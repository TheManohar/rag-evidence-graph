"""Fail-closed probes for the reviewed RAG evidence path (B) and the A/B contrast.

Read-only except FC-bodies (which are rejected). No service is restarted here.
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.error
import urllib.request

A_URL = "http://127.0.0.1:8788/knowledge_search"
B_URL = "http://127.0.0.1:8791/rag_answer"
B_STATUS = "http://127.0.0.1:8791/rag_run_status"
HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"


def req(method: str, url: str, body: dict | None = None, raw: bytes | None = None) -> dict:
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    r = urllib.request.Request(url, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(r, timeout=20) as resp:
            code, text = resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        code, text = exc.code, exc.read().decode()
    except Exception as exc:  # noqa: BLE001
        code, text = -1, json.dumps({"transport_error": f"{type(exc).__name__}: {exc}"})
    ms = round((time.monotonic() - t0) * 1000, 2)
    try:
        parsed = json.loads(text)
    except Exception:  # noqa: BLE001
        parsed = {"unparsable": text}
    return {"http": code, "ms": ms, "body": parsed, "raw": text}


CASES = [
    ("FC1_B_bare_body_no_wrapper", "POST", B_URL, {"query": "Python backend"}),
    ("FC1b_A_bare_body_no_wrapper", "POST", A_URL, {"query": "Python backend"}),
    ("FC2_B_nonexistent_run_id", "GET", B_STATUS + "?run_id=does-not-exist-abc123", None),
    ("FC3_B_empty_result_query", "POST", B_URL,
     {"rag_answer": {"query": "zzqqx flurblewump 99213 nonexistent", "mode": "keyword", "top_k": 5}}),
    ("FC4_B_identity_override", "POST", B_URL,
     {"rag_answer": {"query": "Python", "principal": "owner"}}),
    ("FC5_B_forbidden_token", "POST", B_URL,
     {"rag_answer": {"query": "Python", "workflowId": "abc"}}),
    ("FC6_B_extra_top_key", "POST", B_URL,
     {"rag_answer": {"query": "Python"}, "workflow": {"id": "x"}}),
    ("FC7_B_missing_query", "POST", B_URL, {"rag_answer": {}}),
    ("FC8_B_blank_query", "POST", B_URL, {"rag_answer": {"query": "   "}}),
    ("FC9_B_unknown_filter", "POST", B_URL,
     {"rag_answer": {"query": "Python", "filters": {"collection": "x"}}}),
    ("FC10_B_bad_mode", "POST", B_URL, {"rag_answer": {"query": "Python", "mode": "semantic"}}),
    ("FC11_B_top_k_out_of_range", "POST", B_URL, {"rag_answer": {"query": "Python", "top_k": 99}}),
    ("FC12_B_PUT_method", "PUT", B_URL, {"rag_answer": {"query": "Python"}}),
    ("FC13_B_run_status_missing_run_id", "GET", B_STATUS, None),
    ("FC14_B_rag_answer_GET_method", "GET", B_URL, None),
]


def main() -> int:
    RAW.mkdir(exist_ok=True)
    out = []
    for name, method, url, body in CASES:
        r = req(method, url, body)
        (RAW / f"{name}.json").write_text(r["raw"])
        b = r["body"]
        rec = {"case": name, "method": method, "url": url,
               "request_body": body, "http": r["http"], "ms": r["ms"],
               "ok": b.get("ok"), "status": b.get("status"),
               "error": b.get("error"), "error_class": b.get("error_class"),
               "error_class_present": "error_class" in b,
               "synthesis": b.get("synthesis")}
        out.append(rec)
        print(f"{name:34s} http={r['http']:>4} status={str(b.get('status')):18s} "
              f"error_class={str(b.get('error_class')):10s} err={(str(b.get('error')) or '')[:60]!r}")

    # Idempotency: identical request under the same explicit run_id replays.
    rid = "ab-audit-replay-001"
    r1 = req("POST", B_URL, {"rag_answer": {"query": "Kafka event-driven systems",
                                            "top_k": 3, "run_id": rid}})
    r2 = req("POST", B_URL, {"rag_answer": {"query": "Kafka event-driven systems",
                                           "top_k": 3, "run_id": rid}})
    r3 = req("POST", B_URL, {"rag_answer": {"query": "Terraform infrastructure as code",
                                           "top_k": 3, "run_id": rid}})
    rs = req("GET", B_STATUS + f"?run_id={rid}", None)
    idem = {
        "run_id": rid,
        "first": {"status": r1["body"].get("status"), "replayed": r1["body"].get("replayed"),
                  "n": len(r1["body"].get("evidence") or []),
                  "fingerprint": r1["body"].get("request_fingerprint")},
        "identical_replay": {"status": r2["body"].get("status"), "replayed": r2["body"].get("replayed"),
                             "n": len(r2["body"].get("evidence") or []),
                             "fingerprint": r2["body"].get("request_fingerprint")},
        "different_request_same_id": {"status": r3["body"].get("status"),
                                      "replayed": r3["body"].get("replayed"),
                                      "n": len(r3["body"].get("evidence") or []),
                                      "fingerprint": r3["body"].get("request_fingerprint")},
        "run_status_readback": {"http": rs["http"], "status": rs["body"].get("status"),
                                "completed": rs["body"].get("completed"),
                                "query": rs["body"].get("query"),
                                "n": len(rs["body"].get("evidence") or [])},
    }
    print("idempotency:", json.dumps(idem, indent=2)[:900])
    (RAW / "FC15_idempotency.json").write_text(json.dumps(idem, indent=2))

    (HERE / "fail_closed_results.json").write_text(json.dumps(
        {"generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "cases": out, "idempotency": idem}, indent=2))
    print("wrote", HERE / "fail_closed_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
