"""A/B validation harness: incumbent knowledge_search (A) vs reviewed rag_answer (B).

Read-only probes. Exact wrapped request contracts:
  A: POST http://127.0.0.1:8788/knowledge_search  body {"knowledge_search":{...}}
  B: POST http://127.0.0.1:8791/rag_answer       body {"rag_answer":{...}}

Writes ab_results.json (+ raw/ dir) into the directory holding this script.
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.error
import urllib.request

A_URL = "http://127.0.0.1:8788/knowledge_search"
B_URL = "http://127.0.0.1:8791/rag_answer"
HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"

# Fixed, reproducible query set. Each entry is frozen verbatim.
QUERIES = [
    {"id": "Q1", "class": "candidate_facts", "query": "Python backend API development",
     "mode": "hybrid", "top_k": 5, "filters": {}},
    {"id": "Q2", "class": "candidate_facts", "query": "SQL ETL data pipeline experience",
     "mode": "hybrid", "top_k": 5, "filters": {}},
    {"id": "Q3", "class": "job_requirements_skills", "query": "Kafka event-driven systems",
     "mode": "hybrid", "top_k": 5, "filters": {}},
    {"id": "Q4", "class": "job_requirements_skills", "query": "Terraform infrastructure as code",
     "mode": "hybrid", "top_k": 5, "filters": {}},
    {"id": "Q5", "class": "company_role_lookup", "query": "employer_a senior full-stack engineer python django",
     "mode": "hybrid", "top_k": 5, "filters": {}},
    {"id": "Q6", "class": "company_role_lookup", "query": "employer_b senior python engineer berlin",
     "mode": "hybrid", "top_k": 5, "filters": {}},
    {"id": "Q7", "class": "known_reference", "query": "Hermes MCP connection test",
     "mode": "hybrid", "top_k": 5, "filters": {}},
    {"id": "Q8", "class": "filtered_requirements", "query": "Berlin",
     "mode": "hybrid", "top_k": 5, "filters": {"types": ["job_opening"]}},
    {"id": "Q9", "class": "empty_result", "query": "zzqqx flurblewump 99213 nonexistent",
     "mode": "keyword", "top_k": 5, "filters": {}},
    {"id": "Q10", "class": "empty_result_hybrid_control", "query": "zzqqx flurblewump 99213 nonexistent",
     "mode": "hybrid", "top_k": 5, "filters": {}},
    {"id": "Q11", "class": "coverage_cap", "query": "Python",
     "mode": "hybrid", "top_k": 20, "filters": {}},
]


def call(url: str, body: dict, timeout: float = 30.0) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    code = None
    raw = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = resp.status
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        code = exc.code
        raw = exc.read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        code = -1
        raw = json.dumps({"transport_error": f"{type(exc).__name__}: {exc}"})
    elapsed_ms = round((time.monotonic() - t0) * 1000, 2)
    try:
        parsed = json.loads(raw)
    except Exception:  # noqa: BLE001
        parsed = {"unparsable": raw}
    return {"http": code, "client_latency_ms": elapsed_ms, "body": parsed, "raw": raw}


def ids_a(body: dict) -> list[tuple[str, int]]:
    return [(r.get("source_id"), r.get("chunk_index")) for r in (body.get("results") or [])]


def ids_b(body: dict) -> list[tuple[str, int]]:
    return [(r.get("source_id"), r.get("chunk_index")) for r in (body.get("evidence") or [])]


BATCH = "ab20261014T1755"

def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(exist_ok=True)
    results = []
    for q in QUERIES:
        rid = f"ab-audit-{BATCH}-{q['id'].lower()}"
        a_body = {"knowledge_search": {"query": q["query"], "top_k": q["top_k"],
                                       "mode": q["mode"], "filters": q["filters"]}}
        b_body = {"rag_answer": {"query": q["query"], "top_k": q["top_k"],
                                 "mode": q["mode"], "filters": q["filters"], "run_id": rid}}
        a = call(A_URL, a_body)
        b = call(B_URL, b_body)
        (RAW / f"{q['id']}_A.json").write_text(a["raw"])
        (RAW / f"{q['id']}_B.json").write_text(b["raw"])
        ia, ib = ids_a(a["body"]), ids_b(b["body"])
        sa, sb = set(ia), set(ib)
        entry = {
            "query_id": q["id"], "class": q["class"], "query": q["query"],
            "mode": q["mode"], "top_k": q["top_k"], "filters": q["filters"],
            "A": {
                "http": a["http"], "client_latency_ms": a["client_latency_ms"],
                "ok": a["body"].get("ok"), "count": a["body"].get("count"),
                "reported_latency_ms": a["body"].get("latency_ms"),
                "index_version": a["body"].get("index_version"),
                "mode": a["body"].get("mode"),
                "quarantined": a["body"].get("quarantined"),
                "error_class": a["body"].get("error_class"),
                "identifiers": [{"source_id": s, "chunk_index": c,
                                 "object_id": (a["body"].get("results") or [])[i].get("object_id"),
                                 "type_key": (a["body"].get("results") or [])[i].get("type_key"),
                                 "score": (a["body"].get("results") or [])[i].get("score")}
                                for i, (s, c) in enumerate(ia)],
            },
            "B": {
                "http": b["http"], "client_latency_ms": b["client_latency_ms"],
                "ok": b["body"].get("ok"), "status": b["body"].get("status"),
                "run_id": b["body"].get("run_id"),
                "reported_latency_ms": b["body"].get("latency_ms"),
                "index_version": b["body"].get("index_version"),
                "error": b["body"].get("error"), "error_class": b["body"].get("error_class"),
                "synthesis": b["body"].get("synthesis"),
                "replayed": b["body"].get("replayed"),
                "freshness_status": (b["body"].get("freshness") or {}).get("status"),
                "freshness_fresh": (b["body"].get("freshness") or {}).get("fresh"),
                "freshness_reasons": (b["body"].get("freshness") or {}).get("reasons"),
                "validation": b["body"].get("validation"),
                "retrieval": b["body"].get("retrieval"),
                "quarantined": b["body"].get("quarantined"),
                "citations": [{"source_id": e.get("source_id"), "object_id": e.get("object_id"),
                               "space_id": e.get("space_id"), "type_key": e.get("type_key"),
                               "chunk_index": e.get("chunk_index"),
                               "index_version": e.get("index_version"),
                               "source_url": e.get("source_url"), "score": e.get("score"),
                               "snippet": e.get("snippet")}
                              for e in (b["body"].get("evidence") or [])],
            },
            "compare": {
                "A_n": len(ia), "B_n": len(ib),
                "overlap": len(sa & sb),
                "A_only": [list(x) for x in (sa - sb)],
                "B_only": [list(x) for x in (sb - sa)],
                "identical_order_and_set": ia == ib,
                "scores_identical": (
                    [(r.get("source_id"), r.get("chunk_index"), r.get("score"))
                     for r in (a["body"].get("results") or [])]
                    == [(r.get("source_id"), r.get("chunk_index"), r.get("score"))
                        for r in (b["body"].get("evidence") or [])]
                ),
            },
        }
        results.append(entry)
        print(f"{q['id']:4s} A(n={entry['compare']['A_n']}) B(n={entry['compare']['B_n']}) "
              f"overlap={len(sa & sb)} identical={entry['compare']['identical_order_and_set']} "
              f"B_status={entry['B']['status']} A_ms={entry['A']['client_latency_ms']} "
              f"B_ms={entry['B']['client_latency_ms']}")
    (HERE / "ab_results.json").write_text(json.dumps(
        {"generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "A_url": A_URL, "B_url": B_URL, "queries": results}, indent=2))
    print("wrote", HERE / "ab_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
