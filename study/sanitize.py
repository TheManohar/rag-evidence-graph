#!/usr/bin/env python3
"""Reproducible redaction of the evaluation evidence published in ``study/``.

The full captures of the original evaluation are private: they contain source
text from a personal knowledge base (retrieved snippets), stable source-object
identifiers, and the names of specific employers. The *measurements* are the
part worth publishing, so this script derives publishable summaries from the
raw captures and refuses to emit anything that still carries the private parts.

Run:

    RAG_STUDY_RAW=/path/to/raw python3 study/sanitize.py

It is committed so the curation is auditable: you can read exactly which fields
were dropped, and re-derive the published JSON from the raw captures instead of
trusting a hand-edited file.

Exit codes: 0 = written and clean, 1 = a leak guard tripped, 2 = raw input missing.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

STUDY = Path(__file__).resolve().parent
RAW = Path(os.environ.get("RAG_STUDY_RAW", "/root/rag-langgraph-work"))

# Keys that must never reach the published output. Matched case-insensitively
# against every key at every depth.
FORBIDDEN_KEYS = {
    "snippet",
    "source_id",
    "object_id",
    "space_id",
    "source_url",
    "anytype_name",
    "anytype_snippet",
    "text",
    "content",
}

# Stable object identifiers of the source system, and its URL scheme.
ID_PATTERN = re.compile(r"bafyrei[0-9a-z]{20,}")
URL_PATTERN = re.compile(r"anytype://")
HOST_PATH_PATTERN = re.compile(r"/(root|home|etc/nixos)/")

# Employer names are replaced with role-neutral labels. The *mapping* is private
# by design: a committed table of `"real-employer": "employer_a"` would publish
# exactly what the redaction is meant to hide. It is loaded from
# ``study/employer-labels.local.json`` (gitignored, never committed). Without it
# the script fails closed instead of emitting raw role-lookup query text.
LABELS_PATH = STUDY / "employer-labels.local.json"
ROLE_LOOKUP_MARKERS = ("role_lookup", "company")
LABEL_PREFIX = "employer_"

_employer_labels: dict[str, str] = {}


def load_employer_labels() -> None:
    """Load the private name->label map. Absent is a supported state."""
    if not LABELS_PATH.is_file():
        return
    raw = json.loads(LABELS_PATH.read_text())
    _employer_labels.clear()
    _employer_labels.update({str(k).lower(): str(v) for k, v in raw.items()})


def needs_employer_redaction(diagram_class: Any) -> bool:
    text = str(diagram_class or "").lower()
    return any(marker in text for marker in ROLE_LOOKUP_MARKERS)


def redact_text(value: Any, *, diagram_class: Any = None) -> Any:
    """Replace employer names in a query string; count what was replaced."""
    if not isinstance(value, str):
        return value
    out = value
    for name in sorted(_employer_labels, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
        out, hits = pattern.subn(_employer_labels[name], out)
        for _ in range(hits):
            withheld("employer_names_in_query_text")
    if needs_employer_redaction(diagram_class) and not out.startswith(LABEL_PREFIX):
        raise SystemExit(
            "REFUSING TO PUBLISH: a role-lookup query text has no employer label and "
            f"{LABELS_PATH.name} is absent or does not cover it. The map is private by "
            "design; the committed .sanitized.json outputs were derived with it present."
        )
    return out

_removed: dict[str, int] = {}
_withheld: dict[str, int] = {}


def note(kind: str) -> None:
    _removed[kind] = _removed.get(kind, 0) + 1


def withheld(kind: str, n: int = 1) -> None:
    _withheld[kind] = _withheld.get(kind, 0) + n


def scrub(value: Any) -> Any:
    """Drop forbidden keys recursively; record what was dropped."""
    if isinstance(value, dict):
        out = {}
        for key, val in value.items():
            if key.lower() in FORBIDDEN_KEYS:
                note(key.lower())
                continue
            out[key] = scrub(val)
        return out
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


def assert_clean(payload: Any, label: str) -> None:
    """Leak guard. Raises rather than writing an unsafe file.

    Keys are checked structurally at every depth; *values* are only checked for
    identifier/URL/path patterns. String-matching keys inside the serialised blob
    produces false positives the moment a legitimate value happens to spell a
    forbidden key (e.g. a field literally naming which field the audit used).
    """
    blob = json.dumps(payload)
    problems = []
    for pattern, why in (
        (ID_PATTERN, "source-object identifier"),
        (URL_PATTERN, "source-system URL"),
        (HOST_PATH_PATTERN, "host filesystem path"),
    ):
        if pattern.search(blob):
            problems.append(f"{label}: {why} survived redaction")

    def walk(node: Any, trail: str) -> None:
        if isinstance(node, dict):
            for key, val in node.items():
                if key.lower() in FORBIDDEN_KEYS:
                    problems.append(f"{label}: forbidden key {trail}{key!r} survived redaction")
                walk(val, f"{trail}{key}.")
        elif isinstance(node, list):
            for i, val in enumerate(node):
                walk(val, f"{trail}[{i}].")

    walk(payload, "")
    if problems:
        raise SystemExit("LEAK GUARD: " + "; ".join(problems))


def ids_of(entries: list[dict]) -> list[tuple[str, int]]:
    """(source_id, chunk_index) identity used for overlap maths — never emitted."""
    return [
        (e.get("source_id", ""), int(e.get("chunk_index", 0)))
        for e in entries or []
    ]


def sanitize_ab(raw: dict) -> dict:
    """A/B comparison: keep metrics and per-query provenance, drop all content."""
    queries = []
    for q in raw.get("queries", []):
        a, b = q.get("A", {}), q.get("B", {})
        a_ids = ids_of(a.get("identifiers", []))
        b_ids = ids_of(b.get("citations", []))
        b_scores = [c.get("score") for c in b.get("citations", [])]
        a_scores = [i.get("score") for i in a.get("identifiers", [])]
        withheld("A_identifier_records", len(a_ids))
        withheld("B_citation_records", len(b_ids))
        withheld("retrieved_snippets", sum(1 for c in b.get("citations", []) if c.get("snippet")))

        queries.append(
            {
                "query_id": q.get("query_id"),
                "class": q.get("class"),
                "query": redact_text(q.get("query"), diagram_class=q.get("class")),
                "mode": q.get("mode"),
                "top_k": q.get("top_k"),
                "filters": q.get("filters", {}),
                "A": {
                    "http": a.get("http"),
                    "ok": a.get("ok"),
                    "count": a.get("count"),
                    "client_latency_ms": a.get("client_latency_ms"),
                    "reported_latency_ms": a.get("reported_latency_ms"),
                    "index_version": a.get("index_version"),
                    "mode": a.get("mode"),
                    "quarantined": a.get("quarantined"),
                    "error_class": a.get("error_class"),
                },
                "B": {
                    "http": b.get("http"),
                    "ok": b.get("ok"),
                    "status": b.get("status"),
                    "citation_count": len(b_ids),
                    "client_latency_ms": b.get("client_latency_ms"),
                    "reported_latency_ms": b.get("reported_latency_ms"),
                    "index_version": b.get("index_version"),
                    "error": b.get("error"),
                    "error_class": b.get("error_class"),
                    "synthesis": b.get("synthesis"),
                    "replayed": b.get("replayed"),
                    "freshness_status": b.get("freshness_status"),
                    "freshness_fresh": b.get("freshness_fresh"),
                    "freshness_reasons": b.get("freshness_reasons"),
                    "validation": b.get("validation"),
                    "retrieval": scrub(b.get("retrieval")),
                    "quarantined": b.get("quarantined"),
                },
                "comparison": {
                    "overlap": len(set(a_ids) & set(b_ids)),
                    "a_only": len(set(a_ids) - set(b_ids)),
                    "b_only": len(set(b_ids) - set(a_ids)),
                    "identical_order_and_set": a_ids == b_ids,
                    "scores_identical": a_scores == b_scores,
                },
            }
        )

    return {
        "schema": "rag-evidence-graph/study/ab-summary/1",
        "derived_from": "ab_results.json (raw capture withheld)",
        "generated_at_utc": raw.get("generated_at_utc"),
        "note": (
            "Redacted summary. Source-object identifiers, retrieved snippets and "
            "source-system URLs are dropped; per-query metrics and outcomes kept."
        ),
        "queries": queries,
    }


def sanitize_traceability(raw: dict) -> dict:
    """Citation traceability: keep the audit verdicts, drop the audited objects."""
    # The audit recorded which field of the source object each snippet was checked
    # against. Publish that as a neutral label: the distinction matters (some
    # objects carry no snippet, so the name was compared instead), the vendor's
    # field spelling does not.
    trace_field = {"anytype_snippet": "object_snippet", "anytype_name": "object_name"}

    objects = []
    for o in raw.get("objects", []):
        if o.get("anytype_name"):
            withheld("source_object_names")
        objects.append(
            {
                "type_key": o.get("type_key"),
                "cited_type_key": o.get("cited_type_key"),
                "type_matches": o.get("type_matches"),
                "object_exists": o.get("object_exists"),
                "snippet_traceable": o.get("snippet_traceable"),
                "trace_field": trace_field.get(o.get("trace_field"), o.get("trace_field")),
                "http": o.get("http"),
                "cited_in": o.get("cited_in"),
            }
        )
    return {
        "schema": "rag-evidence-graph/study/citation-traceability/1",
        "derived_from": "citation_traceability.json (raw capture withheld)",
        "note": (
            "Redacted summary. Object names, ids and space ids are dropped; the "
            "per-object verdicts that constitute the audit are kept. trace_field "
            "records whether the cited text was matched against the object's "
            "snippet or, for objects with no snippet, its name."
        ),
        "summary": raw.get("summary", {}),
        "objects": objects,
    }


def sanitize_semantic(raw: dict) -> dict:
    """Index-build quality: timings, counts, per-query retrieval signals."""
    rows = []
    for r in raw.get("rows", []):
        rows.append(
            {
                "id": r.get("id"),
                "class": r.get("class"),
                "query": redact_text(r.get("query"), diagram_class=r.get("class")),
                "mode": r.get("mode"),
                "top_k": r.get("top_k"),
                "filters": r.get("filters", {}),
                "before": {k: scrub(v) for k, v in (r.get("before") or {}).items()},
                "after": scrub(r.get("after")),
            }
        )
    return {
        "schema": "rag-evidence-graph/study/semantic-quality/1",
        "derived_from": "evaluation.json (raw capture withheld)",
        "note": "Redacted summary; retrieval scores/types and build metrics only.",
        "metrics": {
            k: raw.get(k)
            for k in (
                "model_load_s",
                "embedding_build_s",
                "collection_write_s",
                "index_build_s",
                "points",
                "sources",
                "max_rss_kib",
                "content_v2_unchanged",
                "threshold_frozen_before_measurement",
            )
        },
        "rows": rows,
    }


def write(path: Path, payload: Any, label: str) -> None:
    assert_clean(payload, label)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path.relative_to(STUDY.parent)}  ({path.stat().st_size:,} bytes)")


def load(path: Path) -> dict:
    if not path.is_file():
        print(f"missing raw input: {path}", file=sys.stderr)
        raise SystemExit(2)
    return json.loads(path.read_text())


def main() -> int:
    ab_dir = RAW / "ab-validation"
    sem_dir = RAW / "semantic-quality-t56b2d6d6"
    load_employer_labels()

    write(STUDY / "ab-validation" / "ab-summary.sanitized.json",
          sanitize_ab(load(ab_dir / "ab_results.json")), "ab-summary")
    write(STUDY / "ab-validation" / "citation-traceability.sanitized.json",
          sanitize_traceability(load(ab_dir / "citation_traceability.json")), "traceability")

    sem = sem_dir / "evaluation.json"
    if sem.is_file():
        write(STUDY / "semantic-quality" / "evaluation.sanitized.json",
              sanitize_semantic(load(sem)), "semantic")
    else:
        print(f"note: {sem} absent, skipping semantic-quality summary", file=sys.stderr)

    report = {
        "schema": "rag-evidence-graph/study/redaction-report/1",
        "raw_root": "<withheld>",
        "method": (
            "whitelist: each published summary is built field by field from the raw "
            "capture, so private fields are never copied rather than deleted. The "
            "counts below are items deliberately left out."
        ),
        "withheld": dict(sorted(_withheld.items())),
        "withheld_total": sum(_withheld.values()),
        "guards": [
            "source-object identifiers absent",
            "source-system URLs absent",
            "host filesystem paths absent",
            "employer names absent; role-lookup query text carries employer_N labels only",
            "forbidden keys absent at every depth",
        ],
    }
    write(STUDY / "redaction-report.json", report, "redaction-report")
    print(f"leak guards passed; {sum(_withheld.values())} private items withheld")
    return 0


if __name__ == "__main__":
    sys.exit(main())
