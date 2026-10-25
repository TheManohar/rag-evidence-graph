"""Render the per-query comparison table (markdown) from ab_results.json."""
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
ab = json.loads((HERE / "ab_results.json").read_text())
tr = json.loads((HERE / "citation_traceability.json").read_text())
trace = {o["object_id"]: o for o in tr["objects"]}

lines = [
    "| # | class | query (verbatim) | mode / top_k / filters | A n | B n | overlap | A-only | B-only | B status | B freshness | B grounded/total | traceable citations | A ms | B ms |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
]
for q in ab["queries"]:
    c, A, B = q["compare"], q["A"], q["B"]
    cited = [e["object_id"] for e in (B["citations"] or [])]
    n_trace = sum(1 for o in cited if trace.get(o, {}).get("snippet_traceable"))
    flt = q["filters"] or "-"
    v = B["validation"] or {}
    lines.append(
        f"| {q['query_id']} | {q['class']} | `{q['query']}` | "
        f"{q['mode']} / {q['top_k']} / {json.dumps(flt) if flt != '-' else '-'} | "
        f"{c['A_n']} | {c['B_n']} | {c['overlap']} | {len(c['A_only'])} | {len(c['B_only'])} | "
        f"{B['status']} | {B['freshness_status']} | {v.get('grounded')}/{v.get('candidates')} | "
        f"{n_trace}/{len(cited)} | {A['client_latency_ms']} | {B['client_latency_ms']} |"
    )
(HERE / "table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
