# Study — validating an evidence-retrieval path before trusting it

Three questions had to be answered before this graph could be trusted with anything:

1. Does routing retrieval through a validation graph make results **worse** than the plain search
   surface it sits in front of?
2. Does the graph **fail closed** on bad input, or does it fail plausibly?
3. Are the citations it returns **real**, and does a "better" retrieval model actually improve
   retrieval?

The answers, in short: no regression (Q1), yes it fails closed (Q2), and — the interesting one — a
semantically stronger retrieval model **improved precision but cost recall, so it was not promoted**
(Q3). That last result is the reason this directory exists in public form. An evaluation that only
ever reports wins is not an evaluation.

## What is published here

| File | What it is |
|---|---|
| [`ab-validation/RESULT.md`](ab-validation/RESULT.md) | The A/B validation report: method, per-query comparison, fail-closed probes, citation traceability audit, restart/durability checks, verdict, and its own limitations section. |
| [`ab-validation/table.md`](ab-validation/table.md) | The generated per-query comparison table. |
| [`ab-validation/fail_closed_results.json`](ab-validation/fail_closed_results.json) | 15 live fail-closed cases plus the idempotency probes — request bodies, HTTP status, outcome and error class. |
| [`ab-validation/ab-summary.sanitized.json`](ab-validation/ab-summary.sanitized.json) | Per-query metrics and outcomes from the A/B run, with content removed. |
| [`ab-validation/citation-traceability.sanitized.json`](ab-validation/citation-traceability.sanitized.json) | The object-by-object traceability verdicts (39 cited objects). |
| [`semantic-quality/evaluation.sanitized.json`](semantic-quality/evaluation.sanitized.json) | Retrieval scores and types for the incumbent, the coverage rebuild and the semantic rebuild, plus index-build cost. |
| [`redaction-report.json`](redaction-report.json) | Exactly what was withheld, by category. |
| [`sanitize.py`](sanitize.py) | The redactor that produced the `.sanitized.json` files. Committed so the curation is reproducible instead of asserted. |
| [`harness/`](harness/) | The probe scripts that generated the A/B evidence (`run_ab.py`, `fc_probes.py`, `render_table.py`). |

**Withheld:** the raw captures (verbatim retrieval responses), retrieved text snippets, stable
source-object identifiers, the names of the specific source objects, and the employer names in the
frozen query set — those appear as `employer_a` / `employer_b`, because *which* company a query
targeted is not part of any finding here. 213 items in total, counted in `redaction-report.json`. The
measurements, verdicts and error taxonomy are published in full; the *content* of a private knowledge
base is not. `verify_citations.py` is also withheld, because it reads a host credential file — its
results are published as the traceability JSON.

## The setup

Both surfaces ran live and read-only, side by side, on a frozen query set chosen before measurement:

- **A** — the incumbent plain search endpoint (returns raw ranked candidates).
- **B** — this graph's `/rag_answer` endpoint (same candidates, plus freshness gate, grounding
  validation, quarantine and citation assembly).

Eleven queries across six classes: candidate facts, job requirements, company/role lookup, a known
reference, an empty-result control, and a `top_k=20` coverage-cap probe. Every B call carried an
explicit `run_id` so each row is a fresh, non-replayed run that can be read back afterwards.

## Result 1 — no retrieval regression

For **Q1–Q10 the evidence was identical in set, order and score** (`identical_order_and_set=true`,
`scores_identical=true`, `A_only=0`, `B_only=0`). B adds fields to the response; it removes nothing.
The single difference is Q11: at `top_k=20`, A returned 15 results and B returned the top-10 prefix
of exactly that list, because the evidence cap is 10.

Cost: one extra loopback hop, **+25–35 ms** per query (A 9.8–39.3 ms, B 31.3–74.3 ms).

## Result 2 — the trap worth publishing

The negative control is the most instructive row in the whole study. The same nonsense query —
`zzqqx flurblewump 99213 nonexistent` — behaves differently by mode:

- in `keyword` mode it scores nothing and returns **0** results, correctly;
- in `hybrid` mode it returns **5 results from A and 5 fully validated, traceable citations from
  B**.

Nothing was fabricated. Those were real objects. But the vector arm has no score threshold, so
nearest-neighbour search always returns `top_k` regardless of relevance. **"Citation validated" means
traceable, not relevant.** That is why the graph grew a separate calibrated relevance gate that can
fail closed to `low_relevance`, and why the report says plainly that evidence returned is not
evidence of support.

## Result 3 — fail-closed, verified live

15 cases, all issued against the running service, all rejected correctly: bare bodies without the
wrapper, identity overrides (`principal`), workflow/credential-shaped fields, unknown filters, bad
`mode`, `top_k=99`, unknown `run_id` (404), write verbs (405), and the empty-result path as a distinct
`no_evidence` rather than a fabricated `ok`.

Citation audit: **39 unique cited objects, 39/39 existed, 39/39 type-matched, 39/39 traceable**
(31 against an object snippet, 8 against the object name for objects that carry no snippet), zero
invariant violations, `synthesis` null on every response.

Durability: a checkpoint seeded before a restart was read back after it; a deliberate `kill -9`
proved supervision by restarting the service (visible in the report).

## Result 4 — the semantic rebuild that was not promoted

A local ONNX `all-MiniLM-L6-v2` embedding (384-dim, CPU, content-hashed in the deployment) was built
and evaluated against the same frozen queries plus three unrelated-phrase negatives, two
candidate-type-filtered queries and one job-type-filtered query. Cosine floor 0.35, **fixed before
measurement**.

It did what semantic retrieval is supposed to do — and it was still rejected:

| Measure | Before | After |
|---|---|---|
| Topically relevant chunks (independent review of saved output) | 63/90 (70.0%) | 56/57 (98.2%) |
| Unwanted hits on unrelated / unsupported queries | 20 | **0** |
| Q3 / J1 relevant breadth | 5 | **1** |
| C1 / C2 relevant records | 5 / 4 | **3 / 1** |
| Q8 strict-location precision | 5/5 | 4/5 |
| Index build (383 points, 185 sources) | — | 25.8 s, 360 MB peak RSS |

Precision and negative controls improved sharply; **positive-query breadth regressed**, and primary
candidate-fact retrieval stayed at zero because the indexed candidate objects held placeholder
content. A conservative similarity gate throws away long, noisy but relevant chunks. Disposition:
**staged, not promoted to production.** The numbers and the reasons are in the report; the honest
negative is the output.

## Limitations (from the source report, not softened)

- Single warm run per query — latencies are indicative, not tail statistics.
- Relevance is a conservative topical signal, not entailment. The measurement records that
  explicitly (`support=not_assessed`, `scope=topical_similarity_not_entailment`).
- 179 indexed points at A/B time, one space, `acl=["owner"]`. Nothing here speaks to behaviour at
  scale or across principals.
- The A/B says nothing about answer quality, because the graph does not answer. It returns evidence;
  synthesis is downstream.

## Re-deriving the sanitised files

```bash
RAG_STUDY_RAW=/path/to/private/captures python3 study/sanitize.py
```

The script builds each published file field by field (a whitelist, not a delete-list), then runs leak
guards that abort the write if a source-object identifier, source URL or host path survives. If the
guards trip, nothing is written.

The employer-name mapping lives in `study/employer-labels.local.json`, which is **gitignored**: a
committed `"real-employer": "employer_a"` table would publish exactly what it redacts. Without that
file the script still redacts ids, snippets, URLs and paths, but it **refuses to publish** any
role-lookup query text that carries no `employer_N` label — fail-closed, so an absent map can never
cause a name to leak. The committed `.sanitized.json` files are the outputs it produced with the map
present.

## Editorial note

`RESULT.md` and `table.md` are reproduced as written at the time of the work, including internal task
references and working-directory paths (for example in the artifacts section). They were not
rewritten for publication, because an edited report is a different artifact from the one that
supported the decision. Where a path points at the private working directory, the corresponding
sanitised file is in this directory.
