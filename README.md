# rag-evidence-graph

A small, offline-testable [LangGraph](https://langchain-ai.github.io/langgraph/) state graph that
performs **read-only retrieval augmented generation with validated citations** — and deliberately
never writes the answer.

The graph takes a typed retrieval request, checks whether the index is fresh enough to be cited,
retrieves candidates through a bounded client, validates that every returned citation is actually
grounded in retrieved text, quarantines instruction-like or ungroundable candidates, and returns a
deterministic, fully cited evidence payload. Prose synthesis is somebody else's job: on every
response `synthesis` is `null`.

That boundary is the point of the project. Most RAG demos hide retrieval quality problems behind a
fluent paragraph. This one refuses to generate anything, so the retrieval layer has to stand on its
own and be measurable — which is what the [study](study/README.md) does.

![The evidence graph: four stages on the success path, each with its own closed set of terminal
outcomes, and every outcome converging on finalize](docs/evidence-graph.png)

Solid green is the run continuing; dashed red is the run ending early in one of the graph's
deterministic outcomes. Editable source: [`docs/evidence-graph.drawio`](docs/evidence-graph.drawio)
(open it in [diagrams.net](https://app.diagrams.net) or the desktop app).

## What it does, concretely

| Property | Behaviour |
|---|---|
| Output | Validated evidence + citations + freshness verdict + validation report. Never prose. |
| Outcome space | Closed set of 12 deterministic outcomes (`ok`, `invalid_request`, `stale_index`, `low_relevance`, `invalid_citations`, `no_evidence`, `deadline_exceeded`, …). No free-form statuses. |
| Failure mode | **Fail-closed.** A missing, stale or unreadable freshness signal cannot produce cited evidence. A missing relevance signal is `low_relevance`, never `ok`. |
| Network surface | Loopback only. Three routes; `PUT`/`DELETE`/`PATCH` return `405`; non-loopback bind is refused at startup. |
| Credentials | None, by design. No provider, no model, no API key anywhere in the tree. |
| Persistence | File-backed SQLite checkpoints (durable, outside the store, not on tmpfs) or in-memory for tests. Refuses to start non-durable unless told it is a dev run. |
| Provenance | One redacted JSONL line per request: `ts, run_id, status, wall_ms, evidence_count, freshness_age, index_version, replayed, caller`. No query text, no snippets. |
| Idempotency | `run_id` doubles as the checkpoint thread id. An identical request replays its stored terminal state with no second retrieval; a *different* request reusing an `run_id` starts clean rather than inheriting the old evidence. |

## Install

Python 3.11+.

```bash
git clone git@github.com:TheManohar/rag-evidence-graph.git
cd rag-evidence-graph

uv venv && . .venv/bin/activate
uv pip install -e '.[sqlite,dev]'     # or: pip install -e '.[sqlite,dev]'
```

`sqlite` is the optional durable-checkpoint extra (`langgraph-checkpoint-sqlite`). Without it only
the in-memory saver is available.

> The distribution is named `rag-evidence-graph`; the import package is `langgraph_rag`. That is
> deliberate: the published source is byte-for-byte the code that was deployed and measured, so the
> import name was not rewritten for cosmetics. A rename is a follow-up for the point where the graph
> is consumed *from* this repository rather than from the host configuration that produced it.

## Run it

The adapter needs a retrieval endpoint and a freshness endpoint. Both are defined by **protocols**
(`RetrievalClient`, `FreshnessSource`), so the graph is not bound to any particular vector store or
metadata service — swap the implementation and the graph is unchanged.

```bash
# Offline/dev: in-memory checkpoints, provenance sink off
python -m langgraph_rag \
  --host 127.0.0.1 --port 8791 \
  --allow-memory \
  --metrics-path '' \
  --knowledge-search-url http://127.0.0.1:8788/knowledge_search \
  --freshness-url      http://127.0.0.1:8790/freshness
```

```bash
curl -s localhost:8791/rag_health | jq
curl -s -X POST localhost:8791/rag_answer \
  -H 'content-type: application/json' \
  -d '{"rag_answer":{"query":"kafka event driven systems","top_k":5,"mode":"hybrid"}}' | jq
```

Durable production shape:

```bash
mkdir -p "$HOME/.local/state/rag-evidence-graph"
python -m langgraph_rag \
  --checkpoint sqlite \
  --checkpoint-path "$HOME/.local/state/rag-evidence-graph/checkpoints.sqlite3" \
  --metrics-path    "$HOME/.local/state/rag-evidence-graph/metrics.jsonl"
```

### Adapter contract

```text
POST /rag_answer       {"rag_answer": {query, top_k?, mode?, filters?, run_id?}}
GET  /rag_run_status   ?run_id=...
GET  /rag_health
```

Unknown fields, identity overrides (`principal`), workflow/credential-shaped fields and unknown
filters are **rejected with 400**, not ignored. A bare `{"query": …}` without the wrapper is a
contract error — that is tested, not incidental.

## The two ideas worth stealing

**1. Validation is not relevance — and the code says so.** Citation validity and freshness are
checked mechanically, but neither tells you whether the evidence answers the question. The study
demonstrates the trap directly: a nonsense query (`zzqqx flurblewump 99213 nonexistent`) in hybrid
mode returned five *perfectly traceable* citations, because a nearest-neighbour search with no score
threshold always returns `top_k`. So a calibrated `term_coverage` gate runs after citation
validation and fails closed to `low_relevance`, and the README of the study says plainly that
"evidence returned" does not mean "requirement supported".

**2. Budgets are structural, not aspirational.** Every request carries a server-side wall-clock
deadline (default 5 s); upstream calls get only the remaining budget, and the adapter returns HTTP
504 `deadline_exceeded` instead of waiting longer. That replaced a ~21 s worst case. The retrieval
client is bounded (timeout, retry count, evidence cap of 10) and its URL is validated as loopback.

## Tests

The suite is fully offline: fake retrieval clients and injected freshness results, no network, no
index, no credentials.

```bash
python -m pytest -q
# 244 passed, 1 skipped
```

Verified on Python 3.14.7 against `langgraph` from PyPI (1.2.17) as well as the pinned deployment
version (1.2.10): a clean `uv venv` + `pip install -e '.[sqlite,dev]'` needs no Nix, no Hermes and no
network at test time. Nothing from the original host environment is imported by the package.

Test modules cover the graph contract, request validation, freshness gating, citation/grounding
validation and quarantine, relevance calibration, the SQLite checkpointer (including cross-process
readback), run-id idempotency, the request deadline, provenance redaction, the adapter's HTTP
surface, and static checks that the NixOS deployment module carries no credentials.

One module (`tests/test_retrieval_diagnostics.py`) exercises the *host's* retrieval layer rather
than this package and is skipped when that sibling module is absent. It is kept in the tree because
deleting a test to make a green run is not a green run.

## Study

`study/` contains the evaluation work behind the design: an A/B validation against the incumbent
retrieval surface, 15 fail-closed probes, citation traceability checked object-by-object against the
source system, a semantic-quality comparison of two index builds, and the caveats that came out of
it. See [study/README.md](study/README.md) for the findings and, importantly, the limitations.

Headline results (single warm run per query, 11 fixed queries, 2026-10-14):

- **No retrieval regression:** for 10 of 11 queries the new path's evidence was identical to the
  incumbent's in set, order *and* score; it adds fields and removes nothing.
- **39/39 cited objects traced back** to real source objects, with zero invariant violations
  (`synthesis` null on every response; no claim without a citation).
- **Fail-closed verified live**, not just in unit tests: 15 malformed/edge requests rejected with the
  right HTTP status and error class, and a `SIGKILL` proved supervision and checkpoint durability.
- **Cost:** one extra loopback hop, roughly +25–35 ms per query.

The study files are curated. Raw captures contain private knowledge-base text and identifiers, so
they stay out of the repository; `study/sanitize.py` is committed so the redaction is reproducible
and auditable rather than a claim.

## Deployment reference

`langgraph-rag.nix` and `langgraph-version-baseline.json` are the NixOS module and runtime baseline
of the original deployment (systemd unit, own interpreter env, `StateDirectory`, no credentials).
They are included as a worked example of running this as a supervised loopback service. Nothing in
the Python package requires Nix.

## Limitations

- Relevance is gated by a floor fitted on a small labelled set. It is a guardrail, not a ranked
  quality metric, and it will need refitting for a different corpus.
- The evidence cap (10) is deliberate; raising it is a configuration decision, not a workaround.
- Freshness is only as good as the upstream projection job. When that job fails the graph degrades
  to `stale_index` — correct, but it means availability is bounded by the indexer.
- Single-space, single-principal in the measured deployment. Nothing here proves multi-tenant
  access control, and the adapter is loopback-only precisely so that boundary is not exposed.

## License

MIT — see [LICENSE](LICENSE).
