# A/B validation — incumbent `knowledge_search` (A) vs reviewed `rag_answer` evidence path (B)

**Work item:** internal board item (id redacted) · role: system administrator
**Date:** 2026-10-14 UTC (probes run 17:50–17:56 UTC)
**Verdict: A/B validation PASSES.** The reviewed evidence path may become the default, subject to the
caveats in §7 — three of which (evidence cap, relevance-vs-traceability, index content scope) must be
written into the guidance rather than left implicit.

> **Publication note.** This report is published much as it was written, with three changes: employer
> names in the frozen query set are replaced with `employer_a` / `employer_b` (the mapping is not
> published), an internal work-item id is redacted, and artifact paths point at this repository's
> `study/` directory instead of the private working directory. Everything else — measurements, verdict,
> caveats, error strings — is as written. Files named in §8's acceptance table as restart evidence
> (`restart/…`) are withheld; §6 describes what they showed.

Surface A (incumbent): `POST http://127.0.0.1:8788/knowledge_search`, body `{"knowledge_search":{...}}`
Surface B (reviewed): `POST http://127.0.0.1:8791/rag_answer`, body `{"rag_answer":{...}}`

Nothing was mutated. No Anytype content was written, no Qdrant schema change, no repo commit, nothing
pushed. The only service action taken anywhere in this task is the deliberate `langgraph-rag` restart
(§6) plus one SIGKILL of its own process to prove supervision (§6.3).

---

## 1. Environment readback (before any probe)

    $ systemctl is-active hermes-rag-search.service hermes-rag-projection.service langgraph-rag.service
    active
    active
    active
    $ ss -ltnp | grep -E '8788|8790|8791'
    LISTEN 127.0.0.1:8788  (python3, hermes-rag-search)
    LISTEN 127.0.0.1:8790  (python3, hermes-rag-projection)
    LISTEN 127.0.0.1:8791  (python3.14, langgraph-rag)

    A  GET /healthz   -> {"ok":true,"route":"/knowledge_search","read_only":true,"identity_bound":true,"tool":"knowledge_search"}
    B  GET /rag_health -> {"ok":true,"service":"langgraph-rag","checkpointer":{"mode":"sqlite","durable":true},
                           "expected_index_version":"rag-v1","freshness_sla_seconds":900,"max_evidence":10,"synthesis":false}
    f  GET 127.0.0.1:8790/freshness -> ok=true, failure_status="ok", index_version="rag-v1", point_count=179,
                                       last_successful_index_write_at=1791999950, age_seconds=<900, success_count=4, failure_count=0
    i  GET 127.0.0.1:8790/status -> points=179, status=green, optimizer_status=ok, update_queue=0

Both surfaces are live and answering on the loopback-only ports. `point_count` is **179** today (the
guidance/status docs still say 177 — doc staleness, not a defect of either surface).

---

## 2. The fixed query set (verbatim, reproducible)

Queries were frozen before measurement, then run unchanged against both surfaces. Each was sent with its
surface's exact wrapped contract; the B call additionally carries an explicit `run_id`
(`ab-audit-ab20261014T1755-<qid>`) so every row is a fresh, non-replayed run and can be read back later
via `GET /rag_run_status?run_id=...`.

| # | class | query text (verbatim) | mode | top_k | filters |
|---|---|---|---|---|---|
| Q1 | candidate facts | `Python backend API development` | hybrid | 5 | — |
| Q2 | candidate facts | `SQL ETL data pipeline experience` | hybrid | 5 | — |
| Q3 | job requirements / skills | `Kafka event-driven systems` | hybrid | 5 | — |
| Q4 | job requirements / skills | `Terraform infrastructure as code` | hybrid | 5 | — |
| Q5 | company / role lookup | `employer_a senior full-stack engineer python django` | hybrid | 5 | — |
| Q6 | company / role lookup | `employer_b senior python engineer berlin` | hybrid | 5 | — |
| Q7 | known reference | `Hermes MCP connection test` | hybrid | 5 | — |
| Q8 | filtered requirements | `Berlin` | hybrid | 5 | `{"types":["job_opening"]}` |
| Q9 | empty result | `zzqqx flurblewump 99213 nonexistent` | keyword | 5 | — |
| Q10 | empty-result control | `zzqqx flurblewump 99213 nonexistent` | hybrid | 5 | — |
| Q11 | coverage cap | `Python` | hybrid | 20 | — |

Exact reproduction (read-only; the harness only issues GET/POST reads against A and B, and GET against
the Anytype read API for citation spot-checks):

    python3 study/harness/run_ab.py        # per-query A/B comparison
    python3 study/harness/fc_probes.py     # fail-closed probes + idempotency
    python3 study/harness/render_table.py  # rebuilds the table in §3

    # Not published: verify_citations.py (reads a host credential) and
    # restart_check.sh (restarts the live unit). Their results are published
    # as citation-traceability.sanitized.json and §6 below.

---

## 3. Per-query comparison

Identifiers compared as `(source_id, chunk_index)`; `overlap` is the intersection size; `traceable` is
"citation count whose Anytype object was read back and matched" (§5).

| # | class | query (verbatim) | mode / top_k / filters | A n | B n | overlap | A-only | B-only | B status | B freshness | B grounded/total | traceable citations | A ms | B ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q1 | candidate_facts | `Python backend API development` | hybrid / 5 / - | 5 | 5 | 5 | 0 | 0 | ok | fresh | 5/5 | 5/5 | 39.25 | 74.32 |
| Q2 | candidate_facts | `SQL ETL data pipeline experience` | hybrid / 5 / - | 5 | 5 | 5 | 0 | 0 | ok | fresh | 5/5 | 5/5 | 11.9 | 36.25 |
| Q3 | job_requirements_skills | `Kafka event-driven systems` | hybrid / 5 / - | 5 | 5 | 5 | 0 | 0 | ok | fresh | 5/5 | 5/5 | 11.62 | 37.04 |
| Q4 | job_requirements_skills | `Terraform infrastructure as code` | hybrid / 5 / - | 5 | 5 | 5 | 0 | 0 | ok | fresh | 5/5 | 5/5 | 10.98 | 35.88 |
| Q5 | company_role_lookup | `employer_a senior full-stack engineer python django` | hybrid / 5 / - | 5 | 5 | 5 | 0 | 0 | ok | fresh | 5/5 | 5/5 | 10.83 | 40.19 |
| Q6 | company_role_lookup | `employer_b senior python engineer berlin` | hybrid / 5 / - | 5 | 5 | 5 | 0 | 0 | ok | fresh | 5/5 | 5/5 | 10.6 | 40.49 |
| Q7 | known_reference | `Hermes MCP connection test` | hybrid / 5 / - | 5 | 5 | 5 | 0 | 0 | ok | fresh | 5/5 | 5/5 | 14.92 | 43.47 |
| Q8 | filtered_requirements | `Berlin` | hybrid / 5 / `{"types":["job_opening"]}` | 5 | 5 | 5 | 0 | 0 | ok | fresh | 5/5 | 5/5 | 13.98 | 42.9 |
| Q9 | empty_result | `zzqqx flurblewump 99213 nonexistent` | keyword / 5 / - | 0 | 0 | 0 | 0 | 0 | no_evidence | fresh | 0/0 | 0/0 | 10.63 | 38.23 |
| Q10 | empty_result_hybrid_control | `zzqqx flurblewump 99213 nonexistent` | hybrid / 5 / - | 5 | 5 | 5 | 0 | 0 | ok | fresh | 5/5 | 5/5 | 10.31 | 47.82 |
| Q11 | coverage_cap | `Python` | hybrid / 20 / - | 15 | 10 | 10 | 5 | 0 | ok | fresh | 10/15 | 10/10 | 9.78 | 31.25 |

Reading the table:

* **Q1–Q10: byte-identical.** Not merely overlapping — for every one of these ten queries B's evidence
  list is identical to A's result list in **set, order and score** (checked programmatically:
  `identical_order_and_set=true`, `scores_identical=true`, `A_only=0`, `B_only=0`). B adds fields to the
  response; it removes nothing.
* **Q9 is the empty path.** `mode=keyword` with a query whose tokens do not occur scores nothing in the
  keyword arm and, in keyword mode, the vector arm is disabled, so A returns `count=0` and B terminates
  `no_evidence` (HTTP 200, `error_class="retrieval"`) with `freshness.status="fresh"` still attached.
* **Q10 is the control and it is the important one.** The *same* nonsense query in `hybrid` mode returns
  **5 results from A and 5 validated citations from B**. The incumbent's vector arm is a deterministic
  256-dimension feature-hash embedding with **no score threshold**, so hybrid/vector retrieval always
  returns `top_k` nearest points regardless of relevance. B faithfully propagates them (they are real,
  traceable objects — `orca`, `test chat`, `Apotheke am Rathaus Lichtenberg`), so B's "validated cited
  evidence" means **traceable**, not **relevant**. See §7 caveat 2.
* **Q11 exposes the only coverage difference.** At `top_k=20` A returns 15 items and B returns the
  **top-10 prefix** of exactly that list (`B_only=0`, `A_only=5`, order-preserving). B's evidence is
  bounded by `max_evidence=10` (visible in `/rag_health`). This is a documented bound, not a ranking
  disagreement, but it is a real difference for any caller that asked for more than 10.
* **Latency:** B costs one extra loopback hop plus graph/validation/checkpoint work — roughly
  **+25–35 ms** on warm runs (A 9.8–39.3 ms, B 31.3–74.3 ms for the same queries). No timeout risk
  against the adapter's 5 s retrieval timeout or the 900 s freshness SLA.

---

## 4. Fail-closed behaviour (not just the happy path)

Every case below was issued live; `failed_closed_results.json` + `raw/` hold the exact bodies.

| case | request | HTTP | outcome | `error_class` | error |
|---|---|---|---|---|---|
| FC1 | B, bare `{"query":"Python backend"}` (no wrapper) | 400 | `invalid_request` | `contract` | `request must be exactly {'rag_answer': {...}}` |
| FC1b | **A**, the same bare body (contrast) | 400 | — | `contract` | `request must be exactly {'knowledge_search': {...}}` |
| FC2 | B, `GET /rag_run_status?run_id=does-not-exist-abc123` | 404 | — | `not_found` | `unknown run_id` |
| FC3 | B, empty-result query (Q9) | 200 | `no_evidence` | `retrieval` | `retrieval returned no candidates` |
| FC4 | B, `{"rag_answer":{"query":"Python","principal":"owner"}}` | 400 | `invalid_request` | `contract` | `field 'principal' is not permitted…` |
| FC5 | B, `{"rag_answer":{"query":"Python","workflowId":"abc"}}` | 400 | `invalid_request` | `contract` | `field 'workflowId' is not permitted…` |
| FC6 | B, extra top-level key (`workflow`) beside `rag_answer` | 400 | `invalid_request` | `contract` | `only the 'rag_answer' key is allowed…` |
| FC7 | B, `{"rag_answer":{}}` (no query) | 400 | `invalid_request` | `contract` | `query must be a non-empty string` |
| FC8 | B, blank query `"   "` | 400 | `invalid_request` | `contract` | `query must be a non-empty string` |
| FC9 | B, unknown filter `{"collection":"x"}` | 400 | `invalid_request` | `contract` | `field 'collection' is not permitted…` |
| FC10 | B, bad mode `"semantic"` | 400 | `invalid_request` | `contract` | `mode must be one of hybrid\|vector\|keyword` |
| FC11 | B, `top_k=99` | 400 | `invalid_request` | `contract` | `top_k must be an integer in 1..20` |
| FC12 | B, `PUT /rag_answer` | 405 | — | — | `write methods are not available` |
| FC13 | B, `/rag_run_status` with no `run_id` | 400 | — | `contract` | `run_id is required` |
| FC14 | B, `GET /rag_answer` | 404 | — | — | `not found` (no such route) |

Fail-closed is correct in every case: contract violations are `400`/`contract`, a missing run is
`404`/`not_found`, write verbs are `405`, the empty path is a distinct `no_evidence` (not a fabricated
`ok`), and identity/credential/workflow-shaped fields are refused rather than ignored. The card's
premise is reproduced exactly: a bare `{"query":…}` is rejected with `error_class='contract'` on **both**
surfaces.

**Idempotency / thread isolation** (`FC15`): identical request replayed under the same explicit
`run_id=ab-audit-replay-001` returns `replayed=true` with the same fingerprint; the same `run_id` reused
for a *different* request returns a clean run (`replayed=false`, new fingerprint, no inherited evidence);
`GET /rag_run_status?run_id=…` reads the stored terminal state back (`status=ok`, `completed=true`,
`query="Terraform infrastructure as code"`).

---

## 5. Citation traceability and "no synthesised answer"

Every citation B returned across the whole query set was read back through the approved read path
(`GET http://127.0.0.1:31012/v1/spaces/<space>/objects/<id>`, `Anytype-Version: 2025-11-08`):

* **39 unique cited objects → 39/39 exist** in Anytype and the object id matches.
* **39/39 `type_key` match** the cited `type_key` (0 mismatches).
* **39/39 snippets traceable**: 31 match the object's own `snippet`; the remaining 8 are name-only
  objects (`bookmark`/`area`/`chat_derived`) whose Anytype `snippet` field is empty and whose cited text
  equals the object's **name** (e.g. `orca` → `stablyai/orca` bookmark; `test chat` → the `chat_derived`
  object). All 8 are genuine objects, not fabrications.
* **Invariant violations: 0.** Across all 11 responses: `synthesis` is `null` on every response; there is
  no `answer`/`response_text` field; `citations == evidence` (the response never contains a claim that is
  not a citation); every evidence entry carries `source_id`, `object_id`, `space_id`, `type_key`,
  `chunk_index`, `updated_at`, `index_version`, `source_url`, `score`, `snippet`; every `index_version` is
  `rag-v1`. B never returned an uncited claim and never synthesised an answer.

Spot-checked example (Q3, top hit): citation `object_id=bafyreigehmauqrkan3tq2m6lhju7o6hijpv24rmf3vq6e7z4gbusxjh5fe`
read back as Anytype object *“Senior Full-stack Python Engineer (m/f/d) — employer_b”*, type `job_opening`,
snippet beginning `Posting date: 2026-05-29T06:33:36-04:00 / Source: https://employer-b.example/job/…`.

---

## 6. Availability, supervision and durability

### 6.1 Unit state

    $ systemctl show langgraph-rag -p Restart -p Type -p ActiveState -p MainPID -p NRestarts -p StateDirectory
    Restart=on-failure        # NOT Restart=always
    RestartUSec=3s
    Type=simple
    ActiveState=active
    StateDirectory=hermes-langgraph

Answering the card's question directly: the unit **is supervised, but with `Restart=on-failure`, not
`Restart=always`**. For an always-running `Type=simple` daemon this covers the realistic failure modes
(crash, OOM/`signal`, non-zero exit) and auto-restarts after 3 s; it does *not* restart after a clean
`systemctl stop` or a clean `exit(0)`, which is intended. If the guidance wants literal `Restart=always`,
that is a one-line NixOS module change — **not** made here (out of this card's scope).

### 6.2 Deliberate `systemctl restart langgraph-rag` (the only restart of another service in this task)

    before: MainPID=3945985  NRestarts=0  ActiveState=active
    $ systemctl restart langgraph-rag
    after:  MainPID=4015248  NRestarts=0  ActiveState=active (ready within 4 polls × 0.5 s ≈ ≤2 s)

`GET /rag_health` **before** and **after** are byte-identical:

    {"ok":true,"service":"langgraph-rag","checkpointer":{"mode":"sqlite","durable":true},
     "expected_index_version":"rag-v1","freshness_sla_seconds":900,"max_evidence":10,"synthesis":false}

It comes back answering with the same contract: a post-restart `POST /rag_answer` for the same query
returned the **same evidence ids**, `status=ok`, `synthesis=null`, `freshness.status="fresh"`, and the
same response key set as before the restart (`contract_keys_same=true`,
`evidence_identical_after_restart=true`).

**Durability across the restart:** the `run_id=ab-audit-durability-001` run seeded *before* the restart
was read back *after* it — `GET /rag_run_status?run_id=ab-audit-durability-001` → `200`, full stored
response (`status=ok`, `completed=true`, same `request_fingerprint`, same 3 citations). The SQLite
checkpointer at `/var/lib/hermes-langgraph/checkpoints.sqlite3` genuinely survives a restart.

Peers were not disturbed: `hermes-rag-search`, `hermes-rag-projection` and `docker-qdrant`
`ActiveEnterTimestamp` values are unchanged across the restart (17:16:19Z / 17:17:40Z / 2026-09-24).
No reboot was performed.

### 6.3 Supervision proven, not just declared

`kill -9` of the MainPID (`4015248`) → systemd logged `Failed with result 'signal'` →
`Scheduled restart job, restart counter is at 1` → new MainPID `4015782`, `ActiveState=active`,
`NRestarts=1`, back in ~5 s, `GET /rag_health` → the same contract. On-failure supervision works in
practice. (This kills and respawns only `langgraph-rag`; no other unit was touched.)

---

## 7. Verdict, and the precise guidance change it implies

**Verdict: PASS.**

* **B is at least as good as A on the query set.** Every query that returned evidence from A returned
  the same evidence from B — identical ids, order and scores for Q1–Q10; B lost no query and returned no
  empty or degraded result that A had answered.
* **B is strictly better on provenance.** A returns bare citations. B returns the same citations plus:
  a freshness-gate verdict evaluated against the published contract (fail-closed), a validation block
  (`candidates`/`grounded`/`quarantined`/`problems`), a stable `run_id` + `request_fingerprint` with a
  `rag_run_status` readback path, an explicit `index_version`, independent instruction-like quarantine
  and dedup, and `synthesis=null` on every response. It also refuses request shapes A cannot distinguish
  (identity overrides, workflow/credential-shaped fields, unknown filters) — verified in §4.
* **No coverage regression for queries that matter.** For `top_k ≤ 10` (the intended
  requirement→evidence mapping shape, typically a handful of requirements) parity is exact. The single
  difference is the documented `max_evidence=10` cap, which only bites when a caller asks for more than
  10 (Q11: A 15, B top-10 of the same list).
* **B's fail-closed behaviour is correct** on malformed bodies, unknown run ids and the empty-result
  path, its citations are all traceable to real Anytype objects, and it survives a restart with a
  durable checkpointer.

### What the consuming project's guidance had to state

1. **§3.1 — remove the gate sentence.** Replace *“This path stays available while the reviewed graph is
   admitted; it is not replaced until A/B validation passes.”* with a statement that **A/B validation
   passed on 2026-10-14** (evidence: this file and this directory), that the
   reviewed evidence path **is now the default** for JobSearch retrieval, and that the incumbent
   `knowledge_search` surface remains **available as a fallback/deprecation candidate** (it is still what
   B calls internally, so it must not be removed as a first step).
2. **§1 — the conditional is satisfied.** Both halves of *“Until the freshness gate and evidence path are
   verified live…”* now hold (the gate runs live inside the admitted graph and returns `fresh` on every
   probe; the evidence path is admitted and answering). The standing instruction becomes **unconditional**
   for JobSearch work; the “do not treat the index as fresh” caveat is superseded by “read freshness from
   the adapter response”.
3. **§6 — the live-status summary is stale and must be rewritten**: the adapter *is* admitted
   (`langgraph-rag.service` active, loopback `127.0.0.1:8791`), and the collection now holds **179**
   points, not 177.
4. **New: record the evidence cap.** `max_evidence=10`: a `rag_answer` call returns at most 10 citations
   even when `top_k` is larger (verified Q11). Requirement→evidence mapping should call with `top_k ≤ 10`.
5. **New: record that citations are traceability-validated, not relevance-validated.** The retrieval arms
   carry no score threshold, so hybrid/vector mode always returns `top_k` nearest points even for an
   unrelated query (verified Q10: a nonsense query returned 5 traceable citations). Callers must not read
   “evidence returned” as “requirement supported”: for exact-term requirement checks use
   `mode="keyword"` (which does return zero hits, Q9), and treat `supported`/`adjacent`/`gap` signal as
   dependent on the caller's own relevance judgement plus the `score` field.
6. **New: record the index-content scope.** The projected chunk is the object's **snippet prefix**
   (≈≤300 chars, `chunk_count=1` for every point), so titles/company names are largely absent from the
   indexed text (a “company/role lookup” matches mainly on the source URL path), and the `profile`,
   `candidate` and `evidence` objects currently project as archival placeholders
   (*“Historical version preserved by Hermes n8n upsert …”*) — candidate-fact retrieval is therefore
   **not** yet substantive. This is a coverage limit inherited from the incumbent surface, not something
   B fixes, and it should be stated so the adoption is not oversold.

The corresponding `docs/knowledge-retrieval-status.md` (which still says the adapter is *not* admitted)
needs the same refresh; both are repo edits and are **out of my scope**, so they are carded as a
follow-up for the `developer` profile rather than silently edited (per the card's instruction).

---

## 8. Acceptance criteria — per criterion

| criterion | status | evidence |
|---|---|---|
| Query set recorded verbatim and reproducible | pass | §2 table; `run_ab.py` (`QUERIES`), `ab_results.json` |
| Per-query comparison table with both surfaces' outputs and B's citation provenance | pass | §3; `raw/Q*_A.json`, `raw/Q*_B.json`, `ab_results.json`, `table.md` |
| B's fail-closed cases with exact verdicts | pass | §4; `fail_closed_results.json` (15 cases + idempotency) |
| B's restart behaviour with before/after `rag_health` | pass | §6; `restart/{health_before,health_after}.json`, `props_*.txt`, `status_after_restart_raw.txt`, kill-9 `NRestarts=1` |
| Explicit pass/fail verdict + precise implied guidance change | pass | §7 (PASS; 6 numbered changes) |
| No service restarted except the deliberate `langgraph-rag` restart | pass | §6.2/6.3; peer `ActiveEnterTimestamp` unchanged in `restart/peers_{before,after}.txt` |
| Nothing pushed / no mutation | pass | read-only probes only; no git command issued in the JobSearch repo |

## 9. Residual risk and limitations

1. **Relevance is not fixed by B.** B validates traceability and freshness, not semantic match (Q10). A
   caller that treats any returned citation as support for a requirement can still be wrong — this is the
   most important caveat for the JobSearch pilot, and it is why §7.5 must be written into the guidance.
2. **The 10-evidence cap** is a behavioural difference from A at `top_k > 10`; if a future use case needs
   more evidence per call, `max_evidence` must be raised deliberately (a module option), not worked around.
3. **Freshness depends on the projection job succeeding.** B fails closed, but a projection outage
   degrades B to `stale_index`/`freshness_unavailable` rather than returning stale evidence — expected,
   but it means B's availability is bounded by the projection workflow's health.
4. **Single measurement run.** Latencies are one warm run per query (no repetition/tail statistics);
   they are indicative only. Idempotent replay also means two identical requests are not independent
   samples.
5. **Corpus size.** 179 points, single space, `acl=["owner"]` for all points — the A/B says nothing about
   behaviour at scale or under multiple principals.
6. **`Restart=on-failure`, not `Restart=always`** (§6.1): a clean exit is not auto-restarted. Considered
   acceptable for this daemon; change only if the guidance demands `always`.
7. **Docs still stale** (status doc says “not admitted”; 177 vs 179 points) until the carded follow-up lands.

## 10. Artifacts

    study/ab-validation/RESULT.md                    <- this report
    study/ab-validation/ab-summary.sanitized.json    <- per-query A/B metrics (redacted)
    study/ab-validation/fail_closed_results.json     <- 15 fail-closed cases + idempotency
    study/ab-validation/citation-traceability.sanitized.json <- 39 cited objects, verdicts only
    study/ab-validation/table.md                     <- §3 table (generated)
    study/harness/run_ab.py, fc_probes.py,
        render_table.py                              <- reproduction scripts

    WITHHELD from publication: the raw A and B responses, the raw per-query
    comparison (identifiers + snippets), the object-by-object audit, and the
    restart evidence. See study/redaction-report.json for what and how many.
