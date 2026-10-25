# rag-auto plugin — automatic LangGraph evidence for Hermes

Phase 1 skeleton. **Nothing is installed or enabled by this directory.**

## What it does

| Surface | Behaviour |
| --- | --- |
| `pre_llm_call` | Once per turn, before the model plans. Deterministic gate → allowlist, hard skips, dedup, breaker, budget → one loopback `POST /rag_answer` → bounded `<retrieved_evidence trust="untrusted-data">` block in the user message. |
| `rag_answer` tool | Registered via `ctx.register_tool`. Model-selected deep dive for profiles without an MCP `knowledge_search`. Read-only, bounded JSON result. |
| JSONL metrics | One line per evaluated turn (`~/.hermes/rag-auto/metrics.jsonl` by default). No query text, no snippets, no raw session ids: `session_hash`, `turn_id` and `task_id` are one-way digests (`turn_id`/`task_id` are `sha256:`-prefixed 16-hex values, deterministic so repeat turns still correlate). Fail-open appears as `status=timeout|error`, `fail_open=true`. |

Hard skips (reason codes): `profile_not_allowlisted`,
`empty_or_command_turn`, `secret_or_sensitive`, `privileged_or_destructive`,
`not_evidence_dependent`, `duplicate_turn`, `graph_unhealthy`,
`budget_exhausted`.

Caps: ≤ 8 evidence items, ≤ 6,000 injected characters, ≤ 400 chars per snippet
(`life-coach`: 5 items / 3,000 chars). Turn / session / hourly budgets: 1 / 20 /
120. Breaker: 2 consecutive terminal graph failures → open 60 s → `/rag_health`
probe.

## Configuration (all optional)

Read through `plugins.entries.rag-auto.settings`:

```yaml
plugins:
  entries:
    rag-auto:
      settings:
        graph_url: http://127.0.0.1:8791
        profiles: [developer]          # runtime allowlist (default: all owners)
        deadline.seconds: 6.0          # whole hook path
        deadline.graph_seconds: 5.0    # graph leg (must sit inside the server deadline)
        budget.per_session: 20
        budget.per_profile_hourly: 120
        inject: true                   # false = shadow mode (metrics only)
        metrics.path: /var/lib/hermes/rag-auto/metrics.jsonl
```

## Enablement steps (operator, later — NOT performed in phase 1)

1. Copy this directory to `<HERMES_HOME>/plugins/rag-auto/` for one profile
   (canary: `developer`).
2. Add `rag-auto` to that profile's `plugins.enabled` (the only switch).
3. Ensure the graph adapter runs with `--request-deadline 5.0` (default) and a
   writable `--metrics-path`.
4. Restart that profile's agent process; verify with a fresh process:
   `HERMES_HOME=<profile> python -c "from hermes_cli.plugins import get_plugin_manager; \
   get_plugin_manager().discover_and_load(); print('rag-auto' in ...)"`
   or `hermes plugins list` for that profile.
5. Phase 2: run in shadow mode (`inject: false`) for 72 h and read the JSONL
   decision stream. Phase 3 flips `inject` only after G1/G2/G4 pass.

Rollback: remove `rag-auto` from `plugins.enabled` (or delete the directory).
No service change, no schema change, no source change.

## Tests

`tests/test_plugin_*.py` in the parent module suite. Fake transport only; no
live graph call, no socket, no profile mutation.
