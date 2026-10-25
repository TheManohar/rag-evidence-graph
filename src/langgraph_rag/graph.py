"""The bounded LangGraph RAG graph.

Graph shape (``langgraph.graph.StateGraph``)::

    START
      -> validate_request --(invalid_request)--> finalize
                          \\--(ok)---------------> freshness_gate
    freshness_gate --(stale/unavailable)-------> finalize
                   \\--(fresh)------------------> retrieve
    retrieve --(retrieval error)---------------> finalize
             \\--(ok)---------------------------> validate_citations -> finalize
    finalize -> END

Terminal output is **grounded, cited evidence plus freshness/validation status**.
There is deliberately no synthesis node and no provider/model/credential path:
the graph returns evidence, Hermes performs synthesis.

Idempotency: the checkpoint ``thread_id`` is the request's ``run_id`` -- an
explicit kwarg, else a caller-supplied ``rag_answer.run_id``, else a
deterministic id derived from the request fingerprint. Replaying the identical
request under the same id returns the stored terminal state without a second
retrieval. A *different* request that reuses an id starts a clean run under
that id: the stored thread is discarded first, so a new request can never
inherit a previous run's channels.
"""
from __future__ import annotations

import inspect
import time
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import citations as C
from . import freshness as F
from . import relevance
from .checkpoint import CheckpointerFactory, InMemoryCheckpointer
from .errors import ContractError, Outcome, RetrievalError
from .ids import derive_run_id, is_valid_run_id
from .retrieval_client import RetrievalClient
from .schema import validate_request

#: Bounded synthesis/evidence cap: never return a full-corpus dump.
DEFAULT_MAX_EVIDENCE = 10

#: Default server-side wall-clock budget for one request. Chosen to sit inside
#: the rag-auto plugin's 6.0 s hook deadline (transport + formatting headroom),
#: and far below the historical worst case of ~21 s (5 s retrieval x 3 attempts
#: + 3 s freshness). ``None`` disables the deadline (tests/offline callers).
DEFAULT_REQUEST_DEADLINE = 5.0

#: Channels cleared when a reused thread is restarted, so a new request can
#: never inherit a previous run's state. Values mirror an untouched run.
RESET_CHANNELS: dict[str, Any] = {
    "query": None, "filters": {}, "top_k": 0, "mode": "",
    "request_fingerprint": None, "freshness": {}, "freshness_status": "",
    "retrieval": {}, "results": [], "evidence": [], "quarantined": [],
    "validation": {}, "citations_valid": False, "outcome": None,
    "relevance": {},
    "error": None, "error_class": None, "response": {},
    "deadline_at": None,
}


def _accepts_timeout(fn: Any) -> bool:
    """True when *fn* can take a ``timeout`` keyword (explicit or ``**kwargs``)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "timeout" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def _call_within(fn: Any, remaining: float | None, **kwargs: Any) -> Any:
    """Call *fn* with the remaining budget when it exposes a ``timeout`` knob."""
    if remaining is not None and _accepts_timeout(fn):
        kwargs["timeout"] = max(0.05, float(remaining))
    return fn(**kwargs)


def _explicit_run_id(body: Any) -> str | None:
    """Caller-supplied ``rag_answer.run_id`` (validated shape), else ``None``.

    Extracted independently of the other fields so a caller-chosen id stays
    addressable through ``rag_run_status`` even for an otherwise-rejected body.
    """
    if not isinstance(body, dict):
        return None
    params = body.get("rag_answer")
    if not isinstance(params, dict):
        return None
    rid = params.get("run_id")
    return rid if is_valid_run_id(rid) else None


class RagState(TypedDict, total=False):
    """Typed graph state. Serialisable values only (checkpoint-safe)."""

    run_id: str
    raw_body: Any
    query: str
    filters: dict
    top_k: int
    mode: str
    request_fingerprint: str
    freshness: dict
    freshness_status: str
    retrieval: dict
    results: list
    evidence: list
    quarantined: list
    validation: dict
    citations_valid: bool
    relevance: dict
    outcome: str
    error: str | None
    error_class: str | None
    started_monotonic: float
    deadline_at: float | None
    response: dict
    completed: bool


class RagEngine:
    """Owns the compiled graph and its injected collaborators."""

    def __init__(
        self,
        *,
        retrieval: RetrievalClient,
        freshness: F.FreshnessSource,
        checkpointer: CheckpointerFactory | None = None,
        expected_index_version: str = "rag-v1",
        sla_seconds: int = F.DEFAULT_SLA_SECONDS,
        max_evidence: int = DEFAULT_MAX_EVIDENCE,
        relevance_floor: float | None = None,
        request_deadline: float | None = None,
        now=F.time.time,
    ):
        self._retrieval = retrieval
        self._freshness = freshness
        self._checkpointer = checkpointer or InMemoryCheckpointer()
        self._expected_index_version = expected_index_version
        self._sla = sla_seconds
        self._max_evidence = max_evidence
        self._relevance_floor = (
            relevance.DEFAULT_RELEVANCE_FLOOR if relevance_floor is None
            else float(relevance_floor)
        )
        self._request_deadline = (
            None if request_deadline is None else max(0.0, float(request_deadline))
        )
        self._now = now
        self._saver = self._checkpointer.build()
        self._graph = self._build()

    @property
    def checkpointer(self) -> CheckpointerFactory:
        """The active persistence provider (mode + durability)."""
        return self._checkpointer

    @property
    def request_deadline(self) -> float | None:
        """The server-side wall-clock budget for one request (``None`` = off)."""
        return self._request_deadline

    @property
    def expected_index_version(self) -> str:
        return self._expected_index_version

    # -- deadline ----------------------------------------------------------
    @staticmethod
    def _remaining(state: RagState) -> float | None:
        deadline_at = state.get("deadline_at")
        if deadline_at is None:
            return None
        return float(deadline_at) - time.monotonic()

    def _expired(self, state: RagState) -> dict | None:
        """A deadline outcome when the request budget is spent, else ``None``."""
        remaining = self._remaining(state)
        if remaining is not None and remaining <= 0:
            return {
                "outcome": Outcome.DEADLINE_EXCEEDED.value,
                "error": "request exceeded its server-side deadline",
                "error_class": "deadline",
            }
        return None

    # -- construction ------------------------------------------------------
    def _build(self):
        g = StateGraph(RagState)
        g.add_node("validate_request", self._node_validate)
        g.add_node("freshness_gate", self._node_freshness)
        g.add_node("retrieve", self._node_retrieve)
        g.add_node("validate_citations", self._node_citations)
        g.add_node("finalize", self._node_finalize)
        g.add_edge(START, "validate_request")
        g.add_conditional_edges(
            "validate_request", self._route_after_validate,
            {"freshness_gate": "freshness_gate", "finalize": "finalize"},
        )
        g.add_conditional_edges(
            "freshness_gate", self._route_after_freshness,
            {"retrieve": "retrieve", "finalize": "finalize"},
        )
        g.add_conditional_edges(
            "retrieve", self._route_after_retrieve,
            {"validate_citations": "validate_citations", "finalize": "finalize"},
        )
        g.add_edge("validate_citations", "finalize")
        g.add_edge("finalize", END)
        return g.compile(checkpointer=self._saver)

    # -- nodes -------------------------------------------------------------
    def _node_validate(self, state: RagState) -> dict:
        try:
            req = validate_request(state.get("raw_body"))
        except ContractError as exc:
            return {"outcome": Outcome.INVALID_REQUEST.value, "error": str(exc),
                    "error_class": "contract"}
        return {
            "query": req.query, "filters": req.filters, "top_k": req.top_k,
            "mode": req.mode, "request_fingerprint": req.fingerprint,
        }

    def _node_freshness(self, state: RagState) -> dict:
        expired = self._expired(state)
        if expired is not None:
            return expired
        remaining = self._remaining(state)
        try:
            contract = _call_within(self._freshness.read, remaining)
        except Exception as exc:  # noqa: BLE001 - any read failure fails closed
            return {
                "freshness": {"status": F.UNAVAILABLE, "reasons": [str(exc)], "contract": {}},
                "freshness_status": F.UNAVAILABLE,
                "outcome": Outcome.FRESHNESS_UNAVAILABLE.value,
                "error": f"freshness contract unavailable: {exc}",
                "error_class": "freshness",
            }
        verdict = F.evaluate_contract(
            contract, expected_index_version=self._expected_index_version,
            sla_seconds=self._sla, now=self._now(),
        )
        node: dict[str, Any] = {"freshness": verdict.to_dict(), "freshness_status": verdict.status}
        if not verdict.fresh:
            node.update({
                "outcome": F.outcome_for(verdict).value,
                "error": "; ".join(verdict.reasons) or "index not fresh",
                "error_class": "freshness",
            })
        return node

    def _node_retrieve(self, state: RagState) -> dict:
        expired = self._expired(state)
        if expired is not None:
            return expired
        remaining = self._remaining(state)
        try:
            resp = _call_within(
                self._retrieval.search, remaining,
                query=state["query"], filters=state.get("filters") or {},
                top_k=state["top_k"], mode=state["mode"],
            )
        except RetrievalError as exc:
            return {"retrieval": {"error": exc.message},
                    "outcome": exc.outcome.value, "error": exc.message,
                    "error_class": "retrieval"}
        except Exception as exc:  # noqa: BLE001 - defensive, deterministic
            return {"outcome": Outcome.INTERNAL_ERROR.value,
                    "error": f"{type(exc).__name__}", "error_class": "internal"}
        meta = {k: resp.get(k) for k in
                ("count", "mode", "index_version", "quarantined", "latency_ms",
                 "retries", "diagnostics")}
        return {"retrieval": meta, "results": resp.get("results") or []}

    def _node_citations(self, state: RagState) -> dict:
        results = state.get("results") or []
        evidence, quarantined, problems = C.validate_evidence(
            results, expected_index_version=self._expected_index_version,
            limit=self._max_evidence,
        )
        node: dict[str, Any] = {
            "evidence": evidence,
            "quarantined": quarantined,
            "citations_valid": bool(evidence),
            "validation": {"candidates": len(results), "grounded": len(evidence),
                           "quarantined": len(quarantined), "problems": problems},
        }
        if evidence:
            # Evaluated *after* citation validation: the gate is an additional,
            # independent check, never a replacement for grounding.
            verdict = relevance.evaluate(evidence, floor=self._relevance_floor)
            node["relevance"] = verdict
            if verdict["passed"]:
                node.update({"outcome": Outcome.OK.value, "error": None, "error_class": None})
            else:
                # Fail closed: weak/unknown signal never yields citable evidence.
                node.update({
                    "evidence": [],
                    "citations_valid": False,
                    "outcome": Outcome.LOW_RELEVANCE.value,
                    "error": verdict["reason"],
                    "error_class": "relevance",
                })
        elif results:
            node.update({"outcome": Outcome.INVALID_CITATIONS.value,
                         "error": "no citation passed grounding validation",
                         "error_class": "citations"})
        else:
            node.update({"outcome": Outcome.NO_EVIDENCE.value,
                         "error": "retrieval returned no candidates",
                         "error_class": "retrieval"})
        return node

    def _node_finalize(self, state: RagState) -> dict:
        outcome = state.get("outcome") or Outcome.INTERNAL_ERROR.value
        started = state.get("started_monotonic") or time.monotonic()
        evidence = state.get("evidence") or []
        response = {
            "ok": outcome == Outcome.OK.value,
            "status": outcome,
            "run_id": state.get("run_id"),
            "query": state.get("query"),
            "request_fingerprint": state.get("request_fingerprint"),
            "evidence": evidence,
            "citations": evidence,
            "quarantined": state.get("quarantined") or [],
            "freshness": state.get("freshness") or {},
            "validation": state.get("validation") or {},
            "retrieval": state.get("retrieval") or {},
            "relevance": state.get("relevance") or {},
            "index_version": self._expected_index_version,
            "error": state.get("error"),
            "error_class": state.get("error_class"),
            "latency_ms": round((time.monotonic() - started) * 1000, 2),
            "deadline_exceeded": outcome == Outcome.DEADLINE_EXCEEDED.value,
            "completed": True,
            "synthesis": None,  # graph returns evidence only; Hermes synthesises
        }
        return {"response": response, "completed": True}

    # -- routing -----------------------------------------------------------
    @staticmethod
    def _route_after_validate(state: RagState) -> str:
        return "finalize" if state.get("outcome") else "freshness_gate"

    @staticmethod
    def _route_after_freshness(state: RagState) -> str:
        return "finalize" if state.get("outcome") else "retrieve"

    @staticmethod
    def _route_after_retrieve(state: RagState) -> str:
        return "finalize" if state.get("outcome") else "validate_citations"

    # -- public API --------------------------------------------------------
    def _snapshot(self, config: dict) -> dict:
        try:
            snap = self._graph.get_state(config)
        except Exception:  # noqa: BLE001 - absent thread is not an error
            return {}
        return dict(snap.values) if snap and snap.values else {}

    def _discard_thread(self, thread_id: str) -> None:
        """Best-effort purge of a stored thread before a clean re-run.

        Correctness does not depend on the saver exposing ``delete_thread``:
        ``run`` also overwrites every channel via ``RESET_CHANNELS``. The purge
        additionally removes the previous run's checkpoint rows.
        """
        deleter = getattr(self._saver, "delete_thread", None)
        if deleter is None:
            return
        try:
            deleter(thread_id)
        except Exception:  # noqa: BLE001 - channel reset still guarantees isolation
            pass

    def run(self, body: Any, *, run_id: str | None = None,
            deadline_seconds: float | None = None) -> dict:
        """Run the graph once.

        Idempotency is keyed on the checkpoint ``thread_id``:

        * ``run_id`` (kwarg) wins, else a caller-supplied ``rag_answer.run_id``,
          else a deterministic id derived from the request fingerprint;
        * an identical request under the same id replays the stored terminal
          state with no second retrieval;
        * a *different* request reusing the same id starts a clean run under
          that id -- the stored thread is discarded so the new request cannot
          inherit the previous run's outcome/evidence/results.

        ``deadline_seconds`` bounds the whole request: each downstream call is
        given only the remaining budget, and a spent budget terminates with
        ``deadline_exceeded`` instead of starting the next step. ``None`` falls
        back to the engine's configured ``request_deadline``.
        """
        budget = self._request_deadline if deadline_seconds is None else float(deadline_seconds)
        deadline_at = None if not budget or budget <= 0 else time.monotonic() + budget
        explicit = run_id or _explicit_run_id(body)
        fp: str | None = None
        try:
            fp = validate_request(body).fingerprint
        except ContractError:
            fp = None
        rid = explicit or (derive_run_id(fp) if fp else "rag-" + uuid.uuid4().hex)
        config = {"configurable": {"thread_id": rid}}
        prev = self._snapshot(config)
        if (prev.get("completed") and fp is not None
                and prev.get("request_fingerprint") == fp):
            return {**prev.get("response", {}), "replayed": True}
        if prev:
            self._discard_thread(rid)
        init: RagState = {
            "run_id": rid,
            "raw_body": body,
            "completed": False,
            "started_monotonic": time.monotonic(),
            **RESET_CHANNELS,
            "deadline_at": deadline_at,
        }
        final = self._graph.invoke(init, config)
        return {**final.get("response", {}), "replayed": False}

    def status(self, run_id: str) -> dict | None:
        """Read back a stored run's terminal status (or None if unknown)."""
        snap = self._snapshot({"configurable": {"thread_id": run_id}})
        if not snap:
            return None
        response = snap.get("response")
        if isinstance(response, dict):
            return {**response, "run_id": run_id, "completed": bool(snap.get("completed"))}
        return {"run_id": run_id, "completed": bool(snap.get("completed")),
                "status": snap.get("outcome") or "in_progress", "response": None}

    def health(self) -> dict:
        return {
            "ok": True,
            "service": "langgraph-rag",
            "checkpointer": {
                "mode": self._checkpointer.mode,
                "durable": self._checkpointer.durable,
            },
            "expected_index_version": self._expected_index_version,
            "freshness_sla_seconds": self._sla,
            "request_deadline_seconds": self._request_deadline,
            "max_evidence": self._max_evidence,
            "relevance_floor": self._relevance_floor,
            "relevance_gate": relevance.GATE_SIGNAL,
            "idf_floor": relevance.DEFAULT_IDF_FLOOR,
            "relevance_gate_rule": relevance.GATE_RULE,
            "synthesis": False,
        }
