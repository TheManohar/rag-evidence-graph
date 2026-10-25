"""Acceptance: idempotent run ids and status readback."""
from __future__ import annotations

from conftest import FakeRetrieval, QueryAwareRetrieval, citation, retrieval_response

BODY = {"rag_answer": {"query": "same query", "top_k": 3}}


def test_explicit_run_id_replays_without_second_retrieval(engine_factory):
    retrieval = FakeRetrieval(retrieval_response([citation()]))
    engine = engine_factory(retrieval=retrieval)
    first = engine.run(BODY, run_id="run-abc")
    second = engine.run(BODY, run_id="run-abc")
    assert first["replayed"] is False and second["replayed"] is True
    assert second["status"] == "ok" and second["evidence"] == first["evidence"]
    assert retrieval.calls == 1  # idempotent: no second retrieval


def test_derived_run_id_is_deterministic_for_identical_body(engine_factory):
    retrieval = FakeRetrieval(retrieval_response([citation()]))
    engine = engine_factory(retrieval=retrieval)
    a = engine.run(BODY)
    b = engine.run(dict(BODY))
    assert a["run_id"] == b["run_id"]
    assert b["replayed"] is True and retrieval.calls == 1


def test_different_query_is_a_distinct_run(engine_factory):
    retrieval = FakeRetrieval(retrieval_response([citation()]))
    engine = engine_factory(retrieval=retrieval)
    engine.run({"rag_answer": {"query": "one"}})
    engine.run({"rag_answer": {"query": "two"}})
    assert retrieval.calls == 2


def test_status_readback_of_stored_run(engine_factory):
    engine = engine_factory(retrieval=FakeRetrieval(retrieval_response([citation()])))
    resp = engine.run(BODY, run_id="run-x")
    stored = engine.status("run-x")
    assert stored is not None
    assert stored["status"] == resp["status"]
    assert stored["evidence"] == resp["evidence"]


def test_status_of_unknown_run_is_none(engine_factory):
    engine = engine_factory()
    assert engine.status("does-not-exist") is None


# --- regression: explicit run_id reuse (F1/F2) ----------------------------

def test_body_supplied_run_id_is_honoured(engine_factory):
    """F1: a documented body run_id is used, not silently ignored."""
    engine = engine_factory(retrieval=FakeRetrieval(retrieval_response([citation()])))
    resp = engine.run({"rag_answer": {"query": "x", "run_id": "caller-id"}})
    assert resp["run_id"] == "caller-id"
    stored = engine.status("caller-id")
    assert stored is not None and stored["run_id"] == "caller-id"


def test_kwarg_run_id_overrides_body_run_id(engine_factory):
    engine = engine_factory(retrieval=FakeRetrieval(retrieval_response([citation()])))
    resp = engine.run({"rag_answer": {"query": "x", "run_id": "body-id"}}, run_id="kwarg-id")
    assert resp["run_id"] == "kwarg-id"


def test_same_explicit_run_id_different_body_is_isolated(engine_factory):
    """F2: reusing one id for a different request must not inherit evidence."""
    retrieval = QueryAwareRetrieval()
    engine = engine_factory(retrieval=retrieval)
    a = engine.run({"rag_answer": {"query": "alpha"}}, run_id="shared-1")
    b = engine.run({"rag_answer": {"query": "beta"}}, run_id="shared-1")
    assert a["evidence"][0]["source_id"] == "src-alpha"
    assert b["query"] == "beta"
    assert b["evidence"][0]["source_id"] == "src-beta"  # never alpha's evidence
    assert retrieval.calls == 2                          # a real second retrieval
    assert b["replayed"] is False


def test_reused_run_id_status_reflects_the_latest_request(engine_factory):
    retrieval = QueryAwareRetrieval()
    engine = engine_factory(retrieval=retrieval)
    engine.run({"rag_answer": {"query": "alpha"}}, run_id="shared-2")
    engine.run({"rag_answer": {"query": "beta"}}, run_id="shared-2")
    stored = engine.status("shared-2")
    assert stored is not None and stored["query"] == "beta"
    assert stored["evidence"][0]["source_id"] == "src-beta"


def test_same_request_same_id_still_replays(engine_factory):
    """The idempotent replay path is preserved by the isolation fix."""
    retrieval = QueryAwareRetrieval()
    engine = engine_factory(retrieval=retrieval)
    first = engine.run({"rag_answer": {"query": "same"}}, run_id="replay-1")
    second = engine.run({"rag_answer": {"query": "same"}}, run_id="replay-1")
    assert first["replayed"] is False and second["replayed"] is True
    assert second["evidence"] == first["evidence"]
    assert retrieval.calls == 1
