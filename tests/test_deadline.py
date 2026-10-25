"""Acceptance: server-side request deadline and budget-aware upstream calls."""
from __future__ import annotations

import io
import socket
import time
import urllib.error

from conftest import FakeRetrieval, citation, retrieval_response

from langgraph_rag.errors import Outcome, RetrievalError
from langgraph_rag.retrieval_client import HttpKnowledgeSearchClient

URL = "http://127.0.0.1:8788/knowledge_search"


class RecordingRetrieval(FakeRetrieval):
    def __init__(self, response=None):
        super().__init__(response if response is not None
                         else retrieval_response([citation()]))
        self.timeouts: list[float | None] = []

    def search(self, *, query, filters, top_k, mode, timeout=None):
        self.timeouts.append(timeout)
        return super().search(query=query, filters=filters, top_k=top_k, mode=mode)


def test_engine_passes_the_remaining_budget_to_retrieval(engine_factory):
    retrieval = RecordingRetrieval()
    engine = engine_factory(retrieval=retrieval, request_deadline=2.0)
    response = engine.run({"rag_answer": {"query": "budgeted"}})
    assert response["status"] == "ok"
    assert retrieval.timeouts and retrieval.timeouts[0] is not None
    assert 0 < retrieval.timeouts[0] <= 2.0


def test_budget_is_shared_between_freshness_and_retrieval(engine_factory):
    class HalfSecondFreshness:
        def read(self, *, timeout=None):
            time.sleep(0.25)
            from conftest import fresh_contract
            return fresh_contract()

    retrieval = RecordingRetrieval()
    engine = engine_factory(retrieval=retrieval, freshness=HalfSecondFreshness(),
                            request_deadline=1.0)
    assert engine.run({"rag_answer": {"query": "shared"}})["status"] == "ok"
    # freshness consumed ~0.25 s, so retrieval must be given the remainder.
    assert retrieval.timeouts[0] <= 0.8


def test_no_deadline_means_no_timeout_keyword(engine_factory):
    retrieval = RecordingRetrieval()
    engine = engine_factory(retrieval=retrieval, request_deadline=None)
    assert engine.run({"rag_answer": {"query": "plain"}})["status"] == "ok"
    assert retrieval.timeouts == [None]


def _client(monkeypatch, behaviors, captured, **kw):
    def fake(req, timeout=None):
        captured.append(timeout)
        kind = behaviors[min(len(captured) - 1, len(behaviors) - 1)]
        if kind == "timeout":
            raise socket.timeout("timed out")
        if kind == "status":
            raise urllib.error.HTTPError(req.full_url, 503, "err", {}, io.BytesIO(b"body"))
        raise AssertionError(kind)

    monkeypatch.setattr("urllib.request.urlopen", fake)
    sleeps: list[float] = []
    return HttpKnowledgeSearchClient(URL, sleeper=sleeps.append, **kw), sleeps


def test_total_budget_caps_retries_and_attempt_timeouts(monkeypatch):
    captured: list[float] = []
    client, sleeps = _client(monkeypatch, ["timeout"], captured,
                             timeout=5.0, max_retries=5, backoff=0.2)
    started = time.monotonic()
    try:
        client.search(query="q", filters={}, top_k=3, mode="hybrid", timeout=0.3)
    except RetrievalError as exc:
        assert exc.outcome == Outcome.RETRIEVAL_TIMEOUT
    else:  # pragma: no cover - the fake always times out
        raise AssertionError("expected a retrieval timeout")
    elapsed = time.monotonic() - started
    assert len(captured) <= 2
    assert all(t is not None and t <= 0.3 for t in captured)
    assert sleeps and sum(sleeps) <= 0.4
    assert elapsed < 0.8


def test_budget_below_one_attempt_stops_immediately(monkeypatch):
    captured: list[float] = []
    client, sleeps = _client(monkeypatch, ["timeout"], captured, timeout=5.0, max_retries=2)
    try:
        client.search(query="q", filters={}, top_k=3, mode="hybrid", timeout=0.001)
    except RetrievalError as exc:
        assert exc.outcome == Outcome.RETRIEVAL_TIMEOUT
    assert captured == [] or len(captured) == 1  # one clamped 0.05 s attempt at most
    assert sleeps == []


def test_engine_deadline_off_keeps_the_historical_behaviour(engine_factory):
    retrieval = FakeRetrieval(retrieval_response([citation()]))
    engine = engine_factory(retrieval=retrieval, request_deadline=0.0)
    response = engine.run({"rag_answer": {"query": "no budget"}})
    assert response["status"] == "ok"
    assert response["deadline_exceeded"] is False
    assert retrieval.calls == 1
