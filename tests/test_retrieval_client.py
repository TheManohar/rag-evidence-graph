"""Acceptance: bounded retries/timeouts on the read-only retrieval call."""
from __future__ import annotations

import io
import json
import socket
import urllib.error

import pytest

from langgraph_rag.errors import Outcome, RetrievalError
from langgraph_rag.retrieval_client import HttpKnowledgeSearchClient

URL = "http://127.0.0.1:8788/knowledge_search"
OK = {"ok": True, "results": [], "count": 0, "index_version": "rag-v1"}


class FakeResponse:
    def __init__(self, payload):
        self._bytes = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._bytes

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def make_urlopen(behaviors, captured):
    """behaviors: list of ('ok', payload) | ('status', code) | ('timeout',) | ('urlerror', msg)."""
    state = {"i": 0}

    def fake(req, timeout=None):
        captured.append({"url": req.full_url, "data": req.data, "timeout": timeout, "method": req.get_method()})
        idx = min(state["i"], len(behaviors) - 1)
        state["i"] += 1
        kind = behaviors[idx][0]
        if kind == "ok":
            return FakeResponse(behaviors[idx][1])
        if kind == "status":
            raise urllib.error.HTTPError(req.full_url, behaviors[idx][1], "err", {},
                                         io.BytesIO(b"body"))
        if kind == "timeout":
            raise socket.timeout("timed out")
        if kind == "urlerror":
            raise urllib.error.URLError(behaviors[idx][1] if len(behaviors[idx]) > 1 else "refused")
        raise AssertionError(kind)

    return fake


def client(monkeypatch, behaviors, captured, **kw):
    monkeypatch.setattr("urllib.request.urlopen", make_urlopen(behaviors, captured))
    sleeps: list[float] = []
    return HttpKnowledgeSearchClient(URL, sleeper=sleeps.append, **kw), sleeps


def test_rejects_non_loopback_and_non_http():
    with pytest.raises(ValueError):
        HttpKnowledgeSearchClient("http://10.0.0.5:8788/knowledge_search")
    with pytest.raises(ValueError):
        HttpKnowledgeSearchClient("https://127.0.0.1:8788/knowledge_search")
    HttpKnowledgeSearchClient("http://localhost:8788/knowledge_search")  # allowed


def test_success_and_read_only_envelope(monkeypatch):
    captured: list[dict] = []
    c, sleeps = client(monkeypatch, [("ok", OK)], captured)
    out = c.search(query="q", filters={"types": ["note"]}, top_k=3, mode="hybrid")
    assert out == OK
    assert sleeps == []
    assert len(c.attempts) == 1
    sent = json.loads(captured[0]["data"].decode())
    assert set(sent) == {"knowledge_search"}
    assert set(sent["knowledge_search"]) <= {"query", "top_k", "mode", "filters"}
    assert captured[0]["method"] == "POST"


def test_retryable_status_is_retried_then_succeeds(monkeypatch):
    captured: list[dict] = []
    c, sleeps = client(monkeypatch, [("status", 503), ("ok", OK)], captured, max_retries=2)
    assert c.search(query="q", filters={}, top_k=1, mode="hybrid") == OK
    assert len(c.attempts) == 2
    assert len(sleeps) == 1  # exactly one backoff between the two attempts


def test_retryable_status_exhausted_is_unavailable_and_bounded(monkeypatch):
    captured: list[dict] = []
    c, sleeps = client(monkeypatch, [("status", 503)], captured, max_retries=2)
    with pytest.raises(RetrievalError) as exc:
        c.search(query="q", filters={}, top_k=1, mode="hybrid")
    assert exc.value.outcome is Outcome.RETRIEVAL_UNAVAILABLE
    assert len(c.attempts) == 3  # max_retries + 1: hard upper bound
    assert len(sleeps) == 2


def test_timeout_is_classified_and_bounded(monkeypatch):
    captured: list[dict] = []
    c, _ = client(monkeypatch, [("timeout",)], captured, max_retries=1)
    with pytest.raises(RetrievalError) as exc:
        c.search(query="q", filters={}, top_k=1, mode="hybrid")
    assert exc.value.outcome is Outcome.RETRIEVAL_TIMEOUT
    assert len(c.attempts) == 2


def test_client_4xx_is_terminal_contract_error(monkeypatch):
    captured: list[dict] = []
    c, sleeps = client(monkeypatch, [("status", 400)], captured, max_retries=3)
    with pytest.raises(RetrievalError) as exc:
        c.search(query="q", filters={}, top_k=1, mode="hybrid")
    assert exc.value.outcome is Outcome.RETRIEVAL_CONTRACT
    assert len(c.attempts) == 1 and sleeps == []  # 4xx never retried


def test_unreachable_is_unavailable(monkeypatch):
    captured: list[dict] = []
    c, _ = client(monkeypatch, [("urlerror", "connection refused")], captured, max_retries=0)
    with pytest.raises(RetrievalError) as exc:
        c.search(query="q", filters={}, top_k=1, mode="hybrid")
    assert exc.value.outcome is Outcome.RETRIEVAL_UNAVAILABLE


@pytest.mark.parametrize("payload", [{"ok": False}, {}, {"ok": True, "results": "nope"}, "x"])
def test_non_ok_envelope_is_contract_error(monkeypatch, payload):
    captured: list[dict] = []
    c, _ = client(monkeypatch, [("ok", payload)], captured)
    with pytest.raises(RetrievalError) as exc:
        c.search(query="q", filters={}, top_k=1, mode="hybrid")
    assert exc.value.outcome is Outcome.RETRIEVAL_CONTRACT
