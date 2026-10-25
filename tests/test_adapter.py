"""Acceptance: adapter contract and closed typed-operation surface."""
from __future__ import annotations

import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest

from conftest import FakeRetrieval, QueryAwareRetrieval, citation, retrieval_response
from langgraph_rag.adapter import ROUTES, main, make_handler, status_code_for
from langgraph_rag.errors import Outcome


@pytest.fixture
def server(engine_factory):
    engine = engine_factory(retrieval=FakeRetrieval(retrieval_response([citation()])))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd, engine
    finally:
        httpd.shutdown()
        httpd.server_close()


def call(server, method, path, body=None):
    httpd, _ = server
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
    payload = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, body=payload,
                 headers={"Content-Type": "application/json"} if payload else {})
    resp = conn.getresponse()
    data = json.loads(resp.read().decode() or "{}")
    conn.close()
    return resp.status, data


def test_route_surface_is_exactly_three_typed_operations():
    assert ROUTES == {("POST", "/rag_answer"), ("GET", "/rag_run_status"), ("GET", "/rag_health")}


def test_health(server):
    code, body = call(server, "GET", "/rag_health")
    assert code == 200 and body["ok"] is True
    assert body["checkpointer"] == {"mode": "memory", "durable": False}
    assert body["synthesis"] is False


def test_rag_answer_ok(server):
    code, body = call(server, "POST", "/rag_answer", {"rag_answer": {"query": "x"}})
    assert code == 200 and body["ok"] is True and body["status"] == "ok"
    assert body["evidence"] and body["synthesis"] is None


def test_rag_answer_invalid_request_is_400(server):
    code, body = call(server, "POST", "/rag_answer", {"rag_answer": {"query": ""}})
    assert code == 400 and body["status"] == "invalid_request"


def test_rag_answer_rejects_identity_override(server):
    code, body = call(server, "POST", "/rag_answer", {"rag_answer": {"query": "x", "principal": "owner"}})
    assert code == 400 and body["status"] == "invalid_request"


def test_rag_run_status_readback(server):
    _, engine = server
    engine.run({"rag_answer": {"query": "x"}}, run_id="run-1")
    code, body = call(server, "GET", "/rag_run_status?run_id=run-1")
    assert code == 200 and body["ok"] is True and body["run_id"] == "run-1"


def test_rag_run_status_unknown_and_missing(server):
    code, body = call(server, "GET", "/rag_run_status?run_id=nope")
    assert code == 404 and body["ok"] is False
    code, body = call(server, "GET", "/rag_run_status")
    assert code == 400


@pytest.mark.parametrize("method", ["PUT", "DELETE", "PATCH"])
def test_write_methods_are_405(server, method):
    code, body = call(server, method, "/rag_answer")
    assert code == 405 and "write" in body["error"]


def test_unknown_routes_are_404(server):
    assert call(server, "GET", "/anything")[0] == 404
    assert call(server, "POST", "/other", {"x": 1})[0] == 404


def test_status_code_mapping_is_deterministic():
    assert status_code_for({"status": Outcome.OK.value}) == 200
    assert status_code_for({"status": Outcome.INVALID_REQUEST.value}) == 400
    assert status_code_for({"status": Outcome.STALE_INDEX.value}) == 503
    assert status_code_for({"status": Outcome.RETRIEVAL_TIMEOUT.value}) == 504
    assert status_code_for({"status": "unmapped"}) == 500


def test_main_refuses_non_loopback_bind():
    assert main(["--host", "0.0.0.0"]) == 2


def test_main_refuses_memory_checkpointer_without_opt_in():
    assert main(["--checkpoint", "memory"]) == 3


def test_rag_answer_honours_body_run_id_end_to_end(server):
    """F1 (HTTP): the documented body run_id is addressable via run_status."""
    code, body = call(server, "POST", "/rag_answer",
                      {"rag_answer": {"query": "x", "run_id": "my-run-id"}})
    assert code == 200 and body["run_id"] == "my-run-id"
    code, stored = call(server, "GET", "/rag_run_status?run_id=my-run-id")
    assert code == 200 and stored["run_id"] == "my-run-id"


def test_rag_answer_reused_run_id_different_body_is_isolated_end_to_end(engine_factory):
    """F2 (HTTP): a reused id with a different body returns its own evidence."""
    retrieval = QueryAwareRetrieval()
    engine = engine_factory(retrieval=retrieval)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        s = (httpd, engine)
        c1, b1 = call(s, "POST", "/rag_answer", {"rag_answer": {"query": "alpha", "run_id": "h-1"}})
        c2, b2 = call(s, "POST", "/rag_answer", {"rag_answer": {"query": "beta", "run_id": "h-1"}})
        assert c1 == 200 and b1["evidence"][0]["source_id"] == "src-alpha"
        assert c2 == 200 and b2["query"] == "beta"
        assert b2["evidence"][0]["source_id"] == "src-beta"
        assert retrieval.calls == 2
    finally:
        httpd.shutdown()
        httpd.server_close()
