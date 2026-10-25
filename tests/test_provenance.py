"""Acceptance: redacted adapter provenance, one line per /rag_answer request."""
from __future__ import annotations

import http.client
import json
import threading
import time
from http.server import ThreadingHTTPServer

import pytest
from conftest import FakeRetrieval, citation, fresh_contract, retrieval_response

from langgraph_rag.adapter import make_handler, status_code_for
from langgraph_rag.errors import Outcome
from langgraph_rag.freshness import StaticFreshnessSource
from langgraph_rag.graph import RagEngine
from langgraph_rag.checkpoint import InMemoryCheckpointer
from langgraph_rag.metrics import (CALLER_UNKNOWN, PROVENANCE_FIELDS, MetricsWriter,
                                   provenance_record, record_for_response)

SECRET_QUERY = "SECRET-QUERY-TOKEN-9f1c"
SECRET_SNIPPET = "UNIQUE-SNIPPET-6174"


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def call(httpd, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
    payload = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, body=payload,
                 headers={"Content-Type": "application/json"} if payload else {})
    resp = conn.getresponse()
    data = json.loads(resp.read().decode() or "{}")
    code = resp.status
    conn.close()
    return code, data


class SlowRetrieval(FakeRetrieval):
    def __init__(self, delay: float, response=None):
        super().__init__(response)
        self.delay = delay

    def search(self, *, query, filters, top_k, mode, timeout=None):
        time.sleep(self.delay)
        return super().search(query=query, filters=filters, top_k=top_k, mode=mode)


@pytest.fixture
def provenance_server(tmp_path, engine_factory):
    metrics_path = tmp_path / "metrics" / "rag.jsonl"
    engine = engine_factory(retrieval=FakeRetrieval(retrieval_response(
        [citation(snippet=SECRET_SNIPPET)])))
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0), make_handler(engine, metrics=MetricsWriter(metrics_path)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd, engine, metrics_path
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_one_line_per_request_with_the_exact_redacted_schema(provenance_server):
    httpd, _, path = provenance_server
    code, _ = call(httpd, "POST", "/rag_answer", {"rag_answer": {"query": SECRET_QUERY}})
    assert code == 200
    lines = read_lines(path)
    assert len(lines) == 1
    line = lines[0]
    assert tuple(sorted(line)) == tuple(sorted(PROVENANCE_FIELDS))
    assert line["status"] == "ok"
    assert line["evidence_count"] == 1
    assert line["index_version"] == "rag-v1"
    assert line["replayed"] is False
    assert line["caller"] == CALLER_UNKNOWN
    assert isinstance(line["wall_ms"], float) and line["wall_ms"] >= 0
    assert 55 <= line["freshness_age"] <= 65
    assert line["run_id"].startswith("rag-")


def test_metrics_never_contain_query_text_or_snippets(provenance_server):
    httpd, _, path = provenance_server
    call(httpd, "POST", "/rag_answer", {"rag_answer": {"query": SECRET_QUERY}})
    raw = path.read_text()
    assert SECRET_QUERY not in raw
    assert SECRET_SNIPPET not in raw
    assert "snippet" not in raw
    assert "query" not in raw


def test_replay_is_visible_in_provenance(provenance_server):
    httpd, _, path = provenance_server
    body = {"rag_answer": {"query": "same question", "run_id": "auto-1"}}
    assert call(httpd, "POST", "/rag_answer", body)[0] == 200
    assert call(httpd, "POST", "/rag_answer", body)[0] == 200
    lines = read_lines(path)
    assert len(lines) == 2
    assert [line["replayed"] for line in lines] == [False, True]
    assert {line["run_id"] for line in lines} == {"auto-1"}


def test_rejected_request_still_emits_exactly_one_line(provenance_server):
    httpd, _, path = provenance_server
    code, _ = call(httpd, "POST", "/rag_answer", {"rag_answer": {"query": ""}})
    assert code == 400
    lines = read_lines(path)
    assert len(lines) == 1
    assert lines[0]["status"] == "invalid_request"
    assert lines[0]["evidence_count"] == 0


def test_invalid_json_emits_one_line(provenance_server):
    httpd, _, path = provenance_server
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
    conn.request("POST", "/rag_answer", body=b"{not json",
                 headers={"Content-Type": "application/json"})
    assert conn.getresponse().status == 400
    conn.close()
    lines = read_lines(path)
    assert len(lines) == 1 and lines[0]["status"] == "invalid_request"


def test_deadline_emits_a_visible_line(tmp_path, engine_factory):
    metrics_path = tmp_path / "rag.jsonl"
    engine = engine_factory(retrieval=SlowRetrieval(0.5, retrieval_response([citation()])),
                            request_deadline=0.1)
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0), make_handler(engine, metrics=MetricsWriter(metrics_path)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        started = time.monotonic()
        code, body = call(httpd, "POST", "/rag_answer",
                          {"rag_answer": {"query": "slow", "run_id": "auto-slow"}})
        elapsed = time.monotonic() - started
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert code == 504
    assert body["status"] == "deadline_exceeded" and body["deadline_exceeded"] is True
    assert elapsed < 0.5
    lines = read_lines(metrics_path)
    assert len(lines) == 1
    assert lines[0]["status"] == "deadline_exceeded"
    assert lines[0]["run_id"] == "auto-slow"
    assert lines[0]["evidence_count"] == 0
    assert lines[0]["replayed"] is False


def test_metrics_write_failure_never_fails_the_request(tmp_path, engine_factory):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    writer = MetricsWriter(blocker / "rag.jsonl")
    engine = engine_factory(retrieval=FakeRetrieval(retrieval_response([citation()])))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine, metrics=writer))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        code, body = call(httpd, "POST", "/rag_answer", {"rag_answer": {"query": "x"}})
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert code == 200 and body["ok"] is True
    assert writer.errors >= 1


def test_writer_rotates_by_size(tmp_path):
    path = tmp_path / "rag.jsonl"
    writer = MetricsWriter(path, max_bytes=80)
    for i in range(3):
        writer.emit(provenance_record(run_id=f"r{i}", status="ok", wall_ms=1.0))
    assert path.exists()
    assert (tmp_path / "rag.jsonl.1").exists()


def test_provenance_record_drops_unknown_types_and_never_raises():
    record = provenance_record(run_id=None, status=None, wall_ms="nope",
                               evidence_count="7", freshness_age=float("nan"),
                               index_version=3, replayed=1)
    assert record["run_id"] == "unknown"
    assert record["status"] == "unknown"
    assert record["wall_ms"] is None
    assert record["evidence_count"] == 0
    assert record["freshness_age"] is None
    assert record["index_version"] is None
    assert record["replayed"] is True


def test_record_for_response_tolerates_a_non_dict_response():
    record = record_for_response(None, wall_ms=2.0, status="internal_error")
    assert record["evidence_count"] == 0
    assert record["status"] == "internal_error"
    assert record["caller"] == CALLER_UNKNOWN


def test_health_reports_the_request_deadline(engine_factory):
    engine = engine_factory(request_deadline=5.0)
    assert engine.health()["request_deadline_seconds"] == 5.0


def test_deadline_status_maps_to_504():
    assert status_code_for({"status": Outcome.DEADLINE_EXCEEDED.value}) == 504


def test_a_slow_freshness_read_stops_the_request_before_retrieval():
    """The engine checks the budget between nodes, so the retrieve never starts."""

    class SlowFreshness:
        def read(self, *, timeout=None):
            time.sleep(0.15)
            return fresh_contract()

    retrieval = FakeRetrieval(retrieval_response([citation()]))
    engine = RagEngine(retrieval=retrieval, freshness=SlowFreshness(),
                       checkpointer=InMemoryCheckpointer(), request_deadline=0.1)
    started = time.monotonic()
    response = engine.run({"rag_answer": {"query": "budget"}})
    elapsed = time.monotonic() - started
    assert response["status"] == "deadline_exceeded"
    assert response["deadline_exceeded"] is True
    assert retrieval.calls == 0
    assert elapsed < 0.5
