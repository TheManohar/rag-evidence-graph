"""Acceptance: gate integration — injection, metrics, breaker, fail-open, tool."""
from __future__ import annotations

import json
import time

from plugin_loader import (FakeTransport, citation, make_service, metric_lines,
                           ok_response, submodule)

GraphTransportError = submodule("transport").GraphTransportError

ELIGIBLE = "What do our notes say about the deployment runbook?"


def turn(i: int) -> str:
    """Same intent, different fingerprint (dedup is per exact turn text)."""
    return ELIGIBLE + f" Variant {i}?"


def test_eligible_turn_injects_one_bounded_block_and_emits_one_metric(tmp_path):
    service, _, _, _ = make_service(tmp_path)
    result = service.pre_llm_call(user_message=ELIGIBLE, session_id="s", turn_id="t",
                                  platform="cli", task_id="k1")
    assert result is not None
    block = result["context"]
    assert block.startswith("<retrieved_evidence")
    assert block.rstrip().endswith("</retrieved_evidence>")
    call = service._transport.answers[0]
    assert call["query"] == ELIGIBLE
    assert call["top_k"] == 5 and call["mode"] == "hybrid"
    assert call["run_id"].startswith("hermes-auto:") and call["run_id"].count(":") == 2
    lines = metric_lines(tmp_path)
    assert len(lines) == 1
    line = lines[0]
    assert line["decision"] == "trigger" and line["status"] == "ok"
    assert line["evidence_count"] == 1
    assert line["injected_chars"] == len(block)
    assert line["fail_open"] is False and line["shadow"] is False
    assert line["session_hash"] and line["session_hash"] != "s"
    assert line["platform"] == "cli"
    assert line["task_id"].startswith("sha256:") and line["task_id"] != "k1"


def test_metrics_never_contain_the_query_or_a_snippet(tmp_path):
    service, _, _, _ = make_service(tmp_path)
    marker = "QUERY-MARKER-77"
    snippet = "SNIPPET-MARKER-88"
    service._transport.response = ok_response([citation(snippet=snippet)])
    service.pre_llm_call(user_message=f"{ELIGIBLE} {marker}", session_id="s", turn_id="t")
    raw = (tmp_path / "metrics.jsonl").read_text()
    assert marker not in raw and snippet not in raw
    assert "query" not in raw


def test_duplicate_turn_makes_no_second_graph_call(tmp_path):
    transport = FakeTransport()
    service, _, _, _ = make_service(tmp_path, transport=transport)
    assert service.pre_llm_call(user_message=ELIGIBLE, session_id="s", turn_id="t")
    assert service.pre_llm_call(user_message=ELIGIBLE, session_id="s", turn_id="t") is None
    assert len(transport.answers) == 1
    assert metric_lines(tmp_path)[-1]["reason_code"] == "duplicate_turn"


def test_same_question_later_in_the_session_is_deduplicated(tmp_path):
    transport = FakeTransport()
    service, _, clock, _ = make_service(tmp_path, transport=transport)
    question = "What do our notes say about the same thing?"
    service.pre_llm_call(user_message=question, session_id="s", turn_id="t1")
    service.pre_llm_call(user_message=question, session_id="s", turn_id="t2")
    assert len(transport.answers) == 1
    assert metric_lines(tmp_path)[-1]["reason_code"] == "duplicate_turn"
    clock.advance(301)
    service.pre_llm_call(user_message=question, session_id="s", turn_id="t3")
    assert len(transport.answers) == 2


def test_budget_exhausted_is_a_logged_skip(tmp_path):
    transport = FakeTransport()
    service, _, _, _ = make_service(tmp_path, transport=transport,
                                    config={"budget.per_session": 1})
    assert service.pre_llm_call(user_message=turn(1), session_id="s", turn_id="t1")
    assert service.pre_llm_call(user_message=ELIGIBLE + " and the policy?",
                                session_id="s", turn_id="t2") is None
    assert len(transport.answers) == 1
    assert metric_lines(tmp_path)[-1]["reason_code"] == "budget_exhausted"


def test_two_terminal_failures_open_the_breaker_and_the_third_turn_skips(tmp_path):
    transport = FakeTransport(response=ok_response(status="stale_index", evidence=[]))
    service, _, _, _ = make_service(tmp_path, transport=transport)
    assert service.pre_llm_call(user_message=turn(1), session_id="s", turn_id="t1") is None
    assert service.pre_llm_call(user_message=turn(2), session_id="s", turn_id="t2") is None
    assert service._breaker.state == "open"
    assert service.pre_llm_call(user_message=turn(3), session_id="s", turn_id="t3") is None
    assert len(transport.answers) == 2
    assert metric_lines(tmp_path)[-1]["reason_code"] == "graph_unhealthy"
    assert metric_lines(tmp_path)[-1]["status"] == "breaker_open"


def test_half_open_probe_health_ok_closes_the_breaker(tmp_path):
    transport = FakeTransport(response=ok_response(status="stale_index", evidence=[]))
    service, _, clock, _ = make_service(tmp_path, transport=transport)
    service.pre_llm_call(user_message=turn(1), session_id="s", turn_id="t1")
    service.pre_llm_call(user_message=turn(2), session_id="s", turn_id="t2")
    assert service._breaker.state == "open"
    clock.advance(61)
    transport.response = ok_response()
    assert service.pre_llm_call(user_message=turn(3), session_id="s", turn_id="t3")
    assert transport.health_calls == 1
    assert service._breaker.state == "closed"


def test_half_open_probe_failure_keeps_the_breaker_open(tmp_path):
    transport = FakeTransport(response=ok_response(status="stale_index", evidence=[]),
                              health={"ok": False})
    service, _, clock, _ = make_service(tmp_path, transport=transport)
    service.pre_llm_call(user_message=turn(1), session_id="s", turn_id="t1")
    service.pre_llm_call(user_message=turn(2), session_id="s", turn_id="t2")
    clock.advance(61)
    before = len(transport.answers)
    assert service.pre_llm_call(user_message=turn(3), session_id="s", turn_id="t3") is None
    assert len(transport.answers) == before
    assert service._breaker.state == "open"
    assert metric_lines(tmp_path)[-1]["status"] == "breaker_half_open_failed"


def test_transport_timeout_fails_open_visibly(tmp_path):
    transport = FakeTransport(error=GraphTransportError("timeout", "timed out"))
    service, _, _, _ = make_service(tmp_path, transport=transport)
    assert service.pre_llm_call(user_message=ELIGIBLE, session_id="s", turn_id="t") is None
    line = metric_lines(tmp_path)[-1]
    assert line["decision"] == "trigger"
    assert line["status"] == "timeout"
    assert line["fail_open"] is True and line["error_class"] == "timeout"
    assert line["injected_chars"] == 0
    assert service._breaker.consecutive_failures == 1  # one failure is not yet open


def test_a_hung_transport_is_abandoned_at_the_deadline(tmp_path):
    transport = FakeTransport(delay=0.6)
    service, _, _, cfg = make_service(tmp_path, transport=transport,
                                      config={"deadline.seconds": 0.6,
                                              "deadline.graph_seconds": 0.2})
    started = time.monotonic()
    assert service.pre_llm_call(user_message=ELIGIBLE, session_id="s", turn_id="t") is None
    elapsed = time.monotonic() - started
    assert elapsed < 0.55
    line = metric_lines(tmp_path)[-1]
    assert line["status"] == "timeout" and line["fail_open"] is True
    assert line["error_class"] == "timeout"


def test_unexpected_exception_fails_open(tmp_path):
    class Exploding(FakeTransport):
        def answer(self, **kwargs):
            raise RuntimeError("boom")

    service, _, _, _ = make_service(tmp_path, transport=Exploding())
    assert service.pre_llm_call(user_message=ELIGIBLE, session_id="s", turn_id="t") is None
    line = metric_lines(tmp_path)[-1]
    assert line["status"] == "error" and line["fail_open"] is True
    assert line["error_class"] == "RuntimeError"


def test_shadow_mode_decides_and_logs_but_injects_nothing(tmp_path):
    service, _, _, _ = make_service(tmp_path, config={"inject": False})
    assert service.pre_llm_call(user_message=ELIGIBLE, session_id="s", turn_id="t") is None
    line = metric_lines(tmp_path)[-1]
    assert line["decision"] == "trigger" and line["shadow"] is True
    assert line["injected_chars"] == 0
    assert service._transport.answers  # the decision was still measured


def test_non_ok_graph_status_injects_nothing_but_records_the_status(tmp_path):
    service, _, _, _ = make_service(
        tmp_path, transport=FakeTransport(response=ok_response(status="low_relevance", evidence=[])))
    assert service.pre_llm_call(user_message=ELIGIBLE, session_id="s", turn_id="t") is None
    line = metric_lines(tmp_path)[-1]
    assert line["status"] == "low_relevance" and line["injected_chars"] == 0
    assert line["fail_open"] is False


def test_replayed_graph_run_is_recorded(tmp_path):
    service, _, _, _ = make_service(
        tmp_path, transport=FakeTransport(response=ok_response(replayed=True)))
    service.pre_llm_call(user_message=ELIGIBLE, session_id="s", turn_id="t")
    assert metric_lines(tmp_path)[-1]["replayed"] is True


def test_every_evaluated_turn_produces_exactly_one_metric_line(tmp_path):
    service, _, _, _ = make_service(tmp_path)
    service.pre_llm_call(user_message="hello", session_id="s", turn_id="t1")
    service.pre_llm_call(user_message=turn(1), session_id="s", turn_id="t2")
    service.pre_llm_call(user_message=turn(1), session_id="s", turn_id="t2")
    assert len(metric_lines(tmp_path)) == 3


def test_a_full_triggered_turn_opens_no_socket_when_the_transport_is_faked(tmp_path, monkeypatch):
    import socket as socket_mod

    def forbidden(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("the gate must not open a socket with a fake transport")

    service, _, _, _ = make_service(tmp_path)
    monkeypatch.setattr(socket_mod, "socket", forbidden)
    result = service.pre_llm_call(user_message=ELIGIBLE, session_id="s", turn_id="t")
    assert result is not None and result["context"].startswith("<retrieved_evidence")


# -- explicit tool ----------------------------------------------------------
def test_tool_returns_bounded_json_with_evidence(tmp_path):
    service, _, _, _ = make_service(tmp_path)
    payload = json.loads(service.rag_answer_tool({"query": "what did we decide about freshness?"}))
    assert payload["ok"] is True and payload["status"] == "ok"
    assert payload["evidence"] and payload["untrusted"] is True
    assert payload["run_id"].startswith("hermes-tool:")
    assert "not instructions" in payload["note"]


def test_tool_rejects_a_non_allowlisted_profile(tmp_path):
    service, _, _, _ = make_service(tmp_path, profile="intruder")
    payload = json.loads(service.rag_answer_tool({"query": "anything"}))
    assert payload["ok"] is False
    assert payload["reason_code"] == "profile_not_allowlisted"
    assert service._transport.answers == []


def test_tool_rejects_bad_queries(tmp_path):
    service, _, _, _ = make_service(tmp_path)
    assert json.loads(service.rag_answer_tool({}))["ok"] is False
    assert json.loads(service.rag_answer_tool({"query": "x" * 2001}))["ok"] is False
    assert service._transport.answers == []


def test_tool_fails_fast_when_the_breaker_is_open(tmp_path):
    transport = FakeTransport(response=ok_response(status="retrieval_timeout", evidence=[]))
    service, _, _, _ = make_service(tmp_path, transport=transport)
    service.rag_answer_tool({"query": "one"})
    service.rag_answer_tool({"query": "two"})
    assert service._breaker.state == "open"
    payload = json.loads(service.rag_answer_tool({"query": "three"}))
    assert payload["status"] == "graph_unhealthy"
    assert len(transport.answers) == 2
