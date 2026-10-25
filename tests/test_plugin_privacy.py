"""Privacy regression: host-shaped identifiers must never reach persisted metrics.

Hermes builds ``turn_id`` as ``"<session_id>:<task_id>:<8 hex>"`` (installed
``agent.turn_context._bind_turn_identity``), and the CLI can also pass its
session id as ``task_id`` (``cli_session_mixin``), so both fields can carry a
raw session identifier.  Only a deterministic, one-way digest may be written.
"""
from __future__ import annotations

import json
import re

from plugin_loader import make_service, metric_lines, submodule

ELIGIBLE = "What do our notes say about the deployment runbook?"
SKIP = "hello"
# Synthetic host-shaped identifiers; no real user/session value is used here.
SESSION = "fixture-session-marker-4f8c2e"
TASK = "0b6f1a2c-1111-4222-8333-944455556666"
OPAQUE = re.compile(r"^sha256:[0-9a-f]{16}$")


def host_turn(suffix: str) -> str:
    """Same shape the installed host produces in _bind_turn_identity."""
    return f"{SESSION}:{TASK}:{suffix}"


def test_host_shaped_trigger_and_skip_metrics_are_opaque(tmp_path):
    service, _, _, _ = make_service(tmp_path)
    trigger_turn, skip_turn = host_turn("a1b2c3d4"), host_turn("e5f6a7b8")
    service.pre_llm_call(user_message=ELIGIBLE, session_id=SESSION,
                         turn_id=trigger_turn, task_id=TASK)
    service.pre_llm_call(user_message=SKIP, session_id=SESSION,
                         turn_id=skip_turn, task_id=TASK)

    raw = (tmp_path / "metrics.jsonl").read_text()
    lines = metric_lines(tmp_path)
    assert [line["decision"] for line in lines] == ["trigger", "skip"]
    assert SESSION not in raw
    assert TASK not in raw
    assert trigger_turn not in raw and skip_turn not in raw
    for line in lines:
        assert OPAQUE.match(line["turn_id"]), line
        assert OPAQUE.match(line["task_id"]), line
        assert line["session_hash"] and line["session_hash"] != SESSION
    assert lines[0]["turn_id"] != lines[1]["turn_id"]


def test_opaque_turn_id_is_stable_and_internal_dedup_still_works(tmp_path):
    service, _, _, _ = make_service(tmp_path)
    turn_id = host_turn("0a1b2c3d")
    assert service.pre_llm_call(user_message=ELIGIBLE, session_id=SESSION,
                                turn_id=turn_id, task_id=TASK)
    first = metric_lines(tmp_path)[-1]["turn_id"]
    assert service.pre_llm_call(user_message=ELIGIBLE, session_id=SESSION,
                                turn_id=turn_id, task_id=TASK) is None
    lines = metric_lines(tmp_path)
    assert [line["reason_code"] for line in lines] == ["trigger", "duplicate_turn"]
    assert lines[-1]["turn_id"] == first
    assert SESSION not in (tmp_path / "metrics.jsonl").read_text()


def test_metric_sink_sanitizes_records_that_bypass_build_record(tmp_path):
    metrics = submodule("metrics")
    sink = metrics.MetricSink(tmp_path / "metrics.jsonl")
    record = metrics.build_record(profile="developer", platform="cli",
                                  decision="skip", reason_code="fixture",
                                  session_hash="h", turn_id=host_turn("c0ffee00"),
                                  task_id=TASK)
    # A caller that hand-builds or mutates a record must not bypass the sink.
    record["turn_id"] = host_turn("c0ffee00")
    record["task_id"] = SESSION
    assert sink.emit(record)
    raw = (tmp_path / "metrics.jsonl").read_text()
    assert SESSION not in raw
    line = json.loads(raw)
    assert OPAQUE.match(line["turn_id"]) and OPAQUE.match(line["task_id"])
