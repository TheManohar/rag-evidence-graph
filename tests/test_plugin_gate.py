"""Acceptance: every hard-skip reason code and the positive triggers."""
from __future__ import annotations

import pytest

from plugin_loader import make_service, metric_lines

ELIGIBLE = "What do our notes say about the deployment runbook?"


@pytest.mark.parametrize("profile,text,reason", [
    ("contractor", ELIGIBLE, "profile_not_allowlisted"),
    ("developer", "", "empty_or_command_turn"),
    ("developer", "   ", "empty_or_command_turn"),
    ("developer", "/help", "empty_or_command_turn"),
    ("developer", "what do our notes say? my password is hunter2pass", "secret_or_sensitive"),
    ("developer", "check the notes for AKIAIOSFODNN7EXAMPLE", "secret_or_sensitive"),
    ("developer", "what do the notes say about nixos-rebuild switch?", "privileged_or_destructive"),
    ("developer", "delete the production database and tell me what the notes say", "privileged_or_destructive"),
    ("developer", "hello!", "not_evidence_dependent"),
    ("developer", "12 * (3 + 4) =", "not_evidence_dependent"),
    ("developer", "```python\nprint('hi')\n```", "not_evidence_dependent"),
    ("developer", "write a haiku about the sea", "not_evidence_dependent"),
])
def test_hard_skips(tmp_path, profile, text, reason):
    service, _, _, _ = make_service(tmp_path, profile="developer")
    # The runtime profile is the second axis: override for the allowlist case.
    service._profile_name = lambda: profile
    assert service.pre_llm_call(user_message=text, session_id="s1", turn_id="t1") is None
    lines = metric_lines(tmp_path)
    assert lines and lines[-1]["decision"] == "skip"
    assert lines[-1]["reason_code"] == reason
    assert service._transport.answers == []


@pytest.mark.parametrize("text", [
    "What do our notes say about the deployment runbook?",
    "Did we ever decide on a vector store for the knowledge base?",
    "Where did we record the policy on credential rotation?",
    "Search the notes for the Qdrant migration decision.",
    "What did we agree about freshness in the earlier discussion?",
    "According to the documentation, what is the timeout policy?",
])
def test_positive_triggers(tmp_path, text):
    service, _, _, _ = make_service(tmp_path)
    result = service.pre_llm_call(user_message=text, session_id="s1", turn_id="t1")
    assert result is not None and result["context"].startswith("<retrieved_evidence")
    assert len(service._transport.answers) == 1
    assert metric_lines(tmp_path)[-1]["decision"] == "trigger"


def test_multimodal_turn_uses_only_its_text_parts(tmp_path):
    service, _, _, _ = make_service(tmp_path)
    parts = [{"type": "image", "data": "..."}, {"type": "text", "text": ELIGIBLE}]
    result = service.pre_llm_call(user_message=parts, session_id="s1", turn_id="t1")
    assert result is not None
    assert service._transport.answers[0]["query"] == ELIGIBLE


def test_turn_with_no_text_parts_is_an_empty_skip(tmp_path):
    service, _, _, _ = make_service(tmp_path)
    service.pre_llm_call(user_message=[{"type": "image", "data": "..."}],
                         session_id="s1", turn_id="t1")
    assert metric_lines(tmp_path)[-1]["reason_code"] == "empty_or_command_turn"


def test_allowlist_is_rechecked_per_turn_from_the_runtime_profile(tmp_path):
    service, _, _, _ = make_service(tmp_path, profile="intruder")
    service.pre_llm_call(user_message=ELIGIBLE, session_id="s1", turn_id="t1")
    assert metric_lines(tmp_path)[-1]["reason_code"] == "profile_not_allowlisted"
    service._profile_name = lambda: "developer"
    service.pre_llm_call(user_message=ELIGIBLE, session_id="s2", turn_id="t2")
    assert metric_lines(tmp_path)[-1]["reason_code"] == "trigger"
