"""Acceptance: freshness gate (fail-closed)."""
from __future__ import annotations

import io
import time
import urllib.error

import pytest

from conftest import fresh_contract
from langgraph_rag import freshness as F
from langgraph_rag.errors import FreshnessError, Outcome


def test_fresh_contract_passes():
    v = F.evaluate_contract(fresh_contract(), expected_index_version="rag-v1",
                            sla_seconds=900, now=time.time())
    assert v.status == F.FRESH and v.fresh is True


@pytest.mark.parametrize("over,reason", [
    ({"failure_status": "failed"}, "failure"),
    ({"index_version": "rag-v0"}, "index_version"),
    ({"point_count": 0}, "empty"),
])
def test_stale_cases(over, reason):
    v = F.evaluate_contract(fresh_contract(**over), expected_index_version="rag-v1",
                            sla_seconds=900, now=time.time())
    assert v.status == F.STALE
    assert F.outcome_for(v) is Outcome.STALE_INDEX
    assert any(reason in r for r in v.reasons)


def test_stale_on_age_beyond_sla():
    now = time.time()
    v = F.evaluate_contract(fresh_contract(now=now, last_successful_index_write_at=int(now) - 3600),
                            expected_index_version="rag-v1", sla_seconds=900, now=now)
    assert v.status == F.STALE
    assert v.age_seconds and v.age_seconds > 900


@pytest.mark.parametrize("contract", [
    None, [], "x",
    {"ok": False},
    {"ok": True},  # missing fields
    {"ok": True, "last_successful_source_read_at": 1, "last_successful_index_write_at": "no",
     "index_version": "rag-v1", "point_count": 1, "failure_status": "ok"},
    fresh_contract(ok="yes"),
])
def test_unavailable_cases(contract):
    v = F.evaluate_contract(contract, expected_index_version="rag-v1",
                            sla_seconds=900, now=time.time())
    assert v.status == F.UNAVAILABLE
    assert F.outcome_for(v) is Outcome.FRESHNESS_UNAVAILABLE


def test_static_source_can_raise():
    src = F.StaticFreshnessSource(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        src.read()


def test_http_client_classifies_unreadable_contract(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    client = F.HttpFreshnessClient("http://127.0.0.1:8790/freshness")
    with pytest.raises(FreshnessError) as excinfo:
        client.read()
    assert excinfo.value.outcome is Outcome.FRESHNESS_UNAVAILABLE
