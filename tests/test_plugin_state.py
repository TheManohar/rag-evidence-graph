"""Acceptance: dedup, budget and circuit-breaker state machines."""
from __future__ import annotations

import pytest

from plugin_loader import FakeClock, submodule

budget_mod = submodule("budget")
breaker_mod = submodule("breaker")


# -- dedup ------------------------------------------------------------------
def test_same_turn_is_a_duplicate_within_the_ttl():
    clock = FakeClock()
    cache = budget_mod.DedupCache(ttl_seconds=300, clock=clock.time)
    kwargs = {"profile": "developer", "session_id": "s", "turn_id": "t", "fp": "abc"}
    assert cache.is_duplicate(**kwargs) is False
    cache.remember(**kwargs)
    assert cache.is_duplicate(**kwargs) is True


def test_same_fingerprint_in_a_session_is_a_duplicate_across_turns():
    clock = FakeClock()
    cache = budget_mod.DedupCache(ttl_seconds=300, clock=clock.time)
    cache.remember(profile="developer", session_id="s", turn_id="t1", fp="abc")
    assert cache.is_duplicate(profile="developer", session_id="s", turn_id="t2", fp="abc") is True
    assert cache.is_duplicate(profile="developer", session_id="other", turn_id="t2", fp="abc") is False
    assert cache.is_duplicate(profile="reviewer", session_id="s", turn_id="t2", fp="abc") is False


def test_dedup_ttl_expires():
    clock = FakeClock()
    cache = budget_mod.DedupCache(ttl_seconds=10, clock=clock.time)
    cache.remember(profile="developer", session_id="s", turn_id="t", fp="abc")
    clock.advance(11)
    assert cache.is_duplicate(profile="developer", session_id="s", turn_id="t", fp="abc") is False


# -- budget -----------------------------------------------------------------
def test_per_turn_budget_is_one():
    clock = FakeClock()
    tracker = budget_mod.BudgetTracker(per_turn=1, per_session=20, per_profile_hourly=120,
                                       clock=clock.time)
    assert tracker.consume(profile="developer", session_id="s", turn_id="t") is True
    assert tracker.consume(profile="developer", session_id="s", turn_id="t") is False
    assert tracker.consume(profile="developer", session_id="s", turn_id="t2") is True


def test_per_session_budget():
    clock = FakeClock()
    tracker = budget_mod.BudgetTracker(per_turn=100, per_session=2, per_profile_hourly=120,
                                       clock=clock.time)
    assert tracker.consume(profile="developer", session_id="s", turn_id="1")
    assert tracker.consume(profile="developer", session_id="s", turn_id="2")
    assert tracker.consume(profile="developer", session_id="s", turn_id="3") is False
    assert tracker.consume(profile="developer", session_id="other", turn_id="3") is True


def test_per_profile_hourly_budget_resets_next_hour():
    clock = FakeClock()
    tracker = budget_mod.BudgetTracker(per_turn=100, per_session=100,
                                       per_profile_hourly=2, clock=clock.time)
    assert tracker.consume(profile="developer", session_id="a", turn_id="1")
    assert tracker.consume(profile="developer", session_id="b", turn_id="2")
    assert tracker.consume(profile="developer", session_id="c", turn_id="3") is False
    clock.advance(3600)
    assert tracker.consume(profile="developer", session_id="c", turn_id="4") is True


# -- breaker ----------------------------------------------------------------
def test_breaker_opens_after_two_failures_and_blocks():
    clock = FakeClock()
    breaker = breaker_mod.CircuitBreaker(failures=2, cooldown_seconds=60, clock=clock.monotonic)
    assert breaker.check() == breaker_mod.CLOSED
    breaker.record_failure()
    assert breaker.state == breaker_mod.CLOSED
    breaker.record_failure()
    assert breaker.state == breaker_mod.OPEN
    assert breaker.check() == breaker_mod.OPEN


def test_breaker_half_opens_after_cooldown_with_one_probe():
    clock = FakeClock()
    breaker = breaker_mod.CircuitBreaker(failures=1, cooldown_seconds=30, clock=clock.monotonic)
    breaker.record_failure()
    assert breaker.check() == breaker_mod.OPEN
    clock.advance(31)
    assert breaker.check() == breaker_mod.HALF_OPEN
    # Only one probe at a time.
    assert breaker.check() == breaker_mod.OPEN
    breaker.record_success()
    assert breaker.state == breaker_mod.CLOSED
    assert breaker.check() == breaker_mod.CLOSED


def test_failed_half_open_probe_reopens_with_a_fresh_cooldown():
    clock = FakeClock()
    breaker = breaker_mod.CircuitBreaker(failures=1, cooldown_seconds=30, clock=clock.monotonic)
    breaker.record_failure()
    clock.advance(31)
    assert breaker.check() == breaker_mod.HALF_OPEN
    breaker.record_failure()
    assert breaker.state == breaker_mod.OPEN
    assert breaker.check() == breaker_mod.OPEN
    clock.advance(31)
    assert breaker.check() == breaker_mod.HALF_OPEN


def test_success_resets_the_consecutive_failure_count():
    clock = FakeClock()
    breaker = breaker_mod.CircuitBreaker(failures=2, cooldown_seconds=60, clock=clock.monotonic)
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    assert breaker.state == breaker_mod.CLOSED
    assert breaker.consecutive_failures == 1
