"""Bounded turn/session/profile budgets and duplicate suppression."""
from __future__ import annotations

import threading
import time


class BudgetTracker:
    """Per-turn (1), per-session (20) and per-profile hourly retrieval budget.

    ``consume`` both checks and increments under one lock, so two concurrent
    turns cannot both pass the last slot.
    """

    def __init__(self, *, per_turn: int = 1, per_session: int = 20,
                 per_profile_hourly: int = 120, clock=time.time,
                 max_sessions: int = 4096) -> None:
        self.per_turn = max(1, int(per_turn))
        self.per_session = max(1, int(per_session))
        self.per_profile_hourly = max(1, int(per_profile_hourly))
        self._clock = clock
        self._max_sessions = max(8, int(max_sessions))
        self._lock = threading.Lock()
        self._turn: dict[tuple[str, str, str], float] = {}
        self._session: dict[tuple[str, str], list[float]] = {}
        self._hour: dict[tuple[str, int], int] = {}

    def consume(self, *, profile: str, session_id: str, turn_id: str) -> bool:
        now = self._clock()
        turn_key = (profile, session_id, turn_id)
        session_key = (profile, session_id)
        hour_key = (profile, int(now // 3600))
        with self._lock:
            self._prune(now, hour_key)
            if self._turn.get(turn_key, 0.0) > now:
                return False
            if len(self._session.get(session_key, [])) >= self.per_session:
                return False
            if self._hour.get(hour_key, 0) >= self.per_profile_hourly:
                return False
            self._turn[turn_key] = now + 3600.0
            self._session.setdefault(session_key, []).append(now)
            self._hour[hour_key] = self._hour.get(hour_key, 0) + 1
            return True

    def _prune(self, now: float, current_hour: tuple[str, int]) -> None:
        if len(self._turn) > 512:
            self._turn = {k: v for k, v in self._turn.items() if v > now}
        if len(self._session) > self._max_sessions:
            # Keep the most recently used sessions; each list is bounded by per_session.
            for key in list(self._session)[: len(self._session) - self._max_sessions]:
                self._session.pop(key, None)
        stale_hours = [k for k in self._hour if k[1] < current_hour[1] - 1]
        for key in stale_hours:
            self._hour.pop(key, None)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "turns": len(self._turn),
                "sessions": len(self._session),
                "hours": len(self._hour),
            }


class DedupCache:
    """TTL cache of (turn, session-fingerprint) injections.

    Layer 1 of design.md §6: a turn is injected at most once, and the same
    fingerprint is not retrieved twice inside one session within the TTL.
    """

    def __init__(self, *, ttl_seconds: float = 300.0, clock=time.time,
                 max_entries: int = 2048) -> None:
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self._clock = clock
        self._max_entries = max(16, int(max_entries))
        self._lock = threading.Lock()
        self._entries: dict[tuple, float] = {}

    def _turn_key(self, profile: str, session_id: str, turn_id: str) -> tuple:
        return ("turn", profile, session_id, turn_id)

    def _fp_key(self, profile: str, session_id: str, fp: str) -> tuple:
        return ("fp", profile, session_id, fp)

    def is_duplicate(self, *, profile: str, session_id: str, turn_id: str, fp: str) -> bool:
        now = self._clock()
        with self._lock:
            self._prune(now)
            for key in (self._turn_key(profile, session_id, turn_id),
                        self._fp_key(profile, session_id, fp)):
                expiry = self._entries.get(key)
                if expiry is not None and expiry > now:
                    return True
        return False

    def remember(self, *, profile: str, session_id: str, turn_id: str, fp: str) -> None:
        now = self._clock()
        expiry = now + self.ttl_seconds
        with self._lock:
            self._prune(now)
            self._entries[self._turn_key(profile, session_id, turn_id)] = expiry
            self._entries[self._fp_key(profile, session_id, fp)] = expiry
            if len(self._entries) > self._max_entries:
                for key in list(self._entries)[: len(self._entries) - self._max_entries]:
                    self._entries.pop(key, None)

    def _prune(self, now: float) -> None:
        if len(self._entries) <= self._max_entries // 2:
            return
        self._entries = {k: v for k, v in self._entries.items() if v > now}
