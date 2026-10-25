"""Circuit breaker for the graph leg.

Open after ``failures`` consecutive terminal failures (default 2, per PLAN.md
§6.3), cool down, then allow exactly one half-open probe. The probe is a
``/rag_health`` read, not a retrieval: while the graph is unhealthy the gate
must fail fast with ``graph_unhealthy`` instead of burning the turn budget.
"""
from __future__ import annotations

import threading
import time

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, *, failures: int = 2, cooldown_seconds: float = 60.0,
                 clock=time.monotonic) -> None:
        self.failures = max(1, int(failures))
        self.cooldown_seconds = max(1.0, float(cooldown_seconds))
        self._clock = clock
        self._lock = threading.Lock()
        self._state = CLOSED
        self._consecutive = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def consecutive_failures(self) -> int:
        with self._lock:
            return self._consecutive

    def check(self) -> str:
        """Return ``closed``, ``open`` or ``half_open`` for one attempt.

        A ``half_open`` result claims the single probe slot for this caller; the
        caller must report ``record_success``/``record_failure``.
        """
        now = self._clock()
        with self._lock:
            if self._state == CLOSED:
                return CLOSED
            if self._state == OPEN:
                if self._opened_at is not None and now - self._opened_at >= self.cooldown_seconds:
                    if not self._probe_in_flight:
                        self._probe_in_flight = True
                        self._state = HALF_OPEN
                        return HALF_OPEN
                return OPEN
            return OPEN if self._probe_in_flight else OPEN

    def record_success(self) -> None:
        with self._lock:
            self._state = CLOSED
            self._consecutive = 0
            self._opened_at = None
            self._probe_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive += 1
            self._probe_in_flight = False
            if self._state == HALF_OPEN or self._consecutive >= self.failures:
                self._state = OPEN
                self._opened_at = self._clock()

    def reset(self) -> None:
        self.record_success()
