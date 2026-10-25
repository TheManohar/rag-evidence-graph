"""Freshness gate.

The gate consumes the **published freshness contract** served by the
projection boundary — the component the n8n incremental projection workflow
writes the index through. The contract is published by kanban task
``t_f02440bb`` as ``GET /freshness`` on the projection service
(``modules/hermes-rag/rag/projection_service.py``); the evidence behind it is
recording code in ``rag/freshness_state.py``. The gate does not invent an
independent staleness source.

Read path consumed (documented interface)
------------------------------------------
``FreshnessSource.read()`` issues a read-only ``GET`` to the configured
freshness endpoint. The endpoint returns a JSON object with the following
fields (all of them required):

===============================  =======  =====================================
field                            type     meaning
===============================  =======  =====================================
``ok``                           bool     contract readback succeeded
``last_successful_source_read_at``  int   unix seconds (UTC), last successful snapshot read
``last_successful_index_write_at``  int   unix seconds (UTC), last successful index write
``index_version``                str      index version currently projected
``point_count``                  int      points currently in the collection (live)
``failure_status``               str      ``"ok"`` or a failure label
===============================  =======  =====================================

Extra fields published alongside these (``age_seconds``, ``checked_at``,
``recorded``, ``last_attempt_*``, ...) are additive and ignored here. The
endpoint URL is a service option (``--freshness-url``) so it can be reconciled
at admission without a code change. The gate remains **fail-closed**: an
unreadable/absent contract yields ``freshness_unavailable`` and a failed or
over-SLA contract yields ``stale_index``.

Decision rules (all fail-closed):

* contract unreadable / not an object / missing required fields / ``ok`` false
  -> ``freshness_unavailable``
* ``failure_status`` != ``"ok"``                      -> ``stale_index``
* ``index_version`` != expected                      -> ``stale_index``
* ``point_count`` <= 0                               -> ``stale_index``
* ``now - last_successful_index_write_at`` > SLA     -> ``stale_index``
* otherwise                                           -> ``fresh``
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from .errors import FreshnessError, Outcome

REQUIRED_FIELDS = (
    "ok",
    "last_successful_source_read_at",
    "last_successful_index_write_at",
    "index_version",
    "point_count",
    "failure_status",
)

FRESH = "fresh"
STALE = "stale"
UNAVAILABLE = "unavailable"

DEFAULT_SLA_SECONDS = 900  # existing 15-minute freshness SLA


class FreshnessSource(Protocol):
    """Read-only freshness contract source.

    ``timeout`` (optional) is the caller's remaining wall-clock budget; a
    source that cannot honour it may ignore it, but an HTTP source must lower
    its own request timeout to fit.
    """

    def read(self, *, timeout: float | None = None) -> dict: ...


@dataclass(frozen=True)
class FreshnessVerdict:
    status: str
    reasons: tuple[str, ...] = ()
    contract: dict = field(default_factory=dict)
    age_seconds: float | None = None

    @property
    def fresh(self) -> bool:
        return self.status == FRESH

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "fresh": self.fresh,
            "reasons": list(self.reasons),
            "age_seconds": self.age_seconds,
            "contract": self.contract,
        }


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def evaluate_contract(
    contract: Any,
    *,
    expected_index_version: str = "rag-v1",
    sla_seconds: int = DEFAULT_SLA_SECONDS,
    now: float | None = None,
) -> FreshnessVerdict:
    """Evaluate a freshness contract into a deterministic verdict."""
    now = time.time() if now is None else now
    if not isinstance(contract, dict):
        return FreshnessVerdict(UNAVAILABLE, ("freshness contract is not a JSON object",))
    missing = [f for f in REQUIRED_FIELDS if f not in contract]
    if missing:
        return FreshnessVerdict(UNAVAILABLE, (f"freshness contract missing field(s): {missing}",))
    if contract.get("ok") is not True:
        return FreshnessVerdict(UNAVAILABLE, ("freshness contract reports ok=false",), contract)

    failure_status = contract.get("failure_status")
    written_at = _as_int(contract.get("last_successful_index_write_at"))
    point_count = _as_int(contract.get("point_count"))
    index_version = contract.get("index_version")

    if written_at is None:
        return FreshnessVerdict(UNAVAILABLE, ("last_successful_index_write_at is not an integer",), contract)
    if point_count is None:
        return FreshnessVerdict(UNAVAILABLE, ("point_count is not an integer",), contract)

    age = float(now) - float(written_at)
    if failure_status != "ok":
        return FreshnessVerdict(STALE, (f"projection failure_status={failure_status!r}",), contract, age)
    if index_version != expected_index_version:
        return FreshnessVerdict(
            STALE, (f"index_version {index_version!r} != expected {expected_index_version!r}",), contract, age
        )
    if point_count <= 0:
        return FreshnessVerdict(STALE, ("index is empty (point_count<=0)",), contract, age)
    if age > sla_seconds:
        return FreshnessVerdict(
            STALE, (f"last successful index write is {age:.0f}s old (> SLA {sla_seconds}s)",), contract, age
        )
    return FreshnessVerdict(FRESH, (), contract, age)


def outcome_for(verdict: FreshnessVerdict) -> Outcome:
    return Outcome.STALE_INDEX if verdict.status == STALE else Outcome.FRESHNESS_UNAVAILABLE


class HttpFreshnessClient:
    """Read-only client for the published freshness contract endpoint."""

    def __init__(self, url: str = "http://127.0.0.1:8790/freshness", *, timeout: float = 3.0):
        self.url = url
        self.timeout = timeout

    def read(self, *, timeout: float | None = None) -> dict:
        effective = self.timeout if timeout is None else max(0.05, float(timeout))
        req = urllib.request.Request(self.url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=effective) as resp:
                return json.loads(resp.read().decode("utf-8") or "{}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            raise FreshnessError(Outcome.FRESHNESS_UNAVAILABLE,
                                 f"freshness contract unreadable at {self.url}: {exc}") from exc


class StaticFreshnessSource:
    """Deterministic source used by tests and offline smoke runs."""

    def __init__(self, contract: dict | Exception):
        self._contract = contract

    def read(self, *, timeout: float | None = None) -> dict:
        if isinstance(self._contract, Exception):
            raise self._contract
        return self._contract

