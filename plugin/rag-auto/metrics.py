"""JSONL metrics for the rag-auto gate.

One line per evaluated turn, whichever way the decision goes. Payload-free by
construction: the fixed field set below is the only thing that is written, and
it never includes the query text, snippets, raw session ids or credentials.

Host identity fields may embed a raw session id (Hermes builds ``turn_id`` as
``"<session_id>:<task_id>:<random>"`` and CLI compression passes the session id
as ``task_id``), so ``turn_id`` and ``task_id`` are persisted only as
``sha256:<16 hex>`` one-way digests.  The digest is deterministic, which keeps
per-turn correlation stable without disclosing the composite identifier.  The
sanitation runs both in :func:`build_record` and in :meth:`MetricSink.emit`, so
a caller that hand-builds or mutates a record cannot bypass it.

Fail-open is visible here, not silent: a timeout/exception reports
``status`` in {``timeout``, ``error``}, ``fail_open: true`` and the
``error_class``.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

METRIC_FIELDS = (
    "ts", "profile", "platform", "decision", "reason_code", "session_hash",
    "task_id", "turn_id", "run_id", "status", "wall_ms", "evidence_count",
    "freshness_age", "index_version", "replayed", "injected_chars",
    "shadow", "fail_open", "error_class", "fingerprint_prefix",
)

#: Fields whose host value can embed a raw session or caller-supplied composite
#: identifier.  They are projected to opaque digests before they are persisted.
OPAQUE_FIELDS = ("turn_id", "task_id")

OPAQUE_PREFIX = "sha256:"
_OPAQUE_HEX = frozenset("0123456789abcdef")
_MAX_BYTES = 5_000_000


def opaque_identifier(value: Any) -> str:
    """One-way, deterministic projection for a session-bearing identifier.

    Idempotent: a value already in the published ``sha256:<16 hex>`` form is
    returned unchanged, so record builders and the sink can both sanitize.
    """
    text = str(value or "")
    if not text:
        return ""
    if text.startswith(OPAQUE_PREFIX):
        digest = text[len(OPAQUE_PREFIX):]
        if len(digest) == 16 and set(digest) <= _OPAQUE_HEX:
            return text
    return OPAQUE_PREFIX + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def sanitize_record(record: dict) -> dict:
    """Return a copy with session-bearing identifiers projected to digests."""
    clean = dict(record)
    for field in OPAQUE_FIELDS:
        if field in clean:
            clean[field] = opaque_identifier(clean[field])
    return clean


def _iso(now: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return round(value, 2)
    return None


def build_record(*, profile: str, platform: str, decision: str, reason_code: str,
                 session_hash: str = "", turn_id: str = "", run_id: str = "",
                 status: str = "", wall_ms: Any = None, evidence_count: int = 0,
                 freshness_age: Any = None, index_version: str = "",
                 replayed: bool = False, injected_chars: int = 0,
                 shadow: bool = False, fail_open: bool = False,
                 error_class: str = "", fingerprint_prefix: str = "",
                 task_id: str = "", now: float | None = None) -> dict:
    return {
        "ts": _iso(time.time() if now is None else now),
        "profile": str(profile or ""),
        "platform": str(platform or ""),
        "decision": str(decision or ""),
        "reason_code": str(reason_code or ""),
        "session_hash": str(session_hash or ""),
        "task_id": opaque_identifier(task_id),
        "turn_id": opaque_identifier(turn_id),
        "run_id": str(run_id or "")[:128],
        "status": str(status or ""),
        "wall_ms": _number(wall_ms),
        "evidence_count": int(evidence_count or 0),
        "freshness_age": _number(freshness_age),
        "index_version": str(index_version or ""),
        "replayed": bool(replayed),
        "injected_chars": int(injected_chars or 0),
        "shadow": bool(shadow),
        "fail_open": bool(fail_open),
        "error_class": str(error_class or "")[:64],
        "fingerprint_prefix": str(fingerprint_prefix or "")[:24],
    }


class MetricSink:
    """Append-only JSONL sink; a metrics failure never breaks a turn."""

    def __init__(self, path: str | os.PathLike[str] | None, *, max_bytes: int = _MAX_BYTES) -> None:
        self.path = Path(path) if path else None
        self.max_bytes = max(0, int(max_bytes))
        self.errors = 0
        self._lock = threading.Lock()
        self._warned = False

    def emit(self, record: dict) -> bool:
        if self.path is None:
            return False
        try:
            line = json.dumps(sanitize_record(record), separators=(",", ":"),
                              ensure_ascii=False) + "\n"
        except (TypeError, ValueError):
            self.errors += 1
            return False
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                self._rotate_if_needed()
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                try:
                    os.write(fd, line.encode("utf-8"))
                finally:
                    os.close(fd)
            return True
        except OSError:
            self.errors += 1
            return False

    def _rotate_if_needed(self) -> None:
        path = self.path
        if self.max_bytes <= 0 or path is None or not path.exists():
            return
        try:
            if path.stat().st_size < self.max_bytes:
                return
        except OSError:
            return
        try:
            os.replace(path, path.with_name(path.name + ".1"))
        except OSError:
            pass


class NullSink:
    """No-op sink for tests and for a profile with metrics disabled."""

    errors = 0

    def emit(self, record: dict) -> bool:
        return False
