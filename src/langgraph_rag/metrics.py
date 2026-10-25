"""Redacted per-request provenance metrics for the adapter.

One JSON object per adapter request, appended to a JSONL file. The record is
deliberately payload-free: it never contains the query text, retrieved
snippets, filter values, headers, or any credential. It exists so that
"was the graph invoked, for which run, and how did it end" is answerable
without logging content.

``caller`` is a placeholder today. The plugin that will call this service is
not installed in any profile yet, so the adapter cannot know its caller; the
field is part of the schema now so phase 2 can populate it without a schema
change. It is never derived from caller-supplied text.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

#: Placeholder caller identity until the rag-auto plugin supplies provenance.
CALLER_UNKNOWN = "unknown-until-plugin"

#: The complete, fixed field set of a provenance line. Adding a field here is
#: the only way a line grows; nothing from the request payload is copied in.
PROVENANCE_FIELDS = (
    "ts",
    "run_id",
    "status",
    "wall_ms",
    "evidence_count",
    "freshness_age",
    "index_version",
    "replayed",
    "caller",
)

_MAX_RUN_ID_CHARS = 128


def _now_iso(now: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if now is None else now))


def _safe_run_id(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()[:_MAX_RUN_ID_CHARS]
    return "unknown"


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return round(number, 2)


def provenance_record(
    *,
    run_id: Any,
    status: Any,
    wall_ms: Any,
    evidence_count: Any = 0,
    freshness_age: Any = None,
    index_version: Any = None,
    replayed: Any = False,
    caller: str = CALLER_UNKNOWN,
    now: float | None = None,
) -> dict:
    """Build one fixed-shape, payload-free provenance record."""
    count = evidence_count if isinstance(evidence_count, int) and not isinstance(evidence_count, bool) else 0
    return {
        "ts": _now_iso(now),
        "run_id": _safe_run_id(run_id),
        "status": status if isinstance(status, str) and status else "unknown",
        "wall_ms": _safe_float(wall_ms) if wall_ms is not None else None,
        "evidence_count": max(0, count),
        "freshness_age": _safe_float(freshness_age),
        "index_version": index_version if isinstance(index_version, str) and index_version else None,
        "replayed": bool(replayed),
        "caller": caller if isinstance(caller, str) and caller else CALLER_UNKNOWN,
    }


def record_for_response(
    response: dict | None,
    *,
    wall_ms: float,
    fallback_run_id: Any = None,
    status: str | None = None,
    index_version: str | None = None,
    caller: str = CALLER_UNKNOWN,
) -> dict:
    """Project a graph response onto the provenance schema.

    Only aggregate fields are read. ``response["query"]`` and the evidence
    bodies are intentionally never touched here.
    """
    response = response if isinstance(response, dict) else {}
    freshness = response.get("freshness")
    age = freshness.get("age_seconds") if isinstance(freshness, dict) else None
    return provenance_record(
        run_id=response.get("run_id") or fallback_run_id,
        status=status or response.get("status"),
        wall_ms=wall_ms,
        evidence_count=len(response.get("evidence") or []),
        freshness_age=age,
        index_version=response.get("index_version") or index_version,
        replayed=response.get("replayed") is True,
        caller=caller,
    )


class MetricsWriter:
    """Append-only JSONL sink with size-based rotation.

    Every failure is swallowed: a metrics outage must never fail a graph
    request. ``errors`` records the count for the caller/tests to inspect.
    """

    def __init__(self, path: str | os.PathLike[str] | None, *, max_bytes: int = 5_000_000,
                 mode: int = 0o600) -> None:
        self.path = Path(path) if path else None
        self.max_bytes = max(0, int(max_bytes))
        self.mode = mode
        self.errors = 0
        self._lock = threading.Lock()

    def emit(self, record: dict) -> bool:
        if self.path is None:
            return False
        try:
            line = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
        except (TypeError, ValueError):
            self.errors += 1
            return False
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                self._rotate_if_needed()
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, self.mode)
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
        rotated = path.with_name(path.name + ".1")
        try:
            os.replace(path, rotated)
        except OSError:
            pass
