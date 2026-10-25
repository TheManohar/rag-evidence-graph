"""Idempotent run identifiers.

A run is made idempotent by deriving its checkpoint thread id deterministically
from the request. Replaying the identical request reuses the same thread id, so
LangGraph's checkpointer returns the stored terminal state instead of performing
a second retrieval.
"""
from __future__ import annotations

import hashlib
import json
import re

RUN_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def request_fingerprint(query: str, filters: dict, top_k: int, mode: str) -> str:
    """Stable fingerprint of the *retrieval-relevant* request fields only.

    ``run_id`` is deliberately excluded: supplying an id is what lets a caller
    force two otherwise-identical requests to share a thread.
    """
    canonical = json.dumps(
        {"query": query, "filters": filters or {}, "top_k": top_k, "mode": mode},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_valid_run_id(value: object) -> bool:
    return isinstance(value, str) and bool(RUN_ID_RE.match(value))


def derive_run_id(fingerprint: str) -> str:
    """Deterministic thread id for a fingerprinted request."""
    return "rag-" + fingerprint.split(":", 1)[1][:32]


def assert_run_id(value: str) -> str:
    if not is_valid_run_id(value):
        raise ValueError(f"invalid run_id: {value!r}")
    return value
