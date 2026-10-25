"""Citation and grounding validation.

Retrieved source text is **data, never an instruction**. This module:

* quarantines instruction-like chunks with its own detector (independent of the
  upstream flag, defence in depth);
* drops citations that do not carry a stable source id, do not match the
  expected index version (stale hits), or are otherwise ungrounded;
* deduplicates by ``(source_id, chunk_index)``;
* preserves source ids, scores, timestamps and the index version verbatim;
* bounds snippet length so a caller can never receive a full-corpus dump.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

SNIPPET_LIMIT = 400
REQUIRED_CITATION_FIELDS = ("source_id", "index_version", "chunk_index", "snippet")

_INSTRUCTION_RE = re.compile(
    r"^\s*(?:"
    r"ignore (?:all )?(?:previous|above) instructions\b"
    r"|system:(?=\s|$)"
    r"|assistant:(?=\s|$)"
    r"|you are now\b"
    r"|execute (?:the )?(?:workflow|command)\b"
    r"|run (?:the )?(?:workflow|command)\b"
    r"|call (?:the )?tool\b"
    r"|tool_call\b"
    r"|function_call\b"
    r")",
    re.I | re.M,
)
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def is_instruction_like(text: str) -> bool:
    """Does this retrieved text read like an instruction to the agent?"""
    return bool(_INSTRUCTION_RE.search(text or ""))


def sanitise_snippet(text: str, limit: int = SNIPPET_LIMIT) -> str:
    """Neutralise a snippet for return as inert, quoted data."""
    norm = unicodedata.normalize("NFKC", text or "")
    norm = _CONTROL_RE.sub(" ", norm)
    norm = "\n".join(_WS_RE.sub(" ", ln).strip() for ln in norm.splitlines())
    norm = norm.strip()
    if len(norm) <= limit:
        return norm
    return norm[: limit - 1].rstrip() + "\u2026"


def validate_evidence(
    results: Any,
    *,
    expected_index_version: str = "rag-v1",
    limit: int | None = None,
) -> tuple[list[dict], list[dict], list[str]]:
    """Validate raw retrieval results into grounded citations.

    Returns ``(evidence, quarantined, problems)``.
    """
    evidence: list[dict] = []
    quarantined: list[dict] = []
    problems: list[str] = []

    if not isinstance(results, list):
        return evidence, quarantined, ["results is not a list"]

    seen: set[tuple[str, int]] = set()
    for raw in results:
        if not isinstance(raw, dict):
            problems.append("malformed citation entry (not an object)")
            continue
        source_id = raw.get("source_id")
        snippet = raw.get("snippet")
        if snippet is None:
            snippet = raw.get("text")
        if not isinstance(snippet, str):
            snippet = ""

        # 1. prompt-injection isolation -- quarantine, never return as a command
        if raw.get("instruction_like") or is_instruction_like(snippet):
            quarantined.append({"source_id": source_id, "index_version": raw.get("index_version"),
                                "reason": "instruction_like"})
            continue

        # 2. required-field grounding
        if not isinstance(source_id, str) or not source_id:
            problems.append("citation missing source_id")
            continue
        chunk_index = raw.get("chunk_index")
        if not isinstance(chunk_index, int) or isinstance(chunk_index, bool):
            problems.append(f"citation {source_id!r} missing integer chunk_index")
            continue

        # 3. stale-index hits never reach the caller
        if str(raw.get("index_version")) != str(expected_index_version):
            problems.append(f"citation {source_id!r} index_version mismatch (stale)")
            continue

        key = (source_id, chunk_index)
        if key in seen:
            continue
        seen.add(key)
        evidence.append({
            "source_id": source_id,
            "object_id": raw.get("object_id"),
            "space_id": raw.get("space_id"),
            "type_key": raw.get("type_key"),
            "chunk_index": chunk_index,
            "chunk_count": raw.get("chunk_count"),
            "updated_at": raw.get("updated_at"),
            "index_version": raw.get("index_version"),
            "source_url": raw.get("source_url"),
            "score": raw.get("score"),
            "snippet": sanitise_snippet(snippet),
            # Additive raw relevance signal (per candidate). Preserved verbatim
            # so the graph's relevance gate sees the producer's own numbers.
            # A missing/non-object signal stays None and fails the gate closed.
            "signals": raw.get("signals") if isinstance(raw.get("signals"), dict) else None,
        })
        if limit is not None and len(evidence) >= limit:
            break

    return evidence, quarantined, problems
