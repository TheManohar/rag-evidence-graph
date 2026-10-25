"""Deterministic outcome taxonomy for the bounded RAG graph.

Every failed run terminates in exactly one of these outcomes. The taxonomy is
closed: there is no free-form error string acting as a status. This is what
makes the graph's behaviour reproducible and testable offline.
"""
from __future__ import annotations

from enum import Enum


class Outcome(str, Enum):
    """Closed set of terminal run outcomes."""

    OK = "ok"
    INVALID_REQUEST = "invalid_request"
    FRESHNESS_UNAVAILABLE = "freshness_unavailable"
    STALE_INDEX = "stale_index"
    RETRIEVAL_UNAVAILABLE = "retrieval_unavailable"
    RETRIEVAL_TIMEOUT = "retrieval_timeout"
    RETRIEVAL_CONTRACT = "retrieval_contract"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    NO_EVIDENCE = "no_evidence"
    LOW_RELEVANCE = "low_relevance"
    INVALID_CITATIONS = "invalid_citations"
    INTERNAL_ERROR = "internal_error"


#: Every non-success outcome. Success is exactly ``Outcome.OK``.
ERROR_OUTCOMES = frozenset(o for o in Outcome if o is not Outcome.OK)


class RagError(Exception):
    """Base class carrying a deterministic outcome."""

    def __init__(self, outcome: Outcome, message: str):
        super().__init__(message)
        self.outcome = outcome
        self.message = message


class ContractError(RagError):
    """A request violated the typed `rag_answer` contract."""

    def __init__(self, message: str):
        super().__init__(Outcome.INVALID_REQUEST, message)


class FreshnessError(RagError):
    """The freshness contract could not be read or evaluated."""

    def __init__(self, outcome: Outcome, message: str):
        super().__init__(outcome, message)


class RetrievalError(RagError):
    """The read-only retrieval call failed in a classified way."""

    def __init__(self, outcome: Outcome, message: str):
        super().__init__(outcome, message)


class CheckpointerUnavailable(RuntimeError):
    """Requested durable checkpointer is not importable/usable.

    Raised instead of silently degrading to in-memory persistence.
    """
