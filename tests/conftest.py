"""Shared fakes and fixtures for the offline test suite (no network)."""
from __future__ import annotations

import time

import pytest

from langgraph_rag.checkpoint import InMemoryCheckpointer
from langgraph_rag.freshness import StaticFreshnessSource
from langgraph_rag.graph import RagEngine

INDEX_VERSION = "rag-v1"
SLA_SECONDS = 900


def fresh_contract(now: float | None = None, **over) -> dict:
    now = time.time() if now is None else now
    contract = {
        "ok": True,
        "last_successful_source_read_at": int(now) - 60,
        "last_successful_index_write_at": int(now) - 60,
        "index_version": INDEX_VERSION,
        "point_count": 42,
        "failure_status": "ok",
    }
    contract.update(over)
    return contract


class FakeRetrieval:
    """Offline RetrievalClient stand-in; records calls, raises on demand."""

    def __init__(self, response: dict | None = None, error: Exception | None = None):
        self.response = response if response is not None else {
            "ok": True, "results": [], "count": 0, "mode": "hybrid",
            "index_version": INDEX_VERSION, "quarantined": 0,
            "latency_ms": 1.0, "retries": {},
        }
        self.error = error
        self.calls = 0
        self.last: dict | None = None

    def search(self, *, query, filters, top_k, mode) -> dict:
        self.calls += 1
        self.last = {"query": query, "filters": filters, "top_k": top_k, "mode": mode}
        if self.error is not None:
            raise self.error
        return self.response


class QueryAwareRetrieval:
    """Returns query-dependent evidence so cross-request inheritance is visible."""

    def __init__(self):
        self.calls = 0
        self.last: dict | None = None

    def search(self, *, query, filters, top_k, mode) -> dict:
        self.calls += 1
        self.last = {"query": query, "filters": filters, "top_k": top_k, "mode": mode}
        return retrieval_response([
            citation(source_id=f"src-{query}", snippet=f"text for {query}"),
        ])


def signals(keyword_score: float = 1.0, vector_similarity: float = 0.5,
            matched_terms: list[str] | tuple[str, ...] = ("hello",),
            term_coverage: float = 1.0, idf_mass: float | None = None) -> dict:
    """Raw per-candidate relevance signal, as exposed by the retrieval layer."""
    out = {
        "keyword_score": keyword_score,
        "vector_similarity": vector_similarity,
        "matched_terms": list(matched_terms),
        "term_coverage": term_coverage,
    }
    if idf_mass is not None:
        out["idf_mass"] = idf_mass
    return out


def relevance_diagnostics(candidates: list[dict] | None = None, **over) -> dict:
    """Response-level diagnostics block (additive retrieval metadata)."""
    cands = candidates or [{"id": "p1", "keyword_score": 1.0, "vector_similarity": 0.5,
                            "matched_terms": ["hello"], "term_coverage": 1.0}]
    if not cands:
        cands = [{"id": "p0", "keyword_score": 0.0, "vector_similarity": 0.0,
                  "matched_terms": [], "term_coverage": 0.0}]
        empty = True
    else:
        empty = False
    diag = {
        "query_terms": ["hello"],
        "candidate_count": 0 if empty else len(cands),
        "max_keyword_score": max(c["keyword_score"] for c in cands),
        "max_vector_similarity": max(c["vector_similarity"] for c in cands),
        "max_term_coverage": max(c["term_coverage"] for c in cands),
        "candidates": [] if empty else cands,
    }
    diag.update(over)
    return diag


def citation(source_id: str = "s1", chunk_index: int = 0, snippet: str = "hello world",
             index_version: str = INDEX_VERSION, signals: dict | None = None, **over) -> dict:
    item = {
        "source_id": source_id, "object_id": "o1", "space_id": "sp1", "type_key": "note",
        "chunk_index": chunk_index, "chunk_count": 1, "updated_at": 111,
        "index_version": index_version, "source_url": "anytype://space/sp1/object/o1",
        "score": 0.5, "snippet": snippet,
        "signals": {"keyword_score": 1.0, "vector_similarity": 0.5,
                    "matched_terms": ["hello"], "term_coverage": 1.0}
        if signals is None else signals,
    }
    item.update(over)
    return item


def retrieval_response(results: list[dict], **over) -> dict:
    cands = []
    for i, r in enumerate(results):
        sig = r.get("signals") if isinstance(r, dict) else None
        if not isinstance(sig, dict):
            continue
        cands.append({"id": f"p{i}", "keyword_score": sig.get("keyword_score", 0.0),
                      "vector_similarity": sig.get("vector_similarity", 0.0),
                      "matched_terms": sig.get("matched_terms", []),
                      "term_coverage": sig.get("term_coverage", 0.0)})
    resp = {
        "ok": True, "results": results, "count": len(results), "mode": "hybrid",
        "index_version": INDEX_VERSION, "quarantined": 0, "latency_ms": 1.0, "retries": {},
        "diagnostics": relevance_diagnostics(cands),
    }
    resp.update(over)
    return resp


@pytest.fixture
def engine_factory():
    def _make(retrieval=None, freshness=None, **kwargs) -> RagEngine:
        return RagEngine(
            retrieval=retrieval or FakeRetrieval(),
            freshness=freshness or StaticFreshnessSource(fresh_contract()),
            checkpointer=InMemoryCheckpointer(),
            expected_index_version=INDEX_VERSION,
            sla_seconds=SLA_SECONDS,
            **kwargs,
        )

    return _make
