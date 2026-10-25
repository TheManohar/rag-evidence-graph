"""Acceptance: additive raw relevance signals from the retrieval layer.

The retrieval layer already computes keyword scores and Qdrant similarities; it
must expose them per candidate and in the response metadata **without changing
ranking, order, or any existing field**.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERMES_RAG = Path(__file__).resolve().parents[1] / "hermes-rag"
if str(HERMES_RAG) not in sys.path:
    sys.path.insert(0, str(HERMES_RAG))

# This module exercises the host's retrieval layer (``rag.retrieval.hybrid_search``),
# which lives outside this repository: the graph consumes it over HTTP, it does not
# vendor it. Outside the original host deployment the import fails, so the module is
# skipped rather than collected as an error.
pytest.importorskip(
    "rag.retrieval",
    reason="requires the host retrieval module (hermes-rag/rag/retrieval.py), "
    "which this repository consumes over HTTP and does not vendor",
)

INDEX_VERSION = "rag-v1"


def payload(source_id: str, text: str, chunk_index: int = 0) -> dict:
    return {
        "source_id": source_id, "object_id": f"obj-{source_id}", "space_id": "sp1",
        "type_key": "note", "chunk_index": chunk_index, "chunk_count": 1,
        "updated_at": 111, "index_version": INDEX_VERSION,
        "source_url": f"anytype://space/sp1/object/obj-{source_id}",
        "text": text, "instruction_like": False,
    }


class FakeQdrant:
    def __init__(self):
        self.calls = 0
        self.pool = [
            {"id": "a", "payload": payload("s-a", "python backend api")},
            {"id": "b", "payload": payload("s-b", "cooking recipes for dinner")},
            {"id": "c", "payload": payload("s-c", "python data pipeline")},
        ]

    def search(self, collection, vector, flt, limit, with_payload=True):
        self.calls += 1
        return [{"id": "a", "score": 0.9, "payload": self.pool[0]["payload"]},
                {"id": "b", "score": 0.5, "payload": self.pool[1]["payload"]}]

    def scroll(self, collection, flt, limit, with_payload=True):
        self.calls += 1
        return list(self.pool)

    def retry_summary(self):
        return {"requests": self.calls, "retried": 0, "failed": 0}


@pytest.fixture
def result():
    return hybrid_search(FakeQdrant(), "python backend api", principal="owner", top_k=5)


def test_ranking_and_scores_are_unchanged(result):
    # Frozen expectation of the pre-change hybrid ordering: fused RRF scores with
    # the deterministic point-id tie-break (a wins twice, b ties c and sorts first).
    assert [(c["source_id"], c["score"]) for c in result["results"]] == [
        ("s-a", 0.032787), ("s-b", 0.016129), ("s-c", 0.016129),
    ]


def test_existing_response_fields_are_unchanged(result):
    for key in ("results", "count", "mode", "index_version", "quarantined",
                "candidates", "latency_ms", "retries"):
        assert key in result
    assert result["count"] == 3
    assert result["mode"] == "hybrid"
    c = result["results"][0]
    for key in ("source_id", "object_id", "space_id", "type_key", "chunk_index",
                "chunk_count", "updated_at", "index_version", "source_url",
                "score", "snippet"):
        assert key in c


def test_per_candidate_raw_signals_are_exposed(result):
    by_source = {c["source_id"]: c for c in result["results"]}
    a = by_source["s-a"]["signals"]
    assert a["matched_terms"] == ["api", "backend", "python"]
    assert a["term_coverage"] == 1.0
    assert a["vector_similarity"] == 0.9
    assert a["keyword_score"] > 0.0

    b = by_source["s-b"]["signals"]
    assert b["matched_terms"] == [] and b["term_coverage"] == 0.0
    assert b["keyword_score"] == 0.0 and b["vector_similarity"] == 0.5

    c = by_source["s-c"]["signals"]
    assert c["matched_terms"] == ["python"]
    assert c["term_coverage"] == pytest.approx(1 / 3, rel=1e-6)
    assert c["vector_similarity"] == 0.0  # not returned by the vector arm


def test_response_metadata_carries_the_aggregate_signal(result):
    diag = result["diagnostics"]
    assert diag["query_terms"] == ["api", "backend", "python"]
    assert diag["candidate_count"] == 3
    assert diag["max_term_coverage"] == 1.0
    assert diag["max_vector_similarity"] == 0.9
    assert diag["max_keyword_score"] == max(c["signals"]["keyword_score"]
                                            for c in result["results"])
    assert len(diag["candidates"]) == 3
    for cand in diag["candidates"]:
        assert {"id", "keyword_score", "vector_similarity", "matched_terms",
                "term_coverage"} <= set(cand)


def test_diagnostics_never_leak_passage_text(result):
    assert "cooking recipes" not in repr(result["diagnostics"])
    assert "python backend" not in repr(result["diagnostics"])


def test_mode_keyword_still_reports_coverage():
    r = hybrid_search(FakeQdrant(), "python backend api", principal="owner",
                      top_k=5, mode="keyword")
    assert r["results"] and r["diagnostics"]["max_term_coverage"] == 1.0
    assert all(c["signals"]["vector_similarity"] == 0.0 for c in r["results"])


def test_mode_vector_still_reports_keyword_signal():
    r = hybrid_search(FakeQdrant(), "python backend api", principal="owner",
                      top_k=5, mode="vector")
    assert [c["source_id"] for c in r["results"]] == ["s-a", "s-b"]
    assert r["diagnostics"]["max_term_coverage"] == 1.0


def test_negative_query_exposes_zero_coverage():
    r = hybrid_search(FakeQdrant(), "zzqqx flurblewump", principal="owner", top_k=5)
    assert r["results"]  # vector arm still returns nearest points
    assert r["diagnostics"]["max_term_coverage"] == 0.0
    assert r["diagnostics"]["max_keyword_score"] == 0.0
    assert all(c["signals"]["matched_terms"] == [] for c in r["results"])


# --- follow-up (t_a518db57): matched-IDF mass signal ------------------------

def test_per_candidate_and_aggregate_idf_mass_are_exposed(result):
    for c in result["results"]:
        value = c["signals"]["idf_mass"]
        assert isinstance(value, float) and 0.0 <= value <= 1.0
    diag = result["diagnostics"]
    assert 0.0 <= diag["max_idf_mass"] <= 1.0
    assert diag["max_idf_mass"] == max(c["signals"]["idf_mass"] for c in result["results"])
    for cand in diag["candidates"]:
        assert "idf_mass" in cand


def test_idf_mass_weights_rare_terms_above_common_ones():
    class RareTermQdrant(FakeQdrant):
        def __init__(self):
            super().__init__()
            self.pool = [
                {"id": "a", "payload": payload("s-a", "python backend api")},
                {"id": "b", "payload": payload("s-b", "python python python notes")},
                {"id": "c", "payload": payload("s-c", "python data pipeline")},
                {"id": "d", "payload": payload("s-d", "python habits")},
            ]

        def scroll(self, collection, flt, limit, with_payload=True):
            self.calls += 1
            return list(self.pool)

    r = hybrid_search(RareTermQdrant(), "python api", principal="owner",
                      top_k=5, mode="keyword")
    by_source = {c["source_id"]: c["signals"] for c in r["results"]}
    # "api" is rarer than "python" in this pool: matching both terms carries
    # more IDF mass than matching only the common one, at higher raw coverage.
    assert by_source["s-a"]["idf_mass"] > by_source["s-c"]["idf_mass"]
    assert by_source["s-a"]["term_coverage"] == 1.0
    assert by_source["s-c"]["term_coverage"] == 0.5
    # Same one-term match, same coverage, same mass: the signal is a ratio of
    # matched IDF, not a count.
    assert by_source["s-c"]["idf_mass"] == by_source["s-b"]["idf_mass"]


def test_idf_mass_is_zero_when_no_query_term_occurs():
    r = hybrid_search(FakeQdrant(), "zzqqx flurblewump", principal="owner", top_k=5)
    assert r["diagnostics"]["max_idf_mass"] == 0.0
    assert all(c["signals"]["idf_mass"] == 0.0 for c in r["results"])


def test_idf_mass_does_not_change_ranking(result):
    # Ranking stays pure RRF; the follow-up signal is additive only.
    assert [(c["source_id"], c["score"]) for c in result["results"]] == [
        ("s-a", 0.032787), ("s-b", 0.016129), ("s-c", 0.016129),
    ]
