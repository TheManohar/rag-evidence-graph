"""Acceptance: real file-backed SQLite persistence (write/readback)."""
from __future__ import annotations

import sqlite3

from conftest import (FakeRetrieval, QueryAwareRetrieval, citation,
                      fresh_contract, retrieval_response)
from langgraph_rag.checkpoint import SqliteCheckpointer
from langgraph_rag.freshness import StaticFreshnessSource
from langgraph_rag.graph import RagEngine

BODY = {"rag_answer": {"query": "persist me", "top_k": 2}}


def _engine(path: str, retrieval=None) -> RagEngine:
    return RagEngine(
        retrieval=retrieval or FakeRetrieval(retrieval_response([citation()])),
        freshness=StaticFreshnessSource(fresh_contract()),
        checkpointer=SqliteCheckpointer(path),
        expected_index_version="rag-v1",
        sla_seconds=900,
    )


def test_sqlite_checkpoint_survives_a_second_engine_over_the_same_file(tmp_path):
    db = str(tmp_path / "cp.sqlite3")
    first = _engine(db)
    resp = first.run(BODY, run_id="persist-1")
    assert resp["status"] == "ok" and resp["replayed"] is False

    # A brand-new engine + saver reading the same file must see the stored run.
    second = _engine(db)
    stored = second.status("persist-1")
    assert stored is not None
    assert stored["status"] == "ok"
    assert stored["evidence"] == resp["evidence"]

    # And a replay over the file-backed saver does not re-run retrieval.
    retrieval = FakeRetrieval(retrieval_response([citation()]))
    replay = _engine(db, retrieval=retrieval).run(BODY, run_id="persist-1")
    assert replay["replayed"] is True
    assert retrieval.calls == 0


def test_sqlite_file_is_written_outside_the_store_with_table_content(tmp_path):
    db = str(tmp_path / "cp.sqlite3")
    _engine(db).run(BODY, run_id="persist-2")
    assert (tmp_path / "cp.sqlite3").exists()
    assert (tmp_path / "cp.sqlite3").stat().st_size > 0
    conn = sqlite3.connect(db)
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert "checkpoints" in tables


def test_sqlite_reused_run_id_different_body_is_isolated(tmp_path):
    """F2 on the production saver: a reused id never leaks prior evidence."""
    db = str(tmp_path / "cp.sqlite3")
    retrieval = QueryAwareRetrieval()
    engine = _engine(db, retrieval=retrieval)
    a = engine.run({"rag_answer": {"query": "alpha"}}, run_id="iso-1")
    b = engine.run({"rag_answer": {"query": "beta"}}, run_id="iso-1")
    assert a["evidence"][0]["source_id"] == "src-alpha"
    assert b["query"] == "beta" and b["evidence"][0]["source_id"] == "src-beta"
    assert retrieval.calls == 2

    # A brand-new engine over the same file sees only the latest run.
    stored = _engine(db).status("iso-1")
    assert stored["query"] == "beta"
    assert stored["evidence"][0]["source_id"] == "src-beta"
