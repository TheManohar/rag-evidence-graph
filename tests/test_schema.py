"""Acceptance: schema rejection (typed request validation)."""
from __future__ import annotations

import pytest

from langgraph_rag.errors import ContractError
from langgraph_rag.ids import derive_run_id
from langgraph_rag.schema import validate_request


def test_valid_minimal_request_defaults():
    req = validate_request({"rag_answer": {"query": "  what is hermes  "}})
    assert req.query == "what is hermes"
    assert req.top_k == 5
    assert req.mode == "hybrid"
    assert req.filters == {}
    assert req.run_id is None
    assert req.fingerprint.startswith("sha256:")
    # deterministic fingerprint / run id
    assert req.fingerprint == validate_request({"rag_answer": {"query": "what is hermes"}}).fingerprint
    assert derive_run_id(req.fingerprint).startswith("rag-")


@pytest.mark.parametrize("body", [
    None, [], "x", 3,
    {},  # no rag_answer
    {"knowledge_search": {"query": "x"}},  # wrong operation (the graph is not that boundary)
    {"rag_answer": {"query": "x"}, "extra": 1},
    {"rag_answer": "not-an-object"},
])
def test_rejects_malformed_envelopes(body):
    with pytest.raises(ContractError):
        validate_request(body)


@pytest.mark.parametrize("param", ["principal", "owner", "access", "identity", "acl",
                                   "auth", "role", "scope", "collection"])
def test_rejects_identity_override(param):
    with pytest.raises(ContractError):
        validate_request({"rag_answer": {"query": "x", param: "owner"}})


def test_rejects_identity_override_nested_in_filters():
    with pytest.raises(ContractError):
        validate_request({"rag_answer": {"query": "x", "filters": {"principal": "owner"}}})


@pytest.mark.parametrize("param", ["operation", "workflow", "execute", "shell", "command",
                                   "write", "anytype", "credential", "api_key", "eval"])
def test_rejects_generic_execution_tokens(param):
    with pytest.raises(ContractError):
        validate_request({"rag_answer": {"query": "x", param: "y"}})


def test_rejects_unknown_field():
    with pytest.raises(ContractError):
        validate_request({"rag_answer": {"query": "x", "unknown": 1}})


@pytest.mark.parametrize("top_k", [0, 21, -1, True, "5", 1.5, None])
def test_rejects_bad_top_k(top_k):
    with pytest.raises(ContractError):
        validate_request({"rag_answer": {"query": "x", "top_k": top_k}})


@pytest.mark.parametrize("query", ["", "   ", None, 5, "x" * 2001])
def test_rejects_bad_query(query):
    with pytest.raises(ContractError):
        validate_request({"rag_answer": {"query": query}})


def test_rejects_bad_mode_and_unknown_filter():
    with pytest.raises(ContractError):
        validate_request({"rag_answer": {"query": "x", "mode": "sql"}})
    with pytest.raises(ContractError):
        validate_request({"rag_answer": {"query": "x", "filters": {"nope": []}}})


def test_run_id_validation():
    ok = validate_request({"rag_answer": {"query": "x", "run_id": "abc-123"}})
    assert ok.run_id == "abc-123"
    with pytest.raises(ContractError):
        validate_request({"rag_answer": {"query": "x", "run_id": "bad id with spaces"}})
