"""Typed `rag_answer` request validation.

Exactly one operation is accepted: ``rag_answer``. Unknown fields, identity
overrides, and any workflow/execution/write token are rejected, not ignored.
The request can never select an access identity: the retrieval principal is a
server-side property of the downstream ``knowledge_search`` adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import ContractError
from .ids import is_valid_run_id, request_fingerprint

ALLOWED_TOP_KEYS = {"rag_answer"}
ALLOWED_PARAMS = {"query", "top_k", "mode", "filters", "run_id"}
ALLOWED_FILTERS = {"space_ids", "types", "visibility", "updated_after", "source_ids"}

# Keys that would open a generic-execution / mutation / credential surface.
FORBIDDEN_TOKENS = (
    "operation", "workflow", "workflowid", "execute", "exec", "run", "shell",
    "command", "mutation", "write", "delete", "archive", "create", "update",
    "anytype", "credential", "api_key", "apikey", "token", "secret", "password",
    "url", "http", "eval", "import", "subprocess", "provider", "model", "llm",
)
# Any caller-supplied access/identity override is refused outright.
IDENTITY_OVERRIDE_TOKENS = (
    "principal", "owner", "access", "access_override", "identity", "acl",
    "auth", "authorization", "user", "role", "sudo", "as", "impersonate",
    "scope", "visibility_override", "collection", "namespace",
)

MAX_QUERY_CHARS = 2000
MAX_TOP_K = 20
MODES = ("hybrid", "vector", "keyword")


@dataclass(frozen=True)
class RagRequest:
    query: str
    filters: dict
    top_k: int
    mode: str
    run_id: str | None
    fingerprint: str = field(default="")

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "filters": self.filters,
            "top_k": self.top_k,
            "mode": self.mode,
            "run_id": self.run_id,
        }


def _identity_override(body: Any) -> str | None:
    """Return the first caller-supplied identity-override field, if any.

    Scans every nested object so an override cannot hide inside `filters`.
    """
    if not isinstance(body, dict):
        return None
    queue: list[Any] = [body]
    while queue:
        node = queue.pop()
        if not isinstance(node, dict):
            continue
        for key, value in node.items():
            if isinstance(key, str) and key.lower() in IDENTITY_OVERRIDE_TOKENS:
                return key
            if isinstance(value, dict):
                queue.append(value)
    return None


def validate_request(body: Any) -> RagRequest:
    """Strictly validate a `rag_answer` request body.

    Returns a frozen :class:`RagRequest`. Raises :class:`ContractError`.
    """
    if not isinstance(body, dict):
        raise ContractError("body must be a JSON object")
    override = _identity_override(body)
    if override is not None:
        raise ContractError(
            f"field {override!r} is not permitted: access identity is resolved "
            "server-side and cannot be supplied by the caller"
        )
    keys = {k.lower() for k in body if isinstance(k, str)}
    if "rag_answer" not in keys:
        raise ContractError("request must be exactly {'rag_answer': {...}}")
    if len(body) != 1:
        raise ContractError(f"only the 'rag_answer' key is allowed, got {sorted(body)}")
    params = next(v for k, v in body.items() if isinstance(k, str) and k.lower() == "rag_answer")
    if not isinstance(params, dict):
        raise ContractError("rag_answer payload must be an object")
    for k in params:
        if not isinstance(k, str):
            raise ContractError("rag_answer keys must be strings")
        if k in ALLOWED_PARAMS:
            continue
        if k.lower() in FORBIDDEN_TOKENS:
            raise ContractError(f"field {k!r} is not permitted on the read-only boundary")
        if k.lower() in IDENTITY_OVERRIDE_TOKENS:
            raise ContractError(f"field {k!r} is not permitted: identity is server-side")
        raise ContractError(f"unknown field {k!r}")

    query = params.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ContractError("query must be a non-empty string")
    query = query.strip()
    if len(query) > MAX_QUERY_CHARS:
        raise ContractError(f"query too long (max {MAX_QUERY_CHARS} chars)")

    filters = params.get("filters") or {}
    if not isinstance(filters, dict):
        raise ContractError("filters must be an object")
    for k in filters:
        if not isinstance(k, str) or k not in ALLOWED_FILTERS:
            raise ContractError(f"unknown filter {k!r}")

    top_k = params.get("top_k", 5)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= MAX_TOP_K:
        raise ContractError(f"top_k must be an integer in 1..{MAX_TOP_K}")

    mode = params.get("mode", "hybrid")
    if mode not in MODES:
        raise ContractError("mode must be one of hybrid|vector|keyword")

    run_id = params.get("run_id")
    if run_id is not None and not is_valid_run_id(run_id):
        raise ContractError("run_id must match [A-Za-z0-9._:-]{1,128}")

    return RagRequest(
        query=query,
        filters=dict(filters),
        top_k=top_k,
        mode=mode,
        run_id=run_id,
        fingerprint=request_fingerprint(query, filters, top_k, mode),
    )
