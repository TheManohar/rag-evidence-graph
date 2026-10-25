"""Loopback-only transport for the rag-auto plugin.

The plugin talks to the graph service directly (``127.0.0.1:8791``). No MCP
entry, no provider credential, no generic dispatch: exactly one POST
(``/rag_answer``) and one GET (``/rag_health``).
"""
from __future__ import annotations

import ipaddress
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol

DEFAULT_BASE_URL = "http://127.0.0.1:8791"
MAX_QUERY_CHARS = 2000


class GraphTransportError(Exception):
    """Classified transport failure (never carries the payload)."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind  # "timeout" | "unavailable" | "contract"
        self.message = message


class GraphTransport(Protocol):
    def answer(self, *, query: str, top_k: int, mode: str, run_id: str,
               timeout: float) -> dict: ...

    def health(self, *, timeout: float) -> dict: ...


def assert_loopback(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "http":
        raise ValueError("graph URL must be plain http on loopback")
    host = parsed.hostname or ""
    try:
        if not ipaddress.ip_address(host).is_loopback and host != "localhost":
            raise ValueError(f"graph host {host!r} is not loopback")
    except ValueError:
        if host != "localhost":
            raise ValueError(f"graph host {host!r} is not loopback") from None


class HttpGraphTransport:
    """Stdlib transport. Credentials are never read, stored or sent."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, *, timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        assert_loopback(self.base_url)
        self.timeout = max(0.05, float(timeout))

    def _open(self, request: urllib.request.Request, timeout: float) -> dict:
        try:
            with urllib.request.urlopen(request, timeout=max(0.05, float(timeout))) as response:
                return json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            # The graph classifies terminal outcomes in the body (503/504/200);
            # a 4xx is a contract bug in this client.
            try:
                payload = json.loads(exc.read().decode("utf-8") or "{}")
            except Exception:  # noqa: BLE001
                payload = {}
            if exc.code >= 500 and isinstance(payload, dict):
                return payload
            raise GraphTransportError("contract", f"graph rejected the request (HTTP {exc.code})") from exc
        except (socket.timeout, TimeoutError) as exc:
            raise GraphTransportError("timeout", "graph request timed out") from exc
        except (urllib.error.URLError, ConnectionError, OSError, ValueError) as exc:
            kind = "timeout" if "timed out" in str(exc).lower() else "unavailable"
            raise GraphTransportError(kind, f"graph unreachable: {type(exc).__name__}") from exc

    def answer(self, *, query: str, top_k: int, mode: str, run_id: str,
               timeout: float) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise GraphTransportError("contract", "query must be non-empty")
        if len(query) > MAX_QUERY_CHARS:
            raise GraphTransportError("contract", "query too long")
        payload = {"rag_answer": {
            "query": query.strip(), "top_k": int(top_k), "mode": mode, "run_id": run_id,
        }}
        request = urllib.request.Request(
            self.base_url + "/rag_answer",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        return self._open(request, timeout)

    def health(self, *, timeout: float) -> dict:
        request = urllib.request.Request(self.base_url + "/rag_health", method="GET")
        try:
            payload = self._open(request, timeout)
        except GraphTransportError:
            return {"ok": False}
        return payload if isinstance(payload, dict) else {"ok": False}
