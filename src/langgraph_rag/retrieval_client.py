"""Read-only retrieval client for the existing knowledge_search contract.

Calls exactly one route -- ``POST /knowledge_search`` -- on the loopback
adapter documented in ``hermes-rag/rag/adapter.py``. The client:

* refuses a non-loopback URL (the boundary is loopback-only);
* bounds attempts and wall-clock timeout with exponential backoff;
* classifies every failure into a deterministic outcome
  (``retrieval_timeout`` / ``retrieval_unavailable`` / ``retrieval_contract``);
* never performs a write and never carries credentials.

The graph depends only on the :class:`RetrievalClient` protocol, so tests
inject a fake client without any network.
"""
from __future__ import annotations

import ipaddress
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol

from .errors import Outcome, RetrievalError

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class RetrievalClient(Protocol):
    """The single read-only retrieval operation the graph needs.

    ``timeout`` (optional) is the caller's remaining wall-clock budget for the
    *whole* call (all attempts and backoff). A client that cannot honour it may
    ignore it; the HTTP client caps both the per-attempt timeout and the number
    of attempts to fit.
    """

    def search(self, *, query: str, filters: dict, top_k: int, mode: str,
               timeout: float | None = None) -> dict: ...


def _assert_loopback(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "http":
        raise ValueError("knowledge_search URL must use http on loopback")
    host = parsed.hostname or ""
    try:
        if not ipaddress.ip_address(host).is_loopback and host not in ("localhost",):
            raise ValueError(f"knowledge_search host {host!r} is not loopback")
    except ValueError:
        if host != "localhost":
            raise ValueError(f"knowledge_search host {host!r} is not loopback") from None


class HttpKnowledgeSearchClient:
    """Stdlib HTTP client for the loopback read-only knowledge_search route."""

    def __init__(
        self,
        url: str = "http://127.0.0.1:8788/knowledge_search",
        *,
        timeout: float = 5.0,
        max_retries: int = 2,
        backoff: float = 0.2,
        sleeper=time.sleep,
    ):
        _assert_loopback(url)
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self._sleep = sleeper
        #: non-sensitive per-request attempt log (no payloads)
        self.attempts: list[dict[str, Any]] = []

    def search(self, *, query: str, filters: dict, top_k: int, mode: str,
               timeout: float | None = None) -> dict:
        envelope: dict[str, Any] = {"knowledge_search": {"query": query, "top_k": top_k, "mode": mode}}
        if filters:
            envelope["knowledge_search"]["filters"] = dict(filters)
        data = json.dumps(envelope).encode("utf-8")

        #: Total budget for this call (all attempts + backoff), or None for the
        #: historical per-attempt-only behaviour.
        budget = None if timeout is None else max(0.05, float(timeout))
        started_all = time.monotonic()

        def _remaining_budget() -> float | None:
            if budget is None:
                return None
            return budget - (time.monotonic() - started_all)

        def _out_of_budget() -> bool:
            left = _remaining_budget()
            return left is not None and left <= 0.01

        last_error: RetrievalError | None = None
        for attempt in range(self.max_retries + 1):
            if _out_of_budget():
                break
            left = _remaining_budget()
            call_timeout = self.timeout if left is None else min(self.timeout, max(0.05, left))
            started = time.monotonic()
            req = urllib.request.Request(self.url, data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            try:
                with urllib.request.urlopen(req, timeout=call_timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8") or "{}")
                self.attempts.append({"attempt": attempt, "status": 200, "ms": round((time.monotonic() - started) * 1000, 2)})
                return self._normalise(payload)
            except urllib.error.HTTPError as exc:
                text = exc.read().decode("utf-8", "replace")
                self.attempts.append({"attempt": attempt, "status": exc.code,
                                      "ms": round((time.monotonic() - started) * 1000, 2)})
                if exc.code in RETRYABLE_STATUS and attempt < self.max_retries:
                    last_error = RetrievalError(Outcome.RETRIEVAL_UNAVAILABLE, f"knowledge_search HTTP {exc.code}")
                    if self._sleep_within_budget(self.backoff * (2 ** attempt), _remaining_budget()):
                        continue
                    break
                # 4xx (and exhausted 5xx) are terminal and classified as contract errors
                if exc.code in RETRYABLE_STATUS:
                    raise RetrievalError(Outcome.RETRIEVAL_UNAVAILABLE,
                                         f"knowledge_search unavailable after {attempt + 1} attempts (HTTP {exc.code})")
                raise RetrievalError(Outcome.RETRIEVAL_CONTRACT,
                                     f"knowledge_search rejected the request (HTTP {exc.code}): {text[:200]}")
            except (socket.timeout, TimeoutError) as exc:
                self.attempts.append({"attempt": attempt, "status": 0, "timeout": True,
                                      "ms": round((time.monotonic() - started) * 1000, 2)})
                last_error = RetrievalError(Outcome.RETRIEVAL_TIMEOUT, f"knowledge_search timed out: {exc}")
                if attempt < self.max_retries and self._sleep_within_budget(self.backoff * (2 ** attempt), _remaining_budget()):
                    continue
                raise last_error
            except (urllib.error.URLError, ConnectionError, OSError) as exc:
                self.attempts.append({"attempt": attempt, "status": 0,
                                      "ms": round((time.monotonic() - started) * 1000, 2)})
                # A bare timeout inside URLError is a timeout, not an outage.
                outcome = Outcome.RETRIEVAL_TIMEOUT if "timed out" in str(exc).lower() else Outcome.RETRIEVAL_UNAVAILABLE
                last_error = RetrievalError(outcome, f"knowledge_search unreachable: {exc}")
                if attempt < self.max_retries and self._sleep_within_budget(self.backoff * (2 ** attempt), _remaining_budget()):
                    continue
                raise last_error
        if last_error is not None:
            raise last_error
        raise RetrievalError(Outcome.RETRIEVAL_TIMEOUT,
                             "knowledge_search budget exhausted before an attempt completed")

    def _sleep_within_budget(self, delay: float, remaining: float | None) -> bool:
        """Sleep *delay* only if it fits the remaining budget; False means stop."""
        if remaining is not None and delay >= remaining:
            return False
        self._sleep(delay)
        return True

    @staticmethod
    def _normalise(payload: Any) -> dict:
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise RetrievalError(Outcome.RETRIEVAL_CONTRACT, "knowledge_search returned a non-ok envelope")
        results = payload.get("results")
        if not isinstance(results, list):
            raise RetrievalError(Outcome.RETRIEVAL_CONTRACT, "knowledge_search returned no results array")
        return payload

    def retry_summary(self) -> dict:
        return {
            "attempts": len(self.attempts),
            "retried": sum(1 for a in self.attempts if a.get("attempt", 0) > 0),
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout,
        }
