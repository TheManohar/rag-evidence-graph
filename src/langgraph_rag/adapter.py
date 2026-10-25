"""Narrow loopback HTTP adapter for the bounded LangGraph RAG graph.

Typed operations (only these exist)::

    POST /rag_answer         run the graph, return validated cited evidence
    GET  /rag_run_status     read back a stored run's terminal status
    GET  /rag_health         readiness (checkpointer durability, endpoints)

Boundaries
----------
* Binds loopback only; rejects non-loopback peers.
* No generic dispatch, no arbitrary shell, no n8n execution, no Anytype write,
  no credential pass-through.
* The production path refuses to start with a non-durable (in-memory)
  checkpointer unless ``--allow-memory`` is passed explicitly for a
  non-production smoke run.
* Retrieved source text is returned as inert, quoted data; it is never
  evaluated, executed, or fed to any provider.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .checkpoint import DEFAULT_SQLITE_PATH, MEMORY, SQLITE, resolve_checkpointer
from .errors import Outcome
from .freshness import HttpFreshnessClient
from .graph import DEFAULT_REQUEST_DEADLINE, RagEngine
from .metrics import MetricsWriter, record_for_response
from .retrieval_client import HttpKnowledgeSearchClient

MAX_BODY = 64 * 1024

#: Default provenance sink. Every /rag_answer request appends exactly one
#: redacted line here; a write failure never fails the request.
DEFAULT_METRICS_PATH = "/var/lib/hermes-langgraph/rag-metrics.jsonl"

#: The complete operation surface. Nothing else is reachable.
ROUTES = {
    ("POST", "/rag_answer"),
    ("GET", "/rag_run_status"),
    ("GET", "/rag_health"),
}

#: Deterministic HTTP status mapping for terminal outcomes.
STATUS_CODES = {
    Outcome.OK.value: 200,
    Outcome.INVALID_REQUEST.value: 400,
    Outcome.FRESHNESS_UNAVAILABLE.value: 503,
    Outcome.STALE_INDEX.value: 503,
    Outcome.RETRIEVAL_UNAVAILABLE.value: 503,
    Outcome.RETRIEVAL_TIMEOUT.value: 504,
    Outcome.DEADLINE_EXCEEDED.value: 504,
    Outcome.RETRIEVAL_CONTRACT.value: 502,
    Outcome.NO_EVIDENCE.value: 200,
    Outcome.LOW_RELEVANCE.value: 200,
    Outcome.INVALID_CITATIONS.value: 200,
    Outcome.INTERNAL_ERROR.value: 500,
}


def status_code_for(response: dict) -> int:
    return STATUS_CODES.get(str(response.get("status")), 500)


def build_engine(args: argparse.Namespace) -> RagEngine:
    """Assemble the engine from parsed CLI args (also used by tests)."""
    provider = resolve_checkpointer(args.checkpoint, path=args.checkpoint_path)
    retrieval = HttpKnowledgeSearchClient(
        args.knowledge_search_url, timeout=args.retrieval_timeout,
        max_retries=args.max_retries,
    )
    freshness = HttpFreshnessClient(args.freshness_url, timeout=args.freshness_timeout)
    return RagEngine(
        retrieval=retrieval,
        freshness=freshness,
        checkpointer=provider,
        expected_index_version=args.index_version,
        sla_seconds=args.sla_seconds,
        max_evidence=args.max_evidence,
        relevance_floor=args.relevance_floor,
        request_deadline=getattr(args, "request_deadline", None),
    )


def _explicit_run_id(body: Any) -> str | None:
    """Caller-supplied run id, for metric attribution when a run times out."""
    from .ids import is_valid_run_id

    if not isinstance(body, dict):
        return None
    params = body.get("rag_answer")
    if not isinstance(params, dict):
        return None
    rid = params.get("run_id")
    return rid if is_valid_run_id(rid) else None


def _invoke(engine: RagEngine, body: Any, deadline: float | None) -> tuple[dict | None, bool]:
    """Run the engine, never waiting longer than *deadline* wall clock.

    The engine is also told the same budget so it stops between nodes and
    shortens upstream timeouts. The join is the hard guarantee: a pathological
    transport that ignores its socket timeout cannot hold the caller.
    """
    if not deadline or deadline <= 0:
        return engine.run(body), False
    box: dict[str, Any] = {}

    def worker() -> None:
        try:
            box["response"] = engine.run(body, deadline_seconds=deadline)
        except BaseException as exc:  # noqa: BLE001 - surfaced to the caller
            box["error"] = exc

    thread = threading.Thread(target=worker, name="rag-request", daemon=True)
    thread.start()
    thread.join(deadline)
    if thread.is_alive():
        return None, True
    if "error" in box:
        raise box["error"]
    return box.get("response"), False


def make_handler(engine: RagEngine, *, metrics: MetricsWriter | None = None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "langgraph-rag/0.1"

        def log_message(self, fmt, *args):  # keep logs free of payloads
            sys.stderr.write("langgraph_rag_adapter: %s\n" % (fmt % args))

        def _peer_ok(self) -> bool:
            try:
                return ipaddress.ip_address(self.client_address[0]).is_loopback
            except ValueError:
                return False

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _path_only(self) -> str:
            return self.path.split("?", 1)[0]

        def _query(self) -> dict:
            if "?" not in self.path:
                return {}
            from urllib.parse import parse_qs

            return {k: v[0] for k, v in parse_qs(self.path.split("?", 1)[1]).items()}

        def _emit(self, record: dict) -> None:
            if metrics is not None:
                metrics.emit(record)

        def do_GET(self):  # noqa: N802
            if not self._peer_ok():
                self._send(403, {"ok": False, "error": "loopback clients only"})
                return
            path = self._path_only()
            if path == "/rag_health":
                self._send(200, engine.health())
                return
            if path == "/rag_run_status":
                run_id = self._query().get("run_id", "")
                if not run_id:
                    self._send(400, {"ok": False, "error": "run_id is required",
                                     "error_class": "contract"})
                    return
                stored = engine.status(run_id)
                if stored is None:
                    self._send(404, {"ok": False, "error": "unknown run_id",
                                     "error_class": "not_found", "run_id": run_id})
                    return
                self._send(200, {"ok": True, **stored})
                return
            self._send(404, {"ok": False, "error": "not found"})

        def do_POST(self):  # noqa: N802
            if not self._peer_ok():
                self._send(403, {"ok": False, "error": "loopback clients only"})
                return
            if self._path_only() != "/rag_answer":
                self._send(404, {"ok": False, "error": "only /rag_answer exists"})
                return
            started = time.monotonic()
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                self._emit(record_for_response(
                    None, wall_ms=(time.monotonic() - started) * 1000,
                    status=Outcome.INVALID_REQUEST.value,
                    index_version=engine.expected_index_version,
                ))
                self._send(413, {"ok": False, "error": "invalid body size"})
                return
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception:
                self._emit(record_for_response(
                    None, wall_ms=(time.monotonic() - started) * 1000,
                    status=Outcome.INVALID_REQUEST.value,
                    index_version=engine.expected_index_version,
                ))
                self._send(400, {"ok": False, "error": "invalid JSON"})
                return
            fallback_run_id = _explicit_run_id(body)
            try:
                response, timed_out = _invoke(engine, body, engine.request_deadline)
            except Exception as exc:  # noqa: BLE001 - never leak a traceback
                self._emit(record_for_response(
                    None, wall_ms=(time.monotonic() - started) * 1000,
                    fallback_run_id=fallback_run_id, status=Outcome.INTERNAL_ERROR.value,
                    index_version=engine.expected_index_version,
                ))
                self._send(500, {"ok": False, "error": type(exc).__name__,
                                 "error_class": "internal"})
                return
            wall_ms = (time.monotonic() - started) * 1000
            if timed_out:
                self._emit(record_for_response(
                    None, wall_ms=wall_ms, fallback_run_id=fallback_run_id,
                    status=Outcome.DEADLINE_EXCEEDED.value,
                    index_version=engine.expected_index_version,
                ))
                self._send(504, {
                    "ok": False,
                    "status": Outcome.DEADLINE_EXCEEDED.value,
                    "error": "request exceeded the server-side deadline",
                    "error_class": "deadline",
                    "run_id": fallback_run_id or "unknown",
                    "evidence": [],
                    "citations": [],
                    "deadline_exceeded": True,
                    "completed": False,
                })
                return
            self._emit(record_for_response(
                response, wall_ms=wall_ms, fallback_run_id=fallback_run_id,
                index_version=engine.expected_index_version,
            ))
            self._send(status_code_for(response or {}), response or {})

        def do_PUT(self):  # noqa: N802
            self._send(405, {"ok": False, "error": "write methods are not available"})

        do_DELETE = do_PUT
        do_PATCH = do_PUT

    return Handler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="bounded LangGraph RAG adapter")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--knowledge-search-url", default="http://127.0.0.1:8788/knowledge_search")
    ap.add_argument("--freshness-url", default="http://127.0.0.1:8790/freshness")
    ap.add_argument("--index-version", default="rag-v1")
    ap.add_argument("--sla-seconds", type=int, default=900)
    ap.add_argument("--max-evidence", type=int, default=10)
    ap.add_argument("--relevance-floor", type=float, default=None,
                    help="override the calibrated relevance floor (default: code constant)")
    ap.add_argument("--retrieval-timeout", type=float, default=5.0)
    ap.add_argument("--freshness-timeout", type=float, default=3.0)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--request-deadline", type=float, default=DEFAULT_REQUEST_DEADLINE,
                    help="wall-clock budget for one request in seconds; the response "
                         "never waits past it and upstream calls are shortened to fit "
                         "(0 disables the deadline)")
    ap.add_argument("--metrics-path", default=DEFAULT_METRICS_PATH,
                    help="JSONL provenance sink (one redacted line per /rag_answer); "
                         "empty string disables it")
    ap.add_argument("--checkpoint", choices=[MEMORY, SQLITE], default=SQLITE)
    ap.add_argument("--checkpoint-path", default=DEFAULT_SQLITE_PATH)
    ap.add_argument("--allow-memory", action="store_true",
                    help="development only: permit a non-durable in-memory checkpointer")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        print("refusing to bind a non-loopback address", file=sys.stderr)
        return 2
    if args.checkpoint != SQLITE and not args.allow_memory:
        print("refusing to start with a non-durable checkpointer; "
              "use --checkpoint sqlite (production) or --allow-memory (dev only)",
              file=sys.stderr)
        return 3
    try:
        engine = build_engine(args)
    except Exception as exc:  # noqa: BLE001
        print(f"refusing to start: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    if not engine.checkpointer.durable and not args.allow_memory:
        print("refusing to start: production path requires durable persistence", file=sys.stderr)
        return 3
    server = ThreadingHTTPServer((args.host, args.port),
                                 make_handler(engine, metrics=MetricsWriter(args.metrics_path)))
    print(f"langgraph-rag adapter listening on http://{args.host}:{args.port} "
          f"(checkpointer={engine.health()['checkpointer']['mode']}, "
          f"durable={engine.health()['checkpointer']['durable']}, "
          f"request_deadline={engine.request_deadline}s)", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
