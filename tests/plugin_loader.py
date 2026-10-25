"""Load the rag-auto plugin exactly the way Hermes' directory loader does.

The plugin is not on sys.path (it is a directory artifact under
``plugin/rag-auto``), so tests mirror ``_load_directory_module``:
spec_from_file_location with the plugin dir as the package search location.
"""
from __future__ import annotations

import copy
import importlib
import importlib.util
import sys
import time
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "plugin" / "rag-auto"
MODULE_NAME = "hermes_plugins.rag_auto"

_PLUGIN = None


def load_plugin():
    global _PLUGIN
    if "hermes_plugins" not in sys.modules:
        import types

        namespace = types.ModuleType("hermes_plugins")
        namespace.__path__ = []  # type: ignore[attr-defined]
        sys.modules["hermes_plugins"] = namespace
    for key in [name for name in sys.modules if name == MODULE_NAME or name.startswith(MODULE_NAME + ".")]:
        sys.modules.pop(key, None)
    spec = importlib.util.spec_from_file_location(
        MODULE_NAME, PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.__package__ = MODULE_NAME
    module.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    _PLUGIN = module
    return module


def plugin():
    return _PLUGIN or load_plugin()


def submodule(name: str):
    plugin()
    return importlib.import_module(f"{MODULE_NAME}.{name}")


class FakeCtx:
    def __init__(self, profile: str = "developer", config: dict | None = None) -> None:
        self.profile_name = profile
        self.config = dict(config or {})
        self.hooks: list[tuple[str, object]] = []
        self.tools: list[dict] = []
        self.unload_callbacks: list[object] = []

    def get_config(self, key: str, default=None):
        return self.config.get(key, default)

    def register_hook(self, name: str, callback) -> None:
        self.hooks.append((name, callback))

    def register_tool(self, **kwargs) -> None:
        self.tools.append(kwargs)

    def on_unload(self, callback) -> None:
        self.unload_callbacks.append(callback)


def citation(source_id: str = "s1", snippet: str = "a recorded decision", **over) -> dict:
    item = {
        "source_id": source_id, "space_id": "sp1", "updated_at": 111,
        "index_version": "rag-v1", "score": 0.42,
        "source_url": f"anytype://space/sp1/object/{source_id}", "snippet": snippet,
    }
    item.update(over)
    return item


def ok_response(evidence: list[dict] | None = None, **over) -> dict:
    evidence = [citation()] if evidence is None else evidence
    response = {
        "ok": True, "status": "ok", "run_id": "hermes-auto:test:0",
        "index_version": "rag-v1", "evidence": evidence, "citations": evidence,
        "freshness": {"status": "fresh", "age_seconds": 120.5},
        "replayed": False,
    }
    response.update(over)
    return response


class FakeTransport:
    def __init__(self, response: dict | None = None, *, health: dict | Exception | None = None,
                 error: Exception | None = None, delay: float = 0.0) -> None:
        self.response = response if response is not None else ok_response()
        self.health_payload = {"ok": True} if health is None else health
        self.error = error
        self.delay = delay
        self.answers: list[dict] = []
        self.health_calls = 0

    def answer(self, *, query, top_k, mode, run_id, timeout):
        self.answers.append({"query": query, "top_k": top_k, "mode": mode,
                             "run_id": run_id, "timeout": timeout})
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return copy.deepcopy(self.response)

    def health(self, *, timeout):
        self.health_calls += 1
        if isinstance(self.health_payload, Exception):
            raise self.health_payload
        return dict(self.health_payload)


class FakeClock:
    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.now = start

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_service(tmp_path, *, profile: str = "developer", transport: FakeTransport | None = None,
                 config: dict | None = None, clock: FakeClock | None = None):
    mod = plugin()
    clock = clock or FakeClock()
    cfg = mod.PluginConfig.from_ctx(FakeCtx(profile=profile, config=config))
    sink = mod.MetricSink(tmp_path / "metrics.jsonl")
    service = mod.RagAutoService(
        config=cfg, transport=transport or FakeTransport(), sink=sink,
        profile_name=lambda: profile, clock=clock.time, monotonic=clock.monotonic,
    )
    return service, sink, clock, cfg


def metric_lines(tmp_path) -> list[dict]:
    import json

    path = tmp_path / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
