"""Acceptance: the plugin imports in a fresh process and registers safely."""
from __future__ import annotations

import socket
import subprocess
import sys

import yaml

from plugin_loader import PLUGIN_DIR, FakeCtx, plugin


def test_plugin_imports_in_a_fresh_process_with_no_hermes_runtime():
    code = (
        "import importlib.util, sys\n"
        f"d = {str(PLUGIN_DIR)!r}\n"
        "spec = importlib.util.spec_from_file_location('isolated_rag_auto', d + '/__init__.py',\n"
        "                                              submodule_search_locations=[d])\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "m.__package__ = 'isolated_rag_auto'\n"
        "m.__path__ = [d]\n"
        "sys.modules['isolated_rag_auto'] = m\n"
        "spec.loader.exec_module(m)\n"
        "assert callable(m.register)\n"
        "assert m.RAG_ANSWER_SCHEMA['name'] == 'rag_answer'\n"
        "assert not any(k.startswith('hermes_cli') for k in sys.modules)\n"
        "print('fresh-import-ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            timeout=60, cwd=str(PLUGIN_DIR))
    assert result.returncode == 0, result.stderr
    assert "fresh-import-ok" in result.stdout


def test_manifest_declares_exactly_the_hook_and_tool_it_registers():
    manifest = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text())
    assert manifest["name"] == "rag-auto"
    assert manifest["kind"] == "standalone"
    assert manifest["provides_hooks"] == ["pre_llm_call"]
    assert manifest["provides_tools"] == ["rag_answer"]


def test_register_wires_one_hook_and_one_tool_and_opens_no_socket(monkeypatch):
    def forbidden(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("registration must not open a socket")

    monkeypatch.setattr(socket, "socket", forbidden)
    ctx = FakeCtx(profile="developer", config={"metrics.path": ""})
    plugin().register(ctx)
    assert [name for name, _ in ctx.hooks] == ["pre_llm_call"]
    assert callable(ctx.hooks[0][1])
    assert [tool["name"] for tool in ctx.tools] == ["rag_answer"]
    tool = ctx.tools[0]
    assert tool["toolset"] == "rag"
    assert tool["schema"]["parameters"]["required"] == ["query"]
    assert callable(tool["handler"])
    assert len(ctx.unload_callbacks) == 1


def test_tool_schema_is_closed_and_bounded():
    schema = plugin().RAG_ANSWER_SCHEMA
    params = schema["parameters"]
    assert params["additionalProperties"] is False
    assert params["properties"]["top_k"]["maximum"] <= 8


def test_metrics_path_defaults_under_hermes_home_and_empty_disables(monkeypatch):
    monkeypatch.setenv("HERMES_HOME", "/tmp/hermes-rag-auto-test")
    monkeypatch.delenv("HERMES_RAG_AUTO_METRICS", raising=False)
    cfg = plugin().PluginConfig.from_ctx(FakeCtx(config={}))
    assert cfg.metrics_path == "/tmp/hermes-rag-auto-test/rag-auto/metrics.jsonl"
    cfg = plugin().PluginConfig.from_ctx(FakeCtx(config={"metrics.path": ""}))
    assert cfg.metrics_path is None


def test_limits_cannot_be_raised_past_the_hard_caps():
    cfg = plugin().PluginConfig(max_items=99, max_chars=99_999, snippet_chars=9999,
                                deadline_seconds=99.0, graph_budget_seconds=99.0)
    assert cfg.max_items == 8
    assert cfg.max_chars == 6000
    assert cfg.snippet_chars == 400
    assert cfg.deadline_seconds <= 30.0
    assert cfg.graph_budget_seconds <= cfg.deadline_seconds - 0.4
