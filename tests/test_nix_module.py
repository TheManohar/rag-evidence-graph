"""Static boundary checks on the NixOS module (own interpreter, no secrets)."""
from __future__ import annotations

from pathlib import Path

NIX = Path(__file__).resolve().parents[1] / "langgraph-rag.nix"


def read() -> str:
    return NIX.read_text()


def test_module_declares_its_own_langgraph_interpreter():
    src = read()
    # Its own python env with langgraph + the SQLite checkpoint backend;
    # not the shared system-profile python.
    assert "python3.withPackages" in src
    assert "ps.langgraph" in src
    assert "ps.langgraph-checkpoint-sqlite" in src


def test_module_is_inert_unless_enabled():
    src = read()
    assert "mkEnableOption" in src
    assert "mkIf cfg.enable" in src


def test_module_has_no_credentials_or_provider_flags():
    src = read().lower()
    for needle in ("api_key", "api-key", "openai", "anthropic", "token", "password",
                   "secret", "credential", "sops"):
        assert needle not in src


def test_module_binds_loopback_and_enforces_durable_persistence():
    src = read()
    assert 'default = "127.0.0.1"' in src
    assert "--checkpoint sqlite" in src
    assert "/var/lib/hermes-langgraph/checkpoints.sqlite3" in src
    assert 'StateDirectory = "hermes-langgraph"' in src
    assert 'StateDirectoryMode = "0700"' in src
    assert 'UMask = "0077"' in src
