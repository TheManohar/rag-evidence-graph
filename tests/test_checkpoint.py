"""Acceptance: persistence backends (durable SQLite + test memory), no silent fallback."""
from __future__ import annotations

import pytest

from langgraph_rag.checkpoint import (DEFAULT_SQLITE_PATH, CheckpointerFactory,
                                      InMemoryCheckpointer, SqliteCheckpointer,
                                      assert_persistent_path, resolve_checkpointer)

MOUNTS = """\
/dev/vda4 / ext4 rw,relatime 0 0
tmpfs /run tmpfs rw,nosuid,nodev 0 0
tmpfs /dev/shm tmpfs rw,nosuid,nodev 0 0
"""


def _mounts_file(tmp_path) -> str:
    path = tmp_path / "mounts"
    path.write_text(MOUNTS)
    return str(path)


def test_memory_provider_builds_in_memory_saver():
    provider = InMemoryCheckpointer()
    assert provider.mode == "memory" and provider.durable is False
    assert type(provider.build()).__name__ == "InMemorySaver"


def test_sqlite_path_validation_fails_closed(tmp_path):
    mounts = _mounts_file(tmp_path)
    with pytest.raises(ValueError):
        SqliteCheckpointer("", mounts_file=mounts)          # empty
    with pytest.raises(ValueError):
        SqliteCheckpointer("relative.sqlite3", mounts_file=mounts)  # not absolute
    with pytest.raises(ValueError):
        SqliteCheckpointer("/nix/store/abc/checkpoints.sqlite3", mounts_file=mounts)
    with pytest.raises(ValueError):
        SqliteCheckpointer("/run/langgraph/checkpoints.sqlite3", mounts_file=mounts)  # tmpfs
    with pytest.raises(ValueError):
        SqliteCheckpointer("/dev/shm/checkpoints.sqlite3", mounts_file=mounts)


def test_assert_persistent_path_accepts_var_lib(tmp_path):
    mounts = _mounts_file(tmp_path)
    assert assert_persistent_path(str(tmp_path / "cp.sqlite3"), mounts_file=mounts)
    assert assert_persistent_path(DEFAULT_SQLITE_PATH, mounts_file=mounts)


def test_sqlite_provider_default_path_is_var_lib_hermes_langgraph():
    assert DEFAULT_SQLITE_PATH == "/var/lib/hermes-langgraph/checkpoints.sqlite3"
    assert SqliteCheckpointer().durable is True


def test_sqlite_provider_never_falls_back_to_memory(tmp_path):
    provider = SqliteCheckpointer(str(tmp_path / "cp.sqlite3"), mounts_file=_mounts_file(tmp_path))
    saver = provider.build()  # backend is available in the declared env
    assert "Sqlite" in type(saver).__name__


@pytest.mark.parametrize("mode", ["bogus", "", "postgres"])
def test_unknown_mode_fails_closed(mode):
    with pytest.raises(ValueError):
        resolve_checkpointer(mode)


def test_resolve_both_modes(tmp_path):
    assert isinstance(resolve_checkpointer("memory"), InMemoryCheckpointer)
    sqlite = resolve_checkpointer("sqlite", path=str(tmp_path / "cp.sqlite3"),
                                  mounts_file=_mounts_file(tmp_path))
    assert isinstance(sqlite, SqliteCheckpointer)


def test_providers_satisfy_factory_protocol():
    for p in (InMemoryCheckpointer(), SqliteCheckpointer(path="/var/lib/hermes-langgraph/x.sqlite3")):
        assert isinstance(p, CheckpointerFactory)
