"""Persistence backends for the LangGraph checkpointer.

Policy (operator decision, resolved)
------------------------------------
The production path uses a **file-backed SQLite** checkpointer. It is never
silently downgraded to in-memory persistence, and the adapter refuses to start
on a non-durable saver unless ``--allow-memory`` is passed for a non-production
smoke run.

Backend availability
--------------------
``langgraph.checkpoint.sqlite.SqliteSaver`` is provided by a purpose-built
interpreter environment (``langgraph`` + ``langgraph-checkpoint-sqlite``). The
module declares its own ``python3.withPackages`` env; the shared system-profile
python is *not* widened, and this backend is imported only from that env.

Path requirements
-----------------
* absolute, outside the Nix store, and **not on tmpfs** (checked against the
  longest matching mount in ``/proc/mounts``);
* directory owned by the service user, mode ``0700``;
* database (and its ``-wal`` / ``-shm`` sidecars, which inherit permissions)
  written with ``UMask=0077`` -> ``0600``;
* default: ``/var/lib/hermes-langgraph/checkpoints.sqlite3``.

Backup / retention
------------------
* There is no host-level backup unit on this box: ``/var/lib/hermes-langgraph``
  must be included in whatever file-level backup covers ``/var/lib``, and a
  consistent copy is taken with ``sqlite3 <db> ".backup <dest>"`` (never a raw
  ``cp`` while the service holds the file open).
* Checkpoints are reconstructible service state. Prune completed/expired
  threads periodically (e.g. ``DELETE FROM checkpoints WHERE thread_id IN
  (SELECT thread_id FROM checkpoints GROUP BY thread_id HAVING max(rowid) <
  ...)``, or drop old threads via the saver API) or the file grows unbounded.
"""
from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

from .errors import CheckpointerUnavailable

MEMORY = "memory"
SQLITE = "sqlite"

DEFAULT_SQLITE_PATH = "/var/lib/hermes-langgraph/checkpoints.sqlite3"

_FORBIDDEN_PREFIXES = ("/nix/store", "/proc", "/sys", "/dev")


def _mount_type(target: str, mounts_file: str = "/proc/mounts") -> str | None:
    """Filesystem type of the mount containing ``target`` (longest prefix)."""
    resolved = os.path.realpath(target)
    best: str | None = None
    best_len = -1
    try:
        with open(mounts_file, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return None
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        mount_point = parts[1].replace("\\040", " ")
        prefix = mount_point.rstrip("/")
        if resolved == mount_point or resolved.startswith(prefix + "/") or prefix == "":
            if len(mount_point) > best_len:
                best_len = len(mount_point)
                best = parts[2]
    return best


def assert_persistent_path(path: str, *, mounts_file: str = "/proc/mounts") -> str:
    """Fail-closed validation of a durable checkpoint path."""
    if not path:
        raise ValueError("a checkpoint path is required")
    if not os.path.isabs(path):
        raise ValueError("checkpoint path must be absolute")
    for prefix in _FORBIDDEN_PREFIXES:
        if path == prefix or path.startswith(prefix + "/"):
            raise ValueError(f"checkpoint path must not live under {prefix}")
    parent = os.path.dirname(path) or "/"
    fs = _mount_type(parent, mounts_file)
    if fs in ("tmpfs", "ramfs"):
        raise ValueError(f"checkpoint path must not be on {fs}: {path}")
    return path


@runtime_checkable
class CheckpointerFactory(Protocol):
    """Builds a LangGraph BaseCheckpointSaver. Injectable for tests."""

    mode: str
    durable: bool

    def build(self) -> Any: ...


class InMemoryCheckpointer:
    """Ephemeral saver. Unit tests and explicit non-production smoke runs only."""

    mode = MEMORY
    durable = False

    def build(self) -> Any:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()


class SqliteCheckpointer:
    """File-backed SQLite saver (durable). Fails closed when unavailable."""

    mode = SQLITE
    durable = True

    def __init__(self, path: str = DEFAULT_SQLITE_PATH, *, mounts_file: str = "/proc/mounts"):
        self.path = assert_persistent_path(path, mounts_file=mounts_file)
        self._mounts_file = mounts_file

    def build(self) -> Any:
        assert_persistent_path(self.path, mounts_file=self._mounts_file)
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on the env
            raise CheckpointerUnavailable(
                "langgraph.checkpoint.sqlite is not importable from this interpreter; "
                "durable SQLite persistence cannot be enabled. Refusing to run the "
                "production path with in-memory persistence."
            ) from exc
        import sqlite3

        conn = sqlite3.connect(self.path, check_same_thread=False)
        return SqliteSaver(conn)


def resolve_checkpointer(mode: str = SQLITE, *, path: str | None = None,
                         mounts_file: str = "/proc/mounts") -> CheckpointerFactory:
    """Return a checkpointer factory. Unknown modes fail closed."""
    if mode == MEMORY:
        return InMemoryCheckpointer()
    if mode == SQLITE:
        return SqliteCheckpointer(path or DEFAULT_SQLITE_PATH, mounts_file=mounts_file)
    raise ValueError(f"unknown checkpointer mode {mode!r} (expected 'memory' or 'sqlite')")
