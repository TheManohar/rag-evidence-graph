"""Configuration for the rag-auto plugin.

Every value has a safe default, so registration never depends on config.yaml.
Configuration is read through ``ctx.get_config`` which resolves
``plugins.entries.rag-auto.settings.<key>``. Values are clamped, never trusted:
the formatter caps and the hook deadline are hard upper bounds.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Phase-1..3 canary policies are a subset of this list; membership here is
#: the runtime allowlist, and enablement is separately gated by plugins.enabled.
DEFAULT_PROFILE_ALLOWLIST = (
    "developer",
    "default",
    "researcher",
    "reviewer",
    "writer",
    "system-administrator",
    "life-coach",
)

#: Per-profile formatter ceilings (items, characters). PLAN.md §9.1: the
#: conversational life-coach surface gets 5 items / 3,000 chars.
PROFILE_FORMATTER_CAPS: dict[str, tuple[int, int]] = {"life-coach": (5, 3000)}

#: Hard caps from design.md §5. Config may lower these, never raise them.
HARD_MAX_ITEMS = 8
HARD_MAX_CHARS = 6000
HARD_MAX_SNIPPET_CHARS = 400


def default_metrics_path() -> str:
    override = os.environ.get("HERMES_RAG_AUTO_METRICS")
    if override:
        return override
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return str(Path(home) / "rag-auto" / "metrics.jsonl")


def _read(ctx: Any, key: str, default: Any) -> Any:
    getter = getattr(ctx, "get_config", None)
    if not callable(getter):
        return default
    try:
        return getter(key, default)
    except Exception:  # noqa: BLE001 - a broken config read cannot break registration
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        if isinstance(value, bool):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        if isinstance(value, bool):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _as_profiles(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        items = [str(part).strip() for part in value]
    else:
        return default
    cleaned = tuple(item.lower() for item in items if item)
    return cleaned or default


@dataclass(frozen=True)
class PluginConfig:
    graph_url: str = "http://127.0.0.1:8791"
    profiles: tuple[str, ...] = DEFAULT_PROFILE_ALLOWLIST
    deadline_seconds: float = 6.0
    graph_budget_seconds: float = 5.0
    top_k: int = 5
    mode: str = "hybrid"
    per_turn: int = 1
    per_session: int = 20
    per_profile_hourly: int = 120
    dedup_ttl_seconds: float = 300.0
    breaker_failures: int = 2
    breaker_cooldown_seconds: float = 60.0
    max_items: int = HARD_MAX_ITEMS
    max_chars: int = HARD_MAX_CHARS
    snippet_chars: int = HARD_MAX_SNIPPET_CHARS
    inject: bool = True
    metrics_path: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "deadline_seconds", min(30.0, max(0.5, float(self.deadline_seconds))))
        # The graph leg must finish inside the hook deadline, leaving time for
        # the gate, transport and formatting.
        ceiling = max(0.2, self.deadline_seconds - 0.5)
        object.__setattr__(self, "graph_budget_seconds",
                           min(ceiling, max(0.2, float(self.graph_budget_seconds))))
        object.__setattr__(self, "top_k", min(20, max(1, int(self.top_k))))
        object.__setattr__(self, "mode", self.mode if self.mode in {"hybrid", "vector", "keyword"} else "hybrid")
        object.__setattr__(self, "per_turn", max(1, int(self.per_turn)))
        object.__setattr__(self, "per_session", max(1, int(self.per_session)))
        object.__setattr__(self, "per_profile_hourly", max(1, int(self.per_profile_hourly)))
        object.__setattr__(self, "dedup_ttl_seconds", max(1.0, float(self.dedup_ttl_seconds)))
        object.__setattr__(self, "breaker_failures", max(1, int(self.breaker_failures)))
        object.__setattr__(self, "breaker_cooldown_seconds", max(1.0, float(self.breaker_cooldown_seconds)))
        object.__setattr__(self, "max_items", min(HARD_MAX_ITEMS, max(1, int(self.max_items))))
        object.__setattr__(self, "max_chars", min(HARD_MAX_CHARS, max(200, int(self.max_chars))))
        object.__setattr__(self, "snippet_chars", min(HARD_MAX_SNIPPET_CHARS, max(40, int(self.snippet_chars))))
        object.__setattr__(self, "profiles", tuple(p.lower() for p in self.profiles))

    def formatter_limits(self, profile: str) -> tuple[int, int]:
        """Effective (items, chars) for *profile*, honouring the hard caps."""
        cap_items, cap_chars = PROFILE_FORMATTER_CAPS.get(profile, (HARD_MAX_ITEMS, HARD_MAX_CHARS))
        return min(self.max_items, cap_items), min(self.max_chars, cap_chars)

    @classmethod
    def from_ctx(cls, ctx: Any) -> "PluginConfig":
        graph_url = _read(ctx, "graph.url", cls.graph_url)
        sentinel = object()
        metrics_raw = _read(ctx, "metrics.path", sentinel)
        if metrics_raw is sentinel:
            metrics_path: str | None = default_metrics_path()
        elif isinstance(metrics_raw, str) and metrics_raw.strip():
            metrics_path = metrics_raw.strip()
        else:
            metrics_path = None
        return cls(
            graph_url=str(graph_url or cls.graph_url).rstrip("/"),
            profiles=_as_profiles(_read(ctx, "profiles", list(DEFAULT_PROFILE_ALLOWLIST)),
                                  DEFAULT_PROFILE_ALLOWLIST),
            deadline_seconds=_as_float(_read(ctx, "deadline.seconds", 6.0), 6.0),
            graph_budget_seconds=_as_float(_read(ctx, "deadline.graph_seconds", 5.0), 5.0),
            top_k=_as_int(_read(ctx, "graph.top_k", 5), 5),
            mode=str(_read(ctx, "graph.mode", "hybrid") or "hybrid"),
            per_turn=_as_int(_read(ctx, "budget.per_turn", 1), 1),
            per_session=_as_int(_read(ctx, "budget.per_session", 20), 20),
            per_profile_hourly=_as_int(_read(ctx, "budget.per_profile_hourly", 120), 120),
            dedup_ttl_seconds=_as_float(_read(ctx, "dedup.ttl_seconds", 300.0), 300.0),
            breaker_failures=_as_int(_read(ctx, "breaker.failures", 2), 2),
            breaker_cooldown_seconds=_as_float(_read(ctx, "breaker.cooldown_seconds", 60.0), 60.0),
            max_items=_as_int(_read(ctx, "formatter.max_items", HARD_MAX_ITEMS), HARD_MAX_ITEMS),
            max_chars=_as_int(_read(ctx, "formatter.max_chars", HARD_MAX_CHARS), HARD_MAX_CHARS),
            snippet_chars=_as_int(_read(ctx, "formatter.snippet_chars", HARD_MAX_SNIPPET_CHARS),
                                  HARD_MAX_SNIPPET_CHARS),
            inject=_as_bool(_read(ctx, "inject", True), True),
            metrics_path=metrics_path,
        )
