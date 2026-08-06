"""Global, mutable-by-design configuration.

We don't try to make this a sophisticated DI container. It's a single
singleton-ish object that holds defaults the user can override globally or
per-function. Thread-safety: best-effort — the common pattern is "set once at
startup, read many".
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any

from ..utils.logging import get_logger

_log = get_logger("config")


@dataclass
class GlobalConfig:
    """Defaults applied to every ``@agentic_function`` unless overridden."""

    # Backend selection
    default_backend: str = "openai"            # "openai" | "anthropic" | "ollama" | "mock" | custom
    default_model: str = "gpt-4o-mini"

    # Generation defaults
    temperature: float = 0.2
    max_tokens: int | None = None
    top_p: float | None = None

    # Robustness
    max_retries: int = 2
    retry_initial_delay: float = 0.5
    retry_max_delay: float = 8.0
    request_timeout: float = 60.0

    # Cache
    cache_enabled: bool = False
    cache_dir: str | None = None                # if set + cache_enabled, use DiskCache

    # Tracing
    tracing_enabled: bool = True
    trace_include_messages: bool = True         # include prompts/responses in spans

    # Misc
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def clone(self, **overrides: Any) -> "GlobalConfig":
        """Return a copy with the given fields replaced."""
        return replace(self, **overrides)


# Module-level singleton, lazily initialised.
_global_config = GlobalConfig()


def global_config() -> GlobalConfig:
    """Return the process-wide config singleton."""
    return _global_config


def configure(**kwargs: Any) -> GlobalConfig:
    """Override one or more global defaults.

    Example::

        from agentic_function import configure
        configure(default_model="gpt-4o", cache_enabled=True, max_retries=3)
    """
    global _global_config
    if not kwargs:
        return _global_config
    _global_config = _global_config.clone(**kwargs)
    _log.debug("configured: %s", kwargs)
    return _global_config


def _env_defaults() -> dict[str, Any]:
    """Read a few well-known environment variables. Applied at import time."""
    out: dict[str, Any] = {}
    if v := os.environ.get("OPENAI_API_KEY"):
        out["openai_api_key"] = v
    if v := os.environ.get("ANTHROPIC_API_KEY"):
        out["anthropic_api_key"] = v
    if v := os.environ.get("AGENTIC_FUNCTION_MODEL"):
        out["default_model"] = v
    if v := os.environ.get("AGENTIC_FUNCTION_BACKEND"):
        out["default_backend"] = v
    if v := os.environ.get("AGENTIC_FUNCTION_CACHE"):
        out["cache_enabled"] = v.lower() in {"1", "true", "yes", "on"}
    if v := os.environ.get("AGENTIC_FUNCTION_CACHE_DIR"):
        out["cache_dir"] = v
    return out


# Apply env-derived defaults once, at import.
configure(**_env_defaults())