"""LLM backends — adapters for OpenAI, Anthropic, MiniMax (via Anthropic),
plus a Mock.
"""
from .base import (
    LLMBackend, LLMResponse, StreamChunk,
    get_backend, register_backend, known_backends,
    get_default_backend, set_default_backend,
)
from .mock_backend import MockBackend
from .openai_backend import OpenAIBackend

# AnthropicBackend / MiniMaxBackend don't require their SDK to be importable
# at registration time — we defer that to first call. The constructor itself
# doesn't need them.
try:
    from .anthropic_backend import AnthropicBackend
    register_backend("anthropic", AnthropicBackend())
except Exception:  # pragma: no cover — defensive
    AnthropicBackend = None  # type: ignore[assignment]

# Register "minimax" — an alias for the Anthropic protocol pointed at
# MiniMax's MiniMax-China endpoint. Env vars:
#   MINIMAX_CN_API_KEY (required) — or MINIMAX_API_KEY / ANTHROPIC_API_KEY
#   MINIMAX_BASE_URL (optional)   — defaults to https://api.minimaxi.com/anthropic
if AnthropicBackend is not None:  # pragma: no cover — same gate as above
    try:
        import dataclasses as _dc
        import os as _os

        @_dc.dataclass
        class _MiniMaxBackend(AnthropicBackend):
            """AnthropicBackend preset for MiniMax / MiniMax-China.

            Default ``base_url`` is taken from ``MINIMAX_BASE_URL`` if set,
            otherwise ``https://api.minimaxi.com/anthropic``. The default
            model is ``MiniMax-M3`` which is what MiniMax-CN currently
            serves; users can override per-call via ``@agentic_function(
            model="...")``.

            We override the dataclass field ``name`` so the registry lists
            it under the expected key, and patch ``__post_init__`` to wire
            MiniMax-specific env defaults.
            """

            name: str = "minimax"

            def __post_init__(self) -> None:
                # Force the MiniMax base URL if the user hasn't pinned one.
                if self.base_url is None:
                    self.base_url = (
                        _os.environ.get("MINIMAX_BASE_URL")
                        or "https://api.minimaxi.com/anthropic"
                    )
                # Pick up the MiniMax keys (priority over ANTHROPIC_API_KEY).
                if self.api_key is None:
                    self.api_key = (
                        _os.environ.get("MINIMAX_CN_API_KEY")
                        or _os.environ.get("MINIMAX_API_KEY")
                    )
                # Default model name for MiniMax.
                if self.default_model == "claude-3-5-sonnet-latest":
                    self.default_model = "MiniMax-M3"

        register_backend("minimax", _MiniMaxBackend())
    except Exception:  # pragma: no cover — defensive
        pass


# OpenAI backend — registered last so the order of ``known_backends`` is
# deterministic.
try:
    register_backend("openai", OpenAIBackend())
except Exception:  # pragma: no cover — defensive
    pass


# "mock" is always last in the alphabetical listing because it's the
# least surprising default.
register_backend("mock", MockBackend())


__all__ = [
    "LLMBackend",
    "LLMResponse",
    "StreamChunk",
    "get_backend",
    "register_backend",
    "known_backends",
    "get_default_backend",
    "set_default_backend",
    "MockBackend",
    "OpenAIBackend",
    "AnthropicBackend",
]