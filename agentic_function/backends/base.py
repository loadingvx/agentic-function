"""Abstract ``LLMBackend`` and the response container.

A backend is responsible for one thing: take a normalised request (model +
messages + output_schema) and return a normalised ``LLMResponse`` (content +
usage). It MUST NOT know anything about retries, caching, validation, or
agentic function semantics — those are layered on top.

Backends are pluggable: register yours with ``register_backend("name",
YourBackend())`` and reference it by name in ``@agentic_function(backend="name")``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..core.schema import SchemaSpec
from ..errors import BackendError, RegistrationError


# ---------------------------------------------------------------------------
# Response / streaming types
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class StreamChunk:
    """One delta from a streaming response."""
    delta: str
    done: bool = False
    finish_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LLMResponse:
    """Normalised response from any backend."""
    content: Any                                 # str for text, dict when structured
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    finish_reason: str | None = None
    raw: Any = None                              # provider-specific raw response
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------
class LLMBackend(ABC):
    """Base class every backend must implement."""

    name: str = "abstract"

    @abstractmethod
    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        output_schema: SchemaSpec,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        on_chunk: Callable[[StreamChunk], None] | None = None,
        timeout: float | None = None,
    ) -> LLMResponse: ...

    # Async version — backends can opt out by not overriding.
    async def acomplete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        output_schema: SchemaSpec,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        on_chunk: Callable[[StreamChunk], None] | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        import asyncio
        return await asyncio.to_thread(
            self.complete,
            model=model,
            messages=messages,
            output_schema=output_schema,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            on_chunk=on_chunk,
            timeout=timeout,
        )


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------
_registry: dict[str, LLMBackend] = {}


def register_backend(name: str, backend: LLMBackend) -> None:
    if not name or not isinstance(name, str):
        raise RegistrationError(f"backend name must be a non-empty string, got {name!r}")
    if not isinstance(backend, LLMBackend):
        raise RegistrationError(f"backend must be an LLMBackend subclass, got {type(backend).__name__}")
    _registry[name] = backend


def get_backend(name: str) -> LLMBackend:
    backend = _registry.get(name)
    if backend is None:
        raise BackendError(
            f"no backend registered under name {name!r}. "
            f"Known: {sorted(_registry)}",
            backend=name,
        )
    return backend


def known_backends() -> list[str]:
    return sorted(_registry)


# ---------------------------------------------------------------------------
# "Default backend" used when @agentic_function(backend=...) is omitted.
# ---------------------------------------------------------------------------
_default_backend: LLMBackend | None = None


def get_default_backend() -> LLMBackend | None:
    return _default_backend


def set_default_backend(backend: LLMBackend | None) -> None:
    global _default_backend
    _default_backend = backend