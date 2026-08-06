"""OpenAI backend — supports both OpenAI API and OpenAI-compatible servers
(vLLM, LocalAI, Ollama's OpenAI-mode, etc.).

We use the official ``openai`` Python client when available, but we don't make
it a hard dependency: import errors are deferred so that users who only need
Anthropic or the Mock can skip installing it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from ..core.schema import SchemaSpec
from ..errors import BackendError
from ..utils.logging import get_logger
from .base import LLMBackend, LLMResponse, StreamChunk

_log = get_logger("openai_backend")


def _import_openai() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise BackendError(
            "the 'openai' package is required for the OpenAIBackend. "
            "Install with `pip install agentic-function[openai]`.",
            backend="openai",
        ) from exc
    return OpenAI


def _import_async_openai() -> Any:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise BackendError(
            "the 'openai' package is required for the OpenAIBackend. "
            "Install with `pip install agentic-function[openai]`.",
            backend="openai",
        ) from exc
    return AsyncOpenAI


@dataclass
class OpenAIBackend(LLMBackend):
    """Adapter for OpenAI's Chat Completions API.

    Set ``base_url`` to use a compatible server (vLLM, LocalAI, Ollama's
    OpenAI mode, etc.). Set ``api_key`` from env by default.
    """
    name: str = "openai"
    api_key: str | None = None
    base_url: str | None = None
    organisation: str | None = None
    default_model: str = "gpt-4o-mini"
    request_timeout: float = 60.0

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.base_url is None:
            self.base_url = os.environ.get("OPENAI_BASE_URL")
        if self.organisation is None:
            self.organisation = os.environ.get("OPENAI_ORG_ID")

    # ------------------------------------------------------------------
    # Backend interface
    # ------------------------------------------------------------------
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
    ) -> LLMResponse:
        OpenAI = _import_openai()
        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if self.organisation:
            client_kwargs["organisation"] = self.organisation
        client = OpenAI(**client_kwargs)

        request_kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
        }
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if stop:
            request_kwargs["stop"] = stop

        # Use OpenAI's structured-output mode (json_schema) when possible.
        if _supports_json_schema(model):
            request_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "agentic_function_output",
                    "schema": output_schema.to_json_schema(),
                    "strict": True,
                },
            }
        else:
            request_kwargs["response_format"] = {"type": "json_object"}

        try:
            response = client.chat.completions.create(
                **request_kwargs,
                timeout=timeout or self.request_timeout,
            )
        except Exception as exc:
            raise BackendError(
                f"OpenAI request failed: {exc}",
                backend="openai",
                raw=getattr(exc, "response", None),
            ) from exc

        return _response_from_openai(response)

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
        AsyncOpenAI = _import_async_openai()
        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = AsyncOpenAI(**client_kwargs)

        request_kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
        }
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if stop:
            request_kwargs["stop"] = stop
        if _supports_json_schema(model):
            request_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "agentic_function_output",
                    "schema": output_schema.to_json_schema(),
                    "strict": True,
                },
            }
        else:
            request_kwargs["response_format"] = {"type": "json_object"}

        try:
            response = await client.chat.completions.create(
                **request_kwargs,
                timeout=timeout or self.request_timeout,
            )
        except Exception as exc:
            raise BackendError(
                f"OpenAI async request failed: {exc}",
                backend="openai",
                raw=getattr(exc, "response", None),
            ) from exc

        return _response_from_openai(response)


# Models that understand OpenAI's strict json_schema response_format.
# We default to True for modern models and let the user override by subclassing
# if they hit issues.
_STRICT_SCHEMA_MODELS = (
    "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    "o1", "o1-mini", "o3", "o3-mini", "o4-mini",
)


def _supports_json_schema(model: str) -> bool:
    model_l = model.lower()
    return any(token in model_l for token in _STRICT_SCHEMA_MODELS)


def _response_from_openai(response: Any) -> LLMResponse:
    """Convert an OpenAI chat.completions response to our normalised form."""
    try:
        choice = response.choices[0]
        content = choice.message.content or ""
        finish_reason = choice.finish_reason
    except (AttributeError, IndexError, KeyError) as exc:
        raise BackendError(
            f"unexpected OpenAI response shape: {response!r}",
            backend="openai",
            raw=response,
        ) from exc

    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens",
                               prompt_tokens + completion_tokens) or 0)

    return LLMResponse(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        model=getattr(response, "model", ""),
        finish_reason=finish_reason,
        raw=response,
    )