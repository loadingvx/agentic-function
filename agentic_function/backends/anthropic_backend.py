"""Anthropic backend — supports the Anthropic Messages API and
Anthropic-compatible servers (e.g. MiniMax / MiniMax-China at
``https://api.minimaxi.com/anthropic``).

We use the official ``anthropic`` Python client when available, but we don't
make it a hard dependency: import errors are deferred to first call so users
who only need OpenAI or the Mock can skip installing it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from ..core.schema import SchemaSpec
from ..errors import BackendError
from ..utils.logging import get_logger
from .base import LLMBackend, LLMResponse, StreamChunk

_log = get_logger("anthropic_backend")


def _import_anthropic() -> Any:
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise BackendError(
            "the 'anthropic' package is required for the AnthropicBackend. "
            "Install with `pip install agentic-function[anthropic]`.",
            backend="anthropic",
        ) from exc
    return Anthropic


def _import_async_anthropic() -> Any:
    try:
        from anthropic import AsyncAnthropic
    except ImportError as exc:
        raise BackendError(
            "the 'anthropic' package is required for the AnthropicBackend. "
            "Install with `pip install agentic-function[anthropic]`.",
            backend="anthropic",
        ) from exc
    return AsyncAnthropic


# -------------------------------------------------------------------
# Models known to support tool_use / structured output well enough that
# we can use Anthropic's tool-input-schema flow for strict schema
# enforcement. Everything else falls back to "instruct in the system
# prompt to return JSON matching the schema" mode.
# -------------------------------------------------------------------
_STRICT_TOOL_MODELS: tuple[str, ...] = (
    "claude-3-5",
    "claude-3-7",
    "claude-3-opus",
    "claude-4",
    "claude-opus-4",
    "claude-sonnet-4",
    "claude-haiku-4",
    "minimax",  # MiniMax CN minimax family
    "minimax",
)


def _supports_tool_schema(model: str) -> bool:
    ml = model.lower()
    return any(token in ml for token in _STRICT_TOOL_MODELS)


# -------------------------------------------------------------------
# System-prompt suffix we use when a model doesn't support tool schemas.
# It clearly states the JSON shape and forbids anything else.
# -------------------------------------------------------------------
_JSON_ONLY_SUFFIX = (
    "\n\nIMPORTANT: Respond with ONLY a single JSON object that matches the "
    "schema above. Do NOT include prose, markdown fences, or any commentary. "
    "If a value is unknown, return null (or an empty list/object for "
    "containers). Never refuse."
)


@dataclass
class AnthropicBackend(LLMBackend):
    """Adapter for Anthropic's Messages API.

    Parameters
    ----------
    api_key:
        Defaults to ``ANTHROPIC_API_KEY`` (or ``MINIMAX_CN_API_KEY`` /
        ``MINIMAX_API_KEY`` if those are set — convenient for MiniMax).
    base_url:
        Defaults to ``ANTHROPIC_BASE_URL`` (or
        ``https://api.minimaxi.com/anthropic`` when the env says so).
    default_model:
        Defaults to ``claude-3-5-sonnet-latest``; for MiniMax use the model
        name MiniMax returns in their docs (e.g. ``MiniMax-M3``).
    request_timeout:
        Wall-clock timeout for the API call in seconds.
    max_tokens_default:
        Anthropic requires ``max_tokens`` on every request. We use this
        as the default when the user doesn't pin one explicitly.
    """

    name: str = "anthropic"
    api_key: str | None = None
    base_url: str | None = None
    default_model: str = "claude-3-5-sonnet-latest"
    request_timeout: float = 60.0
    max_tokens_default: int = 1024

    def __post_init__(self) -> None:
        if self.api_key is None:
            # Order: ANTHROPIC_API_KEY, MINIMAX_CN_API_KEY, MINIMAX_API_KEY.
            self.api_key = (
                os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("MINIMAX_CN_API_KEY")
                or os.environ.get("MINIMAX_API_KEY")
            )
        if self.base_url is None:
            self.base_url = (
                os.environ.get("ANTHROPIC_BASE_URL")
                or os.environ.get("MINIMAX_BASE_URL")
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return kwargs

    def _messages_to_anthropic(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert OpenAI-style messages → Anthropic ``(system, messages)``.

        Returns ``(system_text_or_None, anthropic_messages)``. ``system`` is
        the concatenation of all messages that had ``role == "system"``;
        everything else becomes a turn in ``messages``.
        """
        system_chunks: list[str] = []
        anthropic_msgs: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                if content:
                    system_chunks.append(str(content))
            elif role in ("user", "assistant"):
                anthropic_msgs.append({"role": role, "content": str(content)})
            else:
                # tool/function messages — flatten into the prior user turn
                # as a system-side note; we don't expose tools at this layer.
                system_chunks.append(f"[{role}] {content}")
        system = "\n\n".join(system_chunks) if system_chunks else None
        return system, anthropic_msgs

    def _request_kwargs(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        output_schema: SchemaSpec,
        temperature: float | None,
        max_tokens: int | None,
        top_p: float | None,
        stop: list[str] | None,
    ) -> dict[str, Any]:
        system, anthropic_msgs = self._messages_to_anthropic(messages)
        schema_json = output_schema.to_json_schema()

        if system is None:
            system = "You are a helpful assistant that returns strictly-typed JSON."
        system = system + _JSON_ONLY_SUFFIX

        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": anthropic_msgs,
            "max_tokens": max_tokens or self.max_tokens_default,
            "system": system,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if stop:
            kwargs["stop_sequences"] = stop

        # If the model is known to handle tool-input schemas well, we can
        # use Anthropic's ``tools`` flow as a stronger schema constraint.
        # The model will populate the tool's input field with the JSON, and
        # we read it back from ``tool_use.input``.
        if _supports_tool_schema(model):
            tool_name = "agentic_function_output"
            kwargs["tools"] = [
                {
                    "name": tool_name,
                    "description": (
                        "Return a JSON object matching the requested schema. "
                        "Always call this tool exactly once with the result."
                    ),
                    "input_schema": schema_json,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": tool_name}

        return kwargs

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
        Anthropic = _import_anthropic()
        client = Anthropic(**self._client_kwargs())

        kwargs = self._request_kwargs(
            model=model,
            messages=messages,
            output_schema=output_schema,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
        )

        # Anthropic SDK uses ``timeout=`` per-request (not in the constructor).
        call_kwargs: dict[str, Any] = dict(kwargs)
        if timeout is not None:
            call_kwargs["timeout"] = timeout
        elif self.request_timeout:
            call_kwargs["timeout"] = self.request_timeout

        try:
            response = client.messages.create(**call_kwargs)
        except Exception as exc:
            raise BackendError(
                f"Anthropic request failed: {exc}",
                backend="anthropic",
                raw=getattr(exc, "response", None),
            ) from exc

        return _response_from_anthropic(response)

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
        AsyncAnthropic = _import_async_anthropic()
        client = AsyncAnthropic(**self._client_kwargs())

        kwargs = self._request_kwargs(
            model=model,
            messages=messages,
            output_schema=output_schema,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
        )
        call_kwargs: dict[str, Any] = dict(kwargs)
        if timeout is not None:
            call_kwargs["timeout"] = timeout
        elif self.request_timeout:
            call_kwargs["timeout"] = self.request_timeout

        try:
            response = await client.messages.create(**call_kwargs)
        except Exception as exc:
            raise BackendError(
                f"Anthropic async request failed: {exc}",
                backend="anthropic",
                raw=getattr(exc, "response", None),
            ) from exc

        return _response_from_anthropic(response)


# -------------------------------------------------------------------
# Response conversion
# -------------------------------------------------------------------
def _extract_text(response: Any) -> str:
    """Concatenate ``text`` blocks in an Anthropic response."""
    chunks: list[str] = []
    for block in getattr(response, "content", []) or []:
        # New SDK uses ``type``; older used duck-typing.
        btype = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if btype == "text":
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text")
            if text:
                chunks.append(text)
    return "".join(chunks)


def _extract_tool_input(response: Any) -> dict[str, Any] | None:
    """Return the first ``tool_use`` block's ``input`` dict, if any."""
    for block in getattr(response, "content", []) or []:
        btype = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if btype == "tool_use":
            inp = getattr(block, "input", None)
            if inp is None and isinstance(block, dict):
                inp = block.get("input")
            if isinstance(inp, dict):
                return inp
    return None


def _response_from_anthropic(response: Any) -> LLMResponse:
    """Convert an Anthropic ``Message`` to our normalised ``LLMResponse``.

    Preference order for the returned ``content``:

    1. The ``tool_use`` block's ``input`` dict — this is the most reliable
       way to get a structured object back when we constrained the model
       with a tool schema.
    2. The concatenated text blocks — used for the "JSON-only" mode.

    We also attempt to parse the text as JSON (best-effort); if parsing
    succeeds, the parsed dict wins over the raw string. This means downstream
    validation sees a dict (which it can validate against the schema)
    instead of having to re-parse on its own.
    """
    finish_reason = getattr(response, "stop_reason", None)
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    total_tokens = prompt_tokens + completion_tokens

    tool_input = _extract_tool_input(response)
    if tool_input is not None:
        content: Any = tool_input
    else:
        text = _extract_text(response)
        # Best-effort JSON parse: if it's wrapped in fences, strip them.
        stripped = text.strip()
        if stripped.startswith("```"):
            # Drop leading ```json (or ```) and trailing ```
            lines = stripped.splitlines()
            if lines and lines[0].lstrip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].lstrip().startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
        try:
            import json

            content = json.loads(stripped) if stripped else {}
        except Exception:
            # Fall back to raw string; downstream will see validation error.
            content = text

    return LLMResponse(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        model=getattr(response, "model", "") or "",
        finish_reason=finish_reason,
        raw=response,
    )