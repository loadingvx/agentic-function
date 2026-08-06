"""In-process mock backend for tests and offline demos.

Users can plug in canned responses (by request signature) or a default JSON
template built from the function's output_schema.

Example::

    backend = MockBackend()
    backend.register(lambda req: {"category": "positive", "confidence": 0.9})
    register_backend("mock", backend)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.schema import SchemaSpec
from ..utils.hashing import stable_hash
from ..utils.logging import get_logger
from .base import LLMBackend, LLMResponse, StreamChunk

_log = get_logger("mock_backend")


RequestSignature = tuple[str, tuple[tuple[str, str], ...]]


@dataclass
class _Registration:
    """A canned response bound to a request signature."""
    signature: RequestSignature
    handler: Callable[[dict[str, Any]], Any]
    call_count: int = 0
    in_order: bool = True


class MockBackend(LLMBackend):
    """A backend that returns canned responses without making any network calls.

    Two ways to register responses:

    1. ``register_with_schema(schema)`` — generate a JSON dump matching the
       schema's JSON Schema. Useful for the simplest "just give me a result"
       test.
    2. ``register(callable)`` — full programmatic control. The callable receives
       the normalised request and returns either a dict (will be validated)
       or a raw LLMResponse.

    If no registration matches, ``MockBackend.default_response`` is returned.
    """

    name = "mock"

    def __init__(self) -> None:
        self._registrations: list[_Registration] = []
        self.calls: list[dict[str, Any]] = []
        self.default_response: Any | None = None
        # Force-canned mode (e.g. for adversarial / robustness tests).
        self.force_string_output: bool = False

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------
    def register(self, handler: Callable[[dict[str, Any]], Any],
                 *, signature: RequestSignature | None = None) -> None:
        """Register a handler. Optionally scope it to a specific signature.

        If ``signature`` is ``None``, the registration acts as a catch-all
        that matches every request (the simplest and most common case for
        tests).
        """
        sig: RequestSignature | None
        if signature is None:
            sig = None  # sentinel for catch-all
        else:
            sig = signature
        self._registrations.append(_Registration(signature=sig, handler=handler,
                                                 in_order=False))

    def register_with_schema(self, schema: SchemaSpec,
                             *, values: dict[str, Any] | None = None) -> None:
        """Register a handler that emits a JSON object matching ``schema``."""
        values = values or self._sensible_defaults(schema)

        def handler(req: dict[str, Any]) -> Any:
            # Return values; the schema-coercion layer will validate.
            return dict(values)

        self.register(handler)

    @staticmethod
    def _sensible_defaults(schema: SchemaSpec) -> dict[str, Any]:
        # Skip reserved/internal fields when building defaults — the handler
        # will produce them automatically if needed.
        reserved = {"extras"}
        defaults: dict[str, Any] = {}
        for name, field in schema.model_class.model_fields.items():
            if name in reserved:
                continue
            annotation = field.annotation
            if annotation is str:
                defaults[name] = f"mock_{name}"
            elif annotation is int:
                defaults[name] = 1
            elif annotation is float:
                defaults[name] = 0.5
            elif annotation is bool:
                defaults[name] = True
            else:
                origin = getattr(annotation, "__origin__", None)
                if origin in (list, tuple, set, frozenset):
                    defaults[name] = []
                elif origin in (dict,):
                    defaults[name] = {}
                else:
                    defaults[name] = None
        return defaults

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
        on_chunk: Any = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        req = {
            "model": model,
            "messages": messages,
            "output_schema": output_schema,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        self.calls.append(req)

        # 1. Find a matching registration.
        #    a) registrations with an explicit signature are matched against
        #       the request's signature, regardless of in_order.
        #    b) catch-all (signature=None) matches everything.
        #    A handler that raises ``_FixtureExhausted`` is silently skipped
        #    so that ``mock_llm`` can stack call-counted fixtures.
        for reg in self._registrations:
            if reg.signature is None:
                # Catch-all.
                try:
                    reg.call_count += 1
                    result = reg.handler(req)
                except Exception as _ex:
                    if _ex.__class__.__name__ == "_FixtureExhausted":
                        continue
                    raise
                return self._materialise(result, output_schema, req)
            if reg.signature == _sig_of(req):
                # Specific match.
                try:
                    reg.call_count += 1
                    result = reg.handler(req)
                except Exception as _ex:
                    if _ex.__class__.__name__ == "_FixtureExhausted":
                        continue
                    raise
                return self._materialise(result, output_schema, req)

        # 2. Fall back to the default response.
        if self.default_response is not None:
            result = self.default_response
            return self._materialise(result, output_schema, req)

        # 3. Last resort: build something from the schema.
        return self._materialise(self._sensible_defaults(output_schema),
                                 output_schema, req)

    def _materialise(self, result: Any, schema: SchemaSpec,
                     req: dict[str, Any]) -> LLMResponse:
        """Wrap a user-returned value into a real ``LLMResponse``."""
        if isinstance(result, LLMResponse):
            return result

        # If the handler returned a string, pass it through as-is — the
        # executor's parser will try to extract JSON from it.
        if isinstance(result, str):
            return LLMResponse(
                content=result,
                prompt_tokens=_approx_tokens(req["messages"]),
                completion_tokens=_approx_tokens(result),
                total_tokens=_approx_tokens(req["messages"]) + _approx_tokens(result),
                model=req["model"],
            )

        # ``force_string_output`` simulates a model that wraps its JSON
        # output in a string (which is what most real LLMs do).
        if self.force_string_output:
            text = json.dumps(result, ensure_ascii=False)
            return LLMResponse(
                content=text,
                prompt_tokens=_approx_tokens(req["messages"]),
                completion_tokens=_approx_tokens(text),
                total_tokens=_approx_tokens(req["messages"]) + _approx_tokens(text),
                model=req["model"],
            )

        # Default — return as a dict (the executor's parser handles both).
        text = json.dumps(result, ensure_ascii=False)
        return LLMResponse(
            content=result,
            prompt_tokens=_approx_tokens(req["messages"]),
            completion_tokens=_approx_tokens(text),
            total_tokens=_approx_tokens(req["messages"]) + _approx_tokens(text),
            model=req["model"],
        )


def _sig_of(req: dict[str, Any]) -> RequestSignature:
    """Derive a deterministic signature from a request."""
    msgs = req["messages"]
    return (
        req["model"],
        tuple(
            (m.get("role", "?"), stable_hash(m.get("content", ""))[:16])
            for m in msgs
        ),
    )


def _approx_tokens(content: Any) -> int:
    """Rough token estimate (~4 chars per token). Good enough for mocks."""
    if isinstance(content, str):
        return max(1, len(content) // 4)
    if isinstance(content, list):
        return sum(_approx_tokens(m.get("content", "")) for m in content
                   if isinstance(m, dict))
    return 1