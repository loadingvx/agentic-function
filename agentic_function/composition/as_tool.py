"""Convert an ``AgenticFunction`` into a tool schema for OpenAI / Anthropic.

This is the bridge that lets Agentic Functions be used **as tools inside a
general-purpose Agent** — the best of both worlds.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable

from ..core.function import AgenticFunction
from ..errors import CompositionError
from ..runtime.executor import ExecutionRequest


def as_openai_tool(fn: AgenticFunction) -> dict[str, Any]:
    """Return the OpenAI ``tools[].function`` JSON schema for ``fn``."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in fn.signature.parameters.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD):
            continue
        ann = param.annotation if param.annotation is not inspect._empty else str
        properties[name] = _annotation_to_schema(ann)
        if param.default is inspect._empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": fn.name,
            "description": fn.description or fn.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def as_anthropic_tool(fn: AgenticFunction) -> dict[str, Any]:
    """Return the Anthropic ``tools[]`` JSON schema for ``fn``."""
    openai = as_openai_tool(fn)["function"]
    return {
        "name": openai["name"],
        "description": openai["description"],
        "input_schema": openai["parameters"],
    }


def tool_invoker(fn: AgenticFunction) -> Callable[..., Any]:
    """Return a plain Python function that, when called, runs ``fn`` synchronously.

    Useful when a tool-call dispatcher wants to invoke a function from raw JSON.
    """
    def invoke(**kwargs: Any) -> Any:
        req = ExecutionRequest(fn=fn, args=(), kwargs=kwargs, is_async=False)
        return fn._executor.execute(req)  # noqa: SLF001 — intentional
    invoke.__name__ = fn.name
    invoke.__doc__ = fn.description or fn.__doc__ or ""
    return invoke


def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
    """Translate a Python annotation into a JSON-Schema fragment."""
    from ..core.schema import _annotation_to_json  # reuse the helper
    return _annotation_to_json(annotation)