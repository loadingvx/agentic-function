"""Result types returned by ``@agentic_function`` calls.

We expose two flavours:

1. **Dynamic result** — when the user declared ``output_schema`` as a plain
   ``dict[str, type]``. The result has attribute access for declared fields
   (and a ``.extras`` dict for anything else).
2. **Pydantic result** — when the user declared ``output_schema`` as a
   ``BaseModel`` subclass (or just used the function's return type annotation).
   The result is a regular pydantic model.

Both flavours carry the same metadata, accessible via private attributes:
    ``raw_output`` — what the LLM actually returned (string or dict)
    ``messages``   — the message list that was sent to the LLM
    ``metrics``    — CallMetrics (tokens, cost, latency, cache_hit, ...)

These are stored as ``PrivateAttr`` so they don't pollute the JSON schema.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator


class _ResultBase(BaseModel):
    """Shared internal base for all agentic-function results.

    Provides private attributes for ``raw_output``, ``messages``, and ``metrics``.
    Subclasses (including user-supplied pydantic models) inherit these.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _raw_output: Any = PrivateAttr(default=None)
    _messages: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _metrics: Any = PrivateAttr(default=None)  # CallMetrics; lazy import

    # Public read-only accessors — users shouldn't have to touch dunders.
    @property
    def metrics(self) -> Any:
        """Execution metrics for this call (latency, tokens, cost, retries)."""
        return self._metrics

    @property
    def raw_output(self) -> Any:
        """What the LLM actually returned (string or dict)."""
        return self._raw_output

    @property
    def messages(self) -> list[dict[str, Any]]:
        """The message list that was sent to the LLM."""
        return self._messages


class AgenticResult(_ResultBase):
    """Marker base class for user-supplied pydantic models used as outputs.

    Inherit from this (instead of raw ``BaseModel``) to get access to
    ``.raw_output``, ``.messages``, and ``.metrics`` populated automatically.
    """


class DynamicResult(_ResultBase):
    """Result object used when ``output_schema`` was a plain dict.

    Field access is dual: declared fields are exposed as attributes; anything
    else is available via ``.extras``. We use ``model_validator`` to capture
    unknown fields into ``extras`` (instead of a custom ``__getattr__`` which
    interferes with Pydantic's ``PrivateAttr`` descriptor protocol).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    extras: dict[str, Any] = {}

    @model_validator(mode="before")
    @classmethod
    def _collect_extras(cls, values: Any) -> Any:
        """Anything not in declared fields goes into ``extras``."""
        if isinstance(values, dict):
            declared = set(cls.model_fields.keys())
            extras = {k: v for k, v in values.items() if k not in declared}
            if extras:
                values = dict(values)
                values["extras"] = {**values.get("extras", {}), **extras}
                for k in extras:
                    values.pop(k, None)
        return values