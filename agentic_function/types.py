"""Common type definitions used across the library.

We keep these in one place to avoid circular imports between modules.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence, TypeAlias, Union

# A JSON-ish primitive tree. We intentionally stay loose — LLM outputs are messy.
JSONValue: TypeAlias = Union[
    None,
    bool,
    int,
    float,
    str,
    list["JSONValue"],
    dict[str, "JSONValue"],
]

# A schema declaration is a mapping from field name -> Python type (or pydantic
# model class). It can also be a pydantic model class directly.
SchemaDeclaration: TypeAlias = Union[
    type,
    Mapping[str, Any],
]

# A type resolver maps a Python type / pydantic field to a JSON-Schema fragment.
# Used for prompt rendering ("output schema") and OpenAI tool conversion.
TypeResolver: TypeAlias = Callable[[Any], dict[str, Any]]

# Prompt context — what we feed to the template engine.
PromptContext: TypeAlias = Mapping[str, Any]

# Sequence of message dicts (OpenAI-style "role" + "content").
Message: TypeAlias = dict[str, Any]
MessageList: TypeAlias = Sequence[Message]

# A function may return either a single content string or a list of content parts
# (the latter is needed when multimodal output is in play).
LLMOutputContent: TypeAlias = Union[str, list[dict[str, Any]]]