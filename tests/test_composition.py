"""Tests for composition helpers (registry, tool conversion)."""
from __future__ import annotations

import pytest

from agentic_function import agentic_function, AgenticResult, MockBackend, register_backend
from agentic_function.composition import (
    as_openai_tool, as_anthropic_tool, get_function, register, get_global_registry,
)
from agentic_function.errors import RegistrationError


@pytest.fixture
def registered_fn(mock_backend):
    mock_backend.register(lambda req: {"label": "x", "score": 0.1})
    @agentic_function(backend="mock", output_schema={"label": str, "score": float})
    def classify(text: str) -> AgenticResult:
        """Classify the sentiment of ``text``."""
    register(classify, name="classify_sentiment")
    return classify


def test_register_and_lookup(registered_fn):
    fn = get_function("classify_sentiment")
    assert fn is registered_fn


def test_register_duplicate_raises(registered_fn):
    with pytest.raises(RegistrationError):
        register(registered_fn, name="classify_sentiment")


def test_register_overwrite(registered_fn):
    register(registered_fn, name="classify_sentiment", allow_overwrite=True)


def test_lookup_missing_raises():
    with pytest.raises(RegistrationError):
        get_function("does_not_exist")


def test_openai_tool_schema(registered_fn):
    tool = as_openai_tool(registered_fn)
    assert tool["type"] == "function"
    fn = tool["function"]
    assert fn["name"] == "classify"
    assert "sentiment" in fn["description"].lower() or "Classify" in fn["description"]
    assert "text" in fn["parameters"]["properties"]
    assert fn["parameters"]["required"] == ["text"]


def test_anthropic_tool_schema(registered_fn):
    tool = as_anthropic_tool(registered_fn)
    assert tool["name"] == "classify"
    assert "input_schema" in tool


def test_registry_global_singleton():
    assert get_global_registry() is get_global_registry()