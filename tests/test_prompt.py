"""Prompt rendering tests."""
from __future__ import annotations

import pytest

from agentic_function.core.function import AgenticFunction
from agentic_function.core.prompt import render_prompt
from agentic_function.core.schema import resolve_schema


def _make_fn(name: str = "f", doc: str = "test doc",
             schema: dict | None = None,
             template: str | None = None,
             system_template: str | None = None,
             few_shots: list | None = None) -> AgenticFunction:
    from agentic_function.runtime.retry import RetryPolicy
    from agentic_function.core.schema import SchemaSpec

    spec = resolve_schema(schema or {"result": str})
    sig = __import__("inspect").signature(lambda text: None)
    return AgenticFunction(
        wrapped=lambda text: None,
        name=name, docstring=doc, signature=sig,
        model="mock", backend="mock", output_schema=spec,
        temperature=0.0, max_tokens=None, top_p=None, stop=None,
        retry_policy=RetryPolicy(), cache=False, timeout=30.0,
        include_schema_in_prompt=True,
        few_shots=few_shots or [], prompt_template=template,
        system_template=system_template, description=None,
        executor=None,  # not used by render_prompt
        qualified_name=f"tests.{name}",
    )


def test_default_prompt_uses_docstring():
    fn = _make_fn(doc="This is the system prompt.")
    msgs = render_prompt(fn, ("hello",), {})
    assert msgs[0]["role"] == "system"
    assert "This is the system prompt." in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert "text: hello" in msgs[1]["content"]


def test_schema_appended_to_system():
    fn = _make_fn(schema={"label": str, "score": float})
    msgs = render_prompt(fn, ("hi",), {})
    sys_content = msgs[0]["content"]
    assert "Output format" in sys_content
    assert "label" in sys_content
    assert "score" in sys_content


def test_explicit_user_template():
    fn = _make_fn(template="Translate to French: {text}")
    msgs = render_prompt(fn, ("hello",), {})
    assert msgs[1]["content"] == "Translate to French: hello"


def test_explicit_system_template():
    fn = _make_fn(system_template="You are a {role}.")
    msgs = render_prompt(fn, ("hi",), {})
    # ``{role}`` is preserved literally because no value is bound — useful
    # as a hint to the LLM.
    assert "You are a {role}." in msgs[0]["content"]


def test_few_shots_appended():
    fn = _make_fn(few_shots=[({"text": "good"}, {"result": "positive"})])
    msgs = render_prompt(fn, ("hi",), {})
    # system, user, few-shot user, few-shot assistant
    assert len(msgs) == 4
    assert msgs[2]["role"] == "user"
    assert msgs[3]["role"] == "assistant"
    import json
    out = json.loads(msgs[3]["content"])
    assert out == {"result": "positive"}


def test_include_schema_disabled():
    fn = _make_fn(schema={"x": int})
    fn.include_schema_in_prompt = False
    msgs = render_prompt(fn, ("hi",), {})
    assert "Output format" not in msgs[0]["content"]