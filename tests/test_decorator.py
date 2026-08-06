"""End-to-end tests for the decorator and executor."""
from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from agentic_function import (
    agentic_function,
    AgenticResult,
    MockBackend,
    register_backend,
)
from agentic_function.errors import RetryExhaustedError


# ---------------------------------------------------------------------------
# Basic happy-path
# ---------------------------------------------------------------------------
def test_dynamic_schema_call(mock_backend):
    mock_backend.register(lambda req: {"label": "positive", "score": 0.9})

    @agentic_function(
        backend="mock",
        output_schema={"label": str, "score": float},
    )
    def classify(text: str) -> AgenticResult:
        """Classify sentiment."""

    r = classify("I love this!")
    assert r.label == "positive"
    assert r.score == 0.9
    assert r._raw_output == {"label": "positive", "score": 0.9}
    assert len(r._messages) == 2
    assert r._metrics.successful


def test_pydantic_schema_call(mock_backend):
    from pydantic import BaseModel

    class Out(BaseModel):
        sentiment: str
        confidence: float

    mock_backend.register(lambda req: {"sentiment": "negative", "confidence": 0.8})

    @agentic_function(backend="mock", output_schema=Out)
    def classify(text: str) -> Out:
        """Classify."""

    r = classify("terrible")
    assert isinstance(r, Out)
    assert r.sentiment == "negative"
    assert r.confidence == 0.8


def test_inherit_from_agentic_result(mock_backend):
    """User's pydantic models can inherit from AgenticResult for auto-attrs."""
    from agentic_function import AgenticResult

    class Out(AgenticResult):
        label: str

    mock_backend.register(lambda req: {"label": "positive"})

    @agentic_function(backend="mock", output_schema=Out)
    def f(text: str) -> Out:
        """Classify."""

    r = f("hi")
    assert r.label == "positive"
    # Auto-private attrs populated:
    assert r._raw_output == {"label": "positive"}
    assert r._metrics.successful


# ---------------------------------------------------------------------------
# Decorator forms
# ---------------------------------------------------------------------------
def test_bare_decorator_with_pydantic_return(mock_backend):
    # ``Out`` lives at module scope so the return annotation can be resolved.
    # (Under ``from __future__ import annotations`` the annotation becomes a
    # string that Python resolves via globals, not the enclosing locals.)
    mock_backend.register(lambda req: {"x": 42})

    @agentic_function(backend="mock")
    def f(text: str) -> _ModuleOut:
        """Bare decorator."""

    assert f("hi").x == 42


class _ModuleOut(BaseModel):
    x: int


def test_decorator_with_no_schema_raises():
    with pytest.raises(Exception):
        @agentic_function(backend="mock")
        def f(text: str) -> str:
            """No schema."""


# ---------------------------------------------------------------------------
# JSON parsing robustness
# ---------------------------------------------------------------------------
def test_response_with_markdown_fence(mock_backend):
    mock_backend.force_string_output = True
    mock_backend.register(lambda req: {"label": "neutral", "score": 0.5})

    @agentic_function(backend="mock", output_schema={"label": str, "score": float})
    def classify(text: str) -> AgenticResult:
        """..."""

    r = classify("hi")
    assert r.label == "neutral"
    assert r.score == 0.5


def test_response_with_garbage_around_json(mock_backend):
    mock_backend.force_string_output = True

    def handler(req):
        return (
            "Sure! Here's my analysis:\n"
            "```json\n"
            '{"label": "negative", "score": 0.2}\n'
            "```\n"
            "Hope that helps!"
        )

    mock_backend.register(handler)

    @agentic_function(backend="mock", output_schema={"label": str, "score": float})
    def classify(text: str) -> AgenticResult:
        """..."""

    r = classify("hi")
    assert r.label == "negative"
    assert r.score == 0.2


# ---------------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------------
def test_retry_then_success(mock_backend):
    """Handler fails first time, succeeds second."""
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("flaky")  # not a transient error → won't retry
        return {"x": 1}

    mock_backend.register(handler)

    @agentic_function(backend="mock", output_schema={"x": int}, max_retries=2)
    def f(text: str) -> AgenticResult:
        """..."""

    with pytest.raises(RetryExhaustedError):
        f("hi")


def test_max_retries_with_transient_error_succeeds(mock_backend):
    """BackendError IS retried by default."""
    from agentic_function.errors import BackendError

    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] < 2:
            raise BackendError("transient")
        return {"x": 99}

    mock_backend.register(handler)

    @agentic_function(backend="mock", output_schema={"x": int}, max_retries=2)
    def f(text: str) -> AgenticResult:
        """..."""

    r = f("hi")
    assert r.x == 99
    assert calls["n"] == 2
    # Retry observability — critical for model unit-test evals
    assert r.metrics.attempts == 2
    assert r.metrics.retries == 1
    assert r.metrics.recovered is True
    assert r.metrics.error is None
    assert r.metrics.error_category is None
    assert len(r.metrics.attempt_errors) == 1
    assert r.metrics.attempt_errors[0].category == "backend"


def test_retry_exhausted_carries_metrics_and_categories(mock_backend):
    from agentic_function.errors import BackendError, RetryExhaustedError

    mock_backend.register(lambda req: (_ for _ in ()).throw(BackendError("always")))

    # max_retries=2 → attempts 0..2 (3 total)
    @agentic_function(backend="mock", output_schema={"x": int}, max_retries=2)
    def f(text: str) -> AgenticResult:
        """..."""

    with pytest.raises(RetryExhaustedError) as ei:
        f("hi")
    exc = ei.value
    assert exc.attempts == 3
    assert exc.retries == 2
    assert exc.error_category == "backend"
    assert exc.metrics is not None
    assert exc.metrics.successful is False
    assert len(exc.attempt_errors) == 3
    assert all(a.category == "backend" for a in exc.attempt_errors)


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------
def test_async_call(mock_backend):
    mock_backend.register(lambda req: {"x": 5})

    @agentic_function(backend="mock", output_schema={"x": int})
    def f(text: str) -> AgenticResult:
        """..."""

    result = asyncio.run(f.acall("hi"))
    assert result.x == 5


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def test_cache_hit(mock_backend):
    mock_backend.register(lambda req: {"x": 1})

    @agentic_function(backend="mock", output_schema={"x": int}, cache=True)
    def f(text: str) -> AgenticResult:
        """..."""

    r1 = f("hello")
    r2 = f("hello")
    assert r1.x == r2.x == 1
    assert mock_backend.calls == [mock_backend.calls[0]]  # called only once


def test_cache_disabled(mock_backend):
    mock_backend.register(lambda req: {"x": 1})

    @agentic_function(backend="mock", output_schema={"x": int}, cache=False)
    def f(text: str) -> AgenticResult:
        """..."""

    f("hello")
    f("hello")
    assert len(mock_backend.calls) == 2


# ---------------------------------------------------------------------------
# Descriptor / class method use
# ---------------------------------------------------------------------------
def test_method_decoration(mock_backend):
    mock_backend.register(lambda req: {"label": "x"})

    class Service:
        @agentic_function(backend="mock", output_schema={"label": str})
        def classify(self, text: str) -> AgenticResult:
            """..."""

    s = Service()
    r = s.classify("hi")
    assert r.label == "x"

    # Class-level access also works (returns the descriptor itself)
    raw = Service.classify
    assert raw is not None