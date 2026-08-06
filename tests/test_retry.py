"""Retry policy tests."""
from __future__ import annotations

from agentic_function.errors import BackendError, ParseError, ValidationError
from agentic_function.runtime.retry import RetryPolicy, is_retryable


def test_delay_increases_with_attempt():
    p = RetryPolicy(max_retries=3, initial_delay=1.0, exponential_base=2.0,
                    jitter=False)
    assert p.delay_for(0) == 0.0
    assert p.delay_for(1) == 1.0
    assert p.delay_for(2) == 2.0
    assert p.delay_for(3) == 4.0


def test_delay_caps_at_max():
    p = RetryPolicy(max_retries=10, initial_delay=1.0, max_delay=3.0,
                    exponential_base=2.0, jitter=False)
    assert p.delay_for(20) == 3.0


def test_is_retryable_transient_errors():
    p = RetryPolicy(max_retries=3)
    assert is_retryable(BackendError("x"), p, 0)
    assert is_retryable(ParseError("x", raw_output=None), p, 0)
    assert is_retryable(ValidationError("x", raw_output=None), p, 0)


def test_is_retryable_exhausted():
    p = RetryPolicy(max_retries=2)
    # attempt 2 == max_retries → no more retries
    assert not is_retryable(BackendError("x"), p, 2)


def test_is_retryable_non_transient():
    p = RetryPolicy(max_retries=3)
    # ValueError is not in TRANSIENT_ERRORS
    assert not is_retryable(ValueError("x"), p, 0)