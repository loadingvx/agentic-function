"""Testing utilities for agentic functions.

Headline features
-----------------

* :func:`mock_llm` — register a canned response for the next call.
* :func:`mock_llm_table` — register a *sequence* of responses indexed by
  the call number or by a custom key (great for parameterised tests).
* :func:`freeze_time` — deterministic ``time.perf_counter`` deltas so
  timing assertions are stable across machines.
* :func:`isolate_execution` — context manager that swaps the default
  executor / cache / backend / aggregator / budget tracker for fresh
  in-process ones, then restores them on exit. Use this for hermetic tests.
* :func:`capture_metrics` — context manager that returns a list to which
  the executor appends every ``CallMetrics`` produced inside the ``with``
  block (success and failure).
* :func:`eval_summary` — turn a list of ``CallMetrics`` into retry-rate /
  failure-category stats for model unit-test evals.

Design note
-----------
``mock_llm`` and ``mock_llm_table`` install a *new* :class:`MockBackend`
on the default registry each call. To layer multiple fixtures across the
same backend, pass ``backend=...`` explicitly.
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from .backends.mock_backend import MockBackend
from .backends.base import set_default_backend
from .runtime.metrics import CallMetrics


# ----------------------------------------------------------------------
# Mock registration
# ----------------------------------------------------------------------
@dataclass(slots=True)
class _FixtureHandle:
    """A handle returned by ``mock_llm`` / ``mock_llm_table`` so callers can
    inspect or reset the fixture mid-test."""
    backend: MockBackend
    remaining: list[int]

    def reset(self) -> None:
        self.remaining.clear()


class _FixtureExhausted(Exception):
    """Internal sentinel — never escapes the MockBackend."""


def mock_llm(
    output: Any,
    *,
    n_calls: int = 1,
    backend: MockBackend | None = None,
) -> MockBackend:
    """Install a global mock backend that returns ``output`` for ``n_calls`` calls.

    See the original docstring in earlier versions. ``mock_llm_table`` is
    preferred for multi-call tests.
    """
    return mock_llm_table([output] * n_calls, backend=backend)


def mock_llm_table(
    rows: list[Any],
    *,
    key_fn: Callable[[dict[str, Any]], int] | None = None,
    backend: MockBackend | None = None,
) -> MockBackend:
    """Register a sequence of responses, indexed by call order (default) or
    by ``key_fn(request) -> index``.

    Parameters
    ----------
    rows:
        Outputs to return. ``rows[0]`` is returned for the first call,
        ``rows[1]`` for the second, etc. If a call arrives after the table
        is exhausted, the last entry is repeated (so you can build
        "always returns X" behaviour with a 1-element table).
    key_fn:
        Optional callable that maps a request to a row index. Useful when
        the call order is non-deterministic but the *content* drives the
        fixture. Example::

            key_fn = lambda req: 0 if "happy" in req.messages[1]["content"] else 1
    backend:
        Optional backend to register on (otherwise a new one is installed
        as the process default).

    Returns
    -------
    MockBackend
        The backend that received the registration. Save it if you want to
        inspect call counts.
    """
    if not rows:
        raise ValueError("mock_llm_table requires at least one row")
    if backend is None:
        backend = MockBackend()
        set_default_backend(backend)

    counter = [0]

    def handler(request: Any) -> Any:
        if key_fn is not None:
            idx = key_fn(request)
        else:
            idx = counter[0]
            counter[0] += 1
        # Clamp to the last row if exhausted.
        idx = min(max(idx, 0), len(rows) - 1)
        return rows[idx]

    backend.register(handler)
    return backend


# ----------------------------------------------------------------------
# Time freezing — deterministic perf_counter deltas.
# ----------------------------------------------------------------------
@contextlib.contextmanager
def freeze_time(start: float = 1000.0) -> Iterator[None]:
    """Make ``time.perf_counter()`` return deterministic values.

    Every ``time.perf_counter()`` call within the ``with`` block returns
    ``start + n * 0.001`` where ``n`` is incremented per call. This is
    enough granularity for ``CallMetrics.timings`` deltas to be exactly
    reproducible across machines.

    Usage::

        with freeze_time():
            r = classify("text")
            assert r.metrics.timings.backend_call_ms == pytest.approx(1.0)
    """
    counter = [0]
    original = time.perf_counter

    def fake_perf_counter() -> float:
        counter[0] += 1
        return start + counter[0] * 0.001

    time.perf_counter = fake_perf_counter  # type: ignore[assignment]
    try:
        yield
    finally:
        time.perf_counter = original  # type: ignore[assignment]


# ----------------------------------------------------------------------
# Isolation — swap executor / cache / backend / aggregator for fresh ones.
# ----------------------------------------------------------------------
@contextlib.contextmanager
def isolate_execution(
    *,
    cache: Any | None = None,
    backend: Any | None = None,
    aggregator: Any | None = None,
    budget_tracker: Any | None = None,
) -> Iterator[dict[str, Any]]:
    """Replace the process-wide executor / cache / backend / aggregator /
    budget tracker for the duration of the ``with`` block.

    Everything that was previously installed is restored on exit. Use this
    when a test mutates globals (e.g. ``set_default_backend(...)``) and
    you don't want the next test to inherit them.

    Returns a ``dict`` with the *fresh* objects, so the test can poke at
    them::

        with isolate_execution() as ctx:
            backend = ctx["backend"]
            backend.register_with_schema(MyOutput)
            out = classify("text")
            assert ctx["aggregator"].summary()["totals"]["call_count"] == 1
    """
    from .core.decorator import get_default_executor, set_default_executor
    from .runtime.cache import get_cache, set_cache
    from .runtime.aggregator import Aggregator, install_default_aggregator
    from .runtime.budget import install_budget_tracker

    saved = {
        "executor": get_default_executor(),
        "cache": get_cache(),
        "aggregator": install_default_aggregator(None),
        "budget_tracker": install_budget_tracker(None),
    }

    if cache is not None:
        set_cache(cache)

    fresh_executor = None
    if cache is not None:
        from .runtime.executor import Executor
        fresh_executor = Executor(cache=cache)
        set_default_executor(fresh_executor)

    if backend is None:
        backend = MockBackend()
    from .backends.base import set_default_backend as _set_db
    _set_db(backend)

    # Set the requested aggregator (or a fresh one). We restore on exit.
    if aggregator is None:
        aggregator = Aggregator()
    install_default_aggregator(aggregator)

    ctx_out = {
        "executor": fresh_executor or saved["executor"],
        "cache": get_cache(),
        "backend": backend,
        "aggregator": aggregator,
        "budget_tracker": budget_tracker,
    }
    try:
        yield ctx_out
    finally:
        set_default_executor(saved["executor"])
        set_cache(saved["cache"])
        install_default_aggregator(saved["aggregator"])
        install_budget_tracker(saved["budget_tracker"])


# ----------------------------------------------------------------------
# Capture metrics — record every CallMetrics produced in the block.
# ----------------------------------------------------------------------
@contextlib.contextmanager
def capture_metrics() -> Iterator[list[CallMetrics]]:
    """Append every ``CallMetrics`` produced inside the ``with`` block.

    Ideal for model unit-test evals::

        with capture_metrics() as bag:
            for sample in dataset:
                try:
                    classify(sample.text)
                except RetryExhaustedError:
                    pass
        retry_rate = sum(1 for m in bag if m.retries > 0) / len(bag)
        assert retry_rate < 0.1

    Implementation: install a capturing aggregator; the executor records into
    it on every completed call (success *and* failure).
    """
    from .runtime.aggregator import Aggregator, install_default_aggregator
    from .runtime.executor import Executor
    from .core.decorator import set_default_executor, get_default_executor

    collected: list[CallMetrics] = []
    agg = Aggregator(on_record=collected.append)
    prev_agg = install_default_aggregator(agg)
    prev_exec = get_default_executor()
    # Fresh executor so it picks up the new aggregator via the module-level
    # lookup (the executor calls get_default_aggregator() at call time).
    set_default_executor(Executor())

    try:
        yield collected
    finally:
        install_default_aggregator(prev_agg)
        set_default_executor(prev_exec)


def eval_summary(metrics_list: list[CallMetrics]) -> dict[str, Any]:
    """Compute retry / failure eval stats from a list of ``CallMetrics``.

    Returns keys that model-quality unit tests typically assert on:

    - ``call_count``, ``success_count``, ``failure_count``, ``recovered_count``
    - ``total_retries``, ``avg_retries``, ``retry_rate``, ``recovery_rate``
    - ``failure_rate``, ``errors_by_category``, ``retries_histogram``
    """
    from .runtime.aggregator import Aggregator

    agg = Aggregator()
    for i, m in enumerate(metrics_list):
        name = m.function_name or f"call_{i}"
        # Use the short name (last segment) for readable grouping when possible.
        short = name.rsplit(".", 1)[-1] if name else f"call_{i}"
        agg.record(m, function_name=short)
    return agg.summary()


def unmock() -> None:
    """Remove the global mock backend and reset back to ``None``."""
    set_default_backend(None)


__all__ = [
    "mock_llm",
    "mock_llm_table",
    "freeze_time",
    "isolate_execution",
    "capture_metrics",
    "eval_summary",
    "unmock",
]