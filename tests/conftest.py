"""Shared test fixtures.

Most tests use the in-process ``MockBackend`` so they can run offline and
deterministically. A few helpers here keep the boilerplate down.
"""
from __future__ import annotations

import pytest

from agentic_function import MockBackend, register_backend
from agentic_function.runtime.cache import InMemoryCache, set_cache
from agentic_function.runtime.config import configure


@pytest.fixture(autouse=True)
def _isolate_global_state():
    """Reset global mutable state between tests."""
    # Cache
    set_cache(InMemoryCache())
    # Config
    configure(max_retries=0, cache_enabled=False, default_model="mock-model",
              default_backend="mock")
    # Process-wide defaults left by mock_llm_table / capture_metrics / budgets.
    from agentic_function.backends.base import set_default_backend
    from agentic_function.runtime.aggregator import install_default_aggregator
    from agentic_function.runtime.budget import install_budget_tracker
    set_default_backend(None)
    install_default_aggregator(None)
    install_budget_tracker(None)
    # Function registry
    from agentic_function.composition.registry import get_global_registry
    get_global_registry()._store.clear()
    # Backend registry — preserve built-ins but reset user-registered ones.
    from agentic_function.backends.base import _registry
    _registry.pop("test_backend", None)
    yield
    set_default_backend(None)
    install_default_aggregator(None)
    install_budget_tracker(None)


@pytest.fixture
def mock_backend():
    """A fresh MockBackend registered as 'mock'."""
    backend = MockBackend()
    register_backend("mock", backend)
    return backend


@pytest.fixture
def canned(mock_backend):
    """A MockBackend pre-loaded with a default response builder.

    Tests can override ``canned.default_response`` or register more handlers.
    """
    mock_backend.default_response = {
        "label": "neutral",
        "score": 0.5,
        "reason": "mock default",
    }
    return mock_backend