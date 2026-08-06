"""Unit tests for ``agentic_function.utils``."""
from agentic_function.utils.hashing import stable_hash, hash_inputs
from agentic_function.utils.cost import estimate_cost, PRICING


def test_stable_hash_deterministic():
    h1 = stable_hash("a", 1, {"x": 1})
    h2 = stable_hash("a", 1, {"x": 1})
    assert h1 == h2


def test_stable_hash_key_order_independent():
    h1 = stable_hash({"a": 1, "b": 2})
    h2 = stable_hash({"b": 2, "a": 1})
    assert h1 == h2


def test_stable_hash_nested():
    h1 = stable_hash({"outer": {"inner": [1, 2, 3]}})
    h2 = stable_hash({"outer": {"inner": [1, 2, 3]}})
    assert h1 == h2


def test_hash_inputs():
    h1 = hash_inputs("hello", x=1, y=2)
    h2 = hash_inputs("hello", y=2, x=1)  # kwargs order
    assert h1 == h2


def test_estimate_cost_known_model():
    cost = estimate_cost("gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)
    # 0.15/1M * 1000 + 0.60/1M * 500
    assert abs(cost - 0.00045) < 1e-9


def test_estimate_cost_prefix_match():
    cost = estimate_cost("gpt-4o-2024-08-06", prompt_tokens=1_000_000, completion_tokens=0)
    assert cost is not None
    assert cost == 2.5


def test_estimate_cost_unknown_returns_none():
    assert estimate_cost("unknown-model-xyz", 100, 100) is None


def test_pricing_has_major_models():
    assert "gpt-4o-mini" in PRICING
    assert "claude-3-5-sonnet-latest" in PRICING
    assert "gemini-1.5-flash" in PRICING