"""Best-effort cost estimation per model, in USD per 1M tokens.

The numbers here are intentionally conservative public-list prices. They exist
to give the user a rough order-of-magnitude cost signal, not for billing.

Update freely as providers change prices; nothing in this library breaks if a
model is missing — ``estimate_cost`` will simply return ``None``.
"""
from __future__ import annotations

from typing import TypedDict


class ModelPricing(TypedDict, total=False):
    """USD per 1M tokens for input and output."""
    input: float
    output: float


# Source: public pricing pages of each provider, snapshot 2025-08.
# All values in USD / 1M tokens.
PRICING: dict[str, ModelPricing] = {
    # OpenAI
    "gpt-4o":                  {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":             {"input": 0.15,  "output": 0.60},
    "gpt-4o-2024-08-06":       {"input": 2.50,  "output": 10.00},
    "gpt-4.1":                 {"input": 2.00,  "output": 8.00},
    "gpt-4.1-mini":            {"input": 0.40,  "output": 1.60},
    "gpt-4.1-nano":            {"input": 0.10,  "output": 0.40},
    "o1":                      {"input": 15.00, "output": 60.00},
    "o1-mini":                 {"input": 3.00,  "output": 12.00},
    "o3":                      {"input": 10.00, "output": 40.00},
    "o3-mini":                 {"input": 1.10,  "output": 4.40},
    "o4-mini":                 {"input": 1.10,  "output": 4.40},
    # Anthropic
    "claude-3-5-sonnet-latest":  {"input": 3.00,  "output": 15.00},
    "claude-3-5-haiku-latest":   {"input": 0.80,  "output": 4.00},
    "claude-3-opus-latest":      {"input": 15.00, "output": 75.00},
    "claude-3-7-sonnet-latest":  {"input": 3.00,  "output": 15.00},
    "claude-sonnet-4-5":         {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5":          {"input": 1.00,  "output": 5.00},
    "claude-opus-4-1":           {"input": 15.00, "output": 75.00},
    # Google (rough)
    "gemini-1.5-pro":            {"input": 1.25,  "output": 5.00},
    "gemini-1.5-flash":          {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash":          {"input": 0.10,  "output": 0.40},
    "gemini-2.5-pro":            {"input": 1.25,  "output": 10.00},
    # Local / OSS — usually free; we record zero cost.
    "llama3":                   {"input": 0.0,   "output": 0.0},
    "qwen2.5":                  {"input": 0.0,   "output": 0.0},
    "deepseek-chat":            {"input": 0.14,  "output": 0.28},
    "deepseek-reasoner":        {"input": 0.55,  "output": 2.19},
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Return estimated USD cost for a single call, or ``None`` if unknown."""
    pricing = PRICING.get(model)
    if pricing is None:
        # Try a fuzzy match on the longest common prefix (handles dated variants).
        for prefix in sorted(PRICING, key=len, reverse=True):
            if model.startswith(prefix):
                pricing = PRICING[prefix]
                break
    if pricing is None:
        return None
    in_rate = float(pricing.get("input", 0.0))
    out_rate = float(pricing.get("output", 0.0))
    return (prompt_tokens / 1_000_000.0) * in_rate + (completion_tokens / 1_000_000.0) * out_rate