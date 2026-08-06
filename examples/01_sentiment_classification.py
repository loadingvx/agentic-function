"""Example 1 — Sentiment classification with full type safety.

Run with:  python examples/01_sentiment_classification.py

This example uses the in-process MockBackend so it requires no API keys.
The mock is wired by importing ``agentic_function.testing``.
"""
from __future__ import annotations

from agentic_function import (
    AgenticResult,
    agentic_function,
    get_default_backend,
    set_default_backend,
)
from agentic_function.backends.mock_backend import MockBackend
from agentic_function.testing import mock_llm


def main() -> None:
    # ---- 1. Install a mock backend that returns the JSON we expect. ----
    set_default_backend(MockBackend())
    # We can also register specific per-call fixtures.
    mock_llm(
        output={
            "label": "positive",
            "confidence": 0.94,
            "reasoning": "Strong positive adjectives, exclamation mark, no negation.",
        }
    )

    # ---- 2. Declare the function. The docstring is the prompt. ----
    @agentic_function(
        output_schema={
            "label": str,           # "positive" | "negative" | "neutral"
            "confidence": float,    # 0.0 .. 1.0
            "reasoning": str,       # <= 80 chars
        },
        temperature=0.0,
    )
    def classify_sentiment(text: str) -> AgenticResult:
        """Classify the sentiment of ``text``.

        Output:
        - label: one of "positive" | "negative" | "neutral"
        - confidence: float between 0.0 and 1.0
        - reasoning: short justification (<= 80 chars)
        """

    # ---- 3. Call it like a normal function. ----
    result = classify_sentiment("The launch today was absolutely amazing!")
    # ``result`` is a dynamic pydantic model instance with attributes matching
    # the declared schema. It is NOT an ``AgenticResult`` when a schema was
    # provided — it IS the schema. (AgenticResult is only used for the
    # schema-less / untyped case.)
    assert result.label == "positive"
    assert 0.0 <= result.confidence <= 1.0
    print(f"label       : {result.label}")
    print(f"confidence  : {result.confidence}")
    print(f"reasoning   : {result.reasoning}")

    # ---- 4. Async path works too. ----
    import asyncio
    aresult = asyncio.run(classify_sentiment.acall("meh."))
    print(f"\n[async] label : {aresult.label}")

    # ---- 5. Inspect execution metadata. ----
    print(f"\nlatency_ms  : {result.metrics.latency_ms}")
    print(f"tokens      : {result.metrics.usage.prompt_tokens} in / "
          f"{result.metrics.usage.completion_tokens} out")


if __name__ == "__main__":
    main()