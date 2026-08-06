"""Example 3 — Summarisation with quality scoring.

Demonstrates:
* multiple output fields of different types
* few-shot exemplars to steer style
* cost + latency reporting from the result
"""
from __future__ import annotations

from agentic_function import AgenticResult, agentic_function, set_default_backend
from agentic_function.backends.mock_backend import MockBackend
from agentic_function.testing import mock_llm


def main() -> None:
    set_default_backend(MockBackend())
    mock_llm(
        output={
            "summary": "OpenAI releases GPT-5 with multimodal support and a 1M token context.",
            "tags": ["AI", "LLM", "OpenAI"],
            "quality": 0.88,
        }
    )

    @agentic_function(
        output_schema={
            "summary": str,        # <= 60 words, neutral tone
            "tags": list[str],     # 3-5 lowercase keywords
            "quality": float,      # 0.0 .. 1.0 quality estimate
        },
        temperature=0.2,
        few_shots=[
            (
                {"article": "Apple announced new MacBooks with M4 chips..."},
                {
                    "summary": "Apple launches new MacBooks powered by M4 chips.",
                    "tags": ["apple", "macbook", "silicon"],
                    "quality": 0.82,
                },
            ),
        ],
    )
    def summarise(article: str) -> AgenticResult:
        """Summarise ``article`` into a single sentence and assign tags + quality."""

    article = (
        "OpenAI today announced GPT-5, the next generation of its flagship "
        "language model. GPT-5 is multimodal from the ground up, supports a "
        "1M-token context window, and is rolling out to ChatGPT Pro users "
        "starting today. Independent benchmarks show a 12-point jump on MMLU "
        "and a 30% reduction in hallucination rate compared to GPT-4o."
    )
    result = summarise(article)
    print(f"summary : {result.summary}")
    print(f"tags    : {result.tags}")
    print(f"quality : {result.quality}")
    print(f"cost    : ${result.metrics.cost_usd:.4f}")
    print(f"latency : {result.metrics.latency_ms} ms")


if __name__ == "__main__":
    main()