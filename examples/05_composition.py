"""Example 5 — Composition: an Agentic Function that calls other Agentic Functions.

This is the killer feature: each piece is a typed, testable, composable
function — and you can wire them together however you want.
"""
from __future__ import annotations

from agentic_function import AgenticResult, agentic_function, set_default_backend
from agentic_function.backends.mock_backend import MockBackend
from agentic_function.testing import mock_llm


def main() -> None:
    backend = MockBackend()
    set_default_backend(backend)

    @agentic_function(output_schema={"topic": str, "tone": str})
    def extract_topic(text: str) -> AgenticResult:
        """Extract the main topic and tone of a text."""

    @agentic_function(
        output_schema={"summary": str, "tags": list[str], "score": float}
    )
    def make_summary(text: str, topic: str, tone: str) -> AgenticResult:
        """Summarise ``text`` guided by the extracted ``topic`` and ``tone``."""

    # A small dispatcher: route based on which schema's name is in play.
    def route(req: dict) -> Any:
        schema_name = req["output_schema"].model_class.__name__
        if "extract_topic" in schema_name:
            return {"topic": "OpenAI", "tone": "informational"}
        return {
            "summary": "OpenAI ships GPT-5 with multimodal and 1M-token support.",
            "tags": ["openai", "gpt-5", "ai"],
            "score": 0.9,
        }

    backend.register(route)

    article = "OpenAI today announced GPT-5..."

    # ---- Composition #1: manual orchestration ----
    t = extract_topic(article)
    s = make_summary(article, t.topic, t.tone)
    print("--- composition #1 ---")
    print(f"topic    : {t.topic}")
    print(f"summary  : {s.summary}")
    print(f"tags     : {s.tags}")

    # ---- Composition #2: expose as OpenAI tool for any agent framework ----
    from agentic_function import as_openai_tool
    tool = as_openai_tool(make_summary)
    print("\n--- composition #2 ---")
    print("OpenAI tool schema:")
    import json
    print(json.dumps(tool, indent=2))


if __name__ == "__main__":
    main()