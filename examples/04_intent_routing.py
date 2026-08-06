"""Example 4 — LLM-as-router.

Replace brittle rule engines ("if regex matches then ...") with a typed
Agentic Function that picks from a closed enum.
"""
from __future__ import annotations

from typing import Literal

from agentic_function import AgenticResult, agentic_function, set_default_backend
from agentic_function.backends.mock_backend import MockBackend
from agentic_function.testing import mock_llm


def main() -> None:
    set_default_backend(MockBackend())
    mock_llm(
        output={"intent": "billing", "confidence": 0.92,
                "followup_question": "Can you share the invoice number?"}
    )

    # The Literal[...] type makes the schema strict — only the four values
    # below are accepted; anything else triggers a schema-validation retry.
    Intent = Literal["billing", "technical", "account", "other"]

    @agentic_function(
        output_schema={
            "intent": Intent,
            "confidence": float,
            "followup_question": str,
        },
        temperature=0.0,
        max_retries=2,
    )
    def classify_intent(user_message: str) -> AgenticResult:
        """Classify the user's intent into one of:
        - "billing": payment / invoice / refund questions
        - "technical": bugs, errors, how-to
        - "account": login, profile, settings
        - "other": anything else

        Also propose a short follow-up question to clarify the request."""

    message = "Hi, I was charged twice for my March invoice — what do I do?"
    result = classify_intent(message)
    print(f"intent             : {result.intent}")
    print(f"confidence         : {result.confidence}")
    print(f"followup_question  : {result.followup_question}")

    # Because intent is a Literal, downstream code is exhaustive.
    match result.intent:
        case "billing":
            department = "Billing Team"
        case "technical":
            department = "Tech Support"
        case "account":
            department = "Account Services"
        case "other":
            department = "General"
    print(f"routed to          : {department}")


if __name__ == "__main__":
    main()