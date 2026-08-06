"""Example 2 — Structured information extraction.

Use an Agentic Function to extract typed fields out of free-form text
(resumes, support tickets, articles, ...).

Run:  python examples/02_information_extraction.py
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from agentic_function import AgenticResult, agentic_function, set_default_backend
from agentic_function.backends.mock_backend import MockBackend
from agentic_function.testing import mock_llm


class ResumeInfo(BaseModel):
    name: str
    years_experience: int = Field(ge=0)
    skills: list[str]
    most_recent_role: str


def main() -> None:
    # Mock backend with a realistic extraction result.
    set_default_backend(MockBackend())
    mock_llm(
        output={
            "name": "Alice Chen",
            "years_experience": 7,
            "skills": ["Python", "PyTorch", "Kubernetes", "Postgres"],
            "most_recent_role": "Senior ML Engineer at Acme Corp",
        }
    )

    # Schema is a Pydantic class — you get strict type checks and IDE help.
    @agentic_function(
        output_schema=ResumeInfo,
        temperature=0.0,
        max_retries=3,           # auto-retry on schema mismatch / parse errors
    )
    def parse_resume(text: str) -> ResumeInfo:
        """Extract structured fields from a resume text block.

        Output fields:
        - name: full name
        - years_experience: integer count of total professional years
        - skills: list of technology / skill keywords
        - most_recent_role: title and company of the most recent position
        """

    resume = """
    Alice Chen — Senior ML Engineer at Acme Corp (2022–present)
    Previously: ML Engineer at Foo (2018–2022). 7 years of experience
    building production ML systems. Skills: Python, PyTorch, Kubernetes,
    Postgres, gRPC.
    """
    info = parse_resume(resume)
    # ``info`` is a ``ResumeInfo`` instance (not just a dict).
    assert isinstance(info, ResumeInfo)
    print(f"name              : {info.name}")
    print(f"years_experience  : {info.years_experience}")
    print(f"skills            : {info.skills}")
    print(f"most_recent_role  : {info.most_recent_role}")
    # Pydantic validation guarantees ``years_experience >= 0``.


if __name__ == "__main__":
    main()