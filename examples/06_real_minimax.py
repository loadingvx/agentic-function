"""Example 6 — Real LLM call against MiniMax / MiniMax-China (Anthropic-compatible).

This example wires up the ``minimax`` backend (which speaks the Anthropic
Messages protocol against ``https://api.minimaxi.com/anthropic``) and runs
two agentic functions end-to-end with the live LLM.

Required environment variables
------------------------------
``MINIMAX_CN_API_KEY``
    The MiniMax / MiniMax-CN API key. The framework reads it automatically
    when you select ``backend="minimax"`` and don't pass an explicit key.

Optional environment variables
------------------------------
``MINIMAX_BASE_URL``
    Override the default Anthropic-compatible endpoint. Defaults to
    ``https://api.minimaxi.com/anthropic``.

``MINIMAX_MODEL``
    Override the default model name. Defaults to ``MiniMax-M3``.

Run with::

    set -a && source ~/.hermes/.env && set +a
    python examples/06_real_minimax.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Allow running the example directly without `pip install -e .`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentic_function import (  # noqa: E402
    AgenticResult,
    agentic_function,
    get_backend,
    set_default_backend,
)


def _check_env() -> None:
    if not os.environ.get("MINIMAX_CN_API_KEY") and not os.environ.get(
        "MINIMAX_API_KEY"
    ):
        print(
            "ERROR: MINIMAX_CN_API_KEY (or MINIMAX_API_KEY) is not set.\n"
            "       Export it before running, e.g.\n"
            "       set -a && source ~/.hermes/.env && set +a\n"
            "       python examples/06_real_minimax.py",
            file=sys.stderr,
        )
        sys.exit(2)


def _print_result(label: str, result, elapsed_ms: float) -> None:
    print(f"\n── {label} ──")
    print(f"latency       : {elapsed_ms:.1f} ms")
    metrics = result.metrics
    print(f"tokens        : {metrics.usage.prompt_tokens} in / "
          f"{metrics.usage.completion_tokens} out "
          f"(total {metrics.usage.total_tokens})")
    print(f"cache_hit     : {metrics.cache_hit}")
    print(f"attempts      : {metrics.attempts}")
    print(f"cost_usd      : {metrics.cost_usd}")
    print(f"result        : {json.dumps(result.model_dump(), ensure_ascii=False, indent=2)}")


# ---------------------------------------------------------------------------
# 1. Sentiment classification — single-turn, simple JSON
# ---------------------------------------------------------------------------
@agentic_function(
    backend="minimax",
    output_schema={
        "label": str,           # "positive" | "negative" | "neutral"
        "confidence": float,    # 0.0 .. 1.0
        "reasoning": str,       # <= 100 chars
    },
    temperature=0.0,
    max_tokens=512,
)
def classify_sentiment(text: str) -> AgenticResult:
    """Classify the sentiment of ``text``.

    Output JSON schema:
      - label:      one of "positive" | "negative" | "neutral"
      - confidence: float in [0.0, 1.0]
      - reasoning:  short justification (<= 100 chars)
    """


# ---------------------------------------------------------------------------
# 2. Information extraction — multi-field, slightly trickier
# ---------------------------------------------------------------------------
@agentic_function(
    backend="minimax",
    output_schema={
        "company": str,
        "ticker": str,
        "amount_usd_millions": float,
        "is_acquisition": bool,
    },
    temperature=0.0,
    max_tokens=512,
)
def extract_deal(text: str) -> AgenticResult:
    """Extract structured fields from a short news blurb about a corporate deal.

    Output JSON schema:
      - company:            the company being discussed
      - ticker:             its stock ticker (empty string if not public)
      - amount_usd_millions: deal size in millions of USD (0.0 if not stated)
      - is_acquisition:     true if it's an acquisition, false if it's
                            investment/IPO/other
    """


# ---------------------------------------------------------------------------
# 3. Async — to prove the async path works against the same backend
# ---------------------------------------------------------------------------
@agentic_function(
    backend="minimax",
    output_schema={
        "language": str,    # BCP-47 code, e.g. "en", "zh", "ja"
        "summary": str,     # <= 60 chars
    },
    temperature=0.0,
    max_tokens=256,
)
async def detect_language(text: str) -> AgenticResult:
    """Detect the language of ``text`` and write a 1-line summary.

    Output JSON schema:
      - language: BCP-47 code (e.g. "en", "zh", "ja", "fr", ...)
      - summary:  short summary of the text (<= 60 chars)
    """


def main() -> None:
    _check_env()

    # Make sure the "minimax" backend is wired up. Importing backends registers
    # the preset; we just confirm it's there.
    backend = get_backend("minimax")
    set_default_backend(backend)
    print(f"minimax backend ready: base_url={backend.base_url}")
    print(f"  default_model={backend.default_model}")
    print(f"  api_key={'set' if backend.api_key else 'MISSING'}")

    # ---- 1. sentiment ----
    t0 = time.perf_counter()
    result = classify_sentiment(
        "The new agentic-function library makes LLM calls feel like normal Python!"
    )
    elapsed = (time.perf_counter() - t0) * 1000
    _print_result("sentiment classification", result, elapsed)

    assert result.label in {"positive", "negative", "neutral"}, result.label
    assert 0.0 <= result.confidence <= 1.0

    # ---- 2. extraction ----
    t0 = time.perf_counter()
    deal = extract_deal(
        "Anthropic is acquiring the small GPU-efficiency startup Xilicon for "
        "$420M in cash, according to people familiar with the matter."
    )
    elapsed = (time.perf_counter() - t0) * 1000
    _print_result("deal extraction", deal, elapsed)

    assert isinstance(deal.amount_usd_millions, float)
    assert isinstance(deal.is_acquisition, bool)

    # ---- 3. async ----
    t0 = time.perf_counter()
    lang = asyncio.run(
        detect_language.acall(
            "Anthropic 是一家总部位于旧金山的人工智能安全公司，其使命是构建可靠、可解释、可控的人工智能系统。"
        )
    )
    elapsed = (time.perf_counter() - t0) * 1000
    _print_result("async language detection", lang, elapsed)

    print("\n✅ All three real LLM calls succeeded against the minimax backend.")


if __name__ == "__main__":
    main()