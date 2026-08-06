"""Post-mortem diagnostics for a single agentic-function call.

When something fails (or is just slow), these helpers turn a ``CallMetrics``
object into a human-readable summary that tells you:

* what happened (success, cache hit, retry, validation error)
* where time was spent (per-phase breakdown of ``latency_ms``)
* what the LLM saw (the request messages, truncated)
* what the LLM said (the raw response, truncated)
* what we tried to coerce it into (the schema hash + field names)

This is the framework's **debug ergonomics surface** — used by:

* ad-hoc REPL sessions: ``diagnose(result)`` returns a printable string
* unit tests: ``explain_failure(exc)`` returns structured info for assertions
* CLI tools: ``snapshot(result)`` returns a JSON-safe dict for log shipping
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..errors import AgenticFunctionError
from .metrics import CallMetrics


@dataclass(slots=True)
class Diagnostic:
    """Structured diagnostic for one call. JSON-serialisable."""
    function: str
    status: str                       # "success" | "cache_hit" | "failed"
    backend: str
    model: str
    attempts: int
    retries: int
    latency_ms: float
    timings_ms: dict[str, float]
    tokens: dict[str, int]
    cost_usd: float | None
    total_cost_usd: float | None
    finish_reason: str | None
    error: str | None
    request_preview: list[dict[str, Any]] | None
    response_preview: Any
    cache_hit: bool
    timestamp: float

    def to_text(self) -> str:
        """A multi-line, human-friendly summary suitable for log output."""
        lines: list[str] = []
        lines.append(f"─ agentic_function: {self.function} [{self.status}]")
        lines.append(f"  backend        : {self.backend}")
        lines.append(f"  model          : {self.model}")
        lines.append(f"  attempts       : {self.attempts}  (retries: {self.retries})")
        lines.append(f"  latency        : {self.latency_ms:.1f} ms")
        timings = self.timings_ms
        if timings:
            lines.append(
                "  timings        : "
                + " | ".join(
                    f"{k}={v:.1f}ms"
                    for k, v in timings.items()
                    if k != "total_ms" and v > 0
                )
            )
            lines.append(
                f"  phase total    : {timings.get('total_ms', 0):.1f}ms "
                f"({self.latency_ms - timings.get('total_ms', 0):.1f}ms overhead)"
            )
        tok = self.tokens
        lines.append(
            f"  tokens         : {tok['prompt_tokens']} in / "
            f"{tok['completion_tokens']} out / {tok['total_tokens']} total"
        )
        if self.cost_usd is not None:
            t = f"${self.total_cost_usd:.6f}" if self.total_cost_usd else ""
            lines.append(f"  cost           : ${self.cost_usd:.6f} (this attempt)  {t} (cumulative)")
        if self.finish_reason:
            lines.append(f"  finish_reason  : {self.finish_reason}")
        if self.cache_hit:
            lines.append("  cache          : HIT")
        if self.error:
            lines.append(f"  error          : {self.error}")
        if self.request_preview:
            lines.append(f"  request        : {len(self.request_preview)} message(s)")
            for i, m in enumerate(self.request_preview[:3]):
                role = m.get("role", "?")
                content_preview = str(m.get("content", ""))[:80].replace("\n", " ")
                lines.append(f"    [{i}] {role}: {content_preview}…")
        if self.response_preview is not None:
            r = self.response_preview
            if isinstance(r, str):
                preview = r[:120].replace("\n", " ")
            elif isinstance(r, dict):
                preview = json.dumps(r, ensure_ascii=False)[:120]
            else:
                preview = repr(r)[:120]
            lines.append(f"  response       : {preview}…")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "status": self.status,
            "backend": self.backend,
            "model": self.model,
            "attempts": self.attempts,
            "retries": self.retries,
            "latency_ms": self.latency_ms,
            "timings_ms": self.timings_ms,
            "tokens": self.tokens,
            "cost_usd": self.cost_usd,
            "total_cost_usd": self.total_cost_usd,
            "finish_reason": self.finish_reason,
            "error": self.error,
            "request_preview": self.request_preview,
            "response_preview": self.response_preview,
            "cache_hit": self.cache_hit,
            "timestamp": self.timestamp,
        }


def diagnose(result: Any) -> Diagnostic:
    """Build a Diagnostic from an AgenticResult (success path).

    Use ``diagnose(result.metrics)`` if you only have the metrics object.
    """
    metrics = getattr(result, "_metrics", None) or getattr(result, "metrics", None)
    if metrics is None:
        raise TypeError(
            "diagnose() expects an AgenticResult or CallMetrics; got "
            f"{type(result).__name__}"
        )
    # Find the function name. The Executor stashes it on the metrics via the
    # request, but we don't keep that handle — so we look at a few likely
    # spots, then fall back to "<agentic_function>".
    function_name = (
        getattr(result, "_function_name", None)
        or getattr(metrics, "function_name", None)
        or "<agentic_function>"
    )
    return _from_metrics(metrics, function_name=function_name)


def diagnose_metrics(metrics: CallMetrics, *, function_name: str = "<agentic_function>") -> Diagnostic:
    """Build a Diagnostic directly from CallMetrics (no result needed)."""
    return _from_metrics(metrics, function_name=function_name)


def _from_metrics(m: CallMetrics, *, function_name: str) -> Diagnostic:
    if m.cache_hit:
        status = "cache_hit"
    elif m.successful:
        status = "success"
    else:
        status = "failed"
    return Diagnostic(
        function=function_name,
        status=status,
        backend=m.backend or "?",
        model=m.model or "?",
        attempts=m.attempts,
        retries=m.retries,
        latency_ms=m.latency_ms,
        timings_ms=m.timings.as_dict(),
        tokens={
            "prompt_tokens": m.usage.prompt_tokens,
            "completion_tokens": m.usage.completion_tokens,
            "total_tokens": m.usage.total_tokens,
        },
        cost_usd=m.cost_usd,
        total_cost_usd=m.total_cost_usd,
        finish_reason=m.finish_reason,
        error=m.error,
        request_preview=m.request_snapshot,
        response_preview=m.response_snapshot,
        cache_hit=m.cache_hit,
        timestamp=0.0,
    )


# ----------------------------------------------------------------------
# Failure explainer
# ----------------------------------------------------------------------
def explain_failure(exc: BaseException) -> dict[str, Any]:
    """Turn an exception raised by the executor into a structured explanation.

    Useful in tests::

        def test_retries_on_validation():
            with pytest.raises(RetryExhaustedError) as ei:
                classify("text")
            info = explain_failure(ei.value)
            assert info["category"] == "validation"
            assert info["attempts"] >= 1
    """
    from ..errors import (
        BackendError,
        ParseError,
        RetryExhaustedError,
        SchemaError,
        ValidationError,
    )

    if isinstance(exc, RetryExhaustedError):
        last = exc.last_exception
        return {
            "category": "retry_exhausted",
            "attempts": exc.attempts,
            "last_exception_type": type(last).__name__,
            "last_exception_message": str(last),
            "last_exception": _explain(last),
        }
    if isinstance(exc, ValidationError):
        return {
            "category": "validation",
            "errors": exc.errors,
            "raw_output_preview": _truncate(exc.raw_output),
        }
    if isinstance(exc, ParseError):
        return {
            "category": "parse",
            "raw_output_preview": _truncate(exc.raw_output),
        }
    if isinstance(exc, BackendError):
        return {
            "category": "backend",
            "backend": exc.backend,
            "status_code": exc.status_code,
            "message": str(exc),
        }
    if isinstance(exc, SchemaError):
        return {"category": "schema", "message": str(exc)}
    if isinstance(exc, AgenticFunctionError):
        return {"category": "agentic", "type": type(exc).__name__, "message": str(exc)}
    return {"category": "unknown", "type": type(exc).__name__, "message": str(exc)}


def _explain(exc: BaseException) -> dict[str, Any]:
    """Recursively explain a wrapped exception."""
    return explain_failure(exc)


def _truncate(value: Any, limit: int = 500) -> Any:
    if isinstance(value, str):
        return value[:limit] + ("…" if len(value) > limit else "")
    if isinstance(value, dict):
        return {k: _truncate(v, limit) for k, v in value.items()}
    return value


# ----------------------------------------------------------------------
# Snapshot — JSON-safe dict suitable for log shipping / replay tools
# ----------------------------------------------------------------------
def snapshot(result_or_metrics: Any) -> dict[str, Any]:
    """Return a JSON-safe dict describing one call.

    Includes everything ``diagnose()`` returns, plus the raw request/response
    snapshots if debug mode was on.
    """
    metrics = getattr(result_or_metrics, "_metrics", None) or getattr(result_or_metrics, "metrics", None)
    if metrics is None:
        raise TypeError(
            "snapshot() expects an AgenticResult or CallMetrics; got "
            f"{type(result_or_metrics).__name__}"
        )
    diag = _from_metrics(metrics, function_name="<agentic_function>")
    out = diag.to_dict()
    if metrics.request_snapshot is not None:
        out["request"] = metrics.request_snapshot
    if metrics.response_snapshot is not None:
        out["response"] = metrics.response_snapshot
    return out


__all__ = [
    "Diagnostic",
    "diagnose",
    "diagnose_metrics",
    "explain_failure",
    "snapshot",
]