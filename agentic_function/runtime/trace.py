"""Tracing primitives.

The goal is twofold:
1. Provide enough context for debugging a single call (prompt, response, timing).
2. Support nested spans so that when one AgenticFunction calls another, we get a
   proper tree.
"""
from __future__ import annotations

import contextvars
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..utils.logging import get_logger

_log = get_logger("trace")

_current_trace: contextvars.ContextVar["TraceContext | None"] = contextvars.ContextVar(
    "agentic_function_current_trace", default=None
)


@dataclass
class TraceSpan:
    """One unit of work inside the executor (could be a full call, or a retry attempt)."""
    name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: float = 0.0
    status: str = "ok"                          # "ok" | "error" | "cached"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, **attrs: Any) -> None:
        self.events.append({"name": name, "time": time.time(), **attrs})

    def finish(self, status: str = "ok") -> None:
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }


@dataclass
class TraceContext:
    """A tree of spans that belong to one logical operation."""
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = "root"
    spans: list[TraceSpan] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    _span_stack: list[str] = field(default_factory=list)

    def start_span(self, name: str, **attrs: Any) -> TraceSpan:
        parent_id = self._span_stack[-1] if self._span_stack else None
        span = TraceSpan(name=name, parent_id=parent_id, attributes=attrs)
        self.spans.append(span)
        self._span_stack.append(span.span_id)
        return span

    def end_span(self, span: TraceSpan, status: str = "ok") -> None:
        span.finish(status)
        if self._span_stack and self._span_stack[-1] == span.span_id:
            self._span_stack.pop()

    def finish(self) -> None:
        self.finished_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": (self.finished_at - self.started_at) * 1000.0 if self.finished_at else None,
            "spans": [s.to_dict() for s in self.spans],
        }


class TraceRecorder:
    """Collects finished traces. The default recorder is a no-op for tests; users
    can plug in their own (e.g. exporting to OpenTelemetry, Langfuse, ...)."""

    def __init__(self) -> None:
        self._traces: list[TraceContext] = []

    def on_finish(self, trace_context: TraceContext) -> None:
        self._traces.append(trace_context)
        _log.debug("trace finished: %s (%.1fms, %d spans)",
                   trace_context.trace_id,
                   (trace_context.finished_at or 0) - trace_context.started_at,
                   len(trace_context.spans))

    def clear(self) -> None:
        self._traces.clear()

    @property
    def traces(self) -> list[TraceContext]:
        return list(self._traces)


_default_recorder = TraceRecorder()


def get_recorder() -> TraceRecorder:
    return _default_recorder


def set_recorder(recorder: TraceRecorder) -> None:
    global _default_recorder
    _default_recorder = recorder


def get_current_trace() -> TraceContext | None:
    return _current_trace.get()


def trace(name: str = "agentic_function") -> Iterator[TraceContext]:
    """Context manager that opens a new trace for the duration of the block.

    Usage::

        with trace("summarise_article") as ctx:
            result = summarise(long_text)
            print(ctx.trace_id)
    """
    parent = _current_trace.get()
    if parent is not None:
        # Nest under the parent — share trace_id.
        ctx = parent
        span = ctx.start_span(name)
        try:
            _current_trace.set(ctx)
            yield ctx
            ctx.end_span(span, "ok")
        except Exception:
            ctx.end_span(span, "error")
            raise
        return

    ctx = TraceContext(name=name)
    token = _current_trace.set(ctx)
    try:
        yield ctx
    finally:
        ctx.finish()
        _current_trace.reset(token)
        _default_recorder.on_finish(ctx)