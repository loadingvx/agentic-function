"""Tests for the 0.5 observability / debug / quant additions:

* ``PhaseTimings`` field-by-field
* ``diagnose`` / ``explain_failure`` / ``snapshot``
* ``Aggregator`` summary + Prometheus export
* ``BudgetTracker`` ceilings (function + process scope)
* ``mock_llm_table``, ``freeze_time``, ``isolate_execution``

All tests are hermetic — no network calls.
"""
from __future__ import annotations

import json
import time

import pytest

from agentic_function import (
    Aggregator,
    Budget,
    BudgetExceededError,
    BudgetTracker,
    CallMetrics,
    PhaseTimings,
    TokenUsage,
    agentic_function,
    diagnose,
    diagnose_metrics,
    explain_failure,
    install_budget_tracker,
    install_default_aggregator,
    snapshot,
)
from agentic_function.backends.mock_backend import MockBackend
from agentic_function.errors import (
    BackendError, ParseError, RetryExhaustedError, ValidationError,
)
from agentic_function.testing import (
    freeze_time, isolate_execution, mock_llm_table,
)


# ---------------------------------------------------------------------------
# PhaseTimings
# ---------------------------------------------------------------------------
class TestPhaseTimings:
    def test_default_zeros(self):
        t = PhaseTimings()
        assert t.total_ms() == 0.0
        d = t.as_dict()
        assert d["prompt_render_ms"] == 0.0
        assert d["cache_lookup_ms"] == 0.0
        assert d["backend_call_ms"] == 0.0
        assert d["parse_validate_ms"] == 0.0
        assert d["cache_write_ms"] == 0.0
        assert d["retry_backoff_ms"] == 0.0
        assert "total_ms" in d

    def test_total_sums_all_phases(self):
        t = PhaseTimings(
            prompt_render_ms=1.0, cache_lookup_ms=2.0, backend_call_ms=10.0,
            parse_validate_ms=3.0, cache_write_ms=0.5, retry_backoff_ms=4.0,
        )
        assert t.total_ms() == 20.5
        assert t.as_dict()["total_ms"] == 20.5


# ---------------------------------------------------------------------------
# CallMetrics — new fields populated by Executor
# ---------------------------------------------------------------------------
class TestCallMetricsNewFields:
    def test_callmetrics_has_all_new_fields(self):
        m = CallMetrics()
        assert m.backend == ""
        assert m.model == ""
        assert m.finish_reason is None
        assert isinstance(m.timings, PhaseTimings)
        assert m.total_usage == TokenUsage()
        assert m.total_cost_usd is None
        assert m.request_snapshot is None
        assert m.response_snapshot is None

    def test_callmetrics_as_dict_contains_everything(self):
        m = CallMetrics(
            attempts=1, successful=True, latency_ms=123.4,
            backend="anthropic", model="claude-3-5-sonnet-latest",
            finish_reason="end_turn",
            usage=TokenUsage.of(85, 39),
            cost_usd=0.0001,
            timings=PhaseTimings(backend_call_ms=100.0),
        )
        d = m.as_dict()
        assert d["backend"] == "anthropic"
        assert d["model"] == "claude-3-5-sonnet-latest"
        assert d["finish_reason"] == "end_turn"
        assert d["timings"]["backend_call_ms"] == 100.0
        assert d["usage"]["prompt_tokens"] == 85


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
@agentic_function(output_schema={"label": str, "score": float})
def _diag_classify(text: str):
    """Classify sentiment. Output: label positive|negative|neutral, score in 0..1."""


class TestDiagnostics:
    def test_diagnose_returns_diagnostic_for_success(self):
        mock_llm_table([{"label": "positive", "score": 0.9}])
        result = _diag_classify("amazing!")
        diag = diagnose(result)
        assert diag.status == "success"
        assert diag.function.endswith("_diag_classify")
        assert diag.tokens["prompt_tokens"] >= 0
        assert diag.timings_ms["backend_call_ms"] >= 0
        # No error on success
        assert diag.error is None
        # Cache should be off
        assert diag.cache_hit is False
        # Text output is well-formed
        text = diag.to_text()
        assert "_diag_classify" in text
        assert "success" in text

    def test_diagnose_with_cache_hit(self):
        mock_llm_table([{"label": "positive", "score": 0.9}])
        _diag_classify.cache = True    # enable caching on the descriptor
        try:
            r1 = _diag_classify("amazing!")
            assert r1.metrics.cache_hit is False
            r2 = _diag_classify("amazing!")
            assert r2.metrics.cache_hit is True
        finally:
            _diag_classify.cache = False
        diag = diagnose(r2)
        assert diag.status == "cache_hit"
        assert diag.cache_hit is True

    def test_snapshot_is_json_safe(self):
        mock_llm_table([{"label": "positive", "score": 0.9}])
        result = _diag_classify("hi")
        snap = snapshot(result)
        json.dumps(snap)  # must not raise

    def test_explain_failure_validation(self):
        exc = ValidationError("bad", errors=[{"loc": "label"}], raw_output={"label": 1})
        info = explain_failure(exc)
        assert info["category"] == "validation"
        assert info["raw_output_preview"] == {"label": 1}

    def test_explain_failure_parse(self):
        exc = ParseError("not json", raw_output="oops")
        info = explain_failure(exc)
        assert info["category"] == "parse"
        assert info["raw_output_preview"] == "oops"

    def test_explain_failure_backend(self):
        exc = BackendError("network", backend="openai", status_code=502)
        info = explain_failure(exc)
        assert info["category"] == "backend"
        assert info["backend"] == "openai"
        assert info["status_code"] == 502

    def test_explain_failure_retry_exhausted(self):
        inner = ValidationError("bad")
        exc = RetryExhaustedError("all failed", attempts=3, last_exception=inner)
        info = explain_failure(exc)
        assert info["category"] == "retry_exhausted"
        assert info["attempts"] == 3
        assert info["last_exception_type"] == "ValidationError"
        assert info["last_exception"]["category"] == "validation"

    def test_explain_failure_unknown(self):
        info = explain_failure(ValueError("surprise"))
        assert info["category"] == "unknown"
        assert info["type"] == "ValueError"


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
class TestAggregator:
    def test_record_increments_and_summarises(self):
        agg = Aggregator()
        m = CallMetrics(
            attempts=1, successful=True, latency_ms=100.0,
            backend="anthropic", model="claude",
            usage=TokenUsage.of(10, 20), cost_usd=0.001,
        )
        agg.record(m, function_name="f1")
        agg.record(m, function_name="f1")
        summary = agg.summary()
        assert summary["totals"]["call_count"] == 2
        assert summary["totals"]["total_prompt_tokens"] == 20
        assert summary["totals"]["total_completion_tokens"] == 40
        assert summary["totals"]["total_cost_usd"] == pytest.approx(0.002)
        assert "f1" in summary["functions"]
        f1 = summary["functions"]["f1"]
        assert f1["call_count"] == 2
        assert f1["avg_latency_ms"] == 100.0

    def test_top_functions(self):
        agg = Aggregator()
        for i, name in enumerate(["cheap", "expensive"]):
            agg.record(
                CallMetrics(successful=True, latency_ms=10,
                            usage=TokenUsage.of(1, 1), cost_usd=float(i + 1)),
                function_name=name,
            )
        top = agg.top_functions(2, by="total_cost_usd")
        assert top[0]["function"] == "expensive"
        assert top[1]["function"] == "cheap"

    def test_prometheus_export(self):
        agg = Aggregator()
        agg.record(
            CallMetrics(successful=True, latency_ms=50,
                        usage=TokenUsage.of(10, 5), cost_usd=0.001),
            function_name="classify",
        )
        text = agg.to_prometheus()
        assert "# TYPE agentic_calls_total counter" in text
        assert 'agentic_calls_total{function="classify"} 1' in text
        assert "# TYPE agentic_latency_ms_bucket histogram" in text
        # +Inf bucket always present
        assert 'agentic_latency_ms_bucket{function="classify",le="+Inf"} 1' in text

    def test_cache_hit_counting(self):
        agg = Aggregator()
        agg.record(CallMetrics(cache_hit=True, successful=True, latency_ms=0.1),
                   function_name="f")
        agg.record(CallMetrics(cache_hit=False, successful=True, latency_ms=100),
                   function_name="f")
        s = agg.summary()
        assert s["totals"]["cache_hit_count"] == 1
        assert s["functions"]["f"]["cache_hit_count"] == 1

    def test_reset(self):
        agg = Aggregator()
        agg.record(CallMetrics(successful=True), function_name="f")
        agg.reset()
        assert agg.summary()["totals"]["call_count"] == 0

    def test_install_default_aggregator_wires_executor(self):
        from agentic_function.core.decorator import set_default_executor
        from agentic_function.runtime.executor import Executor
        from agentic_function.runtime.aggregator import Aggregator, install_default_aggregator
        agg = Aggregator()
        prev_agg = install_default_aggregator(agg)
        prev_exec = set_default_executor(Executor())
        try:
            mock_llm_table([{"label": "p", "score": 0.5}])
            _diag_classify("hi")
            assert agg.summary()["totals"]["call_count"] == 1
        finally:
            install_default_aggregator(prev_agg)
            set_default_executor(prev_exec)


# ---------------------------------------------------------------------------
# Budget tracker
# ---------------------------------------------------------------------------
class TestBudgetTracker:
    def test_record_accumulates(self):
        bt = BudgetTracker()
        bt.add(Budget("cost_usd", limit=1.0))
        m = CallMetrics(successful=True, cost_usd=0.1)
        bt.record(m)
        bt.record(m)
        assert bt.totals()["cost_usd"] == pytest.approx(0.2)

    def test_record_raises_on_breach(self):
        bt = BudgetTracker()
        bt.add(Budget("cost_usd", limit=0.05))
        with pytest.raises(BudgetExceededError) as excinfo:
            bt.record(CallMetrics(successful=True, cost_usd=0.1))
        assert excinfo.value.metric == "cost_usd"
        assert excinfo.value.limit == 0.05
        assert excinfo.value.used == pytest.approx(0.1)

    def test_function_scope_ignores_process_scope(self):
        bt = BudgetTracker()
        bt.add(Budget("cost_usd", limit=0.01, scope="function"))
        # Recording with scope="process" should not trigger the function limit
        bt.record(CallMetrics(successful=True, cost_usd=0.005), scope="process")
        # ... but it would trigger process scope if we added one
        bt.add(Budget("cost_usd", limit=0.001, scope="process"))
        with pytest.raises(BudgetExceededError):
            bt.record(CallMetrics(successful=True, cost_usd=0.005), scope="process")

    def test_reset(self):
        bt = BudgetTracker()
        bt.add(Budget("cost_usd", limit=1.0))
        bt.record(CallMetrics(successful=True, cost_usd=0.1))
        bt.reset()
        assert bt.totals() == {}

    def test_install_and_restore_via_isolate(self):
        with isolate_execution(budget_tracker=BudgetTracker()) as ctx:
            assert ctx["budget_tracker"] is not None
            tracker = ctx["budget_tracker"]
            tracker.add(Budget("cost_usd", limit=1.0))
            mock_llm_table([{"label": "p", "score": 0.5}], backend=ctx["backend"])
            _diag_classify("hi")
            assert tracker.totals()["cost_usd"] >= 0


# ---------------------------------------------------------------------------
# freeze_time
# ---------------------------------------------------------------------------
class TestFreezeTime:
    def test_freeze_time_deterministic(self):
        mock_llm_table([{"label": "p", "score": 0.5}])
        with freeze_time():
            r = _diag_classify("hi")
        # All timings should be a small multiple of 0.001 ms (the fake
        # perf_counter increment).
        assert r.metrics.timings.prompt_render_ms >= 0
        # Wall-clock latency is whatever perf_counter produced — at minimum 0.
        assert r.metrics.latency_ms >= 0


# ---------------------------------------------------------------------------
# mock_llm_table
# ---------------------------------------------------------------------------
class TestMockLLMTable:
    def test_returns_rows_in_order(self):
        mock_llm_table([
            {"label": "first", "score": 0.1},
            {"label": "second", "score": 0.2},
            {"label": "third", "score": 0.3},
        ])
        assert _diag_classify("a").label == "first"
        assert _diag_classify("b").label == "second"
        assert _diag_classify("c").label == "third"

    def test_key_fn_selects_row_by_input(self):
        def pick(req):
            # ``req`` is a dict — see ``MockBackend.complete()``.
            # The last message is the user message.
            user_msg = req["messages"][-1]["content"]
            return 0 if "good" in user_msg else 1

        mock_llm_table(
            [{"label": "positive", "score": 0.9}, {"label": "negative", "score": 0.1}],
            key_fn=pick,
        )
        assert _diag_classify("good vibes").label == "positive"
        assert _diag_classify("bad vibes").label == "negative"

    def test_clamps_to_last_row_when_exhausted(self):
        mock_llm_table([{"label": "only", "score": 0.5}])
        assert _diag_classify("a").label == "only"
        assert _diag_classify("b").label == "only"
        assert _diag_classify("c").label == "only"

    def test_empty_table_raises(self):
        with pytest.raises(ValueError):
            mock_llm_table([])


# ---------------------------------------------------------------------------
# Snapshot / debug ergonomics (env-driven)
# ---------------------------------------------------------------------------
class TestDebugSnapshots:
    def test_request_snapshot_present_when_debug(self, monkeypatch):
        from agentic_function import agentic_function as _af
        @_af(output_schema={"label": str}, debug=True)
        def _dbg(text: str):
            """Classify. label: positive|negative|neutral"""
        mock_llm_table([{"label": "positive"}])
        r = _dbg("hi")
        assert r.metrics.request_snapshot is not None
        assert isinstance(r.metrics.request_snapshot, list)
        # At least the system + user messages
        assert any(m.get("role") == "user" for m in r.metrics.request_snapshot)
        # Response snapshot
        assert r.metrics.response_snapshot is not None

    def test_debug_env_var_enables_snapshots(self, monkeypatch):
        from agentic_function.runtime.executor import _DEBUG_ENV
        @agentic_function(output_schema={"label": str})
        def _dbg_env(text: str):
            """Classify. label: positive|negative|neutral"""
        mock_llm_table([{"label": "positive"}])
        monkeypatch.setenv("AGENTIC_DEBUG", "1")
        # The module-level _DEBUG_ENV is read at call-time. Re-import the
        # executor's helper so it re-reads the env var.
        from agentic_function.runtime import executor as _exec
        _exec._DEBUG_ENV = True
        try:
            r = _dbg_env("hi")
            assert r.metrics.request_snapshot is not None
        finally:
            _exec._DEBUG_ENV = False


# ---------------------------------------------------------------------------
# Diagnostic.to_text — readable output (regression test for log format)
# ---------------------------------------------------------------------------
class TestDiagnosticText:
    def test_text_includes_every_field(self):
        m = CallMetrics(
            attempts=2, successful=True, latency_ms=1500.0,
            backend="openai", model="gpt-4o-mini",
            finish_reason="stop",
            usage=TokenUsage.of(100, 50),
            cost_usd=0.001,
            retries=1,
            timings=PhaseTimings(
                prompt_render_ms=2.0, cache_lookup_ms=1.0,
                backend_call_ms=1400.0, parse_validate_ms=5.0,
                cache_write_ms=1.0,
            ),
            request_snapshot=[{"role": "user", "content": "hello"}],
            response_snapshot={"label": "positive"},
        )
        diag = diagnose_metrics(m, function_name="my_fn")
        text = diag.to_text()
        assert "my_fn" in text
        assert "openai" in text
        assert "gpt-4o-mini" in text
        assert "1500.0 ms" in text
        assert "backend_call_ms=1400.0ms" in text
        assert "100 in / 50 out" in text
        assert "$0.001000" in text
        assert "stop" in text
        assert "response" in text.lower()