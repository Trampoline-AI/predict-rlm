"""Tests for structured trace output."""

import time
from unittest.mock import MagicMock

import pytest

from predict_rlm.trace import (
    IterationStep,
    LMUsage,
    PredictCallDetail,
    PredictCallGroup,
    RunTrace,
    TokenUsage,
    ToolCall,
    _RawPredictCall,
    _sanitize_for_trace,
    drain_predict_calls,
    drain_tool_calls,
    init_predict_call_collector,
    init_tool_call_collector,
    ms_since,
    record_predict_call,
    record_tool_call,
    snapshot_lm_history_len,
    usage_since,
)


class TestTokenUsage:
    def test_defaults_to_zero(self):
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cost == 0.0

    def test_iadd(self):
        a = TokenUsage(input_tokens=10, output_tokens=5, cost=0.01)
        b = TokenUsage(input_tokens=20, output_tokens=10, cost=0.02)
        a += b
        assert a.input_tokens == 30
        assert a.output_tokens == 15
        assert a.cost == pytest.approx(0.03)


class TestPredictCallGroup:
    def test_fields(self):
        group = PredictCallGroup(
            signature="q -> a",
            model="openai/gpt-4o",
            calls=[
                PredictCallDetail(duration_ms=90, usage=TokenUsage(input_tokens=40, output_tokens=20, cost=0.004)),
                PredictCallDetail(duration_ms=110, usage=TokenUsage(input_tokens=60, output_tokens=30, cost=0.006)),
            ],
        )
        assert group.signature == "q -> a"
        assert group.model == "openai/gpt-4o"
        assert len(group.calls) == 2
        assert group.calls[0].duration_ms == 90


class TestIterationStep:
    def test_fields(self):
        step = IterationStep(
            iteration=1,
            reasoning="think",
            code="print(1)",
            output="1",
            untruncated_output="1",
            duration_ms=500,
        )
        assert step.iteration == 1
        assert step.predict_calls == []

    def test_output_vs_untruncated(self):
        long_output = "x" * 10000
        truncated = long_output[:5000] + f"\n... (truncated to 5000/{len(long_output):,} chars)"
        step = IterationStep(
            iteration=1,
            reasoning="",
            code="print('x' * 10000)",
            output=truncated,
            untruncated_output=long_output,
            duration_ms=100,
        )
        assert len(step.untruncated_output) == 10000
        assert len(step.output) < 6000
        assert "truncated" in step.output


class TestRunTrace:
    def test_serialization(self):
        trace = RunTrace(
            status="completed",
            model="openai/gpt-5",
            sub_model="openai/gpt-4o",
            iterations=2,
            max_iterations=5,
            duration_ms=1000,
            usage=LMUsage(
                main=TokenUsage(input_tokens=70, output_tokens=40, cost=0.04),
                sub=TokenUsage(input_tokens=30, output_tokens=10, cost=0.01),
            ),
            steps=[
                IterationStep(
                    iteration=1,
                    reasoning="step 1",
                    code="x = 1",
                    output="",
                    untruncated_output="",
                    duration_ms=400,
                    predict_calls=[
                        PredictCallGroup(
                            signature="q -> a",
                            model="openai/gpt-4o",
                            calls=[PredictCallDetail(duration_ms=200, usage=TokenUsage(input_tokens=30, output_tokens=10, cost=0.01))],
                        )
                    ],
                ),
                IterationStep(
                    iteration=2,
                    reasoning="step 2",
                    code="SUBMIT(x)",
                    output="FINAL: {'answer': 1}",
                    untruncated_output="FINAL: {'answer': 1}",
                    duration_ms=600,
                ),
            ],
        )
        data = trace.model_dump()
        assert data["status"] == "completed"
        assert data["model"] == "openai/gpt-5"
        assert data["sub_model"] == "openai/gpt-4o"
        assert data["iterations"] == 2
        assert len(data["steps"]) == 2
        assert len(data["steps"][0]["predict_calls"]) == 1
        assert data["steps"][0]["predict_calls"][0]["signature"] == "q -> a"
        assert data["steps"][0]["predict_calls"][0]["model"] == "openai/gpt-4o"
        assert data["usage"]["main"]["input_tokens"] == 70
        assert data["usage"]["sub"]["input_tokens"] == 30

    def test_sub_model_optional(self):
        trace = RunTrace(
            status="completed",
            model="openai/gpt-5",
            iterations=1,
            max_iterations=5,
            duration_ms=100,
        )
        assert trace.sub_model is None
        assert trace.usage.sub.input_tokens == 0

    def test_status_literals(self):
        for status in ("completed", "max_iterations", "error"):
            trace = RunTrace(
                status=status, model="openai/gpt-5",
                iterations=1, max_iterations=5, duration_ms=100,
            )
            assert trace.status == status

    def test_to_exportable_json_returns_string(self):
        trace = RunTrace(
            status="completed", model="openai/gpt-5",
            iterations=1, max_iterations=5, duration_ms=100,
        )
        result = trace.to_exportable_json()
        assert isinstance(result, str)
        import json
        data = json.loads(result)
        assert data["status"] == "completed"

    def test_to_exportable_json_sanitizes_base64(self):
        b64 = "A" * 40000
        trace = RunTrace(
            status="completed", model="openai/gpt-5",
            iterations=1, max_iterations=5, duration_ms=100,
            steps=[
                IterationStep(
                    iteration=1, reasoning="", code="",
                    output="", untruncated_output="", duration_ms=100,
                    predict_calls=[
                        PredictCallGroup(
                            signature="page: dspy.Image -> answer",
                            model="openai/gpt-4o",
                            calls=[PredictCallDetail(
                                duration_ms=50,
                                input={"page": f"data:image/png;base64,{b64}"},
                                output={"answer": "hello"},
                            )],
                        )
                    ],
                )
            ],
        )
        result = trace.to_exportable_json()
        assert "AAAA" not in result
        assert "<IMAGE_BASE_64_ENCODED(40000)>" in result
        # model_dump still has the full data
        full = trace.model_dump()
        assert b64 in full["steps"][0]["predict_calls"][0]["calls"][0]["input"]["page"]

    def test_to_exportable_json_writes_file(self, tmp_path):
        trace = RunTrace(
            status="completed", model="openai/gpt-5",
            iterations=1, max_iterations=5, duration_ms=100,
        )
        out = tmp_path / "trace.json"
        result = trace.to_exportable_json(out)
        assert out.exists()
        assert result == out.read_text()


class TestPredictCallCollector:
    def test_different_signatures_not_aggregated(self):
        init_predict_call_collector()
        record_predict_call(_RawPredictCall(
            signature="a -> b", instructions=None, model="openai/gpt-4o",
            duration_ms=50, usage=TokenUsage(), input={"a": "1"}, output={"b": "x"},
        ))
        record_predict_call(_RawPredictCall(
            signature="c -> d", instructions=None, model="openai/gpt-4o",
            duration_ms=30, usage=TokenUsage(), input={"c": "2"}, output={"d": "y"},
        ))
        groups = drain_predict_calls()
        assert len(groups) == 2
        assert groups[0].signature == "a -> b"
        assert groups[1].signature == "c -> d"
        assert len(groups[0].calls) == 1
        assert groups[0].calls[0].output == {"b": "x"}
        # Drain clears the list
        assert drain_predict_calls() == []

    def test_same_signature_aggregated(self):
        init_predict_call_collector()
        record_predict_call(_RawPredictCall(
            signature="page: dspy.Image -> items: list[str]",
            instructions="Extract items",
            model="openai/gpt-4o",
            duration_ms=100,
            usage=TokenUsage(input_tokens=50, output_tokens=20, cost=0.01),
            input={"page": "url1"}, output={"items": ["a"]},
        ))
        record_predict_call(_RawPredictCall(
            signature="page: dspy.Image -> items: list[str]",
            instructions="Extract items",
            model="openai/gpt-4o",
            duration_ms=150,
            usage=TokenUsage(input_tokens=60, output_tokens=30, cost=0.02),
            input={"page": "url2"}, output={"items": ["b", "c"]},
        ))
        record_predict_call(_RawPredictCall(
            signature="page: dspy.Image -> items: list[str]",
            instructions="Extract items",
            model="openai/gpt-4o",
            duration_ms=120,
            usage=TokenUsage(input_tokens=55, output_tokens=25, cost=0.015),
            input={"page": "url3"}, output={"items": []},
        ))
        groups = drain_predict_calls()
        assert len(groups) == 1
        agg = groups[0]
        assert len(agg.calls) == 3
        assert agg.calls[0].duration_ms == 100
        assert agg.calls[0].output == {"items": ["a"]}
        assert agg.calls[1].usage.input_tokens == 60
        assert agg.calls[2].output == {"items": []}

    def test_different_instructions_not_aggregated(self):
        init_predict_call_collector()
        record_predict_call(_RawPredictCall(
            signature="q -> a", instructions="Task A", model="m",
            duration_ms=10, usage=TokenUsage(), input={"q": "x"}, output={"a": "1"},
        ))
        record_predict_call(_RawPredictCall(
            signature="q -> a", instructions="Task B", model="m",
            duration_ms=10, usage=TokenUsage(), input={"q": "y"}, output={"a": "2"},
        ))
        groups = drain_predict_calls()
        assert len(groups) == 2

    def test_record_without_init_is_silent(self):
        # In a fresh context (no collector set), recording is a no-op
        import contextvars

        from predict_rlm.trace import _predict_calls

        token = _predict_calls.set([])
        try:
            # Reset to simulate no collector
            _predict_calls = contextvars.ContextVar("_predict_calls_fresh")
        finally:
            # Restore
            from predict_rlm import trace

            trace._predict_calls.reset(token)


class TestToolCall:
    def test_fields(self):
        call = ToolCall(
            name="read_pdf", args=[], kwargs={"path": "/tmp/doc.pdf"},
            result='{"pages": 5}', duration_ms=200,
        )
        assert call.name == "read_pdf"
        assert call.error is None
        assert call.result == '{"pages": 5}'

    def test_error_field(self):
        call = ToolCall(
            name="bad_tool", args=[], kwargs={},
            result="", error="FileNotFoundError: no such file", duration_ms=10,
        )
        assert call.error == "FileNotFoundError: no such file"

    def test_serialization(self):
        call = ToolCall(
            name="search", args=["query"], kwargs={"limit": 10},
            result='["a", "b"]', duration_ms=50,
        )
        data = call.model_dump()
        assert data["name"] == "search"
        assert data["args"] == ["query"]
        assert data["kwargs"] == {"limit": 10}


class TestToolCallCollector:
    def test_init_drain_cycle(self):
        init_tool_call_collector()
        record_tool_call(ToolCall(
            name="tool_a", args=[], kwargs={"x": 1},
            result="ok", duration_ms=50,
        ))
        record_tool_call(ToolCall(
            name="tool_b", args=[1, 2], kwargs={},
            result="done", duration_ms=30,
        ))
        calls = drain_tool_calls()
        assert len(calls) == 2
        assert calls[0].name == "tool_a"
        assert calls[1].name == "tool_b"
        assert calls[1].args == [1, 2]
        # Drain clears the list
        assert drain_tool_calls() == []

    def test_error_calls_recorded(self):
        init_tool_call_collector()
        record_tool_call(ToolCall(
            name="failing_tool", args=[], kwargs={},
            result="", error="boom", duration_ms=5,
        ))
        calls = drain_tool_calls()
        assert len(calls) == 1
        assert calls[0].error == "boom"

    def test_record_without_init_is_silent(self):
        # No collector initialized — recording should not raise
        record_tool_call(ToolCall(
            name="orphan", args=[], kwargs={}, result="", duration_ms=1,
        ))
        # No crash, no data — just a no-op


class TestSnapshotLmHistoryLen:
    def test_with_history(self):
        lm = MagicMock()
        lm.history = [{"usage": {}}, {"usage": {}}]
        assert snapshot_lm_history_len(lm) == 2

    def test_without_history(self):
        lm = MagicMock(spec=[])
        assert snapshot_lm_history_len(lm) == 0

    def test_empty_history(self):
        lm = MagicMock()
        lm.history = []
        assert snapshot_lm_history_len(lm) == 0


class TestUsageSince:
    def test_sums_new_entries(self):
        lm = MagicMock()
        lm.history = [
            {"usage": {"prompt_tokens": 100, "completion_tokens": 50}, "cost": 0.01},
            {"usage": {"prompt_tokens": 200, "completion_tokens": 100}, "cost": 0.02},
            {"usage": {"prompt_tokens": 300, "completion_tokens": 150}, "cost": 0.03},
        ]
        usage = usage_since(lm, 1)
        assert usage.input_tokens == 500
        assert usage.output_tokens == 250
        assert usage.cost == pytest.approx(0.05)

    def test_since_zero_sums_all(self):
        lm = MagicMock()
        lm.history = [
            {"usage": {"prompt_tokens": 100, "completion_tokens": 50}, "cost": 0.01},
        ]
        usage = usage_since(lm, 0)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_since_beyond_length_returns_zero(self):
        lm = MagicMock()
        lm.history = [{"usage": {"prompt_tokens": 100, "completion_tokens": 50}, "cost": 0.01}]
        usage = usage_since(lm, 5)
        assert usage.input_tokens == 0

    def test_no_history_returns_zero(self):
        lm = MagicMock(spec=[])
        usage = usage_since(lm, 0)
        assert usage.input_tokens == 0


class TestSanitizeForTrace:
    def test_replaces_data_uri(self):
        data_uri = "data:image/png;base64," + "A" * 40000
        result = _sanitize_for_trace(data_uri)
        assert result == "data:image/png;base64,<IMAGE_BASE_64_ENCODED(40000)>"

    def test_replaces_nested_in_dict(self):
        data = {
            "page": "data:image/jpeg;base64," + "B" * 20000,
            "question": "What is this?",
        }
        result = _sanitize_for_trace(data)
        assert result["question"] == "What is this?"
        assert result["page"] == "data:image/jpeg;base64,<IMAGE_BASE_64_ENCODED(20000)>"

    def test_replaces_in_list(self):
        data = ["data:image/png;base64," + "C" * 10000, "normal string"]
        result = _sanitize_for_trace(data)
        assert result[0] == "data:image/png;base64,<IMAGE_BASE_64_ENCODED(10000)>"
        assert result[1] == "normal string"

    def test_leaves_normal_strings(self):
        assert _sanitize_for_trace("hello") == "hello"
        assert _sanitize_for_trace("data:not-base64") == "data:not-base64"

    def test_leaves_non_strings(self):
        assert _sanitize_for_trace(42) == 42
        assert _sanitize_for_trace(None) is None


class TestMsSince:
    def test_returns_positive_int(self):
        start = time.perf_counter()
        ms = ms_since(start)
        assert isinstance(ms, int)
        assert ms >= 0
