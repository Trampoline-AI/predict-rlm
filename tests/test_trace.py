"""Tests for structured trace output."""

import json
from types import SimpleNamespace

import pytest

from predict_rlm.trace import (
    IterationStep,
    LMFinishMetadata,
    LMUsage,
    PredictCallDetail,
    PredictCallGroup,
    RunEvidence,
    RunEvidenceEvent,
    RunTrace,
    TokenUsage,
    ToolCall,
    _RawPredictCall,
    drain_predict_calls,
    drain_tool_calls,
    init_predict_call_collector,
    init_tool_call_collector,
    lm_completion_metadata_since,
    lm_finish_since,
    record_predict_call,
    record_tool_call,
    reset_predict_call_collector,
    reset_tool_call_collector,
    usage_since,
)


class TestRunTrace:
    def test_atomic_export_replaces_in_progress_trace(self, tmp_path):
        from predict_rlm.predict_rlm import PredictRLM

        path = tmp_path / "predict_rlm_trace.json"
        rlm = PredictRLM.__new__(PredictRLM)
        rlm._trace_export_path = path
        rlm._debug = False

        first = RunTrace(
            status="in_progress",
            model="openai/gpt-5",
            iterations=1,
            max_iterations=5,
            duration_ms=100,
        )
        second = RunTrace(
            status="completed",
            model="openai/gpt-5",
            iterations=2,
            max_iterations=5,
            duration_ms=200,
        )

        rlm._export_run_trace(first)
        assert json.loads(path.read_text())["status"] == "in_progress"

        rlm._export_run_trace(second)
        assert json.loads(path.read_text())["status"] == "completed"
        assert not list(tmp_path.glob("*.tmp"))

    def test_to_exportable_json_sanitizes_base64(self):
        b64 = "A" * 40000
        trace = RunTrace(
            status="completed",
            model="openai/gpt-5",
            iterations=1,
            max_iterations=5,
            duration_ms=100,
            steps=[
                IterationStep(
                    iteration=1,
                    reasoning="",
                    code="",
                    output="",
                    untruncated_output="",
                    duration_ms=100,
                    predict_calls=[
                        PredictCallGroup(
                            signature="page: dspy.Image -> answer",
                            model="openai/gpt-4o",
                            calls=[
                                PredictCallDetail(
                                    duration_ms=50,
                                    input={"page": f"data:image/png;base64,{b64}"},
                                    output={"answer": "hello"},
                                )
                            ],
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

    def test_to_proposer_keeps_behavioral_evidence_without_accounting(self):
        trace = RunTrace(
            status="completed",
            model="openai/gpt-5",
            sub_model="openai/gpt-4o",
            iterations=1,
            max_iterations=5,
            duration_ms=100,
            usage=LMUsage(
                main=TokenUsage(input_tokens=100, output_tokens=50, cost=0.01),
                sub=TokenUsage(input_tokens=20, output_tokens=10, cost=0.002),
            ),
            steps=[
                IterationStep(
                    iteration=1,
                    reasoning="inspect",
                    code="answer = tool('x')",
                    output="truncated",
                    untruncated_output="full",
                    error=True,
                    duration_ms=100,
                    lm=LMFinishMetadata(finish_reason="stop"),
                    tool_calls=[
                        ToolCall(
                            name="tool",
                            args=["x"],
                            kwargs={"mode": "fast"},
                            result={"answer": 1},
                            error="tool failed",
                            duration_ms=3,
                        )
                    ],
                    predict_calls=[
                        PredictCallGroup(
                            signature="q -> a",
                            instructions="answer",
                            model="openai/gpt-4o",
                            total_usage=TokenUsage(
                                input_tokens=20,
                                output_tokens=10,
                                cost=0.002,
                                cache_hits=1,
                            ),
                            calls=[
                                PredictCallDetail(
                                    duration_ms=10,
                                    usage=TokenUsage(
                                        input_tokens=20,
                                        output_tokens=10,
                                        cost=0.002,
                                        cache_hits=1,
                                    ),
                                    input={"q": "question"},
                                    output={"a": "answer"},
                                    error="predict failed",
                                    lm=LMFinishMetadata(finish_reason="length"),
                                )
                            ],
                        )
                    ],
                )
            ],
        )

        proposer = trace.to_proposer()
        data = proposer.model_dump()
        step = data["steps"][0]
        predict_call = step["predict_calls"][0]["calls"][0]

        assert step["reasoning"] == "inspect"
        assert step["code"] == "answer = tool('x')"
        assert step["output"] == "truncated"
        assert step["untruncated_output"] == "full"
        assert step["error"] is True
        assert step["lm"] == {"finish_reason": "stop"}
        assert step["tool_calls"][0] == {
            "name": "tool",
            "args": ["x"],
            "kwargs": {"mode": "fast"},
            "result": {"answer": 1},
            "error": "tool failed",
        }
        assert predict_call["input"] == {"q": "question"}
        assert predict_call["output"] == {"a": "answer"}
        assert predict_call["error"] == "predict failed"
        assert predict_call["lm"] == {"finish_reason": "length"}

        serialized = trace.to_proposer_json()
        forbidden = ("usage", "duration_ms", "cost", "cache_hits", "total_usage")
        for field in forbidden:
            assert f'"{field}"' not in serialized

    def test_to_proposer_projects_strict_evidence_without_raw_or_accounting_data(self):
        b64 = "A" * 40000
        trace = RunTrace(
            status="error",
            model="openai/gpt-5",
            iterations=1,
            max_iterations=5,
            duration_ms=100,
            evidence=RunEvidence(
                run_id="run",
                complete=False,
                terminal_outcome="error",
                events=[
                    RunEvidenceEvent(
                        sequence=1,
                        kind="run.started",
                        timestamp_ns=1,
                        data={"inputs": {"image": f"data:image/png;base64,{b64}"}},
                    ),
                    RunEvidenceEvent(
                        sequence=2,
                        kind="iteration.recorded",
                        timestamp_ns=2,
                        data={
                            "step": {
                                "iteration": 1,
                                "duration_ms": 100,
                                "usage": {"input_tokens": 10, "cost": 0.2},
                                "predict_calls": [
                                    {
                                        "total_usage": {
                                            "output_tokens": 3,
                                            "cache_hits": 1,
                                        }
                                    }
                                ],
                            }
                        },
                    ),
                    RunEvidenceEvent(
                        sequence=3,
                        kind="tool.finished",
                        timestamp_ns=3,
                        data={
                            "name": "inspect",
                            "result": {"image": f"data:image/png;base64,{b64}"},
                            "error": "failed usefully",
                            "duration_ms": 9,
                            "cost": 0.1,
                        },
                    ),
                ],
            ),
        )

        proposer = trace.to_proposer().model_dump()
        serialized = trace.to_proposer_json()

        assert proposer["evidence"]["complete"] is False
        assert [event["kind"] for event in proposer["evidence"]["events"]] == [
            "run.started",
            "iteration.recorded",
            "tool.finished",
        ]
        assert proposer["evidence"]["events"][0]["data"] == {}
        assert proposer["evidence"]["events"][1]["data"] == {"iteration": 1}
        assert proposer["evidence"]["events"][2]["data"]["name"] == "inspect"
        assert proposer["evidence"]["events"][2]["data"]["error"] == "failed usefully"
        assert "AAAA" not in serialized
        assert "<IMAGE_BASE_64_ENCODED(40000)>" in serialized
        for field in (
            "timestamp_ns",
            "inputs",
            "step",
            "duration_ms",
            "usage",
            "cost",
            "cache_hits",
            "total_usage",
            "input_tokens",
            "output_tokens",
        ):
            assert f'"{field}"' not in serialized


class TestPredictCallCollector:
    def test_groups_by_signature_instructions_and_model(self):
        token = init_predict_call_collector()
        try:
            for signature, instructions, model, answer in [
                ("q -> a", "extract", "m", "first"),
                ("q -> a", "extract", "m", "second"),
                ("q -> a", "summarize", "m", "different instructions"),
                ("q -> a", "extract", "other", "different model"),
                ("text -> label", "extract", "m", "different signature"),
            ]:
                record_predict_call(
                    _RawPredictCall(
                        signature=signature,
                        instructions=instructions,
                        model=model,
                        duration_ms=10,
                        usage=TokenUsage(input_tokens=5, cost=0.01),
                        input={"q": "input"},
                        output={"a": answer},
                    )
                )
            groups = drain_predict_calls()
            assert [[call.output["a"] for call in group.calls] for group in groups] == [
                ["first", "second"],
                ["different instructions"],
                ["different model"],
                ["different signature"],
            ]
            assert groups[0].total_usage.input_tokens == 10
            assert groups[0].total_usage.cost == pytest.approx(0.02)
            assert drain_predict_calls() == []
        finally:
            reset_predict_call_collector(token)

    def test_nested_collector_restores_parent_calls(self):
        outer_token = init_predict_call_collector()
        record_predict_call(
            _RawPredictCall(
                signature="outer-before",
                instructions=None,
                model="m",
                duration_ms=10,
                usage=TokenUsage(),
                input={},
                output={},
            )
        )

        inner_token = init_predict_call_collector()
        record_predict_call(
            _RawPredictCall(
                signature="inner",
                instructions=None,
                model="m",
                duration_ms=10,
                usage=TokenUsage(),
                input={},
                output={},
            )
        )
        inner_groups = drain_predict_calls()
        reset_predict_call_collector(inner_token)

        record_predict_call(
            _RawPredictCall(
                signature="outer-after",
                instructions=None,
                model="m",
                duration_ms=10,
                usage=TokenUsage(),
                input={},
                output={},
            )
        )
        outer_groups = drain_predict_calls()
        reset_predict_call_collector(outer_token)

        assert [group.signature for group in inner_groups] == ["inner"]
        assert [group.signature for group in outer_groups] == ["outer-before", "outer-after"]


class TestToolCallCollector:
    def test_nested_collector_restores_parent_calls(self):
        outer_token = init_tool_call_collector()
        record_tool_call(
            ToolCall(
                name="outer_before",
                args=[],
                kwargs={},
                result="ok",
                duration_ms=1,
            )
        )

        inner_token = init_tool_call_collector()
        record_tool_call(
            ToolCall(
                name="inner",
                args=[],
                kwargs={},
                result="",
                error="tool failed",
                duration_ms=1,
            )
        )
        inner_calls = drain_tool_calls()
        assert drain_tool_calls() == []
        reset_tool_call_collector(inner_token)

        record_tool_call(
            ToolCall(
                name="outer_after",
                args=[],
                kwargs={},
                result="ok",
                duration_ms=1,
            )
        )
        outer_calls = drain_tool_calls()
        reset_tool_call_collector(outer_token)

        assert [call.name for call in inner_calls] == ["inner"]
        assert inner_calls[0].error == "tool failed"
        assert [call.name for call in outer_calls] == ["outer_before", "outer_after"]


class TestUsageSince:
    def test_history_delta_excludes_cached_cost_without_losing_real_usage(self):
        lm = SimpleNamespace(
            history=[
                {"usage": {"prompt_tokens": 900}, "cost": 1.0},
                {"usage": {"prompt_tokens": 100, "completion_tokens": 20}, "cost": 0.01},
                {"usage": {}, "cost": 0.01, "response": SimpleNamespace(cache_hit=True)},
                {"usage": {}, "cost": 0.01},
                {"usage": {"prompt_tokens": 0, "completion_tokens": 0}, "cost": 0},
                {"usage": {"prompt_tokens": 200, "completion_tokens": 30}, "cost": 0.02},
            ]
        )
        usage = usage_since(lm, 1)
        assert usage.input_tokens == 300
        assert usage.output_tokens == 50
        assert usage.cost == pytest.approx(0.03)
        assert usage.cache_hits == 2

    def test_lm_completion_metadata_includes_prompt_cache_stats(self):
        lm = SimpleNamespace()
        lm.history = [
            {
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 50,
                    "prompt_tokens_details": {"cached_tokens": 750},
                },
                "response": {"choices": [{"finish_reason": "stop"}]},
            },
            {
                "usage": {
                    "input_tokens": 500,
                    "output_tokens": 20,
                    "input_tokens_details": {"cached_tokens": 100},
                },
                "response": {"choices": [{"finish_reason": "stop"}]},
            },
        ]

        metadata = lm_completion_metadata_since(lm, 0)

        assert metadata is not None
        assert metadata.input_tokens == 1500
        assert metadata.cached_input_tokens == 850
        assert metadata.cache_read_ratio == pytest.approx(850 / 1500)
        assert lm_finish_since(lm, 0) == LMFinishMetadata(finish_reason="stop")
