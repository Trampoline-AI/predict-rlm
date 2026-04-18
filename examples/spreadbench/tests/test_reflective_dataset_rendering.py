"""Proposer reflective-dataset rendering.

The proposer's reflective dataset used to render only the flat
``result.trajectory`` (reasoning / code / output per step). Everything
else the RLM did — tool invocations (``recalculate``, ``render``),
sub-LM ``predict`` calls, per-step timing, exit status, aggregate
token usage — was captured in ``RunTrace`` and persisted to
``task_traces/evaluate_NNNN.jsonl``, but never shown to the proposer.

These tests pin the contract after we migrate the proposer to render
directly from ``RunTrace``: the richer fields ARE surfaced, the render
stays deterministic, and the old trajectory path still works as a
fallback when ``run_trace`` is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from predict_rlm.trace import (  # noqa: E402
    IterationStep,
    LMUsage,
    PredictCallDetail,
    PredictCallGroup,
    RunTrace,
    TokenUsage,
    ToolCall,
)


def _make_run_trace() -> RunTrace:
    return RunTrace(
        status="completed",
        model="openai/gpt-5.4-mini",
        sub_model="openai/gpt-5.1-codex-mini",
        iterations=2,
        max_iterations=32,
        duration_ms=45_000,
        usage=LMUsage(
            main=TokenUsage(input_tokens=120_000, output_tokens=4_500, cost=0.11),
            sub=TokenUsage(input_tokens=8_000, output_tokens=300, cost=0.01),
        ),
        steps=[
            IterationStep(
                iteration=1,
                reasoning="Inspect the workbook first.",
                code="wb = openpyxl.load_workbook(path)\nprint(wb.sheetnames)",
                output="['Sheet1']",
                untruncated_output="['Sheet1']",
                duration_ms=8_000,
                tool_calls=[
                    ToolCall(
                        name="render",
                        args=[],
                        kwargs={"cell_range": "A1:J20"},
                        result="data:image/png;base64,<IMAGE_BASE_64_ENCODED(1234)>",
                        duration_ms=1_400,
                    ),
                ],
                predict_calls=[
                    PredictCallGroup(
                        signature="page: dspy.Image -> items: list[str]",
                        model="openai/gpt-5.1-codex-mini",
                        total_usage=TokenUsage(
                            input_tokens=4_000, output_tokens=120, cost=0.005
                        ),
                        calls=[
                            PredictCallDetail(
                                duration_ms=2_200,
                                usage=TokenUsage(input_tokens=4_000, output_tokens=120, cost=0.005),
                                input={"page": "data:image/png;base64,<IMAGE_BASE_64_ENCODED(1234)>"},
                                output={"items": ["A", "B", "C"]},
                            )
                        ],
                    ),
                ],
            ),
            IterationStep(
                iteration=2,
                reasoning="Compute answer and SUBMIT.",
                code="SUBMIT(ManipulateSpreadsheet(output_spreadsheet=File(path='/tmp/out.xlsx')))",
                output="FINAL: ok",
                untruncated_output="FINAL: ok",
                duration_ms=1_200,
            ),
        ],
    )


class TestCaseSummaryFromRunTrace:
    def test_header_includes_status_iterations_and_tokens(self):
        from lib.optimize import SpreadsheetAdapter

        rt = _make_run_trace()
        case = {
            "idx": 0, "score": 1.0, "passed": True, "message": "",
            "run_trace": rt,
        }
        header = SpreadsheetAdapter._case_summary(case)
        assert "case 0" in header
        assert "score=1.00" in header
        assert "PASS" in header
        # Enriched fields from RunTrace
        assert "status=completed" in header
        assert "2/32 iters" in header or "2 iters" in header
        # Token summary: input+output presented compactly
        assert "120k" in header.lower() or "120,000" in header
        # Duration at least hinted
        assert "45" in header  # seconds or ms representation

    def test_header_handles_missing_run_trace(self):
        from lib.optimize import SpreadsheetAdapter

        case = {"idx": 5, "score": 0.0, "passed": False, "message": "crash"}
        header = SpreadsheetAdapter._case_summary(case)
        # Old behaviour preserved when run_trace is missing
        assert "case 5" in header
        assert "score=0.00" in header
        assert "FAIL" in header


class TestRenderTraceFromRunTrace:
    def test_renders_step_with_tool_calls_and_predict_calls(self):
        from lib.optimize import SpreadsheetAdapter

        rt = _make_run_trace()
        rendered = SpreadsheetAdapter._render_trace(rt, "WORST CASE 0")
        # Still contains the classic reasoning/code/output
        assert "Inspect the workbook" in rendered
        assert "wb = openpyxl.load_workbook" in rendered
        assert "['Sheet1']" in rendered
        # Step header carries duration
        assert "8000ms" in rendered or "8.0s" in rendered or "8,000" in rendered
        # Tool calls rendered compactly
        assert "render" in rendered
        assert "cell_range" in rendered
        # predict_call signature surfaced
        assert "page: dspy.Image -> items: list[str]" in rendered
        # sub-LM token count visible
        assert "4000" in rendered or "4,000" in rendered or "4k" in rendered.lower()

    def test_handles_empty_trace(self):
        from lib.optimize import SpreadsheetAdapter

        empty = RunTrace(
            status="error", model="m", iterations=0, max_iterations=32, duration_ms=0
        )
        rendered = SpreadsheetAdapter._render_trace(empty, "CRASHED CASE 3")
        assert "no" in rendered.lower() or "empty" in rendered.lower()

    def test_falls_back_to_legacy_trajectory_when_no_run_trace(self):
        """If run_trace is missing but trajectory is present (e.g. partial
        trajectory recovered from crash before RunTrace was built), render
        should degrade gracefully to the old reasoning/code/output layout.
        """
        from lib.optimize import SpreadsheetAdapter

        trajectory = [
            {"reasoning": "thinking", "code": "x = 1", "output": "ok"},
        ]
        rendered = SpreadsheetAdapter._render_trace(trajectory, "PARTIAL CASE")
        assert "thinking" in rendered
        assert "x = 1" in rendered
        assert "ok" in rendered

    def test_tool_call_error_surfaced(self):
        from lib.optimize import SpreadsheetAdapter

        rt = RunTrace(
            status="completed",
            model="openai/gpt-5.4-mini",
            iterations=1,
            max_iterations=32,
            duration_ms=1000,
            steps=[
                IterationStep(
                    iteration=1,
                    reasoning="try render",
                    code="render(...)",
                    output="[Error] render failed",
                    untruncated_output="[Error] render failed",
                    error=True,
                    duration_ms=900,
                    tool_calls=[
                        ToolCall(
                            name="render",
                            args=[],
                            kwargs={"cell_range": "BOGUS!Z99"},
                            result="",
                            error="ValueError: invalid cell range",
                            duration_ms=80,
                        )
                    ],
                )
            ],
        )
        rendered = SpreadsheetAdapter._render_trace(rt, "WORST")
        # The tool error message must reach the proposer — this is exactly
        # the signal that lets it write "never pass sheet-qualified bogus
        # ranges to render" style rules.
        assert "invalid cell range" in rendered or "ValueError" in rendered
