from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from predict_rlm.telemetry import JsonlTelemetrySink, TelemetryContext
from rlm_gepa import EvaluationContext

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from spreadsheet_rlm.bench.dataset import SpreadsheetTask  # noqa: E402
from spreadsheet_rlm.gepa import project as project_module  # noqa: E402
from spreadsheet_rlm.gepa.config import default_config  # noqa: E402
from spreadsheet_rlm.gepa.project import SpreadsheetGepaProject  # noqa: E402


class _DummyLM:
    model = "dummy/model"


class _FakePredictRLM:
    def __init__(self, *_args, **_kwargs):
        pass

    async def acall(self, **_kwargs):
        output_path = Path(_kwargs["input_spreadsheet"].path).with_name("candidate.xlsx")
        output_path.write_bytes(b"candidate")
        return SimpleNamespace(
            output_spreadsheet=SimpleNamespace(path=str(output_path)),
            trace={"status": "ok"},
        )


def test_spreadsheet_project_writes_case_start_and_end_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    async def run_case() -> float:
        monkeypatch.setattr(project_module, "PredictRLM", _FakePredictRLM)
        monkeypatch.setattr(project_module, "parse_answer_position", lambda *_args: ("Sheet1", "A1"))
        monkeypatch.setattr(project_module, "_best_effort_recalculate", lambda *_args: None)
        monkeypatch.setattr(project_module, "score_workbooks", lambda *_args: (1.0, "ok"))
        input_path = tmp_path / "input.xlsx"
        answer_path = tmp_path / "answer.xlsx"
        input_path.write_bytes(b"input")
        answer_path.write_bytes(b"answer")
        telemetry_context = TelemetryContext(
            sink=JsonlTelemetrySink(tmp_path / "telemetry" / "events.jsonl"),
            trace_id="run_test:candidate",
            run_id="run_test",
            eval_kind="valset",
            eval_idx=3,
            attempt_id="attempt_0003",
            candidate_hash="cand_sha256_test",
        )
        context = EvaluationContext(
            lm=_DummyLM(),
            sub_lm=_DummyLM(),
            max_iterations=1,
            task_timeout=5,
            output_dir=tmp_path,
            kind="valset",
            concurrency=7,
            telemetry_context=telemetry_context,
        )
        task = SpreadsheetTask(
            task_id="581-46",
            instruction="Fill A1",
            instruction_type="edit",
            answer_position="A1",
            spreadsheet_dir=str(tmp_path),
            test_cases=((2, str(input_path), str(answer_path)),),
        )

        project = SpreadsheetGepaProject(default_config())
        score, _feedback, _trace = await project._run_case(
            task,
            2,
            str(input_path),
            str(answer_path),
            skill=object(),
            context=context,
            tmp_dir=tmp_path,
        )
        return score

    score = asyncio.run(run_case())

    assert score == 1.0
    events = [
        json.loads(line)
        for line in (tmp_path / "telemetry" / "events.jsonl").read_text().splitlines()
    ]
    assert [event["name"] for event in events] == [
        "spreadbench.case.start",
        "spreadbench.case.end",
    ]
    attrs = events[1]["attributes"]
    assert attrs["rlm.run_id"] == "run_test"
    assert attrs["rlm.candidate_hash"] == "cand_sha256_test"
    assert attrs["rlm.eval_kind"] == "valset"
    assert attrs["rlm.eval_idx"] == 3
    assert attrs["rlm.attempt_id"] == "attempt_0003"
    assert attrs["rlm.phase"] == "valset"
    assert attrs["spreadbench.example_id"] == "581-46"
    assert attrs["spreadbench.case_idx"] == 2
    assert attrs["rlm.configured_timeout_sec"] == 5
    assert attrs["rlm.concurrency"] == 7
    assert attrs["spreadbench.score"] == 1.0


def test_best_effort_recalculate_preserves_telemetry_when_exception_is_swallowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_recalculate(_path: str, *, telemetry_context: TelemetryContext | None = None):
        assert telemetry_context is not None
        telemetry_context.write_span(
            "host_tool.recalculate",
            event_domain="host_tool",
            status={"code": "ERROR", "message": "synthetic recalc failure"},
            attributes={"failure.class": "evaluator_exception"},
        )
        raise RuntimeError("synthetic recalc failure")

    monkeypatch.setattr(project_module, "recalculate", fake_recalculate)
    telemetry_context = TelemetryContext(
        sink=JsonlTelemetrySink(tmp_path / "telemetry" / "events.jsonl"),
        trace_id="trace-recalc-swallowed",
        run_id="run_test",
    )

    project_module._best_effort_recalculate(tmp_path / "output.xlsx", telemetry_context)

    events = [
        json.loads(line)
        for line in (tmp_path / "telemetry" / "events.jsonl").read_text().splitlines()
    ]
    assert events[0]["name"] == "host_tool.recalculate"
    assert events[0]["status"]["code"] == "ERROR"
    assert events[0]["attributes"]["failure.class"] == "evaluator_exception"
