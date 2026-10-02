from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest

from predict_rlm.evidence import RunEvidence, RunEvidenceEvent
from predict_rlm.trace import RunTrace

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from spreadsheet_rlm.bench import evaluation  # noqa: E402
from spreadsheet_rlm.bench.config import EvalConfig  # noqa: E402
from spreadsheet_rlm.bench.dataset import SpreadsheetTask  # noqa: E402
from spreadsheet_rlm.bench.evaluation import (  # noqa: E402
    CaseResult,
    TaskResult,
    _dump_eval_task_traces,
)


def _trace(status: Literal["completed", "error"] = "completed") -> RunTrace:
    return RunTrace(
        status=status,
        model="test/model",
        iterations=1,
        max_iterations=2,
        duration_ms=10,
    )


def test_trace_export_distinguishes_failed_runs_from_absent_artifacts(tmp_path):
    completed_trace = _trace()
    failed_trace = _trace("error")
    completed_evidence = RunEvidence(
        run_id="completed", complete=True, terminal_outcome="completed"
    )
    failed_evidence = RunEvidence(run_id="failed", complete=True, terminal_outcome="error")
    cases = [
        CaseResult(
            1,
            1.0,
            True,
            "ok",
            recalc_source="test",
            run_trace=completed_trace,
            evidence=completed_evidence,
        ),
        CaseResult(
            2,
            0.0,
            False,
            "RLM error",
            run_trace=failed_trace,
            evidence=failed_evidence,
        ),
        CaseResult(3, 0.0, False, "Pre-run failure"),
    ]

    _dump_eval_task_traces(tmp_path, [TaskResult("task", 1 / 3, 0, cases)])

    rows = [
        json.loads(line) for line in (tmp_path / "task_traces.jsonl").read_text().splitlines()
    ]
    assert rows == [
        {
            "task_id": "task",
            "case_idx": case.idx,
            "score": case.score,
            "passed": case.passed,
            "message": case.message,
            "recalc_source": case.recalc_source,
            "trace": trace,
            "evidence": evidence,
        }
        for case, trace, evidence in zip(
            cases,
            [completed_trace.model_dump(), failed_trace.model_dump(), None],
            [completed_evidence.model_dump(), failed_evidence.model_dump(), None],
            strict=True,
        )
    ]


@pytest.mark.parametrize("serializer_raises", [False, True])
def test_malformed_trace_export_raises_instead_of_recording_null(tmp_path, serializer_raises):
    class BrokenTrace(RunTrace):
        def to_exportable_json(self, path=None, indent=2):
            if serializer_raises:
                raise ValueError("cannot serialize trace")
            return "not JSON"

    case = CaseResult(
        1,
        0.0,
        False,
        "RLM error",
        run_trace=BrokenTrace(**_trace("error").model_dump()),
    )

    with pytest.raises(ValueError):
        _dump_eval_task_traces(tmp_path, [TaskResult("task", 0.0, 0, [case])])

    assert (tmp_path / "task_traces.jsonl").read_text() == ""


def test_unserializable_evidence_export_raises(tmp_path):
    evidence = RunEvidence(
        run_id="failed",
        complete=False,
        events=[
            RunEvidenceEvent(
                sequence=1, kind="invalid", timestamp_ns=0, data={"value": object()}
            )
        ],
    )
    case = CaseResult(1, 0.0, False, "RLM error", evidence=evidence)

    with pytest.raises(TypeError):
        _dump_eval_task_traces(tmp_path, [TaskResult("task", 0.0, 0, [case])])

    assert (tmp_path / "task_traces.jsonl").read_text() == ""


def test_unwritable_trace_export_raises(tmp_path):
    (tmp_path / "task_traces.jsonl").mkdir()
    case = CaseResult(1, 0.0, False, "Pre-run failure")

    with pytest.raises(IsADirectoryError):
        _dump_eval_task_traces(tmp_path, [TaskResult("task", 0.0, 0, [case])])


@pytest.mark.parametrize("failure_stage", ["copy", "score"])
def test_post_run_failure_exports_completed_artifacts(tmp_path, monkeypatch, failure_stage):
    trace = _trace()
    evidence = RunEvidence(run_id="completed", complete=True, terminal_outcome="completed")
    produced = tmp_path / "produced.xlsx"
    produced.write_bytes(b"workbook")

    class FakePredictRLM:
        def __init__(self, *_args, **_kwargs):
            pass

        async def acall(self, **_kwargs):
            return SimpleNamespace(
                output_spreadsheet=SimpleNamespace(path=str(produced)),
                trace=trace,
                evidence=evidence,
            )

    def fail(*_args):
        raise OSError("post-run failure")

    monkeypatch.setattr(evaluation, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(evaluation, "parse_answer_position", lambda *_args: ("Sheet1", "A1"))
    monkeypatch.setattr(
        evaluation, "recalculate", lambda *_args: SimpleNamespace(source="test")
    )
    monkeypatch.setattr(evaluation, "score_workbooks", fail)
    if failure_stage == "copy":
        monkeypatch.setattr(evaluation.shutil, "copy2", fail)

    task = SpreadsheetTask(
        task_id="task",
        instruction="Fill A1",
        instruction_type="edit",
        answer_position="A1",
        spreadsheet_dir=str(tmp_path),
        test_cases=(),
    )
    case = asyncio.run(
        evaluation._run_case(
            task,
            1,
            str(tmp_path / "input.xlsx"),
            str(tmp_path / "answer.xlsx"),
            object,
            object(),
            object(),
            object(),
            asyncio.Semaphore(1),
            str(tmp_path),
            EvalConfig(),
        )
    )

    assert not case.passed
    assert case.score == 0.0
    assert "post-run failure" in case.message
    _dump_eval_task_traces(tmp_path, [TaskResult("task", 0.0, 0, [case])])
    row = json.loads((tmp_path / "task_traces.jsonl").read_text())
    assert row["trace"] == trace.model_dump()
    assert row["evidence"] == evidence.model_dump()
