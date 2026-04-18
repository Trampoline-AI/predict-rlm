"""Task-trace JSONL files grow incrementally as cases finish.

Before: ``_dump_task_traces`` wrote one file per evaluate AFTER all
cases completed — on a 100-case val eval at concurrency=30 that meant
waiting ~5-10 min for any on-disk artifact.

After: ``_write_case_trace_row`` opens the file at evaluate start and
appends one JSONL line as each case finishes (via
``asyncio.as_completed``). Tail the file to watch progress live.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def _make_task(task_id: str = "T1"):
    # Minimal stub with the fields _write_case_trace_row reads. Avoids
    # depending on SpreadsheetTask's full init signature.
    from types import SimpleNamespace

    return SimpleNamespace(
        task_id=task_id,
        instruction_type="Cell-Level Manipulation",
    )


def _make_case(idx: int, score: float = 0.5):
    return {
        "idx": idx,
        "score": score,
        "passed": score == 1.0,
        "message": f"case {idx} message",
        "run_trace": None,  # keeps the test isolated from RunTrace shape
    }


def test_write_case_trace_row_appends_one_jsonl_line():
    from lib.optimize import SpreadsheetAdapter

    adapter = SpreadsheetAdapter.__new__(SpreadsheetAdapter)
    task = _make_task()

    buf = io.StringIO()
    adapter._write_case_trace_row(buf, task, _make_case(idx=0, score=1.0))
    adapter._write_case_trace_row(buf, task, _make_case(idx=1, score=0.0))

    lines = buf.getvalue().splitlines()
    assert len(lines) == 2
    r0 = json.loads(lines[0])
    r1 = json.loads(lines[1])
    assert r0["task_id"] == "T1" and r0["case_idx"] == 0 and r0["passed"] is True
    assert r1["task_id"] == "T1" and r1["case_idx"] == 1 and r1["passed"] is False
    # instruction_type is lifted from the task (not the case) so each
    # row is self-describing even without the parent task context.
    assert r0["instruction_type"] == "Cell-Level Manipulation"


def test_trace_row_handles_missing_run_trace_gracefully():
    from lib.optimize import SpreadsheetAdapter

    adapter = SpreadsheetAdapter.__new__(SpreadsheetAdapter)
    buf = io.StringIO()
    adapter._write_case_trace_row(
        buf, _make_task(), {"idx": 0, "score": 0.0, "passed": False, "message": "crash"}
    )
    row = json.loads(buf.getvalue())
    assert row["trace"] is None
    # Crash reason preserved for postmortem
    assert row["message"] == "crash"


def test_case_trace_flushes_so_tail_sees_partial_writes(tmp_path):
    """If we're going to claim 'tail the file to watch progress', then
    after each write the flush MUST hit the OS — otherwise buffering
    defeats the purpose. Confirm the file is readable after each row.
    """
    from lib.optimize import SpreadsheetAdapter

    adapter = SpreadsheetAdapter.__new__(SpreadsheetAdapter)
    trace_path = tmp_path / "evaluate_0001.jsonl"
    with trace_path.open("w") as f:
        adapter._write_case_trace_row(f, _make_task(), _make_case(idx=0, score=0.5))
        # Read from a SECOND file handle — we should see the first row
        # without closing the writer.
        mid = trace_path.read_text()
        assert mid.strip(), "row wasn't flushed to disk before second write"
        assert json.loads(mid.strip())["case_idx"] == 0

        adapter._write_case_trace_row(f, _make_task(), _make_case(idx=1, score=1.0))

    final = trace_path.read_text().splitlines()
    assert len(final) == 2


def test_multiple_tasks_in_same_file_each_self_describing(tmp_path):
    """One task_traces/evaluate_NNNN.jsonl spans ALL cases from ALL
    tasks in the batch. Each row is keyed by (task_id, case_idx) so
    downstream readers can regroup without needing the original batch.
    """
    from lib.optimize import SpreadsheetAdapter

    adapter = SpreadsheetAdapter.__new__(SpreadsheetAdapter)
    trace_path = tmp_path / "evaluate_0001.jsonl"
    task_a = _make_task("A")
    task_b = _make_task("B")
    with trace_path.open("w") as f:
        # Simulate interleaved arrival under asyncio.as_completed
        adapter._write_case_trace_row(f, task_a, _make_case(idx=0))
        adapter._write_case_trace_row(f, task_b, _make_case(idx=0))
        adapter._write_case_trace_row(f, task_a, _make_case(idx=1))

    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [(r["task_id"], r["case_idx"]) for r in rows] == [
        ("A", 0), ("B", 0), ("A", 1),
    ]
