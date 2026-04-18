"""Regression test: proposer errors must persist an ERROR trace file.

Before this fix, ``_propose_one_component`` only wrote the proposer trace on
the success path. If ``predictor.acall`` raised (sandbox timeout, gemini API
error, empty-output RuntimeError, etc.), the exception propagated to GEPA —
which recorded 0.0 minibatch scores — and no artifact was left behind to
diagnose the failure. The proposer's PredictRLM already attaches
``exc.trace = self._build_run_trace(status='error', ...)`` on its own
internal errors, so the information exists; it just wasn't being persisted.

This test pins the contract: when the proposer errors, an ERROR trace file
lands in ``proposer_trace_dir`` with the exception message and (when
available) the partial RunTrace.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


class _FakeTokenUsage:
    def __init__(self, input_tokens=0, output_tokens=0, cost=0.0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost = cost
        self.cache_hits = 0


class _FakeLMUsage:
    def __init__(self):
        self.main = _FakeTokenUsage(input_tokens=50, output_tokens=5, cost=0.001)
        self.sub = None


class _FakeRunTrace:
    """Minimal stand-in for predict_rlm.trace.RunTrace on the error path."""

    def __init__(self, status: str = "error"):
        self.status = status
        self.model = "gemini/gemini-3.1-pro-preview"
        self.sub_model = None
        self.iterations = 2
        self.max_iterations = 15
        self.duration_ms = 1234
        self.usage = _FakeLMUsage()
        self.steps = []

    def to_exportable_json(self, indent: int = 0) -> str:
        return json.dumps(
            {
                "status": self.status,
                "model": self.model,
                "iterations": self.iterations,
                "duration_ms": self.duration_ms,
            },
            indent=indent,
        )


def _make_adapter(tmp_path: Path):
    """Build a SpreadsheetAdapter instance without triggering its heavy init."""
    from lib.optimize import SpreadsheetAdapter

    adapter = SpreadsheetAdapter.__new__(SpreadsheetAdapter)
    adapter._proposer_call_count = 0
    adapter.proposer_trace_dir = tmp_path
    adapter.cost_log_path = tmp_path / "cost_log.jsonl"
    adapter.proposer_lm = object()
    adapter.proposer_sub_lm = None
    adapter.proposer_max_iterations = 15
    adapter.use_optimize_gen = False
    return adapter


def _mock_predictor_raising(exc: BaseException):
    """Build a PredictRLM stand-in whose acall raises when awaited."""
    from unittest.mock import MagicMock

    async def _acall(**_kw):
        raise exc

    predictor = MagicMock()
    predictor.acall = _acall
    return predictor


def test_proposer_error_writes_error_trace_file(tmp_path):
    adapter = _make_adapter(tmp_path)

    exc = RuntimeError("gemini rate limit hit")
    exc.trace = _FakeRunTrace(status="error")
    predictor = _mock_predictor_raising(exc)

    # Patch PredictRLM inside the optimize module so we don't spin up a real
    # proposer — this test is about what happens AFTER acall raises.
    with patch("lib.optimize.PredictRLM") as mock_cls:
        mock_cls.return_value = predictor

        import pytest

        with pytest.raises(RuntimeError, match="gemini rate limit"):
            adapter._propose_one_component(
                "skill_instructions",
                current_text="# seed skill\n- rule 1",
                records=[{"Inputs": "i", "Generated Outputs": "g", "Feedback": "f"}],
            )

    # An ERROR trace file must exist in proposer_trace_dir
    error_files = list(tmp_path.glob("proposer_*_ERROR.json"))
    assert len(error_files) == 1, (
        f"expected 1 ERROR trace file, got {len(error_files)}: "
        f"{[f.name for f in error_files]}"
    )

    payload = json.loads(error_files[0].read_text())
    # Contract: error file captures the call index, component, error message,
    # and (when available) the partial RunTrace from exc.trace.
    assert payload["call_idx"] == 1
    assert payload["component"] == "skill_instructions"
    assert payload["status"] == "error"
    assert "gemini rate limit" in payload["error"]
    assert payload["run_trace"] is not None
    assert payload["run_trace"]["status"] == "error"


def test_proposer_error_emits_cost_log_event(tmp_path):
    """proposer errors should also land in cost_log.jsonl so the iter/$ table
    shows the failed attempt instead of silently skipping it.
    """
    adapter = _make_adapter(tmp_path)

    exc = RuntimeError("sandbox timeout after 120s (proposer iter 7)")
    exc.trace = _FakeRunTrace(status="error")
    predictor = _mock_predictor_raising(exc)

    with patch("lib.optimize.PredictRLM") as mock_cls:
        mock_cls.return_value = predictor

        import pytest

        with pytest.raises(RuntimeError):
            adapter._propose_one_component(
                "skill_instructions",
                current_text="# seed",
                records=[],
            )

    rows = [
        json.loads(ln)
        for ln in (tmp_path / "cost_log.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    proposer_error_rows = [r for r in rows if r.get("event") == "proposer_error"]
    assert len(proposer_error_rows) == 1
    assert proposer_error_rows[0]["proposer_call_idx"] == 1
    assert proposer_error_rows[0]["component"] == "skill_instructions"
    assert "sandbox timeout" in proposer_error_rows[0]["error"]


def test_proposer_error_without_trace_still_persisted(tmp_path):
    """Some errors (e.g. an exception raised BEFORE PredictRLM attaches a
    trace) won't have ``exc.trace``. The ERROR file should still be written —
    with ``run_trace=None`` — so we don't lose visibility on these failures.
    """
    adapter = _make_adapter(tmp_path)

    exc = ValueError("empty traces file")  # no .trace attribute
    predictor = _mock_predictor_raising(exc)

    with patch("lib.optimize.PredictRLM") as mock_cls:
        mock_cls.return_value = predictor

        import pytest

        with pytest.raises(ValueError, match="empty traces"):
            adapter._propose_one_component("skill_instructions", "seed", [])

    files = list(tmp_path.glob("proposer_*_ERROR.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["run_trace"] is None
    assert "empty traces" in payload["error"]


def test_proposer_timeout_is_tracked(tmp_path):
    adapter = _make_adapter(tmp_path)
    # Force an explicitly low configured timeout so we can assert the value used
    # is clamped and passed through to asyncio.wait_for.
    adapter.proposer_timeout = 0

    async def _acall(**_kw):
        return {
            "new_instructions": "should not be returned"
        }

    async def _mock_wait_for(awaitable, timeout):
        # Simulate a hard timeout without waiting in real time.
        captured["timeout"] = timeout
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise asyncio.TimeoutError("forced timeout")

    captured: dict[str, object] = {}

    with patch("lib.optimize.PredictRLM") as mock_cls, patch(
        "lib.optimize.asyncio.wait_for", new=_mock_wait_for
    ):
        from unittest.mock import MagicMock

        predictor = MagicMock()
        predictor.acall = _acall
        mock_cls.return_value = predictor

        import pytest

        with pytest.raises(asyncio.TimeoutError):
            adapter._propose_one_component(
                "skill_instructions",
                current_text="# seed",
                records=[],
            )

    assert captured["timeout"] == 1
    error_files = list(tmp_path.glob("proposer_*_ERROR.json"))
    assert len(error_files) == 1
    payload = json.loads(error_files[0].read_text())
    assert payload["error_type"] == "TimeoutError"
    assert payload["component"] == "skill_instructions"


def test_proposer_timeout_recovers_trace_from_chain(tmp_path):
    """When a proposer times out, the CancelledError that PredictRLM
    caught carries the partial RunTrace via ``exc.trace``. The outer
    TimeoutError raised by asyncio.wait_for holds it on
    ``__context__``. ``_persist_proposer_error`` walks the chain via
    ``extract_trace_from_exc`` so the ERROR file AND the cost_log
    event both capture the partial cost accumulated before cancellation.
    """
    adapter = _make_adapter(tmp_path)
    adapter.proposer_timeout = 5

    # Build a TimeoutError whose __context__ is a CancelledError
    # carrying a sentinel .trace (what PredictRLM would attach).
    fake_trace = _FakeRunTrace(status="error")
    inner = asyncio.CancelledError()
    inner.trace = fake_trace  # type: ignore[attr-defined]

    outer = asyncio.TimeoutError("wait_for budget exhausted")
    outer.__context__ = inner

    async def _raises_outer(**_kw):
        raise outer

    async def _mock_wait_for(awaitable, timeout):
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise outer

    from unittest.mock import MagicMock

    predictor = MagicMock()
    predictor.acall = _raises_outer
    with patch("lib.optimize.PredictRLM") as mock_cls, patch(
        "lib.optimize.asyncio.wait_for", new=_mock_wait_for
    ):
        mock_cls.return_value = predictor

        import pytest

        with pytest.raises(asyncio.TimeoutError):
            adapter._propose_one_component(
                "skill_instructions",
                current_text="# seed",
                records=[],
            )

    # ERROR file carries the recovered RunTrace (not None)
    error_files = list(tmp_path.glob("proposer_*_ERROR.json"))
    assert len(error_files) == 1
    payload = json.loads(error_files[0].read_text())
    assert payload["run_trace"] is not None, (
        "extract_trace_from_exc should have walked __context__ and "
        "pulled the partial RunTrace the inner CancelledError carried"
    )
    assert payload["run_trace"]["status"] == "error"

    # cost_log event for the timeout includes the partial token/cost
    # pulled from trace.usage.main (0.001 by _FakeLMUsage default)
    rows = [
        json.loads(ln)
        for ln in (tmp_path / "cost_log.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    proposer_error_rows = [r for r in rows if r.get("event") == "proposer_error"]
    assert len(proposer_error_rows) == 1
    assert proposer_error_rows[0]["input_tokens"] == 50
    assert proposer_error_rows[0]["output_tokens"] == 5
    assert proposer_error_rows[0]["cost_usd"] == pytest.approx(0.001)


def test_adapter_resumes_counters_from_existing_artifacts(tmp_path):
    from lib.optimize import SpreadsheetAdapter

    run_dir = tmp_path / "run"
    proposer_dir = run_dir / "proposer_traces"
    proposer_dir.mkdir(parents=True)
    (proposer_dir / "proposer_0003_skill_instructions.json").write_text("{}")
    (proposer_dir / "proposer_0017_skill_instructions_ERROR.json").write_text("{}")

    task_trace_dir = run_dir / "task_traces"
    task_trace_dir.mkdir()
    (task_trace_dir / "minibatch_0005.jsonl").write_text("{}\n")
    (task_trace_dir / "valset_0002.jsonl").write_text("{}\n")

    cost_log = run_dir / "cost_log.jsonl"
    cost_log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "proposer_error",
                        "proposer_call_idx": 20,
                        "component": "skill_instructions",
                    }
                ),
                json.dumps({"event": "minibatch", "evaluate_idx": 7}),
                json.dumps({"event": "valset", "evaluate_idx": 11}),
            ]
        )
        + "\n"
    )

    adapter = SpreadsheetAdapter.__new__(SpreadsheetAdapter)
    adapter._proposer_call_count = 0
    adapter._minibatch_count = -1
    adapter._valset_count = -1
    adapter.proposer_trace_dir = str(proposer_dir)
    adapter.cost_log_path = cost_log
    adapter.task_trace_dir = task_trace_dir

    adapter._resume_from_artifacts()

    assert adapter._proposer_call_count == 20
    assert adapter._minibatch_count == 7
    assert adapter._valset_count == 11


def test_missing_artifacts_keep_default_start_indices(tmp_path):
    from lib.optimize import SpreadsheetAdapter

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    adapter = SpreadsheetAdapter.__new__(SpreadsheetAdapter)
    adapter._proposer_call_count = 0
    adapter._minibatch_count = -1
    adapter._valset_count = -1
    adapter.proposer_trace_dir = str(run_dir / "proposer_traces")
    adapter.cost_log_path = run_dir / "cost_log.jsonl"
    adapter.task_trace_dir = run_dir / "task_traces"

    adapter._resume_from_artifacts()

    assert adapter._proposer_call_count == 0
    assert adapter._minibatch_count == -1
    assert adapter._valset_count == -1
