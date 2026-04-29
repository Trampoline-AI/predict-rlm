"""Tests for PredictRLM lifecycle callbacks.

Covers ``on_rlm_iteration_start`` and ``on_rlm_iteration_end`` handlers
dispatched from both the sync (``forward``) and async (``aforward``)
iteration loops. Uses a mocked interpreter and patched iteration helper
so no Deno sandbox is required — these tests run as pure unit tests.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import dspy
import pytest
from dspy.primitives.repl_types import REPLEntry, REPLHistory
from dspy.utils.callback import BaseCallback

from predict_rlm import IterationStep, PredictRLM

# --- Test signature & helpers ---------------------------------------------


class EchoSignature(dspy.Signature):
    """Echo the query."""

    query: str = dspy.InputField()
    answer: str = dspy.OutputField()


def _make_lm() -> MagicMock:
    """Build a MagicMock that satisfies snapshot_lm_history_len()."""
    lm = MagicMock(spec=dspy.LM)
    lm.history = []
    lm.model = "mock-lm"
    return lm


def _entry(reasoning: str = "r", code: str = "print(1)", output: str = "1") -> REPLEntry:
    return REPLEntry(reasoning=reasoning, code=code, output=output)


def _history_with(*entries: REPLEntry) -> REPLHistory:
    h = REPLHistory()
    for e in entries:
        h.entries.append(e)
    return h


def _final_prediction(answer: str = "ok") -> dspy.Prediction:
    pred = dspy.Prediction(answer=answer)
    # _execute_iteration's returned Prediction normally has a trajectory; the
    # loop reads .trajectory[-1] when it's a Prediction. Provide a minimal one.
    pred.trajectory = [{"reasoning": "done", "code": "SUBMIT(answer='ok')", "output": "(no output)"}]
    return pred


def _build_rlm(**kwargs) -> PredictRLM:
    """Build a PredictRLM with a mocked interpreter so the loop can run
    without spawning Deno."""
    rlm = PredictRLM(
        EchoSignature,
        interpreter=MagicMock(),
        max_iterations=5,
        **kwargs,
    )
    # Bypass the helpers that rely on real interpreter / signature plumbing.
    rlm._validate_inputs = MagicMock(return_value=None)  # type: ignore[method-assign]
    rlm._prepare_execution_tools = MagicMock(return_value={})  # type: ignore[method-assign]
    rlm._build_variables = MagicMock(return_value=[])  # type: ignore[method-assign]
    return rlm


def _drive_sync(rlm: PredictRLM, iteration_returns: list, fallback: dspy.Prediction | None = None):
    """Run rlm.forward() with patched _execute_iteration returning the
    given sequence of values. ``fallback`` is used by _extract_fallback if
    we exhaust max_iterations without a final Prediction."""
    fallback = fallback or _final_prediction(answer="fallback")
    with patch.object(rlm, "_execute_iteration", side_effect=iteration_returns), \
         patch.object(rlm, "_extract_fallback", return_value=fallback), \
         dspy.context(lm=_make_lm()):
        return rlm.forward(query="hi")


async def _drive_async(rlm: PredictRLM, iteration_returns: list, fallback: dspy.Prediction | None = None):
    fallback = fallback or _final_prediction(answer="fallback")
    aexec = AsyncMock(side_effect=iteration_returns)
    aextract = AsyncMock(return_value=fallback)
    with patch.object(rlm, "_aexecute_iteration", aexec), \
         patch.object(rlm, "_aextract_fallback", aextract), \
         dspy.context(lm=_make_lm()):
        return await rlm.aforward(query="hi")


# --- Recording callback ----------------------------------------------------


class RecordingCallback(BaseCallback):
    """Records every RLM iteration event for assertions."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def on_rlm_iteration_start(self, *, call_id, instance, iteration, max_iterations):
        self.events.append((
            "start",
            {"call_id": call_id, "iteration": iteration, "max_iterations": max_iterations},
        ))

    def on_rlm_iteration_end(self, *, call_id, instance, iteration, step, is_final, exception):
        self.events.append((
            "end",
            {
                "call_id": call_id,
                "iteration": iteration,
                "step": step,
                "is_final": is_final,
                "exception": exception,
            },
        ))


class AsyncRecordingCallback(BaseCallback):
    """Async variant — handlers return coroutines."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def on_rlm_iteration_start(self, *, call_id, instance, iteration, max_iterations):
        # await something to prove we're really async
        await asyncio.sleep(0)
        self.events.append(("start", {"iteration": iteration}))

    async def on_rlm_iteration_end(self, *, call_id, instance, iteration, step, is_final, exception):
        await asyncio.sleep(0)
        self.events.append(("end", {"iteration": iteration, "is_final": is_final}))


# --- Sync path tests -------------------------------------------------------


class TestSyncCallbacks:
    def test_events_fire_in_order_and_carry_step(self):
        cb = RecordingCallback()
        rlm = _build_rlm()
        rlm.callbacks = [cb]

        h1 = _history_with(_entry(reasoning="r1", code="x=1", output="1"))
        h2 = _history_with(_entry("r1", "x=1", "1"), _entry("r2", "x=2", "2"))
        final = _final_prediction()

        _drive_sync(rlm, [h1, h2, final])

        assert [name for name, _ in cb.events] == [
            "start", "end", "start", "end", "start", "end",
        ]
        ends = [payload for name, payload in cb.events if name == "end"]
        assert [e["iteration"] for e in ends] == [1, 2, 3]
        assert [e["is_final"] for e in ends] == [False, False, True]
        assert all(isinstance(e["step"], IterationStep) for e in ends)
        assert ends[0]["step"].reasoning == "r1"
        assert ends[1]["step"].code == "x=2"
        # Final iteration's step is built from the Prediction's trajectory.
        assert ends[2]["step"].reasoning == "done"

    def test_global_callback_via_dspy_settings(self):
        cb = RecordingCallback()
        rlm = _build_rlm()  # no instance-level callback
        with dspy.context(callbacks=[cb]):
            _drive_sync(rlm, [_final_prediction()])
        assert [name for name, _ in cb.events] == ["start", "end"]

    def test_instance_callback(self):
        cb = RecordingCallback()
        rlm = _build_rlm()
        rlm.callbacks = [cb]
        _drive_sync(rlm, [_final_prediction()])
        assert [name for name, _ in cb.events] == ["start", "end"]

    def test_no_callbacks_runs_clean(self):
        rlm = _build_rlm()
        result = _drive_sync(rlm, [_final_prediction(answer="ok")])
        assert result.answer == "ok"

    def test_handler_exception_is_isolated(self, caplog):
        class Boom(BaseCallback):
            def on_rlm_iteration_start(self, **_):
                raise RuntimeError("handler boom")
            def on_rlm_iteration_end(self, **_):
                raise RuntimeError("handler boom")

        rlm = _build_rlm()
        rlm.callbacks = [Boom()]
        with caplog.at_level(logging.WARNING, logger="predict_rlm.callbacks"):
            result = _drive_sync(rlm, [_final_prediction(answer="ok")])
        assert result.answer == "ok"
        assert any("handler boom" in rec.message for rec in caplog.records)

    def test_iteration_end_fires_with_exception_when_iteration_raises(self):
        cb = RecordingCallback()
        rlm = _build_rlm()
        rlm.callbacks = [cb]
        # First iteration raises before a step is built.
        with pytest.raises(RuntimeError, match="kaboom"):
            _drive_sync(rlm, [RuntimeError("kaboom")])
        assert [name for name, _ in cb.events] == ["start", "end"]
        end_payload = cb.events[-1][1]
        assert end_payload["step"] is None
        assert end_payload["is_final"] is False
        assert isinstance(end_payload["exception"], RuntimeError)

    def test_max_iterations_status_emits_no_final_flag(self):
        cb = RecordingCallback()
        rlm = _build_rlm()
        rlm.max_iterations = 2
        rlm.callbacks = [cb]
        h = _history_with(_entry())
        _drive_sync(rlm, [h, _history_with(_entry(), _entry())])
        ends = [p for n, p in cb.events if n == "end"]
        assert all(e["is_final"] is False for e in ends)
        assert len(ends) == 2

    def test_async_handler_in_sync_path_warns_and_skips(self, caplog):
        cb = AsyncRecordingCallback()
        rlm = _build_rlm()
        rlm.callbacks = [cb]
        with caplog.at_level(logging.WARNING, logger="predict_rlm.callbacks"):
            _drive_sync(rlm, [_final_prediction(answer="ok")])
        # Coroutines were never executed → no events recorded
        assert cb.events == []
        assert any("Async callback" in rec.message for rec in caplog.records)

    def test_basecallback_subclass_without_rlm_methods_is_noop(self):
        class OnlyLM(BaseCallback):
            def on_lm_start(self, **_): pass

        rlm = _build_rlm()
        rlm.callbacks = [OnlyLM()]
        # Must not raise.
        result = _drive_sync(rlm, [_final_prediction(answer="ok")])
        assert result.answer == "ok"


# --- Async path tests ------------------------------------------------------


class TestAsyncCallbacks:
    @pytest.mark.asyncio
    async def test_sync_handler_in_async_path(self):
        cb = RecordingCallback()
        rlm = _build_rlm()
        rlm.callbacks = [cb]
        await _drive_async(rlm, [_final_prediction(answer="ok")])
        assert [name for name, _ in cb.events] == ["start", "end"]

    @pytest.mark.asyncio
    async def test_async_handler_is_awaited(self):
        cb = AsyncRecordingCallback()
        rlm = _build_rlm()
        rlm.callbacks = [cb]
        h = _history_with(_entry())
        await _drive_async(rlm, [h, _final_prediction()])
        assert [name for name, _ in cb.events] == ["start", "end", "start", "end"]
        assert cb.events[-1][1]["is_final"] is True

    @pytest.mark.asyncio
    async def test_async_handler_exception_isolated(self, caplog):
        class AsyncBoom(BaseCallback):
            async def on_rlm_iteration_end(self, **_):
                raise RuntimeError("async boom")

        rlm = _build_rlm()
        rlm.callbacks = [AsyncBoom()]
        with caplog.at_level(logging.WARNING, logger="predict_rlm.callbacks"):
            result = await _drive_async(rlm, [_final_prediction(answer="ok")])
        assert result.answer == "ok"
        assert any("async boom" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_async_iteration_exception_still_emits_end(self):
        cb = RecordingCallback()
        rlm = _build_rlm()
        rlm.callbacks = [cb]
        with pytest.raises(RuntimeError, match="async kaboom"):
            await _drive_async(rlm, [RuntimeError("async kaboom")])
        assert [name for name, _ in cb.events] == ["start", "end"]
        assert isinstance(cb.events[-1][1]["exception"], RuntimeError)


# --- Multi-callback ordering ----------------------------------------------


class TestMultipleCallbacks:
    def test_global_and_instance_both_invoked(self):
        global_cb = RecordingCallback()
        instance_cb = RecordingCallback()
        rlm = _build_rlm()
        rlm.callbacks = [instance_cb]
        with dspy.context(callbacks=[global_cb]):
            _drive_sync(rlm, [_final_prediction()])
        # Both receive both events.
        assert [n for n, _ in global_cb.events] == ["start", "end"]
        assert [n for n, _ in instance_cb.events] == ["start", "end"]

    def test_one_handler_failing_does_not_block_others(self):
        good = RecordingCallback()

        class Bad(BaseCallback):
            def on_rlm_iteration_start(self, **_):
                raise RuntimeError("bad")

        rlm = _build_rlm()
        rlm.callbacks = [Bad(), good]
        _drive_sync(rlm, [_final_prediction()])
        assert [n for n, _ in good.events] == ["start", "end"]


# --- Integration test (real Deno sandbox) ---------------------------------


@pytest.mark.integration
class TestCallbacksIntegration:
    """Verifies the callback contract end-to-end against a real interpreter.

    The driving LM is mocked (no API calls), but the sandbox is real, so
    the IterationStep delivered to ``on_rlm_iteration_end`` contains the
    actual stdout produced by Pyodide.
    """

    def test_iteration_end_receives_real_sandbox_output(self):
        from predict_rlm.interpreter import JspiInterpreter

        cb = RecordingCallback()

        # Real interpreter — preinstall_packages=False keeps it fast.
        interpreter = JspiInterpreter(tools={}, preinstall_packages=False)
        try:
            rlm = PredictRLM(
                "query -> answer",
                interpreter=interpreter,
                max_iterations=3,
            )
            rlm.callbacks = [cb]

            scripted = [
                dspy.Prediction(reasoning="probe", code="print('hello-from-sandbox')"),
                dspy.Prediction(reasoning="finish", code="SUBMIT(answer='42')"),
            ]
            rlm.generate_action = MagicMock(side_effect=scripted)

            with dspy.context(lm=_make_lm()):
                result = rlm.forward(query="anything")

            assert result.answer == "42"
            ends = [p for n, p in cb.events if n == "end"]
            assert len(ends) == 2
            assert ends[0]["is_final"] is False
            assert "hello-from-sandbox" in ends[0]["step"].output
            assert ends[1]["is_final"] is True
        finally:
            interpreter.shutdown()
