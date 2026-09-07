"""Callback failures, async handlers, and real iteration output."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import dspy
import pytest
from dspy.primitives.repl_types import REPLEntry, REPLHistory
from dspy.utils.callback import BaseCallback

from predict_rlm import PredictRLM


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
    pred.trajectory = [
        {"reasoning": "done", "code": "SUBMIT(answer='ok')", "output": "(no output)"}
    ]
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


def _drive_sync(
    rlm: PredictRLM, iteration_returns: list, fallback: dspy.Prediction | None = None
):
    """Run rlm(...) with patched _execute_iteration returning the
    given sequence of values. ``fallback`` is used by _extract_fallback if
    we exhaust max_iterations without a final Prediction."""
    fallback = fallback or _final_prediction(answer="fallback")
    with (
        patch.object(rlm, "_execute_iteration", side_effect=iteration_returns),
        patch.object(rlm, "_extract_fallback", return_value=fallback),
        dspy.context(lm=_make_lm()),
    ):
        return rlm(query="hi")


async def _drive_async(
    rlm: PredictRLM, iteration_returns: list, fallback: dspy.Prediction | None = None
):
    fallback = fallback or _final_prediction(answer="fallback")
    aexec = AsyncMock(side_effect=iteration_returns)
    aextract = AsyncMock(return_value=fallback)
    with (
        patch.object(rlm, "_aexecute_iteration", aexec),
        patch.object(rlm, "_aextract_fallback", aextract),
        dspy.context(lm=_make_lm()),
    ):
        return await rlm.acall(query="hi")


# --- Recording callback ----------------------------------------------------


class RecordingCallback(BaseCallback):
    """Records every RLM iteration event for assertions."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def on_rlm_iteration_start(self, *, call_id, instance, iteration, max_iterations):
        self.events.append(
            (
                "start",
                {"call_id": call_id, "iteration": iteration, "max_iterations": max_iterations},
            )
        )

    def on_rlm_iteration_end(self, *, call_id, instance, iteration, step, is_final, exception):
        self.events.append(
            (
                "end",
                {
                    "call_id": call_id,
                    "iteration": iteration,
                    "step": step,
                    "is_final": is_final,
                    "exception": exception,
                },
            )
        )


class AsyncRecordingCallback(BaseCallback):
    """Async variant — handlers return coroutines."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def on_rlm_iteration_start(self, *, call_id, instance, iteration, max_iterations):
        # await something to prove we're really async
        await asyncio.sleep(0)
        self.events.append(("start", {"call_id": call_id, "iteration": iteration}))

    async def on_rlm_iteration_end(
        self, *, call_id, instance, iteration, step, is_final, exception
    ):
        await asyncio.sleep(0)
        self.events.append(
            (
                "end",
                {"call_id": call_id, "iteration": iteration, "is_final": is_final},
            )
        )


def _assert_shared_call_id(events: list[tuple[str, dict]]) -> None:
    call_ids = [payload["call_id"] for _, payload in events]
    assert all(call_ids)
    assert len(set(call_ids)) == 1


class TestSyncCallbacks:
    def test_iteration_end_fires_with_exception_when_iteration_raises(self):
        cb = RecordingCallback()
        rlm = _build_rlm()
        rlm.callbacks = [cb]
        # First iteration raises before a step is built.
        with pytest.raises(RuntimeError, match="kaboom"):
            _drive_sync(rlm, [RuntimeError("kaboom")])
        assert [name for name, _ in cb.events] == ["start", "end"]
        _assert_shared_call_id(cb.events)
        end_payload = cb.events[-1][1]
        assert end_payload["step"] is None
        assert end_payload["is_final"] is False
        assert isinstance(end_payload["exception"], RuntimeError)

    def test_async_handler_in_sync_path_warns_and_skips(self, caplog):
        cb = AsyncRecordingCallback()
        rlm = _build_rlm()
        rlm.callbacks = [cb]
        with caplog.at_level(logging.WARNING, logger="predict_rlm.callbacks"):
            _drive_sync(rlm, [_final_prediction(answer="ok")])
        # Coroutines were never executed → no events recorded
        assert cb.events == []
        assert any("Async callback" in rec.message for rec in caplog.records)


class TestAsyncCallbacks:
    @pytest.mark.asyncio
    async def test_async_handler_is_awaited(self):
        cb = AsyncRecordingCallback()
        rlm = _build_rlm()
        rlm.callbacks = [cb]
        h = _history_with(_entry())
        await _drive_async(rlm, [h, _final_prediction()])
        assert [name for name, _ in cb.events] == ["start", "end", "start", "end"]
        _assert_shared_call_id(cb.events)
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
        _assert_shared_call_id(cb.events)
        assert isinstance(cb.events[-1][1]["exception"], RuntimeError)


class TestMultipleCallbacks:
    def test_one_handler_failing_does_not_block_others(self):
        good = RecordingCallback()

        class Bad(BaseCallback):
            def on_rlm_iteration_start(self, **_):
                raise RuntimeError("bad")

        rlm = _build_rlm()
        rlm.callbacks = [Bad(), good]
        _drive_sync(rlm, [_final_prediction()])
        assert [n for n, _ in good.events] == ["start", "end"]


@pytest.mark.integration
class TestCallbacksIntegration:
    """Verifies the callback contract end-to-end against a real interpreter.

    The driving LM is mocked (no API calls), but the sandbox is real, so
    the IterationStep delivered to ``on_rlm_iteration_end`` contains the
    actual stdout produced by Pyodide.
    """

    def test_iteration_end_receives_real_sandbox_output(self):
        from predict_rlm.backends import JspiBackend

        cb = RecordingCallback()

        # Real interpreter — preinstall_packages=False keeps it fast.
        interpreter = JspiBackend(tools={}, preinstall_packages=False)
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
                result = rlm(query="anything")

            assert result.answer == "42"
            ends = [p for n, p in cb.events if n == "end"]
            assert len(ends) == 2
            assert ends[0]["is_final"] is False
            assert "hello-from-sandbox" in ends[0]["step"].output
            assert ends[1]["is_final"] is True
        finally:
            interpreter.shutdown()
