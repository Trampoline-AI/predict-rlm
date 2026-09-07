"""Cancellation preserves completed and pending work without masking the cause."""

import asyncio
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import dspy
import pytest

from predict_rlm import PredictRLM


def _cancelling_run(error):
    class Repl:
        def execute(self, code, variables=None, timeout=None):
            if code == "await slow_tool()":
                raise error
            return "committed output"

        async def aexecute(self, code, variables=None, timeout=None):
            return self.execute(code, variables, timeout)

    @contextmanager
    def session(**kwargs):
        yield Repl()

    rlm = PredictRLM("query -> answer", max_iterations=3)
    actions = [
        dspy.Prediction(reasoning="first", code="print('committed output')"),
        dspy.Prediction(reasoning="pending", code="await slow_tool()"),
    ]
    rlm.generate_action = MagicMock(side_effect=actions)
    rlm.generate_action.acall = AsyncMock(side_effect=actions)
    rlm._interpreter_context = session
    return rlm


def _assert_partial_trace(error):
    assert error.trace.status == "error"
    assert [step.code for step in error.trace.steps] == [
        "print('committed output')",
        "await slow_tool()",
    ]
    assert error.trace.steps[0].untruncated_output == "committed output"
    assert error.trace.steps[1].error is True


def test_keyboard_interrupt_preserves_completed_and_pending_steps():
    error = KeyboardInterrupt("stop")
    rlm = _cancelling_run(error)
    with pytest.raises(KeyboardInterrupt) as caught:
        rlm._forward_traced(None, query="work")
    assert caught.value is error
    _assert_partial_trace(error)


@pytest.mark.asyncio
async def test_async_cancellation_preserves_completed_and_pending_steps():
    error = asyncio.CancelledError("stop")
    rlm = _cancelling_run(error)
    with pytest.raises(asyncio.CancelledError) as caught:
        await rlm._aforward_traced(None, query="work")
    assert caught.value is error
    _assert_partial_trace(error)


@pytest.mark.asyncio
async def test_trace_build_failure_does_not_replace_cancellation():
    error = asyncio.CancelledError("stop")
    rlm = _cancelling_run(error)
    build_trace = rlm._build_run_trace

    def fail_error_trace(*args, **kwargs):
        if kwargs.get("status") == "error":
            raise RuntimeError("trace unavailable")
        return build_trace(*args, **kwargs)

    with patch.object(rlm, "_build_run_trace", side_effect=fail_error_trace):
        with pytest.raises(asyncio.CancelledError) as caught:
            await rlm._aforward_traced(None, query="work")
    assert caught.value is error
