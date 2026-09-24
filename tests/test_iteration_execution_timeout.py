"""Selected iteration deadlines preserve state; failed recovery is bounded."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import dspy
import pytest


class _SequentialActions:
    def __init__(self, *actions: SimpleNamespace) -> None:
        self.actions = list(actions)
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        assert self.actions, "PredictRLM requested more actions than the test provided"
        return self.actions.pop(0)


class _PredictionStub:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def keys(self) -> list[str]:
        return ["answer"]

    def __getitem__(self, key: str) -> str:
        return getattr(self, key)


@pytest.mark.asyncio
async def test_jspi_silent_iteration_timeout_recovery_failure_is_bounded(monkeypatch):
    import predict_rlm.execution_timeout as execution_timeout
    from predict_rlm.backends import JspiBackend
    from predict_rlm.backends.base import SandboxFatalError
    from predict_rlm.execution_timeout import ITERATION_TIMEOUT_FAILURE_CLASS

    monkeypatch.setattr(
        execution_timeout,
        "DEFAULT_RECOVERABLE_EXECUTION_TIMEOUT_GRACE_SECONDS",
        0.2,
    )
    interpreter = JspiBackend.__new__(JspiBackend)
    spans = []
    killed = []
    interpreter._write_telemetry_span = lambda name, **kwargs: spans.append(
        {"name": name, **kwargs}
    )
    interpreter._telemetry_pending_tool_count = lambda: 0
    interpreter._telemetry_pending_file_ops_count = lambda: 0

    async def _kill_sandbox():
        killed.append(True)

    interpreter._akill_sandbox = _kill_sandbox

    async def _silent_execute(_request_id):
        await asyncio.sleep(30)

    interpreter._execute_async = _silent_execute

    start = asyncio.get_running_loop().time()
    with pytest.raises(SandboxFatalError, match="failed to recover"):
        await interpreter._execute_with_timeout(
            8,
            timeout_seconds=0.1,
            timeout_failure_class=ITERATION_TIMEOUT_FAILURE_CLASS,
        )

    elapsed = asyncio.get_running_loop().time() - start
    assert 0.25 <= elapsed < 1.0
    assert killed == [True]
    assert any(
        span["name"] == "sandbox.execute.timeout"
        and span["attributes"]["timeout.recovery_failed"] is True
        for span in spans
    )


@pytest.mark.integration
def test_predict_rlm_jspi_timeout_preserves_state_history_and_predict_tool(monkeypatch):
    from functools import partial

    from predict_rlm import PredictRLM
    from predict_rlm.backends import JspiBackend
    from predict_rlm.predict_rlm import dspy

    monkeypatch.setattr(
        "predict_rlm.backends.jspi.execution.JspiBackend",
        partial(JspiBackend, preinstall_packages=False),
    )

    actions = _SequentialActions(
        SimpleNamespace(
            reasoning="prepare state before the bounded operation",
            code=(
                "first = await predict('question: str -> answer: str', "
                "question='first call')\n"
                "saved = {'first': first['answer'], 'marker': 123}\n"
            ),
        ),
        SimpleNamespace(
            reasoning="run a bounded risky loop using prepared state",
            code=(
                "print('first predict:', saved['first'])\n"
                "print('marker before timeout:', saved['marker'])\n"
                "while True:\n"
                "    pass\n"
            ),
            execution_timeout_seconds=0.2,
        ),
        SimpleNamespace(
            reasoning="continue with preserved state and call predict again",
            code=(
                "print('marker after timeout:', saved['marker'])\n"
                "second = await predict('question: str -> answer: str', "
                "question='second call')\n"
                "SUBMIT(answer=f\"{saved['first']} -> {second['answer']} / {saved['marker']}\")"
            ),
        ),
    )
    mock_lm = MagicMock()
    mock_predictor = MagicMock()
    mock_predictor.acall = AsyncMock(
        side_effect=[
            _PredictionStub("pre-timeout prediction"),
            _PredictionStub("post-timeout prediction"),
        ]
    )
    rlm = PredictRLM(
        "prompt -> answer",
        sub_lm=mock_lm,
        max_iterations=3,
        sandbox_backend="jspi",
    )
    rlm.generate_action = actions

    with patch.object(dspy, "Predict", return_value=mock_predictor):
        prediction = rlm(prompt="exercise deno timeout recovery")

    assert prediction.answer == "pre-timeout prediction -> post-timeout prediction / 123"
    assert len(prediction.trace.steps) == 3
    _, timeout_step, final_step = prediction.trace.steps
    assert (
        "[Timeout] Iteration execution timed out after 0.2s" in timeout_step.untruncated_output
    )
    assert "first predict: pre-timeout prediction" in timeout_step.untruncated_output
    assert "marker before timeout: 123" in timeout_step.untruncated_output
    assert final_step.output == (
        "FINAL: {'answer': 'pre-timeout prediction -> post-timeout prediction / 123'}"
    )
    second_history = str(actions.calls[2]["repl_history"])
    assert "[Timeout] Iteration execution timed out after 0.2s" in second_history
    assert "first predict: pre-timeout prediction" in second_history


@pytest.mark.integration
def test_jspi_timeout_during_async_sleep_preserves_state_and_recovers(monkeypatch):
    import predict_rlm.execution_timeout as execution_timeout
    from predict_rlm.backends import JspiBackend
    from predict_rlm.execution_timeout import RecoverableExecutionTimeout

    monkeypatch.setattr(
        execution_timeout,
        "DEFAULT_RECOVERABLE_EXECUTION_TIMEOUT_GRACE_SECONDS",
        2,
    )
    interpreter = JspiBackend(preinstall_packages=False)
    try:
        interpreter.execute(
            "import asyncio, signal\nsaved = 42\n"
            "previous_sigint = signal.getsignal(signal.SIGINT)"
        )
        result = interpreter.execute(
            "print('before sleep')\nawait asyncio.sleep(0.3)",
            timeout=0.05,
        )
        assert isinstance(result, RecoverableExecutionTimeout)
        assert result.stdout == "before sleep\n"
        assert (
            interpreter.execute(
                "assert signal.getsignal(signal.SIGINT) is previous_sigint\nprint(saved)"
            )
            == "42\n"
        )
    finally:
        interpreter.shutdown()


@pytest.mark.integration
@pytest.mark.parametrize("definition", ["exec", "module", "async-generator"])
def test_jspi_timeout_interrupts_child_code_from_other_filenames(monkeypatch, definition):
    import predict_rlm.execution_timeout as execution_timeout
    from predict_rlm.backends import JspiBackend
    from predict_rlm.execution_timeout import RecoverableExecutionTimeout

    monkeypatch.setattr(
        execution_timeout,
        "DEFAULT_RECOVERABLE_EXECUTION_TIMEOUT_GRACE_SECONDS",
        2,
    )
    if definition == "async-generator":
        source = "async def spin():\n    yield 1\n    while True: pass"
        setup = f"exec({source!r})"
        operation = (
            "async def consume():\n"
            "    async for value in spin():\n"
            "        pass\n"
            "await asyncio.gather(consume())"
        )
    else:
        source = "async def spin():\n    while True: pass"
        if definition == "module":
            setup = (
                "import sys\nsys.path.insert(0, '/tmp')\n"
                f"with open('/tmp/timeout_child.py', 'w') as module:\n"
                f"    module.write({source!r})\n"
                "from timeout_child import spin"
            )
        else:
            setup = f"exec({source!r})"
        operation = "await asyncio.gather(spin())"
    interpreter = JspiBackend(preinstall_packages=False)
    try:
        interpreter.execute(
            "import asyncio, signal\nsaved = 42\n"
            "previous_sigint = signal.getsignal(signal.SIGINT)\n" + setup
        )
        result = interpreter.execute("print('before child')\n" + operation, timeout=0.1)
        assert isinstance(result, RecoverableExecutionTimeout)
        assert result.stdout == "before child\n"
        assert (
            interpreter.execute(
                "assert signal.getsignal(signal.SIGINT) is previous_sigint\nprint(saved)"
            )
            == "42\n"
        )
    finally:
        interpreter.shutdown()


@pytest.mark.asyncio
async def test_nonfinite_model_deadline_fails_before_execution():
    from dspy.primitives.repl_types import REPLHistory

    from predict_rlm import PredictRLM

    rlm = PredictRLM("question -> answer", model_execution_timeout=True)
    rlm.generate_action = MagicMock()
    rlm.generate_action.acall = AsyncMock(
        return_value=dspy.Prediction(
            reasoning="invalid cap",
            code="while True: pass",
            execution_timeout_seconds=float("nan"),
        )
    )
    repl = MagicMock()
    repl.aexecute = AsyncMock(side_effect=AssertionError("unbounded code executed"))
    with pytest.raises(RuntimeError, match="invalid execution_timeout_seconds"):
        await rlm._aexecute_iteration(repl, [], REPLHistory(), 0, {}, ["answer"])
