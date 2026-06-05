from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import dspy
import pytest


class _FakeRepl:
    def __init__(self):
        self.calls = []

    def execute(self, code, variables=None, timeout=None):
        self.calls.append({"code": code, "variables": variables, "timeout": timeout})
        return "[Success] ok"

    async def aexecute(self, code, variables=None, timeout=None):
        self.calls.append({"code": code, "variables": variables, "timeout": timeout})
        return "[Success] ok"


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


def _build_executor():
    from predict_rlm.predict_rlm import PredictRLM

    executor = PredictRLM.__new__(PredictRLM)
    executor.signature = dspy.Signature("question -> answer")
    executor.max_iterations = 3
    executor.verbose = False
    executor._user_tools = {}
    executor.generate_action = MagicMock()
    executor._partial_pending_entry = None
    executor._partial_history = None
    executor._partial_pending_start = None
    def _process_execution_result(pred, *args):
        result = args[-3]
        return {
            "result": result,
            "pred_code": getattr(pred, "code", None),
        }

    executor._process_execution_result = _process_execution_result
    return executor


def test_missing_action_timeout_preserves_existing_execution_call():
    executor = _build_executor()
    executor.generate_action.acall = AsyncMock(
        return_value=SimpleNamespace(reasoning="run it", code="print('ok')")
    )
    repl = _FakeRepl()

    async def _run():
        return await executor._aexecute_iteration(
            repl,
            variables=[],
            history=MagicMock(),
            iteration=0,
            input_args={"question": "q"},
            output_field_names=["answer"],
        )

    result = asyncio.run(_run())

    assert result["result"] == "[Success] ok"
    assert repl.calls == [
        {
            "code": "print('ok')",
            "variables": {"question": "q"},
            "timeout": None,
        }
    ]


def test_positive_action_timeout_is_passed_to_execution():
    executor = _build_executor()
    executor.generate_action.acall = AsyncMock(
        return_value=SimpleNamespace(
            reasoning="run with a cap",
            code="print('ok')",
            execution_timeout_seconds=2.5,
        )
    )
    repl = _FakeRepl()

    async def _run():
        return await executor._aexecute_iteration(
            repl,
            variables=[],
            history=MagicMock(),
            iteration=0,
            input_args={},
            output_field_names=["answer"],
        )

    asyncio.run(_run())

    assert repl.calls[0]["timeout"] == 2.5


def test_positive_action_timeout_is_passed_to_sync_execution():
    executor = _build_executor()
    executor.generate_action = MagicMock(
        return_value=SimpleNamespace(
            reasoning="run with a cap",
            code="print('ok')",
            execution_timeout_seconds=3,
        )
    )
    repl = _FakeRepl()

    executor._execute_iteration(
        repl,
        variables=[],
        history=MagicMock(),
        iteration=0,
        input_args={},
        output_field_names=["answer"],
    )

    assert repl.calls[0]["timeout"] == 3.0


@pytest.mark.parametrize("timeout_value", [None, 1, 2.5])
def test_action_timeout_accepts_null_and_finite_positive_values(timeout_value):
    executor = _build_executor()
    pred_kwargs = {"reasoning": "run it", "code": "print('ok')"}
    if timeout_value is not None:
        pred_kwargs["execution_timeout_seconds"] = timeout_value
    pred = SimpleNamespace(**pred_kwargs)

    result = executor._action_execution_timeout(pred)

    assert result == (None if timeout_value is None else float(timeout_value))


def test_action_timeout_uses_shared_validation_helper(monkeypatch):
    from predict_rlm import predict_rlm

    calls = []

    def fake_validate_execution_timeout(value):
        calls.append(value)
        return 4.0

    monkeypatch.setattr(
        predict_rlm,
        "validate_execution_timeout",
        fake_validate_execution_timeout,
    )
    executor = _build_executor()
    pred = SimpleNamespace(
        reasoning="run it",
        code="print('ok')",
        execution_timeout_seconds=4,
    )

    assert executor._action_execution_timeout(pred) == 4.0
    assert calls == [4]


@pytest.mark.parametrize(
    "timeout_value",
    [True, False, "2", 0, -1, float("nan"), float("inf"), -float("inf")],
)
def test_invalid_action_timeout_fails_before_execution(timeout_value):
    executor = _build_executor()
    executor.generate_action.acall = AsyncMock(
        return_value=SimpleNamespace(
            reasoning="bad cap",
            code="print('ok')",
            execution_timeout_seconds=timeout_value,
        )
    )
    repl = _FakeRepl()

    async def _run():
        return await executor._aexecute_iteration(
            repl,
            variables=[],
            history=MagicMock(),
            iteration=0,
            input_args={},
            output_field_names=["answer"],
        )

    with pytest.raises(RuntimeError, match="invalid execution_timeout_seconds"):
        asyncio.run(_run())
    assert repl.calls == []


@pytest.mark.parametrize(
    ("timeout_value", "expected_log"),
    [(None, "Execution timeout: null"), (2.5, "Execution timeout: 2.5s")],
)
def test_verbose_iteration_log_includes_execution_timeout(timeout_value, expected_log, capsys):
    executor = _build_executor()
    executor.verbose = True
    pred_kwargs = {"reasoning": "run it", "code": "print('ok')"}
    if timeout_value is not None:
        pred_kwargs["execution_timeout_seconds"] = timeout_value
    executor.generate_action.acall = AsyncMock(return_value=SimpleNamespace(**pred_kwargs))
    repl = _FakeRepl()

    async def _run():
        return await executor._aexecute_iteration(
            repl,
            variables=[],
            history=MagicMock(),
            iteration=1,
            input_args={},
            output_field_names=["answer"],
        )

    asyncio.run(_run())

    captured = capsys.readouterr()
    assert expected_log in captured.err


@pytest.mark.asyncio
async def test_jspi_per_iteration_timeout_has_recoverable_host_grace():
    from predict_rlm.execution_timeout import (
        DEFAULT_RECOVERABLE_EXECUTION_TIMEOUT_GRACE_SECONDS,
        ITERATION_TIMEOUT_FAILURE_CLASS,
        format_recoverable_timeout_result,
        recoverable_timeout_host_deadline_seconds,
    )
    from predict_rlm.interpreter import JspiInterpreter

    interpreter = JspiInterpreter.__new__(JspiInterpreter)
    spans = []
    killed = []
    interpreter._write_telemetry_span = lambda name, **kwargs: spans.append(
        {"name": name, **kwargs}
    )
    interpreter._telemetry_pending_tool_count = lambda: 0
    interpreter._telemetry_pending_file_ops_count = lambda: 0
    interpreter._kill_sandbox = lambda: killed.append(True)

    async def _slow_execute(_request_id):
        await asyncio.sleep(1.05)
        return format_recoverable_timeout_result(
            {"timeout": {"seconds": 0.01}, "stdout": "late\n", "stderr": ""}
        )

    interpreter._execute_async = _slow_execute

    start = asyncio.get_running_loop().time()
    result = await interpreter._execute_with_timeout(
        7,
        timeout_seconds=0.01,
        timeout_failure_class=ITERATION_TIMEOUT_FAILURE_CLASS,
    )

    elapsed = asyncio.get_running_loop().time() - start

    assert DEFAULT_RECOVERABLE_EXECUTION_TIMEOUT_GRACE_SECONDS == 30.0
    assert recoverable_timeout_host_deadline_seconds(
        0.01,
        ITERATION_TIMEOUT_FAILURE_CLASS,
    ) == 30.01
    assert elapsed >= 1.0
    assert "[Timeout] Iteration execution timed out after 0.01s" in result
    assert "[stdout]\nlate" in result
    assert killed == []
    assert any(span["name"] == "sandbox.execute.timeout" for span in spans)


@pytest.mark.asyncio
async def test_jspi_silent_iteration_timeout_recovery_failure_is_bounded(monkeypatch):
    import predict_rlm.execution_timeout as execution_timeout
    from predict_rlm.execution_timeout import ITERATION_TIMEOUT_FAILURE_CLASS
    from predict_rlm.interpreter import JspiInterpreter, SandboxFatalError

    monkeypatch.setattr(
        execution_timeout,
        "DEFAULT_RECOVERABLE_EXECUTION_TIMEOUT_GRACE_SECONDS",
        0.2,
    )
    interpreter = JspiInterpreter.__new__(JspiInterpreter)
    spans = []
    killed = []
    interpreter._write_telemetry_span = lambda name, **kwargs: spans.append(
        {"name": name, **kwargs}
    )
    interpreter._telemetry_pending_tool_count = lambda: 0
    interpreter._telemetry_pending_file_ops_count = lambda: 0
    interpreter._kill_sandbox = lambda: killed.append(True)

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


@pytest.mark.asyncio
async def test_jspi_timeout_result_formats_buffered_stdout_and_stderr():
    from predict_rlm.interpreter import JspiInterpreter

    interpreter = JspiInterpreter.__new__(JspiInterpreter)
    interpreter._pending_file_ops = {}
    interpreter._active_tool_count = 0
    interpreter._sync_files = lambda: None
    interpreter._wait_and_send_all_responses = AsyncMock()
    interpreter._send_completed_responses = AsyncMock()
    interpreter._read_with_timeout_async = AsyncMock(
        return_value='{"jsonrpc":"2.0","result":{"timeout":{"seconds":2.5},'
        '"stdout":"out before\\n","stderr":"err before\\n"},"id":9}'
    )

    result = await interpreter._execute_async(9)

    assert "[Timeout] Iteration execution timed out after 2.5s" in result
    assert "[stdout]\nout before" in result
    assert "[stderr]\nerr before" in result


def test_jspi_recoverable_timeout_preserves_output_and_globals():
    from predict_rlm.interpreter import JspiInterpreter

    interpreter = JspiInterpreter(preinstall_packages=False, exec_timeout=5.0)
    try:
        result = interpreter.execute(
            """
import sys
print("stdout before timeout")
print("stderr before timeout", file=sys.stderr)
survived_value = 123
while True:
    pass
""",
            timeout=0.2,
        )

        output = str(result)
        assert "[Timeout] Iteration execution timed out after 0.2s" in output
        assert "stdout before timeout" in output
        assert "stderr before timeout" in output

        followup = interpreter.execute("survived_value")
        assert followup == 123
    finally:
        interpreter.shutdown()


def test_predict_rlm_jspi_timeout_preserves_state_history_and_predict_tool():
    from predict_rlm import PredictRLM
    from predict_rlm.predict_rlm import dspy

    actions = _SequentialActions(
        SimpleNamespace(
            reasoning="call predict before a bounded risky loop",
            code=(
                "first = await predict('question: str -> answer: str', "
                "question='first call')\n"
                "saved = {'first': first['answer'], 'marker': 123}\n"
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
        max_iterations=2,
        sandbox_backend="jspi",
    )
    rlm.generate_action = actions

    with patch.object(dspy, "Predict", return_value=mock_predictor):
        prediction = rlm(prompt="exercise deno timeout recovery")

    assert prediction.answer == "pre-timeout prediction -> post-timeout prediction / 123"
    assert [call["iteration"] for call in actions.calls] == ["1/2", "2/2"]
    assert mock_predictor.acall.await_count == 2
    assert [call.kwargs["question"] for call in mock_predictor.acall.await_args_list] == [
        "first call",
        "second call",
    ]
    assert len(prediction.trace.steps) == 2
    timeout_step, final_step = prediction.trace.steps
    assert "[Timeout] Iteration execution timed out after 0.2s" in timeout_step.untruncated_output
    assert "first predict: pre-timeout prediction" in timeout_step.untruncated_output
    assert "marker before timeout: 123" in timeout_step.untruncated_output
    assert final_step.output == (
        "FINAL: {'answer': 'pre-timeout prediction -> post-timeout prediction / 123'}"
    )
    second_history = str(actions.calls[1]["repl_history"])
    assert "[Timeout] Iteration execution timed out after 0.2s" in second_history
    assert "first predict: pre-timeout prediction" in second_history


def test_jspi_no_timeout_execution_still_returns_output_and_stderr():
    from predict_rlm.interpreter import JspiInterpreter

    interpreter = JspiInterpreter(preinstall_packages=False, exec_timeout=5.0)
    try:
        result = interpreter.execute(
            """
import sys
print("stdout ok")
print("stderr ok", file=sys.stderr)
"""
        )
    finally:
        interpreter.shutdown()

    output = str(result)
    assert "stdout ok" in output
    assert "stderr ok" in output
    assert "[Timeout]" not in output
