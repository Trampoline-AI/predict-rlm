from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from terminal_bench_rlm.tools import tbench_agent  # noqa: E402


def _assert_task_instruction_signature(signature, task_instruction: str) -> None:
    assert list(signature.input_fields) == []
    assert list(signature.output_fields) == ["answer"]
    assert "Terminal-Bench task instruction" in signature.instructions
    assert task_instruction in signature.instructions


def _assert_terminal_bench_submit_confirmation_contract(message: str) -> None:
    normalized = " ".join(message.split())
    assert "re-read the original task" in normalized
    assert "todos" in message
    assert "required verification" in message
    assert "current final state" in message
    assert "fresh evidence" in message
    assert "verifier-shaped" in message
    assert "unverified" in message
    assert "blocker" in message
    assert "stale debug history" in message
    assert "file existence alone" in message
    assert "The second SUBMIT" in message
    assert "same already-verified answer" in message
    assert "Does your solution meet" not in message
    assert "If everything looks good" not in message


def test_agent_exposes_terminal_bench_name() -> None:
    assert tbench_agent.TerminalBenchRLMBaseAgent.name() == "predict-rlm"


def test_import_path_agent_class_exposes_terminal_bench_name() -> None:
    assert tbench_agent.TerminalBenchRLMAgent.name() == "predict-rlm"
    assert issubclass(tbench_agent.TerminalBenchRLMAgent, tbench_agent.TerminalBenchRLMBaseAgent)


def test_agent_constructs_predict_rlm_with_container_interpreter(monkeypatch) -> None:
    events: list[tuple[str, object]] = []

    class FakeInterpreter:
        def __init__(self, container, **kwargs) -> None:
            self.container = container
            self.kwargs = kwargs
            events.append(("interpreter", self))

        def shutdown(self) -> None:
            events.append(("shutdown", self))

    class FakePredictRLM:
        def __init__(self, signature, **kwargs) -> None:
            self.signature = signature
            self.kwargs = kwargs
            events.append(("rlm", self))

        def __call__(self, **_kwargs):
            raise AssertionError("Harbor agents should use PredictRLM.acall()")

        async def acall(self, **kwargs):
            events.append(("acall", kwargs))
            return SimpleNamespace(answer="done")

    monkeypatch.setattr(tbench_agent, "TerminalBenchRunnerInterpreter", FakeInterpreter)
    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)

    agent = tbench_agent.TerminalBenchRLMBaseAgent(
        lm="main",
        sub_lm="sub",
        no_rebuild=False,
    )
    session = SimpleNamespace(container="container")
    result = agent.perform_task("solve it", session)

    interpreter = events[0][1]
    rlm = events[1][1]
    call = events[2][1]
    assert events[2][0] == "acall"
    assert result.total_input_tokens == 0
    assert result.total_output_tokens == 0
    assert result.failure_mode is None
    assert result.timestamped_markers == []
    assert interpreter.container is session
    _assert_task_instruction_signature(rlm.signature, "solve it")
    assert rlm.kwargs["interpreter"] is interpreter
    assert rlm.kwargs["lm"] == "main"
    assert rlm.kwargs["sub_lm"] == "sub"
    assert "no_rebuild" not in rlm.kwargs
    assert callable(rlm.kwargs["submit_confirmation"])
    assert not (
        set(rlm.kwargs.get("tools") or {})
        & tbench_agent.TERMINAL_WRAPPER_TOOL_NAMES
    )
    assert call == {}
    assert events[-1] == ("shutdown", interpreter)


def test_agent_preserves_custom_signature_instructions(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeInterpreter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self) -> None:
            pass

    class FakePredictRLM:
        def __init__(self, signature, **_kwargs) -> None:
            captured["signature"] = signature

        async def acall(self, **kwargs):
            captured["call_kwargs"] = kwargs
            return SimpleNamespace(answer="done")

    monkeypatch.setattr(tbench_agent, "TerminalBenchRunnerInterpreter", FakeInterpreter)
    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)

    base_signature = tbench_agent.dspy.Signature(
        "instruction -> answer",
        "Keep existing benchmark guidance.",
    )
    agent = tbench_agent.TerminalBenchRLMBaseAgent(signature=base_signature)

    agent.perform_task("solve the custom task", SimpleNamespace(container="container"))

    signature = captured["signature"]
    _assert_task_instruction_signature(signature, "solve the custom task")
    assert "Keep existing benchmark guidance." in signature.instructions
    assert captured["call_kwargs"] == {}


def test_agent_terminal_bench_submit_confirmation_mode_passes_callback(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeInterpreter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self) -> None:
            pass

    class FakePredictRLM:
        def __init__(self, _signature, **kwargs) -> None:
            captured["rlm_kwargs"] = kwargs

        async def acall(self, **_kwargs):
            return SimpleNamespace(answer="done")

    monkeypatch.setattr(tbench_agent, "TerminalBenchRunnerInterpreter", FakeInterpreter)
    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)

    agent = tbench_agent.TerminalBenchRLMBaseAgent(
        submit_confirmation_mode="terminal_bench"
    )
    agent.perform_task(
        "Edit the config and verify the service starts.",
        SimpleNamespace(container="container"),
    )

    rlm_kwargs = captured["rlm_kwargs"]
    assert isinstance(rlm_kwargs, dict)
    callback = rlm_kwargs["submit_confirmation"]
    assert callable(callback)

    message = str(
        callback(
            SimpleNamespace(
                submitted_payload={"answer": "changed config"},
                latest_observation="pytest passed",
                iteration=4,
            )
        )
    )
    assert "Original task:" in message
    assert "Edit the config and verify the service starts." in message
    assert "Current submitted payload / terminal state:" in message
    assert '"answer": "changed config"' in message
    assert "pytest passed" in message
    assert "Are you sure you want to mark the task as complete?" in message
    _assert_terminal_bench_submit_confirmation_contract(message)
    assert "call SUBMIT again" in message


def test_agent_installs_codex_lm_before_constructing_predict_rlm(monkeypatch) -> None:
    events: list[tuple[str, object]] = []
    captured_env: dict[str, str | None] = {}

    class FakeInterpreter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self) -> None:
            events.append(("shutdown", None))

    class FakePredictRLM:
        def __init__(self, signature, **kwargs) -> None:
            events.append(("rlm", kwargs))
            self.signature = signature
            self.kwargs = kwargs

        async def acall(self, **_kwargs):
            return SimpleNamespace(answer="done")

    def install_monkeypatch(*, exclude=()):
        events.append(("install", tuple(exclude)))
        captured_env["OPENAI_API_KEY"] = tbench_agent.os.environ.get("OPENAI_API_KEY")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(tbench_agent, "TerminalBenchRunnerInterpreter", FakeInterpreter)
    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)
    monkeypatch.setitem(sys.modules, "dspy_codex_lm", types.ModuleType("dspy_codex_lm"))
    cli_module = types.ModuleType("dspy_codex_lm.cli")
    cli_module.install_monkeypatch = install_monkeypatch
    monkeypatch.setitem(sys.modules, "dspy_codex_lm.cli", cli_module)

    agent = tbench_agent.TerminalBenchRLMBaseAgent(
        lm="main",
        sub_lm="sub",
        codex_lm=True,
        codex_lm_exclude="openai/keep-direct,anthropic/",
        no_rebuild=False,
    )
    agent.perform_task("solve it", SimpleNamespace(container="container"))

    assert events[0] == ("install", ("openai/keep-direct", "anthropic/"))
    assert captured_env["OPENAI_API_KEY"] == "codex-lm"
    rlm_kwargs = events[1][1]
    assert isinstance(rlm_kwargs, dict)
    assert rlm_kwargs["lm"] == "main"
    assert rlm_kwargs["sub_lm"] == "sub"
    assert "codex_lm" not in rlm_kwargs
    assert "codex_lm_exclude" not in rlm_kwargs
    assert "no_rebuild" not in rlm_kwargs


def test_agent_raises_clear_error_when_codex_lm_dependency_missing(monkeypatch) -> None:
    class FakeInterpreter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self) -> None:
            pass

    class FakePredictRLM:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("PredictRLM should not be constructed")

    def fake_import_module(name: str, package: str | None = None):
        if name == "dspy_codex_lm.cli":
            raise ImportError("missing")
        return real_import_module(name, package)

    real_import_module = tbench_agent.importlib.import_module
    monkeypatch.setattr(tbench_agent, "TerminalBenchRunnerInterpreter", FakeInterpreter)
    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(tbench_agent.importlib, "import_module", fake_import_module)

    agent = tbench_agent.TerminalBenchRLMBaseAgent(codex_lm=True)

    with pytest.raises(RuntimeError) as exc_info:
        agent.perform_task("solve it", SimpleNamespace(container="container"))

    message = str(exc_info.value)
    assert "predict-rlm[codex-lm]" in message
    assert "dspy-codex-lm" not in message


def test_agent_exports_predict_rlm_trace_to_logging_dir(monkeypatch, tmp_path: Path) -> None:
    class FakeTrace:
        def to_exportable_json(self) -> str:
            return '{"status":"completed","model":"main","sub_model":null,"iterations":0,"max_iterations":1,"duration_ms":1,"usage":{"main":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0,"cost_usd":0.0},"sub":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0,"cost_usd":0.0}},"telemetry_ref":null,"steps":[]}'

    class FakeInterpreter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self) -> None:
            pass

    class FakePredictRLM:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def acall(self, **_kwargs):
            return SimpleNamespace(answer="done", trace=FakeTrace())

    monkeypatch.setattr(tbench_agent, "TerminalBenchRunnerInterpreter", FakeInterpreter)
    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)

    agent = tbench_agent.TerminalBenchRLMBaseAgent()
    agent.perform_task("solve it", SimpleNamespace(container="container"), logging_dir=tmp_path)

    trace_files = list(tmp_path.glob("predict_rlm_trace*.json"))
    assert len(trace_files) == 1
    assert '\"status\":\"completed\"' in trace_files[0].read_text()


def test_agent_rejects_terminal_wrapper_tools() -> None:
    def run_terminal_command(command: str) -> str:
        return command

    try:
        tbench_agent.TerminalBenchRLMBaseAgent(tools=[run_terminal_command])
    except ValueError as exc:
        assert "terminal wrapper tools" in str(exc)
    else:
        raise AssertionError("expected wrapper tools to be rejected")


def test_agent_factory_returns_upstream_baseagent_subclass(monkeypatch) -> None:
    class FakeBaseAgent:
        pass

    module = types.ModuleType("terminal_bench.agents.base_agent")
    module.BaseAgent = FakeBaseAgent
    monkeypatch.setitem(sys.modules, "terminal_bench.agents.base_agent", module)

    agent_class = tbench_agent.terminal_bench_agent_class()

    assert issubclass(agent_class, FakeBaseAgent)
    assert agent_class.name() == "predict-rlm"
