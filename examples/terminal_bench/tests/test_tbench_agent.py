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

        def __call__(self, **kwargs):
            events.append(("call", kwargs))
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
    assert result.total_input_tokens == 0
    assert result.total_output_tokens == 0
    assert result.failure_mode is None
    assert result.timestamped_markers == []
    assert interpreter.container is session
    assert rlm.signature == "instruction -> answer"
    assert rlm.kwargs["interpreter"] is interpreter
    assert rlm.kwargs["lm"] == "main"
    assert rlm.kwargs["sub_lm"] == "sub"
    assert "no_rebuild" not in rlm.kwargs
    assert not (
        set(rlm.kwargs.get("tools") or {})
        & tbench_agent.TERMINAL_WRAPPER_TOOL_NAMES
    )
    assert events[-1] == ("shutdown", interpreter)


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

        def __call__(self, **_kwargs):
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

    with pytest.raises(RuntimeError, match="dspy-codex-lm.*Terminal-Bench"):
        agent.perform_task("solve it", SimpleNamespace(container="container"))


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

        def __call__(self, **_kwargs):
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
