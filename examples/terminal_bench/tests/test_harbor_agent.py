from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from terminal_bench_rlm.skills import (  # noqa: E402
    DEFAULT_TERMINAL_BENCH_SKILL_INSTRUCTIONS,
    TERMINAL_BENCH_SKILL_NAME,
)
from terminal_bench_rlm.tools import tbench_agent  # noqa: E402


def _assert_terminal_bench_skill_semantics(instructions: str) -> None:
    headings = [
        "Operating principle",
        "Inspection and changes",
        "Timeouts and long-running work",
        "Problem-solving strategy",
        "Verification and final submission",
    ]
    heading_positions = [instructions.index(heading) for heading in headings]

    assert heading_positions == sorted(heading_positions)
    assert "inspect the filesystem before making changes" in instructions
    assert "install missing packages" in instructions
    assert "package managers" in instructions
    assert "small inspectable steps" in instructions
    assert "Do not submit based on an unobserved verification command" in instructions
    assert "run the verification in one iteration" in instructions
    assert "separate later iteration" in instructions
    assert "SUBMIT makes the result final" in instructions
    assert "async def start" in instructions
    assert "async def wait" in instructions
    assert "stdout_tail" in instructions
    assert "stderr_tail" in instructions
    assert "call wait(job, seconds=5) again" in instructions
    assert (
        "# Use run() for bounded foreground commands; inspect output before continuing."
        in instructions
    )
    assert (
        "# Use start()/wait() for longer jobs; poll briefly, return to loop, inspect tails later."
        in instructions
    )
    assert "# Use requests timeouts for network calls." in instructions
    assert (
        "# Use asyncio.wait_for for expensive computations or async work that may hang."
        in instructions
    )
    assert (
        "# Use asyncio.gather for independent non-mutating checks that can run concurrently."
        in instructions
    )
    assert "run_terminal_command" not in instructions
    assert "send_terminal_keys" not in instructions
    assert "read_terminal" not in instructions


def test_harbor_agent_runs_predict_rlm_async_against_harbor_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakePredictRLM:
        def __init__(self, signature, **kwargs) -> None:
            captured["signature"] = signature
            captured["kwargs"] = kwargs

        def __call__(self, **_kwargs):
            raise AssertionError("Harbor adapter must not call sync PredictRLM entrypoint")

        async def acall(self, **kwargs):
            captured["acall_kwargs"] = kwargs
            return SimpleNamespace(answer="done", trace=None)

    class FakeInterpreter:
        def __init__(self, environment, *, loop, **kwargs) -> None:
            captured["environment"] = environment
            captured["loop"] = loop
            captured["interpreter_kwargs"] = kwargs

        def shutdown(self) -> None:
            captured["shutdown"] = True

    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(tbench_agent, "HarborEnvironmentInterpreter", FakeInterpreter)

    environment = object()
    context = SimpleNamespace()
    agent = tbench_agent.HarborPredictRLMAgent(
        logs_dir=tmp_path,
        skill_instructions="Use shell commands and verify outputs.",
        max_iterations="3",
    )

    asyncio.run(agent.run("solve this task", environment, context))

    assert captured["environment"] is environment
    assert captured["signature"] == "instruction -> answer"
    assert captured["kwargs"]["interpreter"] is not None
    assert captured["kwargs"]["max_iterations"] == 3
    assert captured["kwargs"]["skills"][0].name == "terminal-bench"
    assert captured["kwargs"]["skills"][0].instructions == "Use shell commands and verify outputs."
    assert captured["acall_kwargs"] == {"instruction": "solve this task"}
    assert captured["shutdown"] is True


def test_harbor_agent_uses_default_terminal_bench_skill_when_not_overridden(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakePredictRLM:
        def __init__(self, _signature, **kwargs) -> None:
            captured["kwargs"] = kwargs

        async def acall(self, **_kwargs):
            return SimpleNamespace(answer="done", trace=None)

    class FakeInterpreter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(tbench_agent, "HarborEnvironmentInterpreter", FakeInterpreter)

    context = SimpleNamespace()
    agent = tbench_agent.HarborPredictRLMAgent(logs_dir=tmp_path)

    asyncio.run(agent.run("solve", object(), context))

    skills = captured["kwargs"]["skills"]
    assert len(skills) == 1
    assert skills[0].name == TERMINAL_BENCH_SKILL_NAME
    assert skills[0].instructions == DEFAULT_TERMINAL_BENCH_SKILL_INSTRUCTIONS
    _assert_terminal_bench_skill_semantics(skills[0].instructions)
    assert "tools" not in captured["kwargs"]


def test_harbor_agent_keeps_one_terminal_bench_skill(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakePredictRLM:
        def __init__(self, _signature, **kwargs) -> None:
            captured["kwargs"] = kwargs

        async def acall(self, **_kwargs):
            return SimpleNamespace(answer="done", trace=None)

    class FakeInterpreter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(tbench_agent, "HarborEnvironmentInterpreter", FakeInterpreter)

    context = SimpleNamespace()
    agent = tbench_agent.HarborPredictRLMAgent(
        logs_dir=tmp_path,
        skills=[
            SimpleNamespace(name=TERMINAL_BENCH_SKILL_NAME, instructions="stale"),
            SimpleNamespace(name="other", instructions="keep"),
        ],
    )

    asyncio.run(agent.run("solve", object(), context))

    skills = captured["kwargs"]["skills"]
    terminal_bench_skills = [
        skill for skill in skills if getattr(skill, "name", None) == TERMINAL_BENCH_SKILL_NAME
    ]
    assert len(terminal_bench_skills) == 1
    assert terminal_bench_skills[0].instructions == DEFAULT_TERMINAL_BENCH_SKILL_INSTRUCTIONS
    assert [getattr(skill, "name", None) for skill in skills] == ["other", TERMINAL_BENCH_SKILL_NAME]


def test_harbor_agent_does_not_forward_harbor_extra_env_to_predict_rlm(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakePredictRLM:
        def __init__(self, _signature, **kwargs) -> None:
            captured["kwargs"] = kwargs

        async def acall(self, **_kwargs):
            return SimpleNamespace(answer="done", trace=None)

    class FakeInterpreter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(tbench_agent, "HarborEnvironmentInterpreter", FakeInterpreter)

    context = SimpleNamespace()
    agent = tbench_agent.HarborPredictRLMAgent(
        logs_dir=tmp_path,
        extra_env={"OPENAI_API_KEY": "codex-lm"},
    )

    asyncio.run(agent.run("solve", object(), context))

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert "extra_env" not in kwargs
    assert context.answer == "done"



def test_harbor_agent_accepts_harbor_post_run_context_hook(tmp_path: Path) -> None:
    context = SimpleNamespace(metadata={})
    agent = tbench_agent.HarborPredictRLMAgent(logs_dir=tmp_path)

    agent.populate_context_post_run(context)


def test_harbor_agent_populates_context_answer(monkeypatch, tmp_path: Path) -> None:
    class FakePredictRLM:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def acall(self, **_kwargs):
            return SimpleNamespace(answer="finished", trace=None)

    class FakeInterpreter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(tbench_agent, "HarborEnvironmentInterpreter", FakeInterpreter)

    context = SimpleNamespace()
    agent = tbench_agent.HarborPredictRLMAgent(logs_dir=tmp_path)

    asyncio.run(agent.run("solve", object(), context))

    assert context.answer == "finished"


def test_harbor_agent_populates_pydantic_context_metadata(monkeypatch, tmp_path: Path) -> None:
    harbor_context = pytest.importorskip("harbor.models.agent.context")

    class FakePredictRLM:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def acall(self, **_kwargs):
            return SimpleNamespace(answer="finished", trace=None)

    class FakeInterpreter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(tbench_agent, "HarborEnvironmentInterpreter", FakeInterpreter)

    context = harbor_context.AgentContext()
    agent = tbench_agent.HarborPredictRLMAgent(logs_dir=tmp_path)

    asyncio.run(agent.run("solve", object(), context))

    assert context.metadata == {"answer": "finished"}


def test_harbor_agent_writes_setup_and_agent_phase_events(monkeypatch, tmp_path: Path) -> None:
    class FakePredictRLM:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def acall(self, **_kwargs):
            return SimpleNamespace(answer="finished", trace=None)

    class FakeInterpreter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(tbench_agent, "HarborEnvironmentInterpreter", FakeInterpreter)

    phase_log = tmp_path / "task_phase_events.jsonl"
    agent = tbench_agent.HarborPredictRLMAgent(
        logs_dir=tmp_path,
        phase_log_path=str(phase_log),
        task_id="terminal-bench/task",
    )

    asyncio.run(agent.setup(object()))
    asyncio.run(agent.run("solve", object(), SimpleNamespace()))

    events = [json.loads(line) for line in phase_log.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "agent_setup_start",
        "agent_setup_end",
        "agent_run_start",
        "sandbox_setup_start",
        "sandbox_setup_end",
        "agent_run_end",
    ]
    assert {event["task_id"] for event in events} == {"terminal-bench/task"}
    assert events[0]["phase"] == "agent_setup"
    assert events[2]["phase"] == "agent_eval"
    assert events[4]["phase"] == "sandbox_setup"
    assert events[4]["duration_seconds"] >= 0
    assert events[5]["duration_seconds"] >= 0


def test_harbor_agent_shuts_down_and_writes_trace_on_async_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakeTrace:
        def to_exportable_json(self) -> str:
            return '{"status": "failed"}'

    class FakePredictRLM:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __call__(self, **_kwargs):
            raise AssertionError("Harbor adapter must not call sync PredictRLM entrypoint")

        async def acall(self, **_kwargs):
            exc = RuntimeError("rlm failed")
            exc.trace = FakeTrace()
            raise exc

    class FakeInterpreter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self) -> None:
            captured["shutdown"] = True

    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(tbench_agent, "HarborEnvironmentInterpreter", FakeInterpreter)

    phase_log = tmp_path / "task_phase_events.jsonl"
    agent = tbench_agent.HarborPredictRLMAgent(
        logs_dir=tmp_path,
        phase_log_path=str(phase_log),
        task_id="terminal-bench/task",
    )

    with pytest.raises(RuntimeError, match="rlm failed"):
        asyncio.run(agent.run("solve", object(), SimpleNamespace()))

    assert captured["shutdown"] is True
    trace_files = list(tmp_path.glob("predict_rlm_trace_*.json"))
    assert len(trace_files) == 1
    assert json.loads(trace_files[0].read_text(encoding="utf-8")) == {"status": "failed"}
    events = [json.loads(line) for line in phase_log.read_text().splitlines()]
    assert events[-1]["event"] == "agent_run_end"
    assert events[-1]["phase"] == "agent_eval"
    assert events[-1]["status"] == "failed"
