from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from terminal_bench_rlm.tools import tbench_agent  # noqa: E402


def test_harbor_agent_runs_predict_rlm_against_harbor_environment(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakePredictRLM:
        def __init__(self, signature, **kwargs) -> None:
            captured["signature"] = signature
            captured["kwargs"] = kwargs

        def __call__(self, **kwargs):
            captured["call_kwargs"] = kwargs
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
    assert captured["call_kwargs"] == {"instruction": "solve this task"}
    assert captured["shutdown"] is True


def test_harbor_agent_does_not_forward_harbor_extra_env_to_predict_rlm(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakePredictRLM:
        def __init__(self, _signature, **kwargs) -> None:
            captured["kwargs"] = kwargs

        def __call__(self, **_kwargs):
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



def test_harbor_agent_populates_context_answer(monkeypatch, tmp_path: Path) -> None:
    class FakePredictRLM:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __call__(self, **_kwargs):
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

        def __call__(self, **_kwargs):
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
