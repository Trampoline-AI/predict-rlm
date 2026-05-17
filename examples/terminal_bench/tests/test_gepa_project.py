from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from terminal_bench_rlm.gepa import cli as gepa_cli  # noqa: E402
from terminal_bench_rlm.gepa.config import COMPONENT_SKILL, default_config  # noqa: E402
from terminal_bench_rlm.gepa.project import (  # noqa: E402
    HarborSubprocessHarnessRunner,
    TerminalBenchGepaProject,
    TerminalBenchInProcessHarnessRunner,
    TerminalBenchSubprocessHarnessRunner,
    TerminalBenchTaskRunRequest,
    TerminalBenchTaskRunResult,
)
from terminal_bench_rlm.tools import tbench_agent  # noqa: E402

from predict_rlm.trace import RunTrace  # noqa: E402
from rlm_gepa import EvaluationContext, RLMGepaExampleResult  # noqa: E402
from rlm_gepa.schema import validate_project  # noqa: E402


class FakeHarnessRunner:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[object] = []

    async def run(self, request):
        self.calls.append(request)
        return self.result


def test_project_validation_check_has_non_empty_train_and_val_examples() -> None:
    project = TerminalBenchGepaProject(default_config())

    validation = validate_project(project)

    assert validation.seed_candidate[COMPONENT_SKILL].strip()
    assert len(validation.trainset) >= 1
    assert len(validation.valset) >= 1


def test_config_serializes_terminal_bench_fields_for_run_metadata() -> None:
    payload = default_config().to_dict()

    json.dumps(payload)
    assert payload["harness_backend"] == "harbor"
    assert payload["harbor_dataset"] == "terminal-bench/terminal-bench-2-1"
    assert payload["terminal_bench_output_dir"] == "runs/gepa-terminal-bench"
    assert payload["train_task_ids"] == ["configure-git-webserver", "extract-moves-from-video"]
    assert payload["val_task_ids"] == ["super-benchmark-upet"]
    assert payload["max_iterations"] == 30


def test_cli_accepts_harbor_backend_and_executable_args() -> None:
    parser = argparse.ArgumentParser()
    gepa_cli._add_project_args(parser)
    args = parser.parse_args(
        [
            "--harness-backend",
            "harbor",
            "--harbor-executable",
            "uvx harbor",
            "--harbor-dataset",
            "terminal-bench/terminal-bench-2",
        ]
    )

    config = gepa_cli._apply_project_args(default_config(), args)

    assert config.harness_backend == "harbor"
    assert config.harbor_executable == "uvx harbor"
    assert config.harbor_dataset == "terminal-bench/terminal-bench-2"


def test_build_project_uses_harbor_harness_by_default() -> None:
    project = TerminalBenchGepaProject(default_config())

    assert isinstance(project.harness_runner, HarborSubprocessHarnessRunner)


def test_build_project_can_still_use_python_harness() -> None:
    config = default_config()
    config.harness_backend = "python"
    project = TerminalBenchGepaProject(config)

    assert isinstance(project.harness_runner, TerminalBenchInProcessHarnessRunner)


def test_build_project_can_still_use_cli_harness() -> None:
    config = default_config()
    config.harness_backend = "cli"
    project = TerminalBenchGepaProject(config)

    assert isinstance(project.harness_runner, TerminalBenchSubprocessHarnessRunner)


def test_harbor_agent_exposes_agent_info_without_harbor_dependency() -> None:
    agent = tbench_agent.HarborPredictRLMAgent(
        logs_dir=Path("/tmp/logs"),
        model_name="openai/gpt-5.4-mini",
        lm="openai/gpt-5.4-mini",
    )

    assert agent.predict_rlm_kwargs == {"lm": "openai/gpt-5.4-mini"}
    assert agent.to_agent_info() == {
        "name": "predict-rlm",
        "version": "unknown",
        "model_info": None,
    }


def test_harbor_runner_builds_harbor_run_command(monkeypatch, tmp_path: Path) -> None:
    config = default_config()
    config.terminal_bench_output_dir = tmp_path / "harbor-runs"
    config.harbor_dataset = "terminal-bench/terminal-bench-2-1"
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        run_dir = config.terminal_bench_output_dir / "gepa-val-task"
        run_dir.mkdir(parents=True)
        (run_dir / "result.json").write_text(
            json.dumps(
                {
                    "trial_results": [
                        {
                            "task_info": {"name": "task"},
                            "verifier_result": {"rewards": {"reward": 1.0}},
                        }
                    ]
                }
            )
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = HarborSubprocessHarnessRunner(cwd=tmp_path)._run_sync(
        TerminalBenchTaskRunRequest(
            task_id="task",
            instruction="",
            skill_instructions="skill",
            lm="main",
            sub_lm="sub",
            max_iterations=3,
            task_timeout=900,
            verbose_rlm=True,
            output_dir=tmp_path,
            run_id="gepa-val-task",
            config=config,
        )
    )

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[:4] == ["harbor", "run", "-d", "terminal-bench/terminal-bench-2-1"]
    assert "--include-task-name" in cmd
    assert cmd[cmd.index("--include-task-name") + 1] == "task"
    assert "--agent-import-path" in cmd
    assert cmd[cmd.index("--agent-import-path") + 1] == (
        "terminal_bench_rlm.tools.tbench_agent:HarborPredictRLMAgent"
    )
    assert "--jobs-dir" in cmd
    assert cmd[cmd.index("--jobs-dir") + 1] == str(config.terminal_bench_output_dir)
    assert "--job-name" in cmd
    assert cmd[cmd.index("--job-name") + 1] == "gepa-val-task"
    assert "--agent-timeout" not in cmd
    assert "--n-attempts" in cmd
    assert cmd[cmd.index("--n-attempts") + 1] == "1"
    assert "--n-concurrent" in cmd
    assert cmd[cmd.index("--n-concurrent") + 1] == "1"
    assert "--agent-kwarg" in cmd
    assert "exec_timeout=900" in cmd
    assert result.error is None
    assert result.trial_result["verifier_result"]["rewards"]["reward"] == 1.0


def test_harbor_runner_loads_harbor_result_json_without_subprocess(tmp_path: Path) -> None:
    config = default_config()
    config.terminal_bench_output_dir = tmp_path / "harbor-runs"
    run_dir = config.terminal_bench_output_dir / "gepa-val-task"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_results": [
                    {
                        "task_name": "other",
                        "verifier_result": {"rewards": {"reward": 0.0}},
                    },
                    {
                        "task_name": "task",
                        "verifier_result": {"rewards": {"reward": 0.25}},
                    },
                ]
            }
        )
    )

    request = TerminalBenchTaskRunRequest(
        task_id="task",
        instruction="",
        skill_instructions="skill",
        lm="main",
        sub_lm="sub",
        max_iterations=3,
        task_timeout=30,
        verbose_rlm=False,
        output_dir=tmp_path,
        run_id="gepa-val-task",
        config=config,
    )

    result = HarborSubprocessHarnessRunner(cwd=tmp_path)._load_result(request, run_dir)

    assert result.error is None
    assert result.trial_result["verifier_result"]["rewards"]["reward"] == 0.25


def test_harbor_runner_loads_nested_trial_result_with_ctrf_details(tmp_path: Path) -> None:
    config = default_config()
    config.terminal_bench_output_dir = tmp_path / "harbor-runs"
    run_dir = config.terminal_bench_output_dir / "gepa-val-video-processing"
    trial_dir = run_dir / "video-processing__abc123"
    verifier_dir = trial_dir / "verifier"
    verifier_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "id": "job-id",
                "n_total_trials": 1,
                "stats": {
                    "evals": {
                        "predict-rlm__terminal-bench/terminal-bench-2": {
                            "reward_stats": {"reward": {"0.0": ["video-processing__abc123"]}}
                        }
                    }
                },
            }
        )
    )
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "terminal-bench/video-processing",
                "trial_name": "video-processing__abc123",
                "verifier_result": {"rewards": {"reward": 0.0}},
                "exception_info": None,
            }
        )
    )
    (verifier_dir / "ctrf.json").write_text(
        json.dumps(
            {
                "results": {
                    "summary": {"tests": 5, "passed": 4, "failed": 1},
                    "tests": [
                        {"name": "test_a", "status": "passed"},
                        {"name": "test_b", "status": "passed"},
                        {"name": "test_c", "status": "passed"},
                        {"name": "test_d", "status": "passed"},
                        {"name": "test_e", "status": "failed"},
                    ],
                }
            }
        )
    )

    request = TerminalBenchTaskRunRequest(
        task_id="video-processing",
        instruction="",
        skill_instructions="skill",
        lm="main",
        sub_lm="sub",
        max_iterations=3,
        task_timeout=30,
        verbose_rlm=False,
        output_dir=tmp_path,
        run_id="gepa-val-video-processing",
        config=config,
    )

    result = HarborSubprocessHarnessRunner(cwd=tmp_path)._load_result(request, run_dir)
    details = result.trial_result["verifier_result"]["ctrf"]["results"]["summary"]

    assert result.error is None
    assert result.trial_result["trial_name"] == "video-processing__abc123"
    assert details == {"tests": 5, "passed": 4, "failed": 1}


def test_seed_candidate_allows_shell_package_and_tool_workflows() -> None:
    skill = TerminalBenchGepaProject(default_config()).seed_candidate()[COMPONENT_SKILL]

    assert "subprocess.run" in skill
    assert "install missing packages" in skill
    assert "package managers" in skill
    assert "programmatic tools" in skill


def test_seed_candidate_skill_is_passed_to_terminal_bench_agent(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakePredictRLM:
        def __init__(self, signature, **kwargs) -> None:
            captured["signature"] = signature
            captured["kwargs"] = kwargs

        def __call__(self, **_kwargs):
            return SimpleNamespace(answer="done")

    class FakeInterpreter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(tbench_agent, "TerminalBenchRunnerInterpreter", FakeInterpreter)

    agent = tbench_agent.TerminalBenchRLMBaseAgent(
        skill_instructions="Prefer idempotent shell commands and verify files before finishing.",
    )
    agent.perform_task("solve it", SimpleNamespace(container=object()))

    skills = captured["kwargs"]["skills"]
    assert len(skills) == 1
    assert skills[0].name == "terminal-bench"
    assert "verify files before finishing" in skills[0].instructions


def test_evaluate_example_returns_gepa_result_from_fake_harness_runner(tmp_path: Path) -> None:
    parser_result = SimpleNamespace(
        is_resolved=False,
        parser_results={"test_a": "passed", "test_b": "failed"},
    )
    runner = FakeHarnessRunner(
        TerminalBenchTaskRunResult(
            task_id="configure-git-webserver",
            trial_result=parser_result,
            traces=[],
        )
    )
    config = default_config()
    project = TerminalBenchGepaProject(config, harness_runner=runner)
    example = project.load_valset()[0]
    context = EvaluationContext(
        lm="executor",
        sub_lm="sub",
        max_iterations=2,
        task_timeout=30,
        output_dir=tmp_path,
        kind="val",
    )

    result = asyncio.run(
        project.evaluate_example(
            {COMPONENT_SKILL: "Candidate skill: inspect logs, edit files, and run tests."},
            example,
            context,
        )
    )

    assert isinstance(result, RLMGepaExampleResult)
    assert result.score == 0.5
    assert result.objective_scores == {
        "soft_score": 0.5,
        "hard_score": 0.0,
        "passed": 1,
        "total": 2,
        "is_resolved": False,
    }
    assert result.example_id == example.task_id
    assert "soft=0.500 hard=0.000 passed=1/2" in result.feedback
    assert runner.calls[0].skill_instructions.startswith("Candidate skill")
    assert runner.calls[0].lm == "executor"
    assert runner.calls[0].sub_lm == "sub"


def test_subprocess_runner_loads_exported_predict_rlm_trace(
    monkeypatch, tmp_path: Path
) -> None:
    config = default_config()
    config.terminal_bench_output_dir = tmp_path / "tbench-runs"
    run_id = "gepa-val-task"
    run_dir = config.terminal_bench_output_dir / run_id
    logging_dir = run_dir / "logs" / "agent"
    logging_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(
        json.dumps({"results": [{"task_id": "task", "is_resolved": True, "parser_results": {}}]})
    )
    trace = RunTrace(
        status="completed",
        model="main",
        sub_model=None,
        iterations=0,
        max_iterations=1,
        duration_ms=1,
    )
    trace.to_exportable_json(logging_dir / "predict_rlm_trace.json")

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = TerminalBenchSubprocessHarnessRunner(cwd=tmp_path)._run_sync(
        TerminalBenchTaskRunRequest(
            task_id="task",
            instruction="",
            skill_instructions="skill",
            lm="main",
            sub_lm="sub",
            max_iterations=1,
            task_timeout=30,
            verbose_rlm=False,
            output_dir=tmp_path,
            run_id=run_id,
            config=config,
        )
    )

    assert result.error is None
    assert len(result.traces) == 1
    assert result.traces[0].status == "completed"


def test_in_process_runner_calls_terminal_bench_harness_and_loads_results(
    monkeypatch, tmp_path: Path
) -> None:
    config = default_config()
    config.terminal_bench_output_dir = tmp_path / "tbench-runs"
    captured: dict[str, object] = {}

    class FakeHarness:
        def __init__(self, **kwargs) -> None:
            captured["kwargs"] = kwargs

        def run(self):
            run_dir = config.terminal_bench_output_dir / "gepa-val-task"
            run_dir.mkdir(parents=True)
            (run_dir / "results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "task_id": "task",
                                "is_resolved": False,
                                "parser_results": {"test_a": "passed", "test_b": "failed"},
                            }
                        ]
                    }
                )
            )
            return SimpleNamespace()

    monkeypatch.setitem(sys.modules, "terminal_bench", types.ModuleType("terminal_bench"))
    monkeypatch.setitem(sys.modules, "terminal_bench.harness", types.ModuleType("harness"))
    harness_module = types.ModuleType("harness")
    harness_module.Harness = FakeHarness
    monkeypatch.setitem(sys.modules, "terminal_bench.harness.harness", harness_module)

    result = TerminalBenchInProcessHarnessRunner(cwd=tmp_path)._run_sync(
        TerminalBenchTaskRunRequest(
            task_id="task",
            instruction="",
            skill_instructions="skill",
            lm="main",
            sub_lm="sub",
            max_iterations=3,
            task_timeout=900,
            verbose_rlm=True,
            output_dir=tmp_path,
            run_id="gepa-val-task",
            config=config,
        )
    )

    assert result.error is None
    assert result.trial_result["parser_results"] == {"test_a": "passed", "test_b": "failed"}
    kwargs = captured["kwargs"]
    assert kwargs["agent_import_path"] == "terminal_bench_rlm.tools.tbench_agent:TerminalBenchRLMAgent"
    assert kwargs["agent_kwargs"]["skill_instructions"] == "skill"
    assert kwargs["agent_kwargs"]["max_iterations"] == "3"
    assert kwargs["agent_kwargs"]["verbose"] == "true"
    assert kwargs["task_ids"] == ["task"]
    assert kwargs["global_agent_timeout_sec"] == 870


def test_subprocess_runner_passes_codex_lm_agent_kwargs(
    monkeypatch, tmp_path: Path
) -> None:
    config = default_config()
    config.terminal_bench_output_dir = tmp_path / "tbench-runs"
    config.codex_lm = True
    config.codex_lm_exclude = ("openai/keep-direct", "anthropic/")
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    TerminalBenchSubprocessHarnessRunner(cwd=tmp_path)._run_sync(
        TerminalBenchTaskRunRequest(
            task_id="task",
            instruction="",
            skill_instructions="skill",
            lm="main",
            sub_lm="sub",
            max_iterations=3,
            task_timeout=30,
            verbose_rlm=True,
            output_dir=tmp_path,
            run_id="gepa-val-task",
            config=config,
        )
    )

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    agent_kwargs = [
        cmd[index + 1]
        for index, value in enumerate(cmd[:-1])
        if value == "--agent-kwarg"
    ]
    assert "codex_lm=true" in agent_kwargs
    assert "codex_lm_exclude=openai/keep-direct,anthropic/" in agent_kwargs
    assert "verbose=true" in agent_kwargs
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert "capture_output" not in kwargs


def test_subprocess_runner_passes_reasoning_effort_agent_kwargs(
    monkeypatch, tmp_path: Path
) -> None:
    config = default_config()
    config.terminal_bench_output_dir = tmp_path / "tbench-runs"
    captured: dict[str, object] = {}

    class FakeLM:
        model = "openai/gpt-5.5"
        kwargs = {"reasoning_effort": "low"}

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    TerminalBenchSubprocessHarnessRunner(cwd=tmp_path)._run_sync(
        TerminalBenchTaskRunRequest(
            task_id="task",
            instruction="",
            skill_instructions="skill",
            lm=FakeLM(),
            sub_lm=FakeLM(),
            max_iterations=3,
            task_timeout=30,
            verbose_rlm=False,
            output_dir=tmp_path,
            run_id="gepa-val-task",
            config=config,
        )
    )

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    agent_kwargs = [
        cmd[index + 1]
        for index, value in enumerate(cmd[:-1])
        if value == "--agent-kwarg"
    ]
    assert "lm_reasoning_effort=low" in agent_kwargs
    assert "sub_lm_reasoning_effort=low" in agent_kwargs


def test_agent_builds_low_effort_lms_from_agent_kwargs(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakePredictRLM:
        def __init__(self, signature, **kwargs) -> None:
            captured["signature"] = signature
            captured["kwargs"] = kwargs

        def __call__(self, **_kwargs):
            return SimpleNamespace(answer="done")

    class FakeInterpreter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self) -> None:
            pass

    class FakeDspy:
        class LM:
            def __init__(self, model, **kwargs) -> None:
                self.model = model
                self.kwargs = kwargs

    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(tbench_agent, "TerminalBenchRunnerInterpreter", FakeInterpreter)
    monkeypatch.setattr(tbench_agent, "dspy", FakeDspy)

    agent = tbench_agent.TerminalBenchRLMBaseAgent(
        lm="openai/gpt-5.5",
        sub_lm="openai/gpt-5.5",
        lm_reasoning_effort="low",
        sub_lm_reasoning_effort="low",
    )
    agent.perform_task("solve it", SimpleNamespace(container=object()))

    kwargs = captured["kwargs"]
    lm = kwargs["lm"]
    sub_lm = kwargs["sub_lm"]
    assert lm.model == "openai/gpt-5.5"
    assert lm.kwargs["reasoning_effort"] == "low"
    assert sub_lm.model == "openai/gpt-5.5"
    assert sub_lm.kwargs["reasoning_effort"] == "low"


def test_subprocess_runner_synthesizes_trace_when_agent_does_not_export_one(
    monkeypatch, tmp_path: Path
) -> None:
    config = default_config()
    config.terminal_bench_output_dir = tmp_path / "tbench-runs"
    run_id = "gepa-val-task"
    run_dir = config.terminal_bench_output_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(
        json.dumps({"results": [{"task_id": "task", "is_resolved": True, "parser_results": {}}]})
    )

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = TerminalBenchSubprocessHarnessRunner(cwd=tmp_path)._run_sync(
        TerminalBenchTaskRunRequest(
            task_id="task",
            instruction="",
            skill_instructions="skill",
            lm="main",
            sub_lm="sub",
            max_iterations=3,
            task_timeout=30,
            verbose_rlm=False,
            output_dir=tmp_path,
            run_id=run_id,
            config=config,
        )
    )

    assert result.error is None
    assert len(result.traces) == 1
    assert result.traces[0].model == "main"
    assert result.traces[0].sub_model == "sub"
    assert result.traces[0].max_iterations == 3
