from __future__ import annotations

import argparse
import ast
import asyncio
import json
import re
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    _seed_skill_instructions,
    phase_duration_summary,
)
from terminal_bench_rlm.skills import DEFAULT_TERMINAL_BENCH_SKILL_INSTRUCTIONS  # noqa: E402
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
    assert payload["max_iterations"] == 20


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


def test_cli_codex_lm_missing_dependency_points_to_local_extra(monkeypatch) -> None:
    parser = argparse.ArgumentParser()
    gepa_cli._add_project_args(parser)
    args = parser.parse_args(["--codex-lm"])
    monkeypatch.setattr(gepa_cli.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(RuntimeError) as exc_info:
        gepa_cli._install_codex_lm(args)

    message = str(exc_info.value)
    assert "predict-rlm[codex-lm" in message
    assert "dspy-codex-lm" not in message


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
        run_dir.mkdir(parents=True, exist_ok=True)
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
    assert "task_id=task" in cmd
    assert f"phase_log_path={config.terminal_bench_output_dir / 'gepa-val-task' / 'task_phase_events.jsonl'}" in cmd
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["timeout"] == 2760
    assert "stdout" not in kwargs
    assert "stderr" not in kwargs
    assert result.error is None
    assert result.trial_result["verifier_result"]["rewards"]["reward"] == 1.0


def test_harbor_subprocess_runner_retries_transient_registry_exception_result(
    monkeypatch, tmp_path: Path
) -> None:
    config = default_config()
    config.terminal_bench_output_dir = tmp_path / "harbor-runs"
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        run_dir = config.terminal_bench_output_dir / "gepa-val-task"
        run_dir.mkdir(parents=True, exist_ok=True)
        if len(calls) == 1:
            trial_result = {
                "task_name": "task",
                "exception_info": {
                    "exception_type": "RuntimeError",
                    "exception_message": (
                        "failed to fetch anonymous token: unexpected status from GET request "
                        "to https://auth.docker.io/token: 500 Internal Server Error"
                    ),
                },
            }
        else:
            trial_result = {
                "task_name": "task",
                "exception_info": None,
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        (run_dir / "result.json").write_text(json.dumps({"trial_results": [trial_result]}))
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
            task_timeout=30,
            verbose_rlm=False,
            output_dir=tmp_path,
            run_id="gepa-val-task",
            config=config,
        )
    )

    assert len(calls) == 2
    assert result.error is None
    assert result.trial_result["exception_info"] is None
    assert result.trial_result["verifier_result"]["rewards"]["reward"] == 1.0


def test_harbor_subprocess_runner_does_not_retry_non_registry_exception_result(
    monkeypatch, tmp_path: Path
) -> None:
    config = default_config()
    config.terminal_bench_output_dir = tmp_path / "harbor-runs"
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        run_dir = config.terminal_bench_output_dir / "gepa-val-task"
        run_dir.mkdir(parents=True, exist_ok=True)
        trial_result = {
            "task_name": "task",
            "exception_info": {
                "exception_type": "RuntimeError",
                "exception_message": "image parser service returned 500 Internal Server Error",
            },
        }
        (run_dir / "result.json").write_text(json.dumps({"trial_results": [trial_result]}))
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
            task_timeout=30,
            verbose_rlm=False,
            output_dir=tmp_path,
            run_id="gepa-val-task",
            config=config,
        )
    )

    assert len(calls) == 1
    assert result.trial_result["exception_info"]["exception_message"] == (
        "image parser service returned 500 Internal Server Error"
    )


def test_phase_duration_summary_aggregates_task_phase_event_logs(tmp_path: Path) -> None:
    task_a_log = tmp_path / "jobs" / "run-a" / "task_phase_events.jsonl"
    task_a_log.parent.mkdir(parents=True)
    task_a_log.write_text(
        "\n".join(
            json.dumps(event)
            for event in [
                {
                    "task_id": "terminal-bench/a",
                    "phase": "agent_setup",
                    "event": "agent_setup_end",
                    "status": "completed",
                    "duration_seconds": 1.25,
                },
                {
                    "task_id": "terminal-bench/a",
                    "phase": "agent_eval",
                    "event": "agent_run_end",
                    "status": "completed",
                    "duration_seconds": 10.5,
                },
                {
                    "task_id": "terminal-bench/a",
                    "phase": "sandbox_setup",
                    "event": "sandbox_setup_end",
                    "status": "completed",
                    "duration_seconds": 2.0,
                },
            ]
        )
        + "\n"
    )
    task_b_log = tmp_path / "jobs" / "run-b" / "task_phase_events.jsonl"
    task_b_log.parent.mkdir(parents=True)
    task_b_log.write_text(
        json.dumps(
            {
                "task_id": "terminal-bench/b",
                "phase": "agent_eval",
                "event": "agent_run_end",
                "status": "failed",
                "duration_seconds": 4,
            }
        )
        + "\n"
    )

    summary = phase_duration_summary(tmp_path)

    assert summary == {
        "phase_totals": {
            "agent_eval": {"duration_seconds": 14.5, "events": 2},
            "agent_setup": {"duration_seconds": 1.25, "events": 1},
            "sandbox_setup": {"duration_seconds": 2.0, "events": 1},
        },
        "tasks": {
            "terminal-bench/a": {
                "duration_seconds": 13.75,
                "phases": {
                    "agent_eval": {"duration_seconds": 10.5, "events": 1},
                    "agent_setup": {"duration_seconds": 1.25, "events": 1},
                    "sandbox_setup": {"duration_seconds": 2.0, "events": 1},
                },
            },
            "terminal-bench/b": {
                "duration_seconds": 4.0,
                "phases": {"agent_eval": {"duration_seconds": 4.0, "events": 1}},
            },
        },
        "total_logged_duration_seconds": 17.75,
    }


def test_harbor_subprocess_runner_writes_task_phase_events(monkeypatch, tmp_path: Path) -> None:
    config = default_config()
    config.terminal_bench_output_dir = tmp_path / "harbor-runs"
    config.harbor_task_cache_dir = tmp_path / "harbor-cache"
    task_dir = config.harbor_task_cache_dir / "terminal-bench" / "task" / "sha"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(
        """
[agent]
timeout_sec = 900.0

[verifier]
timeout_sec = 900.0

[environment]
build_timeout_sec = 900.0
""".strip()
    )

    def fake_run(cmd, **_kwargs):
        run_dir = config.terminal_bench_output_dir / "gepa-val-task"
        run_dir.mkdir(parents=True, exist_ok=True)
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

    HarborSubprocessHarnessRunner(cwd=tmp_path)._run_sync(
        TerminalBenchTaskRunRequest(
            task_id="terminal-bench/task",
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

    phase_log = config.terminal_bench_output_dir / "gepa-val-task" / "task_phase_events.jsonl"
    events = [json.loads(line) for line in phase_log.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "harbor_subprocess_start",
        "harbor_subprocess_end",
    ]
    assert events[0]["phase"] == "environment_setup"
    assert events[0]["task_id"] == "terminal-bench/task"
    assert events[0]["dataset"] == "terminal-bench/terminal-bench-2-1"
    assert events[0]["agent_timeout_seconds"] == 900
    assert events[0]["outer_timeout_seconds"] == 2760
    assert events[1]["duration_seconds"] >= 0



def test_harbor_subprocess_runner_uses_official_task_timeout_components(
    monkeypatch, tmp_path: Path
) -> None:
    config = default_config()
    config.terminal_bench_output_dir = tmp_path / "harbor-runs"
    config.harbor_task_cache_dir = tmp_path / "harbor-cache"
    task_dir = config.harbor_task_cache_dir / "terminal-bench" / "task" / "sha"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(
        """
[agent]
timeout_sec = 90.0

[verifier]
timeout_sec = 45.0

[environment]
build_timeout_sec = 15.0
""".strip()
    )
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    HarborSubprocessHarnessRunner(cwd=tmp_path)._run_sync(
        TerminalBenchTaskRunRequest(
            task_id="terminal-bench/task",
            instruction="",
            skill_instructions="skill",
            lm="main",
            sub_lm="sub",
            max_iterations=3,
            task_timeout=1800,
            verbose_rlm=False,
            output_dir=tmp_path,
            run_id="gepa-val-task",
            config=config,
        )
    )

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "exec_timeout=90" in cmd
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["timeout"] == 210


def test_harbor_subprocess_runner_uses_global_task_cache_when_run_cache_is_empty(
    monkeypatch, tmp_path: Path
) -> None:
    config = default_config()
    config.terminal_bench_output_dir = tmp_path / "harbor-runs"
    config.harbor_task_cache_dir = tmp_path / "empty-run-cache"
    global_task_dir = tmp_path / "home" / ".cache" / "harbor" / "tasks" / "packages" / "terminal-bench" / "task" / "sha"
    global_task_dir.mkdir(parents=True)
    (global_task_dir / "task.toml").write_text(
        """
[agent]
timeout_sec = 90.0

[verifier]
timeout_sec = 45.0

[environment]
build_timeout_sec = 15.0
""".strip()
    )
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(subprocess, "run", fake_run)

    HarborSubprocessHarnessRunner(cwd=tmp_path)._run_sync(
        TerminalBenchTaskRunRequest(
            task_id="terminal-bench/task",
            instruction="",
            skill_instructions="skill",
            lm="main",
            sub_lm="sub",
            max_iterations=3,
            task_timeout=1800,
            verbose_rlm=False,
            output_dir=tmp_path,
            run_id="gepa-val-task",
            config=config,
        )
    )

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "exec_timeout=90" in cmd
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["timeout"] == 210


def test_harbor_subprocess_runner_fails_fast_without_official_real_task_timeouts(
    monkeypatch, tmp_path: Path
) -> None:
    config = default_config()
    config.terminal_bench_output_dir = tmp_path / "harbor-runs"
    config.harbor_task_cache_dir = tmp_path / "empty-run-cache"

    def fake_run(*_args, **_kwargs):
        raise AssertionError("harbor run should not launch without task.toml timeouts")

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="official Harbor timeouts"):
        HarborSubprocessHarnessRunner(cwd=tmp_path)._run_sync(
            TerminalBenchTaskRunRequest(
                task_id="terminal-bench/task",
                instruction="",
                skill_instructions="skill",
                lm="main",
                sub_lm="sub",
                max_iterations=3,
                task_timeout=1800,
                verbose_rlm=False,
                output_dir=tmp_path,
                run_id="gepa-val-task",
                config=config,
            )
        )


def test_harbor_subprocess_runner_times_out_inside_outer_harbor_budget(monkeypatch, tmp_path: Path) -> None:
    config = default_config()
    config.terminal_bench_output_dir = tmp_path / "harbor-runs"
    captured: dict[str, object] = {}
    long_stdout = "started\n" + ("o" * 5000)
    long_stderr = "still running\n" + ("e" * 5000)

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        raise subprocess.TimeoutExpired(
            cmd=cmd,
            timeout=kwargs["timeout"],
            output=long_stdout,
            stderr=long_stderr,
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = HarborSubprocessHarnessRunner(cwd=tmp_path)._run_sync(
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
            run_id="gepa-val-task",
            config=config,
        )
    )

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["timeout"] == 150
    assert result.error is not None
    assert result.error.startswith("Terminal-Bench CLI timed out after 150s")
    exception_info = result.trial_result["exception_info"]
    assert exception_info["exception_type"] == "HarnessTimeoutError"
    assert exception_info["phase"] == "harness_subprocess"
    assert exception_info["timed_out"] is True
    assert exception_info["timeout_seconds"] == 150
    assert exception_info["stdout_tail"].startswith("o")
    assert exception_info["stdout_tail"].endswith("o" * 20)
    assert len(exception_info["stdout_tail"]) <= 2000
    assert exception_info["stderr_tail"].startswith("e")
    assert exception_info["stderr_tail"].endswith("e" * 20)
    assert len(exception_info["stderr_tail"]) <= 2000


def test_harbor_subprocess_runner_failure_records_bounded_diagnostics(
    monkeypatch, tmp_path: Path
) -> None:
    config = default_config()
    config.terminal_bench_output_dir = tmp_path / "harbor-runs"
    long_stdout = "setup\n" + ("a" * 5000)
    long_stderr = "boom\n" + ("b" * 5000)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=137, stdout=long_stdout, stderr=long_stderr)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = HarborSubprocessHarnessRunner(cwd=tmp_path)._run_sync(
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
            run_id="gepa-val-task",
            config=config,
        )
    )

    exception_info = result.trial_result["exception_info"]
    assert exception_info["exception_type"] == "HarnessSubprocessError"
    assert exception_info["phase"] == "harness_subprocess"
    assert exception_info["returncode"] == 137
    assert exception_info["stdout_tail"].startswith("a")
    assert exception_info["stdout_tail"].endswith("a" * 20)
    assert len(exception_info["stdout_tail"]) <= 2000
    assert exception_info["stderr_tail"].startswith("b")
    assert exception_info["stderr_tail"].endswith("b" * 20)
    assert len(exception_info["stderr_tail"]) <= 2000


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


def test_seed_candidate_uses_shared_default_terminal_bench_skill_text() -> None:
    skill = TerminalBenchGepaProject(default_config()).seed_candidate()[COMPONENT_SKILL]

    assert skill == DEFAULT_TERMINAL_BENCH_SKILL_INSTRUCTIONS
    assert _seed_skill_instructions() == DEFAULT_TERMINAL_BENCH_SKILL_INSTRUCTIONS


def test_default_terminal_bench_skill_includes_concurrent_timeout_snippet() -> None:
    skill = DEFAULT_TERMINAL_BENCH_SKILL_INSTRUCTIONS
    normalized_skill = " ".join(skill.split())
    headings = [
        "Operating principle",
        "Inspection and changes",
        "Timeouts and long-running work",
        "Problem-solving strategy",
        "Required verification and final QA",
        "Verification and final submission",
    ]
    bad_required_verification_prefix = "+Required" + " verification:"
    obsolete_schema_terms = [
        "acceptance" + "_contract",
        "expected" + "_final_state",
        "status: " + '"pending|verified|blocked"',
    ]

    assert [skill.index(heading) for heading in headings] == sorted(
        skill.index(heading) for heading in headings
    )
    assert "command-line tasks in a Linux environment" in skill
    assert "Terminal-Bench tasks inside a Linux task container" not in skill
    assert "inspect the filesystem before making changes" in skill
    assert "package managers" in skill
    assert "small inspectable steps" in skill
    assert "1-5 seconds" in skill
    assert "10-60 seconds" in skill
    assert "several minutes" in skill
    assert "commands, network requests, and computations" in skill
    assert "query-optimize" not in skill.lower()
    assert "sqlite" not in skill.lower()
    assert "unobserved verification command" in skill
    assert bad_required_verification_prefix not in skill
    assert "@dataclass" in skill
    assert "class RequiredVerification" in skill
    assert "requirement: str" in skill
    assert "verification: str" in skill
    assert "required verification list" in skill
    assert "required checks" in skill
    assert "short list" in skill
    assert "extracted from the task" in skill
    assert "verification:" in skill
    assert "schema" not in skill.lower()
    assert "yaml" not in skill.lower()
    assert all(term not in skill for term in obsolete_schema_terms)
    assert "ledger" not in skill.lower()
    assert "task requirements" in skill
    assert "Before SUBMIT" in skill
    assert "proportional evidence" in skill
    assert "literal paths/endpoints" in skill
    assert "config values" in skill
    assert "processes or services" in normalized_skill
    assert "absolute minimum" in skill
    assert "files, processes, services, and configs" in skill
    assert "initial state" in skill
    assert "no extra modified files" in skill
    assert "copied artifacts" in skill
    assert "debug helpers" in skill
    assert "alternate runtime artifacts" in normalized_skill
    assert "temporary services" in skill
    assert "config side effects" in skill
    assert "paths, endpoints, flags, and config values named by the task" in normalized_skill
    assert "visible tests" in skill
    assert "verifier-shaped checks" in skill
    assert "hidden tests" in skill
    assert "parse/load/exercise" in skill
    assert "semantic/reference" in skill
    assert "stdout/progress text" in skill
    assert "command behavior" in skill
    assert "emulator, interpreter, VM, service, or wrapper tasks" in skill
    assert "named binary, program, protocol, or mechanism" in normalized_skill
    assert "shortcut or native/source-level stand-in" in skill
    assert "negative constraints" in normalized_skill
    assert "debug/runtime state" in skill
    assert "stdout/stderr" in skill
    assert "exit code" in normalized_skill or "exit codes" in normalized_skill
    assert "service behavior" in skill
    assert "SUBMIT makes the result final" in skill
    assert "stale debug history" in skill
    assert "Once the observable task contract is satisfied" not in skill
    assert "run the verification in one iteration" not in skill
    assert "separate later iteration" not in skill
    assert "always run the full verifier" not in skill.lower()
    assert "must reproduce the full verifier" not in skill.lower()
    for term in ["windows", "win311", "qemu", "mips", "bmp", "doom", "PIL"]:
        assert re.search(rf"\b{re.escape(term)}\b", skill, re.IGNORECASE) is None

    snippet = skill.split("```python\n", 1)[1].split("\n```", 1)[0]
    compile(snippet, "<terminal-bench-skill>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    comment_anchors = [
        (
            "# Use run() for bounded foreground commands; inspect output before continuing.",
            "async def run(cmd, timeout=60):",
        ),
        (
            "# Use requests timeouts for network calls.",
            "response = requests.get(url, timeout=10)",
        ),
        (
            "# Use asyncio.wait_for for expensive computations or async work that may hang.",
            "computation = await asyncio.wait_for",
        ),
        (
            "# Use asyncio.gather for independent non-mutating checks that can run concurrently.",
            "results = await asyncio.gather",
        ),
    ]
    for comment, code_anchor in comment_anchors:
        assert comment in snippet
        assert code_anchor in snippet
        assert snippet.index(comment) < snippet.index(code_anchor)

    for anchor in [
        "async def run(cmd, timeout=60):",
        "subprocess.run",
        "capture_output=True",
        "timeout=timeout",
        "requests.get(url, timeout=10)",
        "asyncio.wait_for",
        "await asyncio.gather",
        "print(result.returncode)",
        "print(result.stdout[-2000:])",
        "print(result.stderr[-2000:])",
    ]:
        assert anchor in snippet

    for removed_anchor in [
        "async def start",
        "async def wait",
        "await start(",
        "await wait(",
        "subprocess.Popen",
        "stdout_tail",
        "stderr_tail",
        "job = await start",
        "progress = await wait",
    ]:
        assert removed_anchor not in snippet


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
        skill_instructions=(
            "Make task changes boldly in small inspectable steps and verify files "
            "before finishing."
        ),
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
    assert kwargs["global_agent_timeout_sec"] == 900


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
        kwargs = {"reasoning_effort": "low", "service_tier": "priority"}

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
    assert "lm_service_tier=priority" in agent_kwargs
    assert "sub_lm_service_tier=priority" in agent_kwargs


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
        lm_service_tier="priority",
        sub_lm_service_tier="priority",
    )
    agent.perform_task("solve it", SimpleNamespace(container=object()))

    kwargs = captured["kwargs"]
    lm = kwargs["lm"]
    sub_lm = kwargs["sub_lm"]
    assert lm.model == "openai/gpt-5.5"
    assert lm.kwargs["reasoning_effort"] == "low"
    assert lm.kwargs["service_tier"] == "priority"
    assert sub_lm.model == "openai/gpt-5.5"
    assert sub_lm.kwargs["reasoning_effort"] == "low"
    assert sub_lm.kwargs["service_tier"] == "priority"


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
