from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tarfile
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from terminal_bench_rlm.gepa import cli as gepa_cli  # noqa: E402
from terminal_bench_rlm.gepa import project as gepa_project  # noqa: E402
from terminal_bench_rlm.gepa.config import COMPONENT_SKILL, default_config  # noqa: E402
from terminal_bench_rlm.gepa.project import (  # noqa: E402
    DaytonaRemoteControllerEnvironment,
    HarborControllerLocality,
    HarborRemoteControllerHarnessRunner,
    HarborSubprocessHarnessRunner,
    LocalShellRemoteControllerEnvironment,
    RemoteCommandResult,
    SbxRemoteControllerEnvironment,
    SshGcpRemoteControllerEnvironment,
    TerminalBenchExample,
    TerminalBenchGepaProject,
    TerminalBenchInProcessHarnessRunner,
    TerminalBenchSubprocessHarnessRunner,
    TerminalBenchTaskRunRequest,
    TerminalBenchTaskRunResult,
    _agent_kwargs,
    _build_harbor_harness_runner,
    _build_harbor_run_command,
    _extract_tarball,
    _seed_skill_instructions,
    _subprocess_env,
    phase_duration_summary,
    select_harbor_controller_locality,
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


class RecordingHarnessRunner:
    def __init__(self) -> None:
        self.calls: list[TerminalBenchTaskRunRequest] = []

    async def run(self, request):
        self.calls.append(request)
        return TerminalBenchTaskRunResult(
            task_id=request.task_id,
            trial_result={"verifier_result": {"rewards": {"reward": 1.0}}},
            traces=[],
            run_dir=None,
        )


class FakeInteractiveHarborEnvironment:
    def start_exec(self, command, *, workdir=None, timeout=None):
        return None


class FakeOneShotHarborEnvironment:
    def __init__(self, *, run_id: str = "gepa-val-task", remote_root_exists: bool = False) -> None:
        self.run_id = run_id
        self.remote_root_exists = remote_root_exists
        self.commands: list[str] = []
        self.uploads: list[tuple[str, str]] = []
        self.downloads: list[tuple[str, str]] = []
        self.upload_archive_members: list[str] = []
        self.upload_archive_contents: dict[str, bytes] = {}

    def exec(self, *, command: str, timeout_sec: int):
        self.commands.append(command)
        if self.remote_root_exists and command.startswith("test ! -e "):
            return SimpleNamespace(return_code=1, stdout="", stderr="exists")
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    def upload_file(self, host_path: str, remote_path: str) -> None:
        self.uploads.append((host_path, remote_path))
        if host_path.endswith(".tar.gz"):
            with tarfile.open(host_path, "r:gz") as archive:
                self.upload_archive_members = archive.getnames()
                self.upload_archive_contents = {}
                for member in archive.getmembers():
                    if not member.isfile():
                        continue
                    source = archive.extractfile(member)
                    if source is not None:
                        with source:
                            self.upload_archive_contents[member.name] = source.read()

    def download_file(self, remote_path: str, host_path: str) -> None:
        self.downloads.append((remote_path, host_path))
        with tarfile.open(host_path, "w:gz") as archive:
            result_path = Path(host_path).parent / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "trial_results": [
                            {
                                "task_name": "task",
                                "verifier_result": {"rewards": {"reward": 1.0}},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            archive.add(result_path, arcname=f"{self.run_id}/result.json")


class FakeDaytonaSyncSandbox:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def run(self, command, *, timeout=None):
        self.calls.append(("run", (command,), {"timeout": timeout}))
        return RemoteCommandResult(stdout="sync")

    def copy_to(self, host_path, remote_path):
        self.calls.append(("copy_to", (host_path, remote_path), {}))

    def get_file(self, remote_path, host_path):
        self.calls.append(("get_file", (remote_path, host_path), {}))


class FakeDaytonaAsyncSandbox:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    async def exec(self, *, command, timeout_sec):
        self.calls.append(("exec", (), {"command": command, "timeout_sec": timeout_sec}))
        return RemoteCommandResult(stdout="async")

    async def put_file(self, host_path, remote_path):
        self.calls.append(("put_file", (host_path, remote_path), {}))

    async def copy_from(self, remote_path, host_path):
        self.calls.append(("copy_from", (remote_path, host_path), {}))


class FakeDaytonaSdkSandbox:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.process = SimpleNamespace(exec=self._exec)
        self.fs = SimpleNamespace(upload_file=self._upload_file, download_file=self._download_file)

    def _exec(self, command, *, timeout=None):
        self.calls.append(("process.exec", (command,), {"timeout": timeout}))
        return RemoteCommandResult(stdout="sdk")

    def _upload_file(self, host_path, remote_path):
        self.calls.append(("fs.upload_file", (host_path, remote_path), {}))

    def _download_file(self, remote_path, host_path):
        self.calls.append(("fs.download_file", (remote_path, host_path), {}))


class RecordingDaytonaBootstrapEnvironment:
    def __init__(self) -> None:
        self.commands: list[tuple[str, int | None]] = []
        self.uploads: list[tuple[str, str]] = []
        self.upload_archive_contents: dict[str, bytes] = {}

    async def exec(self, *, command, timeout_sec):
        self.commands.append((command, timeout_sec))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    async def upload_file(self, host_path, remote_path):
        self.uploads.append((host_path, remote_path))
        if host_path.endswith(".tar.gz"):
            with tarfile.open(host_path, "r:gz") as archive:
                self.upload_archive_contents = {}
                for member in archive.getmembers():
                    if not member.isfile():
                        continue
                    source = archive.extractfile(member)
                    if source is not None:
                        with source:
                            self.upload_archive_contents[member.name] = source.read()


def _task_request(config, tmp_path: Path, *, run_id: str = "gepa-val-task"):
    return TerminalBenchTaskRunRequest(
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


def test_project_validation_check_has_non_empty_train_and_val_examples() -> None:
    project = TerminalBenchGepaProject(default_config())

    validation = validate_project(project)

    assert validation.seed_candidate[COMPONENT_SKILL].strip()
    assert len(validation.trainset) >= 1
    assert len(validation.valset) >= 1


@pytest.mark.asyncio
async def test_project_loads_task_timeout_and_resources_before_launch(tmp_path: Path) -> None:
    config = default_config()
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
cpus = 2
memory_mb = 4096
storage_mb = 10240
gpus = 1
""".strip(),
        encoding="utf-8",
    )
    harness_runner = RecordingHarnessRunner()
    project = TerminalBenchGepaProject(config, harness_runner=harness_runner)

    await project.evaluate_example(
        {COMPONENT_SKILL: "skill"},
        TerminalBenchExample("terminal-bench/task"),
        EvaluationContext(
            lm="main",
            sub_lm="sub",
            max_iterations=3,
            task_timeout=1800,
            output_dir=tmp_path,
            kind="valset",
        ),
    )

    request = harness_runner.calls[0]
    assert request.task_timeout == 210
    assert request.task_resources == {
        "cpus": 2,
        "memory_mb": 4096,
        "storage_mb": 10240,
        "gpus": 1,
    }



def test_config_serializes_terminal_bench_fields_for_run_metadata() -> None:
    payload = default_config().to_dict()

    json.dumps(payload)
    assert "harness_backend" not in payload
    assert payload["harbor_dataset"] == "terminal-bench/terminal-bench-2-1"
    assert payload["harbor_environment"] == "docker"
    assert payload["harbor_controller_locality"] == "auto"
    assert "harbor_agent_interpreter_mode" not in payload
    assert payload["harbor_remote_workdir"] == "/tmp/predict_rlm_terminal_bench"
    assert payload["terminal_bench_output_dir"] == "runs/gepa-terminal-bench"
    assert payload["train_task_ids"] == ["configure-git-webserver", "extract-moves-from-video"]
    assert payload["val_task_ids"] == ["super-benchmark-upet"]
    assert payload["max_iterations"] == 50


def test_cli_accepts_harbor_executable_args_without_backend_choice() -> None:
    parser = argparse.ArgumentParser()
    gepa_cli._add_project_args(parser)
    args = parser.parse_args(
        [
            "--harbor-executable",
            "uvx harbor",
            "--harbor-dataset",
            "terminal-bench/terminal-bench-2",
            "--harbor-environment",
            "daytona",
            "--harbor-controller-locality",
            "local-controller",
            "--harbor-remote-workdir",
            "/remote/tb",
        ]
    )

    config = gepa_cli._apply_project_args(default_config(), args)

    assert not hasattr(config, "harness_backend")
    assert not hasattr(config, "harbor_agent_interpreter_mode")
    assert config.harbor_executable == "uvx harbor"
    assert config.harbor_dataset == "terminal-bench/terminal-bench-2"
    assert config.harbor_environment == "daytona"
    assert config.harbor_controller_locality == "local-controller"
    assert config.harbor_remote_workdir == "/remote/tb"


def test_cli_help_advertises_remote_controller_as_supplied_machine() -> None:
    parser = argparse.ArgumentParser()
    gepa_cli._add_project_args(parser)

    help_text = parser.format_help()

    assert "the Harbor host process inside a supplied controller" in help_text
    assert "--harness-backend" not in help_text
    assert "--harbor-agent-interpreter-mode" not in help_text
    assert "unsupported for Daytona" not in help_text


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


def test_build_project_ignores_removed_python_harness_backend_attribute() -> None:
    config = default_config()
    config.harness_backend = "python"
    project = TerminalBenchGepaProject(config)

    assert isinstance(project.harness_runner, HarborSubprocessHarnessRunner)


def test_build_project_ignores_removed_cli_harness_backend_attribute() -> None:
    config = default_config()
    config.harness_backend = "cli"
    project = TerminalBenchGepaProject(config)

    assert isinstance(project.harness_runner, HarborSubprocessHarnessRunner)


def test_build_project_explicit_remote_controller_requires_supplied_environment() -> None:
    config = default_config()
    config.harbor_controller_locality = "remote-controller"

    with pytest.raises(RuntimeError) as exc_info:
        gepa_project.build_project(config)

    message = str(exc_info.value)
    assert "Harbor remote-controller requires a Harbor/Daytona environment object" in message
    assert "build_project(config) cannot construct that environment" in message
    assert "must supply it to the lower-level Harbor runner" in message


def test_harbor_controller_auto_chooses_local_controller_for_interactive_exec() -> None:
    selection = select_harbor_controller_locality(
        "auto",
        FakeInteractiveHarborEnvironment(),
    )

    assert selection.locality is HarborControllerLocality.LOCAL_CONTROLLER
    assert "interactive exec" in selection.reason


def test_harbor_controller_auto_chooses_remote_controller_for_one_shot_file_sync() -> None:
    selection = select_harbor_controller_locality(
        "auto",
        FakeOneShotHarborEnvironment(),
    )

    assert selection.locality is HarborControllerLocality.REMOTE_CONTROLLER
    assert "one-shot exec" in selection.reason


def test_harbor_controller_auto_recognizes_remote_controller_adapters() -> None:
    adapters = [
        LocalShellRemoteControllerEnvironment(),
        SshGcpRemoteControllerEnvironment("gcp-vm"),
        SbxRemoteControllerEnvironment(sandbox_id="sandbox-123"),
        DaytonaRemoteControllerEnvironment(FakeDaytonaSyncSandbox()),
    ]

    selections = [select_harbor_controller_locality("auto", adapter) for adapter in adapters]

    assert [selection.locality for selection in selections] == [
        HarborControllerLocality.REMOTE_CONTROLLER,
        HarborControllerLocality.REMOTE_CONTROLLER,
        HarborControllerLocality.REMOTE_CONTROLLER,
        HarborControllerLocality.REMOTE_CONTROLLER,
    ]


def test_harbor_controller_local_controller_fails_without_interactive_exec() -> None:
    env = FakeOneShotHarborEnvironment()

    with pytest.raises(RuntimeError, match="persistent interactive exec"):
        select_harbor_controller_locality("local-controller", env)

    assert env.commands == []


def test_daytona_auto_rejects_host_local_controller_fallback_for_interactive_exec() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        select_harbor_controller_locality(
            "auto",
            FakeInteractiveHarborEnvironment(),
            harbor_environment="daytona",
        )

    message = str(exc_info.value)
    assert "Daytona" in message
    assert "remote-controller" in message
    assert "local-controller" in message


def test_build_harbor_harness_runner_uses_remote_controller_for_auto_one_shot_env() -> None:
    config = default_config()
    runner = _build_harbor_harness_runner(
        config,
        controller_environment=FakeOneShotHarborEnvironment(),
    )

    assert isinstance(runner, HarborRemoteControllerHarnessRunner)


def test_daytona_harbor_config_requires_controller_environment() -> None:
    config = default_config()
    config.harbor_environment = "daytona"

    with pytest.raises(RuntimeError) as exc_info:
        _build_harbor_harness_runner(config)

    message = str(exc_info.value)
    assert "Daytona" in message
    assert "controller environment" in message
    assert "build_project(config) cannot construct it" in message
    assert "host launcher" in message


def test_harbor_controller_auto_rejects_daytona_without_controller_environment() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        select_harbor_controller_locality(
            "auto",
            None,
            harbor_environment="daytona",
        )

    message = str(exc_info.value)
    assert "Daytona" in message
    assert "controller environment" in message
    assert "remote-controller" in message
    assert "local controller" in message


def test_daytona_harbor_config_uses_supplied_remote_controller_environment() -> None:
    config = default_config()
    config.harbor_environment = "daytona"

    runner = _build_harbor_harness_runner(
        config,
        controller_environment=FakeOneShotHarborEnvironment(),
    )

    assert isinstance(runner, HarborRemoteControllerHarnessRunner)


def test_daytona_agent_kwargs_do_not_expose_interpreter_mode(tmp_path: Path) -> None:
    config = default_config()
    config.harbor_environment = "daytona"
    request = _task_request(config, tmp_path)

    kwargs = _agent_kwargs(request)
    cmd = _build_harbor_run_command(request, output_dir=tmp_path / "harbor-runs")

    assert "interpreter_mode" not in kwargs
    assert kwargs["submit_confirmation_mode"] == "terminal_bench"
    assert "submit_confirmation_mode=terminal_bench" in cmd
    assert "interpreter_mode=" not in cmd
    assert "remote-controller" not in cmd


def test_docker_agent_kwargs_do_not_expose_interpreter_mode(tmp_path: Path) -> None:
    config = default_config()
    config.harbor_environment = "docker"

    kwargs = _agent_kwargs(_task_request(config, tmp_path))

    assert "interpreter_mode" not in kwargs
    assert kwargs["submit_confirmation_mode"] == "terminal_bench"


def test_daytona_explicit_remote_controller_requires_supplied_environment() -> None:
    config = default_config()
    config.harbor_environment = "daytona"
    config.harbor_controller_locality = "remote-controller"

    with pytest.raises(RuntimeError) as exc_info:
        _build_harbor_harness_runner(config)

    message = str(exc_info.value)
    assert "requires an explicit Daytona controller environment" in message


def test_local_shell_remote_controller_runs_and_syncs_files(tmp_path: Path) -> None:
    env = LocalShellRemoteControllerEnvironment()
    source = tmp_path / "source.txt"
    source.write_text("artifact\n", encoding="utf-8")
    remote_source = tmp_path / "remote" / "source.txt"
    remote_output = tmp_path / "remote" / "output.txt"
    downloaded = tmp_path / "downloaded.txt"

    env.upload_file(str(source), str(remote_source))
    result = env.exec(
        command=f"cp {remote_source} {remote_output}",
        timeout_sec=10,
    )
    env.download_file(str(remote_output), str(downloaded))

    assert result.returncode == 0
    assert remote_source.read_text(encoding="utf-8") == "artifact\n"
    assert downloaded.read_text(encoding="utf-8") == "artifact\n"


def test_ssh_gcp_remote_controller_builds_ssh_and_scp_commands(monkeypatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(gepa_project.subprocess, "run", fake_run)
    env = SshGcpRemoteControllerEnvironment(
        "gcp-vm",
        ssh_args=("-i", "key.pem"),
        scp_args=("-i", "key.pem"),
    )

    result = env.exec(command="echo ok", timeout_sec=12)
    env.upload_file("/host/repo.tar.gz", "/remote/repo.tar.gz")
    env.download_file("/remote/artifacts.tar.gz", "/host/artifacts.tar.gz")

    assert result.returncode == 0
    assert [call[0] for call in calls] == [
        ["ssh", "-i", "key.pem", "gcp-vm", "echo ok"],
        ["scp", "-i", "key.pem", "/host/repo.tar.gz", "gcp-vm:/remote/repo.tar.gz"],
        ["scp", "-i", "key.pem", "gcp-vm:/remote/artifacts.tar.gz", "/host/artifacts.tar.gz"],
    ]
    assert calls[0][1]["timeout"] == 12


def test_sbx_remote_controller_builds_create_exec_cp_and_rm_commands(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    host_artifacts = tmp_path / "artifacts.tar.gz"

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        stdout = "sandbox-123\n" if cmd[:2] == ["sbx", "create"] else ""
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(gepa_project.subprocess, "run", fake_run)
    env = SbxRemoteControllerEnvironment(
        workspace=tmp_path / "sbx-workspace",
        create_args=("--template", "ubuntu:22.04"),
    )

    result = env.exec(command="echo ok", timeout_sec=13)
    env.upload_file("/host/repo.tar.gz", "/remote/repo.tar.gz")
    env.download_file("/remote/artifacts.tar.gz", str(host_artifacts))
    env.close()

    assert result.returncode == 0
    assert calls == [
        ["sbx", "create", "shell", str(tmp_path / "sbx-workspace"), "--template", "ubuntu:22.04"],
        ["sbx", "exec", "sandbox-123", "sh", "-lc", "echo ok"],
        ["sbx", "cp", "/host/repo.tar.gz", "sandbox-123:/remote/repo.tar.gz"],
        ["sbx", "cp", "sandbox-123:/remote/artifacts.tar.gz", str(host_artifacts)],
        ["sbx", "rm", "sandbox-123"],
    ]


def test_daytona_remote_controller_wraps_sync_async_and_sdk_method_names() -> None:
    sync_sandbox = FakeDaytonaSyncSandbox()
    sync_env = DaytonaRemoteControllerEnvironment(sync_sandbox)

    assert sync_env.exec(command="echo sync", timeout_sec=7).stdout == "sync"
    sync_env.upload_file("/host/a", "/remote/a")
    sync_env.download_file("/remote/a", "/host/a")

    assert sync_sandbox.calls == [
        ("run", ("echo sync",), {"timeout": 7}),
        ("copy_to", ("/host/a", "/remote/a"), {}),
        ("get_file", ("/remote/a", "/host/a"), {}),
    ]

    async_sandbox = FakeDaytonaAsyncSandbox()
    async_env = DaytonaRemoteControllerEnvironment(async_sandbox)

    assert async_env.exec(command="echo async", timeout_sec=8).stdout == "async"
    async_env.upload_file("/host/a", "/remote/a")
    async_env.download_file("/remote/a", "/host/a")

    assert async_sandbox.calls == [
        ("exec", (), {"command": "echo async", "timeout_sec": 8}),
        ("put_file", ("/host/a", "/remote/a"), {}),
        ("copy_from", ("/remote/a", "/host/a"), {}),
    ]

    sdk_sandbox = FakeDaytonaSdkSandbox()
    sdk_env = DaytonaRemoteControllerEnvironment(sdk_sandbox)

    assert sdk_env.exec(command="echo sdk", timeout_sec=9).stdout == "sdk"
    sdk_env.upload_file("/host/a", "/remote/a")
    sdk_env.download_file("/remote/a", "/host/a")

    assert sdk_sandbox.calls == [
        ("process.exec", ("echo sdk",), {"timeout": 9}),
        ("fs.upload_file", ("/host/a", "/remote/a"), {}),
        ("fs.download_file", ("/remote/a", "/host/a"), {}),
    ]


def test_daytona_remote_agent_exposes_agent_info_without_harbor_dependency() -> None:
    agent = tbench_agent.DaytonaRemotePredictRLMAgent(
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


def test_daytona_installed_source_bundle_contains_bootstrap_at_remote_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(tbench_agent, "_source_checkout_root", lambda start=None: None)

    bundle_path = tmp_path / "repo.tar.gz"
    tbench_agent._create_source_bundle(bundle_path)

    with tarfile.open(bundle_path, "r:gz") as archive:
        members = set(archive.getnames())
        bootstrap = archive.extractfile("repo/src/predict_rlm/remote/bootstrap_controller.sh")
        assert bootstrap is not None
        with bootstrap:
            bootstrap_text = bootstrap.read().decode("utf-8")

    assert "repo/pyproject.toml" in members
    assert "repo/examples/terminal_bench/pyproject.toml" in members
    assert "repo/examples/terminal_bench/terminal_bench_rlm/tools/remote_controller.py" in members
    assert bootstrap_text.startswith("#!/bin/sh\n")


@pytest.mark.asyncio
async def test_daytona_bootstrap_command_uses_packaged_asset_and_requests_python312(
    tmp_path: Path,
) -> None:
    env = RecordingDaytonaBootstrapEnvironment()
    agent = tbench_agent.DaytonaRemotePredictRLMAgent(
        logs_dir=tmp_path,
        lm="openai/gpt-5.4-mini",
        remote_root="/remote/controller",
        remote_home="/remote/home",
    )

    await agent._bootstrap_remote_controller(env)

    assert len(env.uploads) == 1
    assert env.uploads[0][1] == "/remote/controller/repo.tar.gz"
    assert (
        "repo/src/predict_rlm/remote/bootstrap_controller.sh" in env.upload_archive_contents
    )
    assert env.commands[0] == (
        "rm -rf /remote/controller && mkdir -p /remote/controller /remote/home",
        120,
    )
    assert env.commands[1] == ("tar -xzf /remote/controller/repo.tar.gz -C /remote/controller", 120)
    setup_command, setup_timeout = env.commands[2]
    assert setup_timeout == 900
    outer_tokens = shlex.split(setup_command)
    inner_command = outer_tokens[outer_tokens.index("-lc") + 1]
    inner_tokens = shlex.split(inner_command)
    assert inner_tokens == [
        "sh",
        "/remote/controller/repo/src/predict_rlm/remote/bootstrap_controller.sh",
        "--root",
        "/remote/controller",
        "--repo",
        "/remote/controller/repo",
        "--python",
        "3.12",
    ]
    assert agent._remote_setup_complete is True


def test_harbor_runner_builds_harbor_run_command(monkeypatch, tmp_path: Path) -> None:
    config = default_config()
    config.terminal_bench_output_dir = tmp_path / "harbor-runs"
    config.harbor_dataset = "terminal-bench/terminal-bench-2-1"
    config.harbor_environment = "daytona"
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
    assert cmd[:6] == [
        "harbor",
        "run",
        "-d",
        "terminal-bench/terminal-bench-2-1",
        "-e",
        "daytona",
    ]
    assert "--include-task-name" in cmd
    assert cmd[cmd.index("--include-task-name") + 1] == "task"
    assert "--agent-import-path" in cmd
    assert cmd[cmd.index("--agent-import-path") + 1] == (
        "terminal_bench_rlm.tools.tbench_agent:DaytonaRemotePredictRLMAgent"
    )
    assert "--jobs-dir" in cmd
    assert cmd[cmd.index("--jobs-dir") + 1] == str(config.terminal_bench_output_dir)
    assert "--job-name" in cmd
    assert cmd[cmd.index("--job-name") + 1] == "gepa-val-task"
    assert "--agent-timeout" not in cmd
    assert "--n-attempts" in cmd
    assert cmd[cmd.index("--n-attempts") + 1] == "1"
    assert "--max-retries" in cmd
    assert cmd[cmd.index("--max-retries") + 1] == "3"
    assert cmd.count("--retry-include") == 2
    retry_include_indices = [i for i, arg in enumerate(cmd) if arg == "--retry-include"]
    assert [cmd[i + 1] for i in retry_include_indices] == [
        "DaytonaError",
        "DownloadVerifierDirError",
    ]
    assert "--n-concurrent" in cmd
    assert cmd[cmd.index("--n-concurrent") + 1] == "1"
    assert "--cpus" in cmd
    assert cmd[cmd.index("--cpus") + 1] == "auto"
    assert "--memory" in cmd
    assert cmd[cmd.index("--memory") + 1] == "auto"
    assert "--agent-kwarg" in cmd
    assert "exec_timeout=900" in cmd
    assert "task_id=task" in cmd
    assert f"phase_log_path={config.terminal_bench_output_dir / 'gepa-val-task' / 'task_phase_events.jsonl'}" in cmd
    assert "tar" not in cmd
    assert "cd" not in cmd
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["timeout"] == 2760
    assert "stdout" not in kwargs
    assert "stderr" not in kwargs
    assert result.error is None
    assert result.trial_result["verifier_result"]["rewards"]["reward"] == 1.0


def test_harbor_remote_controller_builds_remote_command_and_syncs_artifacts(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    cwd = repo / "examples" / "terminal_bench"
    cwd.mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'predict-rlm'\n", encoding="utf-8")
    (cwd / "pyproject.toml").write_text(
        "[project]\nname = 'terminal-bench-rlm'\n",
        encoding="utf-8",
    )
    config = default_config()
    config.terminal_bench_output_dir = tmp_path / "local-runs"
    config.harbor_environment = "docker"
    config.harbor_remote_workdir = "/remote/tb"
    env = FakeOneShotHarborEnvironment()

    result = HarborRemoteControllerHarnessRunner(env, cwd=cwd)._run_sync(
        _task_request(config, tmp_path)
    )

    assert result.error is None
    assert result.trial_result["verifier_result"]["rewards"]["reward"] == 1.0
    assert env.uploads
    assert env.uploads[0][1] == "/remote/tb/gepa-val-task/repo.tar.gz"
    assert env.downloads == [
        (
            "/remote/tb/gepa-val-task/artifacts.tar.gz",
            env.downloads[0][1],
        )
    ]
    joined_commands = "\n".join(env.commands)
    assert "cd /remote/tb/gepa-val-task/repo/examples/terminal_bench" in joined_commands
    assert "harbor run -d terminal-bench/terminal-bench-2-1 -e docker" in joined_commands
    assert "--jobs-dir /remote/tb/gepa-val-task/harbor-runs" in joined_commands
    assert "--job-name gepa-val-task" in joined_commands
    assert "phase_log_path=/remote/tb/gepa-val-task/harbor-runs/gepa-val-task/task_phase_events.jsonl" in joined_commands
    assert (config.terminal_bench_output_dir / "gepa-val-task" / "result.json").exists()


def test_harbor_remote_controller_allows_daytona_when_controller_is_supplied(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    cwd = repo / "examples" / "terminal_bench"
    cwd.mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'predict-rlm'\n", encoding="utf-8")
    config = default_config()
    config.terminal_bench_output_dir = tmp_path / "local-runs"
    config.harbor_environment = "daytona"
    env = FakeOneShotHarborEnvironment()

    result = HarborRemoteControllerHarnessRunner(env, cwd=cwd)._run_sync(
        _task_request(config, tmp_path)
    )

    assert result.error is None
    joined_commands = "\n".join(env.commands)
    assert "harbor run -d terminal-bench/terminal-bench-2-1 -e daytona" in joined_commands
    assert "--max-retries 3" in joined_commands
    assert "--retry-include DaytonaError" in joined_commands
    assert "--retry-include DownloadVerifierDirError" in joined_commands
    assert env.uploads
    assert env.downloads


def test_harbor_remote_controller_refuses_existing_remote_root_before_unpack(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    cwd = repo / "examples" / "terminal_bench"
    cwd.mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'predict-rlm'\n", encoding="utf-8")
    config = default_config()
    config.terminal_bench_output_dir = tmp_path / "local-runs"
    config.harbor_remote_workdir = "/remote/tb"
    env = FakeOneShotHarborEnvironment(remote_root_exists=True)

    with pytest.raises(RuntimeError) as exc_info:
        HarborRemoteControllerHarnessRunner(env, cwd=cwd)._run_sync(_task_request(config, tmp_path))

    message = str(exc_info.value)
    assert "remote root already exists: /remote/tb/gepa-val-task" in message
    assert "Download/preserve artifacts or use a unique run id" in message
    assert env.uploads == []
    assert all("rm -rf" not in command for command in env.commands)


def test_harbor_remote_controller_does_not_recreate_one_shot_json_rpc_shim(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    cwd = repo / "examples" / "terminal_bench"
    cwd.mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'predict-rlm'\n", encoding="utf-8")
    config = default_config()
    config.terminal_bench_output_dir = tmp_path / "local-runs"
    env = FakeOneShotHarborEnvironment()

    HarborRemoteControllerHarnessRunner(env, cwd=cwd)._run_sync(_task_request(config, tmp_path))

    remote_shell = "\n".join(env.commands)
    assert "jsonrpc" not in remote_shell.lower()
    assert "predict_rlm_runner.py" not in remote_shell
    assert "heredoc" not in remote_shell.lower()
    assert "python3 -u /tmp/predict_rlm_runner.py" not in remote_shell


def test_harbor_remote_controller_package_uploads_only_tracked_files(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    cwd = repo / "examples" / "terminal_bench"
    cwd.mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'predict-rlm'\n", encoding="utf-8")
    (cwd / "pyproject.toml").write_text(
        "[project]\nname = 'terminal-bench-rlm'\n",
        encoding="utf-8",
    )
    (repo / "tracked.txt").write_text("working-tree tracked content\n", encoding="utf-8")
    (repo / "early_failures.txt").write_text("untracked sentinel\n", encoding="utf-8")
    (repo / "bug_reports").mkdir()
    (repo / "bug_reports" / "stale.md").write_text("untracked report\n", encoding="utf-8")
    (repo / ".hermes").mkdir()
    (repo / ".hermes" / "checkpoint.json").write_text("checkpoint\n", encoding="utf-8")

    def fake_git_ls_files(cmd, **kwargs):
        assert cmd == ["git", "ls-files", "-z"]
        assert kwargs["cwd"] == repo
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=(
                b"pyproject.toml\0"
                b"examples/terminal_bench/pyproject.toml\0"
                b"tracked.txt\0"
            ),
            stderr=b"",
        )

    monkeypatch.setattr(gepa_project.subprocess, "run", fake_git_ls_files)
    config = default_config()
    config.terminal_bench_output_dir = tmp_path / "local-runs"
    env = FakeOneShotHarborEnvironment()

    HarborRemoteControllerHarnessRunner(env, cwd=cwd)._run_sync(_task_request(config, tmp_path))

    assert "repo/tracked.txt" in env.upload_archive_members
    assert env.upload_archive_contents["repo/tracked.txt"] == b"working-tree tracked content\n"
    assert "repo/early_failures.txt" not in env.upload_archive_members
    assert "repo/bug_reports/stale.md" not in env.upload_archive_members
    assert "repo/.hermes/checkpoint.json" not in env.upload_archive_members


def test_extract_tarball_rejects_path_traversal_without_tar_filter(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("artifact\n", encoding="utf-8")
    safe_archive = tmp_path / "safe.tar.gz"
    with tarfile.open(safe_archive, "w:gz") as archive:
        archive.add(source, arcname="gepa-val-task/result.txt")

    output_dir = tmp_path / "output"
    _extract_tarball(safe_archive, output_dir)

    assert (output_dir / "gepa-val-task" / "result.txt").read_text(encoding="utf-8") == "artifact\n"

    evil_archive = tmp_path / "evil.tar.gz"
    with tarfile.open(evil_archive, "w:gz") as archive:
        archive.add(source, arcname="../escaped.txt")

    with pytest.raises(RuntimeError, match="outside"):
        _extract_tarball(evil_archive, output_dir)

    assert not (tmp_path / "escaped.txt").exists()


def test_harbor_subprocess_env_loads_repo_env_development_without_overrides(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    cwd = repo / "examples" / "terminal_bench"
    cwd.mkdir(parents=True)
    (repo / ".env.development").write_text(
        "\n".join(
            [
                "# ignored",
                "DAYTONA_API_KEY=from-file",
                "export OPENAI_API_KEY='from-file-openai'",
                "EMPTY_LINE_IGNORED",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell-openai")
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)

    env = _subprocess_env(cwd)

    assert env["DAYTONA_API_KEY"] == "from-file"
    assert env["OPENAI_API_KEY"] == "from-shell-openai"
    assert "EMPTY_LINE_IGNORED" not in env


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


def test_harbor_subprocess_runner_retries_daytona_setup_exception_result(
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
                "agent_result": None,
                "verifier_result": None,
                "agent_setup": None,
                "agent_execution": None,
                "exception_info": {
                    "exception_type": "DaytonaError",
                    "exception_message": "Failed to execute session command: ",
                    "exception_traceback": (
                        "harbor/trial/trial.py in _setup_agent_environment\n"
                        "harbor/environments/daytona/environment.py in start\n"
                        "await env.ensure_dirs(env._mount_targets(writable_only=True))"
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
        "## Operating principle",
        "## Inspection and changes",
        "## Timeouts and long-running work",
        "### Command helper pattern",
        "## Required verification and final QA",
        "## Verification and final submission",
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
    assert hashlib.sha256(skill.encode()).hexdigest() == (
        "cfcce3ff3ba4c293f9d3c68cc61b618befa2c2ed8727043633b37b6de255283c"
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
    assert "class Todo" in skill
    assert "task: str" in skill
    assert "done: bool = False" in skill
    assert "class RequiredVerification" in skill
    assert "requirement: str" in skill
    assert "check: Callable[[], bool] | str" in skill
    assert "verified: bool = False" in skill
    assert 'evidence: str = ""' in skill
    assert "verification: str" not in skill
    assert "todos and required verification" in skill
    assert "Mark a todo done" in skill
    assert "required checks" in skill
    assert "both lists short" in skill
    assert "extracted from the task" not in skill
    assert "callable or command check evaluates true" in skill
    assert "passed against the current final state" in skill
    assert "verified:" in skill
    assert "schema" not in skill.lower()
    assert "yaml" not in skill.lower()
    assert all(term not in skill for term in obsolete_schema_terms)
    assert "ledger" not in skill.lower()
    assert "task into todos" in skill
    assert "Before SUBMIT" in skill
    assert "fresh verifier-shaped evidence" in skill
    assert "current final state" in skill
    assert "Any unverified required verification entry is a blocker" in normalized_skill
    assert "file existence alone" in skill
    assert "self-attestation" in skill
    assert "literal paths/endpoints" in normalized_skill
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
    assert "ready_to_submit(todos, required)" in skill
    assert "all(todo.done for todo in todos)" in skill
    assert "all(item.verified for item in required)" in skill
    assert "When every todo is done" in skill
    assert "SUBMIT makes the result final" in skill
    assert "targeted proof is enough" not in skill
    assert "SUBMIT immediately" not in skill
    assert "SUBMIT while budget remains" not in skill
    assert "stale debug history" in normalized_skill
    assert "Once the observable task contract is satisfied" not in skill
    assert "run the verification in one iteration" not in skill
    assert "separate later iteration" not in skill
    assert "always run the full verifier" not in skill.lower()
    assert "must reproduce the full verifier" not in skill.lower()
    for term in ["windows", "win311", "qemu", "mips", "bmp", "doom", "PIL"]:
        assert re.search(rf"\b{re.escape(term)}\b", skill, re.IGNORECASE) is None

    snippet = skill.split("```python\n", 1)[1].split("\n```", 1)[0]
    compile(snippet, "<terminal-bench-skill>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    for anchor in [
        "async def run(cmd):",
        "subprocess.run",
        "capture_output=True",
        "result = await run('python -m pytest tests/unit -q')",
        "requests.get(url, timeout=10)",
        "asyncio.wait_for",
        "print(result.returncode)",
        "print(result.stdout[-2000:])",
        "print(result.stderr[-2000:])",
    ]:
        assert anchor in snippet

    assert "Parallelize only expensive checks" in skill
    assert "await asyncio.gather" not in snippet
    assert "timeout=timeout" not in snippet
    assert "import asyncio" not in snippet
    assert "import subprocess" not in snippet
    assert "import requests" not in snippet

    visual_snippet = skill.split("## Visual perception with predict", 1)[1].split(
        "## Required verification", 1
    )[0]
    for anchor in [
        "await predict(...)",
        "dspy.Image",
        "data:image/png;base64,",
        "base64.b64encode(image_bytes).decode()",
        "image=data_url",
        "print(result.visible_text)",
    ]:
        assert anchor in visual_snippet
    assert "import base64" not in visual_snippet
    assert "from pathlib import Path" not in visual_snippet

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

        async def acall(self, **_kwargs):
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

        async def acall(self, **_kwargs):
            return SimpleNamespace(answer="done")

    class FakeInterpreter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self) -> None:
            pass

    class FakeDspy:
        class Signature:
            def __init__(self, fields, instructions) -> None:
                self.output_fields = {"answer": object()} if isinstance(fields, str) else dict(fields)
                self.instructions = instructions

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
