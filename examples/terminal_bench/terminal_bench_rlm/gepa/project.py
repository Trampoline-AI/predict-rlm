from __future__ import annotations

import asyncio
import inspect
import json
import os
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
import tomllib
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

from predict_rlm.trace import RunTrace
from rlm_gepa import EvaluationContext, RLMGepaExampleResult, RLMGepaProject
from terminal_bench_rlm.scoring import to_gepa_example_result
from terminal_bench_rlm.skills import DEFAULT_TERMINAL_BENCH_SKILL_INSTRUCTIONS

from .config import (
    COMPONENT_SKILL,
    TERMINAL_BENCH_SPEC,
    TerminalBenchGepaConfig,
    default_config,
)


@dataclass(frozen=True)
class TerminalBenchExample:
    task_id: str
    instruction: str = ""


@dataclass(frozen=True)
class TerminalBenchTaskRunRequest:
    task_id: str
    instruction: str
    skill_instructions: str
    lm: Any
    sub_lm: Any
    max_iterations: int
    task_timeout: int
    verbose_rlm: bool
    output_dir: Path
    run_id: str
    config: TerminalBenchGepaConfig
    task_resources: dict[str, Any] | None = None


@dataclass(frozen=True)
class TerminalBenchTaskRunResult:
    task_id: str
    trial_result: Any
    traces: list[Any]
    run_dir: Path | None = None
    error: str | None = None


@dataclass(frozen=True)
class HarborTaskTimeouts:
    environment_setup: int
    agent: int
    verifier: int
    cleanup: int

    @property
    def outer(self) -> int:
        return self.environment_setup + self.agent + self.verifier + self.cleanup


class HarborControllerLocality(StrEnum):
    AUTO = "auto"
    LOCAL_CONTROLLER = "local-controller"
    REMOTE_CONTROLLER = "remote-controller"


@dataclass(frozen=True)
class HarborControllerSelection:
    locality: HarborControllerLocality
    reason: str


@dataclass(frozen=True)
class RemoteCommandResult:
    """Portable result returned by remote-controller environment adapters."""

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""

    @property
    def return_code(self) -> int:
        return self.returncode


class RemoteControllerEnvironment(Protocol):
    """Boundary required by HarborRemoteControllerHarnessRunner.

    Implementations run shell commands in the controller environment and move
    files between the host and controller filesystem. The shared runner owns
    packaging, command composition, artifact download, and result parsing.
    """

    def exec(self, *, command: str, timeout_sec: int) -> Any: ...

    def upload_file(self, host_path: str, remote_path: str) -> Any: ...

    def download_file(self, remote_path: str, host_path: str) -> Any: ...


class LocalShellRemoteControllerEnvironment:
    """Remote-controller adapter that runs against the local shell/filesystem."""

    def __init__(self, *, cwd: Path | None = None) -> None:
        self.cwd = cwd

    def exec(self, *, command: str, timeout_sec: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=self.cwd,
            shell=True,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
        )

    def upload_file(self, host_path: str, remote_path: str) -> None:
        destination = Path(remote_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(host_path, destination)

    def download_file(self, remote_path: str, host_path: str) -> None:
        destination = Path(host_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote_path, destination)


class SshGcpRemoteControllerEnvironment:
    """Remote-controller adapter for GCP VMs reachable through SSH/SCP."""

    def __init__(
        self,
        host: str,
        *,
        ssh_executable: str = "ssh",
        scp_executable: str = "scp",
        ssh_args: Sequence[str] = (),
        scp_args: Sequence[str] = (),
    ) -> None:
        self.host = host
        self.ssh_executable = ssh_executable
        self.scp_executable = scp_executable
        self.ssh_args = tuple(ssh_args)
        self.scp_args = tuple(scp_args)

    def exec(self, *, command: str, timeout_sec: int) -> subprocess.CompletedProcess[str]:
        return _run_controller_subprocess(
            [self.ssh_executable, *self.ssh_args, self.host, command],
            timeout=timeout_sec,
            check=False,
        )

    def upload_file(self, host_path: str, remote_path: str) -> None:
        _run_controller_subprocess(
            [self.scp_executable, *self.scp_args, host_path, f"{self.host}:{remote_path}"],
            timeout=None,
        )

    def download_file(self, remote_path: str, host_path: str) -> None:
        _run_controller_subprocess(
            [self.scp_executable, *self.scp_args, f"{self.host}:{remote_path}", host_path],
            timeout=None,
        )


class GcpSshRemoteControllerEnvironment(SshGcpRemoteControllerEnvironment):
    """Alias with GCP-first naming for remote-controller call sites."""


class SbxRemoteControllerEnvironment:
    """Remote-controller adapter for Docker SBX sandboxes."""

    def __init__(
        self,
        *,
        sandbox_id: str | None = None,
        sbx_executable: str = "sbx",
        workspace: str | Path | None = None,
        create_args: Sequence[str] = (),
        remove_on_close: bool = True,
    ) -> None:
        self.sandbox_id = sandbox_id
        self.sbx_executable = sbx_executable
        self.workspace = Path(workspace) if workspace is not None else None
        self.create_args = tuple(create_args)
        self.remove_on_close = remove_on_close
        self._owns_sandbox = sandbox_id is None

    def create(self) -> str:
        if self.sandbox_id is not None:
            return self.sandbox_id
        workspace = self.workspace or Path(tempfile.mkdtemp(prefix="predict-rlm-sbx-controller-"))
        workspace.mkdir(parents=True, exist_ok=True)
        completed = _run_controller_subprocess(
            [self.sbx_executable, "create", "shell", str(workspace), *self.create_args],
            timeout=None,
        )
        sandbox_id = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
        if not sandbox_id:
            raise RuntimeError("sbx create did not return a sandbox id on stdout")
        self.workspace = workspace
        self.sandbox_id = sandbox_id
        return sandbox_id

    def close(self) -> None:
        if self.sandbox_id is None or not self._owns_sandbox or not self.remove_on_close:
            return
        _run_controller_subprocess([self.sbx_executable, "rm", self.sandbox_id], timeout=None)
        self.sandbox_id = None

    def exec(self, *, command: str, timeout_sec: int) -> subprocess.CompletedProcess[str]:
        sandbox_id = self.create()
        return _run_controller_subprocess(
            [self.sbx_executable, "exec", sandbox_id, "sh", "-lc", command],
            timeout=timeout_sec,
            check=False,
        )

    def upload_file(self, host_path: str, remote_path: str) -> None:
        sandbox_id = self.create()
        _run_controller_subprocess(
            [self.sbx_executable, "cp", host_path, f"{sandbox_id}:{remote_path}"],
            timeout=None,
        )

    def download_file(self, remote_path: str, host_path: str) -> None:
        sandbox_id = self.create()
        Path(host_path).parent.mkdir(parents=True, exist_ok=True)
        _run_controller_subprocess(
            [self.sbx_executable, "cp", f"{sandbox_id}:{remote_path}", host_path],
            timeout=None,
        )

    def __enter__(self) -> SbxRemoteControllerEnvironment:
        self.create()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


class DockerSbxRemoteControllerEnvironment(SbxRemoteControllerEnvironment):
    """Alias with Docker SBX naming for remote-controller call sites."""


class DaytonaRemoteControllerEnvironment:
    """Adapter that treats a Daytona sandbox as the Harbor controller host."""

    def __init__(self, sandbox: Any) -> None:
        self.sandbox = sandbox

    def exec(self, *, command: str, timeout_sec: int) -> Any:
        process = getattr(self.sandbox, "process", None)
        if process is not None:
            return _call_controller_method(
                process,
                ("exec", "run"),
                (
                    ((command,), {"timeout": timeout_sec}),
                    (((), {"command": command, "timeout_sec": timeout_sec})),
                ),
            )
        return _call_controller_method(
            self.sandbox,
            ("exec", "run"),
            (((), {"command": command, "timeout_sec": timeout_sec}), ((command,), {"timeout": timeout_sec})),
        )

    def upload_file(self, host_path: str, remote_path: str) -> Any:
        filesystem = getattr(self.sandbox, "fs", None) or getattr(self.sandbox, "filesystem", None)
        if filesystem is not None:
            return _call_controller_method(
                filesystem,
                ("upload_file", "copy_to", "put_file"),
                (((host_path, remote_path), {}),),
            )
        return _call_controller_method(
            self.sandbox,
            ("upload_file", "copy_to", "put_file"),
            (((host_path, remote_path), {}),),
        )

    def download_file(self, remote_path: str, host_path: str) -> Any:
        filesystem = getattr(self.sandbox, "fs", None) or getattr(self.sandbox, "filesystem", None)
        if filesystem is not None:
            return _call_controller_method(
                filesystem,
                ("download_file", "copy_from", "get_file"),
                (((remote_path, host_path), {}),),
            )
        return _call_controller_method(
            self.sandbox,
            ("download_file", "copy_from", "get_file"),
            (((remote_path, host_path), {}),),
        )


class TerminalBenchHarnessRunner(Protocol):
    async def run(self, request: TerminalBenchTaskRunRequest) -> TerminalBenchTaskRunResult: ...


class HarborSubprocessHarnessRunner:
    """Runs Terminal-Bench 2.x tasks through Harbor's CLI with the PredictRLM agent."""

    def __init__(self, *, cwd: Path | None = None) -> None:
        self.cwd = cwd or Path(__file__).resolve().parents[2]
        self._result_retry_limit = 1

    async def run(self, request: TerminalBenchTaskRunRequest) -> TerminalBenchTaskRunResult:
        return await asyncio.to_thread(self._run_sync, request)

    def _run_sync(self, request: TerminalBenchTaskRunRequest) -> TerminalBenchTaskRunResult:
        output_dir = _resolve_output_dir(request)
        output_dir.mkdir(parents=True, exist_ok=True)
        phase_log_path = _phase_log_path(request, output_dir=output_dir)
        cmd = _build_harbor_run_command(request, output_dir=output_dir)
        max_attempts = self._result_retry_limit + 1
        for attempt in range(1, max_attempts + 1):
            started = time.monotonic()
            _write_phase_event(
                phase_log_path,
                event="harbor_subprocess_start",
                phase="environment_setup",
                request=request,
                status="started",
                agent_timeout_seconds=_agent_timeout(request),
                outer_timeout_seconds=_subprocess_timeout(request),
                harbor_run_dir=str(output_dir / request.run_id),
                attempt=attempt,
                max_attempts=max_attempts,
            )
            try:
                completed = subprocess.run(cmd, **_subprocess_run_kwargs(request, cwd=self.cwd))
            except subprocess.TimeoutExpired as exc:
                run_dir = output_dir / request.run_id
                _write_phase_event(
                    phase_log_path,
                    event="harbor_subprocess_end",
                    phase="harness_subprocess",
                    request=request,
                    status="timeout",
                    duration_seconds=time.monotonic() - started,
                    agent_timeout_seconds=_agent_timeout(request),
                    outer_timeout_seconds=_subprocess_timeout(request),
                    harbor_run_dir=str(run_dir),
                    attempt=attempt,
                    max_attempts=max_attempts,
                )
                return TerminalBenchTaskRunResult(
                    task_id=request.task_id,
                    trial_result=_timeout_trial_result(exc),
                    traces=[],
                    run_dir=run_dir,
                    error=_subprocess_timeout_error(exc),
                )
            run_dir = output_dir / request.run_id
            _write_phase_event(
                phase_log_path,
                event="harbor_subprocess_end",
                phase="harness_subprocess",
                request=request,
                status="completed" if completed.returncode == 0 else "failed",
                duration_seconds=time.monotonic() - started,
                agent_timeout_seconds=_agent_timeout(request),
                outer_timeout_seconds=_subprocess_timeout(request),
                returncode=completed.returncode,
                harbor_run_dir=str(run_dir),
                attempt=attempt,
                max_attempts=max_attempts,
            )
            if completed.returncode != 0:
                return TerminalBenchTaskRunResult(
                    task_id=request.task_id,
                    trial_result=_subprocess_failure_trial_result(completed),
                    traces=[],
                    run_dir=run_dir,
                    error=_subprocess_error(completed),
                )
            result = self._load_result(request, run_dir)
            retry_reason = _retryable_harbor_trial_exception_reason(result.trial_result)
            if retry_reason is None or attempt >= max_attempts:
                return result
            _write_phase_event(
                phase_log_path,
                event="harbor_subprocess_retry",
                phase="environment_setup",
                request=request,
                status="retrying",
                attempt=attempt,
                next_attempt=attempt + 1,
                max_attempts=max_attempts,
                retry_reason=retry_reason,
                harbor_run_dir=str(run_dir),
            )
        raise AssertionError("unreachable Harbor subprocess retry loop exit")

    def _load_result(
        self,
        request: TerminalBenchTaskRunRequest,
        run_dir: Path,
    ) -> TerminalBenchTaskRunResult:
        return _load_task_run_result(request, run_dir)


class HarborRemoteControllerHarnessRunner:
    """Runs the existing Harbor task command inside a Harbor/Daytona machine."""

    def __init__(
        self,
        controller_environment: Any | None = None,
        *,
        cwd: Path | None = None,
    ) -> None:
        self.controller_environment = controller_environment
        self.cwd = cwd or Path(__file__).resolve().parents[2]

    async def run(self, request: TerminalBenchTaskRunRequest) -> TerminalBenchTaskRunResult:
        return await asyncio.to_thread(self._run_sync, request)

    def _run_sync(self, request: TerminalBenchTaskRunRequest) -> TerminalBenchTaskRunResult:
        environment = self._require_controller_environment()
        if not _supports_remote_controller(environment):
            raise RuntimeError(
                "Harbor remote-controller requires one-shot remote command execution "
                "plus upload and download file APIs."
            )

        output_dir = _resolve_output_dir(request)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_dir = output_dir / request.run_id
        remote_root = _remote_run_root(request)
        remote_repo_dir = f"{remote_root}/repo"
        remote_output_dir = f"{remote_root}/harbor-runs"
        remote_archive_path = f"{remote_root}/repo.tar.gz"
        remote_artifact_path = f"{remote_root}/artifacts.tar.gz"
        repo_root = _repo_root_for_cwd(self.cwd)
        remote_cwd = _remote_cwd(repo_root, self.cwd, remote_repo_dir)
        remote_cmd = _build_harbor_run_command(request, output_dir=remote_output_dir)

        with tempfile.TemporaryDirectory(prefix="terminal-bench-remote-controller-") as tmp_dir:
            local_archive_path = Path(tmp_dir) / "repo.tar.gz"
            local_artifact_path = Path(tmp_dir) / "artifacts.tar.gz"
            _create_repo_archive(repo_root, local_archive_path)

            _remote_preflight_new_run_root(
                environment,
                remote_root,
                timeout=request.task_timeout,
            )
            _remote_exec_checked(
                environment,
                f"mkdir -p {shlex.quote(remote_root)} {shlex.quote(remote_output_dir)}",
                timeout=request.task_timeout,
                operation="creating remote controller workdir",
            )
            _remote_upload_file(environment, str(local_archive_path), remote_archive_path)
            _remote_exec_checked(
                environment,
                (
                    f"rm -rf {shlex.quote(remote_repo_dir)} && "
                    f"mkdir -p {shlex.quote(remote_repo_dir)} && "
                    f"tar -xzf {shlex.quote(remote_archive_path)} "
                    f"-C {shlex.quote(remote_repo_dir)} --strip-components=1"
                ),
                timeout=request.task_timeout,
                operation="unpacking remote controller package",
            )
            _remote_exec_checked(
                environment,
                f"cd {shlex.quote(remote_cwd)} && {shlex.join(remote_cmd)}",
                timeout=_subprocess_timeout(request),
                operation="running remote Harbor controller",
            )
            _remote_exec_checked(
                environment,
                (
                    f"tar -czf {shlex.quote(remote_artifact_path)} "
                    f"-C {shlex.quote(remote_output_dir)} {shlex.quote(request.run_id)}"
                ),
                timeout=request.task_timeout,
                operation="packing remote Harbor artifacts",
            )
            _remote_download_file(environment, remote_artifact_path, str(local_artifact_path))
            _extract_tarball(local_artifact_path, output_dir)

        return _load_task_run_result(request, run_dir)

    def _require_controller_environment(self) -> Any:
        if self.controller_environment is None:
            raise RuntimeError(
                "Harbor remote-controller was selected, but no Harbor/Daytona "
                "controller environment was provided. Remote runs are opt-in; "
                "provide an environment with exec/upload/download capabilities and "
                "download artifacts before rerunning a reused job name."
            )
        return self.controller_environment


class TerminalBenchSubprocessHarnessRunner:
    """Runs Terminal-Bench through its CLI with the PredictRLM custom agent."""

    def __init__(self, *, cwd: Path | None = None) -> None:
        self.cwd = cwd or Path(__file__).resolve().parents[2]

    async def run(self, request: TerminalBenchTaskRunRequest) -> TerminalBenchTaskRunResult:
        return await asyncio.to_thread(self._run_sync, request)

    def _run_sync(self, request: TerminalBenchTaskRunRequest) -> TerminalBenchTaskRunResult:
        config = request.config
        output_dir = _resolve_output_dir(request)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset = f"{config.dataset_name}=={config.dataset_version}"
        cmd = [
            config.terminal_bench_executable,
            "run",
            "--agent-import-path",
            "terminal_bench_rlm.tools.tbench_agent:TerminalBenchRLMAgent",
        ]
        for key, value in _agent_kwargs(request).items():
            cmd.extend(["--agent-kwarg", f"{key}={value}"])
        cmd.extend(
            [
                "--dataset",
                dataset,
                "--task-id",
                request.task_id,
                "--n-concurrent",
                str(config.n_concurrent_trials),
                "--n-attempts",
                str(config.n_attempts),
                "--run-id",
                request.run_id,
                "--output-path",
                str(output_dir),
                "--global-agent-timeout-sec",
                str(request.task_timeout),
                "--log-level",
                "info",
            ]
        )
        cmd.append("--upload-results" if config.upload_results else "--no-upload-results")
        cmd.append("--cleanup" if config.cleanup else "--no-cleanup")
        if config.no_rebuild:
            cmd.append("--no-rebuild")
        else:
            cmd.append("--rebuild")

        try:
            completed = subprocess.run(cmd, **_subprocess_run_kwargs(request, cwd=self.cwd))
        except subprocess.TimeoutExpired as exc:
            run_dir = output_dir / request.run_id
            return TerminalBenchTaskRunResult(
                task_id=request.task_id,
                trial_result=_timeout_trial_result(exc),
                traces=[],
                run_dir=run_dir,
                error=_subprocess_timeout_error(exc),
            )
        run_dir = output_dir / request.run_id
        if completed.returncode != 0:
            error = _subprocess_error(completed)
            return TerminalBenchTaskRunResult(
                task_id=request.task_id,
                trial_result=_subprocess_failure_trial_result(completed),
                traces=[],
                run_dir=run_dir,
                error=error,
            )
        return _load_task_run_result(request, run_dir)


class TerminalBenchInProcessHarnessRunner:
    """Runs Terminal-Bench through its Python Harness API."""

    def __init__(self, *, cwd: Path | None = None) -> None:
        self.cwd = cwd or Path(__file__).resolve().parents[2]

    async def run(self, request: TerminalBenchTaskRunRequest) -> TerminalBenchTaskRunResult:
        return await asyncio.to_thread(self._run_sync, request)

    def _run_sync(self, request: TerminalBenchTaskRunRequest) -> TerminalBenchTaskRunResult:
        from terminal_bench.harness.harness import Harness

        config = request.config
        output_dir = _resolve_output_dir(request)
        output_dir.mkdir(parents=True, exist_ok=True)
        agent_timeout = _agent_timeout_with_cleanup_grace(request)
        Harness(
            output_path=output_dir,
            run_id=request.run_id,
            agent_import_path="terminal_bench_rlm.tools.tbench_agent:TerminalBenchRLMAgent",
            dataset_name=config.dataset_name,
            dataset_version=config.dataset_version,
            agent_kwargs=_agent_kwargs(request),
            no_rebuild=config.no_rebuild,
            cleanup=config.cleanup,
            task_ids=[request.task_id],
            n_concurrent_trials=config.n_concurrent_trials,
            upload_results=config.upload_results,
            n_attempts=config.n_attempts,
            global_agent_timeout_sec=agent_timeout,
            log_level=20,
        ).run()
        run_dir = output_dir / request.run_id
        return _load_task_run_result(request, run_dir)


class TerminalBenchGepaProject(RLMGepaProject):
    project_name = "terminal-bench-rlm"
    components = (COMPONENT_SKILL,)
    agent_spec = TERMINAL_BENCH_SPEC

    def __init__(
        self,
        config: TerminalBenchGepaConfig,
        *,
        harness_runner: TerminalBenchHarnessRunner | None = None,
    ) -> None:
        self.config = config
        self.harness_runner = harness_runner or _build_harness_runner(config)

    def seed_candidate(self) -> dict[str, str]:
        return {COMPONENT_SKILL: _seed_skill_instructions()}

    def component_focus(self, component_name: str) -> str:
        if component_name == COMPONENT_SKILL:
            return (
                "terminal/container problem-solving instructions injected into "
                "the PredictRLM agent as a Skill"
            )
        return ""

    def load_trainset(self) -> Sequence[TerminalBenchExample]:
        return _examples(self.config.train_task_ids, limit=self.config.train_limit)

    def load_valset(self) -> Sequence[TerminalBenchExample]:
        return _examples(self.config.val_task_ids, limit=self.config.val_limit)

    def task_timeout_for_example(self, example: TerminalBenchExample, default_timeout: int) -> int:
        return _terminal_bench_task_timeouts(
            self.config,
            example.task_id,
            fallback=default_timeout,
        ).outer

    def task_resources_for_example(self, example: TerminalBenchExample) -> dict[str, Any]:
        return _terminal_bench_task_resources(self.config, example.task_id)

    async def evaluate_example(
        self,
        candidate: dict[str, str],
        example: TerminalBenchExample,
        context: EvaluationContext,
    ) -> RLMGepaExampleResult:
        task_timeout = self.task_timeout_for_example(example, context.task_timeout)
        task_resources = dict(context.task_resources) or self.task_resources_for_example(example)
        request = TerminalBenchTaskRunRequest(
            task_id=example.task_id,
            instruction=example.instruction,
            skill_instructions=candidate[COMPONENT_SKILL],
            lm=context.lm,
            sub_lm=context.sub_lm,
            max_iterations=context.max_iterations,
            task_timeout=task_timeout,
            verbose_rlm=context.verbose_rlm,
            output_dir=context.output_dir,
            run_id=_run_id(context.kind, example.task_id),
            config=self.config,
            task_resources=task_resources,
        )
        run_result = await self.harness_runner.run(request)
        result = to_gepa_example_result(
            run_result.trial_result,
            traces=run_result.traces,
            example_id=example.task_id,
            rlm_inputs={
                "task_id": example.task_id,
                "dataset_name": self.config.dataset_name,
                "dataset_version": self.config.dataset_version,
                "terminal_bench_run_dir": str(run_result.run_dir) if run_result.run_dir else None,
            },
        )
        if run_result.error:
            result.error = run_result.error
        elif not run_result.traces:
            result.error = "Terminal-Bench harness result did not expose PredictRLM RunTrace data"
        return result


def build_project(config: TerminalBenchGepaConfig | None = None) -> RLMGepaProject:
    return TerminalBenchGepaProject(config or default_config())


def _build_harness_runner(config: TerminalBenchGepaConfig) -> TerminalBenchHarnessRunner:
    return _build_harbor_harness_runner(config)


def _build_harbor_harness_runner(
    config: TerminalBenchGepaConfig,
    *,
    controller_environment: Any | None = None,
) -> TerminalBenchHarnessRunner:
    selection = select_harbor_controller_locality(
        config.harbor_controller_locality,
        controller_environment,
        harbor_environment=config.harbor_environment,
    )
    if selection.locality is HarborControllerLocality.REMOTE_CONTROLLER:
        return HarborRemoteControllerHarnessRunner(controller_environment)
    return HarborSubprocessHarnessRunner()


def select_harbor_controller_locality(
    requested: str | HarborControllerLocality,
    controller_environment: Any | None,
    *,
    harbor_environment: str | None = None,
) -> HarborControllerSelection:
    locality = HarborControllerLocality(str(requested))
    is_daytona = _is_daytona_environment(harbor_environment)
    if is_daytona and controller_environment is None:
        raise RuntimeError(
            "Harbor Daytona requires an explicit Daytona controller environment "
            "so the Harbor host/controller runs inside the sandbox via "
            "remote-controller. build_project(config) cannot construct it. "
            "Do not treat the host launcher as a local controller."
        )
    if controller_environment is None:
        if locality is HarborControllerLocality.REMOTE_CONTROLLER:
            raise RuntimeError(
                "Harbor remote-controller requires a Harbor/Daytona environment "
                "object with remote exec, upload, and download APIs. "
                "build_project(config) cannot construct that environment; callers "
                "must supply it to the lower-level Harbor runner."
            )
        return HarborControllerSelection(
            HarborControllerLocality.LOCAL_CONTROLLER,
            "no explicit remote controller environment was provided; using local Harbor launcher",
        )
    if is_daytona:
        if locality is HarborControllerLocality.LOCAL_CONTROLLER:
            raise RuntimeError(
                "Harbor Daytona does not support local-controller; use remote-controller "
                "so the Harbor host/controller runs inside the Daytona sandbox."
            )
        if locality is HarborControllerLocality.AUTO:
            if _supports_remote_controller(controller_environment):
                return HarborControllerSelection(
                    HarborControllerLocality.REMOTE_CONTROLLER,
                    "Daytona auto selected remote-controller so the controller runs inside the sandbox",
                )
            raise RuntimeError(
                "Harbor Daytona auto requires remote-controller exec plus upload/download "
                "file APIs; local-controller fallback is not allowed."
            )
    if locality is HarborControllerLocality.LOCAL_CONTROLLER:
        if not _supports_interactive_exec(controller_environment):
            raise RuntimeError(
                "Harbor local-controller requires persistent interactive exec "
                "(start_exec/exec_stream/popen or docker exec -i)."
            )
        return HarborControllerSelection(
            HarborControllerLocality.LOCAL_CONTROLLER,
            "remote environment exposes persistent interactive exec",
        )
    if locality is HarborControllerLocality.REMOTE_CONTROLLER:
        if not _supports_remote_controller(controller_environment):
            raise RuntimeError(
                "Harbor remote-controller requires one-shot remote exec plus "
                "upload/download file APIs."
            )
        return HarborControllerSelection(
            HarborControllerLocality.REMOTE_CONTROLLER,
            "remote environment exposes one-shot exec and artifact file sync",
        )
    if _supports_interactive_exec(controller_environment):
        return HarborControllerSelection(
            HarborControllerLocality.LOCAL_CONTROLLER,
            "auto selected local-controller because persistent interactive exec is available",
        )
    if _supports_remote_controller(controller_environment):
        return HarborControllerSelection(
            HarborControllerLocality.REMOTE_CONTROLLER,
            "auto selected remote-controller because only one-shot exec plus file sync are available",
        )
    raise RuntimeError(
        "Harbor controller locality auto-detection failed: environment exposes neither "
        "persistent interactive exec nor remote exec/upload/download capabilities."
    )


def _is_daytona_environment(harbor_environment: str | None) -> bool:
    return str(harbor_environment or "").strip().lower() == "daytona"


def _resolve_output_dir(request: TerminalBenchTaskRunRequest) -> Path:
    output_dir = request.config.terminal_bench_output_dir
    if output_dir.is_absolute():
        return output_dir
    return (request.output_dir / output_dir).resolve()




def phase_duration_summary(run_dir: str | Path) -> dict[str, Any]:
    phase_totals: dict[str, dict[str, float | int]] = {}
    tasks: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(run_dir).rglob("task_phase_events.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            phase = str(event.get("phase") or "unknown")
            task_id = str(event.get("task_id") or path.parent.name)
            duration = event.get("duration_seconds")
            if not isinstance(duration, int | float):
                continue
            duration = float(duration)
            _add_phase_duration(phase_totals, phase, duration)
            task = tasks.setdefault(task_id, {"duration_seconds": 0.0, "phases": {}})
            task["duration_seconds"] += duration
            _add_phase_duration(task["phases"], phase, duration)
    return {
        "phase_totals": _sorted_phase_durations(phase_totals),
        "tasks": {
            task_id: {
                "duration_seconds": round(float(task["duration_seconds"]), 6),
                "phases": _sorted_phase_durations(task["phases"]),
            }
            for task_id, task in sorted(tasks.items())
        },
        "total_logged_duration_seconds": round(
            sum(float(phase["duration_seconds"]) for phase in phase_totals.values()), 6
        ),
    }


def _add_phase_duration(target: dict[str, dict[str, float | int]], phase: str, duration: float) -> None:
    bucket = target.setdefault(phase, {"duration_seconds": 0.0, "events": 0})
    bucket["duration_seconds"] = float(bucket["duration_seconds"]) + duration
    bucket["events"] = int(bucket["events"]) + 1


def _sorted_phase_durations(
    durations: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    return {
        phase: {
            "duration_seconds": round(float(values["duration_seconds"]), 6),
            "events": int(values["events"]),
        }
        for phase, values in sorted(durations.items())
    }


def _phase_log_path(request: TerminalBenchTaskRunRequest, *, output_dir: Path) -> Path:
    return output_dir / request.run_id / "task_phase_events.jsonl"


def _phase_log_path_text(
    request: TerminalBenchTaskRunRequest,
    *,
    output_dir: Path | str,
) -> str:
    if isinstance(output_dir, Path):
        return str(_phase_log_path(request, output_dir=output_dir))
    return "/".join(
        [
            output_dir.rstrip("/"),
            request.run_id,
            "task_phase_events.jsonl",
        ]
    )


def _write_phase_event(
    path: Path,
    *,
    event: str,
    phase: str,
    request: TerminalBenchTaskRunRequest,
    status: str,
    **fields: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "event": event,
        "phase": phase,
        "status": status,
        "task_id": request.task_id,
        "run_id": request.run_id,
        "dataset": request.config.harbor_dataset,
        **fields,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    duration = payload.get("duration_seconds")
    duration_text = f" duration_seconds={duration:.3f}" if isinstance(duration, (int, float)) else ""
    print(
        f"phase_event task={request.task_id} phase={phase} event={event} "
        f"status={status}{duration_text}",
        flush=True,
    )


def _build_harbor_run_command(
    request: TerminalBenchTaskRunRequest,
    *,
    output_dir: Path | str,
) -> list[str]:
    config = request.config
    cmd = [
        *shlex.split(config.harbor_executable),
        "run",
        "-d",
        config.harbor_dataset,
        "-e",
        config.harbor_environment,
        "--include-task-name",
        request.task_id,
        "--agent-import-path",
        "terminal_bench_rlm.tools.tbench_agent:DaytonaRemotePredictRLMAgent",
        "--n-attempts",
        str(config.n_attempts),
        "--n-concurrent",
        str(config.n_concurrent_trials),
        "--cpus",
        config.harbor_cpus,
        "--memory",
        config.harbor_memory,
        "--jobs-dir",
        str(output_dir),
        "--job-name",
        request.run_id,
    ]
    for key, value in _agent_kwargs(request, output_dir=output_dir).items():
        cmd.extend(["--agent-kwarg", f"{key}={value}"])
    if config.no_rebuild:
        cmd.append("--no-force-build")
    else:
        cmd.append("--force-build")
    cmd.append("--delete" if config.cleanup else "--no-delete")
    return cmd


def _agent_kwargs(
    request: TerminalBenchTaskRunRequest,
    *,
    output_dir: Path | str | None = None,
) -> dict[str, str]:
    config = request.config
    if output_dir is None:
        output_dir = _resolve_output_dir(request)
    kwargs = {
        "lm": _model_name(request.lm),
        "sub_lm": _model_name(request.sub_lm),
        "max_iterations": str(request.max_iterations),
        "exec_timeout": str(_agent_timeout_with_cleanup_grace(request)),
        "skill_instructions": request.skill_instructions,
        "task_id": request.task_id,
        "phase_log_path": _phase_log_path_text(request, output_dir=output_dir),
    }
    lm_reasoning_effort = _reasoning_effort(request.lm)
    if lm_reasoning_effort is not None:
        kwargs["lm_reasoning_effort"] = lm_reasoning_effort
    sub_lm_reasoning_effort = _reasoning_effort(request.sub_lm)
    if sub_lm_reasoning_effort is not None:
        kwargs["sub_lm_reasoning_effort"] = sub_lm_reasoning_effort
    lm_service_tier = _service_tier(request.lm)
    if lm_service_tier is not None:
        kwargs["lm_service_tier"] = lm_service_tier
    sub_lm_service_tier = _service_tier(request.sub_lm)
    if sub_lm_service_tier is not None:
        kwargs["sub_lm_service_tier"] = sub_lm_service_tier
    if request.verbose_rlm:
        kwargs["verbose"] = "true"
    if config.codex_lm:
        kwargs["codex_lm"] = "true"
        if os.environ.get("CODEX_LM_DEBUG"):
            kwargs["codex_lm_debug"] = os.environ["CODEX_LM_DEBUG"]
        if os.environ.get("CODEX_LM_DEBUG_LOG"):
            kwargs["codex_lm_debug_log"] = os.environ["CODEX_LM_DEBUG_LOG"]
        if os.environ.get("PREDICT_RLM_DEBUG"):
            kwargs["predict_rlm_debug"] = os.environ["PREDICT_RLM_DEBUG"]
        if os.environ.get("PREDICT_RLM_DEBUG_JSON"):
            kwargs["predict_rlm_debug_json"] = os.environ["PREDICT_RLM_DEBUG_JSON"]
        if os.environ.get("PREDICT_RLM_DEBUG_LOG"):
            kwargs["predict_rlm_debug_log"] = os.environ["PREDICT_RLM_DEBUG_LOG"]
        if config.codex_lm_exclude:
            kwargs["codex_lm_exclude"] = ",".join(config.codex_lm_exclude)
    return kwargs


def _agent_timeout(request: TerminalBenchTaskRunRequest) -> int:
    return _harbor_task_timeouts(request).agent


def _agent_timeout_with_cleanup_grace(request: TerminalBenchTaskRunRequest) -> int:
    return _agent_timeout(request)


def _subprocess_timeout(request: TerminalBenchTaskRunRequest) -> int:
    return _harbor_task_timeouts(request).outer


def _harbor_task_timeouts(request: TerminalBenchTaskRunRequest) -> HarborTaskTimeouts:
    return _terminal_bench_task_timeouts(
        request.config,
        request.task_id,
        fallback=max(1, int(request.task_timeout)),
    )


def _terminal_bench_task_timeouts(
    config: TerminalBenchGepaConfig,
    task_id: str,
    *,
    fallback: int,
) -> HarborTaskTimeouts:
    cleanup = max(0, int(config.timeout_cleanup_grace_sec))
    payload = _load_terminal_bench_task_toml(config, task_id)
    if payload is None:
        if task_id.startswith("terminal-bench/"):
            raise RuntimeError(
                f"Cannot determine official Harbor timeouts for {task_id}: "
                "task.toml is missing from the configured task cache and the global Harbor cache."
            )
        return HarborTaskTimeouts(
            environment_setup=fallback,
            agent=fallback,
            verifier=fallback,
            cleanup=cleanup,
        )
    return HarborTaskTimeouts(
        environment_setup=_timeout_from_section(payload, "environment", "build_timeout_sec", fallback),
        agent=_timeout_from_section(payload, "agent", "timeout_sec", fallback),
        verifier=_timeout_from_section(payload, "verifier", "timeout_sec", fallback),
        cleanup=cleanup,
    )


def _terminal_bench_task_resources(config: TerminalBenchGepaConfig, task_id: str) -> dict[str, Any]:
    payload = _load_terminal_bench_task_toml(config, task_id)
    if payload is None:
        return {}
    environment = payload.get("environment")
    if not isinstance(environment, dict):
        return {}
    resources: dict[str, Any] = {}
    for key in ("cpus", "memory_mb", "storage_mb", "gpus"):
        if key in environment:
            resources[key] = environment[key]
    return resources


def _load_harbor_task_toml(request: TerminalBenchTaskRunRequest) -> dict[str, Any] | None:
    return _load_terminal_bench_task_toml(request.config, request.task_id)


def _load_terminal_bench_task_toml(
    config: TerminalBenchGepaConfig,
    task_id: str,
) -> dict[str, Any] | None:
    task_toml = _terminal_bench_task_toml_path(config, task_id)
    if task_toml is None:
        return None
    try:
        return tomllib.loads(task_toml.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None


def _harbor_task_toml_path(request: TerminalBenchTaskRunRequest) -> Path | None:
    return _terminal_bench_task_toml_path(request.config, request.task_id)


def _terminal_bench_task_toml_path(config: TerminalBenchGepaConfig, task_id: str) -> Path | None:
    roots = []
    if config.harbor_task_cache_dir is not None:
        roots.append(config.harbor_task_cache_dir)
    global_cache_root = Path.home() / ".cache" / "harbor" / "tasks" / "packages"
    if global_cache_root not in roots:
        roots.append(global_cache_root)

    task_parts = [part for part in task_id.split("/") if part]
    candidates = []
    for cache_root in roots:
        task_dir = cache_root.joinpath(*task_parts)
        candidates.extend(path for path in task_dir.glob("*/task.toml") if path.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _timeout_from_section(
    payload: dict[str, Any],
    section: str,
    key: str,
    fallback: int,
) -> int:
    section_payload = payload.get(section)
    if not isinstance(section_payload, dict):
        return fallback
    value = section_payload.get(key)
    if value is None:
        return fallback
    return max(1, int(float(value)))


def _subprocess_run_kwargs(request: TerminalBenchTaskRunRequest, *, cwd: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "cwd": cwd,
        "check": False,
        "env": _subprocess_env(cwd),
        "text": True,
        "timeout": _subprocess_timeout(request),
    }
    if not request.verbose_rlm:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    return kwargs


def _supports_interactive_exec(environment: Any) -> bool:
    if _has_any_callable(environment, ("start_exec", "exec_stream", "popen")):
        return True
    container = getattr(environment, "container", None)
    if container is None:
        return False
    if _has_any_callable(container, ("start_exec", "exec_stream", "popen")):
        return True
    if getattr(container, "id", None):
        return True
    attrs = getattr(container, "attrs", None)
    return isinstance(attrs, dict) and bool(attrs.get("Id"))


def _supports_remote_controller(environment: Any) -> bool:
    return (
        _has_any_callable(environment, ("exec", "run"))
        and _has_any_callable(environment, ("upload_file", "copy_to", "put_file"))
        and _has_any_callable(environment, ("download_file", "copy_from", "get_file"))
    )


def _has_any_callable(obj: Any, names: tuple[str, ...]) -> bool:
    return any(callable(getattr(obj, name, None)) for name in names)


def _run_controller_subprocess(
    cmd: Sequence[str],
    *,
    timeout: int | None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(cmd),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            "remote-controller command failed: "
            f"{shlex.join(list(cmd))}; exit code {completed.returncode}; "
            f"stdout: {completed.stdout}; stderr: {completed.stderr}"
        )
    return completed


def _call_controller_method(
    obj: Any,
    names: tuple[str, ...],
    attempts: Sequence[tuple[tuple[Any, ...], dict[str, Any]]],
) -> Any:
    type_errors: list[TypeError] = []
    for name in names:
        method = getattr(obj, name, None)
        if method is None:
            continue
        for args, kwargs in attempts:
            try:
                return _resolve_remote_call(method(*args, **kwargs))
            except TypeError as exc:
                type_errors.append(exc)
    if type_errors:
        raise type_errors[0]
    raise TypeError(f"Remote-controller object does not expose any of: {', '.join(names)}")


def _remote_run_root(request: TerminalBenchTaskRunRequest) -> str:
    root = request.config.harbor_remote_workdir.rstrip("/") or "/tmp/predict_rlm_terminal_bench"
    return f"{root}/{request.run_id}"


def _repo_root_for_cwd(cwd: Path) -> Path:
    cwd = cwd.resolve()
    for directory in (cwd, *cwd.parents):
        if (directory / "pyproject.toml").is_file() and (
            directory / "examples" / "terminal_bench"
        ).is_dir():
            return directory
    return cwd


def _remote_cwd(repo_root: Path, cwd: Path, remote_repo_dir: str) -> str:
    try:
        rel_cwd = cwd.resolve().relative_to(repo_root)
    except ValueError:
        return remote_repo_dir
    if str(rel_cwd) == ".":
        return remote_repo_dir
    return "/".join([remote_repo_dir.rstrip("/"), rel_cwd.as_posix()])


def _create_repo_archive(repo_root: Path, archive_path: Path) -> None:
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in _iter_repo_archive_paths(repo_root):
            archive.add(path, arcname=str(Path("repo") / path.relative_to(repo_root)), recursive=False)


def _iter_repo_archive_paths(repo_root: Path) -> list[Path]:
    tracked_paths = _git_tracked_repo_paths(repo_root)
    if tracked_paths is not None:
        return tracked_paths
    return _fallback_repo_archive_paths(repo_root)


def _git_tracked_repo_paths(repo_root: Path) -> list[Path] | None:
    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    paths: list[Path] = []
    for raw_relpath in completed.stdout.split(b"\0"):
        if not raw_relpath:
            continue
        relpath = Path(os.fsdecode(raw_relpath))
        if relpath.is_absolute() or ".." in relpath.parts:
            continue
        path = repo_root / relpath
        if path.is_file() or path.is_symlink():
            paths.append(path)
    return sorted(paths)


def _fallback_repo_archive_paths(repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    for root, dirnames, filenames in os.walk(repo_root):
        root_path = Path(root)
        dirnames[:] = [
            dirname
            for dirname in sorted(dirnames)
            if not _exclude_from_remote_package(root_path / dirname, repo_root)
        ]
        for filename in sorted(filenames):
            path = root_path / filename
            if not _exclude_from_remote_package(path, repo_root):
                paths.append(path)
    return paths


def _exclude_from_remote_package(path: Path, repo_root: Path) -> bool:
    rel = path.relative_to(repo_root)
    parts = set(rel.parts)
    if parts & {
        ".git",
        ".hermes",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".terminal-bench-venv",
        ".venv",
        "__pycache__",
        "bug_reports",
        "downloaded-gcp-artifacts",
        "ops",
        "runs",
        "run_artifacts",
    }:
        return True
    if path.suffix in {".pyc", ".pyo"}:
        return True
    return rel.as_posix() in {
        "early_failures.txt",
        "failing-short.txt",
        "old-skill-bits.txt",
    }


def _remote_exec_checked(
    environment: Any,
    command: str,
    *,
    timeout: int,
    operation: str,
) -> Any:
    result = _remote_exec(environment, command, timeout=timeout)
    returncode = _remote_returncode(result)
    if returncode not in (0, None):
        stdout = _remote_stdout(result)
        stderr = _remote_stderr(result)
        raise RuntimeError(
            f"Harbor remote-controller failed while {operation}: "
            f"exit code {returncode}; stdout: {stdout}; stderr: {stderr}"
        )
    return result


def _remote_preflight_new_run_root(environment: Any, remote_root: str, *, timeout: int) -> None:
    result = _remote_exec(
        environment,
        f"test ! -e {shlex.quote(remote_root)}",
        timeout=timeout,
    )
    returncode = _remote_returncode(result)
    if returncode not in (0, None):
        stdout = _remote_stdout(result)
        stderr = _remote_stderr(result)
        raise RuntimeError(
            "Harbor remote-controller remote root already exists: "
            f"{remote_root}. Refusing to delete or overwrite it. "
            "Download/preserve artifacts or use a unique run id before rerunning. "
            f"stdout: {stdout}; stderr: {stderr}"
        )


def _remote_exec(environment: Any, command: str, *, timeout: int) -> Any:
    for name in ("exec", "run"):
        method = getattr(environment, name, None)
        if method is None:
            continue
        try:
            return _resolve_remote_call(method(command=command, timeout_sec=int(timeout)))
        except TypeError as keyword_exc:
            try:
                return _resolve_remote_call(method(command, timeout=timeout))
            except TypeError:
                raise keyword_exc
    raise TypeError("Harbor remote-controller environment does not expose exec/run")


def _remote_upload_file(environment: Any, host_path: str, remote_path: str) -> None:
    for name in ("upload_file", "copy_to", "put_file"):
        method = getattr(environment, name, None)
        if method is not None:
            _resolve_remote_call(method(host_path, remote_path))
            return
    raise TypeError("Harbor remote-controller environment does not expose upload_file/copy_to")


def _remote_download_file(environment: Any, remote_path: str, host_path: str) -> None:
    for name in ("download_file", "copy_from", "get_file"):
        method = getattr(environment, name, None)
        if method is not None:
            Path(host_path).parent.mkdir(parents=True, exist_ok=True)
            _resolve_remote_call(method(remote_path, host_path))
            return
    raise TypeError("Harbor remote-controller environment does not expose download_file/copy_from")


def _resolve_remote_call(value: Any) -> Any:
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


def _remote_returncode(result: Any) -> int | None:
    value = getattr(result, "return_code", None)
    if value is None:
        value = getattr(result, "returncode", None)
    if value is None:
        value = getattr(result, "exit_code", None)
    return value


def _remote_stdout(result: Any) -> str:
    stdout = getattr(result, "stdout", None)
    if stdout is not None:
        return str(stdout)
    artifacts = getattr(result, "artifacts", None)
    artifact_stdout = getattr(artifacts, "stdout", None)
    if artifact_stdout is not None:
        return str(artifact_stdout)
    payload = getattr(result, "result", None)
    return "" if payload is None else str(payload)


def _remote_stderr(result: Any) -> str:
    stderr = getattr(result, "stderr", None)
    return "" if stderr is None else str(stderr)


def _extract_tarball(archive_path: Path, output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            _extract_tar_member(archive, member, output_dir)


def _extract_tar_member(archive: tarfile.TarFile, member: tarfile.TarInfo, output_dir: Path) -> None:
    target = (output_dir / member.name).resolve()
    if output_dir != target and output_dir not in target.parents:
        raise RuntimeError(f"Refusing to extract remote artifact outside {output_dir}: {member.name}")
    if member.isdir():
        target.mkdir(parents=True, exist_ok=True)
        return
    if not member.isfile():
        raise RuntimeError(f"Refusing to extract unsupported remote artifact member: {member.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source = archive.extractfile(member)
    if source is None:
        raise RuntimeError(f"Remote artifact member could not be read: {member.name}")
    with source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination)


def _subprocess_env(cwd: Path) -> dict[str, str]:
    env = os.environ.copy()
    env_file = _find_env_development(cwd)
    if env_file is None:
        return env
    for line in env_file.read_text(encoding="utf-8").splitlines():
        key_value = _parse_env_line(line)
        if key_value is None:
            continue
        key, value = key_value
        env.setdefault(key, value)
    return env


def _find_env_development(cwd: Path) -> Path | None:
    cwd = cwd.resolve()
    for directory in (cwd, *cwd.parents):
        candidate = directory / ".env.development"
        if candidate.is_file():
            return candidate
    return None


def _parse_env_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].lstrip()
    key, separator, value = line.partition("=")
    key = key.strip()
    if not separator or not key:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'\"', "'"}:
        value = value[1:-1]
    return key, value


def _timeout_trial_result(exc: subprocess.TimeoutExpired) -> dict[str, Any]:
    stdout_tail = _tail_text(exc.output)
    stderr_tail = _tail_text(exc.stderr)
    return {
        "is_resolved": False,
        "parser_results": {},
        "exception_info": {
            "exception_type": "HarnessTimeoutError",
            "exception_message": _subprocess_timeout_error(exc),
            "phase": "harness_subprocess",
            "timed_out": True,
            "timeout_seconds": exc.timeout,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        },
    }


def _subprocess_failure_trial_result(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "is_resolved": False,
        "parser_results": {},
        "exception_info": {
            "exception_type": "HarnessSubprocessError",
            "exception_message": _subprocess_error(completed),
            "phase": "harness_subprocess",
            "returncode": completed.returncode,
            "stdout_tail": _tail_text(completed.stdout),
            "stderr_tail": _tail_text(completed.stderr),
        },
    }


_NON_RETRYABLE_HARBOR_EXCEPTION_TYPES = {
    "AgentTimeoutError",
    "HarnessSubprocessError",
    "HarnessTimeoutError",
}
_HARBOR_REGISTRY_CONTEXT_MARKERS = (
    "auth.docker.io",
    "containerd",
    "docker",
    "docker hub",
    "docker.io",
    "ghcr.io",
    "pull",
    "registry",
    "registry-1.docker.io",
)


def _retryable_harbor_trial_exception_reason(trial_result: Any) -> str | None:
    if not isinstance(trial_result, dict):
        return None
    exception_info = trial_result.get("exception_info")
    if not isinstance(exception_info, dict):
        return None

    exception_type = str(
        exception_info.get("exception_type")
        or exception_info.get("type")
        or exception_info.get("class")
        or ""
    )
    if exception_type in _NON_RETRYABLE_HARBOR_EXCEPTION_TYPES:
        return None

    text = _exception_info_text(exception_info)
    normalized = text.lower()
    if "agenttimeouterror" in normalized:
        return None
    if exception_type == "DaytonaError":
        if "harbor/environments/daytona" in normalized or "failed to execute session command" in normalized:
            return "daytona_environment_setup"
    if "failed to fetch anonymous token" in normalized:
        return "docker_registry_anonymous_token"
    if "failed to resolve reference" in normalized and _has_harbor_registry_context(normalized):
        return "docker_registry_reference_resolution"
    if "500 internal server error" in normalized and _has_harbor_registry_context(normalized):
        return "docker_registry_500"
    if "too many requests" in normalized and _has_harbor_registry_context(normalized):
        return "docker_registry_rate_limit"
    if "tls handshake timeout" in normalized and _has_harbor_registry_context(normalized):
        return "docker_registry_tls_timeout"
    if "connection reset" in normalized and _has_harbor_registry_context(normalized):
        return "docker_registry_connection_reset"
    if "i/o timeout" in normalized and _has_harbor_registry_context(normalized):
        return "docker_registry_io_timeout"
    return None


def _exception_info_text(exception_info: dict[str, Any]) -> str:
    fields = (
        "exception_type",
        "exception_message",
        "message",
        "diagnostic_text",
        "error",
        "exception_traceback",
    )
    return "\n".join(str(exception_info[field]) for field in fields if exception_info.get(field))


def _has_harbor_registry_context(text: str) -> bool:
    return any(marker in text for marker in _HARBOR_REGISTRY_CONTEXT_MARKERS)


def _load_task_run_result(
    request: TerminalBenchTaskRunRequest,
    run_dir: Path,
) -> TerminalBenchTaskRunResult:
    harbor_result_path = run_dir / "result.json"
    if harbor_result_path.exists():
        payload = json.loads(harbor_result_path.read_text(encoding="utf-8"))
        trial, trial_dir = _load_harbor_trial_result(payload, request.task_id, run_dir)
        return TerminalBenchTaskRunResult(
            task_id=request.task_id,
            trial_result=_attach_harbor_verifier_details(trial, trial_dir),
            traces=_load_run_traces(
                run_dir,
                model=_model_name(request.lm),
                sub_model=_model_name(request.sub_lm),
                max_iterations=request.max_iterations,
            ),
            run_dir=run_dir,
        )

    results_path = run_dir / "results.json"
    if not results_path.exists():
        return TerminalBenchTaskRunResult(
            task_id=request.task_id,
            trial_result={"is_resolved": False, "parser_results": {}},
            traces=[],
            run_dir=run_dir,
            error=f"Terminal-Bench completed but did not write {results_path}",
        )
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    trial = _find_trial_result(payload, request.task_id)
    return TerminalBenchTaskRunResult(
        task_id=request.task_id,
        trial_result=trial,
        traces=_load_run_traces(
            run_dir,
            model=_model_name(request.lm),
            sub_model=_model_name(request.sub_lm),
            max_iterations=request.max_iterations,
        ),
        run_dir=run_dir,
    )


def _examples(task_ids: Sequence[str], *, limit: int | None = None) -> list[TerminalBenchExample]:
    ids = list(task_ids)
    if limit is not None:
        ids = ids[:limit]
    return [TerminalBenchExample(task_id=task_id) for task_id in ids]


def _seed_skill_instructions() -> str:
    return DEFAULT_TERMINAL_BENCH_SKILL_INSTRUCTIONS


def _model_name(model: Any) -> str:
    return str(getattr(model, "model", model))


def _reasoning_effort(model: Any) -> str | None:
    kwargs = getattr(model, "kwargs", None)
    if not isinstance(kwargs, dict):
        return None
    effort = kwargs.get("reasoning_effort")
    return str(effort) if effort else None


def _service_tier(model: Any) -> str | None:
    kwargs = getattr(model, "kwargs", None)
    if not isinstance(kwargs, dict):
        return None
    service_tier = kwargs.get("service_tier")
    return str(service_tier) if service_tier else None


def _run_id(kind: str, task_id: str) -> str:
    safe_task = "".join(char if char.isalnum() or char in "._-" else "-" for char in task_id)
    return f"gepa-{kind}-{safe_task}-{uuid.uuid4().hex[:8]}"


def _find_trial_result(payload: dict[str, Any], task_id: str) -> dict[str, Any]:
    results = payload.get("results")
    if isinstance(results, list):
        for row in results:
            if isinstance(row, dict) and row.get("task_id") == task_id:
                return row
        for row in results:
            if isinstance(row, dict):
                return row
    return payload


def _find_harbor_trial_result(payload: dict[str, Any], task_id: str) -> dict[str, Any]:
    results = payload.get("trial_results")
    if isinstance(results, list):
        for row in results:
            if isinstance(row, dict) and _harbor_task_matches(row, task_id):
                return row
        for row in results:
            if isinstance(row, dict):
                return row
    return payload


def _load_harbor_trial_result(
    payload: dict[str, Any],
    task_id: str,
    run_dir: Path,
) -> tuple[dict[str, Any], Path | None]:
    trial = _find_harbor_trial_result(payload, task_id)
    trial_dir = _harbor_trial_dir(run_dir, trial)
    if _is_harbor_trial_result(trial):
        return trial, trial_dir
    nested = _find_nested_harbor_trial_result(run_dir, task_id)
    if nested is not None:
        return nested
    return trial, trial_dir


def _find_nested_harbor_trial_result(
    run_dir: Path,
    task_id: str,
) -> tuple[dict[str, Any], Path | None] | None:
    for result_path in sorted(run_dir.glob("*/result.json")):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and _harbor_task_matches(payload, task_id):
            return payload, result_path.parent
    return None


def _attach_harbor_verifier_details(
    trial: dict[str, Any],
    trial_dir: Path | None,
) -> dict[str, Any]:
    if trial_dir is None:
        return trial
    ctrf_path = trial_dir / "verifier" / "ctrf.json"
    if not ctrf_path.exists():
        return trial
    try:
        ctrf = json.loads(ctrf_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return trial
    enriched = dict(trial)
    verifier_result = enriched.get("verifier_result")
    if not isinstance(verifier_result, dict):
        verifier_result = {}
    else:
        verifier_result = dict(verifier_result)
    verifier_result["ctrf"] = ctrf
    enriched["verifier_result"] = verifier_result
    return enriched


def _is_harbor_trial_result(row: dict[str, Any]) -> bool:
    return any(
        key in row
        for key in ("agent_result", "verifier_result", "exception_info", "trial_name", "task_name")
    )


def _harbor_task_matches(row: dict[str, Any], task_id: str) -> bool:
    task_name = _harbor_task_name(row)
    if task_name == task_id or (task_name is not None and task_name.endswith(f"/{task_id}")):
        return True
    trial_name = row.get("trial_name")
    if isinstance(trial_name, str) and trial_name.startswith(f"{task_id}__"):
        return True
    task_id_payload = row.get("task_id")
    if isinstance(task_id_payload, dict) and task_id_payload.get("name") == task_id:
        return True
    return False


def _harbor_trial_dir(run_dir: Path, row: dict[str, Any]) -> Path | None:
    trial_uri = row.get("trial_uri")
    if isinstance(trial_uri, str):
        parsed = urlparse(trial_uri)
        if parsed.scheme == "file":
            return Path(unquote(parsed.path))
    trial_name = row.get("trial_name")
    if isinstance(trial_name, str):
        return run_dir / trial_name
    return None


def _harbor_task_name(row: dict[str, Any]) -> str | None:
    task_name = row.get("task_name") or row.get("task_id")
    if task_name is not None:
        return str(task_name)
    task_info = row.get("task_info")
    if isinstance(task_info, dict):
        name = task_info.get("name") or task_info.get("task_name") or task_info.get("id")
        return str(name) if name is not None else None
    return None


def _load_run_traces(run_dir: Path, *, model: str, sub_model: str | None, max_iterations: int) -> list[RunTrace]:
    traces: list[RunTrace] = []
    for path in sorted(run_dir.rglob("predict_rlm_trace*.json")):
        traces.append(RunTrace.model_validate_json(path.read_text(encoding="utf-8")))
    if traces:
        return traces
    return [
        RunTrace(
            status="completed",
            model=model,
            sub_model=sub_model,
            iterations=0,
            max_iterations=max_iterations,
            duration_ms=0,
        )
    ]


def _subprocess_error(completed: subprocess.CompletedProcess[str]) -> str:
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    output = output.strip()
    if len(output) > 4000:
        output = output[-4000:]
    return f"Terminal-Bench CLI exited {completed.returncode}: {output}"


def _subprocess_timeout_error(exc: subprocess.TimeoutExpired) -> str:
    output_parts = []
    for value in (exc.output, exc.stderr):
        if isinstance(value, bytes):
            value = value.decode(errors="replace")
        if value:
            output_parts.append(str(value))
    output = "\n".join(output_parts).strip()
    if len(output) > 4000:
        output = output[-4000:]
    suffix = f": {output}" if output else ""
    return f"Terminal-Bench CLI timed out after {exc.timeout}s{suffix}"


def _tail_text(value: Any, *, limit: int = 2000) -> str:
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    if value is None:
        return ""
    text = str(value)
    return text[-limit:] if len(text) > limit else text
