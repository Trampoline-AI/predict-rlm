from __future__ import annotations

import asyncio
import json
import shlex
import subprocess
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

from predict_rlm.trace import RunTrace
from rlm_gepa import EvaluationContext, RLMGepaExampleResult, RLMGepaProject
from terminal_bench_rlm.scoring import to_gepa_example_result

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


@dataclass(frozen=True)
class TerminalBenchTaskRunResult:
    task_id: str
    trial_result: Any
    traces: list[Any]
    run_dir: Path | None = None
    error: str | None = None


class TerminalBenchHarnessRunner(Protocol):
    async def run(self, request: TerminalBenchTaskRunRequest) -> TerminalBenchTaskRunResult: ...


class HarborSubprocessHarnessRunner:
    """Runs Terminal-Bench 2.x tasks through Harbor's CLI with the PredictRLM agent."""

    def __init__(self, *, cwd: Path | None = None) -> None:
        self.cwd = cwd or Path(__file__).resolve().parents[2]

    async def run(self, request: TerminalBenchTaskRunRequest) -> TerminalBenchTaskRunResult:
        return await asyncio.to_thread(self._run_sync, request)

    def _run_sync(self, request: TerminalBenchTaskRunRequest) -> TerminalBenchTaskRunResult:
        config = request.config
        output_dir = _resolve_output_dir(request)
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            *shlex.split(config.harbor_executable),
            "run",
            "-d",
            config.harbor_dataset,
            "--include-task-name",
            request.task_id,
            "--agent-import-path",
            "terminal_bench_rlm.tools.tbench_agent:HarborPredictRLMAgent",
            "--n-attempts",
            str(config.n_attempts),
            "--n-concurrent",
            str(config.n_concurrent_trials),
            "--jobs-dir",
            str(output_dir),
            "--job-name",
            request.run_id,
        ]
        for key, value in _agent_kwargs(request).items():
            cmd.extend(["--agent-kwarg", f"{key}={value}"])
        if config.no_rebuild:
            cmd.append("--no-force-build")
        else:
            cmd.append("--force-build")
        cmd.append("--delete" if config.cleanup else "--no-delete")
        completed = subprocess.run(cmd, cwd=self.cwd, check=False, text=True)
        run_dir = output_dir / request.run_id
        if completed.returncode != 0:
            return TerminalBenchTaskRunResult(
                task_id=request.task_id,
                trial_result={"is_resolved": False, "parser_results": {}},
                traces=[],
                run_dir=run_dir,
                error=_subprocess_error(completed),
            )
        return self._load_result(request, run_dir)

    def _load_result(
        self,
        request: TerminalBenchTaskRunRequest,
        run_dir: Path,
    ) -> TerminalBenchTaskRunResult:
        return _load_task_run_result(request, run_dir)


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

        completed = subprocess.run(
            cmd,
            cwd=self.cwd,
            check=False,
            text=True,
        )
        run_dir = output_dir / request.run_id
        if completed.returncode != 0:
            error = _subprocess_error(completed)
            trial_result: dict[str, Any] = {"is_resolved": False, "parser_results": {}}
            return TerminalBenchTaskRunResult(
                task_id=request.task_id,
                trial_result=trial_result,
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

    async def evaluate_example(
        self,
        candidate: dict[str, str],
        example: TerminalBenchExample,
        context: EvaluationContext,
    ) -> RLMGepaExampleResult:
        request = TerminalBenchTaskRunRequest(
            task_id=example.task_id,
            instruction=example.instruction,
            skill_instructions=candidate[COMPONENT_SKILL],
            lm=context.lm,
            sub_lm=context.sub_lm,
            max_iterations=context.max_iterations,
            task_timeout=context.task_timeout,
            verbose_rlm=context.verbose_rlm,
            output_dir=context.output_dir,
            run_id=_run_id(context.kind, example.task_id),
            config=self.config,
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
    if config.harness_backend == "harbor":
        return HarborSubprocessHarnessRunner()
    if config.harness_backend == "python":
        return TerminalBenchInProcessHarnessRunner()
    if config.harness_backend == "cli":
        return TerminalBenchSubprocessHarnessRunner()
    raise ValueError(f"Unsupported Terminal-Bench harness backend: {config.harness_backend}")


def _resolve_output_dir(request: TerminalBenchTaskRunRequest) -> Path:
    output_dir = request.config.terminal_bench_output_dir
    if output_dir.is_absolute():
        return output_dir
    return request.output_dir / output_dir


def _agent_kwargs(request: TerminalBenchTaskRunRequest) -> dict[str, str]:
    config = request.config
    kwargs = {
        "lm": _model_name(request.lm),
        "sub_lm": _model_name(request.sub_lm),
        "max_iterations": str(request.max_iterations),
        "exec_timeout": str(request.task_timeout),
        "skill_instructions": request.skill_instructions,
    }
    lm_reasoning_effort = _reasoning_effort(request.lm)
    if lm_reasoning_effort is not None:
        kwargs["lm_reasoning_effort"] = lm_reasoning_effort
    sub_lm_reasoning_effort = _reasoning_effort(request.sub_lm)
    if sub_lm_reasoning_effort is not None:
        kwargs["sub_lm_reasoning_effort"] = sub_lm_reasoning_effort
    if request.verbose_rlm:
        kwargs["verbose"] = "true"
    if config.codex_lm:
        kwargs["codex_lm"] = "true"
        if config.codex_lm_exclude:
            kwargs["codex_lm_exclude"] = ",".join(config.codex_lm_exclude)
    return kwargs


def _agent_timeout_with_cleanup_grace(request: TerminalBenchTaskRunRequest) -> int:
    grace = max(0, int(request.config.timeout_cleanup_grace_sec))
    return max(1, int(request.task_timeout) - grace)


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
    return (
        "You are solving Terminal-Bench tasks inside a Linux task container. "
        "Read the task instruction carefully, inspect the filesystem before making "
        "changes, and use Python as an orchestration layer for real machine work: "
        "call subprocess.run or equivalent shell commands to inspect files, run "
        "programs, edit code, start services, compile artifacts, and execute tests. "
        "When a task needs missing dependencies, install missing packages with the "
        "available package managers (for example apt, pip, npm, cargo, or language-"
        "specific installers) unless the task instruction forbids it. Prefer "
        "repeatable, idempotent commands and keep state changes limited to the exact "
        "files/services requested by the task. Use programmatic tools for binary, "
        "image, audio, video, archive, or other non-text inputs rather than guessing. "
        "Use small verification loops: run available tests, inspect logs, and check "
        "command outputs before finishing. Do not rely on Terminal-Bench wrapper "
        "tools; operate through the PredictRLM interpreter in the container."
    )


def _model_name(model: Any) -> str:
    return str(getattr(model, "model", model))


def _reasoning_effort(model: Any) -> str | None:
    kwargs = getattr(model, "kwargs", None)
    if not isinstance(kwargs, dict):
        return None
    effort = kwargs.get("reasoning_effort")
    return str(effort) if effort else None


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
