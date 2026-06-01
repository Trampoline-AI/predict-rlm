from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlm_gepa import AgentSpec, OptimizeConfig

COMPONENT_SKILL = "skill_instructions"

DEFAULT_TRAIN_TASK_IDS = (
    "configure-git-webserver",
    "extract-moves-from-video",
)
DEFAULT_VAL_TASK_IDS = ("super-benchmark-upet",)


@dataclass
class TerminalBenchGepaConfig(OptimizeConfig):
    """Configuration for optimizing the Terminal-Bench PredictRLM agent."""

    dataset_name: str = "terminal-bench-core"
    dataset_version: str = "0.1.1"
    harbor_dataset: str = "terminal-bench/terminal-bench-2-1"
    harbor_environment: str = "docker"
    train_task_ids: tuple[str, ...] = DEFAULT_TRAIN_TASK_IDS
    val_task_ids: tuple[str, ...] = DEFAULT_VAL_TASK_IDS
    train_limit: int | None = None
    val_limit: int | None = None
    terminal_bench_output_dir: Path = Path("runs/gepa-terminal-bench")
    terminal_bench_executable: str = ".terminal-bench-venv/bin/tb"
    harbor_executable: str = "harbor"
    harness_backend: str = "harbor"
    harbor_controller_locality: str = "auto"
    harbor_agent_interpreter_mode: str = "auto"
    harbor_remote_workdir: str = "/tmp/predict_rlm_terminal_bench"
    harbor_cpus: str = "auto"
    harbor_memory: str = "auto"
    timeout_cleanup_grace_sec: int = 60
    harbor_task_cache_dir: Path | None = None
    n_attempts: int = 1
    n_concurrent_trials: int = 1
    cleanup: bool = True
    no_rebuild: bool = False
    upload_results: bool = False
    codex_lm: bool = False
    codex_lm_exclude: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["terminal_bench_output_dir"] = str(self.terminal_bench_output_dir)
        payload["harness_backend"] = self.harness_backend
        payload["harbor_executable"] = self.harbor_executable
        payload["harbor_dataset"] = self.harbor_dataset
        payload["harbor_environment"] = self.harbor_environment
        payload["harbor_controller_locality"] = self.harbor_controller_locality
        payload["harbor_agent_interpreter_mode"] = self.harbor_agent_interpreter_mode
        payload["harbor_remote_workdir"] = self.harbor_remote_workdir
        payload["harbor_cpus"] = self.harbor_cpus
        payload["harbor_memory"] = self.harbor_memory
        payload["train_task_ids"] = list(self.train_task_ids)
        payload["val_task_ids"] = list(self.val_task_ids)
        payload["codex_lm_exclude"] = list(self.codex_lm_exclude)
        if self.harbor_task_cache_dir is not None:
            payload["harbor_task_cache_dir"] = str(self.harbor_task_cache_dir)
        return payload


def default_config() -> TerminalBenchGepaConfig:
    return TerminalBenchGepaConfig(
        executor_lm="openai/gpt-5.4-mini",
        executor_sub_lm="openai/gpt-5.4-mini",
        proposer_lm="anthropic/claude-sonnet-4-6",
        proposer_sub_lm="openai/gpt-5.4-mini",
        max_metric_calls=2,
        minibatch_size=1,
        concurrency=1,
        max_iterations=20,
        task_timeout=900,
    )


def _terminal_bench_tool_signatures() -> str:
    from terminal_bench_rlm.tools.tbench_agent import HarborPredictRLMAgent

    init_sig = inspect.signature(HarborPredictRLMAgent.__init__)
    task_sig = inspect.signature(HarborPredictRLMAgent.run)
    return (
        f"HarborPredictRLMAgent.__init__{init_sig}\n"
        "Constructs a PredictRLM-backed Harbor agent. The optimized "
        "`skill_instructions` component is passed as a PredictRLM Skill.\n\n"
        f"HarborPredictRLMAgent.run{task_sig}\n"
        "Runs the RLM inside the Harbor BaseEnvironment for Terminal-Bench 2.x tasks."
    )


def _format_runtime_examples() -> dict[str, list[str]]:
    return {
        "container protocol": [
            "work in the provided Terminal-Bench container filesystem",
            "write durable task outputs where the instruction requests them",
        ],
        "shell workflow": [
            "inspect files before editing",
            "make task changes boldly in small inspectable steps",
            "avoid destructive retries or overwriting the best partial solution",
        ],
        "verification": [
            "run task-local checks when available",
            "read logs and command output before declaring completion",
        ],
    }


TERMINAL_BENCH_SPEC = AgentSpec(
    agent_type=(
        "a Terminal-Bench agent that solves Linux/container tasks by writing "
        "and executing code through PredictRLM inside the benchmark container"
    ),
    use_cases=[
        "configure local services and repositories in a task container",
        "extract or transform files using command-line tools",
        "debug failing scripts until benchmark parser tests pass",
    ],
    runtime_grounding_examples=_format_runtime_examples(),
    tool_signatures=_terminal_bench_tool_signatures(),
    target_signature="instruction: str -> answer: str",
    scoring_description=(
        "Terminal-Bench runs parser tests after the agent finishes. The soft "
        "score is passed parser tests divided by total parser tests. The hard "
        "score is 1 only when every parser test passes."
    ),
    domain_conventions_note=(
        "Terminal-Bench guidance must transfer across shell, service, repo, "
        "and file-manipulation tasks rather than memorizing a single task ID."
    ),
)


def coerce_task_ids(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part) for part in value)
