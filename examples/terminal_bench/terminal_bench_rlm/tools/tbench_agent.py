from __future__ import annotations

import asyncio
import importlib
import importlib.resources
import inspect
import io
import json
import logging
import os
import shlex
import tarfile
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, Callable

import dspy

from terminal_bench_rlm.skills import (
    TERMINAL_BENCH_SKILL_NAME,
    build_terminal_bench_skill,
)

TERMINAL_WRAPPER_TOOL_NAMES = frozenset(
    {"run_terminal_command", "send_terminal_keys", "read_terminal"}
)
PredictRLM: Any | None = None
TerminalBenchRunnerClientAdapter: Any | None = None
DAYTONA_REMOTE_ROOT = "/tmp/predict_rlm_controller"
DAYTONA_REMOTE_HOME = "/tmp/predict_rlm_home"
DAYTONA_REMOTE_RESULT_SENTINEL = "PREDICT_RLM_REMOTE_RESULT_JSON="
logger = logging.getLogger(__name__)
_SOURCE_BUNDLE_RELATIVE_PATHS = (
    "pyproject.toml",
    "README.md",
    "src",
    "examples/terminal_bench/pyproject.toml",
    "examples/terminal_bench/terminal_bench_rlm",
)
_SOURCE_CHECKOUT_SENTINELS = (
    "pyproject.toml",
    "src/predict_rlm/remote/bootstrap_controller.sh",
    "examples/terminal_bench/terminal_bench_rlm/tools/remote_controller.py",
)
_INSTALLED_SOURCE_BUNDLE_PACKAGES = (
    ("predict_rlm", "repo/src/predict_rlm"),
    ("rlm_gepa", "repo/src/rlm_gepa"),
    ("dspy_codex_lm", "repo/src/codex-lm/dspy_codex_lm"),
    ("terminal_bench_rlm", "repo/examples/terminal_bench/terminal_bench_rlm"),
)
_GENERATED_SOURCE_BUNDLE_PYPROJECT_TEMPLATE = """\
[project]
name = "predict-rlm"
version = "0.0.0"
requires-python = ">=3.12"
dependencies = [
    "deno>=2",
    "dspy>=3.1.2",
    "nest-asyncio>=1.6.0",
    "pydantic>=2.8.2,<3",
]

[project.optional-dependencies]
codex-lm = [
    "litellm>=1.50",
    "openai>=1.60",
    "tenacity>=9.1.4",
]
gepa = [
    "gepa>=0.0.26",
    "tqdm>=4.0.0",
]
gepa-viz = [
    "plotly>=5.0.0",
    "kaleido>=0.2.0",
]

[tool.hatch.build.targets.wheel]
packages = [
{packages}
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""
_GENERATED_TERMINAL_BENCH_PYPROJECT = """\
[project]
name = "terminal-bench-rlm"
version = "0.0.0"
requires-python = ">=3.12"
dependencies = ["predict-rlm[codex-lm,gepa,gepa-viz]"]

[tool.hatch.build.targets.wheel]
packages = ["terminal_bench_rlm"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""
_SECRET_PAYLOAD_KEY_PARTS = (
    "api_key",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)
_REMOTE_CONTROLLER_ENV_KEYS = (
    "CODEX_LM_AUTH_PROFILE",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_ORGANIZATION",
)
_SUBMIT_CONFIRMATION_MODE_TERMINAL_BENCH = "terminal_bench"


def _tool_name(tool: Callable[..., Any]) -> str:
    return getattr(tool, "__name__", type(tool).__name__)


def _validate_tools(tools: dict[str, Callable[..., Any]] | list[Callable[..., Any]] | None) -> None:
    if tools is None:
        return
    names = set(tools) if isinstance(tools, dict) else {_tool_name(tool) for tool in tools}
    forbidden = names & TERMINAL_WRAPPER_TOOL_NAMES
    if forbidden:
        raise ValueError(
            "Terminal-Bench PredictRLM integration must not register terminal wrapper "
            f"tools: {sorted(forbidden)}"
        )


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_codex_lm_exclude(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part) for part in value if str(part))


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_submit_confirmation_mode(value: Any) -> str | None:
    if value in (None, False):
        return None
    if value is True:
        return _SUBMIT_CONFIRMATION_MODE_TERMINAL_BENCH
    text = str(value).strip().lower()
    if text in {"", "0", "false", "none", "off", "no"}:
        return None
    if text == _SUBMIT_CONFIRMATION_MODE_TERMINAL_BENCH:
        return text
    raise ValueError(
        "submit_confirmation_mode must be 'terminal_bench', True, False, or None."
    )


def _coerce_float(value: Any) -> float:
    if value is None:
        return 900.0
    return float(value)


def _build_lm(value: Any, reasoning_effort: str | None, service_tier: str | None = None) -> Any:
    if not isinstance(value, str) or reasoning_effort is None:
        return value
    kwargs: dict[str, Any] = {"cache": False}
    if reasoning_effort != "none":
        kwargs["reasoning_effort"] = reasoning_effort
    if service_tier:
        kwargs["service_tier"] = service_tier
    return dspy.LM(value, **kwargs)


def _install_codex_lm_monkeypatch(exclude: tuple[str, ...]) -> None:
    try:
        cli_module = importlib.import_module("dspy_codex_lm.cli")
    except ImportError as exc:
        raise RuntimeError(
            "CodexLM was requested for the Terminal-Bench PredictRLM agent, "
            "but predict-rlm[codex-lm] is not importable in the Terminal-Bench "
            "agent process. Run examples/terminal_bench/scripts/setup_terminal_bench.sh "
            "to install the local predict-rlm package with its codex-lm extra."
        ) from exc
    os.environ.setdefault("OPENAI_API_KEY", "codex-lm")
    cli_module.install_monkeypatch(exclude=exclude)


def _get_task_instruction(task: Any) -> str:
    for attr in ("instruction", "prompt", "description", "question"):
        value = getattr(task, attr, None)
        if value:
            return str(value)
    return str(task)


def _signature_with_task_instruction(signature: Any, task_instruction: str) -> Any:
    base_signature = (
        signature if hasattr(signature, "output_fields") else dspy.Signature(signature, "")
    )
    instructions = (getattr(base_signature, "instructions", "") or "").strip()
    task_block = f"## Terminal-Bench task instruction\n\n{task_instruction.strip()}"
    if instructions:
        instructions = f"{instructions}\n\n{task_block}"
    else:
        instructions = task_block
    return dspy.Signature({**base_signature.output_fields}, instructions)


def _format_submit_confirmation_state(context: Any) -> str:
    submitted_payload = getattr(context, "submitted_payload", {})
    try:
        payload_text = json.dumps(submitted_payload, indent=2, sort_keys=True, default=str)
    except TypeError:
        payload_text = str(submitted_payload)
    latest_observation = str(getattr(context, "latest_observation", "") or "").strip()
    iteration = getattr(context, "iteration", None)
    parts = [f"Submitted payload:\n{payload_text}"]
    parts.append(
        "Latest terminal observation:\n"
        f"{latest_observation or '(no terminal observation captured)'}"
    )
    if iteration is not None:
        parts.append(f"Submit iteration: {iteration}")
    return "\n\n".join(parts)


def _build_terminal_bench_submit_confirmation(task_instruction: str) -> Callable[[Any], str]:
    instruction = task_instruction.strip() or "(no task instruction provided)"

    def confirm(context: Any) -> str:
        return (
            "Original task:\n"
            f"{instruction}\n\n"
            "Current submitted payload / terminal state:\n"
            f"{_format_submit_confirmation_state(context)}\n\n"
            "Are you sure you want to mark the task as complete?\n"
            "[!] Final verification gate\n"
            "Before calling SUBMIT again, re-read the original task and compare it "
            "to the current final state. Use the skill's todos and required "
            "verification entries as the contract.\n"
            "- Convert every required outcome, file/path, command behavior, service, "
            "format, semantic expectation, and negative constraint into concrete "
            "required verification predicates.\n"
            "- Mark each todo done and each required verification entry verified only "
            "with fresh evidence from the current final state: task-local tests, "
            "public verifier commands, verifier-shaped emulation, parser/load/exercise "
            "checks, or direct inspection of named paths/processes/configs.\n"
            "- Do not rely on stale debug history, prior partial runs, file existence "
            "alone, plausibility, or self-attestation as verification evidence.\n"
            "- Any unverified todo or required verification entry is a blocker to "
            "SUBMIT; continue checking or fixing instead of confirming.\n"
            "- The second SUBMIT is only to confirm the same already-verified answer; "
            "do not use it to do new work, change the answer, or defer verification.\n"
            "After this point, grading will begin and no further edits will be possible. "
            "If every todo and required verification entry is verified now, call SUBMIT again. "
            "Otherwise continue working."
        )

    return confirm


def _build_submit_confirmation(
    mode: str | None,
    task_instruction: str,
) -> Callable[[Any], str] | None:
    if mode is None:
        return None
    if mode == _SUBMIT_CONFIRMATION_MODE_TERMINAL_BENCH:
        return _build_terminal_bench_submit_confirmation(task_instruction)
    raise ValueError(f"Unsupported submit_confirmation_mode: {mode}")


def _get_runtime(task: Any = None, session: Any = None, **kwargs: Any) -> Any:
    if "container" in kwargs and kwargs["container"] is not None:
        return kwargs["container"]
    if session is not None and getattr(session, "container", None) is not None:
        return session
    if task is not None:
        task_session = getattr(task, "session", None)
        if task_session is not None and getattr(task_session, "container", None) is not None:
            return task_session
        if getattr(task, "container", None) is not None:
            return task.container
    raise ValueError("Could not locate Terminal-Bench session container")


def _coerce_answer(result: Any) -> str:
    if isinstance(result, str):
        return result
    for attr in ("answer", "summary", "output"):
        value = getattr(result, attr, None)
        if value is not None:
            return str(value)
    return str(result)


def _write_trace(trace: Any, logging_dir: Path | None, *, path: Path | None = None) -> None:
    if trace is None or logging_dir is None:
        return
    logging_dir.mkdir(parents=True, exist_ok=True)
    trace_path = path or logging_dir / f"predict_rlm_trace_{uuid.uuid4().hex[:8]}.json"
    if hasattr(trace, "to_exportable_json"):
        trace_path.write_text(trace.to_exportable_json(), encoding="utf-8")
    else:
        trace_path.write_text(str(trace), encoding="utf-8")


def _write_phase_event(
    path: Path | None,
    *,
    task_id: str | None,
    event: str,
    phase: str,
    status: str,
    **fields: Any,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "event": event,
        "phase": phase,
        "status": status,
        "task_id": task_id,
        **fields,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    duration = payload.get("duration_seconds")
    duration_text = f" duration_seconds={duration:.3f}" if isinstance(duration, (int, float)) else ""
    print(
        f"phase_event task={task_id} phase={phase} event={event} "
        f"status={status}{duration_text}",
        flush=True,
    )


def _set_context_answer(context: Any, answer: str) -> None:
    if "metadata" in getattr(type(context), "model_fields", {}):
        metadata = dict(context.metadata or {})
        metadata["answer"] = answer
        context.metadata = metadata
    else:
        setattr(context, "answer", answer)


def _predict_rlm_class() -> Any:
    global PredictRLM
    if PredictRLM is None:
        PredictRLM = getattr(importlib.import_module("predict_rlm"), "PredictRLM")
    return PredictRLM


def _skill_class() -> Any:
    return getattr(importlib.import_module("predict_rlm"), "Skill")


def _with_terminal_bench_skill(
    rlm_kwargs: dict[str, Any],
    skill_instructions: str | None,
) -> None:
    skills = [
        skill
        for skill in list(rlm_kwargs.get("skills") or [])
        if getattr(skill, "name", None) != TERMINAL_BENCH_SKILL_NAME
    ]
    skills.append(build_terminal_bench_skill(_skill_class(), skill_instructions))
    rlm_kwargs["skills"] = skills


def _interpreter_class() -> Any:
    global TerminalBenchRunnerClientAdapter
    if TerminalBenchRunnerClientAdapter is None:
        TerminalBenchRunnerClientAdapter = getattr(
            importlib.import_module(".container_runner", __package__),
            "TerminalBenchRunnerClientAdapter",
        )
    return TerminalBenchRunnerClientAdapter



@dataclass
class LocalAgentResult:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    failure_mode: str | None = None
    timestamped_markers: list[Any] = field(default_factory=list)


def _terminal_bench_base_agent() -> type[Any] | None:
    import importlib

    for module_name in (
        "terminal_bench.agents.base_agent",
        "terminal_bench.agents.base",
        "terminal_bench.agent.base",
        "terminal_bench.agents",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        base_agent = getattr(module, "BaseAgent", None)
        if base_agent is not None:
            return base_agent
    return None


def _terminal_bench_agent_result() -> type[Any]:
    import importlib

    for module_name in (
        "terminal_bench.agents.base_agent",
        "terminal_bench.agents.base",
        "terminal_bench.agent.base",
        "terminal_bench.agents",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        agent_result = getattr(module, "AgentResult", None)
        if agent_result is not None:
            return agent_result
    return LocalAgentResult


def _make_agent_result(
    *,
    total_input_tokens: int = 0,
    total_output_tokens: int = 0,
    failure_mode: str | None = None,
    timestamped_markers: list[Any] | None = None,
) -> Any:
    agent_result = _terminal_bench_agent_result()
    kwargs = {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "timestamped_markers": timestamped_markers or [],
    }
    if failure_mode is not None:
        kwargs["failure_mode"] = failure_mode
    try:
        return agent_result(**kwargs)
    except TypeError:
        return LocalAgentResult(**kwargs)


class _TerminalBenchRLMBaseAgentMixin:
    """Terminal-Bench agent adapter that runs PredictRLM code inside the task container."""

    @staticmethod
    def name() -> str:
        return "predict-rlm"

    def __init__(
        self,
        *,
        signature: Any = "instruction -> answer",
        tools: dict[str, Callable[..., Any]] | list[Callable[..., Any]] | None = None,
        skill_instructions: str | None = None,
        interpreter_kwargs: dict[str, Any] | None = None,
        codex_lm: bool | str = False,
        codex_lm_exclude: tuple[str, ...] | list[str] | str | None = None,
        lm_reasoning_effort: str | None = None,
        sub_lm_reasoning_effort: str | None = None,
        lm_service_tier: str | None = None,
        sub_lm_service_tier: str | None = None,
        exec_timeout: float | str | None = None,
        no_rebuild: bool | None = None,
        phase_log_path: str | Path | None = None,
        task_id: str | None = None,
        codex_lm_debug: bool | str = False,
        codex_lm_debug_log: str | None = None,
        predict_rlm_debug: bool | str = False,
        predict_rlm_debug_json: bool | str = False,
        predict_rlm_debug_log: str | None = None,
        submit_confirmation_mode: str | bool | None = _SUBMIT_CONFIRMATION_MODE_TERMINAL_BENCH,
        **predict_rlm_kwargs: Any,
    ) -> None:
        if "interpreter_mode" in predict_rlm_kwargs:
            raise TypeError("interpreter_mode is not a supported Terminal-Bench PredictRLM agent parameter")
        _validate_tools(tools)
        self.signature = signature
        self.tools = tools
        self.skill_instructions = skill_instructions
        self.interpreter_kwargs = dict(interpreter_kwargs or {})
        self.interpreter_kwargs.setdefault("exec_timeout", _coerce_float(exec_timeout))
        self.codex_lm = _coerce_bool(codex_lm)
        self.codex_lm_exclude = _coerce_codex_lm_exclude(codex_lm_exclude)
        self.lm_reasoning_effort = _coerce_optional_text(lm_reasoning_effort)
        self.sub_lm_reasoning_effort = _coerce_optional_text(sub_lm_reasoning_effort)
        self.lm_service_tier = _coerce_optional_text(lm_service_tier)
        self.sub_lm_service_tier = _coerce_optional_text(sub_lm_service_tier)
        self.phase_log_path = Path(phase_log_path) if phase_log_path is not None else None
        self.task_id = task_id
        self.predict_rlm_kwargs = predict_rlm_kwargs
        self.codex_lm_debug = _coerce_bool(codex_lm_debug)
        self.codex_lm_debug_log = _coerce_optional_text(codex_lm_debug_log)
        self.predict_rlm_debug = _coerce_bool(predict_rlm_debug)
        self.predict_rlm_debug_json = _coerce_bool(predict_rlm_debug_json)
        self.predict_rlm_debug_log = _coerce_optional_text(predict_rlm_debug_log)
        self.submit_confirmation_mode = _coerce_submit_confirmation_mode(submit_confirmation_mode)
        if self.codex_lm_debug:
            os.environ["CODEX_LM_DEBUG"] = "1"
        if self.codex_lm_debug_log:
            os.environ["CODEX_LM_DEBUG_LOG"] = self.codex_lm_debug_log
        if self.predict_rlm_debug:
            os.environ["PREDICT_RLM_DEBUG"] = "1"
        if self.predict_rlm_debug_json:
            os.environ["PREDICT_RLM_DEBUG_JSON"] = "1"
        if self.predict_rlm_debug_log:
            os.environ["PREDICT_RLM_DEBUG_LOG"] = self.predict_rlm_debug_log

    def perform_task(
        self,
        instruction: str | Any = "",
        session: Any | None = None,
        logging_dir: Path | None = None,
        **kwargs: Any,
    ) -> Any:
        if session is None:
            session = kwargs.pop("session", None)
        task = kwargs.pop("task", None)
        if isinstance(instruction, str):
            task_instruction = instruction or kwargs.pop("instruction", "")
            if task_instruction == "" and task is not None:
                task_instruction = _get_task_instruction(task)
        else:
            task = instruction
            task_instruction = kwargs.pop("instruction", None) or _get_task_instruction(task)
        runtime = _get_runtime(task=task, session=session, **kwargs)
        if self.codex_lm:
            _install_codex_lm_monkeypatch(self.codex_lm_exclude)
        interpreter = _interpreter_class()(runtime, **self.interpreter_kwargs)
        try:
            rlm_kwargs = dict(self.predict_rlm_kwargs)
            if "lm" in rlm_kwargs:
                rlm_kwargs["lm"] = _build_lm(
                    rlm_kwargs["lm"], self.lm_reasoning_effort, self.lm_service_tier
                )
            if "sub_lm" in rlm_kwargs:
                rlm_kwargs["sub_lm"] = _build_lm(
                    rlm_kwargs["sub_lm"],
                    self.sub_lm_reasoning_effort,
                    self.sub_lm_service_tier,
                )
            rlm_kwargs["interpreter"] = interpreter
            _with_terminal_bench_skill(rlm_kwargs, self.skill_instructions)
            if "max_iterations" in rlm_kwargs:
                rlm_kwargs["max_iterations"] = int(rlm_kwargs["max_iterations"])
            if self.tools is not None:
                rlm_kwargs["tools"] = self.tools
            submit_confirmation = _build_submit_confirmation(
                self.submit_confirmation_mode,
                task_instruction,
            )
            if submit_confirmation is not None:
                if "submit_confirmation" in rlm_kwargs:
                    raise ValueError(
                        "submit_confirmation_mode cannot be combined with a custom "
                        "submit_confirmation callback."
                    )
                rlm_kwargs["submit_confirmation"] = submit_confirmation
            signature = _signature_with_task_instruction(self.signature, task_instruction)
            rlm = _predict_rlm_class()(signature, **rlm_kwargs)
            result = rlm.acall()
            if inspect.isawaitable(result):
                result = asyncio.run(result)
            _write_trace(getattr(result, "trace", None), logging_dir)
            _coerce_answer(result)
            return _make_agent_result()
        except BaseException as exc:
            _write_trace(getattr(exc, "trace", None), logging_dir)
            raise
        finally:
            interpreter.shutdown()


class HarborPredictRLMBaseAgent(_TerminalBenchRLMBaseAgentMixin, ABC):
    """Shared Harbor BaseAgent-compatible state for remote Terminal-Bench adapters."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        logger: Any | None = None,
        mcp_servers: list[Any] | None = None,
        skills_dir: str | None = None,
        extra_env: dict[str, str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.logs_dir = Path(logs_dir)
        self.model_name = model_name
        self.logger = logger
        self.mcp_servers = mcp_servers or []
        self.skills_dir = skills_dir
        self.extra_env = extra_env or {}
        super().__init__(*args, **kwargs)

    def version(self) -> str | None:
        return None

    def to_agent_info(self) -> Any:
        try:
            agent_info_cls = getattr(importlib.import_module("harbor.models.trial.result"), "AgentInfo")
        except ImportError:
            return {"name": self.name(), "version": self.version() or "unknown", "model_info": None}
        return agent_info_cls(name=self.name(), version=self.version() or "unknown", model_info=None)

    def populate_context_post_run(self, _context: Any) -> None:
        return None

    @abstractmethod
    async def setup(self, environment: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        raise NotImplementedError


def _repo_root() -> Path:
    root = _source_checkout_root()
    if root is None:
        raise RuntimeError("Could not locate predict-rlm source checkout")
    return root


def _source_checkout_root(start: Path | None = None) -> Path | None:
    current = (start or Path(__file__).resolve()).parent
    for candidate in (current, *current.parents):
        if all((candidate / sentinel).exists() for sentinel in _SOURCE_CHECKOUT_SENTINELS):
            return candidate
    return None


def _source_bundle_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(info.name).parts
    ignored = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "downloaded-gcp-artifacts",
        "ops",
    }
    if any(part in ignored for part in parts):
        return None
    if info.name.endswith((".pyc", ".pyo")):
        return None
    return info


def _archive_add_bytes(archive: tarfile.TarFile, arcname: str, data: bytes) -> None:
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mode = 0o644
    if arcname.endswith(".sh"):
        info.mode = 0o755
    archive.addfile(info, io.BytesIO(data))


def _resource_bundle_filter(arcname: str) -> bool:
    parts = Path(arcname).parts
    if any(part in {".git", "__pycache__"} for part in parts):
        return False
    return not arcname.endswith((".pyc", ".pyo"))


def _archive_add_resource_tree(
    archive: tarfile.TarFile,
    resource: Traversable,
    arcroot: str,
) -> None:
    for child in resource.iterdir():
        arcname = f"{arcroot}/{child.name}"
        if not _resource_bundle_filter(arcname):
            continue
        if child.is_dir():
            _archive_add_resource_tree(archive, child, arcname)
        elif child.is_file():
            _archive_add_bytes(archive, arcname, child.read_bytes())


def _generated_source_bundle_pyproject(package_paths: list[str]) -> str:
    packages = "".join(f'    "{path}",\n' for path in package_paths)
    return _GENERATED_SOURCE_BUNDLE_PYPROJECT_TEMPLATE.format(packages=packages.rstrip())


def _create_installed_source_bundle(destination: Path) -> None:
    with tarfile.open(destination, "w:gz") as archive:
        resources: list[tuple[Traversable, str, str]] = []
        package_paths: list[str] = []
        missing: list[str] = []
        for package_name, arcroot in _INSTALLED_SOURCE_BUNDLE_PACKAGES:
            try:
                resource = importlib.resources.files(package_name)
            except ModuleNotFoundError:
                missing.append(package_name)
                continue
            resources.append((resource, arcroot, package_name))
            if arcroot.startswith("repo/"):
                package_paths.append(arcroot[len("repo/") :])
        if "predict_rlm" in missing or "terminal_bench_rlm" in missing:
            raise RuntimeError(
                "Could not build Daytona source bundle from installed packages; "
                f"missing required package resources: {', '.join(missing)}"
            )
        _archive_add_bytes(
            archive,
            "repo/pyproject.toml",
            _generated_source_bundle_pyproject(package_paths).encode("utf-8"),
        )
        _archive_add_bytes(
            archive,
            "repo/examples/terminal_bench/pyproject.toml",
            _GENERATED_TERMINAL_BENCH_PYPROJECT.encode("utf-8"),
        )
        for resource, arcroot, _package_name in resources:
            _archive_add_resource_tree(archive, resource, arcroot)


def _create_source_bundle(destination: Path) -> None:
    root = _source_checkout_root()
    if root is None:
        _create_installed_source_bundle(destination)
        return
    with tarfile.open(destination, "w:gz") as archive:
        for relative in _SOURCE_BUNDLE_RELATIVE_PATHS:
            source = root / relative
            if source.exists():
                archive.add(source, arcname=str(Path("repo") / relative), filter=_source_bundle_filter)


async def _resolve_remote_call(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _remote_exec(environment: Any, command: str, *, timeout: int | None = None) -> Any:
    for name in ("exec", "run"):
        method = getattr(environment, name, None)
        if method is None:
            continue
        if timeout is None:
            try:
                return await _resolve_remote_call(method(command=command))
            except TypeError as keyword_exc:
                try:
                    return await _resolve_remote_call(method(command))
                except TypeError:
                    raise keyword_exc
        try:
            return await _resolve_remote_call(method(command=command, timeout_sec=int(timeout)))
        except TypeError as keyword_exc:
            try:
                return await _resolve_remote_call(method(command, timeout=timeout))
            except TypeError:
                raise keyword_exc
    raise TypeError("Daytona remote PredictRLM environment does not expose exec/run")


async def _remote_upload_file(environment: Any, host_path: str, remote_path: str) -> None:
    for name in ("upload_file", "copy_to", "put_file"):
        method = getattr(environment, name, None)
        if method is not None:
            await _resolve_remote_call(method(host_path, remote_path))
            return
    raise TypeError("Daytona remote PredictRLM environment does not expose upload_file/copy_to")


async def _remote_upload_dir(environment: Any, host_path: str, remote_path: str) -> None:
    method = getattr(environment, "upload_dir", None)
    if method is None:
        raise TypeError("Daytona remote PredictRLM CodexLM auth upload requires upload_dir")
    await _resolve_remote_call(method(host_path, remote_path))


async def _remote_download_file(environment: Any, remote_path: str, host_path: str) -> None:
    for name in ("download_file", "copy_from", "get_file"):
        method = getattr(environment, name, None)
        if method is not None:
            await _resolve_remote_call(method(remote_path, host_path))
            return
    raise TypeError("Daytona remote PredictRLM environment does not expose download_file/copy_from")


def _extract_tar_safely(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Refusing to extract remote trace outside logs dir: {member.name}")
        archive.extractall(destination, filter="data")


def _remote_returncode(result: Any) -> int | None:
    for attr in ("returncode", "return_code", "exit_code"):
        value = getattr(result, attr, None)
        if value is not None:
            return int(value)
    if isinstance(result, dict):
        for key in ("returncode", "return_code", "exit_code"):
            value = result.get(key)
            if value is not None:
                return int(value)
    return None


def _remote_stdout(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return str(result.get("stdout", "") or "")
    return str(getattr(result, "stdout", "") or "")


def _remote_stderr(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("stderr", "") or "")
    return str(getattr(result, "stderr", "") or "")


def _remote_output_tail(result: Any) -> str:
    output = "\n".join(part for part in (_remote_stdout(result), _remote_stderr(result)) if part)
    lines = output.splitlines()[-20:]
    return "\n".join(lines)


def _raise_for_remote_failure(result: Any, operation: str) -> None:
    returncode = _remote_returncode(result)
    if returncode not in (None, 0):
        tail = _remote_output_tail(result)
        details = f":\n{tail}" if tail else ""
        raise RuntimeError(
            f"Daytona remote PredictRLM failed while {operation}: exit code {returncode}{details}"
        )


def _reject_secret_payload_values(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            if any(part in lowered for part in _SECRET_PAYLOAD_KEY_PARTS) and item not in (
                None,
                "",
                False,
            ):
                dotted = ".".join((*path, key_text))
                raise ValueError(f"Refusing to put secret-like payload field in remote JSON: {dotted}")
            _reject_secret_payload_values(item, (*path, key_text))
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _reject_secret_payload_values(item, (*path, str(index)))


def _json_dumps_non_secret_payload(payload: dict[str, Any]) -> str:
    _reject_secret_payload_values(payload)
    try:
        return json.dumps(payload, sort_keys=True)
    except TypeError as exc:
        raise TypeError(
            "Daytona remote PredictRLM payload must be JSON-serializable; "
            "custom Python objects such as callables are not supported by the remote adapter."
        ) from exc


def _parse_remote_result(result: Any) -> dict[str, Any]:
    sentinel_payload = None
    for line in _remote_stdout(result).splitlines():
        if line.startswith(DAYTONA_REMOTE_RESULT_SENTINEL):
            sentinel_payload = line[len(DAYTONA_REMOTE_RESULT_SENTINEL) :]
    if sentinel_payload is None:
        returncode = _remote_returncode(result)
        raise RuntimeError(
            "Daytona remote PredictRLM did not emit the result sentinel"
            f" (exit code {returncode})"
        )
    parsed = json.loads(sentinel_payload)
    if not isinstance(parsed, dict):
        raise RuntimeError("Daytona remote PredictRLM result sentinel was not a JSON object")
    return parsed


class DaytonaRemotePredictRLMAgent(HarborPredictRLMBaseAgent):
    """Harbor adapter that runs the PredictRLM controller inside Daytona."""

    def __init__(
        self,
        *args: Any,
        remote_root: str = DAYTONA_REMOTE_ROOT,
        remote_home: str = DAYTONA_REMOTE_HOME,
        remote_log_stream: bool | str = True,
        remote_log_poll_interval: float | str = 5.0,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("exec_timeout", None)
        self.remote_root = remote_root.rstrip("/") or DAYTONA_REMOTE_ROOT
        self.remote_home = remote_home.rstrip("/") or DAYTONA_REMOTE_HOME
        self.remote_log_stream = _coerce_bool(remote_log_stream)
        self.remote_log_poll_interval = float(remote_log_poll_interval)
        self._remote_setup_complete = False
        super().__init__(*args, **kwargs)
        self.interpreter_kwargs.pop("exec_timeout", None)
        if self.predict_rlm_debug and not self.predict_rlm_debug_log:
            self.predict_rlm_debug_log = f"{self.remote_root}/predict_rlm_debug.jsonl"
        if self.codex_lm_debug and not self.codex_lm_debug_log:
            self.codex_lm_debug_log = f"{self.remote_root}/codex_lm_debug.jsonl"

    async def setup(self, environment: Any) -> None:
        started = time.monotonic()
        _write_phase_event(
            self.phase_log_path,
            task_id=self.task_id,
            event="agent_setup_start",
            phase="agent_setup",
            status="started",
        )
        try:
            await self._bootstrap_remote_controller(environment)
        except BaseException:
            _write_phase_event(
                self.phase_log_path,
                task_id=self.task_id,
                event="agent_setup_end",
                phase="agent_setup",
                status="failed",
                duration_seconds=time.monotonic() - started,
            )
            raise
        _write_phase_event(
            self.phase_log_path,
            task_id=self.task_id,
            event="agent_setup_end",
            phase="agent_setup",
            status="completed",
            duration_seconds=time.monotonic() - started,
        )

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        if not self._remote_setup_complete:
            await self.setup(environment)
        started = time.monotonic()
        _write_phase_event(
            self.phase_log_path,
            task_id=self.task_id,
            event="agent_run_start",
            phase="agent_eval",
            status="started",
        )
        credentials_uploaded = False
        try:
            credentials_uploaded = await self._upload_codex_lm_credentials(environment)
            payload_json = self._build_remote_payload_json(instruction)
            stop_streaming = asyncio.Event()
            stream_task = self._start_remote_log_stream(environment, stop_streaming)
            try:
                result = await self._run_remote_controller(environment, payload_json)
            finally:
                stop_streaming.set()
                if stream_task is not None:
                    try:
                        await stream_task
                    except Exception as exc:
                        logger.warning("Remote log streaming stopped with an error", exc_info=exc)
            parsed = _parse_remote_result(result)
            if not parsed.get("ok"):
                error_type = parsed.get("error_type") or "RemotePredictRLMError"
                error = parsed.get("error") or "remote controller failed"
                raise RuntimeError(f"Daytona remote PredictRLM failed: {error_type}: {error}")
            await self._download_remote_logs(environment)
            answer = str(parsed.get("answer", ""))
        except BaseException:
            await self._try_download_remote_logs(environment)
            _write_phase_event(
                self.phase_log_path,
                task_id=self.task_id,
                event="agent_run_end",
                phase="agent_eval",
                status="failed",
                duration_seconds=time.monotonic() - started,
            )
            raise
        finally:
            if credentials_uploaded:
                await self._cleanup_codex_lm_credentials(environment)
        _write_phase_event(
            self.phase_log_path,
            task_id=self.task_id,
            event="agent_run_end",
            phase="agent_eval",
            status="completed",
            duration_seconds=time.monotonic() - started,
        )
        _set_context_answer(context, answer)

    async def _bootstrap_remote_controller(self, environment: Any) -> None:
        root = shlex.quote(self.remote_root)
        home = shlex.quote(self.remote_home)
        result = await _remote_exec(
            environment,
            f"rm -rf {root} && mkdir -p {root} {home}",
            timeout=120,
        )
        _raise_for_remote_failure(result, "preparing the remote controller root")
        with tempfile.TemporaryDirectory(prefix="predict-rlm-controller-") as tmpdir:
            bundle_path = Path(tmpdir) / "repo.tar.gz"
            await asyncio.to_thread(_create_source_bundle, bundle_path)
            remote_bundle = f"{self.remote_root}/repo.tar.gz"
            await _remote_upload_file(environment, str(bundle_path), remote_bundle)
        result = await _remote_exec(
            environment,
            f"tar -xzf {shlex.quote(remote_bundle)} -C {root}",
            timeout=120,
        )
        _raise_for_remote_failure(result, "unpacking the remote controller bundle")
        extra = "[codex-lm]" if self.codex_lm else ""
        bootstrap_script = shlex.quote(
            f"{self.remote_root}/repo/src/predict_rlm/remote/bootstrap_controller.sh"
        )
        bootstrap_args = [
            f"sh {bootstrap_script}",
            f"--root {shlex.quote(self.remote_root)}",
            f"--repo {shlex.quote(f'{self.remote_root}/repo')}",
        ]
        if extra:
            bootstrap_args.append(f"--extra {shlex.quote(extra)}")
        bootstrap_args.append("--python 3.12")
        setup_command = " ".join(
            [
                f"HOME={home}",
                "PATH=\"$HOME/.local/bin:$PATH\"",
                "sh",
                "-lc",
                shlex.quote(" ".join(bootstrap_args)),
            ]
        )
        result = await _remote_exec(environment, setup_command, timeout=900)
        _raise_for_remote_failure(result, "installing the remote controller bundle")
        self._remote_setup_complete = True

    async def _upload_codex_lm_credentials(self, environment: Any) -> bool:
        if not self.codex_lm:
            return False
        credentials_dir = Path.home() / ".codex-lm"
        if not credentials_dir.is_dir():
            raise RuntimeError(
                "CodexLM was enabled, but local ~/.codex-lm is not available for opaque upload"
            )
        result = await _remote_exec(
            environment,
            f"mkdir -p {shlex.quote(self.remote_home)}",
            timeout=60,
        )
        _raise_for_remote_failure(result, "preparing the remote CodexLM home")
        try:
            await _remote_upload_dir(
                environment,
                str(credentials_dir),
                f"{self.remote_home}/.codex-lm",
            )
        except BaseException as exc:
            raise RuntimeError(
                "CodexLM credential upload failed; refusing to run without remote auth"
            ) from exc
        return True

    async def _cleanup_codex_lm_credentials(self, environment: Any) -> None:
        try:
            await _remote_exec(
                environment,
                f"rm -rf {shlex.quote(f'{self.remote_home}/.codex-lm')}",
                timeout=60,
            )
        except BaseException:
            return

    async def _try_download_remote_logs(self, environment: Any) -> None:
        try:
            await self._download_remote_logs(environment)
        except BaseException as exc:
            logger.warning("Failed to download remote PredictRLM logs", exc_info=exc)

    async def _download_remote_logs(self, environment: Any) -> None:
        archive_remote_path = f"{self.remote_root}/logs.tar.gz"
        logs_dir = f"{self.remote_root}/logs"
        command = " ".join(
            [
                "sh",
                "-lc",
                shlex.quote(
                    "logs_dir=$1; archive=$2; "
                    "if [ -d \"$logs_dir\" ]; then "
                    "tar -czf \"$archive\" -C \"$logs_dir\" .; "
                    "else tar -czf \"$archive\" -T /dev/null; fi"
                ),
                "--",
                shlex.quote(logs_dir),
                shlex.quote(archive_remote_path),
            ]
        )
        result = await _remote_exec(environment, command, timeout=60)
        _raise_for_remote_failure(result, "archiving remote PredictRLM logs")
        archive_local_path = self.logs_dir / "predict_rlm_logs.tar.gz"
        await _remote_download_file(environment, archive_remote_path, str(archive_local_path))
        _extract_tar_safely(archive_local_path, self.logs_dir)

    def _start_remote_log_stream(
        self,
        environment: Any,
        stop_event: asyncio.Event,
    ) -> asyncio.Task[None] | None:
        paths = [path for path in (self.predict_rlm_debug_log, self.codex_lm_debug_log) if path]
        if not self.remote_log_stream or not paths:
            return None
        return asyncio.create_task(self._stream_remote_logs(environment, paths, stop_event))

    async def _stream_remote_logs(
        self,
        environment: Any,
        paths: list[str],
        stop_event: asyncio.Event,
    ) -> None:
        offsets = {path: 0 for path in paths}
        while not stop_event.is_set():
            for path in paths:
                output, offset = await self._read_remote_log_delta(environment, path, offsets[path])
                offsets[path] = offset
                if output:
                    print(output, end="" if output.endswith("\n") else "\n", flush=True)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.remote_log_poll_interval)
            except TimeoutError:
                pass
        for path in paths:
            output, offset = await self._read_remote_log_delta(environment, path, offsets[path])
            offsets[path] = offset
            if output:
                print(output, end="" if output.endswith("\n") else "\n", flush=True)

    async def _read_remote_log_delta(
        self,
        environment: Any,
        remote_path: str,
        offset: int,
    ) -> tuple[str, int]:
        quoted_path = shlex.quote(remote_path)
        command = " ".join(
            [
                "sh",
                "-lc",
                shlex.quote(
                    "path=$1; offset=$2; "
                    "if [ ! -f \"$path\" ]; then "
                    "printf 'PREDICT_RLM_REMOTE_LOG_OFFSET=%s\\n' \"$offset\"; exit 0; "
                    "fi; "
                    "size=$(wc -c < \"$path\" | tr -d ' '); "
                    "if [ \"$size\" -gt \"$offset\" ]; then "
                    "dd if=\"$path\" bs=1 skip=\"$offset\" count=$((size - offset)) 2>/dev/null; "
                    "fi; "
                    "printf '\\nPREDICT_RLM_REMOTE_LOG_OFFSET=%s\\n' \"$size\""
                ),
                "--",
                quoted_path,
                str(offset),
            ]
        )
        result = await _remote_exec(environment, command, timeout=30)
        stdout = _remote_stdout(result)
        marker = "PREDICT_RLM_REMOTE_LOG_OFFSET="
        next_offset = offset
        lines = stdout.splitlines()
        if lines and lines[-1].startswith(marker):
            next_offset = int(lines[-1][len(marker) :])
            output = "\n".join(lines[:-1])
            if output:
                output += "\n"
            return output, next_offset
        return stdout, next_offset

    def _build_remote_payload_json(self, instruction: str) -> str:
        if self.tools is not None:
            raise TypeError(
                "Daytona remote PredictRLM does not support shipping local Python tools"
            )
        payload = {
            "codex_lm": self.codex_lm,
            "codex_lm_debug": self.codex_lm_debug,
            "codex_lm_debug_log": self.codex_lm_debug_log,
            "codex_lm_exclude": list(self.codex_lm_exclude),
            "instruction": instruction,
            "interpreter_kwargs": self.interpreter_kwargs,
            "lm_reasoning_effort": self.lm_reasoning_effort,
            "lm_service_tier": self.lm_service_tier,
            "logging_dir": f"{self.remote_root}/logs",
            "predict_rlm_debug": self.predict_rlm_debug,
            "predict_rlm_debug_json": self.predict_rlm_debug_json,
            "predict_rlm_debug_log": self.predict_rlm_debug_log,
            "predict_rlm_kwargs": self.predict_rlm_kwargs,
            "signature": self.signature,
            "skill_instructions": self.skill_instructions,
            "sub_lm_reasoning_effort": self.sub_lm_reasoning_effort,
            "sub_lm_service_tier": self.sub_lm_service_tier,
        }
        if self.submit_confirmation_mode is not None:
            payload["submit_confirmation_mode"] = self.submit_confirmation_mode
        return _json_dumps_non_secret_payload(payload)

    def _remote_controller_env_assignments(self) -> list[str]:
        env = {key: os.environ[key] for key in _REMOTE_CONTROLLER_ENV_KEYS if os.environ.get(key)}
        env.update({key: value for key, value in self.extra_env.items() if value})
        return [f"{key}={shlex.quote(str(value))}" for key, value in sorted(env.items())]

    async def _run_remote_controller(self, environment: Any, payload_json: str) -> Any:
        payload_remote_path = f"{self.remote_root}/payload-{uuid.uuid4().hex}.json"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tmp:
            tmp.write(payload_json)
            payload_host_path = tmp.name
        try:
            await _remote_upload_file(environment, payload_host_path, payload_remote_path)
        finally:
            try:
                os.unlink(payload_host_path)
            except OSError:
                pass
        pythonpath = ":".join(
            [
                f"{self.remote_root}/repo/examples/terminal_bench",
                f"{self.remote_root}/repo/src",
                f"{self.remote_root}/repo/src/codex-lm",
            ]
        )
        python = shlex.quote(f"{self.remote_root}/.venv/bin/python")
        env_assignments = self._remote_controller_env_assignments()
        command = " ".join(
            [
                f"HOME={shlex.quote(self.remote_home)}",
                *env_assignments,
                "PYTHONUNBUFFERED=1",
                f"PYTHONPATH={shlex.quote(pythonpath)}:${{PYTHONPATH:-}}",
                python,
                "-m",
                "terminal_bench_rlm.tools.remote_controller",
                shlex.quote(payload_remote_path),
            ]
        )
        return await _remote_exec(
            environment,
            command,
        )



class TerminalBenchRLMBaseAgent(_TerminalBenchRLMBaseAgentMixin):
    pass


def terminal_bench_agent_class():
    """Return a Terminal-Bench BaseAgent subclass when terminal-bench is installed."""
    base_agent = _terminal_bench_base_agent()
    if base_agent is not None:
        if issubclass(TerminalBenchRLMBaseAgent, base_agent):
            return TerminalBenchRLMBaseAgent

        class TerminalBenchRLMBaseAgentAdapter(TerminalBenchRLMBaseAgent, base_agent):  # type: ignore[misc, valid-type]
            pass

        return TerminalBenchRLMBaseAgentAdapter
    return TerminalBenchRLMBaseAgent


TerminalBenchRLMAgent = terminal_bench_agent_class()
