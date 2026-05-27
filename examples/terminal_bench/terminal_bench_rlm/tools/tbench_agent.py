from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
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
TerminalBenchRunnerInterpreter: Any | None = None
HarborEnvironmentInterpreter: Any | None = None


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


def _write_trace(trace: Any, logging_dir: Path | None) -> None:
    if trace is None or logging_dir is None:
        return
    logging_dir.mkdir(parents=True, exist_ok=True)
    path = logging_dir / f"predict_rlm_trace_{uuid.uuid4().hex[:8]}.json"
    if hasattr(trace, "to_exportable_json"):
        path.write_text(trace.to_exportable_json(), encoding="utf-8")
    else:
        path.write_text(str(trace), encoding="utf-8")


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
    global TerminalBenchRunnerInterpreter
    if TerminalBenchRunnerInterpreter is None:
        TerminalBenchRunnerInterpreter = getattr(
            importlib.import_module(".container_runner", __package__),
            "TerminalBenchRunnerInterpreter",
        )
    return TerminalBenchRunnerInterpreter


def _harbor_interpreter_class() -> Any:
    global HarborEnvironmentInterpreter
    if HarborEnvironmentInterpreter is None:
        HarborEnvironmentInterpreter = getattr(
            importlib.import_module(".container_runner", __package__),
            "HarborEnvironmentInterpreter",
        )
    return HarborEnvironmentInterpreter


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
        signature: str = "instruction -> answer",
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
        **predict_rlm_kwargs: Any,
    ) -> None:
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
        interpreter = _interpreter_class()(
            runtime,
            **self.interpreter_kwargs,
        )
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
            rlm = _predict_rlm_class()(self.signature, **rlm_kwargs)
            result = rlm(instruction=task_instruction)
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


class HarborPredictRLMAgent(_TerminalBenchRLMBaseAgentMixin):
    """Harbor BaseAgent-compatible adapter for Terminal-Bench 2.x tasks."""

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

    async def setup(self, environment: Any) -> None:
        started = time.monotonic()
        _write_phase_event(
            self.phase_log_path,
            task_id=self.task_id,
            event="agent_setup_start",
            phase="agent_setup",
            status="started",
        )
        _write_phase_event(
            self.phase_log_path,
            task_id=self.task_id,
            event="agent_setup_end",
            phase="agent_setup",
            status="completed",
            duration_seconds=time.monotonic() - started,
        )
        return None

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        loop = asyncio.get_running_loop()
        started = time.monotonic()
        _write_phase_event(
            self.phase_log_path,
            task_id=self.task_id,
            event="agent_run_start",
            phase="agent_eval",
            status="started",
        )
        try:
            result = await self._run_async(instruction, environment, loop)
        except BaseException:
            _write_phase_event(
                self.phase_log_path,
                task_id=self.task_id,
                event="agent_run_end",
                phase="agent_eval",
                status="failed",
                duration_seconds=time.monotonic() - started,
            )
            raise
        _write_phase_event(
            self.phase_log_path,
            task_id=self.task_id,
            event="agent_run_end",
            phase="agent_eval",
            status="completed",
            duration_seconds=time.monotonic() - started,
        )
        answer = _coerce_answer(result)
        if "metadata" in getattr(type(context), "model_fields", {}):
            metadata = dict(context.metadata or {})
            metadata["answer"] = answer
            context.metadata = metadata
        else:
            setattr(context, "answer", answer)

    async def _run_async(
        self,
        instruction: str,
        environment: Any,
        loop: asyncio.AbstractEventLoop,
    ) -> Any:
        if self.codex_lm:
            _install_codex_lm_monkeypatch(self.codex_lm_exclude)
        setup_started = time.monotonic()
        _write_phase_event(
            self.phase_log_path,
            task_id=self.task_id,
            event="sandbox_setup_start",
            phase="sandbox_setup",
            status="started",
        )
        try:
            interpreter = _harbor_interpreter_class()(environment, loop=loop, **self.interpreter_kwargs)
        except BaseException:
            _write_phase_event(
                self.phase_log_path,
                task_id=self.task_id,
                event="sandbox_setup_end",
                phase="sandbox_setup",
                status="failed",
                duration_seconds=time.monotonic() - setup_started,
            )
            raise
        _write_phase_event(
            self.phase_log_path,
            task_id=self.task_id,
            event="sandbox_setup_end",
            phase="sandbox_setup",
            status="completed",
            duration_seconds=time.monotonic() - setup_started,
        )
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
            rlm = _predict_rlm_class()(self.signature, **rlm_kwargs)
            result = rlm.acall(instruction=instruction)
            if inspect.isawaitable(result):
                result = await result
            _write_trace(getattr(result, "trace", None), self.logs_dir)
            return result
        except BaseException as exc:
            _write_trace(getattr(exc, "trace", None), self.logs_dir)
            raise
        finally:
            await asyncio.to_thread(interpreter.shutdown)



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
