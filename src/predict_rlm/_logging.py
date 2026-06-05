"""Logging and human trace rendering for predict-rlm."""

from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from io import StringIO
from typing import Any

PACKAGE_LOGGER_NAME = "predict_rlm"
TRACE_LOGGER_NAME = "predict_rlm.trace"
DEBUG_HANDLER_MARKER = "_predict_rlm_debug_handler"
TRACE_HANDLER_MARKER = "_predict_rlm_trace_handler"
CODE_PREVIEW_CHARS = 4000
OUTPUT_PREVIEW_CHARS = 2000
TOOL_PREVIEW_CHARS = 200
ERROR_COLOR = "31"
ANSI_RESET = "\033[0m"

_live_tool_call_logging: ContextVar[bool] = ContextVar(
    "_live_tool_call_logging", default=False
)
_suppress_interpreter_result_logging: ContextVar[bool] = ContextVar(
    "_suppress_interpreter_result_logging", default=False
)
_trace_logger = logging.getLogger(TRACE_LOGGER_NAME)


@dataclass(frozen=True)
class _LoggerState:
    level: int
    propagate: bool
    disabled: bool


_debug_logger_state: _LoggerState | None = None
_trace_logger_state: _LoggerState | None = None


def configure_predict_rlm_logging(
    *,
    debug: bool | None = None,
    verbose: bool | None = None,
) -> None:
    """Make predict-rlm debug/trace logs visible without touching root logging."""
    if debug is not None:
        _configure_debug_logging(debug)
    if verbose is not None:
        _configure_trace_logging(verbose)


def _configure_debug_logging(enabled: bool) -> None:
    global _debug_logger_state

    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    if not enabled:
        if _debug_logger_state is not None:
            _restore_logger_state(
                package_logger,
                _debug_logger_state,
                marker=DEBUG_HANDLER_MARKER,
            )
            _debug_logger_state = None
        return

    if _debug_logger_state is None:
        _debug_logger_state = _capture_logger_state(package_logger)
    package_logger.setLevel(logging.DEBUG)
    if _has_marked_handler(package_logger, DEBUG_HANDLER_MARKER) or not (
        package_logger.hasHandlers()
    ):
        _ensure_marked_stream_handler(
            package_logger,
            DEBUG_HANDLER_MARKER,
            _PredictRLMDebugFormatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            ),
        )


def _configure_trace_logging(enabled: bool) -> None:
    global _trace_logger_state

    if not enabled:
        if _trace_logger_state is not None:
            _restore_logger_state(
                _trace_logger,
                _trace_logger_state,
                marker=TRACE_HANDLER_MARKER,
            )
            _trace_logger_state = None
        return

    if _trace_logger_state is None:
        _trace_logger_state = _capture_logger_state(_trace_logger)
    _trace_logger.setLevel(logging.INFO)
    _trace_logger.propagate = False
    if not _trace_logger.handlers or _has_marked_handler(
        _trace_logger,
        TRACE_HANDLER_MARKER,
    ):
        _ensure_marked_stream_handler(
            _trace_logger,
            TRACE_HANDLER_MARKER,
            logging.Formatter("%(message)s"),
        )


def _capture_logger_state(logger: logging.Logger) -> _LoggerState:
    return _LoggerState(
        level=logger.level,
        propagate=logger.propagate,
        disabled=logger.disabled,
    )


def _restore_logger_state(
    logger: logging.Logger,
    state: _LoggerState,
    *,
    marker: str,
) -> None:
    logger.setLevel(state.level)
    logger.propagate = state.propagate
    logger.disabled = state.disabled
    for handler in list(logger.handlers):
        if getattr(handler, marker, False):
            logger.removeHandler(handler)
            handler.close()


def _has_marked_handler(logger: logging.Logger, marker: str) -> bool:
    return any(getattr(handler, marker, False) for handler in logger.handlers)


def _ensure_marked_stream_handler(
    logger: logging.Logger,
    marker: str,
    formatter: logging.Formatter,
) -> None:
    for handler in logger.handlers:
        if getattr(handler, marker, False):
            if isinstance(handler, logging.StreamHandler):
                handler.stream = sys.stderr
            handler.setFormatter(formatter)
            return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    setattr(handler, marker, True)
    logger.addHandler(handler)


class _PredictRLMDebugFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if _debug_record_is_error(record):
            return f"\033[{ERROR_COLOR}m{message}{ANSI_RESET}"
        return message


def _debug_record_is_error(record: logging.LogRecord) -> bool:
    if record.levelno >= logging.ERROR:
        return True

    message = record.getMessage()
    if not message:
        return False

    event = message.split(maxsplit=1)[0]
    if event.endswith((".error", ".fatal", ".timeout")):
        return True
    return any(
        marker in message
        for marker in (
            " status=ERROR",
            " status=error",
            " error_type=",
        )
    )


def format_log_fields(fields: dict[str, Any]) -> str:
    clean_fields = {key: value for key, value in fields.items() if value is not None}
    if not clean_fields:
        return ""
    return " " + " ".join(
        f"{key}={_format_log_value(value)}"
        for key, value in sorted(clean_fields.items())
    )


def _format_log_value(value: Any) -> str:
    if isinstance(value, str):
        return value if value and not any(ch.isspace() for ch in value) else repr(value)
    return str(value)


def _preview(value: Any, limit: int | None) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, indent=2, default=str)
    if limit is not None:
        text = text[:limit]
    return text if text else "(no output)"


def _render_rich(renderable: Any) -> str:
    try:
        from rich.console import Console
    except Exception:
        return str(renderable)

    buffer = StringIO()
    Console(
        file=buffer,
        force_terminal=True,
        color_system="256",
        no_color=False,
        legacy_windows=False,
    ).print(renderable)
    return buffer.getvalue().rstrip("\n")


def _render_trace_header(iteration: int, max_iterations: int) -> str:
    try:
        from rich.text import Text
    except Exception:
        return f"RLM turn {iteration}/{max_iterations}"

    text = Text()
    text.append(f"RLM turn {iteration}/{max_iterations}", style="bold bright_black")
    return _render_rich(text)


def _render_reasoning(reasoning: str) -> str:
    try:
        from rich.padding import Padding
        from rich.table import Table
        from rich.text import Text
    except Exception:
        return f"  reasoning: {reasoning.strip()}"

    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(no_wrap=True)
    table.add_column(ratio=1)
    table.add_row(
        Text("reasoning:", style="dim italic"),
        Text(reasoning.strip(), style="dim italic"),
    )
    return _render_rich(Padding(table, (0, 0, 0, 2)))


def _render_trace_detail(
    label: str,
    body: Any,
    *,
    syntax: str | None = None,
    limit: int | None = None,
) -> str:
    if isinstance(body, str):
        text = body
    else:
        text = json.dumps(body, indent=2, default=str)
    if limit is not None:
        text = text[:limit]
    try:
        from rich.console import Group
        from rich.padding import Padding
        from rich.syntax import Syntax
        from rich.text import Text
    except Exception:
        content = text if text else "(empty)"
        return "\n".join([f"  {label}", *(f"    {line}" for line in content.splitlines())])

    if text:
        content: Text | Syntax = (
            Syntax(
                text,
                syntax,
                background_color="default",
                line_numbers=False,
                word_wrap=True,
            )
            if syntax is not None
            else Text(text, style="dim")
        )
    else:
        content = Text("(empty)", style="dim italic")
    return _render_rich(
        Group(
            Padding(Text(label, style="dim italic"), (0, 0, 0, 2)),
            Padding(content, (0, 0, 0, 4)),
        )
    )


def _render_runtime_event(message: str) -> str:
    try:
        from rich.text import Text
    except Exception:
        return f"  {message}"
    return _render_rich(Text.assemble(("  ", "dim"), (message, "magenta")))


def _emit_trace(message: str) -> None:
    _trace_logger.info("%s", message)


def _code_line_count(code: str) -> int:
    return len(code.splitlines()) if code else 0


@contextmanager
def live_tool_call_logging(enabled: bool):
    token = _live_tool_call_logging.set(enabled or _live_tool_call_logging.get())
    try:
        yield
    finally:
        _live_tool_call_logging.reset(token)


def live_tool_call_logging_enabled() -> bool:
    return _live_tool_call_logging.get()


@contextmanager
def suppress_interpreter_result_logging(enabled: bool):
    token = _suppress_interpreter_result_logging.set(
        enabled or _suppress_interpreter_result_logging.get()
    )
    try:
        yield
    finally:
        _suppress_interpreter_result_logging.reset(token)


def interpreter_result_logging_enabled(verbose: bool) -> bool:
    return verbose and not _suppress_interpreter_result_logging.get()


def _tool_call_payload(args: list[Any] | None, kwargs: dict[str, Any] | None) -> Any:
    args = args or []
    kwargs = kwargs or {}
    if args and kwargs:
        return {"args": args, "kwargs": kwargs}
    if args:
        return {"args": args}
    return kwargs


def _format_tool_call_preview(
    tool_name: str | None,
    *,
    args: list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
) -> str:
    preview = json.dumps(_tool_call_payload(args, kwargs), default=str)[:TOOL_PREVIEW_CHARS]
    return f"Tool: {tool_name}({preview})"


def emit_trace_iteration_start(
    *,
    iteration: int,
    max_iterations: int,
    reasoning: str,
    code: str,
) -> None:
    _emit_trace("\n" + _render_trace_header(iteration, max_iterations))
    if reasoning.strip():
        _emit_trace(_render_reasoning(reasoning))
    _emit_trace(_render_trace_detail("code:", code, syntax="python"))


def emit_trace_iteration_output(output: Any) -> None:
    emit_trace_block("output:", output, color="32", limit=None)


def emit_trace_iteration_submit(submit_payload: Any) -> None:
    emit_trace_block("output:", submit_payload, color="32", limit=None)


def emit_trace_iteration_end() -> None:
    return


def emit_trace_block(title: str, body: Any, *, color: str, limit: int | None) -> None:
    del color
    _emit_trace(_render_trace_detail(title, body, limit=limit))


def emit_trace_result(result: dict[str, Any]) -> None:
    if "final" in result:
        emit_trace_block("output:", result["final"], color="32", limit=OUTPUT_PREVIEW_CHARS)
        return
    emit_trace_block("output:", result.get("output", ""), color="32", limit=OUTPUT_PREVIEW_CHARS)


def emit_trace_error(error_type: str, message: Any) -> None:
    emit_trace_block(
        f"error ({error_type}):",
        message,
        color="31",
        limit=OUTPUT_PREVIEW_CHARS,
    )


def emit_trace_tool_call(
    tool_name: str | None,
    *,
    args: list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
) -> None:
    preview = _format_tool_call_preview(tool_name, args=args, kwargs=kwargs)
    _emit_trace(_render_runtime_event(preview))
