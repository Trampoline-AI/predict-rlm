"""Shared helpers for recoverable per-iteration execution timeouts."""

from __future__ import annotations

import math
from typing import Any

DEFAULT_RECOVERABLE_EXECUTION_TIMEOUT_GRACE_SECONDS = 30.0
ITERATION_TIMEOUT_FAILURE_CLASS = "rlm_iteration_execution_timeout"
SANDBOX_EXEC_TIMEOUT_FAILURE_CLASS = "sandbox_exec_timeout"


class RecoverableExecutionTimeout(str):
    """String observation for an interrupted iteration that can continue."""

    def __new__(
        cls,
        message: str,
        *,
        timeout_seconds: float,
        stdout: str = "",
        stderr: str = "",
        state: dict[str, Any] | None = None,
    ) -> "RecoverableExecutionTimeout":
        obj = str.__new__(cls, message)
        obj.timeout_seconds = timeout_seconds
        obj.stdout = stdout
        obj.stderr = stderr
        obj.state = dict(state or {})
        obj.state_preserved = obj.state.get("preserved")
        return obj


def validate_execution_timeout(timeout: float | None) -> float | None:
    """Validate an LM-selected timeout value."""
    if timeout is None:
        return None
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or float(timeout) <= 0
    ):
        raise ValueError("execution timeout must be a positive number of seconds")
    return float(timeout)


def resolve_execution_timeout(
    timeout: float | None,
    *,
    default_timeout: float,
) -> tuple[float, str]:
    """Resolve request timeout and classify whether host expiry is fatal."""
    execution_timeout = validate_execution_timeout(timeout)
    if execution_timeout is None:
        return default_timeout, SANDBOX_EXEC_TIMEOUT_FAILURE_CLASS
    return execution_timeout, ITERATION_TIMEOUT_FAILURE_CLASS


def recoverable_timeout_host_deadline_seconds(
    timeout_seconds: float,
    timeout_failure_class: str,
    *,
    grace_seconds: float | None = None,
) -> float:
    """Return host watchdog budget for an execution request."""
    if timeout_failure_class != ITERATION_TIMEOUT_FAILURE_CLASS:
        return timeout_seconds
    if grace_seconds is None:
        grace_seconds = DEFAULT_RECOVERABLE_EXECUTION_TIMEOUT_GRACE_SECONDS
    return timeout_seconds + grace_seconds


def format_recoverable_timeout_result(
    result: dict[str, Any],
) -> RecoverableExecutionTimeout:
    """Format a runner structured timeout into the observation shown to the LM."""
    timeout_info = result.get("timeout") or {}
    timeout_seconds = float(timeout_info.get("seconds") or 0)
    stdout = str(result.get("stdout") or "")
    stderr = str(result.get("stderr") or "")
    state = result.get("state") if isinstance(result.get("state"), dict) else {}
    message = f"[Timeout] Iteration execution timed out after {timeout_seconds:g}s"
    parts = [message]
    if state:
        if state.get("preserved") is True:
            parts.append("[state]\nPython globals preserved in the live kernel.")
        elif state.get("preserved") is False:
            reason = str(state.get("reason") or "kernel state was discarded")
            if state.get("source") == "pickle_snapshot":
                restored = ", ".join(map(str, state.get("restored_globals") or []))
                lost = ", ".join(map(str, state.get("lost_globals") or []))
                state_lines = [
                    "Full live Python state was not preserved.",
                    f"Reason: {reason}.",
                    "Restored pickleable globals"
                    + (f": {restored}." if restored else ": none."),
                    "Lost globals / imports"
                    + (f": {lost}." if lost else ": none."),
                ]
                parts.append("[state]\n" + "\n".join(state_lines))
            else:
                parts.append(f"[state]\nPython globals were lost: {reason}")
    if stdout:
        parts.append(f"[stdout]\n{stdout.rstrip()}")
    if stderr:
        parts.append(f"[stderr]\n{stderr.rstrip()}")
    return RecoverableExecutionTimeout(
        "\n\n".join(parts),
        timeout_seconds=timeout_seconds,
        stdout=stdout,
        stderr=stderr,
        state=state,
    )
