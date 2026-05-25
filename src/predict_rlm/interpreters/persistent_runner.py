"""Shared persistent JSON-RPC supervisor client helpers."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Protocol

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput

from predict_rlm.execution_timeout import format_recoverable_timeout_result
from predict_rlm.interpreter import SandboxFatalError

from .base import STALE_RESPONSE_DISCARD_LIMIT


class PersistentSupervisorProcess(Protocol):
    """Process-like object used by persistent JSON-RPC supervisor clients."""

    stdin: Any
    stdout: Any
    stderr: Any

    def poll(self) -> int | None: ...


PersistentRunnerProcess = PersistentSupervisorProcess


class PersistentJsonRpcRunnerClient(ABC):
    """Small shared client for persistent Python JSON-RPC supervisors.

    Backend subclasses own process creation, tool callbacks, file semantics,
    timeout policy, and user-facing diagnostics. This class owns request IDs,
    JSON-RPC framing, response resync, and structured-response lifecycle state.
    """

    def __init__(
        self,
        *,
        supervisor_name: str | None = None,
        runner_name: str | None = None,
        stale_response_discard_limit: int = STALE_RESPONSE_DISCARD_LIMIT,
    ) -> None:
        self._persistent_supervisor_name = supervisor_name or runner_name
        if self._persistent_supervisor_name is None:
            raise TypeError("supervisor_name is required")
        self._stale_response_discard_limit = stale_response_discard_limit
        self._request_id = 0
        self._supervisor_exit_recovery_context: dict[str, Any] | None = None

    def _send_json_rpc_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        recovered = self._recover_dead_supervisor_after_structured_execute(method)
        if recovered is not None:
            return recovered
        self._ensure_process_for_request(method)
        return self._send_json_rpc_request_without_ensure(
            method,
            params,
            timeout=timeout,
        )

    def _send_json_rpc_request_without_ensure(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        params = params or {}
        process = self._require_supervisor_process()
        request_timeout = self._request_timeout_seconds(method, params, timeout)
        request_id = self._next_request_id()
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": request_id,
        }
        try:
            process.stdin.write(self._serialize_supervisor_message(payload) + "\n")
            process.stdin.flush()
        except BrokenPipeError as exc:
            self._handle_supervisor_send_error(method, request_id, exc)

        deadline = time.monotonic() + request_timeout
        request_start = time.perf_counter()
        stale_discards = 0
        stdout_tail: list[str] = []
        self._on_supervisor_request_start(
            method,
            params,
            request_id=request_id,
            request_timeout=request_timeout,
        )
        while True:
            self._drain_completed_supervisor_work()
            if process.poll() is not None:
                self._handle_supervisor_exit_during_request(
                    method,
                    request_id=request_id,
                    request_start=request_start,
                    process=process,
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._handle_supervisor_request_timeout(
                    method,
                    params,
                    process,
                    request_id=request_id,
                    request_timeout=request_timeout,
                    request_start=request_start,
                    stdout_tail="".join(stdout_tail)[-4000:],
                )

            line = self._read_supervisor_stdout_line(
                process,
                deadline=deadline,
                timeout=remaining,
            )
            if line is None:
                return self._handle_supervisor_request_timeout(
                    method,
                    params,
                    process,
                    request_id=request_id,
                    request_timeout=request_timeout,
                    request_start=request_start,
                    stdout_tail="".join(stdout_tail)[-4000:],
                )
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                stdout_tail.append(line)
                continue
            if not isinstance(message, dict):
                stdout_tail.append(line)
                continue
            if self._handle_supervisor_control_message(message, deadline=deadline):
                continue
            if message.get("id") == request_id:
                self._record_supervisor_response(
                    method,
                    params,
                    request_id=request_id,
                    request_timeout=request_timeout,
                    response=message,
                )
                self._on_supervisor_request_response(
                    method,
                    request_id=request_id,
                    request_start=request_start,
                    response=message,
                )
                return message
            stale_discards += 1
            if stale_discards > self._stale_response_discard_limit:
                self._handle_stale_response_limit(
                    method,
                    request_id=request_id,
                    request_start=request_start,
                )

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _require_supervisor_process(self) -> PersistentSupervisorProcess:
        process = self._get_supervisor_process()
        if process is None:
            raise SandboxFatalError(f"{self._persistent_supervisor_name} is not started")
        if process.stdin is None or process.stdout is None:
            raise SandboxFatalError(f"{self._persistent_supervisor_name} stdio is unavailable")
        return process

    def _record_supervisor_response(
        self,
        method: str,
        params: dict[str, Any],
        *,
        request_id: int,
        request_timeout: float,
        response: dict[str, Any],
    ) -> None:
        if method != "execute":
            return
        response_kind = self._execute_response_kind(response)
        if response_kind not in {"structured_error", "structured_timeout"}:
            self._supervisor_exit_recovery_context = None
            return
        self._supervisor_exit_recovery_context = {
            "request_id": request_id,
            "method": method,
            "response_kind": response_kind,
            "execution_timeout_seconds": params.get("execution_timeout_seconds"),
            "request_timeout": request_timeout,
        }

    def _execute_response_kind(self, response: dict[str, Any]) -> str:
        if "error" in response:
            return "structured_error"
        result = response.get("result") or {}
        if isinstance(result, dict) and "timeout" in result:
            return "structured_timeout"
        return "ok"

    def _recover_dead_supervisor_after_structured_execute(
        self,
        method: str,
    ) -> dict[str, Any] | None:
        if method != "execute" or self._supervisor_exit_recovery_context is None:
            return None
        process = self._get_supervisor_process()
        if process is None:
            return None
        returncode = process.poll()
        if returncode is None:
            return None

        stderr = self._read_stderr_for_process(process)
        context = self._supervisor_exit_recovery_context
        self._supervisor_exit_recovery_context = None
        self._discard_supervisor_process()
        restart_error: BaseException | None = None
        try:
            self._ensure_process_for_request(method)
        except BaseException as exc:
            restart_error = exc
        if restart_error is not None:
            raise SandboxFatalError(
                f"{self._persistent_supervisor_name} exited after the previous execute "
                "response and could not be restarted: "
                f"{restart_error}. {self._format_supervisor_exit_evidence(returncode, context)}"
            ) from restart_error

        return {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "result": {
                "output": self._format_supervisor_restart_diagnostic(
                    returncode,
                    context,
                    stderr=stderr,
                )
            },
        }

    def _format_supervisor_exit_evidence(
        self,
        returncode: int | None,
        context: dict[str, Any],
    ) -> str:
        parts = [
            f"supervisor_returncode={returncode}",
            f"previous_request_id={context.get('request_id')}",
            f"previous_method={context.get('method')}",
            f"previous_response={context.get('response_kind')}",
        ]
        execution_timeout = context.get("execution_timeout_seconds")
        if execution_timeout is not None:
            parts.append(
                "previous_execution_timeout_seconds="
                f"{self._format_number(execution_timeout)}"
            )
        request_timeout = context.get("request_timeout")
        if request_timeout is not None:
            parts.append(
                f"previous_host_timeout_seconds={self._format_number(request_timeout)}"
            )
        return " ".join(parts)

    def _format_number(self, value: Any) -> str:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f"{float(value):g}"
        return str(value)

    def _unwrap_execute_response(self, response: dict[str, Any]) -> Any:
        if "error" in response:
            self._raise_execute_error(response)
        result = response.get("result") or {}
        if isinstance(result, dict) and "timeout" in result:
            return format_recoverable_timeout_result(result)
        if "final" in result:
            return FinalOutput(result["final"])
        return result.get("output")

    def _serialize_supervisor_message(self, message: dict[str, Any]) -> str:
        return json.dumps(message)

    def _drain_completed_supervisor_work(self) -> None:
        return None

    def _handle_supervisor_control_message(
        self,
        message: dict[str, Any],
        *,
        deadline: float,
    ) -> bool:
        return False

    def _on_supervisor_request_start(
        self,
        method: str,
        params: dict[str, Any],
        *,
        request_id: int,
        request_timeout: float,
    ) -> None:
        return None

    def _on_supervisor_request_response(
        self,
        method: str,
        *,
        request_id: int,
        request_start: float,
        response: dict[str, Any],
    ) -> None:
        return None

    def _handle_supervisor_send_error(
        self,
        method: str,
        request_id: int,
        exc: BrokenPipeError,
    ) -> None:
        raise SandboxFatalError(f"{self._persistent_supervisor_name} pipe broke") from exc

    def _handle_supervisor_exit_during_request(
        self,
        method: str,
        *,
        request_id: int,
        request_start: float,
        process: PersistentSupervisorProcess,
    ) -> None:
        raise SandboxFatalError(
            f"{self._persistent_supervisor_name} exited unexpectedly: "
            f"{self._read_stderr_for_process(process)}"
        )

    def _handle_stale_response_limit(
        self,
        method: str,
        *,
        request_id: int,
        request_start: float,
    ) -> None:
        raise CodeInterpreterError(
            "Too many stale top-level responses while resyncing "
            f"{self._persistent_supervisor_name} request id={request_id}"
        )

    @abstractmethod
    def _get_supervisor_process(self) -> PersistentSupervisorProcess | None:
        raise NotImplementedError

    @abstractmethod
    def _ensure_process_for_request(self, method: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def _request_timeout_seconds(
        self,
        method: str,
        params: dict[str, Any],
        timeout: float | None,
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def _read_supervisor_stdout_line(
        self,
        process: PersistentSupervisorProcess,
        *,
        deadline: float,
        timeout: float,
    ) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def _handle_supervisor_request_timeout(
        self,
        method: str,
        params: dict[str, Any],
        process: PersistentSupervisorProcess,
        *,
        request_id: int,
        request_timeout: float,
        request_start: float,
        stdout_tail: str,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _discard_supervisor_process(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _read_stderr_for_process(self, process: PersistentSupervisorProcess) -> str:
        raise NotImplementedError

    @abstractmethod
    def _format_supervisor_restart_diagnostic(
        self,
        returncode: int | None,
        context: dict[str, Any],
        *,
        stderr: str,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def _raise_execute_error(self, response: dict[str, Any]) -> None:
        raise NotImplementedError
