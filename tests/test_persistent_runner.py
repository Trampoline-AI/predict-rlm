from __future__ import annotations

import json
from typing import Any

import pytest
from dspy.primitives.code_interpreter import CodeInterpreterError

from predict_rlm.interpreters.persistent_runner import (
    PersistentJsonRpcRunnerClient,
    PersistentSupervisorProcess,
)


class FakePipe:
    def __init__(self, lines: list[str] | None = None) -> None:
        self.lines = list(lines or [])
        self.writes: list[str] = []

    def write(self, data: str) -> None:
        self.writes.append(data)

    def flush(self) -> None:
        return None

    def readline(self) -> str:
        if not self.lines:
            return ""
        return self.lines.pop(0)

    def read(self) -> str:
        return "".join(self.lines)


class FakeProcess:
    def __init__(self, stdout_lines: list[dict[str, Any]] | None = None) -> None:
        self.stdin = FakePipe()
        self.stdout = FakePipe(
            [json.dumps(line) + "\n" for line in (stdout_lines or [])]
        )
        self.stderr = FakePipe()
        self.returncode: int | None = None
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class FakeClient(PersistentJsonRpcRunnerClient):
    def __init__(self, processes: list[FakeProcess]) -> None:
        super().__init__(supervisor_name="fake supervisor", stale_response_discard_limit=2)
        self.processes = list(processes)
        self.process: FakeProcess | None = None
        self.started = 0

    def execute(self, code: str, *, timeout: float | None = None) -> Any:
        response = self._send_json_rpc_request(
            "execute",
            {"code": code, "execution_timeout_seconds": timeout} if timeout else {"code": code},
            timeout=timeout,
        )
        return self._unwrap_execute_response(response)

    def _get_supervisor_process(self) -> FakeProcess | None:
        return self.process

    def _ensure_process_for_request(self, method: str) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.process = self.processes.pop(0)
        self.started += 1

    def _request_timeout_seconds(
        self,
        method: str,
        params: dict[str, Any],
        timeout: float | None,
    ) -> float:
        return timeout or 1.0

    def _read_supervisor_stdout_line(
        self,
        process: PersistentSupervisorProcess,
        *,
        deadline: float,
        timeout: float,
    ) -> str | None:
        line = process.stdout.readline()
        return line or None

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
        process.kill()
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "timeout": {"seconds": params["execution_timeout_seconds"]},
                "stdout": stdout_tail,
                "stderr": "fake supervisor restarted",
            },
        }

    def _discard_supervisor_process(self) -> None:
        self.process = None

    def _read_stderr_for_process(self, process: PersistentSupervisorProcess) -> str:
        return process.stderr.read()

    def _format_supervisor_restart_diagnostic(
        self,
        returncode: int | None,
        context: dict[str, Any],
        *,
        stderr: str,
    ) -> str:
        return (
            "fake restart diagnostic\n"
            f"{self._format_supervisor_exit_evidence(returncode, context)}"
        )

    def _raise_execute_error(self, response: dict[str, Any]) -> None:
        error = response.get("error") or {}
        raise CodeInterpreterError(str(error.get("message") or "runner error"))


def test_persistent_client_discards_stale_response_then_returns_fresh() -> None:
    process = FakeProcess(
        [
            {"jsonrpc": "2.0", "id": 99, "result": {"output": "stale"}},
            {"jsonrpc": "2.0", "id": 1, "result": {"output": "fresh"}},
        ]
    )
    client = FakeClient([process])

    assert client.execute("print('fresh')") == "fresh"


def test_persistent_client_discards_stale_error_then_returns_fresh() -> None:
    process = FakeProcess(
        [
            {
                "jsonrpc": "2.0",
                "id": 99,
                "error": {"code": -32000, "message": "late failure"},
            },
            {"jsonrpc": "2.0", "id": 1, "result": {"output": "fresh"}},
        ]
    )
    client = FakeClient([process])

    assert client.execute("print('fresh')") == "fresh"


def test_persistent_client_exhausted_stale_resync_raises_cleanly() -> None:
    process = FakeProcess(
        [
            {"jsonrpc": "2.0", "id": 99, "result": {"output": "stale"}},
            {"jsonrpc": "2.0", "id": 98, "result": {"output": "stale"}},
            {"jsonrpc": "2.0", "id": 97, "result": {"output": "stale"}},
        ]
    )
    client = FakeClient([process])

    with pytest.raises(CodeInterpreterError, match="stale.*resyncing"):
        client.execute("print('fresh')")


def test_persistent_client_recovers_dead_runner_after_structured_timeout() -> None:
    first = FakeProcess(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "timeout": {"seconds": 0.2},
                    "stdout": "before\n",
                    "stderr": "timed out\n",
                },
            }
        ]
    )
    restarted = FakeProcess([])
    client = FakeClient([first, restarted])

    timeout_result = client.execute("slow()", timeout=0.2)
    first.returncode = 137
    restart_result = client.execute("next()", timeout=0.2)

    assert timeout_result.timeout_seconds == 0.2
    assert client.started == 2
    assert "fake restart diagnostic" in restart_result
    assert "supervisor_returncode=137" in restart_result
    assert "previous_response=structured_timeout" in restart_result
    assert restarted.stdin.writes == []


def test_persistent_client_host_timeout_unwraps_recoverable_timeout() -> None:
    process = FakeProcess([])
    client = FakeClient([process])

    result = client.execute("silent()", timeout=0.2)

    assert process.killed is True
    assert result.timeout_seconds == 0.2
    assert "fake supervisor restarted" in result.stderr
