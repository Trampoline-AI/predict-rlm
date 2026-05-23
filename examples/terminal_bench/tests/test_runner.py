from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from terminal_bench_rlm.tools.runner import runner_script_path, runner_source  # noqa: E402


class LocalRunner:
    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-u", str(runner_script_path())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._request_id = 0

    def request(self, method: str, params: dict | None = None) -> dict:
        self._request_id += 1
        self.write(
            {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params or {},
            }
        )
        return self.read()

    def write(self, message: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def read(self) -> dict:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        assert line, self.proc.stderr.read() if self.proc.stderr else ""
        return json.loads(line)

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.request("shutdown")
            finally:
                self.proc.wait(timeout=5)


@pytest.fixture
def runner() -> LocalRunner:
    proc = LocalRunner()
    try:
        yield proc
    finally:
        proc.close()


def test_execute_persists_namespace_and_reset_clears_it(runner: LocalRunner) -> None:
    first = runner.request("execute", {"code": "x = 40\nprint('ready')"})
    second = runner.request("execute", {"code": "x += 2\nprint(x)"})
    reset = runner.request("reset")
    after = runner.request("execute", {"code": "print('x' in globals())"})

    assert first == {"jsonrpc": "2.0", "id": 1, "result": {"output": "ready\n"}}
    assert second == {"jsonrpc": "2.0", "id": 2, "result": {"output": "42\n"}}
    assert reset == {"jsonrpc": "2.0", "id": 3, "result": {}}
    assert after == {"jsonrpc": "2.0", "id": 4, "result": {"output": "False\n"}}


def test_host_tool_call_round_trip(runner: LocalRunner) -> None:
    registered = runner.request("register_tools", {"tools": ["predict"]})
    assert "error" not in registered

    runner.write(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "execute",
            "params": {
                "code": (
                    "result = await predict('question -> answer', question='2+2?')\n"
                    "print(result['answer'])"
                )
            },
        }
    )
    tool_call = runner.read()
    assert tool_call["method"] == "tool_call"
    assert tool_call["params"] == {
        "name": "predict",
        "args": ["question -> answer"],
        "kwargs": {"question": "2+2?"},
    }

    runner.write(
        {
            "jsonrpc": "2.0",
            "id": tool_call["id"],
            "result": {"type": "json", "value": "{\"answer\": \"4\"}"},
        }
    )
    response = runner.read()
    assert response == {"jsonrpc": "2.0", "id": 2, "result": {"output": "4\n"}}


def test_host_tool_errors_propagate_to_execute_error(runner: LocalRunner) -> None:
    runner.request("register_tools", {"tools": ["predict"]})

    runner.write(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "execute",
            "params": {"code": "await predict('question -> answer', question='bad')"},
        }
    )
    tool_call = runner.read()
    runner.write(
        {
            "jsonrpc": "2.0",
            "id": tool_call["id"],
            "error": {"code": -32000, "message": "predict failed"},
        }
    )
    response = runner.read()

    assert response["id"] == 2
    assert response["error"]["data"]["type"] == "RuntimeError"
    assert "predict failed" in response["error"]["message"]


def test_terminal_bench_runner_payload_is_shared_with_sbx() -> None:
    shared_runner = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "predict_rlm"
        / "sandbox"
        / "python_runner.py"
    )

    assert runner_script_path() == shared_runner
    assert runner_source() == shared_runner.read_text(encoding="utf-8")


def test_runner_timeout_recovers_from_blocking_native_call() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-u", str(runner_script_path())],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "execute",
        "params": {
            "execution_timeout_seconds": 0.1,
            "code": (
                "import hashlib, sys\n"
                "print('stdout before native')\n"
                "print('stderr before native', file=sys.stderr)\n"
                "hashlib.pbkdf2_hmac('sha256', b'x', b'y', 10_000_000)\n"
                "print('unreachable')\n"
            ),
        },
    }

    try:
        assert proc.stdin is not None
        assert proc.stdout is not None
        start = time.monotonic()
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()

        ready, _, _ = select.select([proc.stdout], [], [], 1.0)
        assert ready, "runner did not return a recoverable timeout response"
        response = json.loads(proc.stdout.readline())
        followup = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "execute",
            "params": {"code": "print('after timeout')"},
        }
        proc.stdin.write(json.dumps(followup) + "\n")
        proc.stdin.flush()
        followup_ready, _, _ = select.select([proc.stdout], [], [], 1.0)
        assert followup_ready, "runner did not survive the timed-out native call"
        after = json.loads(proc.stdout.readline())
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=2)

    assert time.monotonic() - start < 1.5
    assert response == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "timeout": {"seconds": 0.1},
            "stdout": "stdout before native\n",
            "stderr": "stderr before native\n",
        },
    }
    assert after == {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {"output": "after timeout\n"},
    }


def _process_exited(pid: int, *, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.02)
    return False


def test_runner_timeout_kills_generated_code_child_process(tmp_path: Path) -> None:
    pid_path = tmp_path / "child.pid"
    proc = subprocess.Popen(
        [sys.executable, "-u", str(runner_script_path())],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    child_pid: int | None = None
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "execute",
        "params": {
            "execution_timeout_seconds": 0.1,
            "code": (
                "import pathlib, signal, subprocess, sys, time\n"
                f"pid_path = pathlib.Path({str(pid_path)!r})\n"
                "child = subprocess.Popen([\n"
                "    sys.executable,\n"
                "    '-c',\n"
                "    'import signal, time; '\n"
                "    'signal.signal(signal.SIGINT, signal.SIG_IGN); '\n"
                "    'signal.signal(signal.SIGTERM, signal.SIG_IGN); '\n"
                "    'time.sleep(30)',\n"
                "])\n"
                "pid_path.write_text(str(child.pid), encoding='utf-8')\n"
                "print(f'child={child.pid}')\n"
                "time.sleep(30)\n"
            ),
        },
    }

    try:
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()

        ready, _, _ = select.select([proc.stdout], [], [], 2.0)
        assert ready, "runner did not return a recoverable timeout response"
        response = json.loads(proc.stdout.readline())
        child_pid = int(pid_path.read_text(encoding="utf-8"))
        assert _process_exited(child_pid, timeout=1.0)
    finally:
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=2)

    assert response == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "timeout": {"seconds": 0.1},
            "stdout": f"child={child_pid}\n",
            "stderr": "",
        },
    }
