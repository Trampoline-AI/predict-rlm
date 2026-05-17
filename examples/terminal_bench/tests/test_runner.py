from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from terminal_bench_rlm.tools.runner import runner_script_path  # noqa: E402


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
        self.write({"id": self._request_id, "method": method, "params": params or {}})
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

    assert first == {"id": 1, "ok": True, "result": {"output": "ready\n"}}
    assert second == {"id": 2, "ok": True, "result": {"output": "42\n"}}
    assert reset == {"id": 3, "ok": True, "result": {}}
    assert after == {"id": 4, "ok": True, "result": {"output": "False\n"}}


def test_host_tool_call_round_trip(runner: LocalRunner) -> None:
    registered = runner.request("register_tools", {"tools": ["predict"]})
    assert registered["ok"] is True

    runner.write(
        {
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
            "id": tool_call["id"],
            "ok": True,
            "result": {"type": "json", "value": {"answer": "4"}},
        }
    )
    response = runner.read()
    assert response == {"id": 2, "ok": True, "result": {"output": "4\n"}}


def test_host_tool_errors_propagate_to_execute_error(runner: LocalRunner) -> None:
    runner.request("register_tools", {"tools": ["predict"]})

    runner.write(
        {
            "id": 2,
            "method": "execute",
            "params": {"code": "await predict('question -> answer', question='bad')"},
        }
    )
    tool_call = runner.read()
    runner.write(
        {
            "id": tool_call["id"],
            "ok": False,
            "error": {"type": "RuntimeError", "message": "predict failed"},
        }
    )
    response = runner.read()

    assert response["id"] == 2
    assert response["ok"] is False
    assert response["error"]["type"] == "RuntimeError"
    assert "predict failed" in response["error"]["message"]
