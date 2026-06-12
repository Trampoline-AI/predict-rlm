"""Runtime-hook instrumentation tests against the supervisor payload."""

from __future__ import annotations

import json
import select
import subprocess
import sys
from pathlib import Path

import pytest

PAYLOAD_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "predict_rlm"
    / "backends"
    / "supervisor"
    / "_payload.py"
)


class LocalRunner:
    def __init__(self, tmp_path: Path) -> None:
        env_root = tmp_path / "runner-root"
        env_root.mkdir()
        self.proc = subprocess.Popen(
            [sys.executable, "-u", str(PAYLOAD_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={"PREDICT_RLM_SBX_ROOT": str(env_root), "PATH": "/usr/bin:/bin"},
        )
        self._request_id = 0

    def send(self, method: str, params: dict | None = None) -> int:
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self._request_id,
        }
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        return self._request_id

    def read_message(self, timeout: float = 10) -> dict:
        assert self.proc.stdout is not None
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        assert ready, "timed out waiting for runner message"
        line = self.proc.stdout.readline()
        assert line, "runner stdout closed"
        return json.loads(line)

    def request(self, method: str, params: dict | None = None) -> dict:
        self.send(method, params)
        return self.read_message()

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.request("shutdown")
            finally:
                self.proc.wait(timeout=5)


@pytest.fixture
def runner(tmp_path):
    proc = LocalRunner(tmp_path)
    try:
        yield proc
    finally:
        proc.close()


def test_runtime_hooks_are_opt_in(runner: LocalRunner, tmp_path: Path):
    path = tmp_path / "no-hook.txt"
    result = runner.request("execute", {"code": f"open({str(path)!r}, 'w').close()"})
    assert result["result"]["output"] == ""


def test_runtime_hooks_emit_function_events(runner: LocalRunner, tmp_path: Path):
    path = tmp_path / "hooked.txt"
    registered = runner.request(
        "register_runtime_hooks",
        {
            "hooks": [
                {"target": "pathlib.Path.write_text", "phases": ["before", "after"]},
                {"target": "subprocess.run", "phases": ["before", "after"]},
            ]
        },
    )
    assert registered["result"] == {}

    execute = runner.request(
        "execute",
        {
            "code": (
                "from pathlib import Path\n"
                "import subprocess, sys\n"
                f"Path({str(path)!r}).write_text('hello')\n"
                "subprocess.run([sys.executable, '-c', 'print(123)'], "
                "capture_output=True, text=True)\n"
            )
        },
    )
    events = []
    while "method" in execute:
        events.append(execute["params"])
        execute = json.loads(runner.proc.stdout.readline())

    assert execute["result"]["output"] == ""
    assert [event["target"] for event in events] == [
        "pathlib.Path.write_text",
        "pathlib.Path.write_text",
        "subprocess.run",
        "subprocess.run",
    ]
    assert events[0]["phase"] == "before"
    assert events[1]["phase"] == "after"


def test_runtime_hooks_do_not_emit_internal_capture_file_events(runner: LocalRunner):
    runner.request(
        "register_runtime_hooks",
        {"hooks": [{"target": "os.open", "phases": ["before"]}]},
    )
    execute = runner.request("execute", {"code": "print('user output')"})
    assert "method" not in execute
    assert execute["result"]["output"] == "user output\n"


def test_runtime_hooks_emit_error_events(runner: LocalRunner):
    runner.request(
        "register_runtime_hooks",
        {"hooks": [{"target": "builtins.open", "phases": ["error"]}]},
    )
    execute = runner.request("execute", {"code": "open('/definitely/missing')"})
    event = execute
    assert event["method"] == "runtime_hook_event"
    assert event["params"]["target"] == "builtins.open"
    assert event["params"]["phase"] == "error"


def test_runtime_hooks_reset_on_reregister(runner: LocalRunner, tmp_path: Path):
    runner.request(
        "register_runtime_hooks",
        {"hooks": [{"target": "pathlib.Path.write_text", "phases": ["before"]}]},
    )
    # Re-register with empty set should clear hooks.
    runner.request("register_runtime_hooks", {"hooks": []})
    path = tmp_path / "after-reset.txt"
    execute = runner.request(
        "execute",
        {"code": f"from pathlib import Path\nPath({str(path)!r}).write_text('x')"},
    )
    assert "method" not in execute
    assert execute["result"]["output"] == ""
