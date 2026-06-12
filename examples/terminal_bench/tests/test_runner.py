from __future__ import annotations

import base64
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


def _drain_available_pipe_text(pipe) -> str:
    assert pipe is not None
    chunks: list[str] = []
    while True:
        ready, _, _ = select.select([pipe], [], [], 0)
        if not ready:
            return "".join(chunks)
        chunk = os.read(pipe.fileno(), 65536)
        if not chunk:
            return "".join(chunks)
        chunks.append(chunk.decode("utf-8", errors="replace"))


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


def test_predict_result_supports_attribute_and_subscript_access(runner: LocalRunner) -> None:
    registered = runner.request("register_tools", {"tools": ["predict"]})
    assert "error" not in registered

    runner.write(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "execute",
            "params": {
                "code": (
                    "result = await predict('question -> answer: str, items: list[int]', "
                    "question='2+2?')\n"
                    "print(result.answer)\n"
                    "print(result['answer'])\n"
                    "print(result.items)"
                )
            },
        }
    )
    tool_call = runner.read()
    assert tool_call["method"] == "tool_call"

    runner.write(
        {
            "jsonrpc": "2.0",
            "id": tool_call["id"],
            "result": {"type": "json", "value": "{\"answer\": \"4\", \"items\": [1, 2]}"},
        }
    )
    response = runner.read()

    assert response == {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {"output": "4\n4\n[1, 2]\n"},
    }


def test_predict_image_data_url_round_trips_to_host_tool(runner: LocalRunner) -> None:
    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    registered = runner.request("register_tools", {"tools": ["predict"]})
    assert "error" not in registered

    runner.write(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "execute",
            "params": {
                "code": (
                    "import base64\n"
                    f"open('/sandbox/image.png', 'wb').write({png_bytes!r})\n"
                    "image_bytes = open('/sandbox/image.png', 'rb').read()\n"
                    "data_url = 'data:image/png;base64,' + base64.b64encode(image_bytes).decode()\n"
                    "result = await predict(\n"
                    "    'image: dspy.Image, question: str -> visible_text: str',\n"
                    "    image=data_url,\n"
                    "    question='What text is visible?',\n"
                    ")\n"
                    "print(result.visible_text)"
                )
            },
        }
    )
    tool_call = runner.read()

    assert tool_call["method"] == "tool_call"
    assert tool_call["params"]["name"] == "predict"
    assert tool_call["params"]["args"] == [
        "image: dspy.Image, question: str -> visible_text: str"
    ]
    assert tool_call["params"]["kwargs"]["question"] == "What text is visible?"
    image = tool_call["params"]["kwargs"]["image"]
    assert image.startswith("data:image/png;base64,")
    assert base64.b64decode(image.removeprefix("data:image/png;base64,")) == png_bytes

    runner.write(
        {
            "jsonrpc": "2.0",
            "id": tool_call["id"],
            "result": {"type": "json", "value": '{"visible_text": "hello"}'},
        }
    )
    response = runner.read()

    assert response == {"jsonrpc": "2.0", "id": 2, "result": {"output": "hello\n"}}


def test_pathlib_path_remains_a_type(runner: LocalRunner) -> None:
    result = runner.request(
        "execute",
        {
            "code": (
                "import pathlib\n"
                "print(isinstance(pathlib.Path, type))\n"
                "print(isinstance('/tmp/example', pathlib.Path))"
            )
        },
    )

    assert result == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"output": "True\nFalse\n"},
    }


def test_windows311_visual_predict_path_handles_pillow_style_path_checks(
    runner: LocalRunner,
) -> None:
    registered = runner.request("register_tools", {"tools": ["predict"]})
    assert "error" not in registered

    runner.write(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "execute",
            "params": {
                "code": (
                    "import base64, pathlib\n"
                    "ppm = b'P6\\n1 1\\n255\\n' + bytes([255, 255, 255])\n"
                    "pathlib.Path('/tmp/win311-screen.ppm').write_bytes(ppm)\n"
                    "\n"
                    "class PillowStyleImage:\n"
                    "    def __init__(self, data):\n"
                    "        self.data = data\n"
                    "        self.size = (1, 1)\n"
                    "\n"
                    "    def save(self, path):\n"
                    "        pathlib.Path(path).write_bytes(self.data)\n"
                    "\n"
                    "def image_open_like_pillow(fp):\n"
                    "    isinstance(fp, pathlib.Path)\n"
                    "    return PillowStyleImage(pathlib.Path(fp).read_bytes())\n"
                    "\n"
                    "im = image_open_like_pillow('/tmp/win311-screen.ppm')\n"
                    "im.save('/tmp/win311-screen.png')\n"
                    "data_url = 'data:image/png;base64,' + base64.b64encode(\n"
                    "    pathlib.Path('/tmp/win311-screen.png').read_bytes()\n"
                    ").decode()\n"
                    "vision = await predict(\n"
                    "    'image: dspy.Image, question: str -> visible_text: str, answer: str',\n"
                    "    instructions='Inspect this VM screenshot.',\n"
                    "    image=data_url,\n"
                    "    question='Does this show the Windows 3.11 desktop?',\n"
                    ")\n"
                    "print(vision.visible_text)\n"
                    "print(vision.answer)"
                )
            },
        }
    )
    tool_call = runner.read()

    assert tool_call["method"] == "tool_call"
    assert tool_call["params"]["name"] == "predict"
    assert tool_call["params"]["args"] == [
        "image: dspy.Image, question: str -> visible_text: str, answer: str"
    ]
    assert tool_call["params"]["kwargs"]["instructions"] == "Inspect this VM screenshot."
    assert tool_call["params"]["kwargs"]["question"] == "Does this show the Windows 3.11 desktop?"
    image = tool_call["params"]["kwargs"]["image"]
    assert image.startswith("data:image/png;base64,")
    assert base64.b64decode(image.removeprefix("data:image/png;base64,")) == (
        b"P6\n1 1\n255\n" + bytes([255, 255, 255])
    )

    runner.write(
        {
            "jsonrpc": "2.0",
            "id": tool_call["id"],
            "result": {
                "type": "json",
                "value": '{"visible_text": "desktop", "answer": "yes"}',
            },
        }
    )
    response = runner.read()

    assert response == {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {"output": "desktop\nyes\n"},
    }


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


def test_terminal_bench_payload_is_shared_with_sbx() -> None:
    shared_payload = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "predict_rlm"
        / "backends"
        / "supervisor"
        / "_payload.py"
    )

    assert runner_script_path() == shared_payload
    assert runner_source() == shared_payload.read_text(encoding="utf-8")


def test_runner_attributes_child_process_output_to_execute_result(
    runner: LocalRunner,
) -> None:
    response = runner.request(
        "execute",
        {
            "code": (
                "import subprocess, sys\n"
                "subprocess.run([\n"
                "    sys.executable,\n"
                "    '-c',\n"
                "    \"import sys; print('child stdout'); "
                "print('child stderr', file=sys.stderr)\",\n"
                "])\n"
            ),
            "execution_timeout_seconds": 2,
        },
    )
    followup = runner.request("execute", {"code": "print('runner still usable')"})
    leaked_stderr = _drain_available_pipe_text(runner.proc.stderr)

    assert response == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"output": "child stdout\nchild stderr\n"},
    }
    assert followup == {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {"output": "runner still usable\n"},
    }
    assert leaked_stderr == ""


def test_runner_timeout_preserves_child_process_output(
    runner: LocalRunner,
) -> None:
    response = runner.request(
        "execute",
        {
            "code": (
                "import subprocess, sys\n"
                "subprocess.run([\n"
                "    sys.executable,\n"
                "    '-c',\n"
                "    \"import sys, time; print('child before timeout'); "
                "print('child err before timeout', file=sys.stderr); "
                "sys.stdout.flush(); sys.stderr.flush(); time.sleep(30)\",\n"
                "])\n"
            ),
            "execution_timeout_seconds": 0.2,
        },
    )
    followup = runner.request("execute", {"code": "print('runner survived timeout')"})
    leaked_stderr = _drain_available_pipe_text(runner.proc.stderr)

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert response["result"]["timeout"] == {"seconds": 0.2}
    assert response["result"]["stdout"] == "child before timeout\n"
    assert response["result"]["stderr"].startswith("child err before timeout\n")
    assert followup == {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {"output": "runner survived timeout\n"},
    }
    assert leaked_stderr == ""


def test_runner_unbounded_runner_exit_returns_error_and_supervisor_survives(
    runner: LocalRunner,
) -> None:
    response = runner.request("execute", {"code": "import os\nos._exit(7)"})
    followup = runner.request(
        "execute", {"code": "print('supervisor survived runner exit')"}
    )

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert response["error"]["data"]["type"] == "RuntimeError"
    assert "execution runner exited without a result" in response["error"]["message"]
    assert followup == {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {"output": "supervisor survived runner exit\n"},
    }


def test_runner_survives_subprocess_timeout_error(runner: LocalRunner) -> None:
    response = runner.request(
        "execute",
        {
            "code": (
                "import subprocess, sys\n"
                "subprocess.run(\n"
                "    [sys.executable, '-c', 'import time; time.sleep(30)'],\n"
                "    timeout=0.1,\n"
                "    check=True,\n"
                ")\n"
            ),
            "execution_timeout_seconds": 2,
        },
    )
    followup = runner.request("execute", {"code": "print('runner still alive')"})

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert response["error"]["data"]["type"] == "TimeoutExpired"
    assert "timed out" in response["error"]["message"]
    assert followup == {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {"output": "runner still alive\n"},
    }


def test_runner_timeout_is_not_swallowed_by_user_exception_handler(
    runner: LocalRunner,
) -> None:
    response = runner.request(
        "execute",
        {
            "code": (
                "caught = 0\n"
                "while True:\n"
                "    try:\n"
                "        pass\n"
                "    except Exception:\n"
                "        caught += 1\n"
            ),
            "execution_timeout_seconds": 0.1,
        },
    )
    followup = runner.request("execute", {"code": "print('still alive')"})

    assert response["jsonrpc"] == "2.0"
    assert response["result"] == {
        "timeout": {"seconds": 0.1},
        "stdout": "",
        "stderr": "",
        "state": {"preserved": True, "source": "live_kernel", "scope": "full_live"},
    }
    assert followup == {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {"output": "still alive\n"},
    }


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
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert response["result"]["timeout"] == {"seconds": 0.1}
    assert response["result"]["stdout"] == "stdout before native\n"
    assert response["result"]["stderr"] == "stderr before native\n"
    assert response["result"]["state"]["preserved"] is False
    assert response["result"]["state"]["source"] == "pickle_snapshot"
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
        status = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
        if status.returncode == 0 and status.stdout.strip().startswith("Z"):
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

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert response["result"]["timeout"] == {"seconds": 0.1}
    assert response["result"]["stdout"] == f"child={child_pid}\n"
    assert response["result"]["stderr"] == ""
    assert response["result"]["state"]["preserved"] is False
    assert response["result"]["state"]["source"] == "pickle_snapshot"
