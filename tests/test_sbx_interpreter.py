"""Tests for the Docker Sandboxes execution backend."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import select
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import UserDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("websockets")  # SBX/supervisor backend requires the [sbx] extra

pytestmark = pytest.mark.sbx

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput  # noqa: E402

from predict_rlm.backends import (  # noqa: E402
    DEFAULT_SBX_TEMPLATE,
    SbxBackend,
    SbxConfig,
    SbxPool,
)
from predict_rlm.backends.base import SandboxExecutionError, SandboxFatalError  # noqa: E402
from predict_rlm.backends.supervisor._payload import (  # noqa: E402
    _pickleable_globals_snapshot,
)
from predict_rlm.files import SyncedFile  # noqa: E402
from predict_rlm.workspace import DirectWorkspaceMount  # noqa: E402

PAYLOAD_PATH = Path(__file__).parents[1] / "src" / "predict_rlm" / "backends" / "supervisor" / "_payload.py"


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


def _real_sbx_available() -> bool:
    if os.environ.get("PREDICT_RLM_RUN_SBX_TESTS") != "1":
        return False
    if shutil.which("sbx") is None:
        return False
    return (
        subprocess.run(
            ["sbx", "ls"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).returncode
        == 0
    )


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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
            env={**dict(), "PREDICT_RLM_SBX_ROOT": str(env_root)},
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
        assert self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        return self._request_id

    def write_message(self, message: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def read_message(self, timeout: float = 3) -> dict:
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


class SequentialActions:
    def __init__(self, *actions: SimpleNamespace) -> None:
        self.actions = list(actions)
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        assert self.actions, "PredictRLM requested more actions than the test provided"
        return self.actions.pop(0)


class PredictionStub:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def keys(self) -> list[str]:
        return ["answer"]

    def __getitem__(self, key: str) -> str:
        return getattr(self, key)


def assert_predict_rlm_recovers_after_user_exceptions_and_tools_still_work(
    pool: SbxPool,
) -> None:
    from predict_rlm import PredictRLM

    async def host_echo(text: str) -> dict:
        await asyncio.sleep(0)
        return {"text": f"echo:{text}"}

    actions = SequentialActions(
        SimpleNamespace(
            reasoning="raise a normal exception from inside a loop",
            code=(
                "for idx in range(3):\n"
                "    print('loop idx', idx)\n"
                "    if idx == 1:\n"
                "        raise ValueError(f'bad loop idx {idx}')\n"
            ),
        ),
        SimpleNamespace(
            reasoning="exercise a missing variable path",
            code="print(missing_recovery_variable)\n",
        ),
        SimpleNamespace(
            reasoning="prove host callbacks still work after ordinary exceptions",
            code=(
                "prediction = await predict('question: str -> answer: str', "
                "question='after exceptions')\n"
                "echoed = await host_echo(prediction['answer'])\n"
                "SUBMIT(answer=echoed['text'])"
            ),
        ),
    )

    mock_lm = MagicMock()
    mock_predictor = MagicMock()
    mock_predictor.acall = AsyncMock(return_value=PredictionStub("tool-ok"))
    rlm = PredictRLM(
        "prompt -> answer",
        sub_lm=mock_lm,
        max_iterations=3,
        tools={"host_echo": host_echo},
        sandbox_backend="sbx",
        sbx_pool=pool,
    )
    rlm.generate_action = actions

    with patch("predict_rlm.predict_rlm.dspy.Predict", return_value=mock_predictor):
        prediction = rlm(prompt="exercise exception recovery")

    assert prediction.answer == "echo:tool-ok"
    assert [call["iteration"] for call in actions.calls] == ["1/3", "2/3", "3/3"]
    assert mock_predictor.acall.await_count == 1
    assert mock_predictor.acall.await_args.kwargs["question"] == "after exceptions"
    assert len(prediction.trace.steps) == 3

    value_error_step, name_error_step, final_step = prediction.trace.steps
    assert "for idx in range(3)" in value_error_step.code
    assert "raise ValueError" in value_error_step.code
    assert "[Error]" in value_error_step.untruncated_output
    assert "ValueError" in value_error_step.untruncated_output
    assert "bad loop idx 1" in value_error_step.untruncated_output
    assert "[Error]" in name_error_step.untruncated_output
    assert "NameError" in name_error_step.untruncated_output
    assert "missing_recovery_variable" in name_error_step.untruncated_output
    assert final_step.output == "FINAL: {'answer': 'echo:tool-ok'}"


@pytest.fixture
def runner(tmp_path):
    proc = LocalRunner(tmp_path)
    try:
        yield proc
    finally:
        proc.close()


class TestPythonRunnerSnapshots:
    def test_snapshot_skips_native_like_objects_without_pickling(self):
        class NativeLike:
            reduce_called = False
            __module__ = "mujoco._structs"

            def __reduce__(self):
                type(self).reduce_called = True
                return int, (1,)

        snapshot = _pickleable_globals_snapshot({"native_model": NativeLike()})

        assert snapshot["globals"] == {}
        assert snapshot["restored_globals"] == []
        assert snapshot["lost_globals"] == ["native_model"]
        assert NativeLike.reduce_called is False

    def test_snapshot_preserves_safe_dataclass_and_mapping_values(self):
        @dataclass
        class RunSummary:
            name: str
            scores: list[int]
            output_path: Path

        snapshot = _pickleable_globals_snapshot({
            "summary": RunSummary("mjcf", [1, 2], Path("/app/model.xml")),
            "config": UserDict({"threshold": 0.6, "labels": ("fast", "exact")}),
        })

        assert snapshot["lost_globals"] == []
        assert snapshot["restored_globals"] == ["config", "summary"]
        assert snapshot["globals"] == {
            "summary": {
                "name": "mjcf",
                "scores": [1, 2],
                "output_path": Path("/app/model.xml"),
            },
            "config": {"threshold": 0.6, "labels": ("fast", "exact")},
        }

    def test_snapshot_crosses_runner_queue_after_hard_timeout(self, runner: LocalRunner):
        runner.request(
            "execute",
            {
                "code": (
                    "from dataclasses import dataclass\n"
                    "@dataclass\n"
                    "class RunSummary:\n"
                    "    name: str\n"
                    "    scores: list[int]\n"
                    "summary = RunSummary('mjcf', [1, 2])\n"
                    "print('seeded')\n"
                )
            },
        )
        timeout = runner.request(
            "execute",
            {
                "code": (
                    "import signal\n"
                    "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
                    "while True:\n"
                    "    pass\n"
                ),
                "execution_timeout_seconds": 0.05,
                "execution_timeout_interrupt_grace_seconds": 0.01,
            },
        )
        followup = runner.request(
            "execute",
            {"code": "print(type(summary).__name__)\nprint(summary['name'])"},
        )

        assert timeout["result"]["state"]["preserved"] is False
        assert timeout["result"]["state"]["source"] == "pickle_snapshot"
        assert "summary" in timeout["result"]["state"]["restored_globals"]
        assert followup["result"]["output"] == "dict\nmjcf\n"


class TestPythonRunnerProtocol:
    def test_user_subprocess_stdin_is_isolated_from_runner_protocol(
        self, runner: LocalRunner
    ):
        code = (
            "import subprocess, sys\n"
            "subprocess.run(\n"
            "    [sys.executable, '-c', 'import os; print(os.read(0, 1))'],\n"
            "    capture_output=True,\n"
            "    text=True,\n"
            "    timeout=0.2,\n"
            ")\n"
        )

        result = runner.request("execute", {"code": code})
        followup = runner.request("execute", {"code": "sentinel = 123\nprint(sentinel)"})

        assert result["result"]["output"].strip() == ""
        assert followup["result"]["output"].strip() == "123"

    def test_reset_clears_globals_but_runner_process_survives(self, runner: LocalRunner):
        before = runner.request("execute", {"code": "x = 40\nprint('ready')"})
        reset = runner.request("reset")
        after = runner.request("execute", {"code": "print('x' in globals())"})

        assert before["result"]["output"].strip() == "ready"
        assert reset["result"] == {}
        assert after["result"]["output"].strip() == "False"

    def test_submit_returns_final_payload(self, runner: LocalRunner):
        runner.request(
            "register_output_fields",
            {"fields": [{"name": "answer", "annotation": "str"}]},
        )

        result = runner.request("execute", {"code": "SUBMIT(answer='done')"})

        assert result["result"]["final"] == {"answer": "done"}

    def test_deferred_submit_preserves_background_service_for_confirmation(
        self, runner: LocalRunner
    ):
        runner.request(
            "register_output_fields",
            {"fields": [{"name": "answer", "annotation": "str"}]},
        )
        start_service = runner.request(
            "execute",
            {
                "code": (
                    "import socket, subprocess, sys, time\n"
                    "server_code = "
                    "'import http.server, socketserver; '"
                    "'srv = socketserver.TCPServer((\\\"127.0.0.1\\\", 0), http.server.SimpleHTTPRequestHandler); '"
                    "'print(srv.server_address[1], flush=True); '"
                    "'srv.serve_forever()'\n"
                    "server = subprocess.Popen(\n"
                    "    [sys.executable, '-u', '-c', server_code],\n"
                    "    stdin=subprocess.DEVNULL,\n"
                    "    stdout=subprocess.PIPE,\n"
                    "    stderr=subprocess.DEVNULL,\n"
                    "    text=True,\n"
                    ")\n"
                    "port = int(server.stdout.readline())\n"
                    "deadline = time.time() + 5\n"
                    "while True:\n"
                    "    try:\n"
                    "        with socket.create_connection(('127.0.0.1', port), timeout=0.2):\n"
                    "            break\n"
                    "    except OSError:\n"
                    "        if time.time() > deadline:\n"
                    "            raise\n"
                    "        time.sleep(0.05)\n"
                    "print(port, server.pid)\n"
                )
            },
        )
        port_text, server_pid_text = start_service["result"]["output"].strip().split()
        port = int(port_text)
        server_pid = int(server_pid_text)

        try:
            submitted = runner.request(
                "execute",
                {
                    "code": "SUBMIT(answer='started')",
                    "defer_final_output": True,
                },
            )
            probe = runner.request(
                "execute",
                {
                    "code": (
                        "import socket\n"
                        f"with socket.create_connection(('127.0.0.1', {port}), timeout=1):\n"
                        "    print('alive')\n"
                    )
                },
            )
            final = runner.request("execute", {"code": "SUBMIT(answer='confirmed')"})
            runner.request("shutdown", {"preserve_kernel_process": True})
            runner.proc.wait(timeout=5)

            post_final_socket = socket.socket()
            post_final_socket.settimeout(1)
            post_final_result = post_final_socket.connect_ex(("127.0.0.1", port))
            post_final_socket.close()
        finally:
            try:
                os.kill(server_pid, 15)
            except OSError:
                pass

        assert submitted["result"] == {"submitted": {"answer": "started"}}
        assert probe["result"]["output"].strip() == "alive"
        assert final["result"] == {"final": {"answer": "confirmed"}}
        assert post_final_result == 0

    def test_predict_image_data_url_round_trips_to_host_tool(self, runner: LocalRunner):
        png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        runner.request("register_tools", {"tools": ["predict"]})

        tool_call = runner.request(
            "execute",
            {
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
                    "print(result.visible_text)\n"
                )
            },
        )

        assert tool_call["method"] == "tool_call"
        assert tool_call["params"]["name"] == "predict"
        assert tool_call["params"]["args"] == [
            "image: dspy.Image, question: str -> visible_text: str"
        ]
        assert tool_call["params"]["kwargs"]["question"] == "What text is visible?"
        image = tool_call["params"]["kwargs"]["image"]
        assert image.startswith("data:image/png;base64,")
        assert base64.b64decode(image.removeprefix("data:image/png;base64,")) == png_bytes

        assert runner.proc.stdin is not None
        assert runner.proc.stdout is not None
        runner.proc.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": tool_call["id"],
                    "result": {"type": "json", "value": '{"visible_text": "hello"}'},
                }
            )
            + "\n"
        )
        runner.proc.stdin.flush()
        response = json.loads(runner.proc.stdout.readline())

        assert response["result"]["output"] == "hello\n"

    # TEMPORARY SKIP: drives many concurrent host-tool calls over a local _payload.py
    # subprocess and reads replies on tight (3s) select timeouts. On loaded CI runners
    # the supervisor is slow to respond and the reader/teardown intermittently times out.
    # Not a product bug -- a CI-timing-sensitive subprocess test. Re-enable with
    # CI-tolerant helper timeouts (follow-up).
    # local-only: flaky on CI (subprocess-timing-sensitive); re-enable with CI-tolerant timeouts
    @pytest.mark.local
    def test_stale_concurrent_tool_calls_do_not_poison_later_execute(
        self, runner: LocalRunner
    ):
        runner.request("register_tools", {"tools": ["predict"]})
        first_execute_id = runner.send(
            "execute",
            {
                "code": (
                    "import asyncio\n"
                    "for idx in range(11):\n"
                    "    asyncio.create_task(predict('x: int -> answer: int', x=idx))\n"
                    "await asyncio.sleep(0.05)\n"
                    "print('scheduled stale calls')\n"
                )
            },
        )

        stale_tool_ids: list[int] = []
        while True:
            message = runner.read_message()
            if message.get("method") == "tool_call":
                stale_tool_ids.append(message["id"])
                continue
            if message.get("id") == first_execute_id:
                assert message["result"]["output"] == "scheduled stale calls\n"
                break

        assert len(stale_tool_ids) == 11
        for tool_id in stale_tool_ids:
            runner.write_message(
                {
                    "jsonrpc": "2.0",
                    "id": tool_id,
                    "result": {"type": "json", "value": '{"answer": -1}'},
                }
            )

        followup_execute_id = runner.send(
            "execute",
            {
                "code": (
                    "result = await predict('x: int -> answer: int', x=999)\n"
                    "print(result.answer)\n"
                )
            },
        )

        saw_followup_tool_call = False
        while True:
            message = runner.read_message()
            if message.get("method") == "tool_call":
                assert message["params"]["kwargs"] == {"x": 999}
                saw_followup_tool_call = True
                runner.write_message(
                    {
                        "jsonrpc": "2.0",
                        "id": message["id"],
                        "result": {"type": "json", "value": '{"answer": 123}'},
                    }
                )
                continue
            if message.get("id") == followup_execute_id:
                assert message["result"]["output"] == "123\n"
                break

        assert saw_followup_tool_call

    def test_syntax_error_uses_json_rpc_error(self, runner: LocalRunner):
        result = runner.request("execute", {"code": "for"})

        assert result["error"]["data"]["type"] == "SyntaxError"

    def test_timeout_returns_structured_result_and_runner_survives(
        self, runner: LocalRunner
    ):
        result = runner.request(
            "execute",
            {
                "code": (
                    "import sys\n"
                    "print('before timeout')\n"
                    "print('stderr before timeout', file=sys.stderr)\n"
                    "partial_timeout_state = 42\n"
                    "while True:\n"
                    "    pass\n"
                ),
                "execution_timeout_seconds": 0.1,
            },
        )
        followup = runner.request(
            "execute",
            {"code": "print('partial_timeout_state' in globals())\nprint('still alive')"},
        )

        assert result["result"] == {
            "timeout": {"seconds": 0.1},
            "stdout": "before timeout\n",
            "stderr": "stderr before timeout\n",
            "state": {
                "preserved": True,
                "source": "live_kernel",
                "scope": "full_live",
            },
        }
        assert followup["result"]["output"] == "True\nstill alive\n"

    def test_execute_captures_child_process_output_with_timeout(
        self, runner: LocalRunner
    ):
        result = runner.request(
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

        assert result["result"]["output"] == "child stdout\nchild stderr\n"
        assert followup["result"]["output"] == "runner still usable\n"
        assert leaked_stderr == ""

    def test_timeout_preserves_child_process_output_and_runner_survives(
        self, runner: LocalRunner
    ):
        result = runner.request(
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

        assert result["result"]["timeout"] == {"seconds": 0.2}
        assert result["result"]["stdout"] == "child before timeout\n"
        assert result["result"]["stderr"].startswith("child err before timeout\n")
        assert followup["result"]["output"] == "runner survived timeout\n"
        assert leaked_stderr == ""

    def test_unbounded_execute_runner_exit_returns_error_and_supervisor_survives(
        self, runner: LocalRunner
    ):
        result = runner.request("execute", {"code": "import os\nos._exit(7)"})
        followup = runner.request(
            "execute", {"code": "print('supervisor survived runner exit')"}
        )

        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 1
        assert result["error"]["data"]["type"] == "RuntimeError"
        assert "execution runner exited without a result" in result["error"]["message"]
        assert followup["result"]["output"] == "supervisor survived runner exit\n"

    def test_timeout_is_not_swallowed_by_user_exception_handler(
        self, runner: LocalRunner
    ):
        result = runner.request(
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

        assert result["result"] == {
            "timeout": {"seconds": 0.1},
            "stdout": "",
            "stderr": "",
            "state": {
                "preserved": True,
                "source": "live_kernel",
                "scope": "full_live",
            },
        }
        assert followup["result"]["output"] == "still alive\n"

    def test_timeout_cancels_pending_async_work(self, runner: LocalRunner):
        result = runner.request(
            "execute",
            {
                "code": (
                    "import asyncio\n"
                    "async def mutate_late():\n"
                    "    await asyncio.sleep(1)\n"
                    "    globals()['late_mutation'] = 'leaked'\n"
                    "await mutate_late()\n"
                ),
                "execution_timeout_seconds": 0.1,
            },
        )
        followup = runner.request(
            "execute",
            {
                "code": (
                    "import asyncio\n"
                    "await asyncio.sleep(0.3)\n"
                    "print('late_mutation' in globals())"
                )
            },
        )

        assert result["result"]["timeout"] == {"seconds": 0.1}
        assert followup["result"]["output"] == "False\n"

    def test_file_helpers_preserve_virtual_paths(self, runner: LocalRunner, tmp_path: Path):
        source = tmp_path / "input.txt"
        source.write_text("hello", encoding="utf-8")
        out = tmp_path / "out.txt"

        runner.request(
            "mount_file",
            {"host_path": str(source), "virtual_path": "/sandbox/input/source/input.txt"},
        )
        runner.request("mkdir_p", {"path": "/sandbox/output/result"})
        runner.request(
            "execute",
            {
                "code": (
                    "with open('/sandbox/input/source/input.txt', encoding='utf-8') as f:\n"
                    "    text = f.read()\n"
                    "with open('/sandbox/output/result/output.txt', 'w', encoding='utf-8') as f:\n"
                    "    f.write(text + ' world')"
                )
            },
        )

        files = runner.request("list_dir", {"path": "/sandbox/output/result"})
        runner.request(
            "sync_file",
            {
                "virtual_path": "/sandbox/output/result/output.txt",
                "host_path": str(out),
            },
        )

        assert files["result"]["files"] == ["/sandbox/output/result/output.txt"]
        assert out.read_text(encoding="utf-8") == "hello world"

    def test_pathlib_path_remains_a_type(self, runner: LocalRunner):
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

        assert result["result"]["output"] == "True\nFalse\n"

    def test_windows311_visual_predict_path_handles_pillow_style_path_checks(
        self, runner: LocalRunner
    ):
        runner.request("register_tools", {"tools": ["predict"]})
        tool_call = runner.request(
            "execute",
            {
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
        )

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

        assert runner.proc.stdin is not None
        assert runner.proc.stdout is not None
        runner.proc.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": tool_call["id"],
                    "result": {
                        "type": "json",
                        "value": '{"visible_text": "desktop", "answer": "yes"}',
                    },
                }
            )
            + "\n"
        )
        runner.proc.stdin.flush()
        response = json.loads(runner.proc.stdout.readline())

        assert response["result"]["output"] == "desktop\nyes\n"


class TestSbxBackendCreateNaming:
    """`sbx create` always receives a known `--name`; the name is never scraped
    from stdout (regression guard for issue #39)."""

    def _run_create(
        self, tmp_path: Path, *, config: SbxConfig, create_stdout: str
    ) -> tuple[SbxBackend, list[str]]:
        backend = SbxBackend(
            config=config,
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(PAYLOAD_PATH)],
            _staging_root=tmp_path / "staging",
        )
        captured: dict[str, list[str]] = {}

        def fake_run(cmd, *args, **kwargs):
            captured["cmd"] = cmd
            return SimpleNamespace(returncode=0, stdout=create_stdout, stderr="")

        with (
            patch("predict_rlm.backends.sbx.backend.shutil.which", return_value="/usr/bin/sbx"),
            patch("predict_rlm.backends.sbx.backend.subprocess.run", side_effect=fake_run),
            patch.object(SbxBackend, "_prepare_supervisor_script", return_value=tmp_path / "sup.py"),
            patch.object(SbxBackend, "_apply_network_policy"),
            patch.object(SbxBackend, "_bootstrap_packages"),
            patch.object(SbxBackend, "_setup_direct_workspace_aliases_in_sandbox"),
        ):
            backend._start_sbx_and_prepare_supervisor()
        return backend, captured["cmd"]

    def test_generates_name_and_passes_it_to_create(self, tmp_path: Path):
        backend, cmd = self._run_create(
            tmp_path, config=SbxConfig(), create_stdout="some-auto-name\n"
        )
        assert "--name" in cmd
        name = cmd[cmd.index("--name") + 1]
        assert name.startswith("predict-rlm-")
        assert backend._sandbox_name == name

    def test_uses_explicit_config_name(self, tmp_path: Path):
        backend, cmd = self._run_create(
            tmp_path, config=SbxConfig(name="my-box"), create_stdout="my-box\n"
        )
        assert cmd[cmd.index("--name") + 1] == "my-box"
        assert backend._sandbox_name == "my-box"

    def test_ignores_update_banner_in_stdout(self, tmp_path: Path):
        # The sbx update banner draws a Unicode box; the old code grabbed its
        # bottom border as the name. The name must come from `--name`, not stdout.
        banner = (
            "╭──────────────────────────────╮\n"
            "│  A new version of sbx is out │\n"
            "╰──────────────────────────────╯\n"
        )
        backend, cmd = self._run_create(
            tmp_path, config=SbxConfig(), create_stdout=banner
        )
        name = cmd[cmd.index("--name") + 1]
        assert name.startswith("predict-rlm-")
        assert backend._sandbox_name == name
        assert "╰" not in backend._sandbox_name


class TestSbxBackendLocalRunner:
    def make_interpreter(
        self,
        tmp_path: Path,
        *,
        debug: bool = False,
        verbose: bool = False,
        tools: dict | None = None,
    ) -> SbxBackend:
        return SbxBackend(
            config=SbxConfig(name="local-test"),
            tools=tools,
            preinstall_packages=False,
            debug=debug,
            verbose=verbose,
            _supervisor_command=[sys.executable, "-u", str(PAYLOAD_PATH)],
            _staging_root=tmp_path / "staging",
        )

    def test_execute_and_state_persistence(self, tmp_path: Path):
        interpreter = self.make_interpreter(tmp_path)
        try:
            assert interpreter.execute("x = 7\nprint(x)") == "7\n"
            assert interpreter.execute("x += 1\nprint(x)") == "8\n"
        finally:
            interpreter.shutdown()

    def test_submit_returns_final_output(self, tmp_path: Path):
        interpreter = self.make_interpreter(tmp_path)
        try:
            result = interpreter.execute("SUBMIT(answer='ok')")
        finally:
            interpreter.shutdown()

        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "ok"}

    def test_debug_logs_runner_and_request_events(self, tmp_path: Path, caplog, capsys):
        caplog.set_level(logging.DEBUG, logger="predict_rlm")
        interpreter = self.make_interpreter(tmp_path, debug=True)
        try:
            assert interpreter.execute("print('hi')") == "hi\n"
        finally:
            interpreter.shutdown()

        stderr = capsys.readouterr().err
        assert "output:" not in stderr
        events = [record.getMessage().split()[0] for record in caplog.records]
        assert "sbx.runner.start" in events
        assert "sbx.runner.started" in events
        assert "sbx.request.start" in events
        assert "sbx.request.ok" in events
        assert "sbx.shutdown.complete" in events

    def test_execute_raises_recoverable_code_errors(self, tmp_path: Path):
        interpreter = self.make_interpreter(tmp_path)
        try:
            with pytest.raises(CodeInterpreterError, match="NameError"):
                interpreter.execute("print(missing_name)")
        finally:
            interpreter.shutdown()

    def test_execute_error_includes_partial_output(self, tmp_path: Path):
        interpreter = self.make_interpreter(tmp_path)
        try:
            with pytest.raises(CodeInterpreterError) as exc_info:
                interpreter.execute("print('before failure')\nraise ValueError('bad')")
        finally:
            interpreter.shutdown()

        assert "before failure" in str(exc_info.value)
        assert "ValueError" in str(exc_info.value)
        assert getattr(exc_info.value, "partial_output") == "before failure\n"

    def test_debug_logs_partial_output_on_error(self, tmp_path: Path, caplog):
        caplog.set_level(logging.DEBUG, logger="predict_rlm")
        interpreter = self.make_interpreter(tmp_path, debug=True)
        try:
            with pytest.raises(CodeInterpreterError):
                interpreter.execute("print('before failure')\nraise ValueError('bad')")
        finally:
            interpreter.shutdown()

        messages = "\n".join(record.getMessage() for record in caplog.records)
        assert "sandbox.partial_output" in messages
        assert "before failure" in messages

    # TEMPORARY SKIP: this exercises SbxBackend over its stdin/stdout transport, which is
    # test-only and deprecated (real SBX uses websocket since the exec->ws migration). On
    # loaded CI runners the supervisor request intermittently hangs/times out. The verbose
    # behavior is unique to this class; re-enable by migrating it to the websocket runner
    # when we delete the dead stdin/stdout transport (follow-up).
    # local-only: flaky on CI; deprecated SbxBackend stdin/stdout transport, slated for removal
    @pytest.mark.local
    def test_verbose_prints_output_tool_calls_and_errors(self, tmp_path: Path, capsys):
        async def add(a: int, b: int) -> dict:
            await asyncio.sleep(0)
            return {"total": a + b}

        interpreter = self.make_interpreter(
            tmp_path,
            verbose=True,
            tools={"add": add},
        )
        try:
            output = interpreter.execute("result = await add(2, 3)\nprint(result['total'])")
            with pytest.raises(CodeInterpreterError, match="ValueError"):
                interpreter.execute("raise ValueError('bad')")
        finally:
            interpreter.shutdown()

        assert output.strip() == "5"
        stderr = capsys.readouterr().err
        assert "[INFO]" not in stderr
        assert "predict_rlm.trace" not in stderr
        assert "Tool: add(" in stderr
        assert '"args": [2, 3]' in stderr
        assert "output:" in stderr
        assert "5" in stderr
        assert "error (ValueError):" in stderr
        assert "bad" in stderr

    def test_verbose_prints_partial_output_before_error(self, tmp_path: Path, capsys):
        interpreter = self.make_interpreter(tmp_path, verbose=True)
        try:
            with pytest.raises(CodeInterpreterError, match="ValueError"):
                interpreter.execute("print('before failure')\nraise ValueError('bad')")
        finally:
            interpreter.shutdown()

        stderr = capsys.readouterr().err
        assert "output:" in stderr
        assert "before failure" in stderr
        assert "error (ValueError):" in stderr
        assert "bad" in stderr

    def test_verbose_prints_submit_payload(self, tmp_path: Path, capsys):
        interpreter = self.make_interpreter(tmp_path, verbose=True)
        try:
            result = interpreter.execute("SUBMIT(answer='ok')")
        finally:
            interpreter.shutdown()

        assert isinstance(result, FinalOutput)
        stderr = capsys.readouterr().err
        assert "[INFO]" not in stderr
        assert "predict_rlm.trace" not in stderr
        assert "output:" in stderr
        assert '"answer": "ok"' in stderr

    def test_execute_timeout_returns_recoverable_observation(
        self, tmp_path: Path
    ):
        interpreter = self.make_interpreter(tmp_path)
        try:
            timeout_result = interpreter.execute(
                "import sys\n"
                "print('before timeout')\n"
                "print('stderr before timeout', file=sys.stderr)\n"
                "partial_timeout_state = 42\n"
                "while True:\n"
                "    pass\n",
                timeout=0.1,
            )
            followup = interpreter.execute(
                "print('partial_timeout_state' in globals())\nprint('still alive')"
            )
        finally:
            interpreter.shutdown()

        assert "[Timeout] Iteration execution timed out after 0.1s" in timeout_result
        assert "[stdout]\nbefore timeout" in timeout_result
        assert "[stderr]\nstderr before timeout" in timeout_result
        assert timeout_result.timeout_seconds == 0.1
        assert timeout_result.state == {
            "preserved": True,
            "source": "live_kernel",
            "scope": "full_live",
        }
        assert timeout_result.state_preserved is True
        assert followup == "True\nstill alive\n"

    def test_default_recoverable_timeout_grace_is_shared(self, tmp_path: Path):
        from predict_rlm.execution_timeout import (
            DEFAULT_RECOVERABLE_EXECUTION_TIMEOUT_GRACE_SECONDS,
            ITERATION_TIMEOUT_FAILURE_CLASS,
        )

        interpreter = self.make_interpreter(tmp_path)
        try:
            assert DEFAULT_RECOVERABLE_EXECUTION_TIMEOUT_GRACE_SECONDS == 30.0
            assert (
                interpreter._host_watchdog_timeout(
                    2.0,
                    ITERATION_TIMEOUT_FAILURE_CLASS,
                )
                == 32.0
            )
        finally:
            interpreter.shutdown()

    def test_delayed_structured_timeout_uses_recoverable_grace(self, tmp_path: Path):
        runner_script = tmp_path / "delayed_timeout_runner.py"
        runner_script.write_text(
            """
import json
import sys
import time

for line in sys.stdin:
    request = json.loads(line)
    request_id = request["id"]
    if request["method"] == "execute":
        time.sleep(1.1)
        print(json.dumps({
            "jsonrpc": "2.0",
            "result": {
                "timeout": {"seconds": request["params"]["execution_timeout_seconds"]},
                "stdout": "late timeout\\n",
                "stderr": "",
            },
            "id": request_id,
        }), flush=True)
    elif request["method"] == "shutdown":
        print(json.dumps({"jsonrpc": "2.0", "result": {}, "id": request_id}), flush=True)
        break
""".lstrip(),
            encoding="utf-8",
        )
        interpreter = SbxBackend(
            config=SbxConfig(name="local-test", exec_timeout=3),
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(runner_script)],
            _staging_root=tmp_path / "staging",
        )
        start = time.monotonic()
        try:
            timeout_result = interpreter.execute("while True: pass", timeout=0.05)
        finally:
            interpreter.shutdown()

        assert time.monotonic() - start >= 1.0
        assert "[Timeout] Iteration execution timed out after 0.05s" in timeout_result
        assert "[stdout]\nlate timeout" in timeout_result

    def test_execute_after_structured_timeout_restarts_dead_runner_with_diagnostic(
        self, tmp_path: Path
    ):
        runner_script = tmp_path / "exit_after_timeout_runner.py"
        runner_script.write_text(
            """
import json
import os
import pathlib
import sys

root = pathlib.Path(os.environ["PREDICT_RLM_SBX_ROOT"])
marker = root / "already_exited"

for line in sys.stdin:
    request = json.loads(line)
    request_id = request["id"]
    if request["method"] == "shutdown":
        print(json.dumps({"jsonrpc": "2.0", "result": {}, "id": request_id}), flush=True)
        break
    if request["method"] != "execute":
        print(json.dumps({"jsonrpc": "2.0", "result": {}, "id": request_id}), flush=True)
        continue
    if not marker.exists():
        marker.write_text("yes", encoding="utf-8")
        print(json.dumps({
            "jsonrpc": "2.0",
            "result": {
                "timeout": {"seconds": request["params"]["execution_timeout_seconds"]},
                "stdout": "before timeout\\n",
                "stderr": "command timed out\\n",
            },
            "id": request_id,
        }), flush=True)
        print("runner stderr tail", file=sys.stderr, flush=True)
        raise SystemExit(137)
    print(json.dumps({
        "jsonrpc": "2.0",
        "result": {"output": "fresh runner\\n"},
        "id": request_id,
    }), flush=True)
""".lstrip(),
            encoding="utf-8",
        )
        interpreter = SbxBackend(
            config=SbxConfig(name="local-test", exec_timeout=3),
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(runner_script)],
            _staging_root=tmp_path / "staging",
        )
        try:
            timeout_result = interpreter.execute("run_slow_command()", timeout=0.2)
            deadline = time.monotonic() + 2
            while interpreter._proc and interpreter._proc.poll() is None:
                assert time.monotonic() < deadline
                time.sleep(0.01)
            restart_result = interpreter.execute("print(existing_global)", timeout=0.2)
            followup = interpreter.execute("print('fresh runner')")
        finally:
            interpreter.shutdown()

        assert "[Timeout] Iteration execution timed out after 0.2s" in timeout_result
        assert "Sbx supervisor exited after the previous execute response" in restart_result
        assert "The supervisor process was restarted" in restart_result
        assert "Python globals from the prior supervisor were lost" in restart_result
        assert "supervisor_returncode=137" in restart_result
        assert "previous_request_id=1" in restart_result
        assert "previous_method=execute" in restart_result
        assert "previous_execution_timeout_seconds=0.2" in restart_result
        assert "runner stderr tail" in restart_result
        assert followup == "fresh runner\n"

    def test_iteration_timeout_recovery_failure_is_bounded_by_grace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import predict_rlm.execution_timeout as execution_timeout

        monkeypatch.setattr(
            execution_timeout,
            "DEFAULT_RECOVERABLE_EXECUTION_TIMEOUT_GRACE_SECONDS",
            0.2,
        )
        interpreter = SbxBackend(
            config=SbxConfig(name="silent-test", exec_timeout=30),
            preinstall_packages=False,
            _supervisor_command=[
                sys.executable,
                "-u",
                "-c",
                "import sys, time\nsys.stdin.readline()\ntime.sleep(30)\n",
            ],
            _staging_root=tmp_path / "staging",
        )
        start = time.monotonic()
        try:
            with pytest.raises(SandboxFatalError, match="failed to recover"):
                interpreter.execute("print('never')", timeout=0.1)
        finally:
            interpreter.shutdown()

        assert 0.25 <= time.monotonic() - start < 1.0

    def test_file_helpers_round_trip_virtual_paths(self, tmp_path: Path):
        interpreter = self.make_interpreter(tmp_path)
        source = tmp_path / "input.txt"
        source.write_text("hello", encoding="utf-8")
        output = tmp_path / "output.txt"
        try:
            interpreter.mount_file_at(str(source), "/sandbox/input/source/input.txt")
            interpreter.mkdir_p("/sandbox/output/result")
            result = interpreter.execute(
                "from pathlib import Path\n"
                "print(Path(input_path).exists())\n"
                "text = Path(input_path).read_text()\n"
                "with open('/sandbox/output/result/output.txt', 'w', encoding='utf-8') as f:\n"
                "    f.write(text + ' sbx')",
                variables={"input_path": "/sandbox/input/source/input.txt"},
            )
            files = interpreter.list_dir("/sandbox/output/result")
            interpreter.sync_file_to("/sandbox/output/result/output.txt", str(output))
        finally:
            interpreter.shutdown()

        assert "True" in result
        assert files == ["/sandbox/output/result/output.txt"]
        assert output.read_text(encoding="utf-8") == "hello sbx"

    def test_file_helpers_are_host_side_and_do_not_start_runner(self, tmp_path: Path):
        interpreter = SbxBackend(
            config=SbxConfig(name="file-only-test"),
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-c", "raise SystemExit(99)"],
            _staging_root=tmp_path / "staging",
        )
        source = tmp_path / "host-input.txt"
        source.write_text("host visible", encoding="utf-8")
        output = tmp_path / "host-output.txt"

        try:
            interpreter.mount_file_at(str(source), "/sandbox/input/source/input.txt")
            interpreter.mkdir_p("/sandbox/output/result/nested")
            staged_output = (
                tmp_path / "staging" / "sandbox" / "output" / "result" / "nested" / "output.txt"
            )
            staged_output.write_text("from staging", encoding="utf-8")

            files = interpreter.list_dir("/sandbox/output/result")
            interpreter.sync_file_to(
                "/sandbox/output/result/nested/output.txt",
                str(output),
            )
        finally:
            interpreter.shutdown()

        staged_input = tmp_path / "staging" / "sandbox" / "input" / "source" / "input.txt"
        assert staged_input.read_text(encoding="utf-8") == "host visible"
        assert (tmp_path / "staging" / "sandbox" / "output" / "result" / "nested").is_dir()
        assert files == ["/sandbox/output/result/nested/output.txt"]
        assert output.read_text(encoding="utf-8") == "from staging"
        assert interpreter._proc is None

    def test_shutdown_removes_owned_staging_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.chdir(tmp_path)
        interpreter = SbxBackend(
            config=SbxConfig(name="local-test"),
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(PAYLOAD_PATH)],
        )
        staging_root = interpreter._staging_root
        source = tmp_path / "input.txt"
        source.write_text("host visible", encoding="utf-8")

        try:
            interpreter.mount_file_at(str(source), "/sandbox/input/source/input.txt")
            assert staging_root.is_dir()
        finally:
            interpreter.shutdown()

        assert not staging_root.exists()

    def test_shutdown_preserves_caller_owned_staging_root(self, tmp_path: Path):
        staging_root = tmp_path / "staging"
        interpreter = SbxBackend(
            config=SbxConfig(name="local-test"),
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(PAYLOAD_PATH)],
            _staging_root=staging_root,
        )
        source = tmp_path / "input.txt"
        source.write_text("host visible", encoding="utf-8")

        try:
            interpreter.mount_file_at(str(source), "/sandbox/input/source/input.txt")
        finally:
            interpreter.shutdown()

        assert staging_root.is_dir()
        assert (staging_root / "sandbox" / "input" / "source" / "input.txt").read_text(
            encoding="utf-8"
        ) == "host visible"

    def test_persist_preserves_owned_staging_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.chdir(tmp_path)
        interpreter = SbxBackend(
            config=SbxConfig(name="local-test", persist=True),
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(PAYLOAD_PATH)],
        )
        staging_root = interpreter._staging_root

        try:
            interpreter.mkdir_p("/sandbox/output/result")
        finally:
            interpreter.shutdown()

        assert staging_root.is_dir()

    def test_execute_can_call_registered_host_tools(self, tmp_path: Path):
        async def add(a: int, b: int) -> dict:
            await asyncio.sleep(0)
            return {"total": a + b}

        interpreter = SbxBackend(
            config=SbxConfig(name="local-test"),
            tools={"add": add},
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(PAYLOAD_PATH)],
            _staging_root=tmp_path / "staging",
        )
        try:
            output = interpreter.execute("result = await add(2, 3)\nprint(result['total'])")
        finally:
            interpreter.shutdown()

        assert output.strip() == "5"

    def test_predict_rlm_recovers_after_user_exceptions_and_tools_still_work(
        self, tmp_path: Path
    ):
        pool = SbxPool(
            size=1,
            config=SbxConfig(name="local-test-user-exceptions"),
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(PAYLOAD_PATH)],
            _staging_root=tmp_path / "staging",
        )
        try:
            assert_predict_rlm_recovers_after_user_exceptions_and_tools_still_work(pool)
        finally:
            pool.shutdown()

    def test_host_tool_synced_file_writeback_updates_sandbox_file(self, tmp_path: Path):
        received_paths: list[str] = []

        def mutate(path: Annotated[str, SyncedFile(writeback=True)]) -> str:
            received_paths.append(path)
            file_path = Path(path)
            original = file_path.read_text(encoding="utf-8")
            file_path.write_text(original + " + host", encoding="utf-8")
            return file_path.read_text(encoding="utf-8")

        interpreter = SbxBackend(
            config=SbxConfig(name="local-test"),
            tools={"mutate": mutate},
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(PAYLOAD_PATH)],
            _staging_root=tmp_path / "staging",
        )
        source = tmp_path / "input.txt"
        source.write_text("sandbox", encoding="utf-8")
        try:
            interpreter.mount_file_at(str(source), "/sandbox/input/source/input.txt")
            output = interpreter.execute(
                "result = await mutate('/sandbox/input/source/input.txt')\n"
                "print(result)\n"
                "with open('/sandbox/input/source/input.txt', encoding='utf-8') as f:\n"
                "    print(f.read())"
            )
        finally:
            interpreter.shutdown()

        assert output.strip().splitlines() == ["sandbox + host", "sandbox + host"]
        assert len(received_paths) == 1
        assert received_paths[0].endswith("/input.txt")
        assert not received_paths[0].startswith("/sandbox/")

    def test_host_tool_synced_file_without_writeback_leaves_sandbox_file_unchanged(
        self, tmp_path: Path
    ):
        def mutate(path: Annotated[str, SyncedFile(writeback=False)]) -> str:
            file_path = Path(path)
            file_path.write_text("host only", encoding="utf-8")
            return file_path.read_text(encoding="utf-8")

        interpreter = SbxBackend(
            config=SbxConfig(name="local-test"),
            tools={"mutate": mutate},
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(PAYLOAD_PATH)],
            _staging_root=tmp_path / "staging",
        )
        source = tmp_path / "input.txt"
        source.write_text("sandbox", encoding="utf-8")
        try:
            interpreter.mount_file_at(str(source), "/sandbox/input/source/input.txt")
            output = interpreter.execute(
                "result = await mutate(path='/sandbox/input/source/input.txt')\n"
                "print(result)\n"
                "with open('/sandbox/input/source/input.txt', encoding='utf-8') as f:\n"
                "    print(f.read())"
            )
        finally:
            interpreter.shutdown()

        assert output.strip().splitlines() == ["host only", "sandbox"]

    def test_host_tool_synced_file_host_dir_writeback_uses_configured_directory(
        self, tmp_path: Path
    ):
        host_dir = tmp_path / "synced-host-dir"
        received_paths: list[str] = []

        def mutate(path: str) -> str:
            received_paths.append(path)
            file_path = Path(path)
            file_path.write_text(
                file_path.read_text(encoding="utf-8") + " + configured",
                encoding="utf-8",
            )
            return str(file_path)

        mutate.__annotations__["path"] = Annotated[str, SyncedFile(host_dir=str(host_dir))]

        interpreter = SbxBackend(
            config=SbxConfig(name="local-test"),
            tools={"mutate": mutate},
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(PAYLOAD_PATH)],
            _staging_root=tmp_path / "staging",
        )
        source = tmp_path / "input.txt"
        source.write_text("sandbox", encoding="utf-8")
        try:
            interpreter.mount_file_at(str(source), "/sandbox/input/source/input.txt")
            output = interpreter.execute(
                "received = await mutate(path='/sandbox/input/source/input.txt')\n"
                "print(received)\n"
                "with open('/sandbox/input/source/input.txt', encoding='utf-8') as f:\n"
                "    print(f.read())"
            )
        finally:
            interpreter.shutdown()

        assert received_paths == [str(host_dir / "input.txt")]
        assert output.strip().splitlines() == [
            str(host_dir / "input.txt"),
            "sandbox + configured",
        ]
        assert (host_dir / "input.txt").read_text(encoding="utf-8") == ("sandbox + configured")

    def test_execute_serializes_concurrent_requests(self, tmp_path: Path):
        runner_script = tmp_path / "detect_concurrent_requests.py"
        runner_script.write_text(
            """
import json
import select
import sys
import time


def send(message):
    sys.stdout.write(json.dumps(message) + "\\n")
    sys.stdout.flush()


while True:
    line = sys.stdin.readline()
    if not line:
        break
    request = json.loads(line)
    request_id = request.get("id")
    method = request.get("method")
    if method == "shutdown":
        send({"jsonrpc": "2.0", "result": {"shutdown": True}, "id": request_id})
        break
    if method != "execute":
        send({"jsonrpc": "2.0", "result": {}, "id": request_id})
        continue
    time.sleep(0.2)
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if readable:
        send({
            "jsonrpc": "2.0",
            "error": {
                "code": -32000,
                "message": "concurrent request detected",
                "data": {"type": "RuntimeError", "args": ["concurrent request detected"]},
            },
            "id": request_id,
        })
        continue
    send({
        "jsonrpc": "2.0",
        "result": {"output": request.get("params", {}).get("code", "") + "\\n"},
        "id": request_id,
    })
""".lstrip(),
            encoding="utf-8",
        )
        interpreter = SbxBackend(
            config=SbxConfig(name="local-test", exec_timeout=2),
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(runner_script)],
            _staging_root=tmp_path / "staging",
        )
        barrier = threading.Barrier(3)
        results: list[str] = []
        errors: list[BaseException] = []

        def execute(code: str) -> None:
            barrier.wait()
            try:
                results.append(interpreter.execute(code).strip())
            except BaseException as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=execute, args=("first",)),
            threading.Thread(target=execute, args=("second",)),
        ]
        try:
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join(timeout=3)
        finally:
            interpreter.shutdown()

        assert [thread.is_alive() for thread in threads] == [False, False]
        assert errors == []
        assert sorted(results) == ["first", "second"]

    def test_concurrent_host_tool_calls_do_not_run_serially(self, tmp_path: Path):
        async def slow(value: int) -> int:
            await asyncio.sleep(0.35)
            return value

        interpreter = SbxBackend(
            config=SbxConfig(name="local-test", exec_timeout=3),
            tools={"slow": slow},
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(PAYLOAD_PATH)],
            _staging_root=tmp_path / "staging",
        )
        try:
            interpreter.prewarm()
            interpreter.execute("pass")
            start = time.monotonic()
            output = interpreter.execute(
                "import asyncio\n"
                "results = await asyncio.gather(slow(1), slow(2))\n"
                "print(results)"
            )
            elapsed = time.monotonic() - start
        finally:
            interpreter.shutdown()

        assert output.strip() == "[1, 2]"
        assert elapsed < 0.6

    def test_same_interpreter_tool_reentry_raises_runtimeerror(self, tmp_path: Path):
        observed_errors: list[str] = []

        def reenter() -> str:
            with pytest.raises(RuntimeError, match="host tool callback") as exc_info:
                interpreter.execute("print('nested')")
            observed_errors.append(str(exc_info.value))
            return "blocked"

        interpreter = SbxBackend(
            config=SbxConfig(name="local-test", exec_timeout=3),
            tools={"reenter": reenter},
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(PAYLOAD_PATH)],
            _staging_root=tmp_path / "staging",
        )
        try:
            output = interpreter.execute("result = await reenter()\nprint(result)")
        finally:
            interpreter.shutdown()

        assert output.strip() == "blocked"
        assert len(observed_errors) == 1

    def test_request_timeout_fires_when_runner_stays_silent(self, tmp_path: Path):
        interpreter = SbxBackend(
            config=SbxConfig(name="silent-test", exec_timeout=0.2),
            preinstall_packages=False,
            _supervisor_command=[
                sys.executable,
                "-u",
                "-c",
                "import sys, time\nsys.stdin.readline()\ntime.sleep(30)\n",
            ],
            _staging_root=tmp_path / "staging",
        )
        start = time.monotonic()
        try:
            with pytest.raises(SandboxFatalError, match="timed out"):
                interpreter.execute("print('never')")
        finally:
            interpreter.shutdown()

        assert time.monotonic() - start < 1.0

    def test_reset_clears_globals_and_staging_root(self, tmp_path: Path):
        interpreter = self.make_interpreter(tmp_path)
        source = tmp_path / "input.txt"
        source.write_text("hello", encoding="utf-8")
        try:
            interpreter.execute("x = 7")
            interpreter.mount_file_at(str(source), "/sandbox/input/source/input.txt")

            interpreter.reset()

            assert interpreter.execute("print('x' in globals())").strip() == "False"
            assert interpreter.list_dir("/sandbox") == []
        finally:
            interpreter.shutdown()


class TestSbxBackendLocalWebSocketRunner:
    def make_interpreter(
        self,
        tmp_path: Path,
        *,
        tools: dict | None = None,
        path: str | None = None,
        url_path: str | None = None,
        startup_timeout: float = 3,
    ) -> SbxBackend:
        port = _free_local_port()
        websocket_path = path or f"/predict-rlm-test-{os.getpid()}-{time.time_ns()}"
        command = [
            sys.executable,
            "-u",
            str(PAYLOAD_PATH),
            "--websocket-host",
            "127.0.0.1",
            "--websocket-port",
            str(port),
            "--websocket-path",
            websocket_path,
            "--websocket-max-message-bytes",
            str(32 * 1024 * 1024),
        ]
        return SbxBackend(
            config=SbxConfig(
                name="local-websocket-test",
                exec_timeout=5,
                websocket_startup_timeout=startup_timeout,
                websocket_max_message_bytes=32 * 1024 * 1024,
            ),
            tools=tools,
            preinstall_packages=False,
            _websocket_supervisor_command=command,
            _websocket_url=f"ws://127.0.0.1:{port}{url_path or websocket_path}",
            _staging_root=tmp_path / "ws-staging",
        )

    def test_websocket_execute_and_state_persistence(self, tmp_path: Path):
        interpreter = self.make_interpreter(tmp_path)
        try:
            assert interpreter.execute("x = 7\nprint(x)") == "7\n"
            assert interpreter.execute("x += 1\nprint(x)") == "8\n"
        finally:
            interpreter.shutdown()

    def test_pydantic_model_variable_injected_as_plain_data(self, tmp_path: Path):
        """A pydantic-model input variable must cross the boundary as plain data.

        Input variables are injected into the sandbox via repr(); a model's repr is a
        constructor call (e.g. RfpAnalysis(...)) referencing a class the sandbox does
        not have, so it would raise NameError. to_plain_data() normalizes the model to
        a dict first. Regression for chained-RLM runs that pass a model output as the
        next RLM's input.
        """
        from pydantic import BaseModel

        class KeyDate(BaseModel):
            name: str

        class RfpAnalysis(BaseModel):
            title: str
            key_dates: list[KeyDate]

        interpreter = self.make_interpreter(tmp_path)
        try:
            output = interpreter.execute(
                "print(type(rfp).__name__, rfp['title'], rfp['key_dates'][0]['name'])",
                variables={"rfp": RfpAnalysis(title="T", key_dates=[KeyDate(name="due")])},
            )
        finally:
            interpreter.shutdown()

        assert output == "dict T due\n"

    def test_websocket_host_tool_round_trip(self, tmp_path: Path):
        def add(a: int, b: int) -> dict:
            return {"total": a + b}

        interpreter = self.make_interpreter(tmp_path, tools={"add": add})
        try:
            output = interpreter.execute("result = await add(2, 3)\nprint(result['total'])")
        finally:
            interpreter.shutdown()

        assert output == "5\n"

    def test_websocket_concurrent_host_tool_timeout_recovers_and_shutdowns(
        self, tmp_path: Path
    ):
        def slow_tool() -> str:
            time.sleep(5)
            return "slow"

        interpreter = self.make_interpreter(tmp_path, tools={"slow_tool": slow_tool})
        shutdown_duration = None
        try:
            timeout_result = interpreter.execute(
                "import asyncio\n"
                "await asyncio.gather(slow_tool(), slow_tool())\n",
                timeout=0.1,
            )
            output = interpreter.execute("print('still alive')")
            shutdown_start = time.perf_counter()
            interpreter.shutdown()
            shutdown_duration = time.perf_counter() - shutdown_start
        finally:
            interpreter.shutdown()

        assert "[Timeout] Iteration execution timed out after 0.1s" in timeout_result
        assert output == "still alive\n"
        assert interpreter._pending_tool_calls == {}
        assert shutdown_duration is not None and shutdown_duration < 2

    def test_websocket_large_host_tool_payload_round_trips(self, tmp_path: Path):
        seen_lengths: list[int] = []

        def predict(signature: str, **kwargs) -> dict:
            seen_lengths.append(len(kwargs["text"]))
            return {"answer": "4"}

        interpreter = self.make_interpreter(tmp_path, tools={"predict": predict})
        try:
            output = interpreter.execute(
                "payload = 'x' * 950000\n"
                "result = await predict('text: str -> answer: str', text=payload)\n"
                "print(result['answer'])"
            )
        finally:
            interpreter.shutdown()

        assert output == "4\n"
        assert seen_lengths == [950000]

    def test_predict_forwards_nested_pydantic_schemas_for_custom_types(self, tmp_path: Path):
        """predict() with a custom output type that nests sibling models.

        The host builds the structured-output signature and can't resolve a
        REPL-defined type by name, so the sandbox extracts model_json_schema()
        and forwards it (on par with JSPI). The model is defined at REPL top
        level but predict() is called from inside a function under gather(), and
        the type nests sibling models -- which can't resolve via the mismatched
        __main__ unless rebuilt against the execution globals. This is the exact
        real-world failure mode; without the fix the host falls back to a plain
        string signature and the predict() call raises.
        """
        received: dict = {}

        def predict(signature: str, **kwargs) -> dict:
            received["schemas"] = kwargs.get("pydantic_schemas")
            return {"analysis": {"page_number": 1, "items": [{"name": "x"}]}}

        interpreter = self.make_interpreter(tmp_path, tools={"predict": predict})
        try:
            output = interpreter.execute(
                "import asyncio\n"
                "from pydantic import BaseModel, Field\n"
                "class PageItem(BaseModel):\n"
                "    name: str\n"
                "class PageAnalysis(BaseModel):\n"
                "    page_number: int\n"
                "    items: list[PageItem] = Field(default_factory=list)\n"
                "async def one(i):\n"
                "    return await predict('doc: str -> analysis: PageAnalysis', doc='hi')\n"
                "await asyncio.gather(*[one(i) for i in range(3)])\n"
                "print('ok')"
            )
        finally:
            interpreter.shutdown()

        assert output == "ok\n"
        schemas = received["schemas"]
        assert schemas is not None and "PageAnalysis" in schemas
        # nested sibling model must be present in the forwarded schema's $defs
        assert "PageItem" in json.dumps(schemas["PageAnalysis"])

    def test_predict_result_supports_attribute_and_subscript_access(self, tmp_path: Path):
        """predict() return must work as ``result.page`` and ``result["page"]``.

        The host returns a plain dict over the wire, which only supports
        subscript -- ``result.page`` would raise ``'dict' object has no
        attribute 'page'``. The sandbox wraps it in a Prediction-like object so
        both forms work, matching the core instructions and the JSPI backend.
        """

        def predict(signature: str, **kwargs) -> dict:
            return {"page": "p1", "items": ["a", "b"]}

        interpreter = self.make_interpreter(tmp_path, tools={"predict": predict})
        try:
            output = interpreter.execute(
                "r = await predict('doc: str -> page: str, items: list[str]', doc='hi')\n"
                "print(r.page, r['page'], r.items, r['items'])"
            )
        finally:
            interpreter.shutdown()

        assert output == "p1 p1 ['a', 'b'] ['a', 'b']\n"

    def test_predict_result_reconstructs_nested_pydantic_instances(self, tmp_path: Path):
        """Custom output types arrive as dicts and are revived to instances.

        The host serializes model instances to dicts for transport. The sandbox
        rebuilds them so nested ``item.name`` attribute access works, matching
        the JSPI backend and the core instructions for Pydantic return values.
        """

        def predict(signature: str, **kwargs) -> dict:
            return {"analysis": {"page_number": 2, "items": [{"name": "x"}, {"name": "y"}]}}

        interpreter = self.make_interpreter(tmp_path, tools={"predict": predict})
        try:
            output = interpreter.execute(
                "from pydantic import BaseModel, Field\n"
                "class PageItem(BaseModel):\n"
                "    name: str\n"
                "class PageAnalysis(BaseModel):\n"
                "    page_number: int\n"
                "    items: list[PageItem] = Field(default_factory=list)\n"
                "r = await predict('doc: str -> analysis: PageAnalysis', doc='hi')\n"
                "print(r.analysis.page_number, [i.name for i in r.analysis.items])"
            )
        finally:
            interpreter.shutdown()

        assert output == "2 ['x', 'y']\n"

    def test_predict_reconstruction_preserves_extra_lm_fields(self, tmp_path: Path):
        """Deno parity: fields the LM returns beyond the declared model survive.

        Reconstruction validates into an ``extra='allow'`` subclass (matching the
        JSPI/Deno backend) so an unexpected field like ``bonus`` is kept and
        attribute-accessible rather than dropped. With a plain (extra='ignore')
        model ``r.item.bonus`` would raise AttributeError.
        """

        def predict(signature: str, **kwargs) -> dict:
            return {"item": {"name": "x", "bonus": "kept"}}

        interpreter = self.make_interpreter(tmp_path, tools={"predict": predict})
        try:
            output = interpreter.execute(
                "from pydantic import BaseModel\n"
                "class Item(BaseModel):\n"
                "    name: str\n"
                "r = await predict('doc: str -> item: Item', doc='hi')\n"
                "print(type(r.item).__name__, r.item.name, r.item.bonus)"
            )
        finally:
            interpreter.shutdown()

        assert output == "Item x kept\n"

    def test_predict_reconstruction_raises_on_invalid_model_output(self, tmp_path: Path):
        """A predict() output the declared model rejects must surface loudly.

        When the host returns data that can't satisfy the model (here: missing the
        required ``name``), reconstruction lets the validation error propagate so the
        caller sees the real cause -- rather than silently leaking a dict and failing
        a step later on attribute access with a misleading ``'dict' object has no
        attribute ...``.
        """

        def predict(signature: str, **kwargs) -> dict:
            return {"item": {}}  # missing required 'name'

        interpreter = self.make_interpreter(tmp_path, tools={"predict": predict})
        try:
            with pytest.raises(SandboxExecutionError) as excinfo:
                interpreter.execute(
                    "from pydantic import BaseModel\n"
                    "class Item(BaseModel):\n"
                    "    name: str\n"
                    "r = await predict('doc: str -> item: Item', doc='hi')\n"
                    "print(r.item.name)"
                )
        finally:
            interpreter.shutdown()

        message = str(excinfo.value)
        assert "validation error" in message.lower()
        assert "'dict' object has no attribute" not in message

    def test_predict_reconstruction_gap_function_local_model_not_resolved(
        self, tmp_path: Path
    ):
        """KNOWN GAP: a predict output model defined inside a function isn't revived.

        Reconstruction resolves the output model from the kernel's module globals only
        (no call-stack walk). A model defined at REPL top level -- every real example --
        resolves fine, but one defined INSIDE a function is not in module globals, so
        the field stays a plain dict and attribute access raises. This is a deliberate,
        documented limitation kept out of the hot path: if it ever shows up it fails
        loudly here (not silently), and the fix would be stack-aware resolution at the
        predict() call site. This test pins the gap so a future change is intentional.
        """

        def predict(signature: str, **kwargs) -> dict:
            return {"item": {"name": "x"}}

        interpreter = self.make_interpreter(tmp_path, tools={"predict": predict})
        try:
            # Item is function-local -> absent from kernel module globals -> not revived,
            # so res.item is a plain dict and res.item.name raises.
            with pytest.raises(SandboxExecutionError) as excinfo:
                interpreter.execute(
                    "async def run():\n"
                    "    from pydantic import BaseModel\n"
                    "    class Item(BaseModel):\n"
                    "        name: str\n"
                    "    res = await predict('doc: str -> item: Item', doc='hi')\n"
                    "    return res.item.name\n"
                    "await run()"
                )
        finally:
            interpreter.shutdown()

        assert "'dict' object has no attribute 'name'" in str(excinfo.value)

    def test_predict_reconstructs_single_model_under_gather(self, tmp_path: Path):
        """Real-world repro: predict() inside a function fanned out via gather().

        Matches the RFP-page pattern: a single ``-> notes: PageRfpNotes`` output,
        the model defined at REPL top level with Optional + list fields, predict()
        called inside an async helper, all 38 fanned out with asyncio.gather, then
        ``res.notes.page`` accessed. Without nested reconstruction this raises
        ``'dict' object has no attribute 'page'``.
        """

        def predict(signature: str, **kwargs) -> dict:
            n = kwargs.get("page_number", 0)
            return {"notes": {"page": n, "page_type": "cover", "title_or_heading": None, "key_facts": ["a"]}}

        interpreter = self.make_interpreter(tmp_path, tools={"predict": predict})
        try:
            output = interpreter.execute(
                "import asyncio\n"
                "from pydantic import BaseModel, Field\n"
                "from typing import Optional\n"
                "class PageRfpNotes(BaseModel):\n"
                "    page: int\n"
                "    page_type: str = Field(description='x')\n"
                "    title_or_heading: Optional[str] = None\n"
                "    key_facts: list[str] = Field(default_factory=list)\n"
                "async def analyze(i):\n"
                "    res = await predict('page_image: dspy.Image, page_number: int, nav_hint: str -> notes: PageRfpNotes', page_number=i+1, nav_hint='h')\n"
                "    return res.notes\n"
                "notes = await asyncio.gather(*[analyze(i) for i in range(38)])\n"
                "print(len(notes), notes[0].page, notes[0].page_type, type(notes[0]).__name__)"
            )
        finally:
            interpreter.shutdown()

        assert output == "38 1 cover PageRfpNotes\n"

    def test_predicts_orphaned_by_gather_failure_do_not_hang_next_execute(
        self, tmp_path: Path
    ):
        """A gather() that raises early orphans its other predict() calls.

        Those tasks are left pending on the kernel loop with tool calls already
        in flight. If the loop is closed without cancelling them, they desync the
        host<->kernel protocol and the *next* predict() hangs to the watchdog.
        The kernel must cancel orphans between executes so the follow-up works.
        """
        async def predict(signature: str, **kwargs) -> dict:
            await asyncio.sleep(0.5)
            return {"a": "ok"}

        interpreter = self.make_interpreter(tmp_path, tools={"predict": predict})
        try:
            # gather raises on boom(); the 6 slow predict() calls get orphaned.
            first = interpreter.execute(
                "import asyncio\n"
                "async def boom(): raise ValueError('expected')\n"
                "async def slow(i):\n"
                "    r = await predict('t: str -> a: str', t='x')\n"
                "    return r['a']\n"
                "try:\n"
                "    await asyncio.gather(boom(), *[slow(i) for i in range(6)])\n"
                "except ValueError:\n"
                "    print('caught')"
            )
            assert first == "caught\n"
            # A fresh predict() on the next execute must not hang.
            second = interpreter.execute(
                "r = await predict('t: str -> a: str', t='y')\nprint(r['a'])"
            )
            assert second == "ok\n"
        finally:
            interpreter.shutdown()

    def test_websocket_reset_and_shutdown(self, tmp_path: Path):
        interpreter = self.make_interpreter(tmp_path)
        try:
            assert interpreter.execute("x = 7\nprint(x)") == "7\n"
            interpreter.reset()
            assert interpreter.execute("print('x' in globals())") == "False\n"
            proc = interpreter._proc
            interpreter.shutdown()
        finally:
            interpreter.shutdown()

        assert proc is not None
        assert proc.poll() is not None

    def test_websocket_auth_path_failure_is_reported(self, tmp_path: Path):
        interpreter = self.make_interpreter(
            tmp_path,
            path="/predict-rlm-good",
            url_path="/predict-rlm-bad",
            startup_timeout=0.5,
        )
        try:
            with pytest.raises(SandboxFatalError, match="Timed out connecting"):
                interpreter.prewarm()
        finally:
            interpreter.shutdown()


class TestSbxBackendInterrupt(TestSbxBackendLocalWebSocketRunner):
    """On-demand execution interrupt + cancellation-safe aexecute (issue #42).

    Drives the real ``_payload.py`` over a real websocket (no Docker) via the
    ``TestSbxBackendLocalWebSocketRunner`` seam.
    """

    @pytest.mark.local
    def test_interrupt_unblocks_long_running_cell(self, tmp_path: Path):
        """Criteria 1 + 3: interrupt from another thread aborts a long sleep
        promptly (not ~120s) and the next execute succeeds with no ConcurrencyError.
        """
        interpreter = self.make_interpreter(tmp_path, startup_timeout=5)
        try:
            running_flag: dict[str, bool] = {}

            def fire_interrupt():
                time.sleep(1.0)
                running_flag["was_running"] = interpreter.interrupt(timeout=10.0)

            thread = threading.Thread(target=fire_interrupt)
            thread.start()
            start = time.monotonic()
            # No timeout= -> relies purely on the interrupt to unblock.
            result = interpreter.execute("import time\ntime.sleep(120)\nprint('done')")
            elapsed = time.monotonic() - start
            thread.join(timeout=5)

            assert elapsed < 30, f"interrupt did not unblock promptly: {elapsed:.1f}s"
            assert running_flag.get("was_running") is True
            assert "done" not in str(result)

            # Next execute must succeed on the warm sandbox (no ConcurrencyError).
            assert interpreter.execute("print('alive')") == "alive\n"
        finally:
            interpreter.shutdown()

    @pytest.mark.local
    def test_interrupt_preserves_warm_state(self, tmp_path: Path):
        """Criterion 2: a variable set before the interrupted cell survives."""
        interpreter = self.make_interpreter(tmp_path, startup_timeout=5)
        try:
            assert interpreter.execute("kept = 99\nprint(kept)") == "99\n"

            def fire_interrupt():
                time.sleep(1.0)
                interpreter.interrupt(timeout=10.0)

            thread = threading.Thread(target=fire_interrupt)
            thread.start()
            interpreter.execute("import time\ntime.sleep(120)\nprint('done')")
            thread.join(timeout=5)

            assert interpreter.execute("print(kept)") == "99\n"
        finally:
            interpreter.shutdown()

    @pytest.mark.local
    def test_interrupt_returns_only_after_cell_releases_gate(self, tmp_path: Path):
        """Regression for the Fractal interrupt-recovery race (#42).

        ``interrupt`` must not return until the interrupted cell has released
        the execution gate -- i.e. the worker blocked in the execute ``recv``
        loop has drained and the interpreter is quiescent. Otherwise the next
        request calls ``recv`` concurrently with the still-draining worker and
        trips a websockets ConcurrencyError.

        Asserting the gate state directly (rather than racing a follow-up
        request) makes the contract deterministic: without the wait-for-drain,
        ``interrupt`` returns right after the ``ws.send`` while the cell is still
        being torn down across several IPC hops, so the gate is still held.
        """
        interpreter = self.make_interpreter(tmp_path, startup_timeout=5)
        gate = interpreter._execution_gate
        try:
            interpreter.execute("warm = 1")

            def run_cell() -> None:
                interpreter.execute("import time\ntime.sleep(120)\nprint('done')")

            worker = threading.Thread(target=run_cell)
            worker.start()
            while not gate.is_running():
                time.sleep(0.01)
            time.sleep(0.5)  # let the kernel actually enter the sleep

            was_running = interpreter.interrupt(timeout=10.0)

            assert was_running is True
            assert (
                gate.is_running() is False
            ), "interrupt returned before the interrupted cell released the gate"

            worker.join(timeout=5)
            assert not worker.is_alive()
            # Warm sandbox + ws are immediately reusable, no ConcurrencyError.
            assert interpreter.execute("print(warm)") == "1\n"
        finally:
            interpreter.shutdown()

    @pytest.mark.local
    def test_interrupt_returns_false_when_idle(self, tmp_path: Path):
        """Criterion 5 (client view): interrupt while idle reports no cell ran."""
        interpreter = self.make_interpreter(tmp_path, startup_timeout=5)
        try:
            interpreter.execute("print('warm')")
            assert interpreter.interrupt(timeout=5.0) is False
        finally:
            interpreter.shutdown()

    @pytest.mark.local
    def test_aexecute_cancellation_is_prompt_and_keeps_sandbox_warm(
        self, tmp_path: Path
    ):
        """Criteria 1 + 4: cancelling aexecute mid-cell unwinds the worker
        promptly (no orphaned to_thread worker) and leaves the ws reusable.
        """
        interpreter = self.make_interpreter(tmp_path, startup_timeout=5)

        async def scenario() -> float:
            interpreter.execute("seed = 5")
            task = asyncio.ensure_future(
                interpreter.aexecute("import time\ntime.sleep(120)\nprint('done')")
            )
            await asyncio.sleep(1.0)
            start = time.monotonic()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return time.monotonic() - start

        try:
            elapsed = asyncio.run(scenario())
            assert elapsed < 30, f"cancellation did not unwind promptly: {elapsed:.1f}s"
            # ws still usable and warm state preserved.
            assert interpreter.execute("print(seed)") == "5\n"
            assert (
                threading.active_count() < 10
            ), "orphaned worker thread(s) survived cancellation"
        finally:
            interpreter.shutdown()


class TestSupervisorPayloadInterruptMethod:
    """Criterion 5: server-side ``interrupt`` JSON-RPC method semantics."""

    @pytest.mark.local
    def test_interrupt_method_acks_running_false_when_idle(self, tmp_path: Path):
        import predict_rlm.backends.supervisor._payload as payload

        payload._consume_interrupt_request()  # clear any prior latch
        result = asyncio.run(
            payload._handle_interrupt_request({"id": 1, "method": "interrupt"})
        )
        # Idle -> ack reports no cell was running (the no-op ack contract).
        assert result["result"]["running"] is False
        payload._consume_interrupt_request()  # don't leak the latch to other tests


class TestSbxBackendLocalSupervisorInterrupt(TestSbxBackendLocalWebSocketRunner):
    """Criterion 5 (in-runner): interrupt method trips the interrupt path."""

    @pytest.mark.local
    def test_interrupt_method_trips_interrupt_path_while_running(self, tmp_path: Path):
        interpreter = self.make_interpreter(tmp_path, startup_timeout=5)
        try:
            interpreter.execute("flag = 1")

            def fire_interrupt():
                time.sleep(1.0)
                interpreter.interrupt(timeout=10.0)

            thread = threading.Thread(target=fire_interrupt)
            thread.start()
            start = time.monotonic()
            interpreter.execute("import time\ntime.sleep(120)")
            elapsed = time.monotonic() - start
            thread.join(timeout=5)
            assert elapsed < 30
            # The runner kernel restored globals from snapshot; flag survives.
            assert interpreter.execute("print(flag)") == "1\n"
        finally:
            interpreter.shutdown()


class TestSbxSupervisorSignalIsolation(TestSbxBackendLocalWebSocketRunner):
    """Regression: a terminal SIGINT (Ctrl-C interrupting an RLM turn) must not
    reach the supervisor subprocess.

    The supervisor is launched without ``start_new_session`` it shares the
    host's process group, so a terminal Ctrl-C is delivered to the Go ``sbx``
    child too. The child cancels its context ("ERROR: context canceled") and
    exits, while Python only sees an ``asyncio.CancelledError`` during the LLM
    phase (no execute in flight) and hands the supervisor back as healthy. The
    next request then fails with "Sbx supervisor exited unexpectedly". The
    out-of-band signal bypasses the in-band #42 interrupt machinery entirely;
    detaching the process group is what keeps Ctrl-C off the child.
    """

    @pytest.mark.local
    def test_supervisor_runs_in_its_own_process_group(self, tmp_path: Path):
        interpreter = self.make_interpreter(tmp_path)
        try:
            interpreter.execute("x = 1")
            proc = interpreter._proc
            assert proc is not None and proc.poll() is None
            assert os.getpgid(proc.pid) != os.getpgid(0)
        finally:
            interpreter.shutdown()


class TestSbxCommandConstruction:
    def test_default_template_uses_explicit_non_docker_shell_template(
        self, monkeypatch, tmp_path: Path
    ):
        commands: list[list[str]] = []

        def fake_run(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="created-name\n", stderr="")

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/sbx")
        monkeypatch.setattr(subprocess, "run", fake_run)
        interpreter = SbxBackend(
            config=SbxConfig(name="created-name"),
            preinstall_packages=False,
            _staging_root=tmp_path / "staging",
        )

        interpreter._start_sbx_and_build_supervisor_command()

        create_cmd = commands[0]
        assert SbxConfig().template == DEFAULT_SBX_TEMPLATE
        assert create_cmd[create_cmd.index("--template") + 1] == DEFAULT_SBX_TEMPLATE

    def test_custom_template_overrides_default(self, monkeypatch, tmp_path: Path):
        commands: list[list[str]] = []

        def fake_run(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="created-name\n", stderr="")

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/sbx")
        monkeypatch.setattr(subprocess, "run", fake_run)
        interpreter = SbxBackend(
            config=SbxConfig(
                name="created-name",
                template="docker.io/example/custom-template:latest",
            ),
            preinstall_packages=False,
            _staging_root=tmp_path / "staging",
        )

        interpreter._start_sbx_and_build_supervisor_command()

        create_cmd = commands[0]
        assert create_cmd[create_cmd.index("--template") + 1] == (
            "docker.io/example/custom-template:latest"
        )

    def test_none_template_omits_template_flag(self, monkeypatch, tmp_path: Path):
        commands: list[list[str]] = []

        def fake_run(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="created-name\n", stderr="")

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/sbx")
        monkeypatch.setattr(subprocess, "run", fake_run)
        interpreter = SbxBackend(
            config=SbxConfig(name="created-name", template=None),
            preinstall_packages=False,
            _staging_root=tmp_path / "staging",
        )

        interpreter._start_sbx_and_build_supervisor_command()

        assert "--template" not in commands[0]

    def test_persist_skips_cleanup_but_is_not_create_flag(self, monkeypatch, tmp_path: Path):
        commands: list[list[str]] = []

        def fake_run(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="created-name\n", stderr="")

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/sbx")
        monkeypatch.setattr(subprocess, "run", fake_run)
        interpreter = SbxBackend(
            config=SbxConfig(name="created-name", persist=True),
            preinstall_packages=False,
            _staging_root=tmp_path / "staging",
        )

        command = interpreter._start_sbx_and_build_supervisor_command()
        interpreter.shutdown()

        create_cmd = commands[0]
        assert create_cmd[:3] == ["sbx", "create", "shell"]
        assert "--persist" not in create_cmd
        assert command[:4] == ["sbx", "exec", "-i", "-w"]
        assert not any(cmd[:2] == ["sbx", "rm"] for cmd in commands)

    def test_websocket_supervisor_starts_foreground_sentinel_and_publishes_port(
        self, monkeypatch, tmp_path: Path
    ):
        run_commands: list[list[str]] = []
        popen_commands: list[list[str]] = []

        class FakeProcess:
            stdout = None
            stderr = None
            stdin = None
            pid = 12345

            def poll(self):
                return None

        def fake_run(command, **kwargs):
            run_commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="created-name\n", stderr="")

        def fake_popen(command, **kwargs):
            popen_commands.append(command)
            return FakeProcess()

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/sbx")
        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        interpreter = SbxBackend(
            config=SbxConfig(
                name="created-name",
                websocket_port=8766,
                websocket_max_message_bytes=32 * 1024 * 1024,
            ),
            preinstall_packages=False,
            _staging_root=tmp_path / "staging",
        )
        monkeypatch.setattr(
            interpreter,
            "_publish_websocket_port",
            lambda: "ws://127.0.0.1:49152/test",
        )

        interpreter._start_sbx_websocket_supervisor()

        assert len(popen_commands) == 1
        supervisor_exec = popen_commands[0]
        assert supervisor_exec[:4] == ["sbx", "exec", "-w", str(tmp_path / "staging")]
        assert "-d" not in supervisor_exec
        assert "-i" not in supervisor_exec
        assert "--websocket-host" in supervisor_exec
        assert supervisor_exec[supervisor_exec.index("--websocket-port") + 1] == "8766"
        assert supervisor_exec[supervisor_exec.index("--websocket-max-message-bytes") + 1] == str(
            32 * 1024 * 1024
        )
        assert interpreter._proc is not None
        assert not any(cmd[:3] == ["sbx", "exec", "-d"] for cmd in run_commands)

    def test_websocket_recovery_restarts_detached_supervisor_after_kill(
        self, monkeypatch, tmp_path: Path
    ):
        commands: list[list[str]] = []

        def fake_run(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        interpreter = SbxBackend(
            config=SbxConfig(name="created-name"),
            preinstall_packages=False,
            _staging_root=tmp_path / "staging",
        )
        interpreter._sandbox_name = "created-name"
        interpreter._prepared_supervisor_path = tmp_path / "staging" / ".predict_rlm_supervisor" / "_payload.py"
        interpreter._websocket_url = "ws://127.0.0.1:49152/predict-rlm/old"
        interpreter._published_websocket_url = interpreter._websocket_url

        started: list[bool] = []
        connected: list[str] = []

        def fake_start_sbx_websocket_supervisor():
            started.append(True)
            interpreter._websocket_url = "ws://127.0.0.1:49153/predict-rlm/new"

        def fake_connect_websocket_supervisor(url: str):
            connected.append(url)
            interpreter._ws = object()

        monkeypatch.setattr(
            interpreter,
            "_start_sbx_websocket_supervisor",
            fake_start_sbx_websocket_supervisor,
        )
        monkeypatch.setattr(
            interpreter,
            "_connect_websocket_supervisor",
            fake_connect_websocket_supervisor,
        )

        interpreter._kill_websocket_supervisor()
        interpreter._ensure_websocket_supervisor()

        assert interpreter._published_websocket_url is None
        assert started == [True]
        assert connected == ["ws://127.0.0.1:49153/predict-rlm/new"]
        assert any(
            cmd[:5] == ["sbx", "exec", "-w", str(tmp_path / "staging"), "created-name"]
            for cmd in commands
        )

    def test_published_websocket_endpoint_parses_localhost_port(self, tmp_path: Path):
        interpreter = SbxBackend(
            preinstall_packages=False,
            _staging_root=tmp_path / "staging",
        )
        interpreter._websocket_path = "/predict-rlm/token"

        assert (
            interpreter._parse_published_websocket_endpoint(
                "Published 8765/tcp to localhost:49152\n"
            )
            == "ws://localhost:49152/predict-rlm/token"
        )
        assert (
            interpreter._parse_published_websocket_endpoint("http://127.0.0.1:49153")
            == "ws://127.0.0.1:49153/predict-rlm/token"
        )

    def test_published_websocket_endpoint_parse_failure_is_fatal(self, tmp_path: Path):
        interpreter = SbxBackend(
            preinstall_packages=False,
            _staging_root=tmp_path / "staging",
        )

        with pytest.raises(SandboxFatalError, match="published WebSocket endpoint"):
            interpreter._parse_published_websocket_endpoint("no ports here")

    def test_shutdown_forces_sbx_removal_without_confirmation(self, monkeypatch, tmp_path: Path):
        commands: list[list[str]] = []

        def fake_run(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        interpreter = SbxBackend(
            config=SbxConfig(name="created-name"),
            preinstall_packages=False,
            _staging_root=tmp_path / "staging",
        )
        interpreter._sandbox_name = "created-name"

        interpreter.shutdown()

        assert ["sbx", "rm", "--force", "created-name"] in commands

    def test_workspace_flags_include_read_only_primary_and_extra_workspaces(
        self, monkeypatch, tmp_path: Path
    ):
        commands: list[list[str]] = []
        extra_one = tmp_path / "extra-one"
        extra_two = tmp_path / "extra-two"

        def fake_run(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="created-name\n", stderr="")

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/sbx")
        monkeypatch.setattr(subprocess, "run", fake_run)
        interpreter = SbxBackend(
            config=SbxConfig(
                name="created-name",
                workspace_read_only=True,
                extra_workspaces=[str(extra_one), str(extra_two)],
            ),
            preinstall_packages=False,
            _staging_root=tmp_path / "staging",
        )

        interpreter._start_sbx_and_build_supervisor_command()

        create_cmd = commands[0]
        workspace_arg = f"{tmp_path / 'staging'}:ro"
        assert create_cmd[:4] == ["sbx", "create", "shell", workspace_arg]
        assert create_cmd[4:6] == [str(extra_one), str(extra_two)]

    def test_default_workspace_is_staging_root_not_repo(self, monkeypatch, tmp_path: Path):
        commands: list[list[str]] = []

        def fake_run(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="created-name\n", stderr="")

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/sbx")
        monkeypatch.setattr(subprocess, "run", fake_run)
        interpreter = SbxBackend(
            config=SbxConfig(name="created-name"),
            preinstall_packages=False,
            _staging_root=tmp_path / "staging",
        )

        command = interpreter._start_sbx_and_build_supervisor_command()

        create_cmd = commands[0]
        assert create_cmd[:4] == ["sbx", "create", "shell", str(tmp_path / "staging")]
        assert str(Path.cwd()) not in create_cmd
        assert command[:5] == ["sbx", "exec", "-i", "-w", str(tmp_path / "staging")]

    def test_supervisor_command_uses_python3_executable(self, monkeypatch, tmp_path: Path):
        def fake_run(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, stdout="created-name\n", stderr="")

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/sbx")
        monkeypatch.setattr(subprocess, "run", fake_run)
        interpreter = SbxBackend(
            config=SbxConfig(name="created-name"),
            preinstall_packages=False,
            _staging_root=tmp_path / "staging",
        )

        command = interpreter._start_sbx_and_build_supervisor_command()

        assert "python" not in command
        supervisor_path = tmp_path / "staging" / ".predict_rlm_supervisor" / "_payload.py"
        assert supervisor_path.read_text(encoding="utf-8") == PAYLOAD_PATH.read_text(
            encoding="utf-8"
        )
        assert command[-3:] == ["python3", "-u", str(supervisor_path)]

    def test_runner_restart_reuses_existing_sandbox(
        self, monkeypatch, tmp_path: Path
    ):
        commands: list[list[str]] = []

        def fake_run(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="created-name\n", stderr="")

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/sbx")
        monkeypatch.setattr(subprocess, "run", fake_run)
        interpreter = SbxBackend(
            config=SbxConfig(name="created-name"),
            preinstall_packages=False,
            _staging_root=tmp_path / "staging",
        )
        interpreter._sandbox_name = "created-name"

        command = interpreter._start_sbx_and_build_supervisor_command()

        assert commands == []
        assert command[:5] == ["sbx", "exec", "-i", "-w", str(tmp_path / "staging")]
        assert "created-name" in command

    def test_package_bootstrap_failure_raises_context(self, monkeypatch, tmp_path: Path):
        def fake_run(command, **kwargs):
            return subprocess.CompletedProcess(
                command,
                17,
                stdout="download started",
                stderr="no matching distribution",
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        interpreter = SbxBackend(
            config=SbxConfig(exec_timeout=1),
            skill_packages=["missing-package"],
            preinstall_packages=False,
            _staging_root=tmp_path / "staging",
        )
        interpreter._sandbox_name = "created-name"

        with pytest.raises(SandboxFatalError, match="missing-package"):
            interpreter._bootstrap_packages()

    def test_package_bootstrap_uses_docker_sandbox_safe_pip(self, monkeypatch, tmp_path: Path):
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        interpreter = SbxBackend(
            preinstall_packages=True,
            _staging_root=tmp_path / "staging",
        )
        interpreter._sandbox_name = "created-name"

        interpreter._bootstrap_packages()

        assert commands == [
            [
                "sbx",
                "exec",
                "-w",
                str(tmp_path / "staging"),
                "created-name",
                "python3",
                "-m",
                "pip",
                "install",
                "--break-system-packages",
                "websockets",
                "pydantic",
                "pandas",
            ]
        ]


class TestSbxPool:
    def test_start_prewarms_interpreters_concurrently(self, tmp_path: Path, monkeypatch):
        pool = SbxPool(
            size=2,
            config=SbxConfig(name="pool-test"),
            preinstall_packages=False,
            _staging_root=tmp_path / "pool",
        )
        barrier = threading.Barrier(2)
        active = 0
        max_active = 0
        active_lock = threading.Lock()
        prewarmed_indexes: list[int] = []

        class FakeInterpreter:
            def __init__(self, index: int) -> None:
                self.index = index
                self.shutdown_called = False

            def prewarm(self) -> None:
                nonlocal active, max_active
                with active_lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    barrier.wait(timeout=1)
                    prewarmed_indexes.append(self.index)
                finally:
                    with active_lock:
                        active -= 1

            def shutdown(self) -> None:
                self.shutdown_called = True

        monkeypatch.setattr(
            pool,
            "_create_interpreter",
            lambda index: FakeInterpreter(index),
        )

        try:
            pool.start()

            assert max_active == 2
            assert prewarmed_indexes == [1, 0] or prewarmed_indexes == [0, 1]
            assert [interpreter.index for interpreter in pool._all_interpreters] == [0, 1]
            assert pool._started
            assert pool._available.qsize() == 2
        finally:
            pool.shutdown()

    def test_lease_logging_overrides_do_not_reconfigure_pool(
        self, tmp_path: Path, monkeypatch
    ):
        pool = SbxPool(
            size=2,
            config=SbxConfig(name="pool-test"),
            preinstall_packages=False,
            _staging_root=tmp_path / "pool",
        )
        created = []

        class FakeInterpreter:
            def __init__(self, index: int) -> None:
                self.index = index
                self.configure_debug_calls: list[bool] = []
                self.configure_verbose_calls: list[bool] = []
                self.runtime_calls: list[dict] = []

            def prewarm(self) -> None:
                return None

            def configure_debug(self, enabled: bool) -> None:
                self.configure_debug_calls.append(enabled)

            def configure_verbose(self, enabled: bool) -> None:
                self.configure_verbose_calls.append(enabled)

            def configure_runtime(self, **kwargs) -> None:
                self.runtime_calls.append(kwargs)

            def reset(self) -> None:
                return None

            def shutdown(self) -> None:
                return None

        def create_interpreter(index: int) -> FakeInterpreter:
            interpreter = FakeInterpreter(index)
            created.append(interpreter)
            return interpreter

        monkeypatch.setattr(pool, "_create_interpreter", create_interpreter)

        try:
            pool.start()

            with pool.lease(debug=True, verbose=True) as interpreter:
                assert interpreter is created[0]

            assert created[0].runtime_calls[-1]["debug"] is True
            assert created[0].runtime_calls[-1]["verbose"] is True
            assert created[0].configure_debug_calls == []
            assert created[0].configure_verbose_calls == []
            assert created[1].configure_debug_calls == []
            assert created[1].configure_verbose_calls == []
            assert pool.debug is False
            assert pool.verbose is False
            assert pool._interpreter_kwargs["debug"] is False
            assert pool._interpreter_kwargs["verbose"] is False

            with pool.lease() as interpreter:
                assert interpreter is created[1]
            with pool.lease() as interpreter:
                assert interpreter is created[0]

            assert created[0].runtime_calls[-1]["debug"] is False
            assert created[0].runtime_calls[-1]["verbose"] is False
        finally:
            pool.shutdown()

    def test_start_failure_shuts_down_created_interpreters_and_leaves_pool_stopped(
        self, tmp_path: Path, monkeypatch
    ):
        pool = SbxPool(
            size=3,
            config=SbxConfig(name="pool-test"),
            preinstall_packages=False,
            _staging_root=tmp_path / "pool",
        )
        created = []

        class FakeInterpreter:
            def __init__(self, index: int) -> None:
                self.index = index
                self.shutdown_called = False

            def prewarm(self) -> None:
                if self.index == 1:
                    raise RuntimeError("prewarm failed")

            def shutdown(self) -> None:
                self.shutdown_called = True

        def create_interpreter(index: int) -> FakeInterpreter:
            interpreter = FakeInterpreter(index)
            created.append(interpreter)
            return interpreter

        monkeypatch.setattr(pool, "_create_interpreter", create_interpreter)

        with pytest.raises(RuntimeError, match="prewarm failed"):
            pool.start()

        assert created
        assert all(interpreter.shutdown_called for interpreter in created)
        assert not pool._started
        assert pool._all_interpreters == []
        assert pool._available.qsize() == 0

    def test_shutdown_runs_concurrently_and_attempts_all_interpreters(
        self, tmp_path: Path, monkeypatch
    ):
        pool = SbxPool(
            size=3,
            config=SbxConfig(name="pool-test"),
            preinstall_packages=False,
            _staging_root=tmp_path / "pool",
        )
        barrier = threading.Barrier(3)
        active = 0
        max_active = 0
        active_lock = threading.Lock()
        shutdown_indexes: list[int] = []

        class FakeInterpreter:
            def __init__(self, index: int) -> None:
                self.index = index

            def prewarm(self) -> None:
                return None

            def shutdown(self) -> None:
                nonlocal active, max_active
                with active_lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    barrier.wait(timeout=1)
                    shutdown_indexes.append(self.index)
                    if self.index == 1:
                        raise RuntimeError("shutdown failed")
                finally:
                    with active_lock:
                        active -= 1

        monkeypatch.setattr(pool, "_create_interpreter", lambda index: FakeInterpreter(index))
        pool.start()

        with pytest.raises(RuntimeError, match="shutdown failed"):
            pool.shutdown()

        assert max_active == 3
        assert sorted(shutdown_indexes) == [0, 1, 2]
        assert not pool._started
        assert pool._shutdown
        assert pool._all_interpreters == []
        assert pool._available.qsize() == 0

    def test_pool_prewarms_all_interpreters(self, tmp_path: Path):
        pool = SbxPool(
            size=2,
            config=SbxConfig(name="pool-test"),
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(PAYLOAD_PATH)],
            _staging_root=tmp_path / "pool",
        )

        try:
            pool.start()

            assert len(pool._all_interpreters) == 2
            assert all(interpreter._proc is not None for interpreter in pool._all_interpreters)
            assert [interpreter.config.name for interpreter in pool._all_interpreters] == [
                "pool-test-0",
                "pool-test-1",
            ]
        finally:
            pool.shutdown()

    def test_pool_assigns_unique_names_when_config_name_is_omitted(self, tmp_path: Path):
        pool = SbxPool(
            size=2,
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(PAYLOAD_PATH)],
            _staging_root=tmp_path / "pool",
        )

        try:
            pool.start()

            names = [interpreter.config.name for interpreter in pool._all_interpreters]
            assert names == [
                f"{pool._pool_name_prefix}-0",
                f"{pool._pool_name_prefix}-1",
            ]
            assert len(set(names)) == 2
        finally:
            pool.shutdown()

    def test_lease_is_exclusive_and_release_resets(self, tmp_path: Path):
        pool = SbxPool(
            size=1,
            config=SbxConfig(name="pool-test"),
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(PAYLOAD_PATH)],
            _staging_root=tmp_path / "pool",
        )
        acquired = threading.Event()
        released = threading.Event()

        def second_lease() -> None:
            with pool.lease() as interpreter:
                acquired.set()
                assert interpreter.execute("print('x' in globals())").strip() == "False"

        try:
            pool.start()
            with pool.lease() as interpreter:
                interpreter.execute("x = 7")
                staged = pool._all_interpreters[0]._host_path_for_virtual_path(
                    "/sandbox/output/value.txt"
                )
                staged.parent.mkdir(parents=True, exist_ok=True)
                staged.write_text("leaked", encoding="utf-8")
                thread = threading.Thread(target=second_lease)
                thread.start()
                assert not acquired.wait(0.2)

            released.set()
            thread.join(timeout=5)

            assert released.is_set()
            assert acquired.is_set()
            with pool.lease() as interpreter:
                assert interpreter.list_dir("/sandbox") == []
        finally:
            pool.shutdown()

    def test_shutdown_unblocks_waiting_lease_and_does_not_return_interpreter(
        self, tmp_path: Path, monkeypatch
    ):
        pool = SbxPool(
            size=1,
            config=SbxConfig(name="pool-test"),
            preinstall_packages=False,
            _staging_root=tmp_path / "pool",
        )

        class FakeInterpreter:
            def __init__(self) -> None:
                self.reset_called = False
                self.shutdown_called = False

            def prewarm(self) -> None:
                return None

            def configure_runtime(self, **kwargs) -> None:
                return None

            def reset(self) -> None:
                self.reset_called = True

            def shutdown(self) -> None:
                self.shutdown_called = True

        interpreter = FakeInterpreter()
        monkeypatch.setattr(pool, "_create_interpreter", lambda index: interpreter)

        errors: list[str] = []

        def waiting_lease() -> None:
            try:
                with pool.lease():
                    errors.append("acquired")
            except RuntimeError as exc:
                errors.append(str(exc))

        pool.start()
        with pool.lease():
            thread = threading.Thread(target=waiting_lease)
            thread.start()
            time.sleep(0.1)

            pool.shutdown()
            thread.join(timeout=2)

            assert not thread.is_alive()
            assert errors == ["SbxPool is shut down"]
            assert pool._available.qsize() == 0
            assert pool._all_interpreters == []

        assert interpreter.shutdown_called
        assert not interpreter.reset_called
        assert pool._available.qsize() == 0

    def test_shutdown_requested_during_start_prevents_waiting_lease_acquire(
        self, tmp_path: Path, monkeypatch
    ):
        pool = SbxPool(
            size=1,
            config=SbxConfig(name="pool-test"),
            preinstall_packages=False,
            _staging_root=tmp_path / "pool",
        )
        prewarm_started = threading.Event()
        allow_prewarm = threading.Event()

        class FakeInterpreter:
            def __init__(self) -> None:
                self.shutdown_called = False

            def prewarm(self) -> None:
                prewarm_started.set()
                assert allow_prewarm.wait(timeout=2)

            def configure_runtime(self, **kwargs) -> None:
                return None

            def reset(self) -> None:
                return None

            def shutdown(self) -> None:
                self.shutdown_called = True

        interpreter = FakeInterpreter()
        monkeypatch.setattr(pool, "_create_interpreter", lambda index: interpreter)

        lease_results: list[str] = []

        def lease_during_start() -> None:
            try:
                with pool.lease():
                    lease_results.append("acquired")
            except RuntimeError as exc:
                lease_results.append(str(exc))

        lease_thread = threading.Thread(target=lease_during_start)
        lease_thread.start()
        assert prewarm_started.wait(timeout=2)

        shutdown_thread = threading.Thread(target=pool.shutdown)
        shutdown_thread.start()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with pool._state_changed:
                if pool._shutdown_requested:
                    break
            time.sleep(0.01)
        else:
            pytest.fail("shutdown did not request pool stop while startup was active")
        allow_prewarm.set()

        lease_thread.join(timeout=2)
        shutdown_thread.join(timeout=2)

        assert not lease_thread.is_alive()
        assert not shutdown_thread.is_alive()
        assert lease_results == ["SbxPool is shut down"]
        assert interpreter.shutdown_called
        assert pool._available.qsize() == 0

    def test_shutdown_before_lease_autostart_prevents_acquire(
        self, tmp_path: Path, monkeypatch
    ):
        pool = SbxPool(
            size=1,
            config=SbxConfig(name="pool-test"),
            preinstall_packages=False,
            _staging_root=tmp_path / "pool",
        )

        class FakeInterpreter:
            def prewarm(self) -> None:
                return None

            def configure_runtime(self, **kwargs) -> None:
                return None

            def reset(self) -> None:
                return None

            def shutdown(self) -> None:
                return None

        monkeypatch.setattr(pool, "_create_interpreter", lambda index: FakeInterpreter())
        original_begin_start = pool._begin_start
        begin_start_entered = threading.Event()
        allow_begin_start = threading.Event()
        lease_results: list[str] = []

        def delayed_begin_start(*, allow_restart: bool) -> bool:
            assert not allow_restart
            begin_start_entered.set()
            assert allow_begin_start.wait(timeout=2)
            return original_begin_start(allow_restart=allow_restart)

        monkeypatch.setattr(pool, "_begin_start", delayed_begin_start)

        def lease_during_shutdown() -> None:
            try:
                with pool.lease():
                    lease_results.append("acquired")
            except RuntimeError as exc:
                lease_results.append(str(exc))

        lease_thread = threading.Thread(target=lease_during_shutdown)
        lease_thread.start()
        assert begin_start_entered.wait(timeout=2)

        pool.shutdown()
        allow_begin_start.set()
        lease_thread.join(timeout=2)

        assert not lease_thread.is_alive()
        assert lease_results == ["SbxPool is shut down"]
        assert pool._available.qsize() == 0
        assert pool._all_interpreters == []

    def test_lease_after_shutdown_raises_until_explicit_restart(
        self, tmp_path: Path, monkeypatch
    ):
        pool = SbxPool(
            size=1,
            config=SbxConfig(name="pool-test"),
            preinstall_packages=False,
            _staging_root=tmp_path / "pool",
        )

        class FakeInterpreter:
            def __init__(self, index: int) -> None:
                self.index = index

            def prewarm(self) -> None:
                return None

            def configure_runtime(self, **kwargs) -> None:
                return None

            def reset(self) -> None:
                return None

            def shutdown(self) -> None:
                return None

        created: list[FakeInterpreter] = []

        def create_interpreter(index: int) -> FakeInterpreter:
            interpreter = FakeInterpreter(index)
            created.append(interpreter)
            return interpreter

        monkeypatch.setattr(pool, "_create_interpreter", create_interpreter)

        pool.start()
        pool.shutdown()

        with pytest.raises(RuntimeError, match="SbxPool is shut down"):
            with pool.lease():
                pass

        pool.start()
        try:
            with pool.lease() as interpreter:
                assert interpreter is created[-1]
        finally:
            pool.shutdown()


@pytest.mark.sbx
@pytest.mark.integration
@pytest.mark.skipif(
    not _real_sbx_available(),
    reason="real Docker Sandboxes tests require PREDICT_RLM_RUN_SBX_TESTS=1, sbx CLI, and sbx login",
)
class TestSbxBackendRealSbx:
    def test_real_sbx_executes_basic_python(self):
        interpreter = SbxBackend(
            config=SbxConfig(name=f"predict-rlm-test-{os.getpid()}"),
            preinstall_packages=False,
        )
        try:
            output = interpreter.execute("print(2 + 3)")
        finally:
            interpreter.shutdown()

        assert output.strip() == "5"

    def test_real_sbx_predict_reconstructs_pydantic_output_under_gather(self):
        """End-to-end repro of the RFP-page failure in a real sbx sandbox.

        The host serializes a custom output model to a dict for transport; the
        sandbox must revive it to a real instance so ``res.insight.page`` works.
        This runs reconstruction inside the actual sandbox (its own pip-installed
        pydantic) under asyncio.gather -- the exact path that was returning a bare
        dict and raising ``'dict' object has no attribute 'page'``. preinstall is
        required so pydantic exists in the sandbox.
        """

        def predict(signature: str, **kwargs) -> dict:
            n = kwargs.get("page_number", 1)
            return {
                "insight": {
                    "page": n,
                    "title_or_section": "T",
                    "purpose": "p",
                    "key_facts": ["a"],
                    "proposal_requirements": [],
                }
            }

        interpreter = SbxBackend(
            config=SbxConfig(name=f"predict-rlm-test-predict-{os.getpid()}"),
            tools={"predict": predict},
            preinstall_packages=True,
        )
        try:
            output = interpreter.execute(
                "import asyncio\n"
                "from pydantic import BaseModel\n"
                "from typing import Optional\n"
                "class PageInsight(BaseModel):\n"
                "    page: int\n"
                "    title_or_section: Optional[str] = None\n"
                "    purpose: str\n"
                "    key_facts: list[str] = []\n"
                "    proposal_requirements: list[str] = []\n"
                "async def inspect_page(i):\n"
                "    res = await predict('page: dspy.Image, page_number: int -> insight: PageInsight', page_number=i+1)\n"
                "    return res.insight\n"
                "results = await asyncio.gather(*[inspect_page(i) for i in range(9)])\n"
                # Nested reconstructed values are REAL Pydantic instances: attribute
                # access (not subscript), isinstance, and model_dump all work.
                "print(len(results), type(results[0]).__name__, results[0].page,"
                " isinstance(results[0], PageInsight), results[0].model_dump()['purpose'])",
                timeout=30,
            )
        finally:
            interpreter.shutdown()

        assert output.strip() == "9 PageInsight 1 True p"

    def test_real_sbx_timeout_is_recoverable_and_runner_survives(self):
        interpreter = SbxBackend(
            config=SbxConfig(name=f"predict-rlm-test-timeout-{os.getpid()}"),
            preinstall_packages=False,
        )
        try:
            timeout_result = interpreter.execute(
                "import sys\n"
                "print('before timeout')\n"
                "print('stderr before timeout', file=sys.stderr)\n"
                "partial_timeout_state = 41\n"
                "while True:\n"
                "    pass\n",
                timeout=0.2,
            )
            followup = interpreter.execute(
                "print('partial_timeout_state' in globals())\nprint('still alive')"
            )
        finally:
            interpreter.shutdown()

        assert "[Timeout] Iteration execution timed out after 0.2s" in timeout_result
        assert "[stdout]\nbefore timeout" in timeout_result
        assert "[stderr]\nstderr before timeout" in timeout_result
        assert followup.strip() == "True\nstill alive"

    def test_predict_rlm_lm_selected_timeout_recovers_and_continues(self):
        from predict_rlm import PredictRLM

        actions = SequentialActions(
            SimpleNamespace(
                reasoning="select a short timeout for a risky loop",
                code=(
                    "import sys\n"
                    "print('before rlm timeout')\n"
                    "print('stderr before rlm timeout', file=sys.stderr)\n"
                    "while True:\n"
                    "    pass\n"
                ),
                execution_timeout_seconds=0.2,
            ),
            SimpleNamespace(
                reasoning="continue after the timeout observation",
                code="SUBMIT(answer='continued after timeout')",
            ),
        )
        pool = SbxPool(
            size=1,
            config=SbxConfig(name=f"predict-rlm-test-rlm-timeout-{os.getpid()}"),
            preinstall_packages=False,
        )
        rlm = PredictRLM(
            "prompt -> answer",
            max_iterations=2,
            sandbox_backend="sbx",
            sbx_pool=pool,
        )
        rlm.generate_action = actions
        try:
            prediction = rlm(prompt="exercise per-iteration timeout")
        finally:
            pool.shutdown()

        assert prediction.answer == "continued after timeout"
        assert [call["iteration"] for call in actions.calls] == ["1/2", "2/2"]
        assert len(prediction.trace.steps) == 2
        timeout_step, final_step = prediction.trace.steps
        assert (
            "[Timeout] Iteration execution timed out after 0.2s"
            in timeout_step.untruncated_output
        )
        assert "[stdout]\nbefore rlm timeout" in timeout_step.untruncated_output
        assert "[stderr]\nstderr before rlm timeout" in timeout_step.untruncated_output
        assert final_step.output == "FINAL: {'answer': 'continued after timeout'}"

    def test_predict_rlm_can_use_predict_after_lm_selected_timeout(self):
        from predict_rlm import PredictRLM

        actions = SequentialActions(
            SimpleNamespace(
                reasoning="call predict before the risky loop",
                code=(
                    "first = await predict('question: str -> answer: str', "
                    "question='first call')\n"
                    "print('first predict:', first['answer'])\n"
                    "while True:\n"
                    "    pass\n"
                ),
                execution_timeout_seconds=0.2,
            ),
            SimpleNamespace(
                reasoning="call predict again after timeout recovery",
                code=(
                    "second = await predict('question: str -> answer: str', "
                    "question='second call')\n"
                    "SUBMIT(answer=second['answer'])"
                ),
            ),
        )
        class PredictionStub:
            def __init__(self, answer: str) -> None:
                self.answer = answer

            def keys(self) -> list[str]:
                return ["answer"]

            def __getitem__(self, key: str) -> str:
                return getattr(self, key)

        mock_lm = MagicMock()
        mock_predictor = MagicMock()
        mock_predictor.acall = AsyncMock(side_effect=[
            PredictionStub("pre-timeout prediction"),
            PredictionStub("post-timeout prediction"),
        ])
        pool = SbxPool(
            size=1,
            config=SbxConfig(name=f"predict-rlm-test-predict-timeout-{os.getpid()}", exec_timeout=12.0),
            preinstall_packages=False,
        )
        rlm = PredictRLM(
            "prompt -> answer",
            sub_lm=mock_lm,
            max_iterations=2,
            sandbox_backend="sbx",
            sbx_pool=pool,
        )
        rlm.generate_action = actions
        try:
            with patch("predict_rlm.predict_rlm.dspy.Predict", return_value=mock_predictor):
                prediction = rlm(prompt="exercise predict across timeout recovery")
        finally:
            pool.shutdown()

        assert prediction.answer == "post-timeout prediction"
        assert mock_predictor.acall.await_count == 2
        assert [call.kwargs["question"] for call in mock_predictor.acall.await_args_list] == [
            "first call",
            "second call",
        ]
        timeout_step, final_step = prediction.trace.steps
        assert "[Timeout] Iteration execution timed out after 0.2s" in timeout_step.untruncated_output
        assert "first predict: pre-timeout prediction" in timeout_step.untruncated_output
        assert final_step.output == "FINAL: {'answer': 'post-timeout prediction'}"

    def test_predict_rlm_recovers_after_user_exceptions_and_tools_still_work(self):
        pool = SbxPool(
            size=1,
            config=SbxConfig(name=f"predict-rlm-test-user-exceptions-{os.getpid()}", exec_timeout=12.0),
            preinstall_packages=False,
        )
        try:
            assert_predict_rlm_recovers_after_user_exceptions_and_tools_still_work(pool)
        finally:
            pool.shutdown()


class TestSbxBackendReattachConfig:
    """Config surface for reusable/persistent sandboxes (issue #41)."""

    def test_reuse_requires_name(self):
        with pytest.raises(Exception):
            SbxConfig(reuse=True)

    def test_reuse_implies_persist_and_no_remove(self):
        config = SbxConfig(name="hot-box", reuse=True)
        assert config.reuse is True
        assert config.persist is True
        assert config.remove_on_shutdown is False

    def test_reuse_false_is_unchanged_default(self):
        config = SbxConfig()
        assert config.reuse is False
        assert config.persist is False
        assert config.remove_on_shutdown is True
        assert config.stop_on_shutdown is False


class TestSbxBackendReattachStagingRoot:
    """Deterministic staging root tied to the sandbox name (issue #41)."""

    def test_reuse_staging_root_is_deterministic_from_name(self, tmp_path: Path):
        with patch(
            "predict_rlm.backends.sbx.backend.Path.cwd", return_value=tmp_path
        ):
            backend_a = SbxBackend(config=SbxConfig(name="hot-box", reuse=True))
            backend_b = SbxBackend(config=SbxConfig(name="hot-box", reuse=True))
        assert backend_a._staging_root == backend_b._staging_root
        assert backend_a._staging_root.name == "hot-box"

    def test_reuse_staging_root_not_marked_for_cleanup(self, tmp_path: Path):
        from predict_rlm.backends.sbx import backend as backend_mod

        with patch(
            "predict_rlm.backends.sbx.backend.Path.cwd", return_value=tmp_path
        ):
            backend = SbxBackend(config=SbxConfig(name="hot-box", reuse=True))
        assert (
            str(backend._staging_root)
            not in backend_mod._owned_staging_roots_pending_cleanup
        )

    def test_ephemeral_staging_root_is_unique_uuid(self, tmp_path: Path):
        with patch(
            "predict_rlm.backends.sbx.backend.Path.cwd", return_value=tmp_path
        ):
            backend_a = SbxBackend(config=SbxConfig())
            backend_b = SbxBackend(config=SbxConfig())
        assert backend_a._staging_root != backend_b._staging_root

    def test_reuse_relocated_staging_root_is_deterministic_across_sessions(
        self, tmp_path: Path
    ):
        """Reattach regression: when the deterministic staging root is nested in
        a direct workspace mount it gets relocated out, but the relocated path
        must stay identical across sessions — otherwise the reattached
        container's bind mounts point at the previous session's now-gone temp
        dir and the websocket supervisor never starts (issues #41/#42).
        """
        mounts = [DirectWorkspaceMount(host_path=str(tmp_path), sandbox_path="/work")]

        def _make() -> SbxBackend:
            with patch(
                "predict_rlm.backends.sbx.backend.Path.cwd", return_value=tmp_path
            ):
                return SbxBackend(
                    config=SbxConfig(name="hot-box", reuse=True),
                    direct_workspace_mounts=mounts,
                )

        backend_a = _make()
        backend_b = _make()
        try:
            assert tmp_path not in backend_a._staging_root.parents
            assert backend_a._staging_root == backend_b._staging_root
            assert backend_a._staging_root.name == "predict-rlm-sbx-hot-box"
        finally:
            for backend in (backend_a, backend_b):
                shutil.rmtree(backend._staging_root, ignore_errors=True)

    def test_ephemeral_relocated_staging_root_stays_unique(self, tmp_path: Path):
        """Non-reusable sandboxes still relocate to a random per-run temp dir."""
        mounts = [DirectWorkspaceMount(host_path=str(tmp_path), sandbox_path="/work")]
        with patch(
            "predict_rlm.backends.sbx.backend.Path.cwd", return_value=tmp_path
        ):
            backend_a = SbxBackend(config=SbxConfig(), direct_workspace_mounts=mounts)
            backend_b = SbxBackend(config=SbxConfig(), direct_workspace_mounts=mounts)
        try:
            assert tmp_path not in backend_a._staging_root.parents
            assert backend_a._staging_root != backend_b._staging_root
        finally:
            for backend in (backend_a, backend_b):
                shutil.rmtree(backend._staging_root, ignore_errors=True)


def _reattach_backend(tmp_path: Path, *, name: str = "hot-box") -> SbxBackend:
    return SbxBackend(
        config=SbxConfig(name=name, reuse=True),
        preinstall_packages=False,
        _staging_root=tmp_path / "staging",
    )


class TestSbxBackendReattachDetection:
    """3-way reattach resolution: running / stopped / missing (issue #41)."""

    def _patches(self, backend: SbxBackend, *, ls_output: str):
        runs: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            runs.append(list(cmd))
            if cmd[:2] == ["sbx", "ls"]:
                return SimpleNamespace(returncode=0, stdout=ls_output, stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        cm = [
            patch(
                "predict_rlm.backends.sbx.backend.shutil.which",
                return_value="/usr/bin/sbx",
            ),
            patch(
                "predict_rlm.backends.sbx.backend.subprocess.run",
                side_effect=fake_run,
            ),
            patch.object(
                SbxBackend, "_prepare_supervisor_script", return_value=Path("/sup.py")
            ),
            patch.object(SbxBackend, "_apply_network_policy"),
            patch.object(SbxBackend, "_bootstrap_packages"),
            patch.object(SbxBackend, "_setup_direct_workspace_aliases_in_sandbox"),
            patch.object(SbxBackend, "_sbx_sandbox_healthy", return_value=True),
        ]
        return runs, cm

    def test_running_named_sandbox_reattaches_without_create_or_bootstrap(
        self, tmp_path: Path
    ):
        backend = _reattach_backend(tmp_path)
        runs, cms = self._patches(backend, ls_output="hot-box  running\n")
        with (
            cms[0],
            cms[1],
            cms[2],
            patch.object(SbxBackend, "_apply_network_policy") as net,
            patch.object(SbxBackend, "_bootstrap_packages") as boot,
            patch.object(SbxBackend, "_setup_direct_workspace_aliases_in_sandbox"),
            patch.object(SbxBackend, "_sbx_sandbox_healthy", return_value=True),
        ):
            backend._start_sbx_and_prepare_supervisor()
        assert backend._sandbox_name == "hot-box"
        # No create command was issued.
        assert not any(r[:2] == ["sbx", "create"] for r in runs)
        net.assert_not_called()
        boot.assert_not_called()

    def test_stopped_named_sandbox_is_started_then_reattaches(self, tmp_path: Path):
        backend = _reattach_backend(tmp_path)
        runs, cms = self._patches(backend, ls_output="hot-box  stopped\n")
        with (
            cms[0],
            cms[1],
            cms[2],
            patch.object(SbxBackend, "_apply_network_policy") as net,
            patch.object(SbxBackend, "_bootstrap_packages") as boot,
            patch.object(SbxBackend, "_setup_direct_workspace_aliases_in_sandbox"),
            patch.object(SbxBackend, "_sbx_sandbox_healthy", return_value=True),
        ):
            backend._start_sbx_and_prepare_supervisor()
        assert backend._sandbox_name == "hot-box"
        assert any(
            r[:2] == ["sbx", "start"] and "hot-box" in r for r in runs
        ), runs
        assert not any(r[:2] == ["sbx", "create"] for r in runs)
        net.assert_not_called()
        boot.assert_not_called()

    def test_missing_named_sandbox_falls_through_to_create(self, tmp_path: Path):
        backend = _reattach_backend(tmp_path)
        runs, cms = self._patches(backend, ls_output="other-box  running\n")
        with (
            cms[0],
            cms[1],
            cms[2],
            patch.object(SbxBackend, "_apply_network_policy") as net,
            patch.object(SbxBackend, "_bootstrap_packages") as boot,
            patch.object(SbxBackend, "_setup_direct_workspace_aliases_in_sandbox"),
            patch.object(SbxBackend, "_sbx_sandbox_healthy", return_value=True),
        ):
            backend._start_sbx_and_prepare_supervisor()
        assert backend._sandbox_name == "hot-box"
        assert any(r[:2] == ["sbx", "create"] for r in runs), runs
        net.assert_called_once()
        boot.assert_called_once()

    def test_running_but_unhealthy_recreates(self, tmp_path: Path):
        backend = _reattach_backend(tmp_path)
        runs, _ = self._patches(backend, ls_output="hot-box  running\n")

        def fake_run(cmd, *args, **kwargs):
            runs.append(list(cmd))
            if cmd[:2] == ["sbx", "ls"]:
                return SimpleNamespace(
                    returncode=0, stdout="hot-box  running\n", stderr=""
                )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        runs.clear()
        with (
            patch(
                "predict_rlm.backends.sbx.backend.shutil.which",
                return_value="/usr/bin/sbx",
            ),
            patch(
                "predict_rlm.backends.sbx.backend.subprocess.run",
                side_effect=fake_run,
            ),
            patch.object(
                SbxBackend, "_prepare_supervisor_script", return_value=Path("/sup.py")
            ),
            patch.object(SbxBackend, "_apply_network_policy") as net,
            patch.object(SbxBackend, "_bootstrap_packages") as boot,
            patch.object(SbxBackend, "_setup_direct_workspace_aliases_in_sandbox"),
            patch.object(SbxBackend, "_sbx_sandbox_healthy", return_value=False),
        ):
            backend._start_sbx_and_prepare_supervisor()
        # Unhealthy -> force-remove + recreate + bootstrap.
        assert any(
            r[:2] == ["sbx", "rm"] and "hot-box" in r for r in runs
        ), runs
        assert any(r[:2] == ["sbx", "create"] for r in runs), runs
        net.assert_called_once()
        boot.assert_called_once()


class TestSbxBackendReattachShutdown:
    """Shutdown under reuse must not remove the sandbox or staging root."""

    def test_reuse_shutdown_does_not_rm_or_delete_staging(self, tmp_path: Path):
        backend = _reattach_backend(tmp_path)
        backend._sandbox_name = "hot-box"
        staging = backend._staging_root
        assert staging.exists()
        runs: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            runs.append(list(cmd))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch(
            "predict_rlm.backends.sbx.backend.subprocess.run", side_effect=fake_run
        ):
            backend.shutdown()
        assert not any(r[:2] == ["sbx", "rm"] for r in runs), runs
        assert staging.exists()

    def test_reuse_stop_on_shutdown_stops_container(self, tmp_path: Path):
        backend = SbxBackend(
            config=SbxConfig(name="hot-box", reuse=True, stop_on_shutdown=True),
            preinstall_packages=False,
            _staging_root=tmp_path / "staging",
        )
        backend._sandbox_name = "hot-box"
        runs: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            runs.append(list(cmd))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch(
            "predict_rlm.backends.sbx.backend.subprocess.run", side_effect=fake_run
        ):
            backend.shutdown()
        assert any(
            r[:2] == ["sbx", "stop"] and "hot-box" in r for r in runs
        ), runs
        assert not any(r[:2] == ["sbx", "rm"] for r in runs), runs


class TestSbxBackendDestroy:
    """Explicit teardown API (issue #41)."""

    def test_destroy_removes_sandbox_and_staging_root(self, tmp_path: Path):
        backend = _reattach_backend(tmp_path)
        backend._sandbox_name = "hot-box"
        staging = backend._staging_root
        assert staging.exists()
        runs: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            runs.append(list(cmd))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch(
            "predict_rlm.backends.sbx.backend.subprocess.run", side_effect=fake_run
        ):
            backend.destroy()
        assert any(
            r[:3] == ["sbx", "rm", "--force"] and "hot-box" in r for r in runs
        ), runs
        assert not staging.exists()

    def test_remove_classmethod_force_removes_named_sandbox(self):
        runs: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            runs.append(list(cmd))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch(
            "predict_rlm.backends.sbx.backend.subprocess.run", side_effect=fake_run
        ):
            SbxBackend.remove("hot-box")
        assert any(
            r[:3] == ["sbx", "rm", "--force"] and "hot-box" in r for r in runs
        ), runs


class TestSbxBackendReattachRegression:
    """`reuse=False` default create path is unchanged (issue #41)."""

    def test_default_path_still_creates_without_ls_probe(self, tmp_path: Path):
        backend = SbxBackend(
            config=SbxConfig(),
            preinstall_packages=False,
            _staging_root=tmp_path / "staging",
        )
        runs: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            runs.append(list(cmd))
            return SimpleNamespace(returncode=0, stdout="auto-name\n", stderr="")

        with (
            patch(
                "predict_rlm.backends.sbx.backend.shutil.which",
                return_value="/usr/bin/sbx",
            ),
            patch(
                "predict_rlm.backends.sbx.backend.subprocess.run", side_effect=fake_run
            ),
            patch.object(
                SbxBackend, "_prepare_supervisor_script", return_value=Path("/sup.py")
            ),
            patch.object(SbxBackend, "_apply_network_policy") as net,
            patch.object(SbxBackend, "_bootstrap_packages") as boot,
            patch.object(SbxBackend, "_setup_direct_workspace_aliases_in_sandbox"),
        ):
            backend._start_sbx_and_prepare_supervisor()
        # No reattach probe on the default path.
        assert not any(r[:2] == ["sbx", "ls"] for r in runs), runs
        assert any(r[:2] == ["sbx", "create"] for r in runs), runs
        net.assert_called_once()
        boot.assert_called_once()


@pytest.mark.integration
@pytest.mark.skipif(
    not _real_sbx_available(),
    reason="real Docker Sandboxes tests require PREDICT_RLM_RUN_SBX_TESTS=1, sbx CLI, and sbx login",
)
class TestSbxBackendRealSbxReattach:
    """End-to-end persist + reattach lifecycle against a real sbx sandbox (#41).

    The headline test the user asked for: first prewarm creates+bootstraps and
    writes filesystem state; shutdown leaves the container alive (no `sbx rm`);
    a second backend reattaches WITHOUT create/bootstrap (asserted via lifecycle
    telemetry and a spy on `_bootstrap_packages`), the persisted state survives,
    and finally `destroy()` removes it so a subsequent prewarm does a clean create.
    """

    def _list_names(self) -> list[str]:
        result = subprocess.run(
            ["sbx", "ls"], capture_output=True, text=True, check=False, timeout=15
        )
        return [line.split()[0] for line in result.stdout.splitlines() if line.split()]

    def test_persist_reattach_destroy_lifecycle(self):
        name = f"predict-rlm-reattach-{os.getpid()}"
        config = SbxConfig(name=name, reuse=True)
        marker = f"state-{os.getpid()}"

        # First session: create + bootstrap + write persisted /sandbox state.
        first = SbxBackend(config=config, preinstall_packages=False, debug=True)
        try:
            first.prewarm()
            first.execute(
                "from pathlib import Path\n"
                f"Path('/sandbox/persisted.txt').write_text({marker!r})\n"
                "print('wrote')"
            )
            first.shutdown()
            # Container must still be listed (no `sbx rm` happened).
            assert name in self._list_names()

            # Second session: reattach. Spy on bootstrap/create to prove they are skipped.
            second = SbxBackend(config=config, preinstall_packages=False, debug=True)
            events: list[str] = []
            orig_log = second._log_lifecycle

            def spy_log(event, **fields):
                events.append(event)
                return orig_log(event, **fields)

            with (
                patch.object(second, "_log_lifecycle", side_effect=spy_log),
                patch.object(
                    SbxBackend,
                    "_bootstrap_packages",
                    side_effect=AssertionError("bootstrap must not run on reattach"),
                ),
            ):
                second.prewarm()
                out = second.execute(
                    "from pathlib import Path\n"
                    "print(Path('/sandbox/persisted.txt').read_text())"
                )
            assert out.strip() == marker
            assert any(e.startswith("sbx.reattach") for e in events), events
            assert not any(e == "sbx.create.start" for e in events), events
            second.shutdown()
            assert name in self._list_names()

            # destroy() removes the container + staging root.
            second.destroy()
            assert name not in self._list_names()

            # A fresh backend now does a clean create (reattach miss).
            third = SbxBackend(config=config, preinstall_packages=False, debug=True)
            try:
                third.prewarm()
                fresh = third.execute(
                    "from pathlib import Path\n"
                    "print(Path('/sandbox/persisted.txt').exists())"
                )
                assert fresh.strip() == "False"
            finally:
                third.destroy()
        finally:
            subprocess.run(
                ["sbx", "rm", "--force", name],
                capture_output=True,
                text=True,
                check=False,
            )

    def test_reattach_after_interpreter_error_recovers(self):
        name = f"predict-rlm-recover-{os.getpid()}"
        config = SbxConfig(name=name, reuse=True)

        first = SbxBackend(config=config, preinstall_packages=False, debug=True)
        try:
            first.prewarm()
            first.execute("keep = 7\nprint('ready')")
            with pytest.raises(CodeInterpreterError, match="ValueError"):
                first.execute("raise ValueError('boom')")
            # The supervisor survives the error: same session keeps working and
            # globals defined before the error are intact.
            assert first.execute("print(keep + 1)").strip() == "8"
            first.shutdown()
            assert name in self._list_names()

            # Reattach to the sandbox that errored then detached: still usable.
            second = SbxBackend(config=config, preinstall_packages=False, debug=True)
            second.prewarm()
            assert second.execute("print('recovered')").strip() == "recovered"
            with pytest.raises(CodeInterpreterError, match="ValueError"):
                second.execute("raise ValueError('again')")
            assert second.execute("print(6 * 7)").strip() == "42"
            second.destroy()
            assert name not in self._list_names()
        finally:
            subprocess.run(
                ["sbx", "rm", "--force", name],
                capture_output=True,
                text=True,
                check=False,
            )

