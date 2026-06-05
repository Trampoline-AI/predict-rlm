"""Tests for the Docker Sandboxes interpreter backend."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import select
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated

import pytest
from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput

from predict_rlm.files import SyncedFile
from predict_rlm.interpreter import SandboxFatalError
from predict_rlm.interpreters import DEFAULT_SBX_TEMPLATE, SbxConfig, SbxInterpreter, SbxPool

RUNNER_PATH = Path(__file__).parents[1] / "src" / "predict_rlm" / "sandbox" / "python_runner.py"


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


class LocalRunner:
    def __init__(self, tmp_path: Path) -> None:
        env_root = tmp_path / "runner-root"
        env_root.mkdir()
        self.proc = subprocess.Popen(
            [sys.executable, "-u", str(RUNNER_PATH)],
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


@pytest.fixture
def runner(tmp_path):
    proc = LocalRunner(tmp_path)
    try:
        yield proc
    finally:
        proc.close()


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


class TestSbxInterpreterLocalRunner:
    def make_interpreter(
        self,
        tmp_path: Path,
        *,
        debug: bool = False,
        verbose: bool = False,
        tools: dict | None = None,
    ) -> SbxInterpreter:
        return SbxInterpreter(
            config=SbxConfig(name="local-test"),
            tools=tools,
            preinstall_packages=False,
            debug=debug,
            verbose=verbose,
            _supervisor_command=[sys.executable, "-u", str(RUNNER_PATH)],
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
        interpreter = SbxInterpreter(
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
        interpreter = SbxInterpreter(
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
        interpreter = SbxInterpreter(
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
        interpreter = SbxInterpreter(
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
        interpreter = SbxInterpreter(
            config=SbxConfig(name="local-test"),
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(RUNNER_PATH)],
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
        interpreter = SbxInterpreter(
            config=SbxConfig(name="local-test"),
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(RUNNER_PATH)],
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
        interpreter = SbxInterpreter(
            config=SbxConfig(name="local-test", persist=True),
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(RUNNER_PATH)],
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

        interpreter = SbxInterpreter(
            config=SbxConfig(name="local-test"),
            tools={"add": add},
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(RUNNER_PATH)],
            _staging_root=tmp_path / "staging",
        )
        try:
            output = interpreter.execute("result = await add(2, 3)\nprint(result['total'])")
        finally:
            interpreter.shutdown()

        assert output.strip() == "5"

    def test_host_tool_synced_file_writeback_updates_sandbox_file(self, tmp_path: Path):
        received_paths: list[str] = []

        def mutate(path: Annotated[str, SyncedFile(writeback=True)]) -> str:
            received_paths.append(path)
            file_path = Path(path)
            original = file_path.read_text(encoding="utf-8")
            file_path.write_text(original + " + host", encoding="utf-8")
            return file_path.read_text(encoding="utf-8")

        interpreter = SbxInterpreter(
            config=SbxConfig(name="local-test"),
            tools={"mutate": mutate},
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(RUNNER_PATH)],
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

        interpreter = SbxInterpreter(
            config=SbxConfig(name="local-test"),
            tools={"mutate": mutate},
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(RUNNER_PATH)],
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

        interpreter = SbxInterpreter(
            config=SbxConfig(name="local-test"),
            tools={"mutate": mutate},
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(RUNNER_PATH)],
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
        interpreter = SbxInterpreter(
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

        interpreter = SbxInterpreter(
            config=SbxConfig(name="local-test", exec_timeout=3),
            tools={"slow": slow},
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(RUNNER_PATH)],
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

        interpreter = SbxInterpreter(
            config=SbxConfig(name="local-test", exec_timeout=3),
            tools={"reenter": reenter},
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(RUNNER_PATH)],
            _staging_root=tmp_path / "staging",
        )
        try:
            output = interpreter.execute("result = await reenter()\nprint(result)")
        finally:
            interpreter.shutdown()

        assert output.strip() == "blocked"
        assert len(observed_errors) == 1

    def test_request_timeout_fires_when_runner_stays_silent(self, tmp_path: Path):
        interpreter = SbxInterpreter(
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
        interpreter = SbxInterpreter(
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
        interpreter = SbxInterpreter(
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
        interpreter = SbxInterpreter(
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
        interpreter = SbxInterpreter(
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
        interpreter = SbxInterpreter(
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
        interpreter = SbxInterpreter(
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
        interpreter = SbxInterpreter(
            config=SbxConfig(name="created-name"),
            preinstall_packages=False,
            _staging_root=tmp_path / "staging",
        )

        command = interpreter._start_sbx_and_build_supervisor_command()

        assert "python" not in command
        runner_path = tmp_path / "staging" / ".predict_rlm_runner" / "python_runner.py"
        assert runner_path.read_text(encoding="utf-8") == RUNNER_PATH.read_text(
            encoding="utf-8"
        )
        assert command[-3:] == ["python3", "-u", str(runner_path)]

    def test_runner_restart_reuses_existing_sandbox(
        self, monkeypatch, tmp_path: Path
    ):
        commands: list[list[str]] = []

        def fake_run(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="created-name\n", stderr="")

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/sbx")
        monkeypatch.setattr(subprocess, "run", fake_run)
        interpreter = SbxInterpreter(
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
        interpreter = SbxInterpreter(
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
        interpreter = SbxInterpreter(
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
            _supervisor_command=[sys.executable, "-u", str(RUNNER_PATH)],
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
            _supervisor_command=[sys.executable, "-u", str(RUNNER_PATH)],
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
            _supervisor_command=[sys.executable, "-u", str(RUNNER_PATH)],
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
@pytest.mark.skipif(
    not _real_sbx_available(),
    reason="real Docker Sandboxes tests require PREDICT_RLM_RUN_SBX_TESTS=1, sbx CLI, and sbx login",
)
class TestSbxInterpreterRealSbx:
    def test_real_sbx_executes_basic_python(self):
        interpreter = SbxInterpreter(
            config=SbxConfig(name=f"predict-rlm-test-{os.getpid()}"),
            preinstall_packages=False,
        )
        try:
            output = interpreter.execute("print(2 + 3)")
        finally:
            interpreter.shutdown()

        assert output.strip() == "5"

    def test_real_sbx_timeout_is_recoverable_and_runner_survives(self):
        interpreter = SbxInterpreter(
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
