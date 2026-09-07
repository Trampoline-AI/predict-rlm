"""Tests for the Docker Sandboxes execution backend."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import select
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("websockets")  # SBX/supervisor backend requires the [sbx] extra


pytestmark = pytest.mark.sbx

from predict_rlm.backends import SbxBackend, SbxConfig  # noqa: E402
from predict_rlm.backends.base import (  # noqa: E402
    BackendExecutionGate,
    SandboxExecutionError,
    SandboxFatalError,
)
from predict_rlm.backends.sbx.execution import SbxExecutionBackend  # noqa: E402
from predict_rlm.backends.supervisor._payload import (  # noqa: E402
    _pickleable_globals_snapshot,
)
from predict_rlm.files import SyncedFile  # noqa: E402
from predict_rlm.runtime import (  # noqa: E402
    ExecutionSpec,
    HostDirectoryMount,
    UnsupportedOperationError,
)

PAYLOAD_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "predict_rlm"
    / "backends"
    / "supervisor"
    / "_payload.py"
)


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
    def test_user_subprocess_stdin_is_isolated_from_runner_protocol(self, runner: LocalRunner):
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

    def test_kernel_result_waits_for_tool_reader_handoff(self, monkeypatch):
        from predict_rlm.backends.supervisor import _payload

        class TrackingCondition(threading.Condition):
            def __init__(self) -> None:
                super().__init__()
                self.publisher_waiting = threading.Event()

            def wait(self, timeout=None):
                self.publisher_waiting.set()
                return super().wait(timeout)

        condition = TrackingCondition()
        release_reader = threading.Event()
        reader_started = threading.Event()
        result_queue: queue.Queue = queue.Queue()

        def reader_loop() -> None:
            reader_started.set()
            release_reader.wait()
            with condition:
                _payload.TOOL_RESPONSE_READER_THREAD = None
                condition.notify_all()

        reader = threading.Thread(target=reader_loop)
        monkeypatch.setattr(_payload, "TOOL_RESPONSE_CONDITION", condition)
        monkeypatch.setattr(_payload, "WAITING_TOOL_RESPONSE_IDS", set())
        monkeypatch.setattr(_payload, "TOOL_RESPONSE_READER_THREAD", reader)

        reader.start()
        assert reader_started.wait(timeout=1)
        publisher = threading.Thread(
            target=_payload._publish_kernel_result,
            args=(result_queue, {"ok": True}),
        )
        publisher.start()

        assert condition.publisher_waiting.wait(timeout=1)
        with pytest.raises(queue.Empty):
            result_queue.get_nowait()

        release_reader.set()
        publisher.join(timeout=1)
        reader.join(timeout=1)

        assert not publisher.is_alive()
        assert not reader.is_alive()
        assert result_queue.get_nowait() == {"ok": True}

    @pytest.mark.local
    def test_stale_concurrent_tool_calls_do_not_poison_later_execute(self, runner: LocalRunner):
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
                    'sys.stdout.flush(); sys.stderr.flush(); time.sleep(30)",\n'
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

    def test_timeout_is_not_swallowed_by_user_exception_handler(self, runner: LocalRunner):
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
        port: int | None = None,
        name: str = "local-websocket-test",
        reuse: bool = False,
        staging_root: Path | None = None,
        startup_timeout: float = 3,
    ) -> SbxBackend:
        port = port or _free_local_port()
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
                name=name,
                reuse=reuse,
                exec_timeout=5,
                websocket_startup_timeout=startup_timeout,
                websocket_max_message_bytes=32 * 1024 * 1024,
            ),
            tools=tools,
            preinstall_packages=False,
            _websocket_supervisor_command=command,
            _websocket_url=f"ws://127.0.0.1:{port}{url_path or websocket_path}",
            _staging_root=staging_root or tmp_path / "ws-staging",
        )

    def test_reusable_named_websocket_supervisors_run_concurrently(self, tmp_path: Path):
        staging_root = tmp_path / "shared-staging"
        first = self.make_interpreter(
            tmp_path,
            name="shared-websocket-test",
            reuse=True,
            staging_root=staging_root,
            path="/predict-rlm-first",
        )
        second = self.make_interpreter(
            tmp_path,
            name="shared-websocket-test",
            reuse=True,
            staging_root=staging_root,
            path="/predict-rlm-second",
        )
        barrier = threading.Barrier(3)
        errors: list[BaseException] = []
        outputs: dict[str, str] = {}

        def execute(interpreter: SbxBackend, label: str) -> None:
            try:
                barrier.wait(timeout=2)
                outputs[label] = interpreter.execute(
                    f"owner = {label!r}\nimport time\ntime.sleep(0.2)\nprint(owner)"
                )
            except BaseException as exc:
                errors.append(exc)

        try:
            first.prewarm()
            second.prewarm()

            threads = [
                threading.Thread(target=execute, args=(first, "first")),
                threading.Thread(target=execute, args=(second, "second")),
            ]
            for thread in threads:
                thread.start()
            barrier.wait(timeout=2)
            for thread in threads:
                thread.join(timeout=3)

            assert all(not thread.is_alive() for thread in threads)
            assert errors == []
            assert outputs == {"first": "first\n", "second": "second\n"}
            assert first.execute("print(owner)") == "first\n"
            assert second.execute("print(owner)") == "second\n"
        finally:
            first.shutdown()
            second.shutdown()

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

    def test_predicts_orphaned_by_gather_failure_do_not_hang_next_execute(self, tmp_path: Path):
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

    @pytest.mark.asyncio
    async def test_async_operations_do_not_delegate_to_sync_or_to_thread(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        async def add(a: int, b: int) -> dict:
            await asyncio.sleep(0)
            return {"total": a + b}

        interpreter = self.make_interpreter(tmp_path, tools={"add": add})
        source = tmp_path / "input.txt"
        source.write_text("hello", encoding="utf-8")
        output = tmp_path / "output.txt"

        def forbidden(*args, **kwargs):
            raise AssertionError("async SBX operation delegated to a sync API")

        monkeypatch.setattr(interpreter, "execute", forbidden)
        monkeypatch.setattr(interpreter, "interrupt", forbidden)
        monkeypatch.setattr(interpreter, "shutdown", forbidden)
        monkeypatch.setattr(asyncio, "to_thread", forbidden)
        try:
            await interpreter.amount_file_at(
                str(source),
                "/sandbox/input/source/input.txt",
            )
            await interpreter.amkdir_p("/sandbox/output/result")
            result = await interpreter.aexecute(
                "result = await add(2, 3)\n"
                "print(result['total'])\n"
                "from pathlib import Path\n"
                "Path('/sandbox/output/result/output.txt').write_text('done')"
            )
            files = await interpreter.alist_dir("/sandbox/output/result")
            manifest = await interpreter.aworkspace_manifest("/sandbox/output/result")
            await interpreter.async_file_to(
                "/sandbox/output/result/output.txt",
                str(output),
            )
        finally:
            await interpreter.ashutdown()

        assert result == "5\n"
        assert files == ["/sandbox/output/result/output.txt"]
        assert manifest["output.txt"].size == 4
        assert output.read_text(encoding="utf-8") == "done"

    @pytest.mark.asyncio
    async def test_aexecute_cancellation_uses_native_interrupt_and_recovers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        interpreter = self.make_interpreter(tmp_path, startup_timeout=5)

        def forbidden(*args, **kwargs):
            raise AssertionError("async cancellation delegated to a sync API")

        monkeypatch.setattr(interpreter, "execute", forbidden)
        monkeypatch.setattr(interpreter, "interrupt", forbidden)
        monkeypatch.setattr(asyncio, "to_thread", forbidden)
        try:
            await interpreter.aexecute("seed = 5")
            task = asyncio.create_task(
                interpreter.aexecute("import time\ntime.sleep(120)\nprint('done')")
            )
            while not interpreter._execution_gate.is_running():
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.2)

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            assert await interpreter.aexecute("print(seed)") == "5\n"
        finally:
            await interpreter.ashutdown()


@pytest.mark.asyncio
async def test_cancelled_async_tool_tasks_remain_quarantined_until_done(tmp_path: Path):
    interpreter = SbxBackend.__new__(SbxBackend)
    interpreter._async_pending_tool_calls = {}
    interpreter._quarantined_async_tool_calls = set()
    release = asyncio.Event()

    async def stubborn_tool_task():
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            await release.wait()

    task = asyncio.create_task(stubborn_tool_task())
    interpreter._async_pending_tool_calls[task] = 1
    await asyncio.sleep(0)

    await interpreter._acancel_async_tool_calls(timeout=0.01)

    assert not task.done()
    assert interpreter._pending_tool_count() == 1

    release.set()
    await task
    await asyncio.sleep(0)

    assert interpreter._pending_tool_count() == 0


@pytest.mark.asyncio
async def test_aexecute_skips_post_hooks_after_fatal_failure():
    interpreter = SbxBackend.__new__(SbxBackend)
    interpreter._execution_gate = BackendExecutionGate("SBX backend")
    interpreter._post_execute_hooks = []
    interpreter._uses_websocket_transport = lambda: True  # type: ignore[method-assign]
    interpreter._aexecute_top_level = AsyncMock(  # type: ignore[method-assign]
        side_effect=SandboxFatalError("fatal")
    )
    hook = AsyncMock()
    interpreter.add_post_execute_hook(hook)

    with pytest.raises(SandboxFatalError, match="fatal"):
        await interpreter.aexecute("raise SystemExit")

    hook.assert_not_awaited()


@pytest.mark.asyncio
async def test_aexecute_cancellation_skips_failing_post_hook():
    interpreter = SbxBackend.__new__(SbxBackend)
    interpreter._execution_gate = BackendExecutionGate("SBX backend")
    interpreter._post_execute_hooks = []
    interpreter._active_async_request = None
    interpreter._uses_websocket_transport = lambda: True  # type: ignore[method-assign]
    started = asyncio.Event()

    async def block(code, variables, *, timeout=None):
        started.set()
        await asyncio.Future()

    async def fail_hook(_backend):
        raise OSError("unsafe workspace sync")

    interpreter._aexecute_top_level = block  # type: ignore[method-assign]
    interpreter.add_post_execute_hook(fail_hook)
    execution = asyncio.create_task(interpreter.aexecute("await work()"))
    await started.wait()
    execution.cancel()

    with pytest.raises(asyncio.CancelledError):
        await execution


@pytest.mark.asyncio
async def test_aexecute_preserves_primary_error_when_post_hook_fails():
    interpreter = SbxBackend.__new__(SbxBackend)
    interpreter._execution_gate = BackendExecutionGate("SBX backend")
    interpreter._post_execute_hooks = []
    interpreter._uses_websocket_transport = lambda: True  # type: ignore[method-assign]
    interpreter._aexecute_top_level = AsyncMock(  # type: ignore[method-assign]
        side_effect=ValueError("primary")
    )

    async def fail_hook(_backend):
        raise OSError("sync failed")

    interpreter.add_post_execute_hook(fail_hook)

    with pytest.raises(ValueError, match="primary") as raised:
        await interpreter.aexecute("bad code")

    assert isinstance(raised.value.post_execute_error, OSError)


def test_execute_skips_post_hooks_after_fatal_failure():
    interpreter = SbxBackend.__new__(SbxBackend)
    interpreter._execution_gate = BackendExecutionGate("SBX backend")
    interpreter._post_execute_hooks = []
    interpreter._execute_top_level = MagicMock(  # type: ignore[method-assign]
        side_effect=SandboxFatalError("fatal")
    )
    hook = MagicMock()
    interpreter.add_post_execute_hook(hook)

    with pytest.raises(SandboxFatalError, match="fatal"):
        interpreter.execute("raise SystemExit")

    hook.assert_not_called()


def test_execute_preserves_primary_error_when_post_hook_fails():
    interpreter = SbxBackend.__new__(SbxBackend)
    interpreter._execution_gate = BackendExecutionGate("SBX backend")
    interpreter._post_execute_hooks = []
    interpreter._execute_top_level = MagicMock(  # type: ignore[method-assign]
        side_effect=ValueError("primary")
    )
    interpreter.add_post_execute_hook(MagicMock(side_effect=OSError("sync failed")))

    with pytest.raises(ValueError, match="primary") as raised:
        interpreter.execute("bad code")

    assert isinstance(raised.value.post_execute_error, OSError)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spec",
    [
        ExecutionSpec(
            host_directory_mounts=(HostDirectoryMount("/host/workspace", "/workspace"),)
        ),
        ExecutionSpec(allowed_domains=("service.internal",)),
        ExecutionSpec(extra_read_paths=("/host/input",)),
        ExecutionSpec(extra_write_paths=("/host/output",)),
    ],
)
async def test_reused_sbx_rejects_invocation_policy_before_backend_construction(
    monkeypatch,
    spec: ExecutionSpec,
):
    constructions = 0

    def construct_backend(**kwargs):
        nonlocal constructions
        constructions += 1
        raise AssertionError("unsafe reused SBX policy reached backend construction")

    monkeypatch.setattr(
        "predict_rlm.backends.sbx.execution.SbxBackend",
        construct_backend,
    )
    backend = SbxExecutionBackend(
        config=SbxConfig(name="hot-box", reuse=True),
    )

    with pytest.raises(UnsupportedOperationError, match="reused SBX"):
        async with backend.start(spec, SimpleNamespace(session=None, ownership=None)):
            pass

    assert constructions == 0


@pytest.mark.asyncio
async def test_cancelled_sync_tool_keeps_direct_synced_temp_until_worker_exits(
    tmp_path: Path,
):
    interpreter = SbxBackend.__new__(SbxBackend)
    interpreter.verbose = False
    interpreter._execution_gate = BackendExecutionGate("SBX backend")
    started = threading.Event()
    release = threading.Event()
    temporary_root = tmp_path / "tool-file-sync"
    temporary_root.mkdir()
    temporary_file = temporary_root / "input.txt"
    temporary_file.write_text("input", encoding="utf-8")

    def block() -> str:
        started.set()
        release.wait()
        assert temporary_file.exists()
        return "done"

    async def prepare(tool, args, kwargs):
        return args, kwargs, [], str(temporary_root)

    interpreter.tools = {"block": block}
    interpreter._aprepare_synced_file_tool_args = prepare  # type: ignore[method-assign]
    invocation = asyncio.create_task(
        interpreter._abuild_tool_response(
            {"id": 1, "params": {"name": "block", "args": [], "kwargs": {}}}
        )
    )
    await asyncio.to_thread(started.wait)

    invocation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invocation

    assert temporary_file.exists()
    worker = next(iter(interpreter._sync_worker_set()))
    release.set()
    await asyncio.wait_for(worker.wait(), timeout=1)
    for _ in range(100):
        if not temporary_root.exists():
            break
        await asyncio.sleep(0.01)

    assert not temporary_root.exists()


class TestSbxBackendInterrupt:
    make_interpreter = TestSbxBackendLocalWebSocketRunner.make_interpreter

    @pytest.mark.local
    def test_interrupt_returns_only_after_cell_releases_gate(self, tmp_path: Path):
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
            time.sleep(0.5)

            was_running = interpreter.interrupt(timeout=10.0)

            assert was_running is True
            assert gate.is_running() is False, (
                "interrupt returned before the interrupted cell released the gate"
            )

            worker.join(timeout=5)
            assert not worker.is_alive()
            assert interpreter.execute("print(warm)") == "1\n"
        finally:
            interpreter.shutdown()


@pytest.mark.integration
@pytest.mark.skipif(
    not _real_sbx_available(),
    reason="real Docker Sandboxes tests require PREDICT_RLM_RUN_SBX_TESTS=1, sbx CLI, and sbx login",
)
class TestSbxBackendRealSbxReattach:
    def _list_names(self) -> list[str]:
        result = subprocess.run(
            ["sbx", "ls"], capture_output=True, text=True, check=False, timeout=15
        )
        return [line.split()[0] for line in result.stdout.splitlines() if line.split()]

    def test_persist_reattach_destroy_lifecycle(self):
        name = f"predict-rlm-reattach-{os.getpid()}"
        config = SbxConfig(name=name, reuse=True)
        marker = f"state-{os.getpid()}"

        first = SbxBackend(config=config, preinstall_packages=False, debug=True)
        try:
            first.prewarm()
            first.execute(
                "from pathlib import Path\n"
                f"Path('/sandbox/persisted.txt').write_text({marker!r})\n"
                "print('wrote')"
            )
            first.shutdown()
            assert name in self._list_names()

            second = SbxBackend(config=config, preinstall_packages=False)
            second.prewarm()
            out = second.execute(
                "from pathlib import Path\nprint(Path('/sandbox/persisted.txt').read_text())"
            )
            assert out.strip() == marker
            second.shutdown()
            assert name in self._list_names()

            second.destroy()
            assert name not in self._list_names()

            third = SbxBackend(config=config, preinstall_packages=False, debug=True)
            try:
                third.prewarm()
                fresh = third.execute(
                    "from pathlib import Path\nprint(Path('/sandbox/persisted.txt').exists())"
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
