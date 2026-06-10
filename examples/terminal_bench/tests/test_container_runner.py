from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from dspy.primitives.code_interpreter import CodeInterpreterError

from predict_rlm.debug import reset_debug_logger_for_tests

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from terminal_bench_rlm.tools.container_runner import (  # noqa: E402
    HarborContainerAdapter,
    LocalProcessRunnerClientAdapter,
    TerminalBenchRunnerClientAdapter,
)
from terminal_bench_rlm.tools.runner import runner_source  # noqa: E402


class FakePipe:
    def __init__(self, on_write=None) -> None:
        self.lines: list[str] = []
        self._on_write = on_write

    def write(self, data: str) -> None:
        self.lines.append(data)
        if self._on_write is not None:
            self._on_write(data)

    def flush(self) -> None:
        return None

    def readline(self) -> str:
        if not self.lines:
            return ""
        return self.lines.pop(0)

    def read(self) -> str:
        return "".join(self.lines)


class FakeProcess:
    def __init__(
        self,
        responses: list[dict | tuple[float, dict]],
        *,
        stderr: str = "",
        wait_callback=None,
    ) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.stdout = FakePipe()
        self.stderr = FakePipe()
        if stderr:
            self.stderr.lines.append(stderr)
        self.stdin = FakePipe(self._on_stdin)
        self.killed = False
        self.waited = False
        self.returncode: int | None = None
        self.wait_callback = wait_callback

    def _on_stdin(self, data: str) -> None:
        request = json.loads(data)
        self.requests.append(request)
        if self.responses:
            response = self.responses.pop(0)
            if isinstance(response, tuple):
                delay, payload = response
                threading.Timer(delay, self._append_response, args=(payload,)).start()
            else:
                self._append_response(response)

    def _append_response(self, response: dict) -> None:
        self.stdout.lines.append(json.dumps(response) + "\n")

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        return 1 if self.killed else None

    def wait(self, timeout=None):
        self.waited = True
        if self.wait_callback is not None:
            return self.wait_callback()
        if self.returncode is None:
            self.returncode = 0
        return 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class FakeAdapter:
    def __init__(self, process: FakeProcess | list[FakeProcess]) -> None:
        self.process = process[0] if isinstance(process, list) else process
        self.processes = list(process) if isinstance(process, list) else [process]
        self.started: list[dict] = []
        self.copied_to: list[tuple[str, str]] = []
        self.copied_from: list[tuple[str, str]] = []
        self.exec_calls: list[list[str]] = []
        self.exec_results: list[SimpleNamespace] = []

    def copy_to(self, host_path: str, container_path: str) -> None:
        self.copied_to.append((host_path, container_path))

    def copy_from(self, container_path: str, host_path: str) -> None:
        self.copied_from.append((container_path, host_path))

    def exec(self, command: list[str], *, timeout: float | None = None):
        self.exec_calls.append(command)
        if self.exec_results:
            return self.exec_results.pop(0)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    def start_exec(
        self,
        command: list[str],
        *,
        workdir: str | None = None,
        timeout: float | None = None,
    ) -> FakeProcess:
        self.started.append({"command": command, "workdir": workdir, "timeout": timeout})
        process = self.processes[min(len(self.started) - 1, len(self.processes) - 1)]
        self.process = process
        return process


@pytest.fixture
def local_runner_interpreter(tmp_path: Path) -> LocalProcessRunnerClientAdapter:
    interpreter = LocalProcessRunnerClientAdapter(
        runner_path=str(tmp_path / "predict_rlm_runner.py"),
        workdir=str(tmp_path),
        exec_timeout=10,
        recoverable_timeout_grace=2.0,
    )
    try:
        yield interpreter
    finally:
        interpreter.shutdown()


def test_execute_reset_shutdown_requests_and_maps_success() -> None:
    process = FakeProcess(
        [
            {"id": 1, "ok": True, "result": {"output": "hi\n"}},
            {"id": 2, "ok": True, "result": {}},
            {"id": 3, "ok": True, "result": {"shutdown": True}},
        ]
    )
    adapter = FakeAdapter(process)
    interpreter = TerminalBenchRunnerClientAdapter(
        object(),
        container_adapter=adapter,
        runner_path="/tmp/predict_rlm_runner.py",
    )

    assert interpreter.execute("print('hi')") == "hi\n"
    interpreter.reset()
    interpreter.shutdown()

    assert adapter.started[0]["command"] == [
        "python3",
        "-u",
        "/tmp/predict_rlm_runner.py",
    ]
    assert [request["method"] for request in process.requests] == [
        "execute",
        "reset",
        "shutdown",
    ]
    assert adapter.copied_to[0][1] == "/tmp/predict_rlm_runner.py"
    assert process.waited is True


def test_shutdown_kills_original_process_if_wait_races_with_process_clear() -> None:
    interpreter: TerminalBenchRunnerClientAdapter | None = None

    def clear_process_then_timeout() -> None:
        assert interpreter is not None
        interpreter._process = None
        raise TimeoutError("wait timed out")

    process = FakeProcess(
        [
            {"id": 1, "ok": True, "result": {"output": "hi\n"}},
            {"id": 2, "ok": True, "result": {"shutdown": True}},
        ],
        wait_callback=clear_process_then_timeout,
    )
    interpreter = TerminalBenchRunnerClientAdapter(
        object(),
        container_adapter=FakeAdapter(process),
        runner_path="/tmp/predict_rlm_runner.py",
    )

    assert interpreter.execute("print('hi')") == "hi\n"
    interpreter.shutdown()

    assert process.killed is True
    assert interpreter._process is None


def test_execute_accepts_lm_selected_execution_timeout() -> None:
    process = FakeProcess(
        [{"id": 1, "ok": True, "result": {"output": "bounded\n"}}]
    )
    interpreter = TerminalBenchRunnerClientAdapter(
        object(),
        container_adapter=FakeAdapter(process),
        runner_path="/tmp/predict_rlm_runner.py",
    )

    assert interpreter.execute("print('bounded')", timeout=2.5) == "bounded\n"

    assert process.requests[0]["method"] == "execute"
    assert process.requests[0]["params"] == {
        "code": "print('bounded')",
        "execution_timeout_seconds": 2.5,
    }


def test_execute_debug_logs_empty_output_diagnostic(monkeypatch, tmp_path: Path) -> None:
    log_path = tmp_path / "predict_rlm_debug.jsonl"
    monkeypatch.setenv("PREDICT_RLM_DEBUG", "1")
    monkeypatch.setenv("PREDICT_RLM_DEBUG_JSON", "1")
    monkeypatch.setenv("PREDICT_RLM_DEBUG_LOG", str(log_path))
    reset_debug_logger_for_tests()
    process = FakeProcess([{"id": 1, "ok": True, "result": {"output": ""}}])
    interpreter = TerminalBenchRunnerClientAdapter(
        object(),
        container_adapter=FakeAdapter(process),
        runner_path="/tmp/predict_rlm_runner.py",
    )

    try:
        assert interpreter.execute("print('expected output')") == ""
    finally:
        reset_debug_logger_for_tests()

    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("{")
    ]
    empty_events = [
        record
        for record in records
        if record.get("event") == "terminal_bench.runner.empty_execute_output"
    ]

    assert empty_events
    event = empty_events[0]
    assert event["request_id"] == 1
    assert event["code_len"] == len("print('expected output')")
    assert event["code_hash"]
    assert event["output_len"] == 0
    assert event["code_preview"] == "print('expected output')"


def test_execute_strips_python_and_repl_fences_before_runner_request() -> None:
    process = FakeProcess(
        [
            {"id": 1, "ok": True, "result": {"output": "python\n"}},
            {"id": 2, "ok": True, "result": {"output": "repl\n"}},
        ]
    )
    interpreter = TerminalBenchRunnerClientAdapter(
        object(),
        container_adapter=FakeAdapter(process),
        runner_path="/tmp/predict_rlm_runner.py",
    )

    assert interpreter.execute("```python\nprint('python')\n```") == "python\n"
    assert interpreter.execute("```repl\nprint('repl')\n```") == "repl\n"

    assert process.requests[0]["params"] == {"code": "print('python')"}
    assert process.requests[1]["params"] == {"code": "print('repl')"}


def test_local_runner_preserves_stdout_from_repl_fenced_code(
    local_runner_interpreter: LocalProcessRunnerClientAdapter,
) -> None:
    result = local_runner_interpreter.execute(
        "```repl\nprint('local stdout survives')\n```"
    )

    assert result == "local stdout survives\n"


def test_local_runner_recoverable_timeout_preserves_streams_and_live_state(
    local_runner_interpreter: LocalProcessRunnerClientAdapter,
) -> None:
    assert local_runner_interpreter.execute("state = 'survived'\nprint('set')") == "set\n"

    timeout_result = local_runner_interpreter.execute(
        "import sys, time\n"
        "print('before timeout')\n"
        "print('stderr before timeout', file=sys.stderr)\n"
        "sys.stdout.flush(); sys.stderr.flush()\n"
        "time.sleep(30)\n",
        timeout=0.2,
    )
    followup = local_runner_interpreter.execute(
        "print('state' in globals())\nprint('fresh kernel')"
    )

    assert "[Timeout] Iteration execution timed out after 0.2s" in timeout_result
    assert "[stdout]\nbefore timeout" in timeout_result
    assert "[stderr]\nstderr before timeout" in timeout_result
    assert timeout_result.timeout_seconds == 0.2
    assert timeout_result.state == {
        "preserved": True,
        "source": "live_kernel",
        "scope": "full_live",
    }
    assert timeout_result.state_preserved is True
    assert followup == "True\nfresh kernel\n"


def test_local_runner_surfaces_subprocess_failure_and_survives(
    local_runner_interpreter: LocalProcessRunnerClientAdapter,
) -> None:
    with pytest.raises(CodeInterpreterError) as exc_info:
        local_runner_interpreter.execute(
            "import subprocess, sys\n"
            "subprocess.run(\n"
            "    [sys.executable, '-c', 'import sys; print(\"child failed\", file=sys.stderr); sys.exit(7)'],\n"
            "    capture_output=True,\n"
            "    text=True,\n"
            "    check=True,\n"
            ")\n",
            timeout=2,
        )
    followup = local_runner_interpreter.execute("print('after subprocess failure')")

    message = str(exc_info.value)
    assert "CalledProcessError" in message
    assert "non-zero exit status 7" in message
    assert followup == "after subprocess failure\n"


def test_local_runner_unbounded_runner_exit_returns_error_and_supervisor_survives(
    local_runner_interpreter: LocalProcessRunnerClientAdapter,
) -> None:
    with pytest.raises(CodeInterpreterError) as exc_info:
        local_runner_interpreter.execute("import os\nos._exit(7)")
    followup = local_runner_interpreter.execute("print('after runner child exit')")

    message = str(exc_info.value)
    assert "RuntimeError" in message
    assert "execution runner exited without a result" in message
    assert "exitcode=7" in message
    assert followup == "after runner child exit\n"


def test_local_runner_protocol_stdin_is_isolated_from_user_subprocesses(
    local_runner_interpreter: LocalProcessRunnerClientAdapter,
) -> None:
    local_runner_interpreter.execute("sentinel = 123")

    result = local_runner_interpreter.execute(
        "import subprocess, sys\n"
        "child = subprocess.run(\n"
        "    [sys.executable, '-c', 'import os; print(os.read(0, 1))'],\n"
        "    capture_output=True,\n"
        "    text=True,\n"
        "    timeout=0.5,\n"
        ")\n"
        "print(child.stdout.strip())\n",
        timeout=2,
    )
    followup = local_runner_interpreter.execute("print(sentinel)")

    assert result == "b''\n"
    assert followup == "123\n"


def test_execute_maps_structured_timeout_as_recoverable_observation() -> None:
    process = FakeProcess(
        [
            {
                "id": 1,
                "ok": True,
                "result": {
                    "timeout": {"seconds": 0.2},
                    "stdout": "before\n",
                    "stderr": "warn\n",
                },
            },
            {"id": 2, "ok": True, "result": {"output": "still alive\n"}},
        ]
    )
    interpreter = TerminalBenchRunnerClientAdapter(
        object(),
        container_adapter=FakeAdapter(process),
        runner_path="/tmp/predict_rlm_runner.py",
    )

    timeout_result = interpreter.execute("while True: pass", timeout=0.2)
    followup = interpreter.execute("print('still alive')")

    assert "[Timeout] Iteration execution timed out after 0.2s" in timeout_result
    assert "[stdout]\nbefore" in timeout_result
    assert "[stderr]\nwarn" in timeout_result
    assert timeout_result.timeout_seconds == 0.2
    assert followup == "still alive\n"
    assert process.killed is False


def test_user_subprocess_stdin_is_isolated_from_terminal_bench_runner_protocol() -> None:
    processes: list[subprocess.Popen[str]] = []

    class LocalAdapter:
        def copy_to(self, host_path: str, container_path: str) -> None:
            return None

        def copy_from(self, container_path: str, host_path: str) -> None:
            return None

        def exec(self, command: list[str], *, timeout: float | None = None):
            return SimpleNamespace(stdout="", stderr="", returncode=0)

        def start_exec(
            self,
            command: list[str],
            *,
            workdir: str | None = None,
            timeout: float | None = None,
        ):
            process = subprocess.Popen(
                [sys.executable, "-u", "-c", runner_source()],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            processes.append(process)
            return process

    interpreter = TerminalBenchRunnerClientAdapter(
        object(),
        container_adapter=LocalAdapter(),
        runner_path="/tmp/predict_rlm_runner.py",
        exec_timeout=30,
    )
    try:
        interpreter.execute("sentinel = 123\nprint('set')")
        result = interpreter.execute(
            "import subprocess, sys\n"
            "subprocess.run(\n"
            "    [sys.executable, '-c', 'import os; print(os.read(0, 1))'],\n"
            "    capture_output=True,\n"
            "    text=True,\n"
            "    timeout=0.2,\n"
            ")\n",
            timeout=1,
        )
        followup = interpreter.execute("print(sentinel)")
    finally:
        interpreter.shutdown()
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2)

    assert len(processes) == 1
    assert result == ""
    assert followup == "123\n"


def test_aexecute_accepts_lm_selected_execution_timeout() -> None:
    async def scenario() -> None:
        process = FakeProcess(
            [{"id": 1, "ok": True, "result": {"output": "bounded async\n"}}]
        )
        interpreter = TerminalBenchRunnerClientAdapter(
            object(),
            container_adapter=FakeAdapter(process),
            runner_path="/tmp/predict_rlm_runner.py",
        )

        result = await interpreter.aexecute("print('bounded async')", timeout=3)

        assert result == "bounded async\n"
        assert process.requests[0]["params"] == {
            "code": "print('bounded async')",
            "execution_timeout_seconds": 3.0,
        }

    asyncio.run(scenario())


def test_lm_selected_silent_runner_timeout_is_recoverable_and_restarts() -> None:
    processes: list[subprocess.Popen[str]] = []

    class SilentAdapter:
        def copy_to(self, host_path: str, container_path: str) -> None:
            return None

        def exec(self, command: list[str], *, timeout: float | None = None):
            return SimpleNamespace(stdout="", stderr="", returncode=0)

        def start_exec(
            self,
            command: list[str],
            *,
            workdir: str | None = None,
            timeout: float | None = None,
        ):
            if not processes:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-c",
                        "import sys, time\nsys.stdin.readline()\ntime.sleep(30)\n",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            else:
                process = subprocess.Popen(
                    [sys.executable, "-u", "-c", runner_source()],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            processes.append(process)
            return process

    interpreter = TerminalBenchRunnerClientAdapter(
        object(),
        container_adapter=SilentAdapter(),
        runner_path="/tmp/predict_rlm_runner.py",
        exec_timeout=30,
        recoverable_timeout_grace=0.1,
    )
    start = time.monotonic()
    try:
        timeout_result = interpreter.execute("print('never returns')", timeout=0.2)
        elapsed = time.monotonic() - start
        followup = interpreter.execute("print('fresh runner')")
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2)

    assert 0.25 <= elapsed < 1
    assert len(processes) >= 2
    assert processes[0].poll() is not None
    assert "[Timeout] Iteration execution timed out after 0.2s" in timeout_result
    assert "Terminal-Bench supervisor request timed out after 0.3s" in timeout_result
    assert "copied supervisor process was killed and restarted" in timeout_result
    assert "Python globals from the timed-out supervisor were lost" in timeout_result
    assert timeout_result.timeout_seconds == 0.2
    assert followup == "fresh runner\n"


def test_execute_after_structured_timeout_restarts_dead_runner_with_diagnostic() -> None:
    first_process = FakeProcess(
        [
            {
                "id": 1,
                "ok": True,
                "result": {
                    "timeout": {"seconds": 0.2},
                    "stdout": "before timeout\n",
                    "stderr": "command timed out\n",
                },
            },
        ],
        stderr="runner stderr tail\n",
    )
    restarted_process = FakeProcess([])
    adapter = FakeAdapter([first_process, restarted_process])
    interpreter = TerminalBenchRunnerClientAdapter(
        object(),
        container_adapter=adapter,
        runner_path="/tmp/predict_rlm_runner.py",
    )

    timeout_result = interpreter.execute("run_slow_command()", timeout=0.2)
    first_process.returncode = 137
    restart_result = interpreter.execute("print(existing_global)", timeout=0.2)

    assert "[Timeout] Iteration execution timed out after 0.2s" in timeout_result
    assert len(adapter.started) == 2
    assert restarted_process.requests == []
    assert "Terminal-Bench supervisor exited after the previous execute response" in restart_result
    assert "The copied supervisor process was restarted" in restart_result
    assert "Python globals from the prior supervisor were lost" in restart_result
    assert "supervisor_returncode=137" in restart_result
    assert "previous_request_id=1" in restart_result
    assert "previous_method=execute" in restart_result
    assert "previous_execution_timeout_seconds=0.2" in restart_result
    assert "runner stderr tail" in restart_result


def test_execute_after_structured_error_restarts_dead_runner_with_diagnostic() -> None:
    first_process = FakeProcess(
        [
            {
                "id": 1,
                "error": {
                    "code": -32000,
                    "message": "Command 'node /app/vm.js' timed out after 45 seconds",
                    "data": {
                        "type": "RuntimeError",
                        "message": "Command 'node /app/vm.js' timed out after 45 seconds",
                    },
                },
            },
        ]
    )
    restarted_process = FakeProcess([])
    adapter = FakeAdapter([first_process, restarted_process])
    interpreter = TerminalBenchRunnerClientAdapter(
        object(),
        container_adapter=adapter,
        runner_path="/tmp/predict_rlm_runner.py",
    )

    with pytest.raises(CodeInterpreterError, match="node /app/vm.js"):
        interpreter.execute("run_node()", timeout=45)
    first_process.returncode = 0
    restart_result = interpreter.execute("print('next iteration')", timeout=45)

    assert len(adapter.started) == 2
    assert restarted_process.requests == []
    assert "Terminal-Bench supervisor exited after the previous execute response" in restart_result
    assert "previous_response=structured_error" in restart_result
    assert "previous_execution_timeout_seconds=45" in restart_result
    assert "supervisor_returncode=0" in restart_result


def test_execute_maps_runner_errors() -> None:
    process = FakeProcess(
        [
            {
                "id": 1,
                "ok": False,
                "error": {"type": "NameError", "message": "missing_name"},
            }
        ]
    )
    interpreter = TerminalBenchRunnerClientAdapter(
        object(),
        container_adapter=FakeAdapter(process),
        runner_path="/tmp/predict_rlm_runner.py",
    )

    with pytest.raises(CodeInterpreterError, match="NameError: missing_name"):
        interpreter.execute("print(missing_name)")


def test_file_ops_use_container_adapter_not_host_filesystem(tmp_path: Path) -> None:
    process = FakeProcess([])
    adapter = FakeAdapter(process)
    adapter.exec_results.append(SimpleNamespace(stdout="", stderr="", returncode=0))
    adapter.exec_results.append(
        SimpleNamespace(
            stdout=json.dumps(["/sandbox/output/result.txt"]),
            stderr="",
            returncode=0,
        )
    )
    interpreter = TerminalBenchRunnerClientAdapter(object(), container_adapter=adapter)

    source = tmp_path / "source.txt"
    target = tmp_path / "target.txt"
    source.write_text("input", encoding="utf-8")

    interpreter.mount_file_at(str(source), "/sandbox/input/source.txt")
    interpreter.mkdir_p("/sandbox/output")
    assert interpreter.list_dir("/sandbox/output") == ["/sandbox/output/result.txt"]
    interpreter.sync_file_to("/sandbox/output/result.txt", str(target))

    assert adapter.copied_to == [(str(source), "/sandbox/input/source.txt")]
    assert adapter.exec_calls[0] == ["mkdir", "-p", "/sandbox/output"]
    assert adapter.exec_calls[1][:3] == ["python3", "-c", interpreter._LIST_DIR_SCRIPT]
    assert adapter.exec_calls[1][3] == "/sandbox/output"
    assert adapter.copied_from == [("/sandbox/output/result.txt", str(target))]
    assert not target.exists()


def test_terminal_bench_adapter_writes_runner_source_with_docker_exec(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []
    container = SimpleNamespace(id="container-123")
    session = SimpleNamespace(container=container)

    def fake_run(cmd, **kwargs):
        calls.append({"cmd": cmd, "kwargs": kwargs})
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = HarborContainerAdapter(session)
    adapter.install_runner_script(runner_source(), "/tmp/predict_rlm_runner.py")

    assert calls == [
        {
            "cmd": [
                "docker",
                "exec",
                "-i",
                "container-123",
                "sh",
                "-c",
                "cat > /tmp/predict_rlm_runner.py",
            ],
            "kwargs": {
                "input": runner_source(),
                "text": True,
                "capture_output": True,
                "check": False,
                "timeout": None,
            },
        }
    ]


def test_terminal_bench_adapter_starts_jsonl_runner_with_docker_exec(
    monkeypatch,
) -> None:
    popen_calls: list[dict[str, object]] = []
    process = SimpleNamespace(stdin=object(), stdout=object(), stderr=object())
    container = SimpleNamespace(id="container-123")
    session = SimpleNamespace(container=container, _user="bench-user")

    def fake_popen(cmd, **kwargs):
        popen_calls.append({"cmd": cmd, "kwargs": kwargs})
        return process

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    adapter = HarborContainerAdapter(session)
    assert (
        adapter.start_exec(
            ["python3", "-u", "/tmp/predict_rlm_runner.py"],
            workdir="/workspace",
            timeout=123,
        )
        is process
    )

    assert popen_calls == [
        {
            "cmd": [
                "docker",
                "exec",
                "-i",
                "-w",
                "/workspace",
                "-u",
                "bench-user",
                "container-123",
                "python3",
                "-u",
                "/tmp/predict_rlm_runner.py",
            ],
            "kwargs": {
                "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
            },
        }
    ]


def test_terminal_bench_minimal_adapter_rejects_file_sync_operations() -> None:
    container = SimpleNamespace(id="container-123")
    session = SimpleNamespace(container=container)
    adapter = HarborContainerAdapter(session)
    interpreter = TerminalBenchRunnerClientAdapter(session, container_adapter=adapter)

    with pytest.raises(NotImplementedError, match="minimal smoke adapter.*file sync"):
        interpreter.mount_file_at("/host/input.txt", "/container/input.txt")
    with pytest.raises(NotImplementedError, match="minimal smoke adapter.*file sync"):
        interpreter.sync_file_to("/container/output.txt", "/host/output.txt")
    with pytest.raises(NotImplementedError, match="minimal smoke adapter.*list_dir"):
        interpreter.list_dir("/container")


def test_host_tool_call_timeout_returns_bounded_tool_error() -> None:
    def slow_predict() -> str:
        time.sleep(0.8)
        return "unreachable"

    process = FakeProcess(
        [
            {"id": 1, "ok": True, "result": {}},
            {
                "jsonrpc": "2.0",
                "id": 99,
                "method": "tool_call",
                "params": {"name": "predict", "args": [], "kwargs": {}},
            },
            {"id": 2, "ok": True, "result": {"output": "handled timeout\n"}},
        ]
    )
    interpreter = TerminalBenchRunnerClientAdapter(
        object(),
        container_adapter=FakeAdapter(process),
        tools={"predict": slow_predict},
        runner_path="/tmp/predict_rlm_runner.py",
        recoverable_timeout_grace=0.2,
    )

    start = time.monotonic()
    result = interpreter.execute("await predict()", timeout=0.1)
    elapsed = time.monotonic() - start

    assert elapsed < 0.6
    assert result == "handled timeout\n"
    assert len(process.requests) == 3
    tool_response = process.requests[2]
    assert tool_response["id"] == 99
    assert "error" in tool_response
    assert "timed out" in tool_response["error"]["message"]
    assert process.killed is False
