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

from predict_rlm.interpreter import SandboxFatalError

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from terminal_bench_rlm.tools.container_runner import (  # noqa: E402
    TERMINAL_BENCH_RECOVERABLE_TIMEOUT_GRACE_SECONDS,
    HarborContainerAdapter,
    HarborEnvironmentInterpreter,
    TerminalBenchRunnerInterpreter,
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


def test_execute_reset_shutdown_requests_and_maps_success() -> None:
    process = FakeProcess(
        [
            {"id": 1, "ok": True, "result": {"output": "hi\n"}},
            {"id": 2, "ok": True, "result": {}},
            {"id": 3, "ok": True, "result": {"shutdown": True}},
        ]
    )
    adapter = FakeAdapter(process)
    interpreter = TerminalBenchRunnerInterpreter(
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


def test_execute_accepts_lm_selected_execution_timeout() -> None:
    process = FakeProcess(
        [{"id": 1, "ok": True, "result": {"output": "bounded\n"}}]
    )
    interpreter = TerminalBenchRunnerInterpreter(
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
    interpreter = TerminalBenchRunnerInterpreter(
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

    interpreter = TerminalBenchRunnerInterpreter(
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
        interpreter = TerminalBenchRunnerInterpreter(
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

    interpreter = TerminalBenchRunnerInterpreter(
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
    interpreter = TerminalBenchRunnerInterpreter(
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
    interpreter = TerminalBenchRunnerInterpreter(
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
    interpreter = TerminalBenchRunnerInterpreter(
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
    interpreter = TerminalBenchRunnerInterpreter(object(), container_adapter=adapter)

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
    interpreter = TerminalBenchRunnerInterpreter(session, container_adapter=adapter)

    with pytest.raises(NotImplementedError, match="minimal smoke adapter.*file sync"):
        interpreter.mount_file_at("/host/input.txt", "/container/input.txt")
    with pytest.raises(NotImplementedError, match="minimal smoke adapter.*file sync"):
        interpreter.sync_file_to("/container/output.txt", "/host/output.txt")
    with pytest.raises(NotImplementedError, match="minimal smoke adapter.*list_dir"):
        interpreter.list_dir("/container")


def test_harbor_environment_structured_timeout_is_recoverable_and_runner_survives() -> None:
    loop = asyncio.new_event_loop()

    class FakeHarborEnvironment:
        def __init__(self) -> None:
            self.uploads: list[tuple[str, str]] = []
            self.process = FakeProcess(
                [
                    {
                        "id": 1,
                        "ok": True,
                        "result": {
                            "timeout": {"seconds": 0.1},
                            "stdout": "before\n",
                            "stderr": "warn\n",
                        },
                    },
                    {"id": 2, "ok": True, "result": {"output": "after\n"}},
                ]
            )

        def upload_file(self, host_path: str, environment_path: str) -> None:
            self.uploads.append((host_path, environment_path))

        def start_exec(self, command, *, workdir=None, timeout=None):
            return self.process

    environment = FakeHarborEnvironment()
    interpreter = HarborEnvironmentInterpreter(
        environment,
        loop=loop,
        recoverable_timeout_grace=1.5,
    )

    try:
        timeout_result = interpreter.execute("while True: pass", timeout=0.1)
        followup = interpreter.execute("print('after')")
    finally:
        loop.close()

    assert "[Timeout] Iteration execution timed out after 0.1s" in timeout_result
    assert "[stdout]\nbefore" in timeout_result
    assert "[stderr]\nwarn" in timeout_result
    assert timeout_result.timeout_seconds == 0.1
    assert followup == "after\n"
    assert environment.uploads[0][1] == "/tmp/predict_rlm_runner.py"
    assert environment.process.killed is False


def test_harbor_environment_delayed_structured_timeout_uses_recovery_grace() -> None:
    loop = asyncio.new_event_loop()

    class SlowHarborEnvironment:
        def __init__(self) -> None:
            self.process = FakeProcess(
                [
                    (
                        1.2,
                        {
                            "id": 1,
                            "ok": True,
                            "result": {
                                "timeout": {"seconds": 0.05},
                                "stdout": "late\n",
                                "stderr": "",
                            },
                        },
                    ),
                    {"id": 2, "ok": True, "result": {"output": "after late\n"}},
                ]
            )

        def upload_file(self, host_path: str, environment_path: str) -> None:
            return None

        def start_exec(self, command, *, workdir=None, timeout=None):
            return self.process

    environment = SlowHarborEnvironment()
    interpreter = HarborEnvironmentInterpreter(environment, loop=loop)

    start = time.monotonic()
    try:
        timeout_result = interpreter.execute("while True: pass", timeout=0.05)
        followup = interpreter.execute("print('after late')")
    finally:
        loop.close()

    assert time.monotonic() - start >= 1.0
    assert "[Timeout] Iteration execution timed out after 0.05s" in timeout_result
    assert "[stdout]\nlate" in timeout_result
    assert followup == "after late\n"
    assert environment.process.killed is False


def test_harbor_environment_default_recoverable_timeout_grace_is_30s() -> None:
    from predict_rlm.execution_timeout import (
        DEFAULT_RECOVERABLE_EXECUTION_TIMEOUT_GRACE_SECONDS,
    )

    loop = asyncio.new_event_loop()

    class FakeHarborEnvironment:
        def __init__(self) -> None:
            self.process = FakeProcess([])

        def upload_file(self, host_path: str, environment_path: str) -> None:
            return None

        def start_exec(self, command, *, workdir=None, timeout=None):
            return self.process

    try:
        interpreter = HarborEnvironmentInterpreter(FakeHarborEnvironment(), loop=loop)
    finally:
        loop.close()

    assert DEFAULT_RECOVERABLE_EXECUTION_TIMEOUT_GRACE_SECONDS == 30.0
    assert (
        TERMINAL_BENCH_RECOVERABLE_TIMEOUT_GRACE_SECONDS
        == DEFAULT_RECOVERABLE_EXECUTION_TIMEOUT_GRACE_SECONDS
    )
    assert interpreter.recoverable_timeout_grace == 30.0


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
    interpreter = TerminalBenchRunnerInterpreter(
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


def test_harbor_environment_default_exec_timeout_is_fatal_and_killed_by_watchdog() -> None:
    loop = asyncio.new_event_loop()

    class SilentHarborEnvironment:
        def __init__(self) -> None:
            self.process = FakeProcess([])

        def upload_file(self, host_path: str, environment_path: str) -> None:
            return None

        def start_exec(self, command, *, workdir=None, timeout=None):
            return self.process

    environment = SilentHarborEnvironment()
    interpreter = HarborEnvironmentInterpreter(
        environment,
        loop=loop,
        exec_timeout=0.2,
    )

    start = time.monotonic()
    try:
        with pytest.raises(
            SandboxFatalError,
            match=r"Terminal-Bench supervisor request timed out after 0\.2s",
        ):
            interpreter.execute("print('never')")
        elapsed = time.monotonic() - start
    finally:
        loop.close()

    assert 0.15 <= elapsed < 1
    assert environment.process.killed is True
    assert environment.process.waited is True


def test_harbor_environment_list_dir_uses_python_fallback_resolver() -> None:
    loop = asyncio.new_event_loop()

    class FakeEnvironment:
        def __init__(self) -> None:
            self.commands: list[str] = []

        def exec(self, *, command, cwd=None, timeout_sec=None):
            self.commands.append(command)
            stdout = json.dumps(["/workspace/out.txt"]) if len(self.commands) == 2 else ""
            return SimpleNamespace(stdout=stdout, stderr="", return_code=0)

    environment = FakeEnvironment()
    interpreter = HarborEnvironmentInterpreter(environment, loop=loop)
    interpreter._run_coro = lambda result: result

    try:
        response = interpreter.list_dir("/workspace")
    finally:
        loop.close()

    assert response == ["/workspace/out.txt"]
    assert len(environment.commands) == 2
    command = environment.commands[1]
    assert "command -v python3" in command
    assert "elif command -v python" in command
    assert '"$_predict_rlm_python" -c' in command


def test_harbor_environment_starts_runner_with_docker_compose_exec(monkeypatch, tmp_path: Path) -> None:
    loop = asyncio.new_event_loop()
    popen_calls: list[dict[str, object]] = []
    process = FakeProcess([{"id": 1, "ok": True, "result": {"output": "ok\n"}}])
    compose_path = tmp_path / "compose.yaml"
    compose_path.write_text("services: {}", encoding="utf-8")

    class ComposeOnlyEnvironment:
        environment_dir = tmp_path
        session_id = "Path.Tracing_ABC123"
        default_user = "agent-user"

        def __init__(self) -> None:
            self.commands: list[str] = []
            self.uploads: list[tuple[str, str]] = []

        @property
        def _docker_compose_paths(self) -> list[Path]:
            return [compose_path]

        def _compose_env_vars(self, *, include_os_env: bool = True) -> dict[str, str]:
            return {"COMPOSE_ENV": "1"}

        def exec(self, *, command, cwd=None, timeout_sec=None):
            self.commands.append(command)
            return SimpleNamespace(stdout="", stderr="", return_code=0)

        def upload_file(self, host_path: str, environment_path: str) -> None:
            self.uploads.append((host_path, environment_path))

    def fake_popen(cmd, **kwargs):
        popen_calls.append({"cmd": cmd, "kwargs": kwargs})
        return process

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    environment = ComposeOnlyEnvironment()
    interpreter = HarborEnvironmentInterpreter(environment, loop=loop, workdir="/workspace")
    interpreter._run_coro = lambda result: result

    try:
        response = interpreter.execute("print('ok')")
    finally:
        loop.close()

    assert response == "ok\n"
    assert environment.uploads[0][1] == "/tmp/predict_rlm_runner.py"
    assert popen_calls == [
        {
            "cmd": [
                "docker",
                "compose",
                "--project-name",
                "path-tracing_abc123",
                "--project-directory",
                str(tmp_path),
                "-f",
                str(compose_path),
                "exec",
                "-T",
                "-w",
                "/workspace",
                "-u",
                "agent-user",
                "main",
                "python3",
                "-u",
                "/tmp/predict_rlm_runner.py",
            ],
            "kwargs": {
                "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "env": {"COMPOSE_ENV": "1"},
            },
        }
    ]


def test_harbor_environment_bootstraps_python_before_runner_execution() -> None:
    loop = asyncio.new_event_loop()

    class FakeEnvironment:
        def __init__(self) -> None:
            self.commands: list[str] = []
            self.uploads: list[tuple[str, str]] = []
            self.started: list[dict[str, object]] = []
            self.process = FakeProcess(
                [{"id": 1, "ok": True, "result": {"output": "ok\n"}}]
            )

        def exec(self, *, command, cwd=None, timeout_sec=None):
            self.commands.append(command)
            return SimpleNamespace(stdout="", stderr="", return_code=0)

        def upload_file(self, host_path: str, environment_path: str) -> None:
            self.uploads.append((host_path, environment_path))

        def start_exec(self, command, *, workdir=None, timeout=None):
            self.started.append({"command": command, "workdir": workdir, "timeout": timeout})
            return self.process

    environment = FakeEnvironment()
    interpreter = HarborEnvironmentInterpreter(environment, loop=loop)
    interpreter._run_coro = lambda result: result

    try:
        response = interpreter.execute("print('ok')")
    finally:
        loop.close()

    assert response == "ok\n"
    bootstrap_command = environment.commands[0]
    assert "command -v python3" in bootstrap_command
    assert "apt-get" in bootstrap_command
    assert "apk add" in bootstrap_command
    assert "python3" in bootstrap_command
    assert environment.uploads[0][1] == "/tmp/predict_rlm_runner.py"
    assert environment.started == [
        {
            "command": ["python3", "-u", "/tmp/predict_rlm_runner.py"],
            "workdir": None,
            "timeout": 900.0,
        }
    ]
