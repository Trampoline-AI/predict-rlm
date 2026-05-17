from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from dspy.primitives.code_interpreter import CodeInterpreterError

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from terminal_bench_rlm.tools.container_runner import (  # noqa: E402
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
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.stdout = FakePipe()
        self.stderr = FakePipe()
        self.stdin = FakePipe(self._on_stdin)
        self.killed = False
        self.waited = False

    def _on_stdin(self, data: str) -> None:
        request = json.loads(data)
        self.requests.append(request)
        if self.responses:
            self.stdout.lines.append(json.dumps(self.responses.pop(0)) + "\n")

    def poll(self):
        return 1 if self.killed else None

    def wait(self, timeout=None):
        self.waited = True
        return 0

    def kill(self) -> None:
        self.killed = True


class FakeAdapter:
    def __init__(self, process: FakeProcess) -> None:
        self.process = process
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
        return self.process


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


def test_harbor_environment_runner_uses_python_fallback_resolver() -> None:
    loop = asyncio.new_event_loop()

    class FakeEnvironment:
        def __init__(self) -> None:
            self.commands: list[str] = []

        def exec(self, *, command, cwd=None, timeout_sec=None):
            self.commands.append(command)
            return SimpleNamespace(
                stdout=json.dumps({"id": 1, "ok": True, "result": {"output": "ok"}}) + "\n",
                stderr="",
                return_code=0,
            )

    environment = FakeEnvironment()
    interpreter = HarborEnvironmentInterpreter(environment, loop=loop)
    interpreter._runner_installed = True
    interpreter._run_coro = lambda result: result

    try:
        response = interpreter._run_runner_messages(
            [{"id": 1, "method": "execute", "params": {"code": "print('ok')"}}],
            1,
        )
    finally:
        loop.close()

    assert response["ok"] is True
    assert len(environment.commands) == 2
    command = environment.commands[1]
    assert "command -v python3" in command
    assert "elif command -v python" in command
    assert '"$_predict_rlm_python" -u /tmp/predict_rlm_runner.py' in command


def test_harbor_environment_bootstraps_python_before_runner_execution() -> None:
    loop = asyncio.new_event_loop()

    class FakeEnvironment:
        def __init__(self) -> None:
            self.commands: list[str] = []

        def exec(self, *, command, cwd=None, timeout_sec=None):
            self.commands.append(command)
            if len(self.commands) == 1:
                return SimpleNamespace(stdout="", stderr="", return_code=0)
            return SimpleNamespace(
                stdout=json.dumps({"id": 1, "ok": True, "result": {"output": "ok"}}) + "\n",
                stderr="",
                return_code=0,
            )

    environment = FakeEnvironment()
    interpreter = HarborEnvironmentInterpreter(environment, loop=loop)
    interpreter._runner_installed = True
    interpreter._run_coro = lambda result: result

    try:
        response = interpreter._run_runner_messages(
            [{"id": 1, "method": "execute", "params": {"code": "print('ok')"}}],
            1,
        )
    finally:
        loop.close()

    assert response["ok"] is True
    bootstrap_command = environment.commands[0]
    assert "command -v python3" in bootstrap_command
    assert "apt-get" in bootstrap_command
    assert "apk add" in bootstrap_command
    assert "python3" in bootstrap_command
    assert '"$_predict_rlm_python" -u /tmp/predict_rlm_runner.py' in environment.commands[1]
