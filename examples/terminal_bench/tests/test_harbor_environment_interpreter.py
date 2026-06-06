from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from dspy.primitives.code_interpreter import FinalOutput  # noqa: E402
from terminal_bench_rlm.tools import container_runner  # noqa: E402
from terminal_bench_rlm.tools.container_runner import HarborEnvironmentInterpreter  # noqa: E402


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

    def readline_timeout(self, timeout: float) -> str | None:
        return self.readline() or None

    def read(self) -> str:
        return "".join(self.lines)


class FakeInteractiveProcess:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.stdout = FakePipe()
        self.stderr = FakePipe()
        self.stdin = FakePipe(self._on_stdin)
        self.killed = False
        self.waited = False

    def _on_stdin(self, data: str) -> None:
        self.requests.append(json.loads(data))
        if self.responses:
            self.stdout.lines.append(json.dumps(self.responses.pop(0)) + "\n")

    def poll(self):
        return 1 if self.killed else None

    def wait(self, timeout=None):
        self.waited = True
        return 0

    def kill(self) -> None:
        self.killed = True


class FakeEnvironment:
    def __init__(self, process: FakeInteractiveProcess | None = None) -> None:
        self.uploads: list[tuple[str, str]] = []
        self.downloads: list[tuple[str, str]] = []
        self.commands: list[dict[str, object]] = []
        self.started: list[dict[str, object]] = []
        self.process = process

    async def upload_file(self, host_path: str, environment_path: str) -> None:
        self.uploads.append((host_path, environment_path))

    async def download_file(self, environment_path: str, host_path: str) -> None:
        self.downloads.append((environment_path, host_path))
        Path(host_path).write_text("downloaded", encoding="utf-8")

    async def exec(self, **kwargs):
        self.commands.append(kwargs)
        return SimpleNamespace(
            return_code=0,
            stdout='{"id": 1, "ok": true, "result": {"output": "ok"}}\n',
            stderr="",
        )

    def start_exec(self, command, *, workdir=None, timeout=None):
        self.started.append({"command": command, "workdir": workdir, "timeout": timeout})
        if self.process is None:
            raise AssertionError("no fake process configured")
        return self.process


class FakeDaytonaCommand:
    def __init__(self, exit_code: int | None = None) -> None:
        self.exit_code = exit_code


class FakeDaytonaSessionResponse:
    cmd_id = "cmd-1"


class FakeDaytonaLogs:
    def __init__(self, stdout: str = "", stderr: str = "") -> None:
        self.stdout = stdout
        self.stderr = stderr


class FakeDaytonaProcessApi:
    def __init__(self, responses: list[dict], *, echo_inputs: bool = False) -> None:
        self.responses = list(responses)
        self.echo_inputs = echo_inputs
        self.sessions: list[str] = []
        self.commands: list[tuple[str, object, int | None]] = []
        self.inputs: list[str] = []
        self.stdout = ""
        self.exit_code: int | None = None

    async def create_session(self, session_id: str) -> None:
        self.sessions.append(session_id)

    async def execute_session_command(
        self, session_id: str, req: object, timeout: int | None = None
    ):
        self.commands.append((session_id, req, timeout))
        return FakeDaytonaSessionResponse()

    async def get_session_command(self, session_id: str, command_id: str):
        return FakeDaytonaCommand(self.exit_code)

    async def get_session_command_logs(self, session_id: str, command_id: str):
        return FakeDaytonaLogs(stdout=self.stdout)

    async def send_session_command_input(
        self, session_id: str, command_id: str, data: str
    ) -> None:
        self.inputs.append(data)
        if self.echo_inputs:
            self.stdout += data
        if self.responses:
            self.stdout += json.dumps(self.responses.pop(0)) + "\n"
        try:
            request = json.loads(data)
        except json.JSONDecodeError:
            return
        if request.get("method") == "shutdown":
            self.exit_code = 0


class FakeFragmentedDaytonaProcessApi(FakeDaytonaProcessApi):
    def __init__(self, stdout_snapshots: list[str]) -> None:
        super().__init__([])
        self.stdout_snapshots = list(stdout_snapshots)
        self._snapshot_index = 0

    async def get_session_command_logs(self, session_id: str, command_id: str):
        if self._snapshot_index < len(self.stdout_snapshots):
            self.stdout = self.stdout_snapshots[self._snapshot_index]
            self._snapshot_index += 1
        return FakeDaytonaLogs(stdout=self.stdout)


class FakeDaytonaSandbox:
    def __init__(self, process: FakeDaytonaProcessApi) -> None:
        self.process = process


class FakeDaytonaEnvironment(FakeEnvironment):
    start_exec = None

    def __init__(self, process: FakeDaytonaProcessApi) -> None:
        super().__init__()
        self._sandbox = FakeDaytonaSandbox(process)


def test_harbor_environment_interpreter_uses_public_environment_file_apis(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        env = FakeEnvironment()
        interpreter = HarborEnvironmentInterpreter(env, loop=asyncio.get_running_loop())

        host_path = tmp_path / "input.txt"
        host_path.write_text("hello", encoding="utf-8")
        await asyncio.to_thread(interpreter.mount_file_at, str(host_path), "/tmp/input.txt")
        await asyncio.to_thread(
            interpreter.sync_file_to,
            "/tmp/output.txt",
            str(tmp_path / "output.txt"),
        )

        assert env.uploads == [(str(host_path), "/tmp/input.txt")]
        assert env.downloads == [("/tmp/output.txt", str(tmp_path / "output.txt"))]

    asyncio.run(scenario())


def test_harbor_environment_interpreter_uses_daytona_session_exec(monkeypatch) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(
            container_runner,
            "_daytona_session_execute_request",
            lambda command, *, run_async: SimpleNamespace(command=command, run_async=run_async),
        )
        process = FakeDaytonaProcessApi([{"id": 1, "ok": True, "result": {"output": "ok\n"}}])
        env = FakeDaytonaEnvironment(process)
        interpreter = HarborEnvironmentInterpreter(env, loop=asyncio.get_running_loop())

        result = await asyncio.to_thread(interpreter.execute, "print('ok')")

        assert result == "ok\n"
        assert len(process.sessions) == 1
        session_id, request, timeout = process.commands[0]
        assert session_id == process.sessions[0]
        assert request.run_async is True
        assert request.command == "python3 -u /tmp/predict_rlm_runner.py"
        assert timeout == 900
        assert [json.loads(item)["method"] for item in process.inputs] == ["execute"]

    asyncio.run(scenario())


def test_harbor_environment_interpreter_ignores_echoed_daytona_stdin(monkeypatch) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(
            container_runner,
            "_daytona_session_execute_request",
            lambda command, *, run_async: SimpleNamespace(command=command, run_async=run_async),
        )
        process = FakeDaytonaProcessApi(
            [{"jsonrpc": "2.0", "id": 1, "result": {"output": "printed\n"}}],
            echo_inputs=True,
        )
        env = FakeDaytonaEnvironment(process)
        interpreter = HarborEnvironmentInterpreter(env, loop=asyncio.get_running_loop())
        try:
            result = await asyncio.to_thread(
                interpreter.execute, "```repl\nprint('printed')\n```"
            )
        finally:
            await asyncio.to_thread(interpreter.shutdown)

        assert result == "printed\n"
        assert json.loads(process.inputs[0])["params"] == {"code": "print('printed')"}

    asyncio.run(scenario())


def test_harbor_environment_interpreter_preserves_fragmented_response_after_stale_daytona_logs(
    monkeypatch,
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(
            container_runner,
            "_daytona_session_execute_request",
            lambda command, *, run_async: SimpleNamespace(command=command, run_async=run_async),
        )
        stale_output = (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "result": {"output": "old request\n"},
                }
            )
            + "\n"
        )
        stale_error = (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "error": {
                        "code": -32000,
                        "message": "Unknown method: None",
                        "data": {"type": "ValueError", "message": "Unknown method: None"},
                    },
                }
            )
            + "\n"
        )
        current = (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "result": {"output": "printed after stale\n"},
                    "id": 1,
                }
            )
            + "\n"
        )
        split_at = len('{"jsonrpc": "2.0", "')
        process = FakeFragmentedDaytonaProcessApi(
            [
                stale_output + stale_error + current[:split_at],
                stale_output + stale_error + current,
            ]
        )
        env = FakeDaytonaEnvironment(process)
        interpreter = HarborEnvironmentInterpreter(
            env,
            loop=asyncio.get_running_loop(),
            exec_timeout=0.2,
        )

        result = await asyncio.to_thread(interpreter.execute, "print('printed after stale')")

        assert result == "printed after stale\n"
        assert json.loads(process.inputs[0])["params"]["code"] == "print('printed after stale')"

    asyncio.run(scenario())


def test_harbor_environment_interpreter_cancels_active_supervisor_without_shutdown_race() -> None:
    class BlockingPipe(FakePipe):
        def __init__(self, process: FakeInteractiveProcess) -> None:
            super().__init__()
            self.process = process

        def readline_timeout(self, timeout: float) -> str | None:
            end = time.monotonic() + min(timeout, 1.0)
            while not self.process.killed and time.monotonic() < end:
                time.sleep(0.01)
            return "" if self.process.killed else None

    async def scenario() -> None:
        process = FakeInteractiveProcess([])
        process.stdout = BlockingPipe(process)
        env = FakeEnvironment(process)
        interpreter = HarborEnvironmentInterpreter(env, loop=asyncio.get_running_loop())

        task = asyncio.create_task(interpreter.aexecute("while True: pass"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert process.killed
        assert interpreter._process is None
        assert [request["method"] for request in process.requests] == ["execute"]

    asyncio.run(scenario())


def test_harbor_environment_interpreter_uses_persistent_exec_for_predict_round_trip() -> None:
    async def scenario() -> None:
        process = FakeInteractiveProcess(
            [
                {"id": 1, "ok": True, "result": {}},
                {
                    "id": 1,
                    "method": "tool_call",
                    "params": {
                        "name": "predict",
                        "args": ["question -> answer"],
                        "kwargs": {"question": "2+2?"},
                    },
                },
                {"id": 2, "ok": True, "result": {"output": "4\n"}},
            ]
        )
        env = FakeEnvironment(process)
        predict_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def predict(*args, **kwargs):
            predict_calls.append((args, kwargs))
            return {"answer": "4"}

        interpreter = HarborEnvironmentInterpreter(
            env,
            loop=asyncio.get_running_loop(),
            tools={"predict": predict},
        )

        result = await asyncio.to_thread(
            interpreter.execute,
            "answer = await predict('question -> answer', question='2+2?')\n"
            "print(answer['answer'])",
        )

        assert result == "4\n"
        assert env.uploads[0][1] == "/tmp/predict_rlm_runner.py"
        assert env.started == [
            {
                "command": ["python3", "-u", "/tmp/predict_rlm_runner.py"],
                "workdir": None,
                "timeout": 900.0,
            }
        ]
        assert predict_calls == [(("question -> answer",), {"question": "2+2?"})]
        assert [request.get("method") for request in process.requests[:2]] == [
            "register_tools",
            "execute",
        ]
        assert process.requests[0]["params"] == {"tools": ["predict"]}
        assert process.requests[2] == {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"type": "json", "value": '{"answer": "4"}'},
        }

    asyncio.run(scenario())


def test_harbor_environment_interpreter_preserves_namespace_across_iterations() -> None:
    async def scenario() -> None:
        process = FakeInteractiveProcess(
            [
                {"id": 1, "ok": True, "result": {"output": ""}},
                {"id": 2, "ok": True, "result": {"output": "1\n"}},
            ]
        )
        env = FakeEnvironment(process)
        interpreter = HarborEnvironmentInterpreter(env, loop=asyncio.get_running_loop())

        first = await asyncio.to_thread(interpreter.execute, "x = 1")
        second = await asyncio.to_thread(interpreter.execute, "print(x)")

        assert first == ""
        assert second == "1\n"
        assert len(env.started) == 1
        assert [request["method"] for request in process.requests] == ["execute", "execute"]

    asyncio.run(scenario())


def test_harbor_environment_interpreter_unwraps_submit_on_persistent_runner() -> None:
    async def scenario() -> None:
        process = FakeInteractiveProcess(
            [{"id": 1, "ok": True, "result": {"final": {"answer": "done"}}}]
        )
        env = FakeEnvironment(process)
        interpreter = HarborEnvironmentInterpreter(env, loop=asyncio.get_running_loop())

        result = await asyncio.to_thread(interpreter.execute, 'SUBMIT(answer="done")')

        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "done"}

    asyncio.run(scenario())


def test_harbor_environment_interpreter_can_defer_next_submit_finalization() -> None:
    async def scenario() -> None:
        process = FakeInteractiveProcess(
            [{"id": 1, "ok": True, "result": {"submitted": {"answer": "done"}}}]
        )
        env = FakeEnvironment(process)
        interpreter = HarborEnvironmentInterpreter(env, loop=asyncio.get_running_loop())
        interpreter.defer_next_submit_finalization()

        result = await asyncio.to_thread(interpreter.execute, 'SUBMIT(answer="done")')

        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "done"}
        assert process.requests[0]["params"]["defer_final_output"] is True

    asyncio.run(scenario())


def test_harbor_environment_interpreter_requires_interactive_exec() -> None:
    class OneShotOnlyEnvironment(FakeEnvironment):
        start_exec = None

    async def scenario() -> None:
        interpreter = HarborEnvironmentInterpreter(
            OneShotOnlyEnvironment(),
            loop=asyncio.get_running_loop(),
            tools={"lookup": lambda: "value"},
        )
        with pytest.raises(TypeError, match="interactive exec"):
            await asyncio.to_thread(interpreter.execute, "print(await lookup())")

    asyncio.run(scenario())
