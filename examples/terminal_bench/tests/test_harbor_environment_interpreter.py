from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from dspy.primitives.code_interpreter import FinalOutput  # noqa: E402
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


def test_harbor_environment_interpreter_uses_public_environment_file_apis(tmp_path: Path) -> None:
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
            "id": 1,
            "ok": True,
            "result": {"type": "json", "value": {"answer": "4"}},
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
