from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput  # noqa: E402
from terminal_bench_rlm.tools.container_runner import HarborEnvironmentInterpreter  # noqa: E402


class FakeEnvironment:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str]] = []
        self.downloads: list[tuple[str, str]] = []
        self.commands: list[dict[str, object]] = []

    async def upload_file(self, host_path: str, environment_path: str) -> None:
        self.uploads.append((host_path, environment_path))

    async def download_file(self, environment_path: str, host_path: str) -> None:
        self.downloads.append((environment_path, host_path))
        Path(host_path).write_text("downloaded", encoding="utf-8")

    async def exec(self, **kwargs):
        self.commands.append(kwargs)
        command = str(kwargs["command"])
        if "SUBMIT" in command:
            return SimpleNamespace(
                return_code=0,
                stdout='{"id": 1, "ok": true, "result": {"final": {"answer": "done"}}}\n',
                stderr="",
            )
        return SimpleNamespace(return_code=0, stdout='{"id": 1, "ok": true, "result": {"output": "ok"}}\n', stderr="")


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


def test_harbor_environment_interpreter_executes_one_shot_runner_and_unwraps_submit() -> None:
    async def scenario() -> None:
        env = FakeEnvironment()
        interpreter = HarborEnvironmentInterpreter(env, loop=asyncio.get_running_loop())

        result = await asyncio.to_thread(interpreter.execute, 'SUBMIT(answer="done")')

        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "done"}
        assert env.uploads[0][1] == "/tmp/predict_rlm_runner.py"
        assert env.commands[-1]["cwd"] is None
        assert env.commands[-1]["timeout_sec"] == 900

    asyncio.run(scenario())


def test_harbor_environment_interpreter_rejects_host_callback_tools() -> None:
    try:
        HarborEnvironmentInterpreter(object(), loop=asyncio.new_event_loop(), tools={"lookup": lambda: None})
    except ValueError as exc:
        assert "host callback tools" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_harbor_environment_interpreter_rejects_tool_call_output() -> None:
    class ToolCallEnvironment(FakeEnvironment):
        async def exec(self, **kwargs):
            return SimpleNamespace(
                return_code=0,
                stdout='{"id": 99, "method": "tool_call", "params": {"name": "lookup"}}\n',
                stderr="",
            )

    async def scenario() -> None:
        interpreter = HarborEnvironmentInterpreter(ToolCallEnvironment(), loop=asyncio.get_running_loop())
        try:
            await asyncio.to_thread(interpreter.execute, "await lookup()")
        except CodeInterpreterError as exc:
            assert "host callback tools" in str(exc)
        else:
            raise AssertionError("expected CodeInterpreterError")

    asyncio.run(scenario())
