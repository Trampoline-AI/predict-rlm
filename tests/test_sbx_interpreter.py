"""Tests for the Docker Sandboxes interpreter backend."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Annotated

import pytest
from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput

from predict_rlm.files import SyncedFile
from predict_rlm.interpreter import SandboxFatalError
from predict_rlm.interpreters import DEFAULT_SBX_TEMPLATE, SbxConfig, SbxInterpreter, SbxPool

RUNNER_PATH = Path(__file__).parents[1] / "src" / "predict_rlm" / "sandbox" / "python_runner.py"


def _real_sbx_available() -> bool:
    if os.environ.get("PREDICT_RLM_RUN_SBX_TESTS") != "1":
        return False
    if shutil.which("sbx") is None:
        return False
    return subprocess.run(
        ["sbx", "ls"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    ).returncode == 0


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

    def request(self, method: str, params: dict | None = None) -> dict:
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
        return json.loads(self.proc.stdout.readline())

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


class TestPythonRunnerProtocol:
    def test_execute_captures_output_and_persists_globals(self, runner: LocalRunner):
        first = runner.request("execute", {"code": "x = 40\nprint('ready')"})
        second = runner.request("execute", {"code": "x += 2\nprint(x)"})

        assert first["result"]["output"].strip() == "ready"
        assert second["result"]["output"].strip() == "42"

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

    def test_syntax_error_uses_json_rpc_error(self, runner: LocalRunner):
        result = runner.request("execute", {"code": "for"})

        assert result["error"]["data"]["type"] == "SyntaxError"

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
                    "from pathlib import Path\n"
                    "text = Path('/sandbox/input/source/input.txt').read_text()\n"
                    "Path('/sandbox/output/result/output.txt').write_text(text + ' world')"
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


class TestSbxInterpreterLocalRunner:
    def make_interpreter(self, tmp_path: Path) -> SbxInterpreter:
        return SbxInterpreter(
            config=SbxConfig(name="local-test"),
            preinstall_packages=False,
            _runner_command=[sys.executable, "-u", str(RUNNER_PATH)],
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

    def test_execute_raises_recoverable_code_errors(self, tmp_path: Path):
        interpreter = self.make_interpreter(tmp_path)
        try:
            with pytest.raises(CodeInterpreterError, match="NameError"):
                interpreter.execute("print(missing_name)")
        finally:
            interpreter.shutdown()

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
                "Path('/sandbox/output/result/output.txt').write_text(text + ' sbx')",
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
            _runner_command=[sys.executable, "-c", "raise SystemExit(99)"],
            _staging_root=tmp_path / "staging",
        )
        source = tmp_path / "host-input.txt"
        source.write_text("host visible", encoding="utf-8")
        output = tmp_path / "host-output.txt"

        try:
            interpreter.mount_file_at(str(source), "/sandbox/input/source/input.txt")
            interpreter.mkdir_p("/sandbox/output/result/nested")
            staged_output = (
                tmp_path
                / "staging"
                / "sandbox"
                / "output"
                / "result"
                / "nested"
                / "output.txt"
            )
            staged_output.write_text("from staging", encoding="utf-8")

            files = interpreter.list_dir("/sandbox/output/result")
            interpreter.sync_file_to(
                "/sandbox/output/result/nested/output.txt",
                str(output),
            )
        finally:
            interpreter.shutdown()

        staged_input = (
            tmp_path / "staging" / "sandbox" / "input" / "source" / "input.txt"
        )
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
            _runner_command=[sys.executable, "-u", str(RUNNER_PATH)],
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
            _runner_command=[sys.executable, "-u", str(RUNNER_PATH)],
            _staging_root=staging_root,
        )
        source = tmp_path / "input.txt"
        source.write_text("host visible", encoding="utf-8")

        try:
            interpreter.mount_file_at(str(source), "/sandbox/input/source/input.txt")
        finally:
            interpreter.shutdown()

        assert staging_root.is_dir()
        assert (
            staging_root / "sandbox" / "input" / "source" / "input.txt"
        ).read_text(encoding="utf-8") == "host visible"

    def test_persist_preserves_owned_staging_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.chdir(tmp_path)
        interpreter = SbxInterpreter(
            config=SbxConfig(name="local-test", persist=True),
            preinstall_packages=False,
            _runner_command=[sys.executable, "-u", str(RUNNER_PATH)],
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
            _runner_command=[sys.executable, "-u", str(RUNNER_PATH)],
            _staging_root=tmp_path / "staging",
        )
        try:
            output = interpreter.execute(
                "result = await add(2, 3)\n"
                "print(result['total'])"
            )
        finally:
            interpreter.shutdown()

        assert output.strip() == "5"

    def test_host_tool_synced_file_writeback_updates_sandbox_file(
        self, tmp_path: Path
    ):
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
            _runner_command=[sys.executable, "-u", str(RUNNER_PATH)],
            _staging_root=tmp_path / "staging",
        )
        source = tmp_path / "input.txt"
        source.write_text("sandbox", encoding="utf-8")
        try:
            interpreter.mount_file_at(str(source), "/sandbox/input/source/input.txt")
            output = interpreter.execute(
                "from pathlib import Path\n"
                "result = await mutate('/sandbox/input/source/input.txt')\n"
                "print(result)\n"
                "print(Path('/sandbox/input/source/input.txt').read_text())"
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
            _runner_command=[sys.executable, "-u", str(RUNNER_PATH)],
            _staging_root=tmp_path / "staging",
        )
        source = tmp_path / "input.txt"
        source.write_text("sandbox", encoding="utf-8")
        try:
            interpreter.mount_file_at(str(source), "/sandbox/input/source/input.txt")
            output = interpreter.execute(
                "from pathlib import Path\n"
                "result = await mutate(path='/sandbox/input/source/input.txt')\n"
                "print(result)\n"
                "print(Path('/sandbox/input/source/input.txt').read_text())"
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
            _runner_command=[sys.executable, "-u", str(RUNNER_PATH)],
            _staging_root=tmp_path / "staging",
        )
        source = tmp_path / "input.txt"
        source.write_text("sandbox", encoding="utf-8")
        try:
            interpreter.mount_file_at(str(source), "/sandbox/input/source/input.txt")
            output = interpreter.execute(
                "from pathlib import Path\n"
                "received = await mutate(path='/sandbox/input/source/input.txt')\n"
                "print(received)\n"
                "print(Path('/sandbox/input/source/input.txt').read_text())"
            )
        finally:
            interpreter.shutdown()

        assert received_paths == [str(host_dir / "input.txt")]
        assert output.strip().splitlines() == [
            str(host_dir / "input.txt"),
            "sandbox + configured",
        ]
        assert (host_dir / "input.txt").read_text(encoding="utf-8") == (
            "sandbox + configured"
        )

    def test_concurrent_host_tool_calls_do_not_run_serially(self, tmp_path: Path):
        async def slow(value: int) -> int:
            await asyncio.sleep(0.35)
            return value

        interpreter = SbxInterpreter(
            config=SbxConfig(name="local-test", exec_timeout=3),
            tools={"slow": slow},
            preinstall_packages=False,
            _runner_command=[sys.executable, "-u", str(RUNNER_PATH)],
            _staging_root=tmp_path / "staging",
        )
        start = time.monotonic()
        try:
            output = interpreter.execute(
                "import asyncio\n"
                "results = await asyncio.gather(slow(1), slow(2))\n"
                "print(results)"
            )
        finally:
            interpreter.shutdown()
        elapsed = time.monotonic() - start

        assert output.strip() == "[1, 2]"
        assert elapsed < 0.6

    def test_request_timeout_fires_when_runner_stays_silent(self, tmp_path: Path):
        interpreter = SbxInterpreter(
            config=SbxConfig(name="silent-test", exec_timeout=0.2),
            preinstall_packages=False,
            _runner_command=[
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

        interpreter._start_sbx_and_build_runner_command()

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

        interpreter._start_sbx_and_build_runner_command()

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

        interpreter._start_sbx_and_build_runner_command()

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

        command = interpreter._start_sbx_and_build_runner_command()
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

        interpreter._start_sbx_and_build_runner_command()

        create_cmd = commands[0]
        workspace_arg = f"{tmp_path / 'staging'}:ro"
        assert create_cmd[:4] == ["sbx", "create", "shell", workspace_arg]
        assert create_cmd[4:6] == [str(extra_one), str(extra_two)]

    def test_default_workspace_is_staging_root_not_repo(
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

        command = interpreter._start_sbx_and_build_runner_command()

        create_cmd = commands[0]
        assert create_cmd[:4] == ["sbx", "create", "shell", str(tmp_path / "staging")]
        assert str(Path.cwd()) not in create_cmd
        assert command[:5] == ["sbx", "exec", "-i", "-w", str(tmp_path / "staging")]

    def test_runner_command_uses_python3_executable(self, monkeypatch, tmp_path: Path):
        def fake_run(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, stdout="created-name\n", stderr="")

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/sbx")
        monkeypatch.setattr(subprocess, "run", fake_run)
        interpreter = SbxInterpreter(
            config=SbxConfig(name="created-name"),
            preinstall_packages=False,
            _staging_root=tmp_path / "staging",
        )

        command = interpreter._start_sbx_and_build_runner_command()

        assert "python" not in command
        runner_path = tmp_path / "staging" / ".predict_rlm_runner" / "python_runner.py"
        assert runner_path.read_text(encoding="utf-8") == RUNNER_PATH.read_text(
            encoding="utf-8"
        )
        assert command[-3:] == ["python3", "-u", str(runner_path)]

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

    def test_package_bootstrap_uses_docker_sandbox_safe_pip(
        self, monkeypatch, tmp_path: Path
    ):
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
            _runner_command=[sys.executable, "-u", str(RUNNER_PATH)],
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
            _runner_command=[sys.executable, "-u", str(RUNNER_PATH)],
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
            _runner_command=[sys.executable, "-u", str(RUNNER_PATH)],
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
