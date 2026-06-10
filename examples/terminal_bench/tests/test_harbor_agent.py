from __future__ import annotations

import asyncio
import inspect
import json
import logging
import shlex
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from terminal_bench_rlm.tools import remote_controller, tbench_agent  # noqa: E402


def _bootstrap_command_args(command: str) -> list[str]:
    return shlex.split(shlex.split(command)[-1])


def test_daytona_remote_agent_constructor_has_no_interpreter_mode_parameter() -> None:
    signature = inspect.signature(tbench_agent.DaytonaRemotePredictRLMAgent)

    assert "interpreter_mode" not in signature.parameters


def test_harbor_predict_rlm_base_agent_is_abstract(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="abstract"):
        tbench_agent.HarborPredictRLMBaseAgent(logs_dir=tmp_path)


class FakeDaytonaRemoteEnvironment:
    def __init__(self, *, answer: str = "remote answer") -> None:
        self.commands: list[str] = []
        self.uploads: list[tuple[str, str]] = []
        self.upload_dirs: list[tuple[str, str]] = []
        self.downloads: list[tuple[str, str]] = []
        self.payloads: list[dict[str, object]] = []
        self.command_timeouts: list[int | None] = []
        self.answer = answer

    def exec(self, *, command: str, timeout_sec: int | None = None):
        self.command_timeouts.append(timeout_sec)
        self.commands.append(command)
        if "terminal_bench_rlm.tools.remote_controller" in command:
            stdout = (
                "controller log\n"
                f"{tbench_agent.DAYTONA_REMOTE_RESULT_SENTINEL}"
                f"{json.dumps({'ok': True, 'answer': self.answer})}\n"
            )
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def upload_file(self, host_path: str, remote_path: str) -> None:
        self.uploads.append((host_path, remote_path))
        if remote_path.endswith(".json"):
            self.payloads.append(json.loads(Path(host_path).read_text(encoding="utf-8")))

    def upload_dir(self, host_path: str, remote_path: str) -> None:
        self.upload_dirs.append((host_path, remote_path))

    def download_file(self, remote_path: str, host_path: str) -> None:
        self.downloads.append((remote_path, host_path))
        with tarfile.open(host_path, "w:gz"):
            pass


def _write_remote_trace_archive(env: FakeDaytonaRemoteEnvironment, tmp_path: Path) -> None:
    def download_file(remote_path: str, host_path: str) -> None:
        env.downloads.append((remote_path, host_path))

        with tarfile.open(host_path, "w:gz") as archive:
            trace_path = tmp_path / "predict_rlm_trace_remote.json"
            trace_path.write_text('{"status":"completed","cost_usd":1.23}')
            archive.add(trace_path, arcname="predict_rlm_trace_remote.json")

    env.download_file = download_file


def test_daytona_remote_agent_downloads_remote_predict_rlm_traces(tmp_path: Path) -> None:
    env = FakeDaytonaRemoteEnvironment(answer="remote done")
    _write_remote_trace_archive(env, tmp_path)
    context = SimpleNamespace()
    agent = tbench_agent.DaytonaRemotePredictRLMAgent(logs_dir=tmp_path)

    asyncio.run(agent.run("solve remotely", env, context))

    assert context.answer == "remote done"
    assert env.downloads == [
        ("/tmp/predict_rlm_controller/logs.tar.gz", str(tmp_path / "predict_rlm_logs.tar.gz"))
    ]
    assert (tmp_path / "predict_rlm_trace_remote.json").read_text() == (
        '{"status":"completed","cost_usd":1.23}'
    )


def test_daytona_remote_agent_downloads_remote_predict_rlm_traces_on_cancellation(
    tmp_path: Path,
) -> None:
    env = FakeDaytonaRemoteEnvironment(answer="remote done")
    _write_remote_trace_archive(env, tmp_path)

    original_exec = env.exec

    def exec_cancel_controller(*, command: str, timeout_sec: int | None = None):
        if "terminal_bench_rlm.tools.remote_controller" in command:
            raise asyncio.CancelledError
        return original_exec(command=command, timeout_sec=timeout_sec)

    env.exec = exec_cancel_controller
    context = SimpleNamespace()
    agent = tbench_agent.DaytonaRemotePredictRLMAgent(logs_dir=tmp_path)

    try:
        asyncio.run(agent.run("solve remotely", env, context))
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("expected cancellation")

    assert env.downloads == [
        ("/tmp/predict_rlm_controller/logs.tar.gz", str(tmp_path / "predict_rlm_logs.tar.gz"))
    ]
    assert (tmp_path / "predict_rlm_trace_remote.json").read_text() == (
        '{"status":"completed","cost_usd":1.23}'
    )


def test_daytona_remote_agent_payload_is_non_secret_and_uses_remote_home(tmp_path: Path) -> None:
    env = FakeDaytonaRemoteEnvironment(answer="remote done")
    context = SimpleNamespace()
    agent = tbench_agent.DaytonaRemotePredictRLMAgent(
        logs_dir=tmp_path,
        extra_env={"OPENAI_API_KEY": "super-secret-token"},
        exec_timeout="123",
        interpreter_kwargs={"cwd": "/tmp/task"},
        lm="openai/gpt-5-mini",
        max_iterations="2",
    )

    asyncio.run(agent.setup(env))
    asyncio.run(agent.run("solve remotely", env, context))

    assert context.answer == "remote done"
    assert env.payloads
    payload_text = json.dumps(env.payloads[-1], sort_keys=True)
    assert "super-secret-token" not in payload_text
    assert "OPENAI_API_KEY" not in payload_text
    assert "submit_confirmation\":" not in payload_text
    assert env.payloads[-1]["submit_confirmation_mode"] == "terminal_bench"
    assert env.payloads[-1]["interpreter_kwargs"] == {"cwd": "/tmp/task"}
    remote_command = next(
        command
        for command in env.commands
        if "terminal_bench_rlm.tools.remote_controller" in command
    )
    remote_command_index = env.commands.index(remote_command)
    assert "HOME=/tmp/predict_rlm_home" in remote_command
    assert "PYTHONPATH=" in remote_command
    assert env.command_timeouts[remote_command_index] is None
    assert "OPENAI_API_KEY=super-secret-token" in remote_command


def test_daytona_remote_agent_forwards_host_openai_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CODEX_LM_AUTH_PROFILE", "gabriel-at-trampoline")
    monkeypatch.setenv("OPENAI_API_KEY", "host-secret-token")
    env = FakeDaytonaRemoteEnvironment(answer="remote done")
    context = SimpleNamespace()
    agent = tbench_agent.DaytonaRemotePredictRLMAgent(logs_dir=tmp_path)

    asyncio.run(agent.run("solve remotely", env, context))

    payload_text = json.dumps(env.payloads[-1], sort_keys=True)
    assert "host-secret-token" not in payload_text
    assert "gabriel-at-trampoline" not in payload_text
    remote_command = next(
        command
        for command in env.commands
        if "terminal_bench_rlm.tools.remote_controller" in command
    )
    assert "CODEX_LM_AUTH_PROFILE=gabriel-at-trampoline" in remote_command
    assert "OPENAI_API_KEY=host-secret-token" in remote_command


def test_daytona_remote_agent_sentinel_parsing_sets_answer(tmp_path: Path) -> None:
    env = FakeDaytonaRemoteEnvironment(answer="sentinel answer")
    context = SimpleNamespace()
    agent = tbench_agent.DaytonaRemotePredictRLMAgent(logs_dir=tmp_path)

    asyncio.run(agent.run("solve remotely", env, context))

    assert context.answer == "sentinel answer"


def test_remote_controller_writes_status_before_rlm_returns(monkeypatch, tmp_path: Path) -> None:
    class FakeInterpreter:
        def shutdown(self) -> None:
            pass

    class FakePredictRLM:
        def __init__(self, _signature, **_kwargs) -> None:
            pass

        async def acall(self):
            status_path = tmp_path / "predict_rlm_status.json"
            assert status_path.exists()
            status = json.loads(status_path.read_text())
            assert status["status"] == "running"
            return SimpleNamespace(answer="done", trace=None)

    monkeypatch.setattr(remote_controller, "_local_process_interpreter_class", lambda: FakeInterpreter)
    monkeypatch.setattr(remote_controller, "_predict_rlm_class", lambda: FakePredictRLM)

    answer = remote_controller._run_predict_rlm(
        {
            "instruction": "solve",
            "logging_dir": str(tmp_path),
        }
    )

    assert answer == "done"
    status = json.loads((tmp_path / "predict_rlm_status.json").read_text())
    assert status["status"] == "completed"


def test_remote_controller_writes_failed_status_without_trace(monkeypatch, tmp_path: Path) -> None:
    class FakeInterpreter:
        def shutdown(self) -> None:
            pass

    class FakePredictRLM:
        def __init__(self, _signature, **_kwargs) -> None:
            pass

        async def acall(self):
            raise RuntimeError("agent stopped before trace")

    monkeypatch.setattr(remote_controller, "_local_process_interpreter_class", lambda: FakeInterpreter)
    monkeypatch.setattr(remote_controller, "_predict_rlm_class", lambda: FakePredictRLM)

    try:
        remote_controller._run_predict_rlm(
            {
                "instruction": "solve",
                "logging_dir": str(tmp_path),
            }
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected remote controller failure")

    status = json.loads((tmp_path / "predict_rlm_status.json").read_text())
    assert status["status"] == "failed"
    assert status["error_type"] == "RuntimeError"
    assert status["error"] == "agent stopped before trace"


def test_remote_controller_verbose_streams_rlm_iteration_logs(monkeypatch, tmp_path: Path) -> None:
    class FakeInterpreter:
        def shutdown(self) -> None:
            pass

    class FakePredictRLM:
        def __init__(self, _signature, **kwargs) -> None:
            self.verbose = kwargs["verbose"]

        def __call__(self):
            raise AssertionError("remote controller must use PredictRLM.acall")

        async def acall(self):
            logging.getLogger("predict_rlm.trace").info("RLM turn 1/2\nCode:\nprint(1)")
            return SimpleNamespace(answer="done", trace=None)

    log_path = tmp_path / "predict_rlm_debug.jsonl"
    monkeypatch.setenv("PREDICT_RLM_DEBUG_LOG", str(log_path))
    monkeypatch.setattr(remote_controller, "_local_process_interpreter_class", lambda: FakeInterpreter)
    monkeypatch.setattr(remote_controller, "_predict_rlm_class", lambda: FakePredictRLM)

    answer = remote_controller._run_predict_rlm(
        {
            "instruction": "solve",
            "predict_rlm_kwargs": {"verbose": True},
        }
    )

    assert answer == "done"
    assert "RLM turn 1/2" in log_path.read_text()


def test_remote_controller_wires_live_trace_export_path(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeInterpreter:
        def shutdown(self) -> None:
            pass

    class FakePredictRLM:
        def __init__(self, _signature, **kwargs) -> None:
            captured["rlm_kwargs"] = kwargs

        async def acall(self):
            return SimpleNamespace(answer="done", trace=None)

    monkeypatch.setattr(remote_controller, "_local_process_interpreter_class", lambda: FakeInterpreter)
    monkeypatch.setattr(remote_controller, "_predict_rlm_class", lambda: FakePredictRLM)

    answer = remote_controller._run_predict_rlm(
        {
            "instruction": "solve",
            "logging_dir": str(tmp_path),
            "predict_rlm_kwargs": {},
        }
    )

    assert answer == "done"
    rlm_kwargs = captured["rlm_kwargs"]
    assert isinstance(rlm_kwargs, dict)
    assert rlm_kwargs["trace_export_path"] == tmp_path / "predict_rlm_trace.json"


def test_daytona_remote_agent_payload_carries_submit_confirmation_mode(tmp_path: Path) -> None:
    env = FakeDaytonaRemoteEnvironment(answer="remote done")
    agent = tbench_agent.DaytonaRemotePredictRLMAgent(
        logs_dir=tmp_path,
        submit_confirmation_mode="terminal_bench",
    )

    asyncio.run(agent.run("solve remotely", env, SimpleNamespace()))

    payload = env.payloads[-1]
    assert payload["submit_confirmation_mode"] == "terminal_bench"
    assert "submit_confirmation" not in payload


def test_remote_controller_reconstructs_terminal_bench_submit_confirmation_callback(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeInterpreter:
        def shutdown(self) -> None:
            pass

    class FakePredictRLM:
        def __init__(self, _signature, **kwargs) -> None:
            captured["rlm_kwargs"] = kwargs

        async def acall(self):
            return SimpleNamespace(answer="done", trace=None)

    monkeypatch.setattr(remote_controller, "_local_process_interpreter_class", lambda: FakeInterpreter)
    monkeypatch.setattr(remote_controller, "_predict_rlm_class", lambda: FakePredictRLM)

    answer = remote_controller._run_predict_rlm(
        {
            "instruction": "Fix the script and run the tests.",
            "submit_confirmation_mode": "terminal_bench",
        }
    )

    assert answer == "done"
    rlm_kwargs = captured["rlm_kwargs"]
    assert isinstance(rlm_kwargs, dict)
    callback = rlm_kwargs["submit_confirmation"]
    assert callable(callback)
    message = callback(
        SimpleNamespace(
            submitted_payload={"answer": "script fixed"},
            latest_observation="tests passed",
            iteration=2,
        )
    )
    assert "Original task:" in message
    assert "Fix the script and run the tests." in message
    assert '"answer": "script fixed"' in message
    assert "tests passed" in message
    assert "After this point, grading will begin" in message


def test_daytona_remote_agent_streams_remote_debug_logs(tmp_path: Path, capsys) -> None:
    class StreamingRemoteEnvironment(FakeDaytonaRemoteEnvironment):
        def __init__(self) -> None:
            super().__init__(answer="streamed answer")
            self.polls = 0

        async def exec(self, *, command: str, timeout_sec: int | None = None):
            self.command_timeouts.append(timeout_sec)
            self.commands.append(command)
            if "terminal_bench_rlm.tools.remote_controller" in command:
                await asyncio.sleep(0.05)
                stdout = (
                    f"{tbench_agent.DAYTONA_REMOTE_RESULT_SENTINEL}"
                    f"{json.dumps({'ok': True, 'answer': self.answer})}\n"
                )
                return SimpleNamespace(returncode=0, stdout=stdout, stderr="")
            if "PREDICT_RLM_REMOTE_LOG_OFFSET" in command:
                self.polls += 1
                stdout = "remote debug line\nPREDICT_RLM_REMOTE_LOG_OFFSET=18\n"
                return SimpleNamespace(returncode=0, stdout=stdout, stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    env = StreamingRemoteEnvironment()
    context = SimpleNamespace()
    agent = tbench_agent.DaytonaRemotePredictRLMAgent(
        logs_dir=tmp_path,
        predict_rlm_debug=True,
        remote_log_poll_interval=0.01,
    )

    asyncio.run(agent.run("solve remotely", env, context))

    assert context.answer == "streamed answer"
    assert env.payloads[-1]["predict_rlm_debug_log"] == "/tmp/predict_rlm_controller/predict_rlm_debug.jsonl"
    assert env.polls > 0
    assert "remote debug line" in capsys.readouterr().out


def test_daytona_remote_agent_log_stream_shutdown_failure_does_not_block_answer(
    tmp_path: Path,
) -> None:
    class FailingLogStreamEnvironment(FakeDaytonaRemoteEnvironment):
        async def exec(self, *, command: str, timeout_sec: int | None = None):
            self.command_timeouts.append(timeout_sec)
            self.commands.append(command)
            if "terminal_bench_rlm.tools.remote_controller" in command:
                await asyncio.sleep(0.05)
                stdout = (
                    f"{tbench_agent.DAYTONA_REMOTE_RESULT_SENTINEL}"
                    f"{json.dumps({'ok': True, 'answer': self.answer})}\n"
                )
                return SimpleNamespace(returncode=0, stdout=stdout, stderr="")
            if "PREDICT_RLM_REMOTE_LOG_OFFSET" in command:
                raise RuntimeError("remote log read failed")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    env = FailingLogStreamEnvironment(answer="controller finished")
    context = SimpleNamespace()
    agent = tbench_agent.DaytonaRemotePredictRLMAgent(
        logs_dir=tmp_path,
        predict_rlm_debug=True,
        remote_log_poll_interval=0.01,
    )

    asyncio.run(agent.run("solve remotely", env, context))

    assert context.answer == "controller finished"


def test_daytona_remote_agent_bootstrap_invokes_packaged_script(tmp_path: Path) -> None:
    env = FakeDaytonaRemoteEnvironment(answer="remote done")
    agent = tbench_agent.DaytonaRemotePredictRLMAgent(logs_dir=tmp_path)

    asyncio.run(agent.setup(env))

    setup_command = next(command for command in env.commands if "bootstrap_controller.sh" in command)
    bootstrap_args = _bootstrap_command_args(setup_command)
    assert bootstrap_args == [
        "sh",
        "/tmp/predict_rlm_controller/repo/src/predict_rlm/remote/bootstrap_controller.sh",
        "--root",
        "/tmp/predict_rlm_controller",
        "--repo",
        "/tmp/predict_rlm_controller/repo",
        "--python",
        "3.12",
    ]
    assert "apt-get install -y python3 python3-pip python3-venv" not in setup_command
    assert "apk add --no-cache python3 py3-pip" not in setup_command
    assert "python3 -m venv /tmp/predict_rlm_controller/uv-bootstrap" not in setup_command


def test_daytona_remote_agent_codex_lm_uploads_opaque_auth_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    credentials_dir = home / ".codex-lm"
    credentials_dir.mkdir(parents=True)
    (credentials_dir / "auth.json").write_text('{"token": "do-not-copy-into-payload"}')
    monkeypatch.setenv("HOME", str(home))
    env = FakeDaytonaRemoteEnvironment(answer="codex answer")
    context = SimpleNamespace()
    agent = tbench_agent.DaytonaRemotePredictRLMAgent(
        logs_dir=tmp_path,
        codex_lm=True,
    )

    asyncio.run(agent.setup(env))
    asyncio.run(agent.run("solve remotely", env, context))

    setup_command = next(command for command in env.commands if "bootstrap_controller.sh" in command)
    bootstrap_args = _bootstrap_command_args(setup_command)
    assert bootstrap_args[bootstrap_args.index("--extra") + 1] == "[codex-lm]"
    assert context.answer == "codex answer"
    assert env.upload_dirs == [
        (str(credentials_dir), "/tmp/predict_rlm_home/.codex-lm"),
    ]
    payload_text = json.dumps(env.payloads[-1], sort_keys=True)
    assert "do-not-copy-into-payload" not in payload_text
    assert any("rm -rf /tmp/predict_rlm_home/.codex-lm" in command for command in env.commands)



def test_daytona_remote_agent_accepts_harbor_post_run_context_hook(tmp_path: Path) -> None:
    context = SimpleNamespace(metadata={})
    agent = tbench_agent.DaytonaRemotePredictRLMAgent(logs_dir=tmp_path)

    agent.populate_context_post_run(context)


def test_daytona_remote_agent_populates_pydantic_context_metadata(tmp_path: Path) -> None:
    harbor_context = pytest.importorskip("harbor.models.agent.context")
    env = FakeDaytonaRemoteEnvironment(answer="finished")
    context = harbor_context.AgentContext()
    agent = tbench_agent.DaytonaRemotePredictRLMAgent(logs_dir=tmp_path)

    asyncio.run(agent.run("solve", env, context))

    assert context.metadata == {"answer": "finished"}


def test_daytona_remote_agent_writes_setup_and_agent_phase_events(tmp_path: Path) -> None:
    env = FakeDaytonaRemoteEnvironment(answer="finished")
    phase_log = tmp_path / "task_phase_events.jsonl"
    agent = tbench_agent.DaytonaRemotePredictRLMAgent(
        logs_dir=tmp_path,
        phase_log_path=str(phase_log),
        task_id="terminal-bench/task",
    )

    asyncio.run(agent.run("solve", env, SimpleNamespace()))

    events = [json.loads(line) for line in phase_log.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "agent_setup_start",
        "agent_setup_end",
        "agent_run_start",
        "agent_run_end",
    ]
    assert {event["task_id"] for event in events} == {"terminal-bench/task"}
    assert events[0]["phase"] == "agent_setup"
    assert events[2]["phase"] == "agent_eval"
    assert events[1]["duration_seconds"] >= 0
    assert events[3]["duration_seconds"] >= 0
