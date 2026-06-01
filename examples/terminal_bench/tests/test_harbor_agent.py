from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from terminal_bench_rlm.skills import (  # noqa: E402
    DEFAULT_TERMINAL_BENCH_SKILL_INSTRUCTIONS,
    TERMINAL_BENCH_SKILL_NAME,
)
from terminal_bench_rlm.tools import remote_controller, tbench_agent  # noqa: E402


def _assert_task_instruction_signature(signature, task_instruction: str) -> None:
    assert list(signature.input_fields) == []
    assert list(signature.output_fields) == ["answer"]
    assert "Terminal-Bench task instruction" in signature.instructions
    assert task_instruction in signature.instructions


def _assert_terminal_bench_skill_semantics(instructions: str) -> None:
    headings = [
        "## Operating principle",
        "## Inspection and changes",
        "## Evidence preservation and stopping discipline",
        "## Timeouts and long-running work",
        "### Command helper pattern",
        "## Problem-solving strategy",
        "## Required verification and final QA",
        "## Verification and final submission",
    ]
    heading_positions = [instructions.index(heading) for heading in headings]
    normalized_instructions = " ".join(instructions.split())
    bad_required_verification_prefix = "+Required" + " verification:"
    obsolete_schema_terms = [
        "acceptance" + "_contract",
        "expected" + "_final_state",
        "status: " + '"pending|verified|blocked"',
    ]

    assert heading_positions == sorted(heading_positions)
    assert "command-line tasks in a Linux environment" in instructions
    assert "Terminal-Bench tasks inside a Linux task container" not in instructions
    assert "inspect the filesystem before making changes" in instructions
    assert "install missing packages" in instructions
    assert "package managers" in instructions
    assert "small inspectable steps" in instructions
    assert "preserve the raw inputs or sidecar files" in instructions
    assert "reversible working copies" in instructions
    assert "Create the requested artifact or service as soon as a plausible solution exists" in instructions
    assert "do not keep changing a working artifact" in instructions
    assert "remaining uncertainty is only speculative" in normalized_instructions
    assert "keep the artifact stable" in instructions
    assert "clean temporary side effects" in instructions
    assert "while budget remains" in instructions
    assert "direct, sampled, analytical, or tool-assisted" in instructions
    assert "choose elegant, smart, effective strategies" in instructions
    assert "exhaustive loops" in instructions
    assert "unobserved verification command" in instructions
    assert bad_required_verification_prefix not in instructions
    assert "@dataclass" in instructions
    assert "class RequiredVerification" in instructions
    assert "requirement: str" in instructions
    assert "verified: bool = False" in instructions
    assert 'evidence: str = ""' in instructions
    assert "verification: str" not in instructions
    assert "required verification list" in instructions
    assert "required checks" in instructions
    assert "Mark an" in instructions
    assert "short list" in instructions
    assert "extracted from the task" in instructions
    assert "verified:" in instructions
    assert "schema" not in instructions.lower()
    assert "yaml" not in instructions.lower()
    assert all(term not in instructions for term in obsolete_schema_terms)
    assert "ledger" not in instructions.lower()
    assert "task requirements" in instructions
    assert "Before SUBMIT" in instructions
    assert "proportional evidence" in instructions
    assert "literal paths/endpoints" in instructions
    assert "config values" in instructions
    assert "processes or services" in normalized_instructions
    assert "absolute minimum" in instructions
    assert "files, processes, services, and configs" in instructions
    assert "initial state" in instructions
    assert "no extra modified files" in instructions
    assert "copied artifacts" in instructions
    assert "debug helpers" in instructions
    assert "alternate runtime artifacts" in normalized_instructions
    assert "temporary services" in instructions
    assert "config side effects" in instructions
    assert (
        "paths, endpoints, flags, and config values named by the task"
        in normalized_instructions
    )
    assert "visible tests" in instructions
    assert "verifier-shaped checks" in instructions
    assert "hidden tests" in instructions
    assert "parse/load/exercise" in instructions
    assert "semantic/reference" in instructions
    assert "stdout/progress text" in instructions
    assert "command behavior" in instructions
    assert "emulator, interpreter, VM, service, or wrapper tasks" in instructions
    assert "named binary, program, protocol, or mechanism" in normalized_instructions
    assert "shortcut or native/source-level stand-in" in instructions
    assert "negative constraints" in normalized_instructions
    assert "debug/runtime state" in instructions
    assert "stdout/stderr" in instructions
    assert "exit code" in normalized_instructions or "exit codes" in normalized_instructions
    assert "service behavior" in instructions
    assert "SUBMIT makes the result final" in instructions
    assert "stale debug history" in instructions
    assert "Once the observable task contract is satisfied" not in instructions
    assert "run the verification in one iteration" not in instructions
    assert "separate later iteration" not in instructions
    assert "always run the full verifier" not in instructions.lower()
    assert "must reproduce the full verifier" not in instructions.lower()
    for term in ["windows", "win311", "qemu", "mips", "bmp", "doom", "PIL"]:
        assert re.search(rf"\b{re.escape(term)}\b", instructions, re.IGNORECASE) is None
    assert "async def start" not in instructions
    assert "async def wait" not in instructions
    assert "await start(" not in instructions
    assert "await wait(" not in instructions
    assert "subprocess.Popen" not in instructions
    assert "stdout_tail" not in instructions
    assert "stderr_tail" not in instructions
    assert "job = await start" not in instructions
    assert "progress = await wait" not in instructions
    assert "poll it again" not in instructions
    assert "For foreground commands, use async/await run()" in instructions
    assert (
        "# Use run() for bounded foreground commands; inspect output before continuing."
        in instructions
    )
    assert "# Use requests timeouts for network calls." in instructions
    assert (
        "# Use asyncio.wait_for for expensive computations or async work that may hang."
        in instructions
    )
    assert (
        "# Use asyncio.gather for independent non-mutating checks that can run concurrently."
        in instructions
    )
    assert "run_terminal_command" not in instructions
    assert "send_terminal_keys" not in instructions
    assert "read_terminal" not in instructions


class FakeDaytonaRemoteEnvironment:
    def __init__(self, *, answer: str = "remote answer") -> None:
        self.commands: list[str] = []
        self.uploads: list[tuple[str, str]] = []
        self.upload_dirs: list[tuple[str, str]] = []
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


def test_harbor_agent_runs_predict_rlm_async_against_harbor_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakePredictRLM:
        def __init__(self, signature, **kwargs) -> None:
            captured["signature"] = signature
            captured["kwargs"] = kwargs

        def __call__(self, **_kwargs):
            raise AssertionError("Harbor adapter must not call sync PredictRLM entrypoint")

        async def acall(self, **kwargs):
            captured["acall_kwargs"] = kwargs
            return SimpleNamespace(answer="done", trace=None)

    class FakeInterpreter:
        def __init__(self, environment, *, loop, **kwargs) -> None:
            captured["environment"] = environment
            captured["loop"] = loop
            captured["interpreter_kwargs"] = kwargs

        def shutdown(self) -> None:
            captured["shutdown"] = True

    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(tbench_agent, "HarborEnvironmentInterpreter", FakeInterpreter)

    environment = object()
    context = SimpleNamespace()
    agent = tbench_agent.HarborPredictRLMAgent(
        logs_dir=tmp_path,
        skill_instructions="Use shell commands and verify outputs.",
        max_iterations="3",
    )

    asyncio.run(agent.run("solve this task", environment, context))

    assert captured["environment"] is environment
    _assert_task_instruction_signature(captured["signature"], "solve this task")
    assert captured["kwargs"]["interpreter"] is not None
    assert captured["kwargs"]["max_iterations"] == 3
    assert captured["kwargs"]["skills"][0].name == "terminal-bench"
    assert captured["kwargs"]["skills"][0].instructions == "Use shell commands and verify outputs."
    assert captured["acall_kwargs"] == {}
    assert captured["shutdown"] is True


def test_harbor_agent_local_process_mode_does_not_use_environment_exec(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakePredictRLM:
        def __init__(self, _signature, **kwargs) -> None:
            captured["kwargs"] = kwargs

        async def acall(self, **_kwargs):
            return SimpleNamespace(answer="done", trace=None)

    class ForbiddenHarborInterpreter:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("local-process mode must not build HarborEnvironmentInterpreter")

    class FakeLocalProcessInterpreter:
        def __init__(self, **kwargs) -> None:
            captured["local_process_kwargs"] = kwargs

        def shutdown(self) -> None:
            captured["shutdown"] = True

    class EnvironmentWithoutExec:
        def start_exec(self, *_args, **_kwargs):
            raise AssertionError("local-process mode must not call environment.start_exec")

        def exec_stream(self, *_args, **_kwargs):
            raise AssertionError("local-process mode must not call environment.exec_stream")

    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(tbench_agent, "HarborEnvironmentInterpreter", ForbiddenHarborInterpreter)
    monkeypatch.setattr(tbench_agent, "LocalProcessRunnerInterpreter", FakeLocalProcessInterpreter)

    agent = tbench_agent.HarborPredictRLMAgent(
        logs_dir=tmp_path,
        interpreter_mode="local-process",
        max_iterations="3",
    )

    asyncio.run(agent.run("solve", EnvironmentWithoutExec(), SimpleNamespace()))

    interpreter = captured["kwargs"]["interpreter"]
    assert isinstance(interpreter, FakeLocalProcessInterpreter)
    assert captured["local_process_kwargs"]["exec_timeout"] == 900.0
    assert captured["shutdown"] is True


def test_harbor_agent_local_process_mode_uses_daytona_session_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakePredictRLM:
        def __init__(self, _signature, **kwargs) -> None:
            captured["kwargs"] = kwargs

        async def acall(self, **_kwargs):
            return SimpleNamespace(answer="done", trace=None)

    class FakeHarborInterpreter:
        def __init__(self, environment, *, loop, **kwargs) -> None:
            captured["environment"] = environment
            captured["loop"] = loop
            captured["interpreter_kwargs"] = kwargs

        def shutdown(self) -> None:
            captured["shutdown"] = True

    class ForbiddenLocalProcessInterpreter:
        def __init__(self, **_kwargs) -> None:
            raise AssertionError("Daytona local-process mode must execute inside the task environment")

    class FakeDaytonaProcess:
        def create_session(self):
            pass

        def execute_session_command(self):
            pass

        def get_session_command(self):
            pass

        def get_session_command_logs(self):
            pass

        def send_session_command_input(self):
            pass

    environment = SimpleNamespace(_sandbox=SimpleNamespace(process=FakeDaytonaProcess()))

    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(tbench_agent, "HarborEnvironmentInterpreter", FakeHarborInterpreter)
    monkeypatch.setattr(tbench_agent, "LocalProcessRunnerInterpreter", ForbiddenLocalProcessInterpreter)

    agent = tbench_agent.HarborPredictRLMAgent(
        logs_dir=tmp_path,
        interpreter_mode="local-process",
        max_iterations="3",
    )

    asyncio.run(agent.run("solve", environment, SimpleNamespace()))

    assert captured["environment"] is environment
    assert captured["interpreter_kwargs"]["exec_timeout"] == 900.0
    assert captured["shutdown"] is True


def test_harbor_agent_uses_default_terminal_bench_skill_when_not_overridden(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakePredictRLM:
        def __init__(self, _signature, **kwargs) -> None:
            captured["kwargs"] = kwargs

        async def acall(self, **_kwargs):
            return SimpleNamespace(answer="done", trace=None)

    class FakeInterpreter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(tbench_agent, "HarborEnvironmentInterpreter", FakeInterpreter)

    context = SimpleNamespace()
    agent = tbench_agent.HarborPredictRLMAgent(logs_dir=tmp_path)

    asyncio.run(agent.run("solve", object(), context))

    skills = captured["kwargs"]["skills"]
    assert len(skills) == 1
    assert skills[0].name == TERMINAL_BENCH_SKILL_NAME
    assert skills[0].instructions == DEFAULT_TERMINAL_BENCH_SKILL_INSTRUCTIONS
    _assert_terminal_bench_skill_semantics(skills[0].instructions)
    assert "tools" not in captured["kwargs"]


def test_harbor_agent_keeps_one_terminal_bench_skill(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakePredictRLM:
        def __init__(self, _signature, **kwargs) -> None:
            captured["kwargs"] = kwargs

        async def acall(self, **_kwargs):
            return SimpleNamespace(answer="done", trace=None)

    class FakeInterpreter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(tbench_agent, "HarborEnvironmentInterpreter", FakeInterpreter)

    context = SimpleNamespace()
    agent = tbench_agent.HarborPredictRLMAgent(
        logs_dir=tmp_path,
        skills=[
            SimpleNamespace(name=TERMINAL_BENCH_SKILL_NAME, instructions="stale"),
            SimpleNamespace(name="other", instructions="keep"),
        ],
    )

    asyncio.run(agent.run("solve", object(), context))

    skills = captured["kwargs"]["skills"]
    terminal_bench_skills = [
        skill for skill in skills if getattr(skill, "name", None) == TERMINAL_BENCH_SKILL_NAME
    ]
    assert len(terminal_bench_skills) == 1
    assert terminal_bench_skills[0].instructions == DEFAULT_TERMINAL_BENCH_SKILL_INSTRUCTIONS
    assert [getattr(skill, "name", None) for skill in skills] == ["other", TERMINAL_BENCH_SKILL_NAME]


def test_harbor_agent_does_not_forward_harbor_extra_env_to_predict_rlm(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakePredictRLM:
        def __init__(self, _signature, **kwargs) -> None:
            captured["kwargs"] = kwargs

        async def acall(self, **_kwargs):
            return SimpleNamespace(answer="done", trace=None)

    class FakeInterpreter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(tbench_agent, "HarborEnvironmentInterpreter", FakeInterpreter)

    context = SimpleNamespace()
    agent = tbench_agent.HarborPredictRLMAgent(
        logs_dir=tmp_path,
        extra_env={"OPENAI_API_KEY": "codex-lm"},
    )

    asyncio.run(agent.run("solve", object(), context))

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert "extra_env" not in kwargs
    assert context.answer == "done"


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


def test_daytona_remote_agent_sentinel_parsing_sets_answer(tmp_path: Path) -> None:
    env = FakeDaytonaRemoteEnvironment(answer="sentinel answer")
    context = SimpleNamespace()
    agent = tbench_agent.DaytonaRemotePredictRLMAgent(logs_dir=tmp_path)

    asyncio.run(agent.run("solve remotely", env, context))

    assert context.answer == "sentinel answer"


def test_remote_controller_verbose_streams_rlm_iteration_logs(monkeypatch, tmp_path: Path) -> None:
    class FakeInterpreter:
        def shutdown(self) -> None:
            pass

    class FakePredictRLM:
        def __init__(self, _signature, **kwargs) -> None:
            self.verbose = kwargs["verbose"]

        def __call__(self):
            logging.getLogger("dspy.predict.rlm").info("RLM iteration 1/2\nCode:\nprint(1)")
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
    assert "RLM iteration 1/2" in log_path.read_text()


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


def test_daytona_remote_agent_bootstrap_installs_python_before_uv(tmp_path: Path) -> None:
    env = FakeDaytonaRemoteEnvironment(answer="remote done")
    agent = tbench_agent.DaytonaRemotePredictRLMAgent(logs_dir=tmp_path)

    asyncio.run(agent.setup(env))

    setup_command = next(command for command in env.commands if "$UV_COMMAND venv" in command)
    assert "apt-get install -y python3 python3-pip python3-venv" in setup_command
    assert "apk add --no-cache python3 py3-pip" in setup_command
    assert "python3 -m venv /tmp/predict_rlm_controller/uv-bootstrap" in setup_command
    assert "/tmp/predict_rlm_controller/uv-bootstrap/bin/python -m pip install" in setup_command
    assert setup_command.index("command -v python3") < setup_command.index("python3 -m venv")


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

    assert context.answer == "codex answer"
    assert env.upload_dirs == [
        (str(credentials_dir), "/tmp/predict_rlm_home/.codex-lm"),
    ]
    payload_text = json.dumps(env.payloads[-1], sort_keys=True)
    assert "do-not-copy-into-payload" not in payload_text
    assert any("rm -rf /tmp/predict_rlm_home/.codex-lm" in command for command in env.commands)



def test_harbor_agent_accepts_harbor_post_run_context_hook(tmp_path: Path) -> None:
    context = SimpleNamespace(metadata={})
    agent = tbench_agent.HarborPredictRLMAgent(logs_dir=tmp_path)

    agent.populate_context_post_run(context)


def test_harbor_agent_populates_context_answer(monkeypatch, tmp_path: Path) -> None:
    class FakePredictRLM:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def acall(self, **_kwargs):
            return SimpleNamespace(answer="finished", trace=None)

    class FakeInterpreter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(tbench_agent, "HarborEnvironmentInterpreter", FakeInterpreter)

    context = SimpleNamespace()
    agent = tbench_agent.HarborPredictRLMAgent(logs_dir=tmp_path)

    asyncio.run(agent.run("solve", object(), context))

    assert context.answer == "finished"


def test_harbor_agent_populates_pydantic_context_metadata(monkeypatch, tmp_path: Path) -> None:
    harbor_context = pytest.importorskip("harbor.models.agent.context")

    class FakePredictRLM:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def acall(self, **_kwargs):
            return SimpleNamespace(answer="finished", trace=None)

    class FakeInterpreter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(tbench_agent, "HarborEnvironmentInterpreter", FakeInterpreter)

    context = harbor_context.AgentContext()
    agent = tbench_agent.HarborPredictRLMAgent(logs_dir=tmp_path)

    asyncio.run(agent.run("solve", object(), context))

    assert context.metadata == {"answer": "finished"}


def test_harbor_agent_writes_setup_and_agent_phase_events(monkeypatch, tmp_path: Path) -> None:
    class FakePredictRLM:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def acall(self, **_kwargs):
            return SimpleNamespace(answer="finished", trace=None)

    class FakeInterpreter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(tbench_agent, "HarborEnvironmentInterpreter", FakeInterpreter)

    phase_log = tmp_path / "task_phase_events.jsonl"
    agent = tbench_agent.HarborPredictRLMAgent(
        logs_dir=tmp_path,
        phase_log_path=str(phase_log),
        task_id="terminal-bench/task",
    )

    asyncio.run(agent.setup(object()))
    asyncio.run(agent.run("solve", object(), SimpleNamespace()))

    events = [json.loads(line) for line in phase_log.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "agent_setup_start",
        "agent_setup_end",
        "agent_run_start",
        "sandbox_setup_start",
        "sandbox_setup_end",
        "agent_run_end",
    ]
    assert {event["task_id"] for event in events} == {"terminal-bench/task"}
    assert events[0]["phase"] == "agent_setup"
    assert events[2]["phase"] == "agent_eval"
    assert events[4]["phase"] == "sandbox_setup"
    assert events[4]["duration_seconds"] >= 0
    assert events[5]["duration_seconds"] >= 0


def test_harbor_agent_shuts_down_and_writes_trace_on_async_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakeTrace:
        def to_exportable_json(self) -> str:
            return '{"status": "failed"}'

    class FakePredictRLM:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __call__(self, **_kwargs):
            raise AssertionError("Harbor adapter must not call sync PredictRLM entrypoint")

        async def acall(self, **_kwargs):
            exc = RuntimeError("rlm failed")
            exc.trace = FakeTrace()
            raise exc

    class FakeInterpreter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def shutdown(self) -> None:
            captured["shutdown"] = True

    monkeypatch.setattr(tbench_agent, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(tbench_agent, "HarborEnvironmentInterpreter", FakeInterpreter)

    phase_log = tmp_path / "task_phase_events.jsonl"
    agent = tbench_agent.HarborPredictRLMAgent(
        logs_dir=tmp_path,
        phase_log_path=str(phase_log),
        task_id="terminal-bench/task",
    )

    with pytest.raises(RuntimeError, match="rlm failed"):
        asyncio.run(agent.run("solve", object(), SimpleNamespace()))

    assert captured["shutdown"] is True
    trace_files = list(tmp_path.glob("predict_rlm_trace_*.json"))
    assert len(trace_files) == 1
    assert json.loads(trace_files[0].read_text(encoding="utf-8")) == {"status": "failed"}
    events = [json.loads(line) for line in phase_log.read_text().splitlines()]
    assert events[-1]["event"] == "agent_run_end"
    assert events[-1]["phase"] == "agent_eval"
    assert events[-1]["status"] == "failed"
