"""Unit tests for interpreter helpers (no Deno required)."""

import asyncio
import subprocess
import tempfile
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from dspy.primitives.code_interpreter import CodeInterpreterError

from predict_rlm.backends import JspiBackend
from predict_rlm.backends.base import SandboxFatalError
from predict_rlm.backends.jspi.backend import RUNNER_PATH, _needs_jspi_flag
from predict_rlm.telemetry import TelemetryContext


class ListTelemetrySink:
    def __init__(self):
        self.records: list[dict[str, Any]] = []

    def write(self, record: dict[str, Any]) -> None:
        self.records.append(record)


class TestNeedsJspiFlag:
    @patch.object(subprocess, "check_output")
    def test_old_v8_needs_flag(self, mock_check):
        mock_check.return_value = "deno 2.0.0\nv8 12.9.245.12-rusty\ntypescript 5.6.2"
        assert _needs_jspi_flag() is True

    @patch.object(subprocess, "check_output")
    def test_v8_13_6_needs_flag(self, mock_check):
        mock_check.return_value = "deno 2.1.0\nv8 13.6.100.0\ntypescript 5.6.2"
        assert _needs_jspi_flag() is True

    @patch.object(subprocess, "check_output")
    def test_v8_13_7_no_flag(self, mock_check):
        mock_check.return_value = "deno 2.2.0\nv8 13.7.0.0\ntypescript 5.6.2"
        assert _needs_jspi_flag() is False

    @patch.object(subprocess, "check_output")
    def test_v8_14_0_no_flag(self, mock_check):
        mock_check.return_value = "deno 3.0.0\nv8 14.0.0.0\ntypescript 5.6.2"
        assert _needs_jspi_flag() is False

    @patch.object(subprocess, "check_output")
    def test_deno_not_found_returns_true(self, mock_check):
        mock_check.side_effect = FileNotFoundError("deno not found")
        assert _needs_jspi_flag() is True

    @patch.object(subprocess, "check_output")
    def test_unexpected_output_returns_true(self, mock_check):
        mock_check.return_value = "some garbage output"
        assert _needs_jspi_flag() is True


def _make_interpreter():
    """Create a JspiBackend without running __init__ (no Deno subprocess)."""
    return JspiBackend.__new__(JspiBackend)


def _attach_telemetry(interp: JspiBackend) -> ListTelemetrySink:
    sink = ListTelemetrySink()
    interp._telemetry_context = TelemetryContext(sink=sink, trace_id="trace-1")
    interp._interpreter_id = "jspi-test"
    return sink


class TestBuildDenoCommand:
    @patch("predict_rlm.backends.jspi.backend._needs_jspi_flag", return_value=True)
    def test_includes_jspi_flag_when_needed(self, _):
        interp = _make_interpreter()
        with patch.object(interp, "_get_deno_dir", return_value=[]):
            cmd = interp._build_deno_command([], [], [], [])
        assert "--v8-flags=--experimental-wasm-jspi" in cmd

    @patch("predict_rlm.backends.jspi.backend._needs_jspi_flag", return_value=False)
    def test_excludes_jspi_flag_when_not_needed(self, _):
        interp = _make_interpreter()
        with patch.object(interp, "_get_deno_dir", return_value=[]):
            cmd = interp._build_deno_command([], [], [], [])
        assert "--v8-flags=--experimental-wasm-jspi" not in cmd

    @patch("predict_rlm.backends.jspi.backend._needs_jspi_flag", return_value=False)
    def test_runner_path_in_command(self, _):
        interp = _make_interpreter()
        with patch.object(interp, "_get_deno_dir", return_value=[]):
            cmd = interp._build_deno_command([], [], [], [])
        assert str(RUNNER_PATH) in cmd

    @patch("predict_rlm.backends.jspi.backend._needs_jspi_flag", return_value=False)
    def test_allow_read_includes_runner_and_user_paths(self, _):
        interp = _make_interpreter()
        with patch.object(interp, "_get_deno_dir", return_value=[]):
            cmd = interp._build_deno_command(["/data/input"], [], [], [])
        read_arg = [a for a in cmd if a.startswith("--allow-read=")][0]
        read_paths = read_arg.split("=", 1)[1].split(",")
        assert str(RUNNER_PATH) in read_paths
        assert "/data/input" in read_paths

    @patch("predict_rlm.backends.jspi.backend._needs_jspi_flag", return_value=False)
    def test_write_paths_also_in_allow_read(self, _):
        interp = _make_interpreter()
        with patch.object(interp, "_get_deno_dir", return_value=[]):
            cmd = interp._build_deno_command([], ["/data/output"], [], [])
        read_arg = [a for a in cmd if a.startswith("--allow-read=")][0]
        assert "/data/output" in read_arg

    @patch("predict_rlm.backends.jspi.backend._needs_jspi_flag", return_value=False)
    def test_allow_write_includes_tempdir_and_user_paths(self, _):
        interp = _make_interpreter()
        with patch.object(interp, "_get_deno_dir", return_value=[]):
            cmd = interp._build_deno_command([], ["/data/output"], [], [])
        write_arg = [a for a in cmd if a.startswith("--allow-write=")][0]
        write_paths = write_arg.split("=", 1)[1].split(",")
        assert "/data/output" in write_paths
        assert tempfile.gettempdir() in write_paths
        assert "/tmp" in write_paths

    @patch("predict_rlm.backends.jspi.backend._needs_jspi_flag", return_value=False)
    def test_allow_net_with_domains(self, _):
        interp = _make_interpreter()
        with patch.object(interp, "_get_deno_dir", return_value=[]):
            cmd = interp._build_deno_command([], [], ["pypi.org", "api.example.com"], [])
        assert "--allow-net=pypi.org,api.example.com" in cmd

    @patch("predict_rlm.backends.jspi.backend._needs_jspi_flag", return_value=False)
    def test_no_allow_net_when_empty(self, _):
        interp = _make_interpreter()
        with patch.object(interp, "_get_deno_dir", return_value=[]):
            cmd = interp._build_deno_command([], [], [], [])
        assert not any(a.startswith("--allow-net") for a in cmd)

    @patch("predict_rlm.backends.jspi.backend._needs_jspi_flag", return_value=False)
    def test_always_includes_allow_env_and_no_prompt(self, _):
        interp = _make_interpreter()
        with patch.object(interp, "_get_deno_dir", return_value=[]):
            cmd = interp._build_deno_command([], [], [], [])
        assert "--allow-env" in cmd
        assert "--no-prompt" in cmd

    @patch("predict_rlm.backends.jspi.backend._needs_jspi_flag", return_value=False)
    def test_env_vars_as_final_arg(self, _):
        interp = _make_interpreter()
        with patch.object(interp, "_get_deno_dir", return_value=[]):
            cmd = interp._build_deno_command(
                [], [], [], ["PYODIDE_PREINSTALL", "SKILL_PACKAGES"]
            )
        assert cmd[-1] == "PYODIDE_PREINSTALL,SKILL_PACKAGES"

    @patch("predict_rlm.backends.jspi.backend._needs_jspi_flag", return_value=False)
    def test_no_env_vars_runner_is_last(self, _):
        interp = _make_interpreter()
        with patch.object(interp, "_get_deno_dir", return_value=[]):
            cmd = interp._build_deno_command([], [], [], [])
        assert cmd[-1] == str(RUNNER_PATH)


class TestGetDenoDir:
    def test_includes_home_cache_paths(self):
        interp = _make_interpreter()
        with patch.dict("os.environ", {"HOME": "/home/test"}, clear=False):
            dirs = interp._get_deno_dir()
        assert "/home/test/.cache/deno" in dirs
        assert "/home/test/Library/Caches/deno" in dirs

    def test_includes_deno_dir_env(self):
        interp = _make_interpreter()
        with patch.dict("os.environ", {"DENO_DIR": "/custom/deno"}, clear=False):
            dirs = interp._get_deno_dir()
        assert "/custom/deno" in dirs


class TestSandboxFatalError:
    """SandboxFatalError must NOT inherit from CodeInterpreterError.

    DSPy's RLM._execute_iteration catches (CodeInterpreterError, SyntaxError)
    and converts the exception into an "[Error] ..." string that gets fed
    back to the model as regular iteration output. For ordinary in-sandbox
    errors (NameError, tool raised, etc.) that's correct — the model can
    self-correct. But when the sandbox subprocess itself dies (exec timeout,
    BrokenPipe), the per-run file_plan mounts and output dirs are gone, so
    subsequent iterations trip over FileNotFoundError with no way to recover.

    Keeping SandboxFatalError a sibling of CodeInterpreterError ensures the
    base class's catch tuple does not swallow it — it propagates out of
    rlm.forward() and the run fails fast.
    """

    def test_is_runtime_error(self):
        assert issubclass(SandboxFatalError, RuntimeError)

    def test_is_not_code_interpreter_error(self):
        assert not issubclass(SandboxFatalError, CodeInterpreterError)


class TestJspiLoggingConfig:
    def test_configure_debug_and_verbose_are_independent(self):
        interp = _make_interpreter()
        interp._debug = False
        interp._verbose = False

        interp.configure_debug(True)
        assert interp._debug is True
        assert interp._verbose is False

        interp.configure_verbose(True)
        assert interp._debug is True
        assert interp._verbose is True

        interp.configure_debug(False)
        assert interp._debug is False
        assert interp._verbose is True

    def test_configure_runtime_updates_debug_and_verbose(self):
        interp = _make_interpreter()
        interp._debug = False
        interp._verbose = False

        interp.configure_runtime(debug=True, verbose=True)

        assert interp._debug is True
        assert interp._verbose is True


class TestJspiTelemetry:
    def test_health_check_no_response_emits_lifecycle_failure(self):
        interp = _make_interpreter()
        sink = _attach_telemetry(interp)
        interp._request_id = 0
        interp._stdin_fd = -1
        interp._stdout_fd = -1
        interp._read_buf = ""
        interp.deno_process = types.SimpleNamespace(
            stdin=types.SimpleNamespace(write=lambda _data: None, flush=lambda: None),
            stdout=None,
            poll=lambda: None,
            pid=1234,
        )

        with patch.object(interp, "_read_with_timeout", return_value=None):
            with pytest.raises(CodeInterpreterError, match="No response"):
                interp._send_request("health_check", {}, "during health check")

        names = [record["name"] for record in sink.records]
        assert names == [
            "sandbox.health_check.start",
            "sandbox.health_check.no_response",
        ]
        no_response = sink.records[-1]
        assert no_response["status"]["code"] == "ERROR"
        assert no_response["attributes"]["failure.class"] == "sandbox_lifecycle_failure"
        assert no_response["attributes"]["process.pid"] == 1234
        assert no_response["attributes"]["rpc.request_id"] == 1

    def test_execute_timeout_emits_timeout_and_kill_events(self):
        interp = _make_interpreter()
        sink = _attach_telemetry(interp)
        interp._exec_timeout = 0.01
        interp._pending_file_ops = {}
        interp.deno_process = types.SimpleNamespace(
            kill=lambda: None,
            wait=lambda timeout=None: None,
            pid=4321,
        )

        async def _never_returns(_request_id):
            await asyncio.sleep(60)

        interp._execute_async = _never_returns

        with pytest.raises(SandboxFatalError):
            asyncio.run(interp._execute_with_timeout(7, 1_000_000_000))

        names = [record["name"] for record in sink.records]
        assert "sandbox.execute.timeout" in names
        assert "sandbox.shutdown.kill" in names
        timeout = next(
            record for record in sink.records if record["name"] == "sandbox.execute.timeout"
        )
        assert timeout["status"]["code"] == "ERROR"
        assert timeout["attributes"]["failure.class"] == "sandbox_exec_timeout"
        assert timeout["attributes"]["rpc.request_id"] == 7
        assert timeout["attributes"]["process.pid"] == 4321

    def test_tool_timeout_emits_host_tool_failure(self, monkeypatch):
        import predict_rlm.backends.jspi.backend as rlm_interpreter

        monkeypatch.setattr(rlm_interpreter, "TOOL_CALL_TIMEOUT_SEC", 0.01)

        async def slow_tool():
            await asyncio.sleep(60)

        interp = _make_interpreter()
        sink = _attach_telemetry(interp)
        interp.tools = {"slow_tool": slow_tool}
        interp._debug = False
        interp._pending_file_ops = {}

        response = asyncio.run(
            interp._execute_tool_async("slow_tool", {"args": [], "kwargs": {}}, "tool-1")
        )

        assert "error" in response
        names = [record["name"] for record in sink.records]
        assert names == ["sandbox.tool_call.start", "sandbox.tool_call.timeout"]
        timeout = sink.records[-1]
        assert timeout["attributes"]["failure.class"] == "host_tool_timeout_or_leak"
        assert timeout["attributes"]["tool.name"] == "slow_tool"
        assert timeout["attributes"]["tool.id"] == "tool-1"


class TestAsyncExecuteEof:
    def test_stdout_eof_does_not_drain_stderr_with_unbounded_read(self):
        interp = _make_interpreter()

        class BlockingStderr:
            def read(self):
                raise AssertionError("stderr.read() would block if the pipe is still open")

        async def no_completed_responses(_pending_tasks):
            return None

        async def stdout_eof(_timeout):
            return ""

        interp.deno_process = SimpleNamespace(stderr=BlockingStderr())
        interp._pending_file_ops = {}
        interp._send_completed_responses = no_completed_responses
        interp._read_with_timeout_async = stdout_eof

        try:
            asyncio.run(interp._execute_async(execute_request_id=1))
        except SandboxFatalError as exc:
            assert "Deno subprocess stopped producing stdout" in str(exc)
        else:
            raise AssertionError("expected SandboxFatalError on Deno stdout EOF")
