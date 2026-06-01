from __future__ import annotations

import importlib
import json
import os
import select
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from dspy.primitives.code_interpreter import CodeInterpreterError

from predict_rlm.interpreter import JspiInterpreter
from predict_rlm.interpreters import SbxConfig, SbxInterpreter

_ROOT = Path(__file__).resolve().parents[1]
_TERMINAL_BENCH_DIR = _ROOT / "examples" / "terminal_bench"
if str(_TERMINAL_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_TERMINAL_BENCH_DIR))

_container_runner = importlib.import_module(
    "terminal_bench_rlm.tools.container_runner"
)
_runner = importlib.import_module("terminal_bench_rlm.tools.runner")
LocalProcessRunnerInterpreter = _container_runner.LocalProcessRunnerInterpreter
TerminalBenchRunnerInterpreter = _container_runner.TerminalBenchRunnerInterpreter
runner_script_path = _runner.runner_script_path


FENCES = "fences"
PROTOCOL_STDERR = "protocol_stderr"
RECOVERABLE_TIMEOUT = "recoverable_timeout"
SUBPROCESS_ERROR = "subprocess_error"
RUNNER_DEATH = "runner_death"
STDIN_ISOLATION = "stdin_isolation"
HOST_CALLBACKS = "host_callbacks"
VISUAL_PATH_PREDICT = "visual_path_predict"


def _predict_tool(signature: str, **kwargs: Any) -> dict[str, Any]:
    if "visible_text" in signature:
        image = kwargs.get("image")
        assert isinstance(image, str)
        assert image.startswith("data:image/png;base64,")
        return {"visible_text": "hello"}
    return {"answer": "4"}


def _drain_available_pipe_text(pipe: Any) -> str:
    if pipe is None:
        return ""
    fileno = getattr(pipe, "fileno", None)
    if fileno is None:
        return ""
    chunks: list[str] = []
    try:
        fd = fileno()
    except (OSError, ValueError, AttributeError):
        return ""
    while True:
        ready, _, _ = select.select([fd], [], [], 0)
        if not ready:
            return "".join(chunks)
        chunk = os.read(fd, 65536)
        if not chunk:
            return "".join(chunks)
        chunks.append(chunk.decode("utf-8", errors="replace"))


def _assert_timeout_observation(result: Any, seconds: float) -> None:
    if isinstance(result, dict):
        assert result["timeout"] == {"seconds": seconds}
        assert result["stdout"] == "before timeout\n"
        assert result["stderr"].startswith("stderr before timeout\n")
        return

    text = str(result)
    assert f"[Timeout] Iteration execution timed out after {seconds:g}s" in text
    assert "[stdout]\nbefore timeout" in text
    assert "[stderr]\nstderr before timeout" in text
    assert getattr(result, "timeout_seconds") == seconds


class RunnerBackend:
    name: str
    capabilities: frozenset[str]
    unsupported: dict[str, str]

    def execute(self, code: str, *, timeout: float | None = None) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def protocol_stderr(self) -> str:
        return ""

    def assert_executed_codes(self, expected: list[str]) -> None:
        return None


class JsonRpcPythonRunnerBackend(RunnerBackend):
    name = "jsonrpc-python-runner"
    capabilities = frozenset({
        PROTOCOL_STDERR,
        RECOVERABLE_TIMEOUT,
        SUBPROCESS_ERROR,
        RUNNER_DEATH,
        STDIN_ISOLATION,
        HOST_CALLBACKS,
        VISUAL_PATH_PREDICT,
    })
    unsupported = {
        FENCES: "inapplicable: raw JSON-RPC runner payloads receive code after wrapper normalization",
    }

    def __init__(self, tmp_path: Path) -> None:
        env_root = tmp_path / "jsonrpc-runner-root"
        env_root.mkdir()
        self.proc = subprocess.Popen(
            [sys.executable, "-u", str(runner_script_path())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={**os.environ, "PREDICT_RLM_SBX_ROOT": str(env_root)},
        )
        self._request_id = 0
        self._request("register_tools", {"tools": ["predict"]})

    def execute(self, code: str, *, timeout: float | None = None) -> Any:
        params: dict[str, Any] = {"code": code}
        if timeout is not None:
            params["execution_timeout_seconds"] = timeout
        response = self._request("execute", params)
        if "error" in response:
            error = response["error"]
            data = error.get("data") or {}
            raise CodeInterpreterError(
                f"{data.get('type', 'RuntimeError')}: {error.get('message', '')}"
            )
        result = response.get("result") or {}
        if "timeout" in result:
            return result
        return result.get("output")

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self._request("shutdown", {})
            finally:
                self.proc.wait(timeout=5)

    def protocol_stderr(self) -> str:
        return _drain_available_pipe_text(self.proc.stderr)

    def _request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._request_id += 1
        request_id = self._request_id
        self._write({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        })
        while True:
            message = self._read()
            if message.get("method") == "tool_call":
                self._write(self._tool_response(message))
                continue
            if message.get("id") == request_id:
                return message

    def _write(self, message: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def _read(self) -> dict[str, Any]:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        assert line, self.proc.stderr.read() if self.proc.stderr else ""
        return json.loads(line)

    def _tool_response(self, message: dict[str, Any]) -> dict[str, Any]:
        params = message.get("params") or {}
        result = _predict_tool(*params.get("args", []), **params.get("kwargs", {}))
        return {
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {"type": "json", "value": json.dumps(result)},
        }


class InterpreterBackend(RunnerBackend):
    def __init__(
        self,
        *,
        name: str,
        capabilities: set[str],
        interpreter: Any,
        unsupported: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.capabilities = frozenset(capabilities)
        self.unsupported = unsupported or {}
        self.interpreter = interpreter

    def execute(self, code: str, *, timeout: float | None = None) -> Any:
        return self.interpreter.execute(code, timeout=timeout)

    def close(self) -> None:
        self.interpreter.shutdown()

    def protocol_stderr(self) -> str:
        process = getattr(self.interpreter, "_process", None) or getattr(
            self.interpreter, "_proc", None
        )
        return _drain_available_pipe_text(getattr(process, "stderr", None))


class FakePipe:
    def __init__(self, on_write: Callable[[str], None] | None = None) -> None:
        self.lines: list[str] = []
        self._on_write = on_write

    def write(self, data: str) -> None:
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


class ContractFakeProcess:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.stdout = FakePipe()
        self.stderr = FakePipe()
        self.stdin = FakePipe(self._on_stdin)
        self.killed = False
        self.returncode: int | None = None

    def _on_stdin(self, data: str) -> None:
        request = json.loads(data)
        self.requests.append(request)
        request_id = request["id"]
        method = request["method"]
        params = request.get("params") or {}
        code = params.get("code", "")
        if method == "shutdown":
            self._append({"jsonrpc": "2.0", "id": request_id, "result": {"shutdown": True}})
        elif method != "execute":
            self._append({"jsonrpc": "2.0", "id": request_id, "result": {}})
        elif "print('python')" in code:
            self._append({"jsonrpc": "2.0", "id": request_id, "result": {"output": "python\n"}})
        elif "print('repl')" in code:
            self._append({"jsonrpc": "2.0", "id": request_id, "result": {"output": "repl\n"}})
        elif "survivor = 'ok'" in code:
            self._append({"jsonrpc": "2.0", "id": request_id, "result": {"output": "set\n"}})
        elif "before timeout" in code:
            self._append({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "timeout": {"seconds": params["execution_timeout_seconds"]},
                    "stdout": "before timeout\n",
                    "stderr": "stderr before timeout\n",
                },
            })
        elif "print(survivor)" in code:
            self._append({"jsonrpc": "2.0", "id": request_id, "result": {"output": "ok\n"}})
        elif "child failed" in code:
            self._append({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32000,
                    "message": "Command returned non-zero exit status 7",
                    "data": {
                        "type": "CalledProcessError",
                        "message": "Command returned non-zero exit status 7",
                    },
                },
            })
        elif "after subprocess failure" in code:
            self._append({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"output": "after subprocess failure\n"},
            })
        else:
            self._append({"jsonrpc": "2.0", "id": request_id, "result": {"output": ""}})

    def _append(self, response: dict[str, Any]) -> None:
        self.stdout.lines.append(json.dumps(response) + "\n")

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        return -9 if self.killed else None

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class ContractFakeAdapter:
    def __init__(self, process: ContractFakeProcess) -> None:
        self.process = process

    def copy_to(self, host_path: str, container_path: str) -> None:
        return None

    def copy_from(self, container_path: str, host_path: str) -> None:
        return None

    def exec(self, command: list[str], *, timeout: float | None = None) -> Any:
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    def start_exec(
        self,
        command: list[str],
        *,
        workdir: str | None = None,
        timeout: float | None = None,
    ) -> ContractFakeProcess:
        return self.process


class FakeTerminalBenchBackend(InterpreterBackend):
    def __init__(self) -> None:
        self.process = ContractFakeProcess()
        super().__init__(
            name="terminal-bench-fake-adapter",
            capabilities={FENCES, RECOVERABLE_TIMEOUT, SUBPROCESS_ERROR},
            interpreter=TerminalBenchRunnerInterpreter(
                object(),
                container_adapter=ContractFakeAdapter(self.process),
                runner_path="/tmp/predict_rlm_runner.py",
            ),
            unsupported={
                PROTOCOL_STDERR: "inapplicable: FakeAdapter validates wrapper mapping without real protocol stderr",
                RUNNER_DEATH: "inapplicable: FakeAdapter has no real child process to kill",
                STDIN_ISOLATION: "inapplicable: FakeAdapter has no real JSON-RPC stdin or OS fd 0",
                HOST_CALLBACKS: "inapplicable: FakeAdapter does not execute code or issue tool calls",
                VISUAL_PATH_PREDICT: "inapplicable: FakeAdapter does not provide filesystem or callback execution",
            },
        )

    def assert_executed_codes(self, expected: list[str]) -> None:
        actual = [
            request.get("params", {}).get("code")
            for request in self.process.requests
            if request.get("method") == "execute"
        ]
        assert actual[: len(expected)] == expected


@dataclass(frozen=True)
class BackendSpec:
    name: str
    factory: Callable[[Path], RunnerBackend]
    unsupported: dict[str, str] = field(default_factory=dict)


def _local_process_backend(tmp_path: Path) -> RunnerBackend:
    return InterpreterBackend(
        name="terminal-bench-local-process",
        capabilities={
            FENCES,
            PROTOCOL_STDERR,
            RECOVERABLE_TIMEOUT,
            SUBPROCESS_ERROR,
            RUNNER_DEATH,
            STDIN_ISOLATION,
            HOST_CALLBACKS,
            VISUAL_PATH_PREDICT,
        },
        interpreter=LocalProcessRunnerInterpreter(
            tools={"predict": _predict_tool},
            runner_path=str(tmp_path / "predict_rlm_runner.py"),
            workdir=str(tmp_path),
            exec_timeout=10,
            recoverable_timeout_grace=1.0,
        ),
    )


def _sbx_local_backend(tmp_path: Path) -> RunnerBackend:
    return InterpreterBackend(
        name="sbx-local-supervisor",
        capabilities={
            FENCES,
            PROTOCOL_STDERR,
            RECOVERABLE_TIMEOUT,
            SUBPROCESS_ERROR,
            RUNNER_DEATH,
            STDIN_ISOLATION,
            HOST_CALLBACKS,
            VISUAL_PATH_PREDICT,
        },
        interpreter=SbxInterpreter(
            config=SbxConfig(name="contract-local", exec_timeout=10),
            tools={"predict": _predict_tool},
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(runner_script_path())],
            _staging_root=tmp_path / "sbx-staging",
        ),
    )


def _real_sbx_backend(tmp_path: Path) -> RunnerBackend:
    if os.environ.get("PREDICT_RLM_RUN_SBX_TESTS") != "1":
        pytest.skip(
            "opt-in/integration-only: set PREDICT_RLM_RUN_SBX_TESTS=1 to run real Docker SBX contracts"
        )
    if shutil.which("sbx") is None:
        pytest.skip("opt-in/integration-only: real Docker SBX contracts require the sbx CLI")
    return InterpreterBackend(
        name="sbx-real-docker",
        capabilities={
            FENCES,
            PROTOCOL_STDERR,
            RECOVERABLE_TIMEOUT,
            SUBPROCESS_ERROR,
            RUNNER_DEATH,
            STDIN_ISOLATION,
            HOST_CALLBACKS,
            VISUAL_PATH_PREDICT,
        },
        interpreter=SbxInterpreter(
            config=SbxConfig(name="contract-real-sbx", exec_timeout=10),
            tools={"predict": _predict_tool},
            preinstall_packages=False,
            _staging_root=tmp_path / "real-sbx-staging",
        ),
    )


def _jspi_backend(tmp_path: Path) -> RunnerBackend:
    del tmp_path
    if os.environ.get("PREDICT_RLM_RUN_JSPI_CONTRACTS") != "1":
        pytest.skip(
            "opt-in/integration-only: set PREDICT_RLM_RUN_JSPI_CONTRACTS=1 to run Deno/JSPI contracts"
        )
    return InterpreterBackend(
        name="jspi-deno",
        capabilities={FENCES, HOST_CALLBACKS},
        interpreter=JspiInterpreter(
            tools={"predict": _predict_tool},
            preinstall_packages=False,
            exec_timeout=10,
        ),
        unsupported={
            PROTOCOL_STDERR: "inapplicable: Deno/JSPI is not the shared CPython JSON-RPC runner",
            RECOVERABLE_TIMEOUT: "inapplicable here: JSPI timeout behavior is covered by Deno integration timeout tests",
            SUBPROCESS_ERROR: "inapplicable: Pyodide/Deno unit contracts do not expose CPython subprocess semantics",
            RUNNER_DEATH: "inapplicable: Pyodide cannot execute os._exit in a CPython child runner",
            STDIN_ISOLATION: "inapplicable: Pyodide/Deno has no CPython child process fd 0 contract",
            VISUAL_PATH_PREDICT: "inapplicable: this contract covers shared runner /sandbox path mapping",
        },
    )


BACKENDS = [
    BackendSpec("jsonrpc-python-runner", JsonRpcPythonRunnerBackend),
    BackendSpec("terminal-bench-local-process", _local_process_backend),
    BackendSpec("terminal-bench-fake-adapter", lambda tmp_path: FakeTerminalBenchBackend()),
    BackendSpec("sbx-local-supervisor", _sbx_local_backend),
    BackendSpec("sbx-real-docker", _real_sbx_backend),
    BackendSpec("jspi-deno", _jspi_backend),
]


@pytest.fixture(params=BACKENDS, ids=lambda spec: spec.name)
def runner_backend(request: pytest.FixtureRequest, tmp_path: Path) -> RunnerBackend:
    backend = request.param.factory(tmp_path)
    try:
        yield backend
    finally:
        backend.close()


def _require(backend: RunnerBackend, capability: str) -> None:
    if capability not in backend.capabilities:
        pytest.skip(
            backend.unsupported.get(
                capability,
                f"inapplicable: {backend.name} does not support {capability}",
            )
        )


def test_contract_fenced_python_and_repl_code_normalized_before_execution(
    runner_backend: RunnerBackend,
) -> None:
    _require(runner_backend, FENCES)

    python_result = runner_backend.execute("```python\nprint('python')\n```")
    repl_result = runner_backend.execute("```repl\nprint('repl')\n```")

    assert "python\n" in str(python_result)
    assert "repl\n" in str(repl_result)
    runner_backend.assert_executed_codes([
        "print('python')",
        "print('repl')",
    ])


def test_contract_child_stdout_stderr_stay_in_execute_result(
    runner_backend: RunnerBackend,
) -> None:
    _require(runner_backend, PROTOCOL_STDERR)

    result = runner_backend.execute(
        "import subprocess, sys\n"
        "subprocess.run([\n"
        "    sys.executable,\n"
        "    '-c',\n"
        "    \"import sys; print('child stdout'); print('child stderr', file=sys.stderr)\",\n"
        "])\n",
        timeout=2,
    )
    followup = runner_backend.execute("print('runner still usable')")

    assert str(result) == "child stdout\nchild stderr\n"
    assert str(followup) == "runner still usable\n"
    assert runner_backend.protocol_stderr() == ""


def test_contract_fd_print_and_child_output_stay_in_execute_result(
    runner_backend: RunnerBackend,
) -> None:
    _require(runner_backend, PROTOCOL_STDERR)

    result = runner_backend.execute(
        "import os, subprocess\n"
        "os.write(1, b'FD1\\n')\n"
        "os.write(2, b'FD2\\n')\n"
        "print('PRINT', flush=True)\n"
        "subprocess.run(['bash', '-lc', 'echo CHILD_OUT; echo CHILD_ERR >&2'])\n",
        timeout=2,
    )

    assert str(result) == "FD1\nPRINT\nCHILD_OUT\nFD2\nCHILD_ERR\n"
    assert runner_backend.protocol_stderr() == ""


def test_contract_recoverable_timeout_returns_observation_and_survives(
    runner_backend: RunnerBackend,
) -> None:
    _require(runner_backend, RECOVERABLE_TIMEOUT)

    assert str(runner_backend.execute("survivor = 'ok'\nprint('set')")) == "set\n"
    timeout_result = runner_backend.execute(
        "import sys, time\n"
        "print('before timeout')\n"
        "print('stderr before timeout', file=sys.stderr)\n"
        "sys.stdout.flush(); sys.stderr.flush()\n"
        "time.sleep(30)\n",
        timeout=0.2,
    )
    followup = runner_backend.execute("print(survivor)")

    _assert_timeout_observation(timeout_result, 0.2)
    assert str(followup) == "ok\n"


def test_contract_nonzero_user_subprocess_error_is_recoverable(
    runner_backend: RunnerBackend,
) -> None:
    _require(runner_backend, SUBPROCESS_ERROR)

    with pytest.raises(CodeInterpreterError) as exc_info:
        runner_backend.execute(
            "import subprocess, sys\n"
            "subprocess.run(\n"
            "    [sys.executable, '-c', 'import sys; print(\"child failed\", file=sys.stderr); sys.exit(7)'],\n"
            "    capture_output=True,\n"
            "    text=True,\n"
            "    check=True,\n"
            ")\n",
            timeout=2,
        )
    followup = runner_backend.execute("print('after subprocess failure')")

    message = str(exc_info.value)
    assert "CalledProcessError" in message
    assert "non-zero exit status 7" in message or "non-zero" in message or "7" in message
    assert str(followup) == "after subprocess failure\n"


def test_contract_runner_child_death_surfaces_error_and_supervisor_survives(
    runner_backend: RunnerBackend,
) -> None:
    _require(runner_backend, RUNNER_DEATH)

    with pytest.raises(CodeInterpreterError) as exc_info:
        runner_backend.execute("import os\nos._exit(7)")
    followup = runner_backend.execute("print('after runner child exit')")

    message = str(exc_info.value)
    assert "RuntimeError" in message
    assert "execution runner exited without a result" in message
    assert "exitcode=7" in message
    assert str(followup) == "after runner child exit\n"


def test_contract_protocol_stdin_is_isolated_from_user_subprocesses(
    runner_backend: RunnerBackend,
) -> None:
    _require(runner_backend, STDIN_ISOLATION)

    runner_backend.execute("sentinel = 123")
    result = runner_backend.execute(
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
    followup = runner_backend.execute("print(sentinel)")

    assert str(result) == "b''\n"
    assert str(followup) == "123\n"


def test_contract_host_tool_callback_round_trips(
    runner_backend: RunnerBackend,
) -> None:
    _require(runner_backend, HOST_CALLBACKS)

    result = runner_backend.execute(
        "result = await predict('question -> answer', question='2+2?')\n"
        "print(result['answer'])"
    )

    assert str(result) == "4\n"


def test_contract_visual_data_url_path_round_trips_to_host_tool(
    runner_backend: RunnerBackend,
) -> None:
    _require(runner_backend, VISUAL_PATH_PREDICT)
    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"

    result = runner_backend.execute(
        "import base64\n"
        f"open('/sandbox/image.png', 'wb').write({png_bytes!r})\n"
        "image_bytes = open('/sandbox/image.png', 'rb').read()\n"
        "data_url = 'data:image/png;base64,' + base64.b64encode(image_bytes).decode()\n"
        "result = await predict(\n"
        "    'image: dspy.Image, question: str -> visible_text: str',\n"
        "    image=data_url,\n"
        "    question='What text is visible?',\n"
        ")\n"
        "print(result.visible_text)"
    )

    assert str(result) == "hello\n"
