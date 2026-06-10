from __future__ import annotations

import importlib
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

import pytest

from predict_rlm.interpreter import JspiClientAdapter
from predict_rlm.interpreters import (
    DirectProcessRunnerClientAdapter,
    SbxClientAdapter,
    SbxConfig,
)

ROOT = Path(__file__).resolve().parents[2]
TERMINAL_BENCH_DIR = ROOT / "examples" / "terminal_bench"
if str(TERMINAL_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(TERMINAL_BENCH_DIR))

runner_module = importlib.import_module("terminal_bench_rlm.tools.runner")
runner_script_path = runner_module.runner_script_path


CAPABILITIES = frozenset(
    {
        "execute",
        "state",
        "reset",
        "code_fences",
        "submit",
        "deferred_submit",
        "recoverable_errors",
        "host_tools",
        "recoverable_iteration_timeout",
        "files",
    }
)


class RuntimeHandle(Protocol):
    spec: RuntimeSpec

    def require(self, capability: str) -> None: ...

    def configure(
        self,
        *,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict[str, Any]] | None = None,
    ) -> None: ...

    def execute(self, code: str, *, timeout: float | None = None) -> Any: ...

    def output(self, result: Any) -> str: ...

    def timeout_observation(self, result: Any) -> dict[str, Any]: ...

    def defer_next_submit_finalization(self) -> None: ...

    def reset(self) -> None: ...

    def shutdown(self) -> None: ...

    def mount_file_at(self, host_path: str, sandbox_path: str) -> None: ...

    def mkdir_p(self, sandbox_path: str) -> None: ...

    def list_dir(self, sandbox_path: str) -> list[str]: ...

    def sync_file_to(self, sandbox_path: str, host_path: str) -> None: ...


@dataclass(frozen=True)
class RuntimeSpec:
    name: str
    adapter: Literal[
        "jspi-process",
        "sbx-cli",
        "direct-process",
        "test-only-local-supervisor",
    ]
    environment: Literal[
        "deno-subprocess",
        "sbx-sandbox",
        "direct-process",
        "local-supervisor-seam",
    ]
    engine: Literal["pyodide-jspi", "python-runner"]
    make: Callable[[Path, "RuntimeSpec"], RuntimeHandle]
    capabilities: frozenset[str]
    opt_in: bool = False
    skip_reason: str | None = None
    xfail_contracts: dict[str, str] = field(default_factory=dict)


def _predict_tool(signature: str, **kwargs: Any) -> dict[str, Any]:
    del signature, kwargs
    return {"answer": "4"}


def _shape_tool(kind: str) -> Any:
    if kind == "list":
        return [1, 2]
    if kind == "dict":
        return {"ok": True}
    if kind == "none":
        return None
    if kind == "text":
        return "hello"
    raise ValueError(f"unknown shape: {kind}")


def _failing_tool() -> str:
    raise ValueError("host tool failed")


def _default_tools() -> dict[str, Callable[..., Any]]:
    return {
        "predict": _predict_tool,
        "shape_tool": _shape_tool,
        "failing_tool": _failing_tool,
    }


class InterpreterRuntimeHandle:
    def __init__(self, spec: RuntimeSpec, interpreter: Any) -> None:
        self.spec = spec
        self.interpreter = interpreter

    def require(self, capability: str) -> None:
        if capability not in CAPABILITIES:
            raise AssertionError(f"unknown runtime capability: {capability}")
        if capability in self.spec.xfail_contracts:
            pytest.xfail(self.spec.xfail_contracts[capability])
        if capability not in self.spec.capabilities:
            pytest.skip(
                self.spec.skip_reason
                or f"{self.spec.name} does not advertise {capability}"
            )

    def configure(
        self,
        *,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict[str, Any]] | None = None,
    ) -> None:
        configure_runtime = getattr(self.interpreter, "configure_runtime", None)
        if configure_runtime is None:
            pytest.skip(f"{self.spec.name} does not support runtime reconfiguration")
        configure_runtime(tools=tools, output_fields=output_fields)

    def execute(self, code: str, *, timeout: float | None = None) -> Any:
        return self.interpreter.execute(code, timeout=timeout)

    def output(self, result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, dict) and "output" in result:
            return str(result["output"])
        return str(result)

    def timeout_observation(self, result: Any) -> dict[str, Any]:
        if isinstance(result, dict) and "timeout" in result:
            return {
                "seconds": result["timeout"]["seconds"],
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "state": result.get("state"),
            }
        return {
            "seconds": getattr(result, "timeout_seconds"),
            "stdout": getattr(result, "stdout", ""),
            "stderr": getattr(result, "stderr", ""),
            "state": getattr(result, "state", None),
        }

    def defer_next_submit_finalization(self) -> None:
        defer = getattr(self.interpreter, "defer_next_submit_finalization", None)
        if defer is None:
            pytest.skip(f"{self.spec.name} does not support deferred submit")
        defer()

    def reset(self) -> None:
        reset = getattr(self.interpreter, "reset", None)
        if reset is None:
            self.interpreter.shutdown()
            return
        reset()

    def shutdown(self) -> None:
        self.interpreter.shutdown()

    def mount_file_at(self, host_path: str, sandbox_path: str) -> None:
        self.interpreter.mount_file_at(host_path, sandbox_path)

    def mkdir_p(self, sandbox_path: str) -> None:
        self.interpreter.mkdir_p(sandbox_path)

    def list_dir(self, sandbox_path: str) -> list[str]:
        return self.interpreter.list_dir(sandbox_path)

    def sync_file_to(self, sandbox_path: str, host_path: str) -> None:
        self.interpreter.sync_file_to(sandbox_path, host_path)


def _make_jspi(tmp_path: Path, spec: RuntimeSpec) -> RuntimeHandle:
    del tmp_path
    if shutil.which("deno") is None:
        pytest.skip("JSPI contracts require Deno")
    return InterpreterRuntimeHandle(
        spec,
        JspiClientAdapter(
            tools=_default_tools(),
            preinstall_packages=False,
            exec_timeout=10,
        ),
    )


def _make_direct_process(tmp_path: Path, spec: RuntimeSpec) -> RuntimeHandle:
    return InterpreterRuntimeHandle(
        spec,
        DirectProcessRunnerClientAdapter(
            tools=_default_tools(),
            runner_path=str(tmp_path / "predict_rlm_runner.py"),
            workdir=str(tmp_path),
            exec_timeout=10,
            recoverable_timeout_grace=1.0,
        ),
    )


def _make_sbx(tmp_path: Path, spec: RuntimeSpec) -> RuntimeHandle:
    if os.environ.get("PREDICT_RLM_RUN_SBX_TESTS") != "1":
        pytest.skip(
            "real SBX runtime contracts require PREDICT_RLM_RUN_SBX_TESTS=1, "
            "the sbx CLI, and sbx login"
        )
    if shutil.which("sbx") is None:
        pytest.skip("real SBX runtime contracts require the sbx CLI")
    return InterpreterRuntimeHandle(
        spec,
        SbxClientAdapter(
            config=SbxConfig(name="runtime-contract-sbx", exec_timeout=10),
            tools=_default_tools(),
            preinstall_packages=False,
            _staging_root=tmp_path / "sbx-staging",
        ),
    )


def _make_internal_jsonrpc(tmp_path: Path, spec: RuntimeSpec) -> RuntimeHandle:
    return InterpreterRuntimeHandle(
        spec,
        SbxClientAdapter(
            config=SbxConfig(name="runtime-contract-local-supervisor", exec_timeout=10),
            tools=_default_tools(),
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(runner_script_path())],
            _staging_root=tmp_path / "internal-jsonrpc-staging",
        ),
    )


def runtime_specs() -> list[RuntimeSpec]:
    return [
        RuntimeSpec(
            name="jspi",
            adapter="jspi-process",
            environment="deno-subprocess",
            engine="pyodide-jspi",
            make=_make_jspi,
            capabilities=frozenset(
                {
                    "execute",
                    "state",
                    "reset",
                    "code_fences",
                    "recoverable_errors",
                    "host_tools",
                    "recoverable_iteration_timeout",
                }
            ),
        ),
        RuntimeSpec(
            name="python-runner/direct-process",
            adapter="direct-process",
            environment="direct-process",
            engine="python-runner",
            make=_make_direct_process,
            capabilities=frozenset(CAPABILITIES),
        ),
        RuntimeSpec(
            name="sbx",
            adapter="sbx-cli",
            environment="sbx-sandbox",
            engine="python-runner",
            make=_make_sbx,
            capabilities=frozenset(CAPABILITIES),
            opt_in=True,
            xfail_contracts={
                "tool_timeout": (
                    "real SBX per-host-tool timeout is not implemented in the "
                    "shared contract matrix yet"
                )
            },
        ),
        RuntimeSpec(
            name="internal/python-runner-jsonrpc",
            adapter="test-only-local-supervisor",
            environment="local-supervisor-seam",
            engine="python-runner",
            make=_make_internal_jsonrpc,
            capabilities=frozenset(CAPABILITIES),
        ),
    ]
