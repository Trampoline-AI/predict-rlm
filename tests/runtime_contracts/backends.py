from __future__ import annotations

import os
import shutil
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from predict_rlm.backends import DirectPythonBackend, JspiBackend

PAYLOAD_PATH = (
    Path(__file__).resolve().parents[2] / "src/predict_rlm/backends/supervisor/_payload.py"
)


def _predict_tool(signature: str, **kwargs: Any) -> dict[str, Any]:
    return {"answer": "4"}


def _shape_tool(kind: str) -> Any:
    return {"list": [1, 2], "dict": {"ok": True}, "none": None, "text": "hello"}[kind]


def _failing_tool() -> None:
    raise ValueError("host tool failed")


def _default_tools() -> dict[str, Callable[..., Any]]:
    return {"predict": _predict_tool, "shape_tool": _shape_tool, "failing_tool": _failing_tool}


@dataclass(frozen=True)
class RuntimeSpec:
    name: str
    make: Callable[[Path, "RuntimeSpec"], "RuntimeHandle"]
    unsupported: frozenset[str] = frozenset()


class RuntimeHandle:
    """Normalize the two legacy result/reset interfaces, not backend behavior."""

    def __init__(self, spec: RuntimeSpec, interpreter: Any) -> None:
        self.spec = spec
        self.interpreter = interpreter

    def __getattr__(self, name: str) -> Any:
        return getattr(self.interpreter, name)

    def require(self, capability: str) -> None:
        if capability in self.spec.unsupported:
            pytest.skip(f"{self.spec.name} does not support {capability}")

    def configure(self, **kwargs: Any) -> None:
        self.interpreter.configure_runtime(**kwargs)

    def reset(self) -> None:
        if isinstance(self.interpreter, JspiBackend):
            self.interpreter.shutdown()
        else:
            self.interpreter.reset()

    @staticmethod
    def output(result: Any) -> str:
        return result["output"] if isinstance(result, dict) else str(result)

    @staticmethod
    def timeout_observation(result: Any) -> dict[str, Any]:
        return {
            "seconds": result.timeout_seconds,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "state": result.state,
        }


def _make_jspi(tmp_path: Path, spec: RuntimeSpec) -> RuntimeHandle:
    if shutil.which("deno") is None:
        pytest.skip("JSPI contracts require Deno")
    return RuntimeHandle(
        spec,
        JspiBackend(
            tools=_default_tools(),
            preinstall_packages=False,
            exec_timeout=10,
        ),
    )


def _make_direct(tmp_path: Path, spec: RuntimeSpec) -> RuntimeHandle:
    return RuntimeHandle(
        spec,
        DirectPythonBackend(
            tools=_default_tools(),
            runner_path=str(tmp_path / "predict_rlm_runner.py"),
            workdir=str(tmp_path),
            exec_timeout=10,
            recoverable_timeout_grace=1.0,
        ),
    )


def _make_sbx(tmp_path: Path, spec: RuntimeSpec) -> RuntimeHandle:
    if spec.name == "sbx" and (
        os.environ.get("PREDICT_RLM_RUN_SBX_TESTS") != "1" or shutil.which("sbx") is None
    ):
        pytest.skip(
            "real SBX contracts require PREDICT_RLM_RUN_SBX_TESTS=1, sbx CLI, and login"
        )
    pytest.importorskip("websockets")
    from predict_rlm.backends import SbxBackend, SbxConfig

    kwargs: dict[str, Any] = {}
    if spec.name == "sbx/local-websocket":
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        path = f"/runtime-contract-{os.getpid()}-{port}"
        kwargs = {
            "_websocket_supervisor_command": [
                sys.executable,
                "-u",
                str(PAYLOAD_PATH),
                "--websocket-host",
                "127.0.0.1",
                "--websocket-port",
                str(port),
                "--websocket-path",
                path,
            ],
            "_websocket_url": f"ws://127.0.0.1:{port}{path}",
        }
    return RuntimeHandle(
        spec,
        SbxBackend(
            config=SbxConfig(name="runtime-contract-sbx", exec_timeout=10),
            tools=_default_tools(),
            preinstall_packages=False,
            _staging_root=tmp_path / "sbx-staging",
            **kwargs,
        ),
    )


def runtime_specs() -> list[RuntimeSpec]:
    return [
        RuntimeSpec("jspi", _make_jspi, frozenset({"deferred_submit"})),
        RuntimeSpec(
            "python-runner/direct-process",
            _make_direct,
            frozenset({"partial_error_output", "concurrent_tools"}),
        ),
        RuntimeSpec("sbx/local-websocket", _make_sbx, frozenset({"deferred_submit"})),
        RuntimeSpec("sbx", _make_sbx, frozenset({"deferred_submit"})),
    ]
