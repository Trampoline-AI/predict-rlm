"""Execution backends for predict-rlm."""

from predict_rlm.runtime import (
    ExecutionBackend,
    ExecutionSession,
)

from .adapters import (
    ExistingExecutionBackendAdapter,
    InterpreterBackendAdapter,
    InterpreterExecutionSession,
    NativeInterpreterExecutionSession,
)
from .base import BackendName, LegacyExecutionBackend
from .jspi import JspiBackend, JspiExecutionBackend, JspiExecutionSession
from .supervisor import DirectPythonBackend, PythonSupervisor

_SBX_EXTRA_INSTALL_HINT = (
    "SbxBackend requires the optional SBX dependencies. "
    "Install them with `pip install 'predict-rlm[sbx]'` or "
    "`uv pip install 'predict-rlm[sbx]'`."
)


def __getattr__(name: str):
    try:
        if name == "DEFAULT_SBX_TEMPLATE":
            from .sbx import DEFAULT_SBX_TEMPLATE

            return DEFAULT_SBX_TEMPLATE
        if name == "SbxConfig":
            from .sbx import SbxConfig

            return SbxConfig
        if name == "SbxBackend":
            from .sbx import SbxBackend

            return SbxBackend
        if name == "SbxPool":
            from .sbx import SbxPool

            return SbxPool
    except ImportError as exc:
        raise ImportError(_SBX_EXTRA_INSTALL_HINT) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_SBX_TEMPLATE",
    "DirectPythonBackend",
    "ExecutionBackend",
    "ExecutionSession",
    "ExistingExecutionBackendAdapter",
    "InterpreterBackendAdapter",
    "InterpreterExecutionSession",
    "NativeInterpreterExecutionSession",
    "JspiBackend",
    "JspiExecutionBackend",
    "JspiExecutionSession",
    "LegacyExecutionBackend",
    "PythonSupervisor",
    "BackendName",
    "SbxConfig",
    "SbxBackend",
    "SbxPool",
]
