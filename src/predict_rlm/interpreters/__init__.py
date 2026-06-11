"""Client adapters for predict-rlm execution backends."""

from .base import DEFAULT_SBX_TEMPLATE, PredictRLMClientAdapter, SandboxBackend, SbxConfig
from .python_runner import DirectProcessRunnerClientAdapter, PythonRunnerClientAdapter
from .sbx_pool import SbxPool

_SBX_EXTRA_INSTALL_HINT = (
    "SbxClientAdapter requires the optional SBX dependencies. "
    "Install them with `pip install 'predict-rlm[sbx]'` or "
    "`uv pip install 'predict-rlm[sbx]'`."
)


def __getattr__(name: str):
    if name == "SbxClientAdapter":
        try:
            from .sbx import SbxClientAdapter
        except ImportError as exc:
            raise ImportError(_SBX_EXTRA_INSTALL_HINT) from exc

        return SbxClientAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_SBX_TEMPLATE",
    "DirectProcessRunnerClientAdapter",
    "PredictRLMClientAdapter",
    "PythonRunnerClientAdapter",
    "SandboxBackend",
    "SbxConfig",
    "SbxClientAdapter",
    "SbxPool",
]
