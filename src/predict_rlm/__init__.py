"""predict-rlm — Production-grade RLMs with tool use, built on DSPy.

Core classes:
    PredictRLM — RLM with a ``predict()`` tool for running DSPy signatures
    Skill — Reusable bundle of instructions, packages, and tools

File I/O:
    File — Unified file type for inputs (mount into sandbox) and outputs
           (sync from sandbox). Use ``list[File]`` for multiple files.
    Workspace — Mutable input directory mounted into the sandbox and synced back.

Lifecycle callbacks:
    IterationStep — payload passed to ``on_rlm_iteration_end`` callbacks.
                    Subclass ``dspy.utils.callback.BaseCallback`` and
                    implement ``on_rlm_iteration_start``/``on_rlm_iteration_end``
                    to broadcast progress (e.g. to a websocket).
"""

from .backends import (
    BackendName,
    DirectPythonBackend,
    PythonSupervisor,
)
from .files import File, LocalDir, LocalFile, OutputDir, OutputFile, SyncedFile
from .predict_rlm import PredictRLM, SubmitConfirmationContext
from .rlm_skills import Skill
from .runtime_hooks import RuntimeHook, RuntimeHookEvent
from .trace import IterationStep, RunTrace
from .workspace import Workspace, WorkspaceMode

_SBX_EXTRA_INSTALL_HINT = (
    "SbxBackend requires the optional SBX dependencies. "
    "Install them with `pip install 'predict-rlm[sbx]'` or "
    "`uv pip install 'predict-rlm[sbx]'`."
)


def __getattr__(name: str):
    try:
        if name == "DEFAULT_SBX_TEMPLATE":
            from .backends.sbx import DEFAULT_SBX_TEMPLATE

            return DEFAULT_SBX_TEMPLATE
        if name == "SbxConfig":
            from .backends.sbx import SbxConfig

            return SbxConfig
        if name == "SbxBackend":
            from .backends.sbx import SbxBackend

            return SbxBackend
        if name == "SbxPool":
            from .backends.sbx import SbxPool

            return SbxPool
    except ImportError as exc:
        raise ImportError(_SBX_EXTRA_INSTALL_HINT) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "File",
    "DirectPythonBackend",
    "IterationStep",
    "LocalDir",
    "LocalFile",
    "OutputDir",
    "OutputFile",
    "PredictRLM",
    "PythonSupervisor",
    "RunTrace",
    "SubmitConfirmationContext",
    "DEFAULT_SBX_TEMPLATE",
    "BackendName",
    "Skill",
    "SbxConfig",
    "SbxBackend",
    "SbxPool",
    "SyncedFile",
    "RuntimeHook",
    "RuntimeHookEvent",
    "Workspace",
    "WorkspaceMode",
]
