"""predict-rlm — Production-grade RLMs with tool use, built on DSPy.

Core classes:
    PredictRLM — RLM with a ``predict()`` tool for running DSPy signatures
    Skill — Reusable bundle of instructions, packages, and tools

File I/O:
    File — Unified file type for inputs (mount into sandbox) and outputs
           (sync from sandbox). Use ``list[File]`` for multiple files.

Lifecycle callbacks:
    IterationStep — payload passed to ``on_rlm_iteration_end`` callbacks.
                    Subclass ``dspy.utils.callback.BaseCallback`` and
                    implement ``on_rlm_iteration_start``/``on_rlm_iteration_end``
                    to broadcast progress (e.g. to a websocket).
"""

from .files import File, LocalDir, LocalFile, OutputDir, OutputFile, SyncedFile
from .interpreters import (
    DEFAULT_SBX_TEMPLATE,
    DirectProcessRunnerClientAdapter,
    PythonRunnerClientAdapter,
    SandboxBackend,
    SbxConfig,
    SbxPool,
)
from .predict_rlm import PredictRLM, SubmitConfirmationContext
from .rlm_skills import Skill
from .trace import IterationStep, RunTrace

_SBX_EXTRA_INSTALL_HINT = (
    "SbxClientAdapter requires the optional SBX dependencies. "
    "Install them with `pip install 'predict-rlm[sbx]'` or "
    "`uv pip install 'predict-rlm[sbx]'`."
)


def __getattr__(name: str):
    if name == "SbxClientAdapter":
        try:
            from .interpreters.sbx import SbxClientAdapter
        except ImportError as exc:
            raise ImportError(_SBX_EXTRA_INSTALL_HINT) from exc

        return SbxClientAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "File",
    "DirectProcessRunnerClientAdapter",
    "IterationStep",
    "LocalDir",
    "LocalFile",
    "OutputDir",
    "OutputFile",
    "PredictRLM",
    "PythonRunnerClientAdapter",
    "RunTrace",
    "SubmitConfirmationContext",
    "DEFAULT_SBX_TEMPLATE",
    "SandboxBackend",
    "Skill",
    "SbxConfig",
    "SbxClientAdapter",
    "SbxPool",
    "SyncedFile",
]
