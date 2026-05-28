"""predict-rlm — Production-grade RLMs with tool use, built on DSPy.

Core classes:
    PredictRLM — RLM with a ``predict()`` tool for running DSPy signatures
    Skill — Reusable bundle of instructions, packages, and tools

File I/O:
    File — Unified file type for inputs (mount into sandbox) and outputs
           (sync from sandbox). Use ``list[File]`` for multiple files.
    Workspace — Mutable input directory mounted into the sandbox.
"""

from .files import (
    File,
    LocalDir,
    LocalFile,
    OutputDir,
    OutputFile,
    SyncedFile,
    Workspace,
    WorkspaceMode,
)
from .interpreters import (
    DEFAULT_SBX_TEMPLATE,
    SandboxBackend,
    SbxConfig,
    SbxInterpreter,
    SbxPool,
)
from .predict_rlm import PredictRLM
from .rlm_skills import Skill
from .runtime_hooks import RuntimeHook, RuntimeHookEvent
from .trace import RunTrace

__all__ = [
    "File",
    "LocalDir",
    "LocalFile",
    "OutputDir",
    "OutputFile",
    "PredictRLM",
    "RunTrace",
    "RuntimeHook",
    "RuntimeHookEvent",
    "DEFAULT_SBX_TEMPLATE",
    "SandboxBackend",
    "Skill",
    "SbxConfig",
    "SbxInterpreter",
    "SbxPool",
    "SyncedFile",
    "Workspace",
    "WorkspaceMode",
]
