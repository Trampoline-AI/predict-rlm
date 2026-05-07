"""predict-rlm — Production-grade RLMs with tool use, built on DSPy.

Core classes:
    PredictRLM — RLM with a ``predict()`` tool for running DSPy signatures
    Skill — Reusable bundle of instructions, packages, and tools

File I/O:
    File — Unified file type for inputs (mount into sandbox) and outputs
           (sync from sandbox). Use ``list[File]`` for multiple files.
"""

from .files import File, LocalDir, LocalFile, OutputDir, OutputFile, SyncedFile
from .interpreters import (
    DEFAULT_SBX_TEMPLATE,
    SandboxBackend,
    SbxConfig,
    SbxInterpreter,
    SbxPool,
)
from .predict_rlm import PredictRLM
from .rlm_skills import Skill
from .trace import RunTrace

__all__ = [
    "File",
    "LocalDir",
    "LocalFile",
    "OutputDir",
    "OutputFile",
    "PredictRLM",
    "RunTrace",
    "DEFAULT_SBX_TEMPLATE",
    "SandboxBackend",
    "Skill",
    "SbxConfig",
    "SbxInterpreter",
    "SbxPool",
    "SyncedFile",
]
