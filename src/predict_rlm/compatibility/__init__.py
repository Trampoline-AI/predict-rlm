"""Built-in modules that preserve pre-kernel PredictRLM behavior."""

from typing import Literal

from predict_rlm.runtime import PathMode, RuntimeContribution

from .backends import execution_from_options
from .files import (
    FileInputAdapter,
    FileOutputAdapter,
    ValueInputAdapter,
    WorkspaceInputAdapter,
    validate_file_workspace_signature,
)
from .synced_files import SyncedFileToolOperation


def files(
    *,
    input_mode: PathMode | Literal["copy", "mount"] = PathMode.COPY,
    output_dir: str | None = None,
) -> RuntimeContribution:
    """Configure File ingress and the host root for generated File outputs."""
    return RuntimeContribution(
        adapters=(
            FileInputAdapter(mode=input_mode),
            WorkspaceInputAdapter(),
            FileOutputAdapter(output_dir),
        ),
        validators=(validate_file_workspace_signature,),
    )


def synced_files() -> RuntimeContribution:
    """Enable the legacy SyncedFile tool annotation compatibility behavior."""
    return RuntimeContribution(tool_operations=(SyncedFileToolOperation(),))


__all__ = [
    "FileInputAdapter",
    "FileOutputAdapter",
    "ValueInputAdapter",
    "WorkspaceInputAdapter",
    "files",
    "execution_from_options",
    "synced_files",
    "SyncedFileToolOperation",
    "validate_file_workspace_signature",
]
