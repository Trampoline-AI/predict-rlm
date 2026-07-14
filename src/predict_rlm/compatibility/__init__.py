"""Built-in modules that preserve pre-kernel PredictRLM behavior."""

from predict_rlm.runtime import RuntimeContribution

from .backends import execution_from_legacy_options
from .files import (
    FileInputAdapter,
    FileOutputAdapter,
    ValueInputAdapter,
    WorkspaceInputAdapter,
    validate_file_workspace_signature,
)
from .synced_files import SyncedFileToolOperation


def files(*, output_dir: str | None = None) -> RuntimeContribution:
    return RuntimeContribution(
        inputs=(FileInputAdapter(), WorkspaceInputAdapter()),
        outputs=(FileOutputAdapter(output_dir),),
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
    "execution_from_legacy_options",
    "synced_files",
    "SyncedFileToolOperation",
    "validate_file_workspace_signature",
]
