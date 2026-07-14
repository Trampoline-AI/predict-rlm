"""Declarative file I/O types for PredictRLM signatures.

Use ``File`` as the type for file-typed fields in DSPy signatures.
The behavior is determined by whether the field is an input or output:

- **Input field** (``dspy.InputField``): the file is mounted from the host
  into the sandbox at ``/sandbox/input/{field_name}/``.
- **Output field** (``dspy.OutputField``): the RLM writes to
  ``/sandbox/output/{field_name}/`` and the file is synced back to the host.

``list[File]`` works for both multiple inputs and multiple outputs.

Example::

    class ConvertPDF(dspy.Signature):
        source: File = dspy.InputField(desc="PDF to convert")
        result: File = dspy.OutputField(desc="Generated Excel file")

    rlm = PredictRLM(ConvertPDF, lm="openai/gpt-5.4", sub_lm="openai/gpt-5.1")
    prediction = await rlm.acall(source=File(path="report.pdf"))
    print(prediction.result.path)  # host path to the generated file
"""

from __future__ import annotations

import os
import typing
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, Field

from .runtime import FieldDescriptor
from .workspace import (
    DEFAULT_WORKSPACE_EXCLUDES,
    DirectWorkspaceMount,
    Workspace,
    WorkspaceFileInfo,
    WorkspaceMode,
    WorkspaceSyncConflict,
    WorkspaceSyncConflictError,
    WorkspaceSyncState,
)

__all__ = [
    "DEFAULT_WORKSPACE_EXCLUDES",
    "File",
    "DirectWorkspaceMount",
    "LocalDir",
    "LocalFile",
    "OutputDir",
    "OutputFile",
    "SyncedFile",
    "Workspace",
    "WorkspaceFileInfo",
    "WorkspaceMode",
    "WorkspaceSyncConflict",
    "WorkspaceSyncConflictError",
    "WorkspaceSyncState",
    "build_file_instructions",
    "build_file_plan",
    "get_synced_file_params",
    "is_file_type",
    "is_input_file_type",
    "is_output_file_type",
    "is_workspace_type",
    "scan_file_fields",
    "scan_workspace_fields",
]


class File(BaseModel):
    """A file reference for PredictRLM signatures.

    Behavior depends on the field position in the signature:
    - As an input field: mounts the file from the host into the sandbox.
    - As an output field: syncs the file from the sandbox back to the host.
    """

    path: str | None = Field(
        default=None,
        description="Path to the file. For inputs, the host path to mount. "
        "For outputs, populated after execution with the host path.",
    )

    @classmethod
    def from_dir(cls, path: str) -> list[File]:
        """Create a list of File references from all files in a directory.

        Walks the directory recursively and returns a File for each file found.
        """
        files: list[File] = []
        for root, _dirs, filenames in os.walk(path):
            for fname in sorted(filenames):
                files.append(cls(path=os.path.join(root, fname)))
        return files


# Deprecated aliases — kept for backwards compatibility
LocalFile = File
LocalDir = File
OutputFile = File
OutputDir = File


@dataclass(frozen=True)
class SyncedFile:
    """Annotation marker for tool parameters that need sandbox-host file sync.

    Use with ``typing.Annotated`` on tool function parameters to declare that
    a parameter is a sandbox file path. The framework automatically syncs the
    file from the sandbox to the host before calling the tool, and optionally
    mounts the modified file back into the sandbox after the tool returns.

    Example::

        def recalculate(
            workbook: Annotated[Path, SyncedFile(host_dir="/tmp/wb")],
            reference: Annotated[Path, SyncedFile(writeback=False)],
        ) -> str:
            ...
    """

    writeback: bool = True
    """If True (default), mount the file back into the sandbox after the tool
    returns. Set to False for read-only access (skip the mount-after step)."""

    host_dir: str | None = None
    """Host directory for the synced file. If None, a temporary directory is
    created and cleaned up after the call. If specified, the directory is used
    as-is and not cleaned up."""


def is_file_type(annotation: Any) -> bool:
    """Check if a field annotation is File or list[File]."""
    return FieldDescriptor("", annotation).matches(File)


def is_workspace_type(annotation: Any) -> bool:
    """Check if a field annotation is Workspace or list[Workspace]."""
    return FieldDescriptor("", annotation).matches(Workspace)


# Deprecated aliases
is_input_file_type = is_file_type
is_output_file_type = is_file_type


def scan_file_fields(
    signature: Any,
) -> tuple[dict[str, str], dict[str, str]]:
    """Scan a DSPy signature for file-typed fields.

    Returns:
        (input_file_fields, output_file_fields) — dicts mapping field names
        to 'file' or 'list_file'.
    """
    input_file_fields: dict[str, str] = {}
    output_file_fields: dict[str, str] = {}

    for name, field in signature.input_fields.items():
        descriptor = FieldDescriptor(name, field.annotation)
        if descriptor.matches(File):
            kind = "list_file" if descriptor.is_list else "file"
            input_file_fields[name] = kind

    for name, field in signature.output_fields.items():
        descriptor = FieldDescriptor(name, field.annotation)
        if descriptor.matches(File):
            kind = "list_file" if descriptor.is_list else "file"
            output_file_fields[name] = kind

    return input_file_fields, output_file_fields


def scan_workspace_fields(signature: Any) -> dict[str, str]:
    """Scan a DSPy signature for Workspace-typed input fields."""
    input_workspace_fields: dict[str, str] = {}

    for name, field in signature.input_fields.items():
        descriptor = FieldDescriptor(name, field.annotation)
        if descriptor.matches(Workspace):
            kind = "list_workspace" if descriptor.is_list else "workspace"
            input_workspace_fields[name] = kind

    output_workspace_fields = [
        name
        for name, field in signature.output_fields.items()
        if is_workspace_type(field.annotation)
    ]
    if output_workspace_fields:
        names = ", ".join(output_workspace_fields)
        raise TypeError(
            f"Workspace fields are input-only: {names}. Use Workspace as an "
            "InputField for mutable directory edits, or File/list[File] as "
            "OutputField for generated artifacts."
        )

    return input_workspace_fields


def build_file_instructions(
    input_mounts: dict[str, str | list[str]],
    output_dirs: dict[str, str],
    workspace_mounts: dict[str, str | list[str]] | None = None,
) -> str:
    """Generate the '## Files' instructions block for the RLM.

    Args:
        input_mounts: Maps field names to sandbox paths (str for file, list for dir).
        output_dirs: Maps field names to sandbox output directory paths.
    """
    lines = ["## Files\n"]

    if input_mounts:
        lines.append(
            "Input files (available in the sandbox filesystem "
            "— use standard Python file I/O):"
        )
        for field_name, sandbox_path in input_mounts.items():
            if isinstance(sandbox_path, list):
                lines.append(f"- `{field_name}`: directory at /sandbox/input/{field_name}/")
                for p in sandbox_path:
                    lines.append(f"  - {p}")
            else:
                lines.append(f"- `{field_name}`: {sandbox_path}")
        lines.append("")

    if output_dirs:
        lines.append(
            "Output directories (write your output files here, "
            "then SUBMIT the sandbox path you wrote to):"
        )
        for field_name, sandbox_dir in output_dirs.items():
            lines.append(f"- `{field_name}`: write to {sandbox_dir}")
        lines.append("")

    if workspace_mounts:
        lines.append(
            "Workspace directories (mutable host directories mounted in the sandbox):"
        )
        for field_name, sandbox_path in workspace_mounts.items():
            if isinstance(sandbox_path, list):
                lines.append(f"- `{field_name}`: workspaces at {', '.join(sandbox_path)}")
            else:
                lines.append(f"- `{field_name}`: workspace at {sandbox_path}")
        lines.append(
            "Edit workspace files using standard Python/os/pathlib APIs under the "
            "mounted path. Mirror-mode workspace changes sync back to the host after "
            "each code block, including failed code blocks when the sandbox remains "
            "alive. Direct SBX workspaces update host files immediately. "
            "Do not submit workspace files as `File` outputs; they sync automatically."
        )
        lines.append("")

    return "\n".join(lines)


def _uses_default_workspace_mount_path(workspace: Workspace) -> bool:
    return (
        "mount_path" not in workspace.model_fields_set
        and workspace.mount_path == Workspace.model_fields["mount_path"].default
    )


def _direct_workspace_sandbox_path(workspace: Workspace, workspace_root: str) -> str:
    if os.path.islink(workspace_root):
        raise ValueError(f"Workspace path cannot be a symlink: {workspace.path}")
    if _uses_default_workspace_mount_path(workspace):
        return workspace_root
    mount_path = workspace.mount_path
    if mount_path == "/sandbox" or mount_path.startswith("/sandbox/"):
        raise ValueError(
            "Workspace(mode='direct') mount_path must not be under /sandbox; "
            "omit mount_path to use the SBX-mounted host path, or pass an absolute "
            "sandbox path such as /workspace."
        )
    if not mount_path.startswith("/"):
        raise ValueError("Workspace(mode='direct') mount_path must be absolute.")
    return mount_path


def build_file_plan(
    input_args: dict[str, Any],
    input_file_fields: dict[str, str],
    output_file_fields: dict[str, str],
    output_dir: str | None = None,
    input_workspace_fields: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Build the file plan for mounting/syncing.

    Returns None if there are no file fields. Otherwise returns:
        {
            "mounts": [(host_path, virtual_path), ...],
            "read_paths": [host_path, ...],
            "output_dirs": [virtual_path, ...],
            "write_dir": str | None,  # host output base dir
            "output_field_map": {field_name: {"virtual_dir": str, "host_dir": str, "kind": str}},
            "input_mounts_for_instructions": {field_name: sandbox_path_str | [paths]},
            "output_dirs_for_instructions": {field_name: sandbox_dir_str},
            "instructions": str,
        }
    """
    input_workspace_fields = input_workspace_fields or {}
    if not input_file_fields and not output_file_fields and not input_workspace_fields:
        return None

    import tempfile

    mounts: list[tuple[str, str]] = []
    read_paths: list[str] = []
    write_paths: list[str] = []
    input_mounts_for_instructions: dict[str, str | list[str]] = {}
    workspace_mounts_for_instructions: dict[str, str | list[str]] = {}
    workspace_states: list[WorkspaceSyncState] = []
    direct_workspace_mounts: list[DirectWorkspaceMount] = []

    # Process input file fields
    for field_name, kind in input_file_fields.items():
        value = input_args.get(field_name)
        if value is None:
            continue

        if kind == "list_file":
            # list[File] — mount each file
            file_paths: list[str] = []
            for item in value:
                host_path = item.path
                basename = os.path.basename(host_path)
                virtual_path = f"/sandbox/input/{field_name}/{basename}"
                mounts.append((host_path, virtual_path))
                read_paths.append(host_path)
                file_paths.append(virtual_path)
            input_mounts_for_instructions[field_name] = file_paths
        elif kind == "file":
            host_path = value.path
            basename = os.path.basename(host_path)
            virtual_path = f"/sandbox/input/{field_name}/{basename}"
            mounts.append((host_path, virtual_path))
            read_paths.append(host_path)
            input_mounts_for_instructions[field_name] = virtual_path

    # Process output file fields
    output_field_map: dict[str, dict[str, str]] = {}
    output_dirs_virtual: list[str] = []
    output_dirs_for_instructions: dict[str, str] = {}

    # Determine host output base directory
    if output_file_fields:
        host_output_base = output_dir or tempfile.mkdtemp(prefix="predict-rlm-")
    else:
        host_output_base = None

    for field_name, kind in output_file_fields.items():
        virtual_dir = f"/sandbox/output/{field_name}"
        output_dirs_virtual.append(virtual_dir)
        output_dirs_for_instructions[field_name] = f"{virtual_dir}/"

        # Check if user specified a path on the File
        output_value = input_args.get(field_name)
        if output_value and hasattr(output_value, "path") and output_value.path:
            host_dir = output_value.path
        else:
            host_dir = os.path.join(host_output_base, field_name)

        output_field_map[field_name] = {
            "virtual_dir": virtual_dir,
            "host_dir": host_dir,
            "kind": kind,
        }

    for field_name, kind in input_workspace_fields.items():
        value = input_args.get(field_name)
        if value is None:
            continue

        workspaces = value if kind == "list_workspace" else [value]
        mount_paths: list[str] = []
        for workspace in workspaces:
            workspace_root = os.path.abspath(workspace.path)
            if workspace.mode is WorkspaceMode.DIRECT:
                sandbox_path = _direct_workspace_sandbox_path(workspace, workspace_root)
                direct_workspace_mounts.append(
                    DirectWorkspaceMount(
                        host_path=workspace_root,
                        sandbox_path=sandbox_path,
                    )
                )
                mount_paths.append(sandbox_path)
            else:
                state = WorkspaceSyncState(workspace)
                workspace_states.append(state)
                mount_paths.append(workspace.mount_path)
            read_paths.append(workspace_root)
            write_paths.append(workspace_root)
        workspace_mounts_for_instructions[field_name] = (
            mount_paths if kind == "list_workspace" else mount_paths[0]
        )

    seen_mount_paths: dict[str, str] = {}
    for field_name, mount_paths in workspace_mounts_for_instructions.items():
        paths = mount_paths if isinstance(mount_paths, list) else [mount_paths]
        for mount_path in paths:
            if mount_path in seen_mount_paths:
                raise ValueError(
                    "Duplicate Workspace.mount_path values are not supported: "
                    f"{mount_path!r} appears in both {seen_mount_paths[mount_path]!r} "
                    f"and {field_name!r}"
                )
            seen_mount_paths[mount_path] = field_name

    instructions = build_file_instructions(
        input_mounts_for_instructions,
        output_dirs_for_instructions,
        workspace_mounts_for_instructions,
    )

    return {
        "mounts": mounts,
        "read_paths": read_paths,
        "write_paths": write_paths,
        "output_dirs": output_dirs_virtual,
        "write_dir": host_output_base,
        "output_field_map": output_field_map,
        "workspace_states": workspace_states,
        "direct_workspace_mounts": direct_workspace_mounts,
        "workspace_mounts_for_instructions": workspace_mounts_for_instructions,
        "instructions": instructions,
    }


def get_synced_file_params(fn: Any) -> dict[str, SyncedFile]:
    """Extract SyncedFile annotations from a tool function's type hints.

    Returns a dict mapping parameter names to their ``SyncedFile`` marker
    for all parameters annotated with ``Annotated[..., SyncedFile(...)]``.
    """
    try:
        hints = typing.get_type_hints(fn, include_extras=True)
    except (TypeError, NameError):
        return {}

    result: dict[str, SyncedFile] = {}
    for name, hint in hints.items():
        if name == "return":
            continue
        if typing.get_origin(hint) is Annotated:
            for arg in typing.get_args(hint)[1:]:
                if isinstance(arg, SyncedFile):
                    result[name] = arg
                    break
    return result
