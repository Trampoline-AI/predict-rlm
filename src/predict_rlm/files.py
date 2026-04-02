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
from typing import Any

from pydantic import BaseModel, Field


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


def _unwrap_annotation(annotation: Any) -> Any:
    """Unwrap Optional/Annotated/list to get the inner file type."""
    origin = typing.get_origin(annotation)
    if origin is typing.Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _unwrap_annotation(args[0])
    if origin is typing.Annotated:
        return _unwrap_annotation(typing.get_args(annotation)[0])
    if origin is list:
        args = typing.get_args(annotation)
        if args:
            return _unwrap_annotation(args[0])
    return annotation


def _is_list_annotation(annotation: Any) -> bool:
    """Check if an annotation is list[...] (possibly wrapped in Optional)."""
    origin = typing.get_origin(annotation)
    if origin is typing.Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _is_list_annotation(args[0])
    if origin is typing.Annotated:
        return _is_list_annotation(typing.get_args(annotation)[0])
    return origin is list


def is_file_type(annotation: Any) -> bool:
    """Check if a field annotation is File or list[File]."""
    inner = _unwrap_annotation(annotation)
    return isinstance(inner, type) and issubclass(inner, File)


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
        annotation = field.annotation
        if is_file_type(annotation):
            kind = "list_file" if _is_list_annotation(annotation) else "file"
            input_file_fields[name] = kind

    for name, field in signature.output_fields.items():
        annotation = field.annotation
        if is_file_type(annotation):
            kind = "list_file" if _is_list_annotation(annotation) else "file"
            output_file_fields[name] = kind

    return input_file_fields, output_file_fields


def build_file_instructions(
    input_mounts: dict[str, str | list[str]],
    output_dirs: dict[str, str],
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

    return "\n".join(lines)


def build_file_plan(
    input_args: dict[str, Any],
    input_file_fields: dict[str, str],
    output_file_fields: dict[str, str],
    output_dir: str | None = None,
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
    if not input_file_fields and not output_file_fields:
        return None

    import tempfile

    mounts: list[tuple[str, str]] = []
    read_paths: list[str] = []
    input_mounts_for_instructions: dict[str, str | list[str]] = {}

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

    instructions = build_file_instructions(
        input_mounts_for_instructions, output_dirs_for_instructions
    )

    return {
        "mounts": mounts,
        "read_paths": read_paths,
        "output_dirs": output_dirs_virtual,
        "write_dir": host_output_base,
        "output_field_map": output_field_map,
        "instructions": instructions,
    }
