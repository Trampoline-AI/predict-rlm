"""File and Workspace compatibility adapters built on opaque artifacts."""

from __future__ import annotations

import os
import tempfile
import typing
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from predict_rlm.files import (
    File,
    _direct_workspace_sandbox_path,
)
from predict_rlm.runtime import (
    Artifact,
    ExecutionSession,
    FieldDescriptor,
    InputAdapter,
    OutputAdapter,
    OutputReservation,
    PreparedInput,
    RunContext,
)
from predict_rlm.workspace import (
    DirectWorkspaceMount,
    Workspace,
    WorkspaceMode,
    WorkspaceSyncState,
)


def _annotation_contains(annotation: Any, value_type: type[Any]) -> bool:
    if annotation is value_type:
        return True
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return _annotation_contains(typing.get_args(annotation)[0], value_type)
    return any(
        _annotation_contains(item, value_type)
        for item in typing.get_args(annotation)
        if item not in (type(None), Ellipsis)
    )


def validate_file_workspace_signature(signature: Any) -> None:
    for field_name, field_info in signature.input_fields.items():
        field = FieldDescriptor(field_name, field_info.annotation)
        for value_type in (File, Workspace):
            if _annotation_contains(field.annotation, value_type) and not field.matches(
                value_type
            ):
                raise ValueError(
                    f"Unsupported {value_type.__name__} annotation for input "
                    f"{field_name!r}: {field.annotation!r}"
                )
    for field_name, field_info in signature.output_fields.items():
        field = FieldDescriptor(field_name, field_info.annotation)
        if _annotation_contains(field.annotation, Workspace):
            raise ValueError(
                f"Workspace is input-only and cannot annotate output {field_name!r}"
            )
        if _annotation_contains(field.annotation, File) and not field.matches(File):
            raise ValueError(
                f"Unsupported File annotation for output {field_name!r}: "
                f"{field.annotation!r}"
            )


def _output_destination(artifact: Artifact, sandbox_path: str) -> Path:
    destination = Path(str(artifact.metadata["destination_path"]))
    source = PurePosixPath(sandbox_path)
    sandbox_root = PurePosixPath(str(artifact.metadata["sandbox_path"]))
    if not source.is_absolute() or ".." in source.parts:
        raise ValueError(
            f"Submitted output path {sandbox_path!r} is outside the reserved output root "
            f"{str(sandbox_root)!r}"
        )
    try:
        relative = source.relative_to(sandbox_root)
    except ValueError as exc:
        raise ValueError(
            f"Submitted output path {sandbox_path!r} is outside the reserved output root "
            f"{str(sandbox_root)!r}"
        ) from exc
    if not relative.parts:
        raise ValueError(
            f"Submitted output path must identify a file under the reserved output root "
            f"{str(sandbox_root)!r}"
        )
    return destination.joinpath(*relative.parts)


class ValueInputAdapter(InputAdapter[Any]):
    """Compatibility owner for ordinary model-visible input values."""

    name = "value"
    value_type = object
    fallback = True

    def supports(self, field: FieldDescriptor, value: Any) -> bool:
        return True

    async def prepare(
        self,
        field: FieldDescriptor,
        value: Any,
        ctx: RunContext,
    ) -> PreparedInput:
        return PreparedInput(model_value=value)


class FileInputAdapter(InputAdapter[File]):
    """Prepare current File values as opaque file artifacts."""

    name = "file"
    value_type = File

    async def prepare(
        self,
        field: FieldDescriptor,
        value: File | list[File] | None,
        ctx: RunContext,
    ) -> PreparedInput:
        if value is None:
            if field.allows_none:
                return PreparedInput(model_value=None)
            raise ValueError(f"File input {field.name!r} cannot be None")

        artifacts: list[Artifact] = []
        model_paths: list[str | None] = []
        for item in field.unpack(value):
            if item is None and field.item_allows_none:
                model_paths.append(None)
                continue
            if not isinstance(item, File) or not item.path:
                raise TypeError("File inputs must contain File values with a path")
            if not os.path.isfile(item.path):
                raise FileNotFoundError(item.path)
            ctx.state.setdefault("extra_read_paths", []).append(os.path.abspath(item.path))
            sandbox_path = f"/sandbox/input/{field.name}/{os.path.basename(item.path)}"
            artifacts.append(
                Artifact(
                    id=f"file-input-{uuid.uuid4().hex}",
                    kind="compat.file",
                    metadata={
                        "source_path": item.path,
                        "sandbox_path": sandbox_path,
                    },
                )
            )
            model_paths.append(sandbox_path)

        model_value = field.pack(model_paths)
        return PreparedInput(
            model_value=model_value,
            artifacts=tuple(artifacts),
            instructions=(
                f"Input file field `{field.name}` is mounted at {model_value!r}.",
            ),
        )


class WorkspaceInputAdapter(InputAdapter[Workspace]):
    """Prepare input-only Workspace directories without durable-provider knowledge."""

    name = "workspace"
    value_type = Workspace

    async def prepare(
        self,
        field: FieldDescriptor,
        value: Workspace | list[Workspace] | None,
        ctx: RunContext,
    ) -> PreparedInput:
        if value is None:
            if field.allows_none:
                return PreparedInput(model_value=None)
            raise ValueError(f"Workspace input {field.name!r} cannot be None")

        artifacts: list[Artifact] = []
        model_paths: list[str | None] = []
        mount_owners = ctx.state.setdefault("workspace_mount_owners", {})
        for workspace in field.unpack(value):
            if workspace is None and field.item_allows_none:
                model_paths.append(None)
                continue
            if not isinstance(workspace, Workspace):
                raise TypeError("Workspace inputs must contain Workspace values")
            if not os.path.isdir(workspace.path):
                raise FileNotFoundError(workspace.path)
            workspace_root = os.path.abspath(workspace.path)
            ctx.state.setdefault("extra_read_paths", []).append(workspace_root)
            ctx.state.setdefault("extra_write_paths", []).append(workspace_root)
            if workspace.mode is WorkspaceMode.DIRECT:
                sandbox_path = _direct_workspace_sandbox_path(workspace, workspace_root)
                kind = "compat.workspace.direct"
                workspace_binding: Any = DirectWorkspaceMount(
                    host_path=workspace_root,
                    sandbox_path=sandbox_path,
                )
            else:
                sandbox_path = workspace.mount_path
                kind = "compat.workspace.mirror"
                workspace_binding = WorkspaceSyncState(workspace)
            owner = mount_owners.get(sandbox_path)
            if owner is not None:
                raise ValueError(
                    "Duplicate Workspace.mount_path values are not supported: "
                    f"{sandbox_path!r} appears in both {owner!r} and {field.name!r}"
                )
            mount_owners[sandbox_path] = field.name
            artifacts.append(
                Artifact(
                    id=f"workspace-input-{uuid.uuid4().hex}",
                    kind=kind,
                    metadata={
                        "source_path": workspace_root,
                        "sandbox_path": sandbox_path,
                        "workspace": workspace,
                        "workspace_binding": workspace_binding,
                    },
                )
            )
            model_paths.append(sandbox_path)

        model_value = field.pack(model_paths)
        return PreparedInput(
            model_value=model_value,
            artifacts=tuple(artifacts),
            instructions=(
                f"Workspace field `{field.name}` is mounted at {model_value!r}. Edit files in "
                "that directory; mirror-mode changes sync back after every code block.",
            ),
        )


class FileOutputAdapter(OutputAdapter[File]):
    """Reserve and materialize current File/list[File] outputs."""

    name = "file"
    value_type = File

    def __init__(self, output_dir: str | None = None) -> None:
        self.output_dir = output_dir

    async def prepare_session(
        self,
        field: FieldDescriptor,
        value: File | list[File] | None,
        ctx: RunContext,
    ) -> None:
        import tempfile

        host_base = self.output_dir or tempfile.mkdtemp(prefix="predict-rlm-")
        host_dir = value.path if isinstance(value, File) and value.path else os.path.join(
            host_base, field.name
        )
        Path(host_dir).mkdir(parents=True, exist_ok=True)
        ctx.state.setdefault("output_host_dirs", {})[field.name] = host_dir
        ctx.state.setdefault("extra_write_paths", []).append(host_dir)

    async def reserve(
        self,
        field: FieldDescriptor,
        value: File | list[File] | None,
        ctx: RunContext,
        session: ExecutionSession,
    ) -> OutputReservation:
        host_dir = ctx.state["output_host_dirs"][field.name]
        sandbox_dir = f"/sandbox/output/{field.name}"
        artifact = Artifact(
            id=f"file-output-{uuid.uuid4().hex}",
            kind="compat.output.directory",
            metadata={
                "sandbox_path": sandbox_dir,
                "destination_path": host_dir,
                "directory": True,
                "source_path": host_dir,
            },
        )
        Path(host_dir).mkdir(parents=True, exist_ok=True)
        binding = await session.mount(artifact)
        ctx.bind(binding)
        artifact = Artifact(
            id=artifact.id,
            kind=artifact.kind,
            metadata={
                **artifact.metadata,
                "sandbox_path": binding.path,
            },
        )
        model_value = f"{binding.path.rstrip('/')}/"
        return OutputReservation(
            field=field,
            artifact=artifact,
            model_value=model_value,
        )

    async def materialize(
        self,
        reservation: OutputReservation,
        submitted_value: Any,
        ctx: RunContext,
        session: ExecutionSession,
    ) -> Any:
        del ctx
        artifact = reservation.artifact
        if reservation.field.is_list:
            for item in submitted_value if isinstance(submitted_value, list) else ():
                if item is None and reservation.field.item_allows_none:
                    continue
                submitted_path = getattr(item, "path", item)
                if not isinstance(submitted_path, str):
                    raise TypeError(
                        f"File output {reservation.field.name!r} must contain file paths"
                    )
                if PurePosixPath(submitted_path).is_absolute():
                    _output_destination(artifact, submitted_path)
            return await self._collect_reserved_directory(artifact, session)

        submitted_path = getattr(submitted_value, "path", submitted_value)
        if isinstance(submitted_path, str) and submitted_path.strip():
            submitted_path = submitted_path.strip()
            if PurePosixPath(submitted_path).is_absolute():
                destination = _output_destination(artifact, submitted_path)
                selected = Artifact(
                    id=artifact.id,
                    kind="compat.file",
                    metadata={
                        "sandbox_path": submitted_path,
                        "destination_path": str(destination),
                    },
                )
                try:
                    return File(path=await session.collect(selected))
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                    pass

        generated = await self._collect_reserved_directory(artifact, session)
        if len(generated) == 1:
            return generated[0]
        if generated:
            return File(path=str(artifact.metadata["destination_path"]))
        return submitted_value

    async def _collect_reserved_directory(
        self,
        artifact: Artifact,
        session: ExecutionSession,
    ) -> list[File]:
        destination = Path(str(artifact.metadata["destination_path"]))
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{destination.name}-collect-",
            dir=destination.parent,
        ) as staging_dir:
            staging = Path(staging_dir)
            await session.collect(
                Artifact(
                    id=f"{artifact.id}-directory",
                    kind=artifact.kind,
                    metadata={
                        **artifact.metadata,
                        "destination_path": str(staging),
                        "directory": True,
                    },
                )
            )
            generated: list[File] = []
            for source in sorted(
                path for path in staging.rglob("*") if path.is_file() and not path.is_symlink()
            ):
                relative = source.relative_to(staging)
                target = destination.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
                generated.append(File(path=str(target)))
            return generated
