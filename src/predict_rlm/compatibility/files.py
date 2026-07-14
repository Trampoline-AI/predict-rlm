"""File and Workspace compatibility adapters built on opaque artifacts."""

from __future__ import annotations

import os
import tempfile
import typing
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from predict_rlm.files import File
from predict_rlm.runtime import (
    Artifact,
    ArtifactBinding,
    DirectoryCreationSession,
    ExecutionResult,
    ExecutionSession,
    FieldDescriptor,
    FileTransfer,
    FileTransferSession,
    HostDirectoryMount,
    HostDirectorySession,
    InputAdapter,
    MountedInput,
    MutableDirectorySession,
    OutputAdapter,
    OutputReservation,
    PreparedInput,
    RunContext,
    SandboxRootReservation,
    SessionRequirements,
)
from predict_rlm.workspace import Workspace, WorkspaceMode, WorkspaceSyncState


@dataclass
class _WorkspaceMountState:
    artifact: Artifact
    sandbox_path: str
    sync_state: WorkspaceSyncState | None
    host_directory_mount: HostDirectoryMount | None = None
    mounted: bool = False
    finalized: bool = False


@dataclass(frozen=True, kw_only=True)
class _WorkspacePreparedInput(PreparedInput):
    mount_states: tuple[_WorkspaceMountState, ...]


def _workspace_mount_states(
    prepared: PreparedInput,
) -> tuple[_WorkspaceMountState, ...]:
    if not isinstance(prepared, _WorkspacePreparedInput):
        raise TypeError("WorkspaceInputAdapter requires typed Workspace prepared state")
    return prepared.mount_states


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


def _replace_model_path(value: Any, planned: str, actual: str) -> Any:
    if value == planned:
        return actual
    if isinstance(value, list):
        return [_replace_model_path(item, planned, actual) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_model_path(item, planned, actual) for item in value)
    return value


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
        read_paths: list[str] = []
        model_paths: list[str | None] = []
        for item in field.unpack(value):
            if item is None and field.item_allows_none:
                model_paths.append(None)
                continue
            if not isinstance(item, File) or not item.path:
                raise TypeError("File inputs must contain File values with a path")
            if not os.path.isfile(item.path):
                raise FileNotFoundError(item.path)
            read_paths.append(os.path.abspath(item.path))
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
            sandbox_roots=(
                SandboxRootReservation(f"/sandbox/input/{field.name}"),
            ),
            requirements=SessionRequirements(extra_read_paths=tuple(read_paths)),
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
                return _WorkspacePreparedInput(
                    model_value=None,
                    mount_states=(),
                )
            raise ValueError(f"Workspace input {field.name!r} cannot be None")

        mount_states = []
        host_directory_mounts = []
        workspace_roots = []
        writable_workspace_roots = []
        sandbox_roots = []
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
            workspace_roots.append(workspace_root)
            if workspace.mode is WorkspaceMode.DIRECT:
                sandbox_path = _direct_workspace_sandbox_path(workspace, workspace_root)
                sync_state = None
                host_directory_mount = HostDirectoryMount(
                    host_path=workspace_root,
                    sandbox_path=sandbox_path,
                )
                host_directory_mounts.append(host_directory_mount)
                writable_workspace_roots.append(workspace_root)
            else:
                sandbox_path = workspace.mount_path
                sync_state = WorkspaceSyncState(workspace)
                host_directory_mount = None
                if workspace.sync_back:
                    writable_workspace_roots.append(workspace_root)
            owner = mount_owners.get(sandbox_path)
            if owner is not None:
                raise ValueError(
                    "Duplicate Workspace.mount_path values are not supported: "
                    f"{sandbox_path!r} appears in both {owner!r} and {field.name!r}"
                )
            mount_owners[sandbox_path] = field.name
            sandbox_roots.append(SandboxRootReservation(sandbox_path))
            artifact = Artifact(
                id=f"workspace-input-{uuid.uuid4().hex}",
                kind="compat.workspace",
                metadata={"sandbox_path": sandbox_path},
            )
            mount_states.append(
                _WorkspaceMountState(
                    artifact=artifact,
                    sandbox_path=sandbox_path,
                    sync_state=sync_state,
                    host_directory_mount=host_directory_mount,
                )
            )
            model_paths.append(sandbox_path)

        model_value = field.pack(model_paths)
        return _WorkspacePreparedInput(
            model_value=model_value,
            mount_states=tuple(mount_states),
            artifacts=tuple(state.artifact for state in mount_states),
            host_directory_mounts=tuple(host_directory_mounts),
            sandbox_roots=tuple(sandbox_roots),
            requirements=SessionRequirements(
                extra_read_paths=tuple(workspace_roots),
                extra_write_paths=tuple(writable_workspace_roots),
            ),
            instructions=(
                f"Workspace field `{field.name}` is mounted at {model_value!r}. Edit files in "
                "that directory; mirror changes sync back after every code block when "
                "sync_back is enabled.",
            ),
        )

    async def mount(
        self,
        field: FieldDescriptor,
        prepared: PreparedInput,
        ctx: RunContext,
        session: ExecutionSession,
    ) -> MountedInput:
        del field, ctx
        states = _workspace_mount_states(prepared)
        bindings = []
        model_value = prepared.model_value
        for state in states:
            if state.host_directory_mount is not None:
                if not isinstance(session, HostDirectorySession):
                    raise TypeError(
                        f"Execution session {session.name!r} does not support host "
                        "directory mounts"
                    )
                sandbox_path = await session.mount_host_directory(
                    state.host_directory_mount
                )
            else:
                if not isinstance(session, DirectoryCreationSession):
                    raise TypeError(
                        f"Execution session {session.name!r} does not support directory "
                        "creation"
                    )
                sandbox_path = state.sandbox_path
                await session.create_directory(state.sandbox_path)
                if not isinstance(session, FileTransferSession):
                    raise TypeError(
                        f"Execution session {session.name!r} does not support file transfer"
                    )
                sync_state = state.sync_state
                assert sync_state is not None
                for source_path, target_path in await sync_state.aiter_mounts():
                    await session.transfer_file(
                        FileTransfer(
                            source_path=source_path,
                            sandbox_path=target_path,
                        )
                    )
            state.mounted = True
            model_value = _replace_model_path(
                model_value,
                state.sandbox_path,
                sandbox_path,
            )
            bindings.append(
                ArtifactBinding(
                    artifact_id=state.artifact.id,
                    path=sandbox_path,
                )
            )
        return MountedInput(model_value=model_value, bindings=tuple(bindings))

    async def after_execution(
        self,
        field: FieldDescriptor,
        prepared: PreparedInput,
        ctx: RunContext,
        session: ExecutionSession,
        result: ExecutionResult | None,
        error: BaseException | None,
    ) -> None:
        del field, ctx, result, error
        states = _workspace_mount_states(prepared)
        mirror_states = [
            state
            for state in states
            if state.sync_state is not None and state.sync_state.workspace.sync_back
        ]
        if not mirror_states:
            return
        if not isinstance(session, MutableDirectorySession):
            raise TypeError(
                f"Execution session {session.name!r} does not support mutable directory inputs"
            )
        first_error: BaseException | None = None
        additional_errors = []
        for state in mirror_states:
            sync_state = state.sync_state
            assert sync_state is not None
            try:
                await sync_state.async_sync_from_sandbox(session)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                else:
                    additional_errors.append(exc)
        if first_error is not None:
            if additional_errors:
                setattr(first_error, "workspace_sync_errors", tuple(additional_errors))
            raise first_error

    async def finalize(
        self,
        field: FieldDescriptor,
        prepared: PreparedInput,
        ctx: RunContext,
        session: ExecutionSession | None,
        error: BaseException | None,
    ) -> None:
        del field, ctx, error
        states = _workspace_mount_states(prepared)
        first_error: BaseException | None = None
        additional_errors = []
        for state in states:
            if state.finalized:
                continue
            try:
                if (
                    state.sync_state is not None
                    and state.sync_state.workspace.sync_back
                    and state.mounted
                ):
                    if not isinstance(session, MutableDirectorySession):
                        raise TypeError(
                            "Execution session does not support mutable "
                            "directory inputs"
                        )
                    await state.sync_state.async_sync_from_sandbox(session)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                else:
                    additional_errors.append(exc)
            finally:
                state.finalized = True
        if first_error is not None:
            if additional_errors:
                setattr(first_error, "workspace_finalize_errors", tuple(additional_errors))
            raise first_error


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
    ) -> SessionRequirements:
        import tempfile

        host_base = self.output_dir or tempfile.mkdtemp(prefix="predict-rlm-")
        host_dir = value.path if isinstance(value, File) and value.path else os.path.join(
            host_base, field.name
        )
        Path(host_dir).mkdir(parents=True, exist_ok=True)
        ctx.state.setdefault("output_host_dirs", {})[field.name] = host_dir
        return SessionRequirements(extra_write_paths=(host_dir,))

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
