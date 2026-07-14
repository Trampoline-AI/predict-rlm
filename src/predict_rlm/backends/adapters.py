"""Adapters from existing interpreter objects to the async session contract."""

from __future__ import annotations

import asyncio
import inspect
import os
import threading
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractContextManager, asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from predict_rlm.runtime import (
    Artifact,
    ArtifactBinding,
    ArtifactFileInfo,
    ExecutionResult,
    ExecutionSession,
    ExecutionSpec,
    FileTransfer,
    HostDirectoryMount,
    RunContext,
    SandboxRootReservation,
    SessionOwnership,
    UnsupportedOperationError,
)


def _artifact_path(artifact: Artifact, name: str) -> str:
    value = artifact.metadata.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Artifact {artifact.id!r} requires string metadata {name!r}")
    return value


def _validate_sandbox_path(path: str) -> None:
    parsed = PurePosixPath(path)
    if not parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError(f"Artifact sandbox path must be absolute and traversal-free: {path!r}")


def _require_reserved_sandbox_path(
    path: str,
    reservations: Sequence[SandboxRootReservation],
) -> None:
    _validate_sandbox_path(path)
    normalized = str(PurePosixPath(path))
    if not any(
        normalized == reservation.path
        or normalized.startswith(reservation.path.rstrip("/") + "/")
        for reservation in reservations
    ):
        raise UnsupportedOperationError(
            f"Sandbox transfer path {path!r} is outside the input adapter's "
            "declared sandbox roots"
        )


_UNSET_MOUNT_SET = object()


class InterpreterExecutionSession:
    """Compatibility session over a caller-supplied synchronous interpreter."""

    def __init__(
        self,
        interpreter: Any,
        *,
        name: str,
        ownership: SessionOwnership,
        host_directory_mounts: Sequence[HostDirectoryMount] = (),
        sandbox_roots: Sequence[SandboxRootReservation] = (),
    ) -> None:
        self.interpreter = interpreter
        self.name = name
        self.ownership = ownership
        self._finalized = False
        self._cancelled = False
        self._sync_tasks: set[asyncio.Task[Any]] = set()
        self._host_directory_mounts = list(host_directory_mounts)
        self._sandbox_roots = tuple(sandbox_roots)

    async def _run_sync(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        self._sync_tasks.add(task)
        task.add_done_callback(self._sync_tasks.discard)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError as cancellation:
            try:
                await task
            except BaseException as worker_error:
                setattr(cancellation, "sync_worker_error", worker_error)
            raise

    async def wait_for_idle(self) -> None:
        while self._sync_tasks:
            await asyncio.gather(*tuple(self._sync_tasks), return_exceptions=True)
        await_host_work = getattr(self.interpreter, "await_host_work", None)
        if callable(await_host_work):
            result = await_host_work()
            if inspect.isawaitable(result):
                await result

    async def install_packages(self, packages: Sequence[str]) -> None:
        if not packages:
            return
        install = getattr(self.interpreter, "ensure_skill_packages", None)
        if callable(install):
            await self._run_sync(install, list(packages))

    async def mount(self, artifact: Artifact) -> ArtifactBinding:
        source_path = _artifact_path(artifact, "source_path")
        sandbox_path = _artifact_path(artifact, "sandbox_path")
        _validate_sandbox_path(sandbox_path)

        if artifact.kind == "compat.output.directory":
            await self._run_sync(self.interpreter.mkdir_p, sandbox_path)
            directory = True
        elif os.path.isdir(source_path):
            await self._mount_directory(source_path, sandbox_path)
            directory = True
        elif os.path.isfile(source_path):
            await self._run_sync(
                self.interpreter.mount_file_at,
                source_path,
                sandbox_path,
            )
            directory = False
        else:
            raise FileNotFoundError(f"Artifact source does not exist: {source_path}")

        return ArtifactBinding(
            artifact_id=artifact.id,
            path=sandbox_path,
            metadata={"directory": directory},
        )

    async def run_code(
        self,
        code: str,
        variables: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> ExecutionResult:
        kwargs: dict[str, Any] = {"variables": dict(variables or {})}
        if timeout is not None:
            kwargs["timeout"] = timeout
        execute_async = getattr(self.interpreter, "aexecute", None)
        uses_websocket = getattr(self.interpreter, "_uses_websocket_transport", None)
        prefer_sync = callable(uses_websocket) and not uses_websocket()
        if callable(execute_async) and not prefer_sync:
            value = await execute_async(code, **kwargs)
        else:
            value = await self._run_sync(self.interpreter.execute, code, **kwargs)
        return ExecutionResult(value=value)

    async def collect(self, artifact: Artifact) -> Any:
        sandbox_path = _artifact_path(artifact, "sandbox_path")
        destination_path = _artifact_path(artifact, "destination_path")
        _validate_sandbox_path(sandbox_path)
        if bool(artifact.metadata.get("directory")):
            return await self._collect_directory(sandbox_path, destination_path)
        destination = Path(destination_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        await self._run_sync(
            self.interpreter.sync_file_to,
            sandbox_path,
            str(destination),
        )
        return str(destination)

    async def configure_host_directory_mounts(
        self,
        mounts: Sequence[HostDirectoryMount],
    ) -> None:
        if not mounts:
            return
        configure = getattr(self.interpreter, "configure_direct_workspace_mounts", None)
        if not callable(configure):
            raise UnsupportedOperationError(
                f"Execution session {self.name!r} does not support host directory mounts"
            )
        await self._run_sync(configure, list(mounts))
        self._host_directory_mounts = list(mounts)

    async def mount_host_directory(self, mount: HostDirectoryMount) -> str:
        if mount not in self._host_directory_mounts:
            raise UnsupportedOperationError(
                "Host directory mounts must be declared before backend acquisition"
            )
        return mount.sandbox_path

    async def transfer_file(self, transfer: FileTransfer) -> str:
        _require_reserved_sandbox_path(transfer.sandbox_path, self._sandbox_roots)
        if not os.path.isfile(transfer.source_path):
            raise FileNotFoundError(transfer.source_path)
        await self._run_sync(
            self.interpreter.mount_file_at,
            transfer.source_path,
            transfer.sandbox_path,
        )
        return transfer.sandbox_path

    async def create_directory(self, sandbox_path: str) -> None:
        _require_reserved_sandbox_path(sandbox_path, self._sandbox_roots)
        await self._run_sync(self.interpreter.mkdir_p, sandbox_path)

    async def inspect_directory(
        self,
        sandbox_path: str,
    ) -> Mapping[str, ArtifactFileInfo]:
        _require_reserved_sandbox_path(sandbox_path, self._sandbox_roots)
        inspect_directory = getattr(self.interpreter, "workspace_manifest", None)
        if not callable(inspect_directory):
            raise UnsupportedOperationError(
                f"Execution session {self.name!r} cannot inspect directories"
            )
        return await self._run_sync(inspect_directory, sandbox_path)

    async def collect_file(self, sandbox_path: str, host_path: str) -> None:
        _require_reserved_sandbox_path(sandbox_path, self._sandbox_roots)
        destination = Path(host_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        await self._run_sync(
            self.interpreter.sync_file_to,
            sandbox_path,
            str(destination),
        )

    async def finalize(self) -> None:
        self._finalized = True

    async def cancel(self) -> None:
        if self._cancelled:
            return
        self._cancelled = True
        interrupt_async = getattr(self.interpreter, "ainterrupt", None)
        if callable(interrupt_async):
            await interrupt_async()
            return
        interrupt = getattr(self.interpreter, "interrupt", None)
        if callable(interrupt):
            await self._run_sync(interrupt)

    async def _mount_directory(self, source_path: str, sandbox_path: str) -> None:
        await self._run_sync(self.interpreter.mkdir_p, sandbox_path)
        source_root = Path(source_path)
        for host_path in sorted(path for path in source_root.rglob("*") if path.is_file()):
            relative = host_path.relative_to(source_root).as_posix()
            target = str(PurePosixPath(sandbox_path) / relative)
            _validate_sandbox_path(target)
            await self._run_sync(
                self.interpreter.mount_file_at,
                str(host_path),
                target,
            )

    async def _collect_directory(self, sandbox_path: str, destination_path: str) -> str:
        destination = Path(destination_path)
        destination.mkdir(parents=True, exist_ok=True)
        files = await self._run_sync(self.interpreter.list_dir, sandbox_path)
        for source in sorted(files):
            relative = PurePosixPath(source).relative_to(PurePosixPath(sandbox_path))
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            await self._run_sync(self.interpreter.sync_file_to, source, str(target))
        return str(destination)


AcquireInterpreter = Callable[
    [ExecutionSpec, RunContext], AbstractContextManager[Any]
]


class InterpreterBackendAdapter:
    """Thread-bridged boundary for a caller-supplied synchronous interpreter."""

    def __init__(
        self,
        name: str,
        acquire: AcquireInterpreter,
        *,
        ownership: SessionOwnership,
        supports_host_directory_mounts: bool = False,
    ) -> None:
        self.name = name
        self._acquire = acquire
        self._ownership = ownership
        self._supports_host_directory_mounts = supports_host_directory_mounts
        self._invocation_lock = (
            threading.Lock() if ownership is SessionOwnership.INJECTED else None
        )
        self._mount_set_key: object = _UNSET_MOUNT_SET

    async def validate_host_directory_mounts(
        self,
        mounts: Sequence[HostDirectoryMount],
        ctx: RunContext,
    ) -> None:
        del ctx
        if self._ownership is SessionOwnership.POOLED:
            if mounts:
                raise UnsupportedOperationError(
                    "Host directory mounts require a per-call interpreter; "
                    "pooled interpreters cannot add mounts after creation"
                )
            return
        if mounts and not self._supports_host_directory_mounts:
            raise UnsupportedOperationError(
                f"Execution backend {self.name!r} does not support host directory mounts"
            )

    @asynccontextmanager
    async def start(
        self,
        spec: ExecutionSpec,
        ctx: RunContext,
    ) -> AsyncIterator[ExecutionSession]:
        lock_acquired = False
        manager: AbstractContextManager[Any] | None = None
        entered = False
        try:
            await self.validate_host_directory_mounts(spec.host_directory_mounts, ctx)
            if self._invocation_lock is not None:
                await self._acquire_invocation_lock()
                lock_acquired = True
            mount_key = tuple(
                sorted(
                    spec.host_directory_mounts,
                    key=lambda mount: (
                        mount.host_path,
                        mount.sandbox_path,
                        mount.read_only,
                    ),
                )
            )
            if self._ownership is SessionOwnership.INJECTED:
                self._validate_mount_key(mount_key)
            manager = self._acquire(spec, ctx)
            enter_task = asyncio.create_task(asyncio.to_thread(manager.__enter__))
            try:
                interpreter = await asyncio.shield(enter_task)
            except asyncio.CancelledError:
                await enter_task
                entered = True
                entered = False
                await self._exit_manager(manager, None, None, None)
                raise
            entered = True
            session = InterpreterExecutionSession(
                interpreter,
                name=self.name,
                ownership=self._ownership,
                host_directory_mounts=spec.host_directory_mounts,
                sandbox_roots=spec.sandbox_roots,
            )
            await session.configure_host_directory_mounts(spec.host_directory_mounts)
            if (
                self._ownership is SessionOwnership.INJECTED
                and self._mount_set_key is _UNSET_MOUNT_SET
            ):
                self._mount_set_key = mount_key
            ctx.session = session
            ctx.ownership = self._ownership
            try:
                yield session
            except BaseException as exc:
                await session.wait_for_idle()
                entered = False
                suppress = await self._exit_manager(
                    manager,
                    type(exc),
                    exc,
                    exc.__traceback__,
                )
                if not suppress:
                    raise
            else:
                await session.wait_for_idle()
                entered = False
                await self._exit_manager(manager, None, None, None)
        finally:
            try:
                if entered and manager is not None:
                    entered = False
                    await self._exit_manager(manager, None, None, None)
            finally:
                ctx.session = None
                if lock_acquired and self._invocation_lock is not None:
                    self._invocation_lock.release()

    def _validate_mount_key(
        self,
        mount_key: tuple[HostDirectoryMount, ...],
    ) -> None:
        if self._mount_set_key is not _UNSET_MOUNT_SET and self._mount_set_key != mount_key:
            raise ValueError(
                "Injected interpreters cannot change a host-directory mount set "
                "across sequential invocations"
            )

    async def _acquire_invocation_lock(self) -> None:
        if self._invocation_lock is None:
            return
        while True:
            acquire_task = asyncio.create_task(
                asyncio.to_thread(self._invocation_lock.acquire, True, 0.05)
            )
            try:
                acquired = await asyncio.shield(acquire_task)
            except asyncio.CancelledError:
                if await acquire_task:
                    self._invocation_lock.release()
                raise
            if acquired:
                return

    async def _exit_manager(
        self,
        manager: AbstractContextManager[Any],
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> Any:
        exit_task = asyncio.create_task(
            asyncio.to_thread(manager.__exit__, exc_type, exc, traceback)
        )
        try:
            return await asyncio.shield(exit_task)
        except asyncio.CancelledError:
            await exit_task
            raise


class NativeInterpreterExecutionSession:
    """Session over a maintained interpreter's native async operations."""

    def __init__(
        self,
        interpreter: Any,
        *,
        name: str,
        ownership: SessionOwnership,
        host_directory_mounts: Sequence[HostDirectoryMount] = (),
        sandbox_roots: Sequence[SandboxRootReservation] = (),
    ) -> None:
        self.interpreter = interpreter
        self.name = name
        self.ownership = ownership
        self._finalized = False
        self._cancelled = False
        self._host_directory_mounts = list(host_directory_mounts)
        self._sandbox_roots = tuple(sandbox_roots)

    async def install_packages(self, packages: Sequence[str]) -> None:
        if packages:
            await self.interpreter.aensure_skill_packages(list(packages))

    async def mount(self, artifact: Artifact) -> ArtifactBinding:
        source_path = _artifact_path(artifact, "source_path")
        sandbox_path = _artifact_path(artifact, "sandbox_path")
        _validate_sandbox_path(sandbox_path)

        if artifact.kind == "compat.output.directory":
            await self.interpreter.amkdir_p(sandbox_path)
            directory = True
        elif os.path.isdir(source_path):
            await self._mount_directory(source_path, sandbox_path)
            directory = True
        elif os.path.isfile(source_path):
            await self.interpreter.amount_file_at(source_path, sandbox_path)
            directory = False
        else:
            raise FileNotFoundError(f"Artifact source does not exist: {source_path}")

        return ArtifactBinding(
            artifact_id=artifact.id,
            path=sandbox_path,
            metadata={"directory": directory},
        )

    async def run_code(
        self,
        code: str,
        variables: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> ExecutionResult:
        value = await self.interpreter.aexecute(
            code,
            variables=dict(variables or {}),
            timeout=timeout,
        )
        return ExecutionResult(value=value)

    async def collect(self, artifact: Artifact) -> Any:
        sandbox_path = _artifact_path(artifact, "sandbox_path")
        destination_path = _artifact_path(artifact, "destination_path")
        _validate_sandbox_path(sandbox_path)
        if bool(artifact.metadata.get("directory")):
            return await self._collect_directory(sandbox_path, destination_path)
        destination = Path(destination_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        await self.interpreter.async_file_to(sandbox_path, str(destination))
        return str(destination)

    async def configure_host_directory_mounts(
        self,
        mounts: Sequence[HostDirectoryMount],
    ) -> None:
        if not mounts:
            return
        configure = getattr(self.interpreter, "aconfigure_direct_workspace_mounts", None)
        if not callable(configure):
            raise UnsupportedOperationError(
                f"Execution session {self.name!r} does not support host directory mounts"
            )
        await configure(list(mounts))
        self._host_directory_mounts = list(mounts)

    async def mount_host_directory(self, mount: HostDirectoryMount) -> str:
        if mount not in self._host_directory_mounts:
            raise UnsupportedOperationError(
                "Host directory mounts must be declared before backend acquisition"
            )
        return mount.sandbox_path

    async def transfer_file(self, transfer: FileTransfer) -> str:
        _require_reserved_sandbox_path(transfer.sandbox_path, self._sandbox_roots)
        if not os.path.isfile(transfer.source_path):
            raise FileNotFoundError(transfer.source_path)
        await self.interpreter.amount_file_at(
            transfer.source_path,
            transfer.sandbox_path,
        )
        return transfer.sandbox_path

    async def create_directory(self, sandbox_path: str) -> None:
        _require_reserved_sandbox_path(sandbox_path, self._sandbox_roots)
        await self.interpreter.amkdir_p(sandbox_path)

    async def inspect_directory(
        self,
        sandbox_path: str,
    ) -> Mapping[str, ArtifactFileInfo]:
        _require_reserved_sandbox_path(sandbox_path, self._sandbox_roots)
        inspect_directory = getattr(self.interpreter, "aworkspace_manifest", None)
        if not callable(inspect_directory):
            raise UnsupportedOperationError(
                f"Execution session {self.name!r} cannot inspect directories"
            )
        return await inspect_directory(sandbox_path)

    async def collect_file(self, sandbox_path: str, host_path: str) -> None:
        _require_reserved_sandbox_path(sandbox_path, self._sandbox_roots)
        destination = Path(host_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        await self.interpreter.async_file_to(sandbox_path, str(destination))

    async def finalize(self) -> None:
        await_host_work = getattr(self.interpreter, "await_host_work", None)
        if callable(await_host_work):
            result = await_host_work(cancel=True)
            if inspect.isawaitable(result):
                await result
        self._finalized = True

    async def cancel(self) -> None:
        if self._cancelled:
            return
        self._cancelled = True
        await self.interpreter.ainterrupt()

    async def _mount_directory(self, source_path: str, sandbox_path: str) -> None:
        await self.interpreter.amkdir_p(sandbox_path)
        source_root = Path(source_path)
        for host_path in sorted(path for path in source_root.rglob("*") if path.is_file()):
            await asyncio.sleep(0)
            relative = host_path.relative_to(source_root).as_posix()
            target = str(PurePosixPath(sandbox_path) / relative)
            _validate_sandbox_path(target)
            await self.interpreter.amount_file_at(str(host_path), target)

    async def _collect_directory(self, sandbox_path: str, destination_path: str) -> str:
        destination = Path(destination_path)
        destination.mkdir(parents=True, exist_ok=True)
        files = await self.interpreter.alist_dir(sandbox_path)
        for source in sorted(files):
            await asyncio.sleep(0)
            relative = PurePosixPath(source).relative_to(PurePosixPath(sandbox_path))
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            await self.interpreter.async_file_to(source, str(target))
        return str(destination)


class ExistingExecutionBackendAdapter:
    """Validate and expose a user-supplied final execution backend."""

    def __init__(self, backend: Any) -> None:
        if not callable(getattr(backend, "start", None)):
            raise TypeError("execution must implement start(spec, ctx)")
        self.backend = backend
        self.name = str(getattr(backend, "name", type(backend).__name__))

    def start(self, spec: ExecutionSpec, ctx: RunContext) -> Any:
        return self.backend.start(spec, ctx)

    async def validate_host_directory_mounts(
        self,
        mounts: Sequence[HostDirectoryMount],
        ctx: RunContext,
    ) -> None:
        validate = getattr(self.backend, "validate_host_directory_mounts", None)
        if not callable(validate):
            if mounts:
                raise UnsupportedOperationError(
                    f"Execution backend {self.name!r} does not support host directory mounts"
                )
            return
        result = validate(mounts, ctx)
        if inspect.isawaitable(result):
            await result
