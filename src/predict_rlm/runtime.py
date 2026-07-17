"""Small-kernel runtime contracts for PredictRLM."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import fnmatch
import inspect
import os
import posixpath
import threading
import types
import typing
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Generic, Literal, Protocol, TypeVar, runtime_checkable

import dspy
from pydantic import BaseModel, ConfigDict, field_validator

AdapterValue = TypeVar("AdapterValue")


class ExecutionFatalError(RuntimeError):
    """The execution session is no longer safe to continue."""


class UnsupportedOperationError(RuntimeError):
    """The selected execution transport cannot perform an optional operation."""


class SyncWorker:
    """Daemon-thread work whose completion can outlive an asyncio deadline."""

    def __init__(
        self,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        context = contextvars.copy_context()

        def run() -> None:
            try:
                result = context.run(function, *args, **kwargs)
            except BaseException as exc:
                self.future.set_exception(exc)
            else:
                self.future.set_result(result)

        self.thread = threading.Thread(
            target=run,
            name="predict-rlm-host-tool",
            daemon=True,
        )
        self.thread.start()

    @property
    def done(self) -> bool:
        return self.future.done()

    async def wait(self) -> Any:
        return await asyncio.shield(asyncio.wrap_future(self.future))

    def add_done_callback(self, callback: Callable[[SyncWorker], Any]) -> None:
        self.future.add_done_callback(lambda _future: callback(self))


class SyncWorkerTracker:
    """Own sync workers until completion and retire state without reusing it."""

    def _sync_worker_lock(self) -> threading.Lock:
        lock = getattr(self, "_sync_workers_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._sync_workers_lock = lock
        return lock

    def _sync_worker_set(self) -> set[SyncWorker]:
        workers = getattr(self, "_sync_workers", None)
        if workers is None:
            workers = set()
            self._sync_workers = workers
        return workers

    def _live_sync_workers(self) -> tuple[SyncWorker, ...]:
        with self._sync_worker_lock():
            return tuple(
                worker for worker in self._sync_worker_set() if not worker.done
            )

    def _start_sync_worker(
        self,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> SyncWorker:
        worker = SyncWorker(function, *args, **kwargs)
        with self._sync_worker_lock():
            self._sync_worker_set().add(worker)

        def discard(completed: SyncWorker) -> None:
            with self._sync_worker_lock():
                self._sync_worker_set().discard(completed)

        worker.add_done_callback(discard)
        return worker

    def has_live_sync_workers(self) -> bool:
        return bool(self._live_sync_workers())

    def defer_until_sync_workers_finish(
        self,
        callback: Callable[[], Any],
        *additional_workers: SyncWorker | None,
    ) -> bool:
        workers = {
            worker
            for worker in (*self._live_sync_workers(), *additional_workers)
            if worker is not None and not worker.done
        }
        if not workers:
            return False

        def finish_if_idle(_worker: SyncWorker) -> None:
            if all(worker.done for worker in workers):
                callback()

        for worker in workers:
            worker.add_done_callback(finish_if_idle)
        return True

    def retire_when_sync_workers_finish(self) -> bool:
        workers = self._live_sync_workers()
        if not workers:
            return False

        def retire() -> None:
            for worker in workers:
                try:
                    worker.future.result()
                except BaseException:
                    pass
            self.shutdown()

        threading.Thread(
            target=retire,
            name="predict-rlm-retired-backend",
            daemon=True,
        ).start()
        return True


@dataclass(frozen=True)
class HostSyncWorkerPolicy:
    owner: SyncWorkerTracker
    timeout: float | None = None
    detach_on_cancel: bool = False


_HOST_SYNC_WORKER_POLICY: ContextVar[HostSyncWorkerPolicy | None] = ContextVar(
    "predict_rlm_host_sync_worker_policy",
    default=None,
)


@contextmanager
def host_sync_worker_policy(
    owner: SyncWorkerTracker,
    *,
    timeout: float | None = None,
    detach_on_cancel: bool = False,
):
    token = _HOST_SYNC_WORKER_POLICY.set(
        HostSyncWorkerPolicy(
            owner=owner,
            timeout=timeout,
            detach_on_cancel=detach_on_cancel,
        )
    )
    try:
        yield
    finally:
        _HOST_SYNC_WORKER_POLICY.reset(token)


def callable_has_sync_leaf(function: Callable[..., Any]) -> bool:
    classified = getattr(function, "__predict_rlm_sync_leaf__", None)
    if isinstance(classified, bool):
        return classified
    return not inspect.iscoroutinefunction(inspect.unwrap(function))


def preserve_sync_leaf(
    wrapper: Callable[..., Any],
    wrapped: Callable[..., Any],
) -> None:
    wrapper.__predict_rlm_sync_leaf__ = callable_has_sync_leaf(wrapped)


def immutable_mapping(value: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    """Return a shallow immutable snapshot of a contract mapping."""
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True)
class Artifact:
    """Opaque host artifact passed from an adapter to an execution session."""

    id: str
    kind: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


@dataclass(frozen=True)
class ArtifactBinding:
    """Invocation-local, sandbox-visible binding for an opaque artifact."""

    artifact_id: str
    path: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


@dataclass(frozen=True)
class HostDirectoryMount:
    """Host directory exposed directly at a sandbox path for one invocation."""

    host_path: str
    sandbox_path: str
    read_only: bool = False


@dataclass(frozen=True)
class FileTransfer:
    """One host file copied into an invocation's sandbox filesystem."""

    source_path: str
    sandbox_path: str


@dataclass(frozen=True)
class ArtifactFileInfo:
    """Portable manifest entry returned by directory inspection."""

    type: str
    sha256: str
    size: int


@dataclass(frozen=True)
class SessionRequirements:
    """Immutable execution requirements contributed by an adapter."""

    allowed_domains: tuple[str, ...] = ()
    extra_read_paths: tuple[str, ...] = ()
    extra_write_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_domains", tuple(self.allowed_domains))
        object.__setattr__(self, "extra_read_paths", tuple(self.extra_read_paths))
        object.__setattr__(self, "extra_write_paths", tuple(self.extra_write_paths))


@dataclass(frozen=True)
class SandboxRootReservation:
    """Exclusive sandbox directory owned by one prepared input."""

    path: str

    def __post_init__(self) -> None:
        if not self.path.startswith("/") or ".." in self.path.split("/"):
            raise ValueError(
                "Sandbox root reservations must be absolute and traversal-free: "
                f"{self.path!r}"
            )
        path = posixpath.normpath(self.path)
        if path == "/":
            raise ValueError("Sandbox root reservations cannot reserve the filesystem root")
        object.__setattr__(self, "path", path)


def merge_session_requirements(
    requirements: Sequence[SessionRequirements],
) -> SessionRequirements:
    return SessionRequirements(
        allowed_domains=tuple(
            dict.fromkeys(
                domain for item in requirements for domain in item.allowed_domains
            )
        ),
        extra_read_paths=tuple(
            dict.fromkeys(
                path for item in requirements for path in item.extra_read_paths
            )
        ),
        extra_write_paths=tuple(
            dict.fromkeys(
                path for item in requirements for path in item.extra_write_paths
            )
        ),
    )


def normalize_host_directory_mounts(
    mounts: Sequence[HostDirectoryMount],
) -> tuple[HostDirectoryMount, ...]:
    for mount in mounts:
        if not mount.sandbox_path.startswith("/") or ".." in mount.sandbox_path.split("/"):
            raise ValueError(
                "Host directory mount paths must be absolute and traversal-free: "
                f"{mount.sandbox_path!r}"
            )
    normalized = tuple(
        HostDirectoryMount(
            host_path=os.path.abspath(mount.host_path),
            sandbox_path=posixpath.normpath(mount.sandbox_path),
            read_only=mount.read_only,
        )
        for mount in mounts
    )
    destinations: dict[str, HostDirectoryMount] = {}
    host_access: dict[str, bool] = {}
    for mount in normalized:
        previous = destinations.get(mount.sandbox_path)
        if previous is not None:
            raise ValueError(
                "Duplicate host-directory sandbox destination: "
                f"{mount.sandbox_path!r}"
            )
        destinations[mount.sandbox_path] = mount
        previous_access = host_access.get(mount.host_path)
        if previous_access is not None and previous_access != mount.read_only:
            raise ValueError(
                "A host directory cannot be exposed with conflicting access modes: "
                f"{mount.host_path!r}"
            )
        host_access[mount.host_path] = mount.read_only
    return normalized


class PathMode(str, Enum):
    COPY = "copy"
    MOUNT = "mount"


class PreparedPath(BaseModel):
    """One host path the kernel will expose inside an execution session."""

    model_config = ConfigDict(frozen=True)

    source: Path
    target: str | None = None
    relative_target: str | None = None
    mode: PathMode = PathMode.COPY
    read_only: bool = True

    @field_validator("source")
    @classmethod
    def normalize_source(cls, source: Path) -> Path:
        return source.expanduser().resolve(strict=True)

    @field_validator("target", "relative_target")
    @classmethod
    def normalize_target(cls, target: str | None) -> str | None:
        if target is None:
            return None
        path = PurePosixPath(target)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Sandbox destinations must be relative and traversal-free")
        normalized = posixpath.normpath(target)
        if normalized in ("", "."):
            raise ValueError("Sandbox destinations cannot be empty")
        return normalized


def _glob_patterns(value: str | Sequence[str], *, name: str) -> tuple[str, ...]:
    patterns = (value,) if isinstance(value, str) else tuple(value)
    if not patterns:
        raise ValueError(f"PreparedInput.glob() {name} patterns cannot be empty")
    for pattern in patterns:
        parsed = PurePosixPath(pattern)
        if not pattern or parsed == PurePosixPath("."):
            raise ValueError(f"PreparedInput.glob() {name} patterns cannot be empty")
        if parsed.is_absolute() or ".." in parsed.parts:
            raise ValueError(f"Glob {name} patterns must stay under the source root")
    return patterns


def _glob_paths(
    root: Path,
    *,
    include: str | Sequence[str],
    exclude: str | Sequence[str],
) -> tuple[tuple[Path, str], ...]:
    root = root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("PreparedInput.glob() source_root must be a directory")
    include_patterns = _glob_patterns(include, name="include")
    exclude_patterns = () if not exclude else _glob_patterns(exclude, name="exclude")
    matches: dict[str, Path] = {}
    for pattern in include_patterns:
        for candidate in root.glob(pattern):
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(root).as_posix()
            if any(fnmatch.fnmatchcase(relative, item) for item in exclude_patterns):
                continue
            resolved = candidate.resolve(strict=True)
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"Glob match escapes source root through a symlink: {candidate}"
                ) from exc
            matches[relative] = resolved
    return tuple((matches[relative], relative) for relative in sorted(matches))


def _strip_annotated(annotation: Any) -> Any:
    while typing.get_origin(annotation) is typing.Annotated:
        annotation = typing.get_args(annotation)[0]
    return annotation


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    annotation = _strip_annotated(annotation)
    origin = typing.get_origin(annotation)
    if origin not in (typing.Union, types.UnionType):
        return annotation, False
    args = typing.get_args(annotation)
    non_none = [item for item in args if item is not type(None)]
    if len(args) != 2 or len(non_none) != 1:
        return annotation, False
    return _strip_annotated(non_none[0]), True


@dataclass(frozen=True)
class FieldDescriptor:
    """Normalized field shape supplied to runtime adapters."""

    name: str
    annotation: Any
    item_annotation: Any = field(init=False)
    is_list: bool = field(init=False)
    allows_none: bool = field(init=False)
    item_allows_none: bool = field(init=False)

    def __post_init__(self) -> None:
        normalized, allows_none = _unwrap_optional(self.annotation)
        is_list = typing.get_origin(normalized) is list
        if is_list:
            args = typing.get_args(normalized)
            item = args[0] if args else Any
        else:
            item = normalized
        item, item_allows_none = _unwrap_optional(item)
        object.__setattr__(self, "item_annotation", item)
        object.__setattr__(self, "is_list", is_list)
        object.__setattr__(self, "allows_none", allows_none)
        object.__setattr__(self, "item_allows_none", item_allows_none)

    def matches(self, value_type: type[Any]) -> bool:
        return isinstance(self.item_annotation, type) and issubclass(
            self.item_annotation, value_type
        )

    def unpack(self, value: Any) -> list[Any]:
        return list(value) if self.is_list else [value]

    def pack(self, values: Sequence[Any]) -> Any:
        return list(values) if self.is_list else values[0]

    def replace_type(self, value_type: type[Any]) -> Any:
        item_type = value_type | None if self.item_allows_none else value_type
        replacement = list[item_type] if self.is_list else item_type
        return replacement | None if self.allows_none else replacement


@dataclass(frozen=True)
class PreparedInput:
    """One adapter's model-visible value and opaque artifacts."""

    model_value: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)
    instructions: tuple[str, ...] = ()
    artifacts: tuple[Artifact, ...] = ()
    host_directory_mounts: tuple[HostDirectoryMount, ...] = ()
    sandbox_roots: tuple[SandboxRootReservation, ...] = ()
    requirements: SessionRequirements = field(default_factory=SessionRequirements)
    prepared_paths: tuple[PreparedPath, ...] = field(default=(), repr=False)
    path_model_value: Literal["single", "list"] | None = field(
        default=None,
        repr=False,
    )

    @classmethod
    def path(
        cls,
        source: str | os.PathLike[str],
        *,
        at: str | None = None,
        mode: PathMode | Literal["copy", "mount"] = PathMode.COPY,
        read_only: bool = True,
        metadata: Mapping[str, Any] | None = None,
        instructions: Sequence[str] = (),
    ) -> PreparedInput:
        """Expose one host path and pass its sandbox path to the model."""

        return cls(
            model_value=None,
            metadata=metadata or {},
            instructions=tuple(instructions),
            prepared_paths=(
                PreparedPath(
                    source=Path(source),
                    target=at,
                    mode=PathMode(mode),
                    read_only=read_only,
                ),
            ),
            path_model_value="single",
        )

    @classmethod
    def paths(
        cls,
        sources: Sequence[str | os.PathLike[str]],
        *,
        at: str | None = None,
        mode: PathMode | Literal["copy", "mount"] = PathMode.COPY,
        read_only: bool = True,
        metadata: Mapping[str, Any] | None = None,
        instructions: Sequence[str] = (),
    ) -> PreparedInput:
        """Expose explicit host paths and pass their sandbox paths as a list."""

        return cls(
            model_value=[],
            metadata=metadata or {},
            instructions=tuple(instructions),
            prepared_paths=tuple(
                PreparedPath(
                    source=Path(source),
                    target=at,
                    relative_target=Path(source).name,
                    mode=PathMode(mode),
                    read_only=read_only,
                )
                for source in sources
            ),
            path_model_value="list",
        )

    @classmethod
    def glob(
        cls,
        root: str | os.PathLike[str],
        *,
        include: str | Sequence[str],
        exclude: str | Sequence[str] = (),
        at: str | None = None,
        allow_empty: bool = False,
        metadata: Mapping[str, Any] | None = None,
        instructions: Sequence[str] = (),
    ) -> PreparedInput:
        """Copy a deterministic host-side glob and pass sandbox paths as a list."""

        matches = _glob_paths(Path(root), include=include, exclude=exclude)
        if not matches and not allow_empty:
            raise ValueError(
                f"PreparedInput.glob() did not match any files under {Path(root)!s}"
            )
        return cls(
            model_value=[],
            metadata=metadata or {},
            instructions=tuple(instructions),
            prepared_paths=tuple(
                PreparedPath(
                    source=source,
                    target=at,
                    relative_target=relative,
                )
                for source, relative in matches
            ),
            path_model_value="list",
        )

    def __post_init__(self) -> None:
        if not isinstance(self.requirements, SessionRequirements):
            raise TypeError("PreparedInput.requirements must be SessionRequirements")
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))
        object.__setattr__(self, "instructions", tuple(self.instructions))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "prepared_paths", tuple(self.prepared_paths))
        object.__setattr__(self, "sandbox_roots", tuple(self.sandbox_roots))
        if not all(
            isinstance(reservation, SandboxRootReservation)
            for reservation in self.sandbox_roots
        ):
            raise TypeError(
                "PreparedInput.sandbox_roots must contain SandboxRootReservation values"
            )
        object.__setattr__(
            self,
            "host_directory_mounts",
            normalize_host_directory_mounts(self.host_directory_mounts),
        )


def _prepared_path_destination(field: FieldDescriptor, path: PreparedPath) -> str:
    root = f"/sandbox/{path.target}" if path.target else f"/sandbox/input/{field.name}"
    if path.relative_target:
        return posixpath.join(root, path.relative_target)
    if path.target is None and path.source.is_file():
        return posixpath.join(root, path.source.name)
    return root


def compile_prepared_input(
    field: FieldDescriptor,
    prepared: PreparedInput,
) -> PreparedInput:
    """Lower adapter-authored paths into backend-facing execution state."""

    if prepared.path_model_value is None:
        return prepared

    destinations = tuple(
        _prepared_path_destination(field, path) for path in prepared.prepared_paths
    )
    artifacts = list(prepared.artifacts)
    mounts = list(prepared.host_directory_mounts)
    reservations = list(prepared.sandbox_roots)
    read_paths = list(prepared.requirements.extra_read_paths)
    write_paths = list(prepared.requirements.extra_write_paths)

    for path, destination in zip(prepared.prepared_paths, destinations, strict=True):
        source = str(path.source)
        reservations.append(SandboxRootReservation(destination))
        read_paths.append(source)
        if path.mode is PathMode.COPY:
            artifacts.append(
                Artifact(
                    id=f"prepared-path-{uuid.uuid4().hex}",
                    kind="runtime.path",
                    metadata={
                        "source_path": source,
                        "sandbox_path": destination,
                    },
                )
            )
            continue
        if not path.read_only:
            write_paths.append(source)
        mounts.append(
            HostDirectoryMount(
                host_path=source,
                sandbox_path=destination,
                read_only=path.read_only,
            )
        )

    if prepared.path_model_value == "single":
        if len(destinations) != 1:
            raise ValueError("PreparedInput.path() must resolve exactly one path")
        model_value: Any = destinations[0]
    else:
        model_value = list(destinations)

    return replace(
        prepared,
        model_value=model_value,
        artifacts=tuple(artifacts),
        host_directory_mounts=tuple(mounts),
        sandbox_roots=tuple(reservations),
        requirements=SessionRequirements(
            allowed_domains=prepared.requirements.allowed_domains,
            extra_read_paths=tuple(dict.fromkeys(read_paths)),
            extra_write_paths=tuple(dict.fromkeys(write_paths)),
        ),
        prepared_paths=(),
        path_model_value=None,
    )


def _replace_prepared_path(value: Any, planned: str, actual: str) -> Any:
    if value == planned:
        return actual
    if isinstance(value, list):
        return [_replace_prepared_path(item, planned, actual) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_prepared_path(item, planned, actual) for item in value)
    if isinstance(value, dict):
        return {
            key: _replace_prepared_path(item, planned, actual)
            for key, item in value.items()
        }
    return value


@dataclass(frozen=True)
class BoundInput:
    """Model-visible value and bindings produced by an input adapter bind."""

    model_value: Any
    bindings: tuple[ArtifactBinding, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", tuple(self.bindings))


@dataclass
class PreparedInputBinding:
    """Invocation-local ownership record for one prepared input field."""

    field: FieldDescriptor
    adapter: InputAdapter[Any]
    prepared: PreparedInput
    open_entered: bool = False
    bound: bool = False
    finalized: bool = False


def validate_sandbox_root_reservations(
    bindings: Mapping[str, PreparedInputBinding],
) -> None:
    """Reject cross-adapter sandbox destination overlap before acquisition."""
    claimed: list[tuple[str, str]] = []
    for field_name, binding in bindings.items():
        reservation_paths = [
            reservation.path for reservation in binding.prepared.sandbox_roots
        ]
        if len(reservation_paths) != len(set(reservation_paths)):
            raise ValueError(
                "Input sandbox destinations overlap: "
                f"{field_name!r} reserves the same path more than once"
            )
        destinations = {
            *reservation_paths,
            *(
                mount.sandbox_path
                for mount in binding.prepared.host_directory_mounts
            ),
        }
        for path in sorted(destinations):
            for previous_field, previous_path in claimed:
                if _sandbox_paths_overlap(path, previous_path):
                    raise ValueError(
                        "Input sandbox destinations overlap: "
                        f"{previous_field!r} reserves {previous_path!r} and "
                        f"{field_name!r} reserves {path!r}"
                    )
            claimed.append((field_name, path))


def _sandbox_paths_overlap(first: str, second: str) -> bool:
    return (
        first == second
        or first.startswith(second.rstrip("/") + "/")
        or second.startswith(first.rstrip("/") + "/")
    )


@dataclass(frozen=True)
class OutputReservation:
    """Session-bound destination owned by exactly one output adapter."""

    field: FieldDescriptor
    artifact: Artifact
    model_value: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


def validate_output_sandbox_root_reservation(
    input_bindings: Mapping[str, PreparedInputBinding],
    output_reservations: Mapping[str, OutputReservation],
    candidate: OutputReservation,
) -> None:
    """Reject an output destination that overlaps an input or prior output."""
    sandbox_path = candidate.artifact.metadata.get("sandbox_path")
    if not isinstance(sandbox_path, str):
        return
    candidate_path = SandboxRootReservation(sandbox_path).path
    for field_name, binding in input_bindings.items():
        paths = {
            *(reservation.path for reservation in binding.prepared.sandbox_roots),
            *(
                mount.sandbox_path
                for mount in binding.prepared.host_directory_mounts
            ),
        }
        for path in paths:
            if _sandbox_paths_overlap(candidate_path, path):
                raise ValueError(
                    "Input/output sandbox destinations overlap: "
                    f"input {field_name!r} reserves {path!r} and output "
                    f"{candidate.field.name!r} reserves {candidate_path!r}"
                )
    for field_name, reservation in output_reservations.items():
        path = reservation.artifact.metadata.get("sandbox_path")
        if isinstance(path, str) and _sandbox_paths_overlap(
            candidate_path,
            SandboxRootReservation(path).path,
        ):
            raise ValueError(
                "Output sandbox destinations overlap: "
                f"{field_name!r} and {candidate.field.name!r}"
            )


@dataclass(frozen=True)
class ExecutionSpec:
    """Invocation configuration supplied when a backend opens a session."""

    tools: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
    output_fields: tuple[Mapping[str, Any], ...] = ()
    packages: tuple[str, ...] = ()
    allowed_domains: tuple[str, ...] = ()
    extra_read_paths: tuple[str, ...] = ()
    extra_write_paths: tuple[str, ...] = ()
    host_directory_mounts: tuple[HostDirectoryMount, ...] = ()
    sandbox_roots: tuple[SandboxRootReservation, ...] = ()
    debug: bool = False
    verbose: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tools", immutable_mapping(self.tools))
        object.__setattr__(
            self,
            "output_fields",
            tuple(immutable_mapping(item) for item in self.output_fields),
        )
        object.__setattr__(self, "packages", tuple(self.packages))
        object.__setattr__(self, "allowed_domains", tuple(self.allowed_domains))
        object.__setattr__(self, "extra_read_paths", tuple(self.extra_read_paths))
        object.__setattr__(self, "extra_write_paths", tuple(self.extra_write_paths))
        object.__setattr__(
            self,
            "host_directory_mounts",
            normalize_host_directory_mounts(self.host_directory_mounts),
        )
        object.__setattr__(self, "sandbox_roots", tuple(self.sandbox_roots))
        if not all(
            isinstance(reservation, SandboxRootReservation)
            for reservation in self.sandbox_roots
        ):
            raise TypeError(
                "ExecutionSpec.sandbox_roots must contain SandboxRootReservation values"
            )
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


@dataclass(frozen=True)
class ExecutionResult:
    """Backend-neutral result of one generated-code execution."""

    value: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


class SessionOwnership(str, Enum):
    OWNED = "owned"
    INJECTED = "injected"
    POOLED = "pooled"


class InputAdapter(ABC, Generic[AdapterValue]):
    """Shared configuration for invocation-local input lifecycles.

    Keep mutable state in ``PreparedInput`` or ``RunContext``. The lifecycle is
    prepare, open, bind, per-attempt durability, then
    reverse-order finalization.
    """

    name: ClassVar[str]
    value_type: ClassVar[type[Any]]
    fallback: ClassVar[bool] = False

    def supports(self, field: FieldDescriptor, value: Any) -> bool:
        return field.matches(self.value_type)

    def append_prompt(
        self,
        prompt: str,
        field: FieldDescriptor,
        prepared: PreparedInput,
        ctx: RunContext,
    ) -> str:
        """Return this invocation's prompt after optional adapter customization."""
        return prompt

    def _transform_prompt_signature(
        self,
        signature: type[dspy.Signature],
        field: FieldDescriptor,
        prepared: PreparedInput,
        ctx: RunContext,
    ) -> type[dspy.Signature]:
        """Return the invocation-local signature used only to construct prompts."""
        return signature

    @abstractmethod
    async def prepare(
        self,
        field: FieldDescriptor,
        value: AdapterValue | list[AdapterValue] | None,
        ctx: RunContext,
    ) -> PreparedInput:
        """Describe the model value and session needs before acquisition.

        Finalization is guaranteed only after ``open`` is entered, so
        clean up partial failures here, use ``ctx.add_cleanup``, or defer owned
        resource acquisition to that hook.
        """
        ...

    async def open(
        self,
        field: FieldDescriptor,
        prepared: PreparedInput,
        ctx: RunContext,
        backend: ExecutionBackend,
    ) -> None:
        """Optionally acquire adapter-owned state before backend acquisition.

        Once this hook is entered, ``finalize`` runs even when the hook raises.
        """

    async def bind(
        self,
        field: FieldDescriptor,
        prepared: PreparedInput,
        ctx: RunContext,
        session: ExecutionSession,
    ) -> BoundInput:
        """Bind the prepared input after acquisition without owning the session.

        The default mounts every prepared artifact. A failure still proceeds
        through adapter and framework finalization.
        """
        del field
        model_value = prepared.model_value
        bindings = []
        for artifact in prepared.artifacts:
            binding = await session.mount(artifact)
            bindings.append(binding)
            planned = artifact.metadata.get("sandbox_path")
            if isinstance(planned, str):
                model_value = _replace_prepared_path(
                    model_value,
                    planned,
                    binding.path,
                )
        for mount in prepared.host_directory_mounts:
            if not isinstance(session, HostDirectorySession):
                raise UnsupportedOperationError(
                    f"Input {mount.sandbox_path!r} requires a live host-path mount"
                )
            path = await session.mount_host_directory(mount)
            bindings.append(
                ArtifactBinding(
                    artifact_id=f"host-path-{uuid.uuid4().hex}",
                    path=path,
                )
            )
            model_value = _replace_prepared_path(
                model_value,
                mount.sandbox_path,
                path,
            )
        return BoundInput(model_value=model_value, bindings=tuple(bindings))

    async def after_execution(
        self,
        field: FieldDescriptor,
        prepared: PreparedInput,
        ctx: RunContext,
        session: ExecutionSession,
        result: ExecutionResult | None,
        error: BaseException | None,
    ) -> None:
        """Durably observe a completed generated-code attempt.

        ``result`` or ``error`` describes its outcome. Framework bootstrap and
        cancelled attempts are excluded. A hook failure aborts the run or is
        attached to an execution failure; use ``finalize`` for the last flush
        after cancellation.
        """

    async def finalize(
        self,
        field: FieldDescriptor,
        prepared: PreparedInput,
        ctx: RunContext,
        session: ExecutionSession | None,
        error: BaseException | None,
    ) -> None:
        """Release adapter-owned state exactly once before session finalization.

        Adapters whose ``open`` hook was entered finalize in reverse
        preparation order on success, failure, or cancellation. Failures do not
        skip later adapter or session finalization. ``session`` is ``None`` when
        acquisition never completed; the framework owns its finalization and
        release.
        """


class OutputAdapter(ABC, Generic[AdapterValue]):
    """Shared configuration for reserving and materializing output destinations.

    Pre-acquisition requirements are followed by session-bound reservation before
    generated execution and materialization after the final prediction.
    """

    name: ClassVar[str]
    value_type: ClassVar[type[Any]]

    def supports(self, field: FieldDescriptor) -> bool:
        return field.matches(self.value_type)

    async def prepare_session(
        self,
        field: FieldDescriptor,
        value: AdapterValue | list[AdapterValue] | None,
        ctx: RunContext,
    ) -> SessionRequirements | None:
        """Contribute policy or prepare a host destination before acquisition.

        Output adapters have no finalization hook: clean partial failures locally
        and use ``ctx.add_cleanup`` for host or provider state that can outlive
        the session.
        """

    @abstractmethod
    async def reserve(
        self,
        field: FieldDescriptor,
        value: AdapterValue | list[AdapterValue] | None,
        ctx: RunContext,
        session: ExecutionSession,
    ) -> OutputReservation:
        """Bind one destination after acquisition and before generated execution.

        Return its artifact and model-visible value; a failure aborts the run.
        """
        ...

    @abstractmethod
    async def materialize(
        self,
        reservation: OutputReservation,
        submitted_value: Any,
        ctx: RunContext,
        session: ExecutionSession,
    ) -> Any:
        """Collect or translate a submitted value while the session is active.

        This runs after the final prediction and returns the public value that
        replaces the submitted field; a failure aborts the run.
        """
        ...


Adapter = InputAdapter[Any] | OutputAdapter[Any]


@runtime_checkable
class RuntimeTool(Protocol):
    name: str
    description: str
    schema: Mapping[str, Any]

    async def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class CallableTool:
    """Async-first adapter for existing callables and DSPy tools."""

    name: str
    function: Callable[..., Any] = field(compare=False, repr=False)
    description: str = ""
    schema: Mapping[str, Any] = field(default_factory=dict)
    extra_read_paths: tuple[str, ...] = ()
    extra_write_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", immutable_mapping(self.schema))
        object.__setattr__(self, "extra_read_paths", tuple(self.extra_read_paths))
        object.__setattr__(self, "extra_write_paths", tuple(self.extra_write_paths))

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return await invoke_host_callable(self.function, *args, **kwargs)


async def invoke_host_callable(
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    if inspect.iscoroutinefunction(function):
        return await function(*args, **kwargs)
    policy = _HOST_SYNC_WORKER_POLICY.get()
    worker = (
        policy.owner._start_sync_worker(function, *args, **kwargs)
        if policy is not None
        else SyncWorker(function, *args, **kwargs)
    )
    try:
        if policy is not None and policy.timeout is not None:
            result = await asyncio.wait_for(worker.wait(), timeout=policy.timeout)
        else:
            result = await worker.wait()
    except TimeoutError as timeout:
        setattr(timeout, "sync_worker", worker)
        raise
    except asyncio.CancelledError as cancellation:
        setattr(cancellation, "sync_worker", worker)
        if policy is not None and policy.detach_on_cancel:
            raise
        await worker.wait()
        raise
    if inspect.isawaitable(result):
        return await result
    return result


@runtime_checkable
class ExecutionSession(Protocol):
    name: str
    ownership: SessionOwnership

    async def install_packages(self, packages: Sequence[str]) -> None: ...

    async def mount(self, artifact: Artifact) -> ArtifactBinding: ...

    async def run_code(
        self,
        code: str,
        variables: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> ExecutionResult: ...

    async def collect(self, artifact: Artifact) -> Any: ...

    async def finalize(self) -> None: ...

    async def cancel(self) -> None: ...


@runtime_checkable
class HostDirectorySession(Protocol):
    """Optional session operation for exposing host directories directly."""

    async def mount_host_directory(self, mount: HostDirectoryMount) -> str: ...


@runtime_checkable
class FileTransferSession(Protocol):
    """Optional session operation for copying typed host files into a sandbox."""

    async def transfer_file(self, transfer: FileTransfer) -> str: ...


@runtime_checkable
class DirectoryCreationSession(Protocol):
    """Optional session operation for creating a copy-in directory root."""

    async def create_directory(self, sandbox_path: str) -> None: ...


@runtime_checkable
class MutableDirectorySession(Protocol):
    """Optional session operations used to sync mutable directories back."""

    async def inspect_directory(
        self,
        sandbox_path: str,
    ) -> Mapping[str, ArtifactFileInfo]: ...

    async def collect_file(self, sandbox_path: str, host_path: str) -> None: ...


@runtime_checkable
class ExecutionBackend(Protocol):
    name: str

    def start(
        self,
        spec: ExecutionSpec,
        ctx: RunContext,
    ) -> Any:
        """Return an async context manager yielding one invocation session."""
        ...


@runtime_checkable
class EventSink(Protocol):
    strict: bool

    async def emit(self, event: Any) -> None: ...

    async def flush(self, run_id: str) -> None: ...

    async def close(self, run_id: str, terminal_event: Any | None = None) -> None: ...


SignatureValidator = Callable[[Any], None]


class ToolOperation(ABC):
    """Construction-time operation that wraps host tools portably."""

    name: ClassVar[str]

    @abstractmethod
    def apply(self, tool: RuntimeTool) -> RuntimeTool: ...


@dataclass(frozen=True)
class RuntimeContribution:
    """Construction-time values contributed by a module factory."""

    instructions: tuple[str, ...] = ()
    adapters: Sequence[Adapter] = ()
    tools: tuple[RuntimeTool, ...] = ()
    packages: tuple[str, ...] = ()
    execution: ExecutionBackend | None = None
    events: tuple[EventSink, ...] = ()
    validators: tuple[SignatureValidator, ...] = ()
    tool_operations: tuple[ToolOperation, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapters", tuple(self.adapters))


RuntimeModule = Callable[[], RuntimeContribution]


@dataclass(frozen=True)
class RuntimeSpec:
    """Immutable construction-time configuration consumed by every run."""

    instructions: tuple[str, ...]
    adapters: tuple[Adapter, ...]
    tools: tuple[RuntimeTool, ...]
    packages: tuple[str, ...]
    execution: ExecutionBackend
    events: tuple[EventSink, ...]
    validators: tuple[SignatureValidator, ...] = ()
    tool_operations: tuple[ToolOperation, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "instructions", tuple(self.instructions))
        object.__setattr__(self, "adapters", tuple(self.adapters))
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "packages", tuple(self.packages))
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "validators", tuple(self.validators))
        object.__setattr__(self, "tool_operations", tuple(self.tool_operations))
        _validate_adapters(self.adapters)
        _validate_unique_names(self.tools, "tool")
        _validate_unique_names(self.input_adapters, "input adapter")
        _validate_unique_names(self.output_adapters, "output adapter")
        _validate_unique_names(self.tool_operations, "tool operation")

    @property
    def input_adapters(self) -> tuple[InputAdapter[Any], ...]:
        return tuple(
            adapter for adapter in self.adapters if isinstance(adapter, InputAdapter)
        )

    @property
    def output_adapters(self) -> tuple[OutputAdapter[Any], ...]:
        return tuple(
            adapter for adapter in self.adapters if isinstance(adapter, OutputAdapter)
        )


Cleanup = Callable[[], Awaitable[None] | None]


@dataclass
class RunContext:
    """Fresh invocation-local state. A context is never reused."""

    spec: RuntimeSpec
    input_values: Mapping[str, Any]
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    input_bindings: dict[str, PreparedInputBinding] = field(default_factory=dict)
    bound_input_bindings: list[PreparedInputBinding] = field(
        default_factory=list,
        repr=False,
    )
    artifact_bindings: dict[str, ArtifactBinding] = field(default_factory=dict)
    output_reservations: dict[str, OutputReservation] = field(default_factory=dict)
    output_adapters: dict[str, OutputAdapter[Any]] = field(default_factory=dict, repr=False)
    output_requirements: dict[str, SessionRequirements] = field(
        default_factory=dict,
        repr=False,
    )
    session: ExecutionSession | None = None
    ownership: SessionOwnership | None = None
    local_paths: dict[str, str] = field(default_factory=dict)
    credentials: dict[str, Any] = field(default_factory=dict, repr=False)
    cleanup_callbacks: list[Cleanup] = field(default_factory=list, repr=False)
    state: dict[str, Any] = field(default_factory=dict, repr=False)
    session_finalized: bool = False
    evidence_complete: bool = True
    terminal_outcome: str | None = None

    def __post_init__(self) -> None:
        self.input_values = immutable_mapping(self.input_values)

    def bind(self, binding: ArtifactBinding) -> None:
        if binding.artifact_id in self.artifact_bindings:
            raise RuntimeError(f"Artifact {binding.artifact_id!r} is already bound")
        self.artifact_bindings[binding.artifact_id] = binding

    def reserve(
        self,
        reservation: OutputReservation,
        adapter: OutputAdapter[Any],
    ) -> None:
        field_name = reservation.field.name
        if field_name in self.output_reservations:
            raise RuntimeError(
                f"Output field {field_name!r} already has a reservation"
            )
        self.output_reservations[field_name] = reservation
        self.output_adapters[field_name] = adapter

    def add_cleanup(self, callback: Cleanup) -> None:
        self.cleanup_callbacks.append(callback)

    async def cleanup(self) -> None:
        first_error: BaseException | None = None
        while self.cleanup_callbacks:
            callback = self.cleanup_callbacks.pop()
            try:
                result = callback()
                if inspect.isawaitable(result):
                    await result
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


_CURRENT_RUN_CONTEXT: ContextVar[RunContext | None] = ContextVar(
    "predict_rlm_run_context", default=None
)


def current_run_context() -> RunContext | None:
    return _CURRENT_RUN_CONTEXT.get()


@asynccontextmanager
async def use_run_context(ctx: RunContext) -> AsyncIterator[RunContext]:
    token: Token[RunContext | None] = _CURRENT_RUN_CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _CURRENT_RUN_CONTEXT.reset(token)


@contextmanager
def use_run_context_sync(ctx: RunContext):
    token: Token[RunContext | None] = _CURRENT_RUN_CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _CURRENT_RUN_CONTEXT.reset(token)


def resolve_runtime_spec(
    *,
    direct: RuntimeContribution,
    modules: Sequence[RuntimeModule | RuntimeContribution] = (),
) -> RuntimeSpec:
    """Expand contribution factories once and resolve deterministic ownership."""
    contributions = [direct]
    for module in modules:
        contribution = module() if callable(module) else module
        if not isinstance(contribution, RuntimeContribution):
            raise TypeError("Runtime modules must return RuntimeContribution")
        contributions.append(contribution)

    executions = [item.execution for item in contributions if item.execution is not None]
    if len(executions) != 1:
        raise ValueError(
            "Runtime configuration must select exactly one execution backend; "
            f"got {len(executions)}"
        )

    packages: list[str] = []
    seen_packages: set[str] = set()
    for contribution in contributions:
        for package in contribution.packages:
            if package not in seen_packages:
                seen_packages.add(package)
                packages.append(package)

    tool_operations = tuple(
        operation for item in contributions for operation in item.tool_operations
    )
    _validate_unique_names(tool_operations, "tool operation")
    tools: tuple[RuntimeTool, ...] = tuple(
        tool for item in contributions for tool in item.tools
    )
    for operation in tool_operations:
        tools = tuple(operation.apply(tool) for tool in tools)

    return RuntimeSpec(
        instructions=tuple(
            instruction
            for contribution in contributions
            for instruction in contribution.instructions
            if instruction
        ),
        adapters=tuple(adapter for item in contributions for adapter in item.adapters),
        tools=tools,
        packages=tuple(packages),
        execution=executions[0],
        events=tuple(sink for item in contributions for sink in item.events),
        validators=tuple(
            validator for item in contributions for validator in item.validators
        ),
        tool_operations=tool_operations,
    )


def resolve_input_adapter(
    adapters: Sequence[InputAdapter[Any]], field: FieldDescriptor, value: Any
) -> InputAdapter[Any]:
    matches = [adapter for adapter in adapters if adapter.supports(field, value)]
    concrete = [adapter for adapter in matches if not adapter.fallback]
    if concrete:
        matches = concrete
    exact = [adapter for adapter in matches if adapter.value_type is field.item_annotation]
    if exact:
        matches = exact
    else:
        matches = [
            adapter
            for adapter in matches
            if not any(
                other.value_type is not adapter.value_type
                and issubclass(other.value_type, adapter.value_type)
                for other in matches
            )
        ]
    return _resolve_adapter(matches, "input", field)


def resolve_output_adapter(
    adapters: Sequence[OutputAdapter[Any]], field: FieldDescriptor
) -> OutputAdapter[Any]:
    matches = [adapter for adapter in adapters if adapter.supports(field)]
    return _resolve_adapter(matches, "output", field)


def _resolve_adapter(
    matches: Sequence[Any], kind: str, field: FieldDescriptor
) -> Any:
    if not matches:
        raise ValueError(
            f"No {kind} adapter supports field {field.name!r} "
            f"with annotation {field.annotation!r}"
        )
    if len(matches) > 1:
        names = ", ".join(adapter.name for adapter in matches)
        raise ValueError(
            f"Multiple {kind} adapters support field {field.name!r} "
            f"with annotation {field.annotation!r}: {names}"
        )
    return matches[0]


def _validate_unique_names(values: Sequence[Any], kind: str) -> None:
    seen: set[str] = set()
    for value in values:
        name = getattr(value, "name", None)
        if not isinstance(name, str) or not name:
            raise TypeError(f"Each {kind} must define a non-empty name")
        if name in seen:
            raise ValueError(f"Duplicate {kind} name: {name!r}")
        seen.add(name)


def _validate_adapters(adapters: Sequence[Any]) -> None:
    for adapter in adapters:
        if not isinstance(adapter, (InputAdapter, OutputAdapter)):
            raise TypeError("Adapters must inherit from InputAdapter or OutputAdapter")
        if not isinstance(getattr(adapter, "value_type", None), type):
            raise TypeError("Each adapter must define value_type as a class")
