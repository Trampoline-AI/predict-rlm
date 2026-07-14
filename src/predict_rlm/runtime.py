"""Small-kernel runtime contracts for PredictRLM."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import inspect
import threading
import types
import typing
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Generic, Protocol, TypeVar, runtime_checkable

RUNTIME_SPI_VERSION = "1"

AdapterValue = TypeVar("AdapterValue")


class ExecutionFatalError(RuntimeError):
    """The execution session is no longer safe to continue."""


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


# The research document used MountedArtifact. ArtifactBinding is the final name,
# while this alias keeps the documented draft import working.
MountedArtifact = ArtifactBinding


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))
        object.__setattr__(self, "instructions", tuple(self.instructions))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))


@dataclass(frozen=True)
class OutputReservation:
    """Session-bound destination owned by exactly one output adapter."""

    field: FieldDescriptor
    artifact: Artifact
    model_value: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


@dataclass(frozen=True)
class ExecutionSpec:
    """Invocation configuration supplied when a backend opens a session."""

    tools: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
    output_fields: tuple[Mapping[str, Any], ...] = ()
    packages: tuple[str, ...] = ()
    allowed_domains: tuple[str, ...] = ()
    extra_read_paths: tuple[str, ...] = ()
    extra_write_paths: tuple[str, ...] = ()
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
    """Base class for values prepared before an execution session starts."""

    name: ClassVar[str]
    value_type: ClassVar[type[Any]]
    fallback: ClassVar[bool] = False

    def supports(self, field: FieldDescriptor, value: Any) -> bool:
        return field.matches(self.value_type)

    @abstractmethod
    async def prepare(
        self,
        field: FieldDescriptor,
        value: AdapterValue | list[AdapterValue] | None,
        ctx: RunContext,
    ) -> PreparedInput: ...


class OutputAdapter(ABC, Generic[AdapterValue]):
    """Base class for session-bound output destinations."""

    name: ClassVar[str]
    value_type: ClassVar[type[Any]]

    def supports(self, field: FieldDescriptor) -> bool:
        return field.matches(self.value_type)

    async def prepare_session(
        self,
        field: FieldDescriptor,
        value: AdapterValue | list[AdapterValue] | None,
        ctx: RunContext,
    ) -> None:
        """Contribute host paths or other requirements before session startup."""

    @abstractmethod
    async def reserve(
        self,
        field: FieldDescriptor,
        value: AdapterValue | list[AdapterValue] | None,
        ctx: RunContext,
        session: ExecutionSession,
    ) -> OutputReservation: ...

    @abstractmethod
    async def materialize(
        self,
        reservation: OutputReservation,
        submitted_value: Any,
        ctx: RunContext,
        session: ExecutionSession,
    ) -> Any: ...


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
    spi_version: str = RUNTIME_SPI_VERSION

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
    prepared_inputs: dict[str, PreparedInput] = field(default_factory=dict)
    artifact_bindings: dict[str, ArtifactBinding] = field(default_factory=dict)
    output_reservations: dict[str, OutputReservation] = field(default_factory=dict)
    output_adapters: dict[str, OutputAdapter[Any]] = field(default_factory=dict, repr=False)
    session: ExecutionSession | None = None
    ownership: SessionOwnership | None = None
    local_paths: dict[str, str] = field(default_factory=dict)
    credentials: dict[str, Any] = field(default_factory=dict, repr=False)
    cleanup_callbacks: list[Cleanup] = field(default_factory=list, repr=False)
    state: dict[str, Any] = field(default_factory=dict, repr=False)
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
