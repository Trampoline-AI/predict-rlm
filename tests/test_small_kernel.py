from __future__ import annotations

import asyncio
import inspect
import shutil
import threading
from collections.abc import Sequence
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock

import dspy
import pytest
from dspy.primitives.code_interpreter import FinalOutput

from predict_rlm.backends.adapters import (
    InterpreterBackendAdapter,
    InterpreterExecutionSession,
)
from predict_rlm.compatibility import (
    FileInputAdapter,
    FileOutputAdapter,
    SyncedFileToolOperation,
)
from predict_rlm.evidence import (
    EvidenceIncompleteError,
    EvidenceRecorder,
    RunEventKind,
)
from predict_rlm.files import File as RuntimeFile
from predict_rlm.files import SyncedFile
from predict_rlm.runtime import (
    Artifact,
    ArtifactBinding,
    CallableTool,
    ExecutionSpec,
    HostDirectoryMount,
    InputAdapter,
    MountedInput,
    OutputAdapter,
    PreparedInput,
    RunContext,
    RuntimeContribution,
    RuntimeSpec,
    SessionOwnership,
    SessionRequirements,
    callable_has_sync_leaf,
    current_run_context,
    resolve_runtime_spec,
    use_run_context,
)
from predict_rlm.workspace import Workspace


class KernelFileSignature(dspy.Signature):
    source: RuntimeFile = dspy.InputField()
    result: RuntimeFile = dspy.OutputField()


class FileUnionInputSignature(dspy.Signature):
    source: RuntimeFile | str = dspy.InputField()
    answer: str = dspy.OutputField()


class NestedFileInputSignature(dspy.Signature):
    source: list[list[RuntimeFile]] = dspy.InputField()
    answer: str = dspy.OutputField()


class WorkspaceUnionInputSignature(dspy.Signature):
    source: Workspace | str = dspy.InputField()
    answer: str = dspy.OutputField()


class NestedWorkspaceInputSignature(dspy.Signature):
    source: list[list[Workspace]] = dspy.InputField()
    answer: str = dspy.OutputField()


class TupleFileInputSignature(dspy.Signature):
    source: tuple[RuntimeFile, ...] = dspy.InputField()
    answer: str = dspy.OutputField()


class SetFileInputSignature(dspy.Signature):
    source: set[RuntimeFile] = dspy.InputField()
    answer: str = dspy.OutputField()


class DictFileInputSignature(dspy.Signature):
    source: dict[str, RuntimeFile] = dspy.InputField()
    answer: str = dspy.OutputField()


class SequenceFileInputSignature(dspy.Signature):
    source: Sequence[RuntimeFile] = dspy.InputField()
    answer: str = dspy.OutputField()


class NestedGenericFileInputSignature(dspy.Signature):
    source: list[dict[str, tuple[RuntimeFile, ...]]] = dspy.InputField()
    answer: str = dspy.OutputField()


class TupleWorkspaceInputSignature(dspy.Signature):
    source: tuple[Workspace, ...] = dspy.InputField()
    answer: str = dspy.OutputField()


class SetWorkspaceInputSignature(dspy.Signature):
    source: set[Workspace] = dspy.InputField()
    answer: str = dspy.OutputField()


class DictWorkspaceInputSignature(dspy.Signature):
    source: dict[str, Workspace] = dspy.InputField()
    answer: str = dspy.OutputField()


class SequenceWorkspaceInputSignature(dspy.Signature):
    source: Sequence[Workspace] = dspy.InputField()
    answer: str = dspy.OutputField()


class NestedGenericWorkspaceInputSignature(dspy.Signature):
    source: list[dict[str, tuple[Workspace, ...]]] = dspy.InputField()
    answer: str = dspy.OutputField()


class StubBackend:
    name = "stub"

    def start(self, spec, ctx):
        raise NotImplementedError


def make_spec(*, events=()) -> RuntimeSpec:
    return RuntimeSpec(
        instructions=(),
        adapters=(),
        tools=(),
        packages=(),
        execution=StubBackend(),
        events=events,
    )


def test_predict_rlm_exposes_one_adapter_extension_parameter():
    from predict_rlm import PredictRLM

    parameters = inspect.signature(PredictRLM.__init__).parameters

    assert "adapters" in parameters
    assert parameters["events"].annotation == "Sequence[EventSink]"
    assert "inputs" not in parameters
    assert "outputs" not in parameters


def test_runtime_spec_partitions_an_immutable_adapter_snapshot_by_role():
    class SharedInputAdapter(InputAdapter[str]):
        name = "shared"
        value_type = str

        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=value)

    class SharedOutputAdapter(OutputAdapter[str]):
        name = "shared"
        value_type = str

        async def reserve(self, field, value, ctx, session):
            raise NotImplementedError

        async def materialize(self, reservation, submitted_value, ctx, session):
            raise NotImplementedError

    input_adapter = SharedInputAdapter()
    output_adapter = SharedOutputAdapter()

    def output_module():
        return RuntimeContribution(adapters=[output_adapter])

    contribution = RuntimeContribution(
        adapters=[input_adapter],
        execution=StubBackend(),
    )
    spec = resolve_runtime_spec(direct=contribution, modules=(output_module,))

    assert isinstance(contribution.adapters, tuple)
    assert isinstance(spec.adapters, tuple)
    assert spec.adapters == (input_adapter, output_adapter)
    assert spec.input_adapters == (input_adapter,)
    assert spec.output_adapters == (output_adapter,)


def test_runtime_spec_rejects_duplicate_adapter_names_within_one_role():
    class DuplicateInputAdapter(InputAdapter[str]):
        name = "duplicate"
        value_type = str

        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=value)

    with pytest.raises(ValueError, match="Duplicate input adapter name"):
        resolve_runtime_spec(
            direct=RuntimeContribution(
                adapters=[DuplicateInputAdapter(), DuplicateInputAdapter()],
                execution=StubBackend(),
            )
        )


def test_configured_adapter_names_suppress_compatibility_defaults_per_role():
    from predict_rlm import PredictRLM

    class CustomFileInputAdapter(InputAdapter[RuntimeFile]):
        name = "file"
        value_type = RuntimeFile

        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=value)

    class CustomFileOutputAdapter(OutputAdapter[RuntimeFile]):
        name = "file"
        value_type = RuntimeFile

        async def reserve(self, field, value, ctx, session):
            raise NotImplementedError

        async def materialize(self, reservation, submitted_value, ctx, session):
            raise NotImplementedError

    custom_input = CustomFileInputAdapter()
    input_override = PredictRLM(KernelFileSignature, adapters=[custom_input])
    assert custom_input in input_override.runtime_spec.input_adapters
    assert not any(
        isinstance(adapter, FileInputAdapter)
        for adapter in input_override.runtime_spec.input_adapters
    )
    assert any(
        isinstance(adapter, FileOutputAdapter)
        for adapter in input_override.runtime_spec.output_adapters
    )

    custom_output = CustomFileOutputAdapter()
    output_override = PredictRLM(KernelFileSignature, adapters=[custom_output])
    assert any(
        isinstance(adapter, FileInputAdapter)
        for adapter in output_override.runtime_spec.input_adapters
    )
    assert custom_output in output_override.runtime_spec.output_adapters
    assert not any(
        isinstance(adapter, FileOutputAdapter)
        for adapter in output_override.runtime_spec.output_adapters
    )


def test_runtime_spec_expands_modules_once_and_deduplicates_packages():
    calls = 0

    def module():
        nonlocal calls
        calls += 1
        return RuntimeContribution(instructions=("module",), packages=("b", "a"))

    spec = resolve_runtime_spec(
        direct=RuntimeContribution(
            instructions=("direct",),
            packages=("a",),
            execution=StubBackend(),
        ),
        modules=(module,),
    )

    assert calls == 1
    assert spec.instructions == ("direct", "module")
    assert spec.packages == ("a", "b")


def test_runtime_spec_rejects_duplicate_tool_names():
    first = CallableTool(name="same", function=lambda: 1)
    second = CallableTool(name="same", function=lambda: 2)

    with pytest.raises(ValueError, match="Duplicate tool name"):
        resolve_runtime_spec(
            direct=RuntimeContribution(
                tools=(first, second),
                execution=StubBackend(),
            )
        )


@pytest.mark.asyncio
async def test_run_context_is_invocation_local():
    first = RunContext(make_spec(), {"value": 1})
    second = RunContext(make_spec(), {"value": 2})

    async with use_run_context(first):
        assert current_run_context() is first
        first.state["marker"] = "first"
        async with use_run_context(second):
            assert current_run_context() is second
            assert "marker" not in second.state
        assert current_run_context() is first

    assert current_run_context() is None
    assert first.run_id != second.run_id


class RecordingSink:
    strict = True

    def __init__(
        self,
        *,
        fail_flush: bool = False,
        fail_close: bool = False,
        fail_emit: RunEventKind | None = None,
    ) -> None:
        self.events = []
        self.fail_flush = fail_flush
        self.fail_close = fail_close
        self.fail_emit = fail_emit
        self.closed = False

    async def emit(self, event) -> None:
        if event.kind is self.fail_emit:
            raise OSError("emit failed")
        self.events.append(event)

    async def flush(self, run_id: str) -> None:
        if self.fail_flush:
            raise OSError("flush failed")

    async def close(self, run_id: str, terminal_event=None) -> None:
        if self.fail_close:
            raise OSError("close failed")
        if terminal_event is not None and terminal_event.kind is self.fail_emit:
            raise OSError("emit failed")
        if terminal_event is not None:
            self.events.append(terminal_event)
        self.closed = True


@pytest.mark.asyncio
async def test_strict_evidence_requires_session_finalization_before_success():
    sink = RecordingSink()
    ctx = RunContext(make_spec(events=(sink,)), {})
    recorder = EvidenceRecorder(ctx, (sink,))
    await recorder.emit(RunEventKind.RUN_STARTED)
    await recorder.emit(
        RunEventKind.SESSION_STARTED,
        backend="stub",
        ownership="owned",
    )

    with pytest.raises(EvidenceIncompleteError, match="finalization"):
        await recorder.finish_success()

    assert not ctx.evidence_complete
    assert sink.closed


@pytest.mark.asyncio
async def test_strict_evidence_flush_failure_prevents_success():
    sink = RecordingSink(fail_flush=True)
    ctx = RunContext(make_spec(events=(sink,)), {})
    recorder = EvidenceRecorder(ctx, (sink,))
    await recorder.emit(RunEventKind.RUN_STARTED)
    await recorder.emit(RunEventKind.SESSION_STARTED)
    await recorder.emit(RunEventKind.SESSION_FINALIZED)
    await recorder.emit(RunEventKind.SESSION_RELEASED)

    with pytest.raises(EvidenceIncompleteError, match="flush"):
        await recorder.finish_success()

    assert not ctx.evidence_complete
    assert sink.closed
    assert all(event.kind is not RunEventKind.RUN_SUCCEEDED for event in sink.events)


@pytest.mark.asyncio
async def test_strict_evidence_close_failure_cannot_publish_success():
    sink = RecordingSink(fail_close=True)
    ctx = RunContext(make_spec(events=(sink,)), {})
    recorder = EvidenceRecorder(ctx, (sink,))
    await recorder.emit(RunEventKind.RUN_STARTED)

    with pytest.raises(EvidenceIncompleteError, match="close"):
        await recorder.finish_success()

    assert all(event.kind is not RunEventKind.RUN_SUCCEEDED for event in sink.events)
    assert not ctx.evidence_complete


@pytest.mark.asyncio
async def test_evidence_failure_does_not_replace_primary_failure():
    sink = RecordingSink(fail_flush=True)
    ctx = RunContext(make_spec(events=(sink,)), {})
    recorder = EvidenceRecorder(ctx, (sink,))
    primary = RuntimeError("primary")

    await recorder.emit(RunEventKind.RUN_STARTED)
    await recorder.finish_failure(primary)

    assert ctx.terminal_outcome == "error"
    assert not ctx.evidence_complete


@pytest.mark.asyncio
async def test_terminal_evidence_emit_failure_still_closes_sink():
    sink = RecordingSink(fail_emit=RunEventKind.RUN_SUCCEEDED)
    ctx = RunContext(make_spec(events=(sink,)), {})
    recorder = EvidenceRecorder(ctx, (sink,))
    await recorder.emit(RunEventKind.RUN_STARTED)

    with pytest.raises(EvidenceIncompleteError, match="emit failed"):
        await recorder.finish_success()

    assert sink.closed
    assert ctx.terminal_outcome == "error"


@pytest.mark.asyncio
async def test_strict_evidence_rejects_lossy_event_serialization():
    sink = RecordingSink()
    ctx = RunContext(make_spec(events=(sink,)), {})
    recorder = EvidenceRecorder(ctx, (sink,))

    with pytest.raises(EvidenceIncompleteError, match="serializ"):
        await recorder.emit(RunEventKind.RUN_STARTED, unsupported=object())

    assert sink.events == []


@pytest.mark.asyncio
async def test_code_cancellation_emits_paired_terminal_evidence():
    from predict_rlm import PredictRLM

    started = asyncio.Event()
    sink = RecordingSink()

    class BlockingSession(FinalSession):
        async def run_code(self, code, variables=None, timeout=None):
            started.set()
            await asyncio.Future()

    backend = FinalBackend()
    backend.session = BlockingSession()
    rlm = PredictRLM(
        "question: str -> answer: str",
        lm=MagicMock(history=[]),
        execution=backend,
        events=[sink],
        max_iterations=1,
        verbose=False,
    )
    rlm.generate_action.acall = AsyncMock(
        return_value=dspy.Prediction(reasoning="block", code="await block()")
    )

    task = asyncio.create_task(rlm.aforward(question="test"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as raised:
        await task

    events = sink.events
    generated = next(event for event in events if event.kind is RunEventKind.CODE_GENERATED)
    executed = next(event for event in events if event.kind is RunEventKind.CODE_EXECUTED)
    assert generated.data["operation_id"] == executed.data["operation_id"]
    assert executed.data["cancelled"] is True
    assert events[-1].kind is RunEventKind.RUN_CANCELLED
    assert raised.value.trace.evidence.complete


@pytest.mark.asyncio
async def test_predict_cancellation_emits_paired_terminal_evidence(monkeypatch):
    from predict_rlm import PredictRLM

    started = asyncio.Event()
    sink = RecordingSink()
    rlm = PredictRLM(
        "question: str -> answer: str",
        lm=MagicMock(history=[]),
        sub_lm=MagicMock(history=[]),
        execution=FinalBackend(),
        events=(sink,),
        max_iterations=1,
        verbose=False,
    )
    ctx = rlm._new_run_context({"question": "test"})
    recorder = EvidenceRecorder(ctx, (sink,))
    ctx.state["evidence"] = recorder

    async def block_predict(self, **kwargs):
        started.set()
        await asyncio.Future()

    monkeypatch.setattr(dspy.Predict, "acall", block_predict)
    predict = rlm._create_predict_tool()

    async with use_run_context(ctx):
        task = asyncio.create_task(predict("value: str -> answer: str", value="x"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    started_event = next(
        event for event in sink.events if event.kind is RunEventKind.PREDICT_STARTED
    )
    finished_event = next(
        event for event in sink.events if event.kind is RunEventKind.PREDICT_FINISHED
    )
    assert started_event.data["call_id"] == finished_event.data["call_id"]
    assert finished_event.data["cancelled"] is True


class FileInterpreter:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.shutdown_calls = 0
        self.direct_mounts = []

    def configure_direct_workspace_mounts(self, mounts) -> None:
        self.direct_mounts = list(mounts)

    def _path(self, sandbox_path: str) -> Path:
        return self.root.joinpath(*Path(sandbox_path).parts[1:])

    def execute(self, code: str, variables=None, timeout=None) -> str:
        return f"{code}:{variables['value']}"

    def mount_file_at(self, host_path: str, sandbox_path: str) -> None:
        target = self._path(sandbox_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(host_path, target)

    def mkdir_p(self, sandbox_path: str) -> None:
        self._path(sandbox_path).mkdir(parents=True, exist_ok=True)

    def list_dir(self, sandbox_path: str) -> list[str]:
        root = self._path(sandbox_path)
        return [
            f"{sandbox_path.rstrip('/')}/{path.relative_to(root).as_posix()}"
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ]

    def sync_file_to(self, sandbox_path: str, host_path: str) -> None:
        target = Path(host_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._path(sandbox_path), target)

    def shutdown(self) -> None:
        self.shutdown_calls += 1


@pytest.mark.asyncio
async def test_interpreter_session_round_trips_directory_artifacts(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "nested").mkdir()
    (source / "nested" / "value.txt").write_text("value", encoding="utf-8")
    interpreter = FileInterpreter(tmp_path / "sandbox")
    session = InterpreterExecutionSession(
        interpreter,
        name="test",
        ownership=SessionOwnership.INJECTED,
    )
    input_artifact = Artifact(
        id="directory-input",
        kind="opaque",
        metadata={
            "source_path": str(source),
            "sandbox_path": "/sandbox/input/source",
        },
    )

    binding = await session.mount(input_artifact)
    output_dir = tmp_path / "collected"
    collected = await session.collect(
        Artifact(
            id="directory-output",
            kind="opaque",
            metadata={
                "sandbox_path": binding.path,
                "destination_path": str(output_dir),
                "directory": True,
            },
        )
    )

    assert collected == str(output_dir)
    assert (output_dir / "nested" / "value.txt").read_text(encoding="utf-8") == "value"


@pytest.mark.asyncio
async def test_interpreter_backend_adapter_preserves_injected_ownership(tmp_path: Path):
    interpreter = FileInterpreter(tmp_path / "sandbox")
    exits = 0

    @contextmanager
    def acquire(spec: ExecutionSpec, ctx: RunContext):
        nonlocal exits
        yield interpreter
        exits += 1

    backend = InterpreterBackendAdapter(
        "injected",
        acquire,
        ownership=SessionOwnership.INJECTED,
        supports_host_directory_mounts=True,
    )
    spec = make_spec()
    ctx = RunContext(spec, {})

    async with backend.start(ExecutionSpec(), ctx) as session:
        result = await session.run_code("code", {"value": 3})
        assert result.value == "code:3"

    assert exits == 1
    assert interpreter.shutdown_calls == 0


@pytest.mark.asyncio
async def test_injected_backend_cancellation_does_not_leak_waiting_lock(tmp_path: Path):
    interpreter = FileInterpreter(tmp_path / "sandbox")

    @contextmanager
    def acquire(spec: ExecutionSpec, ctx: RunContext):
        yield interpreter

    backend = InterpreterBackendAdapter(
        "injected",
        acquire,
        ownership=SessionOwnership.INJECTED,
        supports_host_directory_mounts=True,
    )
    first_ctx = RunContext(make_spec(), {})
    waiting_ctx = RunContext(make_spec(), {})

    async def wait_for_session() -> None:
        async with backend.start(ExecutionSpec(), waiting_ctx):
            pass

    async with backend.start(ExecutionSpec(), first_ctx):
        waiting = asyncio.create_task(wait_for_session())
        await asyncio.sleep(0.05)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting

    await asyncio.sleep(0.05)
    assert backend._invocation_lock is not None
    assert backend._invocation_lock.acquire(blocking=False)
    backend._invocation_lock.release()


@pytest.mark.asyncio
async def test_injected_backend_releases_lock_when_acquisition_raises():
    def acquire(spec: ExecutionSpec, ctx: RunContext):
        raise RuntimeError("acquire failed")

    backend = InterpreterBackendAdapter(
        "injected",
        acquire,
        ownership=SessionOwnership.INJECTED,
        supports_host_directory_mounts=True,
    )

    with pytest.raises(RuntimeError, match="acquire failed"):
        async with backend.start(ExecutionSpec(), RunContext(make_spec(), {})):
            pass

    assert backend._invocation_lock is not None
    assert backend._invocation_lock.acquire(blocking=False)
    backend._invocation_lock.release()


@pytest.mark.asyncio
async def test_failed_injected_acquisition_does_not_commit_mount_set(
    tmp_path: Path,
):
    from predict_rlm.runtime import HostDirectoryMount

    attempts = 0

    @contextmanager
    def acquire(spec: ExecutionSpec, ctx: RunContext):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("acquire failed")
        yield FileInterpreter(tmp_path / "sandbox")

    backend = InterpreterBackendAdapter(
        "injected",
        acquire,
        ownership=SessionOwnership.INJECTED,
        supports_host_directory_mounts=True,
    )

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    first_ctx = RunContext(make_spec(), {})
    first_mount = HostDirectoryMount(str(first), "/workspace")
    with pytest.raises(RuntimeError, match="acquire failed"):
        async with backend.start(
            ExecutionSpec(host_directory_mounts=(first_mount,)), first_ctx
        ):
            pass

    second_ctx = RunContext(make_spec(), {})
    second_mount = HostDirectoryMount(str(second), "/workspace")
    async with backend.start(
        ExecutionSpec(host_directory_mounts=(second_mount,)), second_ctx
    ):
        pass

    assert attempts == 2


@pytest.mark.asyncio
async def test_compatibility_pool_rejects_host_mounts_before_acquisition(tmp_path: Path):
    acquisitions = 0

    @contextmanager
    def acquire(spec, ctx):
        nonlocal acquisitions
        acquisitions += 1
        yield FileInterpreter(tmp_path / "sandbox")

    backend = InterpreterBackendAdapter(
        "pooled",
        acquire,
        ownership=SessionOwnership.POOLED,
    )
    ctx = RunContext(make_spec(), {})
    mount = HostDirectoryMount(str(tmp_path), "/workspace")

    with pytest.raises(RuntimeError, match="pooled interpreters"):
        async with backend.start(
            ExecutionSpec(host_directory_mounts=(mount,)), ctx
        ):
            pass

    assert acquisitions == 0


@pytest.mark.asyncio
async def test_unsupported_injected_mount_is_rejected_before_acquisition(tmp_path: Path):
    acquisitions = 0

    @contextmanager
    def acquire(spec, ctx):
        nonlocal acquisitions
        acquisitions += 1
        yield FileInterpreter(tmp_path / "sandbox")

    backend = InterpreterBackendAdapter(
        "injected",
        acquire,
        ownership=SessionOwnership.INJECTED,
    )
    mount = HostDirectoryMount(str(tmp_path), "/workspace")

    with pytest.raises(RuntimeError, match="does not support host directory mounts"):
        async with backend.start(
            ExecutionSpec(host_directory_mounts=(mount,)),
            RunContext(make_spec(), {}),
        ):
            pass

    assert acquisitions == 0


@pytest.mark.asyncio
async def test_backend_without_mount_capability_rejects_before_start(tmp_path: Path):
    from predict_rlm.backends.adapters import ExistingExecutionBackendAdapter

    backend = FinalBackend()
    wrapped = ExistingExecutionBackendAdapter(backend)

    with pytest.raises(RuntimeError, match="does not support host directory mounts"):
        await wrapped.validate_host_directory_mounts(
            (HostDirectoryMount(str(tmp_path), "/workspace"),),
            RunContext(make_spec(), {}),
        )

    assert backend.spec is None


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["add", "removal", "access"])
async def test_injected_backend_rejects_semantic_mount_aggregate_changes(
    tmp_path: Path,
    change: str,
):
    acquisitions = 0
    interpreter = FileInterpreter(tmp_path / "sandbox")

    @contextmanager
    def acquire(spec, ctx):
        nonlocal acquisitions
        acquisitions += 1
        yield interpreter

    backend = InterpreterBackendAdapter(
        "injected",
        acquire,
        ownership=SessionOwnership.INJECTED,
        supports_host_directory_mounts=True,
    )
    first = HostDirectoryMount(str(tmp_path / "first"), "/first")
    second = HostDirectoryMount(str(tmp_path / "second"), "/second")
    if change == "add":
        initial = ()
        changed = (first,)
    elif change == "removal":
        initial = (first, second)
        changed = (first,)
    else:
        initial = (first,)
        changed = (
            HostDirectoryMount(first.host_path, first.sandbox_path, read_only=True),
        )

    async with backend.start(
        ExecutionSpec(host_directory_mounts=initial),
        RunContext(make_spec(), {}),
    ):
        pass

    with pytest.raises(ValueError, match="mount set"):
        async with backend.start(
            ExecutionSpec(host_directory_mounts=changed),
            RunContext(make_spec(), {}),
        ):
            pass

    assert acquisitions == 1


@pytest.mark.asyncio
async def test_injected_backend_accepts_semantically_reordered_mount_aggregate(
    tmp_path: Path,
):
    acquisitions = 0

    @contextmanager
    def acquire(spec, ctx):
        nonlocal acquisitions
        acquisitions += 1
        yield FileInterpreter(tmp_path / "sandbox")

    backend = InterpreterBackendAdapter(
        "injected",
        acquire,
        ownership=SessionOwnership.INJECTED,
        supports_host_directory_mounts=True,
    )
    first = HostDirectoryMount(str(tmp_path / "first"), "/first")
    second = HostDirectoryMount(str(tmp_path / "second"), "/second")

    async with backend.start(
        ExecutionSpec(host_directory_mounts=(first, second)),
        RunContext(make_spec(), {}),
    ):
        pass
    async with backend.start(
        ExecutionSpec(host_directory_mounts=(second, first)),
        RunContext(make_spec(), {}),
    ):
        pass

    assert acquisitions == 2


@pytest.mark.asyncio
async def test_async_only_injected_mount_configuration_is_rejected_before_acquisition(
    tmp_path: Path,
):
    from predict_rlm.compatibility.backends import execution_from_options

    acquisitions = 0

    class AsyncOnlyInterpreter:
        async def aconfigure_direct_workspace_mounts(self, mounts):
            raise AssertionError("sync compatibility adapter cannot call this hook")

    @contextmanager
    def acquire(spec, ctx):
        nonlocal acquisitions
        acquisitions += 1
        yield AsyncOnlyInterpreter()

    owner = MagicMock()
    owner._acquire_runtime_interpreter = acquire
    backend = execution_from_options(
        owner=owner,
        interpreter=AsyncOnlyInterpreter(),
        sandbox_backend=MagicMock(value="injected"),
        sbx_config=None,
        sbx_pool=None,
        allowed_domains=None,
        runtime_hooks=[],
        on_runtime_hook_event=None,
    )
    mount = HostDirectoryMount(str(tmp_path), "/workspace")

    with pytest.raises(RuntimeError, match="does not support host directory mounts"):
        async with backend.start(
            ExecutionSpec(host_directory_mounts=(mount,)),
            RunContext(make_spec(), {}),
        ):
            pass

    assert acquisitions == 0


@pytest.mark.asyncio
async def test_session_rejects_undeclared_host_directory_mount(tmp_path: Path):
    session = InterpreterExecutionSession(
        FileInterpreter(tmp_path / "sandbox"),
        name="injected",
        ownership=SessionOwnership.INJECTED,
    )

    with pytest.raises(RuntimeError, match="declared before backend acquisition"):
        await session.mount_host_directory(
            HostDirectoryMount(str(tmp_path), "/workspace")
        )


@pytest.mark.asyncio
async def test_backend_acquisition_receives_body_exception(tmp_path: Path):
    interpreter = FileInterpreter(tmp_path / "sandbox")
    exit_args = None

    class Manager:
        def __enter__(self):
            return interpreter

        def __exit__(self, exc_type, exc, traceback):
            nonlocal exit_args
            exit_args = (exc_type, exc, traceback)

    backend = InterpreterBackendAdapter(
        "owned",
        lambda spec, ctx: Manager(),
        ownership=SessionOwnership.OWNED,
    )

    with pytest.raises(ValueError, match="body failed"):
        async with backend.start(ExecutionSpec(), RunContext(make_spec(), {})):
            raise ValueError("body failed")

    assert exit_args is not None
    assert exit_args[0] is ValueError
    assert str(exit_args[1]) == "body failed"


@pytest.mark.asyncio
async def test_injected_backend_waits_for_cancelled_sync_execution_before_release():
    class BlockingInterpreter:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()
            self.finished = threading.Event()

        def execute(self, code, variables=None, timeout=None):
            self.started.set()
            self.release.wait()
            self.finished.set()
            return "finished"

    interpreter = BlockingInterpreter()
    finished_on_exit = False

    @contextmanager
    def acquire(spec: ExecutionSpec, ctx: RunContext):
        nonlocal finished_on_exit
        try:
            yield interpreter
        finally:
            finished_on_exit = interpreter.finished.is_set()

    backend = InterpreterBackendAdapter(
        "injected",
        acquire,
        ownership=SessionOwnership.INJECTED,
        supports_host_directory_mounts=True,
    )
    async with backend.start(ExecutionSpec(), RunContext(make_spec(), {})) as session:
        execution = asyncio.create_task(session.run_code("block"))
        await asyncio.to_thread(interpreter.started.wait)
        execution.cancel()
        await asyncio.sleep(0.05)
        assert not execution.done()

        interpreter.release.set()
        with pytest.raises(asyncio.CancelledError):
            await execution

    assert finished_on_exit
    assert interpreter.finished.is_set()


@pytest.mark.asyncio
async def test_jspi_cancellation_cancels_pending_host_calls():
    from predict_rlm.backends.jspi import JspiBackend

    started = asyncio.Event()
    cancelled = asyncio.Event()
    child_task = None

    class FakeJspi:
        _execute_async_loop = JspiBackend._execute_async_loop

        def __init__(self) -> None:
            self._active_tool_count = 0
            self._pending_file_ops = {}
            self.reads = 0

        async def _send_completed_responses(self, pending_tasks) -> None:
            return None

        async def _read_with_timeout_async(self, timeout):
            self.reads += 1
            if self.reads == 1:
                return (
                    '{"jsonrpc":"2.0","id":"tool-1","method":"tool_call",'
                    '"params":{"name":"slow","args":[],"kwargs":{}}}'
                )
            await asyncio.Event().wait()

        async def _execute_tool_async(self, name, params, request_id):
            nonlocal child_task
            child_task = asyncio.current_task()
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    fake = FakeJspi()
    execution = asyncio.create_task(JspiBackend._execute_async(fake, 1))
    await started.wait()
    execution.cancel()

    with pytest.raises(asyncio.CancelledError):
        await execution

    try:
        assert cancelled.is_set()
        assert fake._active_tool_count == 0
    finally:
        if child_task is not None and not child_task.done():
            child_task.cancel()
            await asyncio.gather(child_task, return_exceptions=True)


def test_backends_execution_backend_exports_final_session_contract():
    from predict_rlm.backends import ExecutionBackend

    assert hasattr(ExecutionBackend, "start")
    assert not hasattr(ExecutionBackend, "execute")


def test_maintained_backends_select_native_session_adapter():
    from predict_rlm import PredictRLM
    from predict_rlm.backends.jspi import JspiExecutionBackend

    lm = MagicMock()
    lm.copy.return_value = lm
    lm.history = []

    rlm = PredictRLM("question -> answer", lm=lm)

    assert isinstance(rlm.runtime_spec.execution, JspiExecutionBackend)


@pytest.mark.sbx
def test_maintained_sbx_backends_select_final_execution_ownership_seam():
    from predict_rlm import PredictRLM
    from predict_rlm.backends.sbx import (
        SbxExecutionBackend,
        SbxPool,
        SbxPoolExecutionBackend,
    )

    lm = MagicMock()
    lm.copy.return_value = lm
    lm.history = []

    owned = PredictRLM("question -> answer", lm=lm, sandbox_backend="sbx")
    pool = SbxPool(size=1, preinstall_packages=False)
    pooled = PredictRLM(
        "question -> answer",
        lm=lm,
        sandbox_backend="sbx",
        sbx_pool=pool,
    )

    assert isinstance(owned.runtime_spec.execution, SbxExecutionBackend)
    assert isinstance(pooled.runtime_spec.execution, SbxPoolExecutionBackend)


@pytest.mark.asyncio
async def test_input_adapter_after_execution_covers_generated_success_and_failure_only(
    tmp_path: Path,
):
    from predict_rlm import PredictRLM

    completed = []

    class LifecycleInputAdapter(InputAdapter[str]):
        name = "lifecycle"
        value_type = str

        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=value)

        async def after_execution(
            self,
            field,
            prepared,
            ctx,
            session,
            result,
            error,
        ):
            completed.append((field.name, result, error))

    class FailingRunSession(FinalSession):
        async def run_code(self, code, variables=None, timeout=None):
            if code == "raise RuntimeError":
                raise RuntimeError("execution failed")
            return await super().run_code(code, variables, timeout=timeout)

    backend = FinalBackend()
    backend.session = FailingRunSession()
    rlm = PredictRLM(
        "value: str -> answer: str",
        lm=MagicMock(history=[]),
        execution=backend,
        adapters=[LifecycleInputAdapter()],
    )
    module_path = tmp_path / "helper.py"
    module_path.write_text("VALUE = 1\n", encoding="utf-8")
    rlm._skill_modules = {"helper": str(module_path)}
    ctx = rlm._new_run_context({"value": "input"})

    async with use_run_context(ctx):
        await rlm._prepare_runtime_inputs(ctx, ctx.input_values)
        async with rlm._execution_session({}) as repl:
            await rlm._bind_runtime_inputs(ctx)
            await repl.aexecute("success")
            with pytest.raises(RuntimeError, match="execution failed"):
                await repl.aexecute("raise RuntimeError")
            await rlm._setup_runtime_modules(ctx)

    assert len(completed) == 2
    outcomes = [
        (result is not None, type(error) if error else None)
        for _, result, error in completed
    ]
    assert outcomes == [
        (True, None),
        (False, RuntimeError),
    ]


@pytest.mark.asyncio
async def test_cancelled_execution_defers_adapter_work_until_session_is_idle():
    from predict_rlm import PredictRLM

    started = asyncio.Event()
    after_calls = []
    finalize_live_states = []

    class LifecycleInputAdapter(InputAdapter[str]):
        name = "cancel-lifecycle"
        value_type = str

        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=value)

        async def after_execution(
            self,
            field,
            prepared,
            ctx,
            session,
            result,
            error,
        ):
            after_calls.append(session.live)

        async def finalize(self, field, prepared, ctx, session, error):
            finalize_live_states.append(session.live)

    class BlockingSession(FinalSession):
        def __init__(self):
            super().__init__()
            self.live = False

        async def run_code(self, code, variables=None, timeout=None):
            self.live = True
            started.set()
            await asyncio.Future()

        async def cancel(self):
            self.live = False
            await super().cancel()

    backend = FinalBackend()
    backend.session = BlockingSession()
    rlm = PredictRLM(
        "value: str -> answer: str",
        lm=MagicMock(history=[]),
        execution=backend,
        adapters=[LifecycleInputAdapter()],
    )
    ctx = rlm._new_run_context({"value": "input"})

    async def invoke():
        async with use_run_context(ctx):
            await rlm._prepare_runtime_inputs(ctx, ctx.input_values)
            async with rlm._execution_session({}) as repl:
                await rlm._bind_runtime_inputs(ctx)
                await repl.aexecute("block")

    invocation = asyncio.create_task(invoke())
    await started.wait()
    invocation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invocation

    assert after_calls == []
    assert finalize_live_states == [False]


@pytest.mark.asyncio
@pytest.mark.parametrize("execution_fails", [False, True])
async def test_input_adapter_after_execution_failure_is_fatal(execution_fails: bool):
    from predict_rlm import PredictRLM

    class DurabilityAdapter(InputAdapter[str]):
        name = "durability"
        value_type = str

        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=value)

        async def after_execution(
            self,
            field,
            prepared,
            ctx,
            session,
            result,
            error,
        ):
            raise OSError("remote flush failed")

    class ExecutionSession(FinalSession):
        async def run_code(self, code, variables=None, timeout=None):
            if execution_fails:
                raise ValueError("generated code failed")
            return await super().run_code(code, variables, timeout=timeout)

    backend = FinalBackend()
    backend.session = ExecutionSession()
    rlm = PredictRLM(
        "value: str -> answer: str",
        lm=MagicMock(history=[]),
        execution=backend,
        adapters=[DurabilityAdapter()],
        max_iterations=1,
        verbose=False,
    )
    rlm.generate_action.acall = AsyncMock(
        return_value=dspy.Prediction(reasoning="run", code="work")
    )

    expected = ValueError if execution_fails else OSError
    message = "generated code failed" if execution_fails else "remote flush failed"
    with pytest.raises(expected, match=message) as raised:
        await rlm.aforward(value="input")

    if execution_fails:
        assert isinstance(raised.value.input_adapter_after_execution_error, OSError)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_owner", ["adapter", "session"])
async def test_cancellation_remains_primary_when_finalization_fails(
    failure_owner: str,
):
    from predict_rlm import PredictRLM

    finalize_started = asyncio.Event()
    allow_finalize = asyncio.Event()

    class FailingAdapter(InputAdapter[str]):
        name = "failing-cancel-finalize"
        value_type = str

        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=value)

        async def finalize(self, field, prepared, ctx, session, error):
            if failure_owner != "adapter":
                return
            finalize_started.set()
            await allow_finalize.wait()
            raise OSError("adapter finalization failed")

    class FailingSession(FinalSession):
        async def finalize(self):
            if failure_owner != "session":
                return await super().finalize()
            finalize_started.set()
            await allow_finalize.wait()
            raise OSError("session finalization failed")

    backend = FinalBackend()
    backend.session = FailingSession()
    rlm = PredictRLM(
        "value: str -> answer: str",
        lm=MagicMock(history=[]),
        execution=backend,
        adapters=[FailingAdapter()],
        max_iterations=1,
        verbose=False,
    )
    rlm.generate_action.acall = AsyncMock(
        return_value=dspy.Prediction(reasoning="submit", code="SUBMIT(answer=value)")
    )

    invocation = asyncio.create_task(rlm.aforward(value="input"))
    await finalize_started.wait()
    invocation.cancel()
    allow_finalize.set()

    with pytest.raises(asyncio.CancelledError) as raised:
        await invocation

    attribute = (
        "input_adapter_finalize_error"
        if failure_owner == "adapter"
        else "session_finalize_error"
    )
    assert isinstance(getattr(raised.value, attribute), OSError)


@pytest.mark.asyncio
async def test_input_adapter_finalization_is_reverse_and_idempotent():
    from predict_rlm import PredictRLM

    calls = []

    class StringInputAdapter(InputAdapter[str]):
        name = "string-lifecycle"
        value_type = str

        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=value)

        async def finalize(self, field, prepared, ctx, session, error):
            calls.append(field.name)

    class IntegerInputAdapter(InputAdapter[int]):
        name = "integer-lifecycle"
        value_type = int

        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=value)

        async def finalize(self, field, prepared, ctx, session, error):
            calls.append(field.name)

    class RecordingFinalizeSession(FinalSession):
        async def finalize(self):
            calls.append("session")
            await super().finalize()

    backend = FinalBackend()
    backend.session = RecordingFinalizeSession()
    rlm = PredictRLM(
        "first: str, second: int -> answer: str",
        lm=MagicMock(history=[]),
        execution=backend,
        adapters=[StringInputAdapter(), IntegerInputAdapter()],
    )
    ctx = rlm._new_run_context({"first": "one", "second": 2})
    ctx.session = backend.session

    async with use_run_context(ctx):
        await rlm._prepare_runtime_inputs(ctx, ctx.input_values)
        await rlm._prepare_runtime_input_sessions(ctx)
        await rlm._bind_runtime_inputs(ctx)
        await rlm._finalize_runtime_inputs(ctx, backend.session, None)
        await rlm._finalize_runtime_inputs(ctx, backend.session, None)

    assert calls == ["second", "first", "session"]


class FinalSession:
    name = "final"
    ownership = SessionOwnership.OWNED

    def __init__(self) -> None:
        self.finalized = 0
        self.cancelled = 0
        self.mounted = []
        self.variables = None
        self.final_payload = None

    async def install_packages(self, packages) -> None:
        return None

    async def mount(self, artifact):
        self.mounted.append(artifact)
        return ArtifactBinding(
            artifact_id=artifact.id,
            path=artifact.metadata["sandbox_path"],
        )

    async def run_code(self, code, variables=None, timeout=None):
        from predict_rlm.runtime import ExecutionResult

        self.variables = variables
        payload = self.final_payload or {
            "answer": (variables or {}).get("question", "async-path")
        }
        return ExecutionResult(FinalOutput(payload))

    async def collect(self, artifact):
        destination = Path(artifact.metadata["destination_path"])
        if artifact.metadata.get("directory"):
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("generated", encoding="utf-8")
        return str(destination)

    async def finalize(self) -> None:
        self.finalized += 1

    async def cancel(self) -> None:
        self.cancelled += 1


class FinalBackend:
    name = "final"

    def __init__(self) -> None:
        self.session = FinalSession()
        self.spec = None

    @asynccontextmanager
    async def start(self, spec, ctx):
        self.spec = spec
        self.session.spec = spec
        yield self.session


class FailingExitBackend(FinalBackend):
    @asynccontextmanager
    async def start(self, spec, ctx):
        self.spec = spec
        self.session.spec = spec
        try:
            yield self.session
        finally:
            raise OSError("release failed")


@pytest.mark.asyncio
async def test_external_input_adapter_owns_session_lifecycle():
    from predict_rlm import PredictRLM
    from predict_rlm.runtime import HostDirectoryMount

    calls = []
    host_mount = HostDirectoryMount("/host/external", "/external")

    class LifecycleInputAdapter(InputAdapter[str]):
        name = "lifecycle"
        value_type = str

        async def prepare(self, field, value, ctx):
            calls.append("prepare")
            return PreparedInput(
                model_value="planned",
                artifacts=(Artifact(id="external", kind="external"),),
                host_directory_mounts=(host_mount,),
            )

        async def prepare_session(self, field, prepared, ctx, backend):
            calls.append(("prepare_session", field.name, backend.name))

        async def mount(self, field, prepared, ctx, session):
            calls.append("mount")
            path = await session.mount_host_directory(host_mount)
            return MountedInput(
                model_value="mounted",
                bindings=(ArtifactBinding(artifact_id="external", path=path),),
            )

        async def after_execution(
            self,
            field,
            prepared,
            ctx,
            session,
            result,
            error,
        ):
            calls.append(("after_execution", field.name, result.value, error))

        async def finalize(self, field, prepared, ctx, session, error):
            calls.append(("finalize", field.name, error))

    class RecordingBackend(FinalBackend):
        async def validate_host_directory_mounts(self, mounts, ctx):
            calls.append(("validate_host_directory_mounts", tuple(mounts)))

        @asynccontextmanager
        async def start(self, spec, ctx):
            calls.append("start")
            async with super().start(spec, ctx) as session:
                yield session
            calls.append("release")

    class RecordingHostMountSession(FinalSession):
        def __init__(self):
            super().__init__()
            self.host_directory_mounts = []

        async def mount_host_directory(self, mount):
            self.host_directory_mounts.append(mount)
            return mount.sandbox_path

    backend = RecordingBackend()
    backend.session = RecordingHostMountSession()
    rlm = PredictRLM(
        "value: str -> answer: str",
        lm=MagicMock(history=[]),
        execution=backend,
        adapters=[LifecycleInputAdapter()],
    )
    ctx = rlm._new_run_context({"value": "input"})

    async with use_run_context(ctx):
        await rlm._prepare_runtime_inputs(ctx, ctx.input_values)
        async with rlm._execution_session({}) as repl:
            await rlm._bind_runtime_inputs(ctx)
            assert ctx.input_bindings["value"].prepared.model_value == "mounted"
            await repl.aexecute("print('run')")

    assert calls[:5] == [
        "prepare",
        ("prepare_session", "value", "final"),
        ("validate_host_directory_mounts", (host_mount,)),
        "start",
        "mount",
    ]
    assert calls[5][0] == "after_execution"
    assert calls[6:] == [("finalize", "value", None), "release"]
    assert backend.session.host_directory_mounts == [host_mount]
    assert backend.session.mounted == []
    assert backend.session.finalized == 1


@pytest.mark.asyncio
async def test_session_finalizes_after_input_adapter_failure_in_reverse_order():
    from predict_rlm import PredictRLM

    calls = []

    class StringInputAdapter(InputAdapter[str]):
        name = "string-finalize"
        value_type = str

        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=value)

        async def finalize(self, field, prepared, ctx, session, error):
            calls.append(field.name)

    class IntegerInputAdapter(InputAdapter[int]):
        name = "integer-finalize"
        value_type = int

        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=value)

        async def finalize(self, field, prepared, ctx, session, error):
            calls.append(field.name)
            raise OSError("second finalize failed")

    class RecordingFinalizeSession(FinalSession):
        async def finalize(self):
            calls.append("session")
            await super().finalize()

    backend = FinalBackend()
    backend.session = RecordingFinalizeSession()
    rlm = PredictRLM(
        "first: str, second: int -> answer: str",
        lm=MagicMock(history=[]),
        execution=backend,
        adapters=[StringInputAdapter(), IntegerInputAdapter()],
    )
    ctx = rlm._new_run_context({"first": "one", "second": 2})

    with pytest.raises(OSError, match="second finalize failed"):
        async with use_run_context(ctx):
            await rlm._prepare_runtime_inputs(ctx, ctx.input_values)
            async with rlm._execution_session({}):
                await rlm._bind_runtime_inputs(ctx)

    assert calls == ["second", "first", "session"]
    assert backend.session.finalized == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_stage", "expected_fields", "session_available"),
    [
        ("prepare_session", ["second", "first"], False),
        ("acquisition", ["third", "second", "first"], False),
        ("install", ["third", "second", "first"], True),
        ("mount", ["third", "second", "first"], True),
    ],
)
async def test_all_prepare_session_adapters_finalize_after_startup_failure(
    failure_stage: str,
    expected_fields: list[str],
    session_available: bool,
):
    from predict_rlm import PredictRLM

    finalized = []
    entered = []

    class LifecycleAdapter(InputAdapter[object]):
        name = "startup-lifecycle"
        value_type = object

        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=value)

        async def prepare_session(self, field, prepared, ctx, backend):
            entered.append(field.name)
            if failure_stage == "prepare_session" and field.name == "second":
                raise RuntimeError("prepare session failed")

        async def mount(self, field, prepared, ctx, session):
            if failure_stage == "mount" and field.name == "second":
                raise RuntimeError("mount failed")
            return MountedInput(model_value=prepared.model_value)

        async def finalize(self, field, prepared, ctx, session, error):
            finalized.append((field.name, session is not None))

    class StartupSession(FinalSession):
        async def install_packages(self, packages):
            if failure_stage == "install":
                raise RuntimeError("install failed")

    class StartupBackend(FinalBackend):
        @asynccontextmanager
        async def start(self, spec, ctx):
            if failure_stage == "acquisition":
                raise RuntimeError("acquisition failed")
            async with super().start(spec, ctx) as session:
                yield session

    backend = StartupBackend()
    backend.session = StartupSession()
    rlm = PredictRLM(
        "first: str, second: int, third: bool -> answer: str",
        lm=MagicMock(history=[]),
        execution=backend,
        adapters=[LifecycleAdapter()],
    )
    ctx = rlm._new_run_context({"first": "one", "second": 2, "third": True})

    with pytest.raises(RuntimeError, match=failure_stage.replace("_", " ")):
        async with use_run_context(ctx):
            await rlm._prepare_runtime_inputs(ctx, ctx.input_values)
            async with rlm._execution_session({}):
                await rlm._bind_runtime_inputs(ctx)

    assert finalized == [
        (field_name, session_available) for field_name in expected_fields
    ]
    if failure_stage == "prepare_session":
        assert entered == ["first", "second"]


@pytest.mark.asyncio
async def test_pre_session_finalize_failure_is_recorded_as_incomplete_evidence():
    from predict_rlm import PredictRLM

    finalized = []

    class LifecycleAdapter(InputAdapter[object]):
        name = "pre-session-evidence"
        value_type = object

        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=value)

        async def prepare_session(self, field, prepared, ctx, backend):
            if field.name == "second":
                raise RuntimeError("prepare failed")

        async def finalize(self, field, prepared, ctx, session, error):
            finalized.append(field.name)
            if field.name == "first":
                raise OSError("abort cleanup failed")

    rlm = PredictRLM(
        "first: str, second: int -> answer: str",
        lm=MagicMock(history=[]),
        execution=FinalBackend(),
        adapters=[LifecycleAdapter()],
    )

    with pytest.raises(RuntimeError, match="prepare failed") as raised:
        await rlm.aforward(first="one", second=2)

    assert finalized == ["second", "first"]
    assert isinstance(raised.value.input_adapter_finalize_error, OSError)
    assert raised.value.trace.evidence.complete is False
    assert RunEventKind.SESSION_FINALIZE_FAILED in {
        event.kind for event in raised.value.trace.evidence.events
    }


@pytest.mark.asyncio
async def test_input_sandbox_root_collision_is_rejected_before_acquisition():
    from predict_rlm import PredictRLM, SandboxRootReservation

    acquisitions = 0

    class FirstAdapter(InputAdapter[str]):
        name = "first-root"
        value_type = str

        async def prepare(self, field, value, ctx):
            return PreparedInput(
                model_value=value,
                sandbox_roots=(SandboxRootReservation("/repository"),),
            )

    class SecondAdapter(InputAdapter[int]):
        name = "second-root"
        value_type = int

        async def prepare(self, field, value, ctx):
            return PreparedInput(
                model_value=value,
                sandbox_roots=(SandboxRootReservation("/repository/subdir"),),
            )

    class AcquisitionBackend(FinalBackend):
        @asynccontextmanager
        async def start(self, spec, ctx):
            nonlocal acquisitions
            acquisitions += 1
            async with super().start(spec, ctx) as session:
                yield session

    rlm = PredictRLM(
        "first: str, second: int -> answer: str",
        lm=MagicMock(history=[]),
        execution=AcquisitionBackend(),
        adapters=[FirstAdapter(), SecondAdapter()],
    )
    ctx = rlm._new_run_context({"first": "one", "second": 2})

    with pytest.raises(ValueError, match="sandbox destination.*overlap"):
        async with use_run_context(ctx):
            await rlm._prepare_runtime_inputs(ctx, ctx.input_values)
            async with rlm._execution_session({}):
                pass

    assert acquisitions == 0


@pytest.mark.asyncio
async def test_transfer_root_host_mount_collision_is_rejected_before_acquisition(
    tmp_path: Path,
):
    from predict_rlm import PredictRLM, SandboxRootReservation

    acquisitions = 0

    class TransferAdapter(InputAdapter[str]):
        name = "transfer-root"
        value_type = str

        async def prepare(self, field, value, ctx):
            return PreparedInput(
                model_value=value,
                sandbox_roots=(SandboxRootReservation("/repository/cache"),),
            )

    class MountAdapter(InputAdapter[int]):
        name = "host-mount"
        value_type = int

        async def prepare(self, field, value, ctx):
            return PreparedInput(
                model_value=value,
                host_directory_mounts=(
                    HostDirectoryMount(str(tmp_path), "/repository"),
                ),
            )

    class AcquisitionBackend(FinalBackend):
        @asynccontextmanager
        async def start(self, spec, ctx):
            nonlocal acquisitions
            acquisitions += 1
            async with super().start(spec, ctx) as session:
                yield session

    rlm = PredictRLM(
        "transfer: str, mount: int -> answer: str",
        lm=MagicMock(history=[]),
        execution=AcquisitionBackend(),
        adapters=[TransferAdapter(), MountAdapter()],
    )
    ctx = rlm._new_run_context({"transfer": "one", "mount": 2})

    with pytest.raises(ValueError, match="sandbox destination.*overlap"):
        async with use_run_context(ctx):
            await rlm._prepare_runtime_inputs(ctx, ctx.input_values)
            async with rlm._execution_session({}):
                pass

    assert acquisitions == 0


@pytest.mark.asyncio
async def test_interpreter_transfer_operations_require_declared_sandbox_root(
    tmp_path: Path,
):
    from predict_rlm import (
        FileTransfer,
        SandboxRootReservation,
        UnsupportedOperationError,
    )

    source = tmp_path / "source.txt"
    source.write_text("contents", encoding="utf-8")
    session = InterpreterExecutionSession(
        FileInterpreter(tmp_path / "sandbox"),
        name="injected",
        ownership=SessionOwnership.INJECTED,
        sandbox_roots=(SandboxRootReservation("/repository"),),
    )

    await session.create_directory("/repository")
    await session.transfer_file(
        FileTransfer(str(source), "/repository/source.txt")
    )

    with pytest.raises(UnsupportedOperationError, match="declared sandbox roots"):
        await session.create_directory("/other")
    with pytest.raises(UnsupportedOperationError, match="declared sandbox roots"):
        await session.transfer_file(FileTransfer(str(source), "/other/source.txt"))


@pytest.mark.asyncio
async def test_input_adapter_finalize_failure_preserves_primary_and_evidence():
    from predict_rlm import PredictRLM

    class FailingInputAdapter(InputAdapter[str]):
        name = "failing-finalize"
        value_type = str

        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=value)

        async def finalize(self, field, prepared, ctx, session, error):
            raise OSError("adapter finalize failed")

    backend = FinalBackend()
    rlm = PredictRLM(
        "question: str -> answer: str",
        lm=MagicMock(history=[]),
        execution=backend,
        adapters=[FailingInputAdapter()],
        max_iterations=1,
        verbose=False,
    )
    rlm.generate_action.acall = AsyncMock(side_effect=ValueError("primary failure"))

    with pytest.raises(ValueError, match="primary failure") as raised:
        await rlm.aforward(question="test")

    assert isinstance(raised.value.input_adapter_finalize_error, OSError)
    assert raised.value.trace.evidence.complete is False
    assert "session.finalize_failed" in {
        event.kind for event in raised.value.trace.evidence.events
    }


class BlockingFinalizeSession(FinalSession):
    def __init__(self) -> None:
        super().__init__()
        self.finalize_started = asyncio.Event()
        self.allow_finalize = asyncio.Event()

    async def finalize(self) -> None:
        self.finalize_started.set()
        await self.allow_finalize.wait()
        await super().finalize()


class BlockingFinalizeBackend(FinalBackend):
    def __init__(self) -> None:
        self.session = BlockingFinalizeSession()
        self.spec = None
        self.released = False

    @asynccontextmanager
    async def start(self, spec, ctx):
        self.spec = spec
        self.session.spec = spec
        try:
            yield self.session
        finally:
            self.released = True


class FailingFinalizeSession(FinalSession):
    async def finalize(self) -> None:
        raise OSError("finalization failed")


class FailingFinalizeBackend(FinalBackend):
    def __init__(self) -> None:
        self.session = FailingFinalizeSession()
        self.spec = None


class SyncedFinalSession(FinalSession):
    def __init__(self) -> None:
        super().__init__()
        self.files = {"/sandbox/work.txt": "before"}
        self.synced_mounts = []
        self.tool_name = "mutate"
        self.captured_host_path = None
        self.spec = None

    async def run_code(self, code, variables=None, timeout=None):
        tool = self.spec.tools[self.tool_name]
        await tool("/sandbox/work.txt")
        return await super().run_code(code, variables, timeout=timeout)

    async def collect(self, artifact):
        destination = Path(artifact.metadata["destination_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            self.files[artifact.metadata["sandbox_path"]],
            encoding="utf-8",
        )
        return str(destination)

    async def mount(self, artifact):
        if artifact.kind == "compat.file" and "destination_path" not in artifact.metadata:
            source = Path(artifact.metadata["source_path"])
            self.files[artifact.metadata["sandbox_path"]] = source.read_text(
                encoding="utf-8"
            )
            self.synced_mounts.append(artifact)
        return ArtifactBinding(
            artifact_id=artifact.id,
            path=artifact.metadata["sandbox_path"],
        )


class SyncedFinalBackend(FinalBackend):
    def __init__(self) -> None:
        self.session = SyncedFinalSession()
        self.spec = None


class MaintainedSyncedInterpreter:
    def __init__(self) -> None:
        self.files = {"/sandbox/work.txt": "before"}
        self.tools = {}
        self.shutdown_calls = 0

    async def aensure_skill_packages(self, packages) -> None:
        return None

    async def aexecute(self, code, variables=None, timeout=None):
        await self.tools["mutate"]("/sandbox/work.txt")
        return FinalOutput({"answer": "done"})

    async def async_file_to(self, sandbox_path: str, host_path: str) -> None:
        destination = Path(host_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.files[sandbox_path], encoding="utf-8")

    async def amount_file_at(self, host_path: str, sandbox_path: str) -> None:
        self.files[sandbox_path] = Path(host_path).read_text(encoding="utf-8")

    async def _sync_file_during_tool(self, sandbox_path: str, host_path: str) -> None:
        await self.async_file_to(sandbox_path, host_path)

    async def _mount_file_during_tool(self, host_path: str, sandbox_path: str) -> None:
        await self.amount_file_at(host_path, sandbox_path)

    async def ainterrupt(self) -> None:
        return None

    def retire_when_sync_workers_finish(self) -> bool:
        return False

    def retire_when_host_work_finishes(self) -> bool:
        return False

    async def ashutdown(self) -> None:
        self.shutdown_calls += 1


class MaintainedSyncedPool:
    def __init__(self, interpreter: MaintainedSyncedInterpreter) -> None:
        self._interpreter_kwargs = {}
        self.session_requirements = SessionRequirements()
        self.interpreter = interpreter
        self.released = False

    @asynccontextmanager
    async def alease(self, **kwargs):
        self.interpreter.tools = kwargs["tools"]
        try:
            yield self.interpreter
        finally:
            self.released = True


def test_predict_rlm_sync_entry_uses_async_session_contract():
    from predict_rlm import PredictRLM

    lm = MagicMock()
    lm.copy.return_value = lm
    lm.history = []
    backend = FinalBackend()
    rlm = PredictRLM(
        "question -> answer",
        lm=lm,
        execution=backend,
        max_iterations=1,
        verbose=False,
    )
    rlm.generate_action.acall = AsyncMock(
        return_value=dspy.Prediction(
            reasoning="submit",
            code="SUBMIT(answer='async-path')",
        )
    )
    rlm._build_signatures_with_files = MagicMock(
        return_value=(rlm.generate_action, rlm.extract)
    )

    result = rlm.forward(question="test")

    assert result.answer == "test"
    assert result.trace.status == "completed"
    assert backend.session.finalized >= 1


@pytest.mark.asyncio
async def test_failed_backend_exit_emits_release_failure_not_release_success():
    from predict_rlm import PredictRLM

    sink = RecordingSink()
    backend = FailingExitBackend()
    rlm = PredictRLM(
        "question: str -> answer: str",
        lm=MagicMock(history=[]),
        execution=backend,
        events=(sink,),
        max_iterations=1,
        verbose=False,
    )
    rlm.generate_action.acall = AsyncMock(
        return_value=dspy.Prediction(
            reasoning="submit",
            code="SUBMIT(answer=question)",
        )
    )

    with pytest.raises(OSError, match="release failed"):
        await rlm.aforward(question="test")

    kinds = [event.kind for event in sink.events]
    assert RunEventKind.SESSION_RELEASED not in kinds
    assert RunEventKind.SESSION_RELEASE_FAILED in kinds


@pytest.mark.asyncio
async def test_failed_backend_exit_does_not_mask_primary_execution_error():
    from predict_rlm import PredictRLM

    backend = FailingExitBackend()
    rlm = PredictRLM(
        "question: str -> answer: str",
        lm=MagicMock(history=[]),
        execution=backend,
        max_iterations=1,
        verbose=False,
    )
    rlm.generate_action.acall = AsyncMock(
        side_effect=ValueError("primary execution failure")
    )

    with pytest.raises(ValueError, match="primary execution failure") as raised:
        await rlm.aforward(question="test")

    assert isinstance(raised.value.session_release_error, OSError)


@pytest.mark.asyncio
async def test_cancellation_waits_for_owned_finalization_before_backend_release():
    from predict_rlm import PredictRLM

    backend = BlockingFinalizeBackend()
    rlm = PredictRLM(
        "question: str -> answer: str",
        lm=MagicMock(history=[]),
        execution=backend,
        max_iterations=1,
        verbose=False,
    )
    rlm.generate_action.acall = AsyncMock(
        return_value=dspy.Prediction(
            reasoning="submit",
            code="SUBMIT(answer=question)",
        )
    )

    invocation = asyncio.create_task(rlm.aforward(question="test"))
    await backend.session.finalize_started.wait()
    invocation.cancel()
    await asyncio.sleep(0)
    completed_before_finalize = invocation.done()
    released_before_finalize = backend.released
    backend.session.allow_finalize.set()
    with pytest.raises(asyncio.CancelledError):
        await invocation

    assert not completed_before_finalize
    assert not released_before_finalize
    assert backend.session.finalized == 1
    assert backend.released


@pytest.mark.asyncio
async def test_finalization_failure_preserves_primary_and_marks_evidence_incomplete():
    from predict_rlm import PredictRLM

    backend = FailingFinalizeBackend()
    rlm = PredictRLM(
        "question: str -> answer: str",
        lm=MagicMock(history=[]),
        execution=backend,
        max_iterations=1,
        verbose=False,
    )
    rlm.generate_action.acall = AsyncMock(side_effect=ValueError("primary failure"))

    with pytest.raises(ValueError, match="primary failure") as raised:
        await rlm.aforward(question="test")

    assert isinstance(raised.value.session_finalize_error, OSError)
    assert raised.value.trace.evidence.complete is False
    assert "session.finalize_failed" in {
        event.kind for event in raised.value.trace.evidence.events
    }


def test_run_trace_strict_evidence_reaches_rlm_gepa_consumer():
    from predict_rlm import PredictRLM
    from rlm_gepa.runtime.adapter import reflective_record
    from rlm_gepa.schema import RLMGepaExampleResult

    backend = FinalBackend()
    rlm = PredictRLM(
        "question: str -> answer: str",
        lm=MagicMock(history=[]),
        execution=backend,
        max_iterations=1,
        verbose=False,
    )
    rlm.generate_action.acall = AsyncMock(
        return_value=dspy.Prediction(
            reasoning="submit",
            code="SUBMIT(answer=question)",
        )
    )

    prediction = rlm.forward(question="test")
    result = RLMGepaExampleResult(
        score=1.0,
        feedback="ok",
        traces=[prediction.trace],
    )
    record = reflective_record(result)
    evidence = record["Traces"][0]["evidence"]
    kinds = [event["kind"] for event in evidence["events"]]

    assert evidence["complete"] is True
    assert kinds.index("session.finalized") < kinds.index("session.released")
    assert kinds.index("session.released") < kinds.index("run.succeeded")


def test_rlm_gepa_rejects_present_but_incomplete_strict_evidence():
    from predict_rlm.trace import RunEvidence, RunTrace
    from rlm_gepa.schema import RLMGepaExampleResult, validate_example_result

    trace = RunTrace(
        status="completed",
        model="test",
        iterations=1,
        max_iterations=1,
        duration_ms=1,
        evidence=RunEvidence(run_id="run", complete=False),
    )
    result = RLMGepaExampleResult(score=1.0, feedback="ok", traces=[trace])

    with pytest.raises(ValueError, match="incomplete strict evidence"):
        validate_example_result(result)


class TransformInputAdapter(InputAdapter[str]):
    name = "transform"
    value_type = str

    async def prepare(self, field, value, ctx):
        return PreparedInput(
            model_value=f"prepared:{value}",
            instructions=("Use the prepared value.",),
        )


def test_custom_contributions_drive_invocation_runtime():
    from predict_rlm import PredictRLM

    lm = MagicMock()
    lm.copy.return_value = lm
    lm.history = []
    backend = FinalBackend()
    module_calls = 0

    async def extra_tool(value: str) -> str:
        return value

    def module():
        nonlocal module_calls
        module_calls += 1
        return RuntimeContribution(
            instructions=("Module instruction.",),
            adapters=[TransformInputAdapter()],
            tools=(CallableTool(name="extra_tool", function=extra_tool),),
            packages=("module-package",),
        )

    rlm = PredictRLM(
        "question: str -> answer: str",
        lm=lm,
        execution=backend,
        modules=(module,),
        max_iterations=1,
        verbose=False,
    )
    rlm._build_signatures_with_files = MagicMock(
        return_value=(rlm.generate_action, rlm.extract)
    )
    rlm.generate_action.acall = AsyncMock(
        return_value=dspy.Prediction(
            reasoning="submit",
            code="SUBMIT(answer=question)",
        )
    )

    result = rlm.forward(question="value")

    assert module_calls == 1
    assert result.answer == "prepared:value"
    assert backend.spec.packages == ("module-package",)
    assert "extra_tool" in backend.spec.tools
    assert "Module instruction." in rlm.runtime_spec.instructions


def test_module_contributions_build_scalar_baseline_predictors():
    from predict_rlm import PredictRLM

    async def module_tool(value: str) -> str:
        """Return the module value unchanged."""
        return value

    backend = FinalBackend()
    rlm = PredictRLM(
        "question: str -> answer: str",
        lm=MagicMock(history=[]),
        execution=backend,
        modules=(
            RuntimeContribution(
                instructions=("Always use the module tool.",),
                tools=(
                    CallableTool(
                        name="module_tool",
                        function=module_tool,
                        description="Return the module value unchanged.",
                    ),
                ),
            ),
        ),
        max_iterations=1,
        verbose=False,
    )

    action_instructions = rlm.generate_action.signature.instructions
    extract_instructions = rlm.extract.signature.instructions

    assert "Always use the module tool." in action_instructions
    assert "Always use the module tool." in extract_instructions
    assert "module_tool" in action_instructions
    assert "module_tool" in extract_instructions


@pytest.mark.parametrize(
    "signature",
    [
        FileUnionInputSignature,
        NestedFileInputSignature,
        TupleFileInputSignature,
        SetFileInputSignature,
        DictFileInputSignature,
        SequenceFileInputSignature,
        NestedGenericFileInputSignature,
        WorkspaceUnionInputSignature,
        NestedWorkspaceInputSignature,
        TupleWorkspaceInputSignature,
        SetWorkspaceInputSignature,
        DictWorkspaceInputSignature,
        SequenceWorkspaceInputSignature,
        NestedGenericWorkspaceInputSignature,
    ],
)
def test_unsupported_compatibility_input_shapes_fail_before_backend_start(signature):
    from predict_rlm import PredictRLM

    backend = FinalBackend()

    with pytest.raises(ValueError, match="unsupported.*annotation|Unsupported.*annotation"):
        PredictRLM(signature, lm=MagicMock(history=[]), execution=backend)

    assert backend.spec is None


def test_workspace_output_fails_before_backend_start():
    from predict_rlm import PredictRLM

    class WorkspaceOutputSignature(dspy.Signature):
        question: str = dspy.InputField()
        workspace: Workspace = dspy.OutputField()

    backend = FinalBackend()

    with pytest.raises(ValueError, match="Workspace.*input-only"):
        PredictRLM(
            WorkspaceOutputSignature,
            lm=MagicMock(history=[]),
            execution=backend,
        )

    assert backend.spec is None


@pytest.mark.parametrize(
    "annotation",
    [
        tuple[RuntimeFile, ...],
        set[RuntimeFile],
        dict[str, RuntimeFile],
        Sequence[RuntimeFile],
        list[dict[str, tuple[RuntimeFile, ...]]],
    ],
)
def test_unsupported_file_generic_output_shapes_fail_before_backend_start(annotation):
    from predict_rlm import PredictRLM

    FileOutputSignature = type(
        "FileOutputSignature",
        (dspy.Signature,),
        {
            "__annotations__": {"question": str, "result": annotation},
            "question": dspy.InputField(),
            "result": dspy.OutputField(),
        },
    )
    backend = FinalBackend()

    with pytest.raises(ValueError, match="Unsupported File annotation"):
        PredictRLM(FileOutputSignature, lm=MagicMock(history=[]), execution=backend)

    assert backend.spec is None


@pytest.mark.parametrize(
    "annotation",
    [
        tuple[Workspace, ...],
        set[Workspace],
        dict[str, Workspace],
        Sequence[Workspace],
        list[dict[str, tuple[Workspace, ...]]],
    ],
)
def test_workspace_remains_input_only_inside_every_generic_output_shape(annotation):
    from predict_rlm import PredictRLM

    WorkspaceOutputSignature = type(
        "WorkspaceOutputSignature",
        (dspy.Signature,),
        {
            "__annotations__": {"question": str, "workspace": annotation},
            "question": dspy.InputField(),
            "workspace": dspy.OutputField(),
        },
    )

    backend = FinalBackend()

    with pytest.raises(ValueError, match="Workspace.*input-only"):
        PredictRLM(
            WorkspaceOutputSignature,
            lm=MagicMock(history=[]),
            execution=backend,
        )

    assert backend.spec is None


@pytest.mark.asyncio
async def test_injected_mount_set_change_is_rejected_before_reacquisition(
    tmp_path: Path,
):
    from predict_rlm.runtime import HostDirectoryMount

    acquisitions = 0

    interpreter = FileInterpreter(tmp_path / "sandbox")

    @contextmanager
    def acquire(spec, ctx):
        nonlocal acquisitions
        acquisitions += 1
        yield interpreter

    backend = InterpreterBackendAdapter(
        "injected",
        acquire,
        ownership=SessionOwnership.INJECTED,
        supports_host_directory_mounts=True,
    )

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    first_ctx = RunContext(make_spec(), {})
    first_mount = HostDirectoryMount(str(first), "/workspace")
    async with backend.start(
        ExecutionSpec(host_directory_mounts=(first_mount,)), first_ctx
    ):
        pass

    second_ctx = RunContext(make_spec(), {})
    second_mount = HostDirectoryMount(str(second), "/workspace")
    with pytest.raises(ValueError, match="mount set"):
        async with backend.start(
            ExecutionSpec(host_directory_mounts=(second_mount,)), second_ctx
        ):
            pass
    with pytest.raises(ValueError, match="mount set"):
        async with backend.start(ExecutionSpec(), second_ctx):
            pass

    assert acquisitions == 1


def test_output_adapter_ambiguity_fails_before_backend_start():
    from predict_rlm import PredictRLM

    class StringOutputAdapter(OutputAdapter[str]):
        value_type = str

        def __init__(self, name: str) -> None:
            self.name = name

        async def reserve(self, field, value, ctx, session):
            raise NotImplementedError

        async def materialize(self, reservation, submitted_value, ctx, session):
            raise NotImplementedError

    lm = MagicMock()
    lm.copy.return_value = lm
    lm.history = []
    backend = FinalBackend()

    with pytest.raises(ValueError, match="Multiple output adapters"):
        PredictRLM(
            "question: str -> answer: str",
            lm=lm,
            execution=backend,
            adapters=[StringOutputAdapter("first"), StringOutputAdapter("second")],
        )

    assert backend.spec is None


@pytest.mark.asyncio
async def test_kernel_cancels_and_finalizes_session_on_failure():
    from predict_rlm import PredictRLM

    lm = MagicMock()
    lm.copy.return_value = lm
    lm.history = []
    backend = FinalBackend()
    rlm = PredictRLM(
        "question -> answer",
        lm=lm,
        execution=backend,
        max_iterations=1,
        verbose=False,
    )
    rlm.generate_action.acall = AsyncMock(side_effect=RuntimeError("action failed"))
    rlm._build_signatures_with_files = MagicMock(
        return_value=(rlm.generate_action, rlm.extract)
    )

    with pytest.raises(RuntimeError, match="action failed"):
        await rlm.aforward(question="test")

    assert backend.session.cancelled == 1
    assert backend.session.finalized == 1


def test_explicit_backend_mounts_and_collects_file_artifacts(tmp_path: Path):
    from predict_rlm import File, PredictRLM

    source = tmp_path / "source.txt"
    source.write_text("source", encoding="utf-8")
    lm = MagicMock()
    lm.copy.return_value = lm
    lm.history = []
    backend = FinalBackend()
    backend.session.final_payload = {
        "result": "/sandbox/output/result/generated.txt"
    }
    rlm = PredictRLM(
        KernelFileSignature,
        lm=lm,
        execution=backend,
        output_dir=tmp_path / "outputs",
        max_iterations=1,
        verbose=False,
    )
    rlm._build_signatures_with_files = MagicMock(
        return_value=(rlm.generate_action, rlm.extract)
    )
    rlm._prepare_file_io = MagicMock(
        side_effect=AssertionError("the kernel must not build a legacy file plan")
    )
    rlm.generate_action.acall = AsyncMock(
        return_value=dspy.Prediction(
            reasoning="write",
            code="SUBMIT(result='/sandbox/output/result/generated.txt')",
        )
    )

    prediction = rlm.forward(source=File(path=str(source)))

    assert prediction.result.path.endswith("result/generated.txt")
    assert Path(prediction.result.path).read_text(encoding="utf-8") == "generated"
    assert len(backend.session.mounted) == 2
    assert {artifact.kind for artifact in backend.session.mounted} == {
        "compat.file",
        "compat.output.directory",
    }
    rlm._prepare_file_io.assert_not_called()


def test_synced_file_operation_is_portable_to_custom_final_backend():
    from predict_rlm import PredictRLM

    backend = SyncedFinalBackend()

    def mutate(path: Annotated[Path, SyncedFile()]) -> str:
        backend.session.captured_host_path = Path(path)
        assert backend.session.captured_host_path.read_text(encoding="utf-8") == "before"
        backend.session.captured_host_path.write_text("after", encoding="utf-8")
        return "mutated"

    rlm = PredictRLM(
        "question: str -> answer: str",
        lm=MagicMock(history=[]),
        execution=backend,
        tools={"mutate": mutate},
        max_iterations=1,
        verbose=False,
    )
    rlm.generate_action.acall = AsyncMock(
        return_value=dspy.Prediction(reasoning="mutate", code="mutate work")
    )

    result = rlm.forward(question="test")

    assert result.answer == "test"
    assert backend.session.files["/sandbox/work.txt"] == "after"
    assert len(backend.session.synced_mounts) == 1
    assert backend.session.captured_host_path is not None
    assert not backend.session.captured_host_path.exists()


@pytest.mark.parametrize(
    "backend_name",
    [
        "jspi",
        pytest.param("sbx", marks=pytest.mark.sbx),
        pytest.param("sbx-pool", marks=pytest.mark.sbx),
    ],
)
def test_synced_file_operation_runs_on_maintained_final_backend_lifecycles(
    backend_name: str,
    monkeypatch,
):
    from predict_rlm import PredictRLM
    from predict_rlm.backends.jspi import execution as jspi_execution

    interpreter = MaintainedSyncedInterpreter()

    def build_interpreter(**kwargs):
        interpreter.tools = kwargs["tools"]
        return interpreter

    kwargs = {}
    pool = None
    if backend_name == "jspi":
        monkeypatch.setattr(jspi_execution, "JspiBackend", build_interpreter)
    elif backend_name == "sbx":
        from predict_rlm.backends.sbx import execution as sbx_execution

        monkeypatch.setattr(sbx_execution, "SbxBackend", build_interpreter)
        kwargs["sandbox_backend"] = "sbx"
    else:
        pool = MaintainedSyncedPool(interpreter)
        kwargs.update(sandbox_backend="sbx", sbx_pool=pool)

    def mutate(path: Annotated[Path, SyncedFile()]) -> str:
        file_path = Path(path)
        file_path.write_text("after", encoding="utf-8")
        return "mutated"

    rlm = PredictRLM(
        "question: str -> answer: str",
        lm=MagicMock(history=[]),
        tools={"mutate": mutate},
        max_iterations=1,
        verbose=False,
        **kwargs,
    )
    rlm.generate_action.acall = AsyncMock(
        return_value=dspy.Prediction(reasoning="mutate", code="mutate work")
    )

    result = rlm.forward(question="test")

    assert result.answer == "done"
    assert interpreter.files["/sandbox/work.txt"] == "after"
    if pool is None:
        assert interpreter.shutdown_calls == 1
    else:
        assert pool.released


@pytest.mark.sbx
def test_sync_forward_awaits_owned_sbx_host_retirement_before_loop_teardown(
    monkeypatch,
):
    from predict_rlm import PredictRLM
    from predict_rlm.backends.sbx import execution as sbx_execution
    from predict_rlm.backends.sbx.backend import SbxBackend

    interpreter = SbxBackend.__new__(SbxBackend)
    interpreter._async_pending_tool_calls = {}
    interpreter._quarantined_async_tool_calls = set()
    interpreter._pending_tool_calls = {}
    interpreter._quarantined_tool_calls = set()
    interpreter._host_work_retirement = None
    cleanup_finished = threading.Event()
    shutdown_saw_cleanup = False

    async def stubborn_tool_task():
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            await asyncio.sleep(0)
            cleanup_finished.set()

    async def ensure_skill_packages(packages):
        return None

    async def execute(code, variables=None, timeout=None):
        task = asyncio.create_task(stubborn_tool_task())
        interpreter._async_pending_tool_calls[task] = 1
        await asyncio.sleep(0)
        return FinalOutput({"answer": "done"})

    async def shutdown():
        nonlocal shutdown_saw_cleanup
        shutdown_saw_cleanup = cleanup_finished.is_set()

    interpreter.aensure_skill_packages = ensure_skill_packages  # type: ignore[method-assign]
    interpreter.aexecute = execute  # type: ignore[method-assign]
    interpreter.ashutdown = shutdown  # type: ignore[method-assign]
    monkeypatch.setattr(sbx_execution, "SbxBackend", lambda **kwargs: interpreter)

    rlm = PredictRLM(
        "question: str -> answer: str",
        lm=MagicMock(history=[]),
        sandbox_backend="sbx",
        max_iterations=1,
        verbose=False,
    )
    rlm.generate_action.acall = AsyncMock(
        return_value=dspy.Prediction(reasoning="submit", code="SUBMIT(answer='done')")
    )

    result = rlm.forward(question="test")

    assert result.answer == "done"
    assert cleanup_finished.is_set()
    assert shutdown_saw_cleanup


def test_synced_file_operation_preserves_read_only_and_custom_host_dir(tmp_path: Path):
    from predict_rlm import PredictRLM

    backend = SyncedFinalBackend()
    backend.session.tool_name = "inspect_file"
    host_dir = tmp_path / "synced"

    def inspect_file(path: Path) -> str:
        assert Path(path).parent == host_dir
        Path(path).write_text("host-only", encoding="utf-8")
        return "inspected"

    inspect_file.__annotations__["path"] = Annotated[
        Path,
        SyncedFile(writeback=False, host_dir=str(host_dir)),
    ]

    rlm = PredictRLM(
        "question: str -> answer: str",
        lm=MagicMock(history=[]),
        execution=backend,
        tools={"inspect_file": inspect_file},
        max_iterations=1,
        verbose=False,
    )
    rlm.generate_action.acall = AsyncMock(
        return_value=dspy.Prediction(reasoning="inspect", code="inspect work")
    )

    rlm.forward(question="test")

    assert backend.session.files["/sandbox/work.txt"] == "before"
    assert backend.session.synced_mounts == []
    assert (host_dir / "work.txt").read_text(encoding="utf-8") == "host-only"
    assert str(host_dir) in backend.spec.extra_write_paths


def test_synced_file_operation_cleans_temporary_file_after_tool_failure():
    from predict_rlm import PredictRLM

    backend = SyncedFinalBackend()
    captured = None

    def fail(path: Annotated[Path, SyncedFile()]) -> str:
        nonlocal captured
        captured = Path(path)
        raise RuntimeError("tool failed")

    backend.session.tool_name = "fail"
    rlm = PredictRLM(
        "question: str -> answer: str",
        lm=MagicMock(history=[]),
        execution=backend,
        tools={"fail": fail},
        max_iterations=1,
        verbose=False,
    )
    rlm.generate_action.acall = AsyncMock(
        return_value=dspy.Prediction(reasoning="fail", code="fail work")
    )
    rlm.extract.acall = AsyncMock(return_value=dspy.Prediction(answer="fallback"))

    result = rlm.forward(question="test")

    assert result.answer == "fallback"
    assert captured is not None and not captured.exists()
    assert backend.session.synced_mounts == []


def test_sync_leaf_classification_survives_synced_file_and_evidence_wrappers():
    from predict_rlm import PredictRLM

    def inspect_file(path: Annotated[Path, SyncedFile()]) -> str:
        return Path(path).name

    synced = SyncedFileToolOperation().apply(
        CallableTool(name="inspect_file", function=inspect_file)
    ).function
    owner = type("EvidenceOwner", (), {"_evidence": lambda self: None})()
    wrapped = PredictRLM._wrap_evidence_tool(owner, "inspect_file", synced)

    assert callable_has_sync_leaf(synced)
    assert callable_has_sync_leaf(wrapped)


@pytest.mark.asyncio
async def test_sync_tool_cancellation_holds_custom_final_backend_lease_until_worker_stops():
    from predict_rlm import PredictRLM

    started = threading.Event()
    release = threading.Event()
    exited = False
    host_path = None
    path_survived_cancellation = False
    sink = RecordingSink()

    def block(path: Annotated[Path, SyncedFile(writeback=False)]) -> str:
        nonlocal host_path, path_survived_cancellation
        host_path = Path(path)
        started.set()
        release.wait()
        path_survived_cancellation = host_path.exists()
        return "done"

    class BlockingToolBackend(FinalBackend):
        def __init__(self) -> None:
            self.session = SyncedFinalSession()
            self.session.tool_name = "block"
            self.spec = None

        @asynccontextmanager
        async def start(self, spec, ctx):
            nonlocal exited
            self.spec = spec
            self.session.spec = spec
            try:
                yield self.session
            finally:
                exited = True

    backend = BlockingToolBackend()
    rlm = PredictRLM(
        "question: str -> answer: str",
        lm=MagicMock(history=[]),
        execution=backend,
        tools={"block": block},
        events=(sink,),
        max_iterations=1,
        verbose=False,
    )
    rlm.generate_action.acall = AsyncMock(
        return_value=dspy.Prediction(reasoning="block", code="block()")
    )

    invocation = asyncio.create_task(rlm.aforward(question="test"))
    await asyncio.to_thread(started.wait)
    invocation.cancel()
    await asyncio.sleep(0.05)
    assert not invocation.done()
    assert not exited
    assert host_path is not None and host_path.exists()

    release.set()
    with pytest.raises(asyncio.CancelledError) as raised:
        await invocation
    assert exited
    assert path_survived_cancellation
    assert host_path is not None and not host_path.exists()
    started_event = next(
        event for event in sink.events if event.kind is RunEventKind.TOOL_STARTED
    )
    finished_event = next(
        event for event in sink.events if event.kind is RunEventKind.TOOL_FINISHED
    )
    assert started_event.data["call_id"] == finished_event.data["call_id"]
    assert finished_event.data["cancelled"] is True
    assert raised.value.trace.evidence.complete


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_operation", ["collect", "writeback"])
async def test_synced_file_temp_survives_cancelled_legacy_io_worker(blocked_operation):
    started = threading.Event()
    release = threading.Event()
    captured_path: Path | None = None
    path_survived_worker = False

    class LegacyInterpreter:
        def sync_file_to(self, sandbox_path, host_path):
            nonlocal captured_path, path_survived_worker
            target = Path(host_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("before", encoding="utf-8")
            if blocked_operation == "collect":
                captured_path = target
                started.set()
                release.wait()
                path_survived_worker = target.exists()

        def mount_file_at(self, host_path, sandbox_path):
            nonlocal captured_path, path_survived_worker
            source = Path(host_path)
            if blocked_operation == "writeback":
                captured_path = source
                started.set()
                release.wait()
                path_survived_worker = source.exists()

    def mutate(path: Annotated[Path, SyncedFile()]) -> str:
        Path(path).write_text("after", encoding="utf-8")
        return "done"

    session = InterpreterExecutionSession(
        LegacyInterpreter(),
        name="legacy",
        ownership=SessionOwnership.INJECTED,
    )
    wrapped = SyncedFileToolOperation().apply(
        CallableTool(name="mutate", function=mutate)
    )
    ctx = RunContext(spec=make_spec(), input_values={})
    ctx.session = session

    async with use_run_context(ctx):
        invocation = asyncio.create_task(wrapped.function("/sandbox/work.txt"))
        await asyncio.to_thread(started.wait)
        invocation.cancel()
        await asyncio.sleep(0)
        completed_while_worker_live = invocation.done()
        path_removed_while_worker_live = captured_path is None or not captured_path.exists()
        release.set()
        try:
            await invocation
        except asyncio.CancelledError:
            pass
        await session.wait_for_idle()

    assert not completed_while_worker_live
    assert not path_removed_while_worker_live
    assert path_survived_worker
    assert captured_path is not None and not captured_path.exists()


def test_file_output_destination_is_not_exposed_as_sandbox_input(tmp_path: Path):
    from predict_rlm import File, PredictRLM

    source = tmp_path / "source.txt"
    source.write_text("source", encoding="utf-8")
    backend = FinalBackend()
    backend.session.final_payload = {
        "result": "/sandbox/output/result/generated.txt"
    }
    lm = MagicMock()
    lm.copy.return_value = lm
    lm.history = []
    rlm = PredictRLM(
        KernelFileSignature,
        lm=lm,
        execution=backend,
        max_iterations=1,
        verbose=False,
    )
    rlm._build_signatures_with_files = MagicMock(
        return_value=(rlm.generate_action, rlm.extract)
    )
    rlm.generate_action.acall = AsyncMock(
        return_value=dspy.Prediction(
            reasoning="write",
            code="SUBMIT(result='/sandbox/output/result/generated.txt')",
        )
    )

    rlm.forward(
        source=File(path=str(source)),
        result=File(path=str(tmp_path / "destination")),
    )

    assert "result" not in backend.session.variables
