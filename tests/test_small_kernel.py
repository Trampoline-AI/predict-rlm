from __future__ import annotations

import asyncio
import shutil
import threading
from contextlib import asynccontextmanager, contextmanager
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock

import dspy
import pytest
from dspy.primitives.code_interpreter import FinalOutput
from pydantic import BaseModel

from predict_rlm.backends.adapters import (
    InterpreterBackendAdapter,
    InterpreterExecutionSession,
)
from predict_rlm.compatibility import SyncedFileToolOperation
from predict_rlm.evidence import (
    EvidenceIncompleteError,
    EvidenceRecorder,
    RunEventKind,
)
from predict_rlm.files import SyncedFile
from predict_rlm.runtime import (
    ArtifactBinding,
    BoundInput,
    CallableTool,
    ExecutionSpec,
    HostDirectoryMount,
    InputAdapter,
    PreparedInput,
    RunContext,
    RuntimeSpec,
    SessionOwnership,
    use_run_context,
)


def make_spec(*, events=()) -> RuntimeSpec:
    return RuntimeSpec(
        instructions=(),
        adapters=(),
        tools=(),
        packages=(),
        execution=FinalBackend(),
        events=events,
    )


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
async def test_strict_evidence_serializes_nested_pydantic_decimals():
    class FinancialEvidence(BaseModel):
        asking_price_cad: Decimal
        field_confidences: dict[str, Decimal]
        pac: date

    sink = RecordingSink()
    ctx = RunContext(make_spec(events=(sink,)), {})
    recorder = EvidenceRecorder(ctx, (sink,))

    await recorder.emit(
        RunEventKind.RUN_STARTED,
        evidence=FinancialEvidence(
            asking_price_cad=Decimal("2150000.00"),
            field_confidences={"unit_mix": Decimal("0.7880")},
            pac=date(2026, 10, 31),
        ),
    )

    assert sink.events[0].data["evidence"] == {
        "asking_price_cad": "2150000.00",
        "field_confidences": {"unit_mix": "0.7880"},
        "pac": "2026-10-31",
    }


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

    def shutdown(self) -> None:
        self.shutdown_calls += 1


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

    async def wait_for_session() -> str:
        async with backend.start(ExecutionSpec(), waiting_ctx) as session:
            return (await session.run_code("code", {"value": 3})).value

    async with backend.start(ExecutionSpec(), first_ctx):
        waiting = asyncio.create_task(wait_for_session())
        await asyncio.sleep(0.05)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting

    assert await asyncio.wait_for(wait_for_session(), timeout=2) == "code:3"
    assert interpreter.shutdown_calls == 0


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
    async with backend.start(ExecutionSpec(host_directory_mounts=(second_mount,)), second_ctx):
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
        async with backend.start(ExecutionSpec(host_directory_mounts=(mount,)), ctx):
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
        await session.mount_host_directory(HostDirectoryMount(str(tmp_path), "/workspace"))


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
        max_iterations=1,
        verbose=False,
    )
    rlm.generate_action.acall = AsyncMock(
        return_value=dspy.Prediction(reasoning="block", code="await block()")
    )

    invocation = asyncio.create_task(rlm.aforward(value="input"))
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


class FinalSession:
    name = "final"
    ownership = SessionOwnership.OWNED

    def __init__(self) -> None:
        self.finalized = 0
        self.cancelled = 0

    async def install_packages(self, packages) -> None:
        return None

    async def mount(self, artifact):
        return ArtifactBinding(
            artifact_id=artifact.id,
            path=artifact.metadata["sandbox_path"],
        )

    async def run_code(self, code, variables=None, timeout=None):
        from predict_rlm.runtime import ExecutionResult

        payload = {"answer": (variables or {}).get("question", "async-path")}
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
        ("open", ["second", "first"], False),
        ("bind", ["third", "second", "first"], True),
    ],
)
async def test_all_open_adapters_finalize_after_startup_failure(
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

        async def open(self, field, prepared, ctx, backend):
            entered.append(field.name)
            if failure_stage == "open" and field.name == "second":
                raise RuntimeError("open failed")

        async def bind(self, field, prepared, ctx, session):
            if failure_stage == "bind" and field.name == "second":
                raise RuntimeError("bind failed")
            return BoundInput(model_value=prepared.model_value)

        async def finalize(self, field, prepared, ctx, session, error):
            finalized.append((field.name, session is not None))

    backend = FinalBackend()
    rlm = PredictRLM(
        "first: str, second: int, third: bool -> answer: str",
        lm=MagicMock(history=[]),
        execution=backend,
        adapters=[LifecycleAdapter()],
    )

    with pytest.raises(RuntimeError, match=failure_stage):
        await rlm.aforward(first="one", second=2, third=True)

    assert finalized == [(field_name, session_available) for field_name in expected_fields]
    if failure_stage == "open":
        assert entered == ["first", "second"]


@pytest.mark.asyncio
async def test_pre_acquisition_finalize_failure_is_recorded_as_incomplete_evidence():
    from predict_rlm import PredictRLM

    finalized = []

    class LifecycleAdapter(InputAdapter[object]):
        name = "pre-acquisition-evidence"
        value_type = object

        async def prepare(self, field, value, ctx):
            return PreparedInput(model_value=value)

        async def open(self, field, prepared, ctx, backend):
            if field.name == "second":
                raise RuntimeError("open failed")

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

    with pytest.raises(RuntimeError, match="open failed") as raised:
        await rlm.aforward(first="one", second=2)

    assert finalized == ["second", "first"]
    assert isinstance(raised.value.input_adapter_finalize_error, OSError)
    assert raised.value.trace.evidence.complete is False
    assert RunEventKind.SESSION_FINALIZE_FAILED in {
        event.kind for event in raised.value.trace.evidence.events
    }


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
                host_directory_mounts=(HostDirectoryMount(str(tmp_path), "/repository"),),
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
    await session.transfer_file(FileTransfer(str(source), "/repository/source.txt"))

    with pytest.raises(UnsupportedOperationError, match="declared sandbox roots"):
        await session.create_directory("/other")
    with pytest.raises(UnsupportedOperationError, match="declared sandbox roots"):
        await session.transfer_file(FileTransfer(str(source), "/other/source.txt"))


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
    async def run_code(self, code, variables=None, timeout=None):
        tool = self.spec.tools["block"]
        await tool("/sandbox/work.txt")
        return await super().run_code(code, variables, timeout=timeout)

    async def collect(self, artifact):
        destination = Path(artifact.metadata["destination_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("before", encoding="utf-8")
        return str(destination)


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
    rlm.generate_action.acall = AsyncMock(side_effect=ValueError("primary execution failure"))

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
    async with backend.start(ExecutionSpec(host_directory_mounts=(first_mount,)), first_ctx):
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
    wrapped = SyncedFileToolOperation().apply(CallableTool(name="mutate", function=mutate))
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
