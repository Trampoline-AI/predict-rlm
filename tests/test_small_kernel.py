from __future__ import annotations

import asyncio
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
    NativeInterpreterExecutionSession,
)
from predict_rlm.compatibility import SyncedFileToolOperation, WorkspaceInputAdapter
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
    FieldDescriptor,
    InputAdapter,
    OutputAdapter,
    PreparedInput,
    RunContext,
    RuntimeContribution,
    RuntimeSpec,
    SessionOwnership,
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
        inputs=(),
        outputs=(),
        tools=(),
        packages=(),
        execution=StubBackend(),
        events=events,
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
        events=(sink,),
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
    )

    with pytest.raises(RuntimeError, match="acquire failed"):
        async with backend.start(ExecutionSpec(), RunContext(make_spec(), {})):
            pass

    assert backend._invocation_lock is not None
    assert backend._invocation_lock.acquire(blocking=False)
    backend._invocation_lock.release()


@pytest.mark.asyncio
async def test_failed_injected_acquisition_does_not_pin_direct_workspace_state(
    tmp_path: Path,
):
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
        supports_direct_workspaces=True,
    )

    def context_for(path: Path) -> RunContext:
        ctx = RunContext(make_spec(), {})
        ctx.prepared_inputs["workspace"] = PreparedInput(
            model_value=str(path),
            artifacts=(
                Artifact(
                    id=path.name,
                    kind="compat.workspace.direct",
                    metadata={
                        "source_path": str(path),
                        "sandbox_path": str(path),
                        "workspace_binding": MagicMock(),
                    },
                ),
            ),
        )
        return ctx

    with pytest.raises(RuntimeError, match="acquire failed"):
        async with backend.start(ExecutionSpec(), context_for(tmp_path / "first")):
            pass

    async with backend.start(ExecutionSpec(), context_for(tmp_path / "second")):
        pass

    assert attempts == 2


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
async def test_native_session_mounts_workspace_without_sync_bridge(
    tmp_path: Path,
    monkeypatch,
):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    source = workspace_root / "source.txt"
    source.write_text("source", encoding="utf-8")

    class NativeInterpreter:
        def __init__(self) -> None:
            self.mounted = []
            self.hooks = []

        async def amkdir_p(self, path):
            self.mounted.append(("directory", path))

        async def amount_file_at(self, host_path, sandbox_path):
            self.mounted.append((host_path, sandbox_path))

        def add_post_execute_hook(self, hook):
            self.hooks.append(hook)

        def remove_post_execute_hook(self, hook):
            self.hooks.remove(hook)

    def reject_sync_bridge(*args, **kwargs):
        raise AssertionError("native sessions must not use asyncio.to_thread")

    monkeypatch.setattr(asyncio, "to_thread", reject_sync_bridge)
    interpreter = NativeInterpreter()
    session = NativeInterpreterExecutionSession(
        interpreter,
        name="native",
        ownership=SessionOwnership.OWNED,
    )
    ctx = RunContext(make_spec(), {})
    prepared = await WorkspaceInputAdapter().prepare(
        FieldDescriptor("workspace", Workspace),
        Workspace(path=str(workspace_root)),
        ctx,
    )

    binding = await session.mount(prepared.artifacts[0])

    assert binding.path == "/sandbox/workspace"
    assert (str(source), "/sandbox/workspace/source.txt") in interpreter.mounted
    assert len(interpreter.hooks) == 1
    assert asyncio.iscoroutinefunction(interpreter.hooks[0])

    await session.finalize()

    assert interpreter.hooks == []


@pytest.mark.asyncio
async def test_native_session_accumulates_direct_workspace_mounts(tmp_path: Path):
    configured = []

    class NativeInterpreter:
        async def aconfigure_direct_workspace_mounts(self, mounts):
            configured.append(list(mounts))

    session = NativeInterpreterExecutionSession(
        NativeInterpreter(),
        name="sbx",
        ownership=SessionOwnership.OWNED,
    )
    for index in range(2):
        source = tmp_path / f"workspace-{index}"
        source.mkdir()
        await session.mount(
            Artifact(
                id=f"workspace-{index}",
                kind="compat.workspace.direct",
                metadata={
                    "source_path": str(source),
                    "sandbox_path": f"/workspace-{index}",
                    "workspace_binding": MagicMock(),
                },
            )
        )

    assert [len(mounts) for mounts in configured] == [1, 2]


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
            tools=(CallableTool(name="extra_tool", function=extra_tool),),
            packages=("module-package",),
        )

    rlm = PredictRLM(
        "question: str -> answer: str",
        lm=lm,
        execution=backend,
        inputs=(TransformInputAdapter(),),
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
async def test_injected_direct_workspace_change_is_rejected_before_reacquisition(
    tmp_path: Path,
):
    acquisitions = 0

    class DirectInterpreter:
        def configure_direct_workspace_mounts(self, mounts):
            return None

    interpreter = DirectInterpreter()

    @contextmanager
    def acquire(spec, ctx):
        nonlocal acquisitions
        acquisitions += 1
        yield interpreter

    backend = InterpreterBackendAdapter(
        "injected",
        acquire,
        ownership=SessionOwnership.INJECTED,
    )

    def context_for(path: Path | None) -> RunContext:
        ctx = RunContext(make_spec(), {})
        if path is not None:
            ctx.prepared_inputs["workspace"] = PreparedInput(
                model_value=str(path),
                artifacts=(
                    Artifact(
                        id=f"workspace-{path.name}",
                        kind="compat.workspace.direct",
                        metadata={
                            "source_path": str(path),
                            "sandbox_path": str(path),
                            "workspace_binding": MagicMock(
                                host_path=str(path), sandbox_path=str(path)
                            ),
                        },
                    ),
                ),
            )
        return ctx

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    async with backend.start(ExecutionSpec(), context_for(first)):
        pass

    with pytest.raises(ValueError, match="direct Workspace.*sequential"):
        async with backend.start(ExecutionSpec(), context_for(second)):
            pass
    with pytest.raises(ValueError, match="direct Workspace.*sequential"):
        async with backend.start(ExecutionSpec(), context_for(None)):
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
            outputs=(StringOutputAdapter("first"), StringOutputAdapter("second")),
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
