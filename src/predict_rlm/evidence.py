"""Strict per-invocation evidence and best-effort observation sinks."""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .runtime import EventSink, RunContext, immutable_mapping
from .serialization import to_plain_data


class RunEventKind(str, Enum):
    RUN_STARTED = "run.started"
    INPUT_PREPARED = "input.prepared"
    SESSION_STARTED = "session.started"
    PACKAGES_INSTALLED = "packages.installed"
    ARTIFACT_MOUNTED = "artifact.mounted"
    OUTPUT_RESERVED = "output.reserved"
    CODE_GENERATED = "code.generated"
    CODE_EXECUTED = "code.executed"
    ITERATION_RECORDED = "iteration.recorded"
    PREDICT_STARTED = "predict.started"
    PREDICT_FINISHED = "predict.finished"
    TOOL_STARTED = "tool.started"
    TOOL_FINISHED = "tool.finished"
    ARTIFACT_COLLECTED = "artifact.collected"
    OUTPUT_MATERIALIZED = "output.materialized"
    SESSION_FINALIZED = "session.finalized"
    SESSION_FINALIZE_FAILED = "session.finalize_failed"
    SESSION_RELEASED = "session.released"
    SESSION_RELEASE_FAILED = "session.release_failed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"
    RUN_SUCCEEDED = "run.succeeded"


@dataclass(frozen=True)
class RunEvent:
    """Ordered evidence emitted by one invocation."""

    run_id: str
    sequence: int
    kind: RunEventKind
    timestamp_ns: int
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", immutable_mapping(self.data))


class EvidenceIncompleteError(RuntimeError):
    """Raised when correctness-critical evidence cannot be completed."""


class InMemoryEvidenceSink:
    """Concurrency-safe sink useful as the default strict evidence ledger."""

    strict = True

    def __init__(self) -> None:
        self.events: dict[str, list[RunEvent]] = {}

    async def emit(self, event: RunEvent) -> None:
        self.events.setdefault(event.run_id, []).append(event)

    async def flush(self, run_id: str) -> None:
        return None

    async def close(
        self,
        run_id: str,
        terminal_event: RunEvent | None = None,
    ) -> None:
        if terminal_event is not None:
            self.events.setdefault(run_id, []).append(terminal_event)


class EvidenceRecorder:
    """Serializes events and enforces strict sink completeness for one run."""

    def __init__(self, ctx: RunContext, sinks: tuple[EventSink, ...]) -> None:
        self.ctx = ctx
        self.sinks = sinks
        self.events: list[RunEvent] = []
        self._sequence = 0
        self._lock = asyncio.Lock()
        self._finalized = False
        self._session_started = False
        self._closed = False
        self._failure: BaseException | None = None

    @property
    def complete(self) -> bool:
        return self.ctx.evidence_complete and self._failure is None

    async def emit(self, kind: RunEventKind, **data: Any) -> RunEvent:
        async with self._lock:
            if self._closed:
                raise EvidenceIncompleteError("Cannot emit evidence after close")
            event = self._make_event(kind, data)
            self.events.append(event)
            if kind is RunEventKind.SESSION_FINALIZED:
                self._finalized = True
            elif kind is RunEventKind.SESSION_STARTED:
                self._session_started = True
            await self._fan_out("emit", event)
            return event

    async def finish_success(self, **data: Any) -> None:
        try:
            self._validate_terminal_state(success=True)
            terminal = self._make_event(RunEventKind.RUN_SUCCEEDED, data)
            await self._commit_terminal(terminal)
        except BaseException as exc:
            self.ctx.terminal_outcome = "error"
            failure = self._make_event(
                RunEventKind.RUN_FAILED,
                {"error_type": type(exc).__name__, "error": str(exc)},
            )
            if self._closed:
                await self._close_after_failed_commit(failure)
            else:
                try:
                    await self._commit_terminal(failure)
                except BaseException:
                    pass
            self.events.append(failure)
            raise
        self.events.append(terminal)
        self.ctx.terminal_outcome = "completed"
        self.require_complete()

    async def finish_failure(self, exc: BaseException) -> None:
        kind = (
            RunEventKind.RUN_CANCELLED
            if isinstance(exc, asyncio.CancelledError)
            else RunEventKind.RUN_FAILED
        )
        try:
            self._validate_terminal_state(success=False)
        except BaseException as evidence_error:
            self._attach_evidence_error(exc, evidence_error)
        terminal = self._make_event(
            kind,
            {"error_type": type(exc).__name__, "error": str(exc)},
        )
        self.ctx.terminal_outcome = (
            "cancelled" if kind is RunEventKind.RUN_CANCELLED else "error"
        )
        try:
            await self._commit_terminal(terminal)
        except BaseException as evidence_error:
            self._attach_evidence_error(exc, evidence_error)
        else:
            self.events.append(terminal)

    def require_complete(self) -> None:
        if self.complete:
            return
        error = EvidenceIncompleteError(
            f"Strict evidence is incomplete for run {self.ctx.run_id}"
        )
        if self._failure is not None:
            raise error from self._failure
        raise error

    def mark_incomplete(self, exc: BaseException) -> None:
        self._mark_incomplete(exc)

    async def _commit_terminal(self, terminal: RunEvent) -> None:
        if self._closed:
            return
        try:
            await self._fan_out("flush", self.ctx.run_id)
            await self._fan_out_terminal(terminal)
        finally:
            self._closed = True

    async def _fan_out_terminal(self, terminal: RunEvent) -> None:
        for sink in self.sinks:
            close = getattr(sink, "close", None)
            if close is None:
                if getattr(sink, "strict", False):
                    error = TypeError(
                        f"Strict evidence sink {type(sink).__name__} has no close()"
                    )
                    self._mark_incomplete(error)
                    raise EvidenceIncompleteError(str(error)) from error
                continue
            try:
                result = close(self.ctx.run_id, terminal)
                if inspect.isawaitable(result):
                    await result
            except BaseException as exc:
                if getattr(sink, "strict", False):
                    self._mark_incomplete(exc)
                    raise EvidenceIncompleteError(
                        f"Strict evidence sink {type(sink).__name__}.close failed: {exc}"
                    ) from exc

    async def _close_after_failed_commit(self, terminal: RunEvent) -> None:
        for sink in self.sinks:
            close = getattr(sink, "close", None)
            if close is None:
                continue
            try:
                result = close(self.ctx.run_id, terminal)
                if inspect.isawaitable(result):
                    await result
            except BaseException:
                continue

    async def _fan_out(self, method_name: str, value: Any) -> None:
        for sink in self.sinks:
            method = getattr(sink, method_name, None)
            if method is None:
                if getattr(sink, "strict", False):
                    error = TypeError(
                        f"Strict evidence sink {type(sink).__name__} has no {method_name}()"
                    )
                    self._mark_incomplete(error)
                    raise EvidenceIncompleteError(str(error)) from error
                continue
            try:
                result = method(value)
                if inspect.isawaitable(result):
                    await result
            except BaseException as exc:
                if getattr(sink, "strict", False):
                    self._mark_incomplete(exc)
                    raise EvidenceIncompleteError(
                        f"Strict evidence sink {type(sink).__name__}.{method_name} failed"
                    ) from exc

    def _mark_incomplete(self, exc: BaseException) -> None:
        self.ctx.evidence_complete = False
        if self._failure is None:
            self._failure = exc

    def _append_local(self, kind: RunEventKind, **data: Any) -> None:
        self.events.append(self._make_event(kind, data))

    def _make_event(self, kind: RunEventKind, data: dict[str, Any]) -> RunEvent:
        normalized = to_plain_data(data)
        try:
            json.dumps(normalized)
        except (TypeError, ValueError) as exc:
            error = EvidenceIncompleteError(
                f"Strict evidence serialization failed for {kind.value}"
            )
            self._mark_incomplete(exc)
            raise error from exc
        self._sequence += 1
        return RunEvent(
            run_id=self.ctx.run_id,
            sequence=self._sequence,
            kind=kind,
            timestamp_ns=time.time_ns(),
            data=normalized,
        )

    def _validate_terminal_state(self, *, success: bool) -> None:
        open_operations: dict[tuple[str, Any], RunEvent] = {}
        pairs = {
            RunEventKind.PREDICT_STARTED: ("predict", RunEventKind.PREDICT_FINISHED, "call_id"),
            RunEventKind.TOOL_STARTED: ("tool", RunEventKind.TOOL_FINISHED, "call_id"),
            RunEventKind.CODE_GENERATED: ("code", RunEventKind.CODE_EXECUTED, "operation_id"),
        }
        terminals = {terminal: (label, key) for _, (label, terminal, key) in pairs.items()}
        for event in self.events:
            if event.kind in pairs:
                label, _, key = pairs[event.kind]
                open_operations[(label, event.data.get(key))] = event
            elif event.kind in terminals:
                label, key = terminals[event.kind]
                open_operations.pop((label, event.data.get(key)), None)
        if open_operations:
            names = ", ".join(f"{name}:{operation_id}" for name, operation_id in open_operations)
            error = EvidenceIncompleteError(f"Unmatched strict evidence operations: {names}")
            self._mark_incomplete(error)
            raise error
        if not success or not self._session_started:
            return
        kinds = [event.kind for event in self.events]
        required = (RunEventKind.SESSION_FINALIZED, RunEventKind.SESSION_RELEASED)
        if any(kind not in kinds for kind in required):
            error = EvidenceIncompleteError(
                "Terminal success requires session finalization and release evidence"
            )
            self._mark_incomplete(error)
            raise error
        if kinds.index(RunEventKind.SESSION_FINALIZED) > kinds.index(
            RunEventKind.SESSION_RELEASED
        ):
            error = EvidenceIncompleteError("Session release preceded finalization evidence")
            self._mark_incomplete(error)
            raise error

    def _attach_evidence_error(
        self,
        primary: BaseException,
        evidence_error: BaseException,
    ) -> None:
        self._mark_incomplete(evidence_error)
        try:
            setattr(primary, "evidence_error", evidence_error)
        except BaseException:
            pass
