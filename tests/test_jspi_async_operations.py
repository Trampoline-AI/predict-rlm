from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from predict_rlm.backends.base import SandboxExecutionError, SandboxFatalError
from predict_rlm.backends.jspi import JspiBackend
from predict_rlm.backends.jspi.execution import JspiExecutionBackend
from predict_rlm.execution_timeout import ITERATION_TIMEOUT_FAILURE_CLASS
from predict_rlm.runtime import ExecutionSpec


@pytest.fixture
def interpreter() -> JspiBackend:
    backend = JspiBackend(preinstall_packages=False)
    return backend


@pytest.mark.asyncio
async def test_cancel_execution_retries_interrupt_until_execution_quiesces(
    interpreter: JspiBackend,
):
    interrupted = asyncio.Event()
    signals = 0

    def send_signal(signal_number):
        del signal_number
        nonlocal signals
        signals += 1
        if signals == 2:
            interrupted.set()

    async def finish_execution(request_id):
        assert request_id == 7
        await interrupted.wait()
        raise SandboxExecutionError("KeyboardInterrupt")

    interpreter.deno_process = SimpleNamespace(
        poll=lambda: None,
        send_signal=send_signal,
    )
    interpreter._active_execute_request_id = 7
    interpreter._execute_async = finish_execution  # type: ignore[method-assign]
    interpreter._akill_sandbox = AsyncMock()  # type: ignore[method-assign]

    await interpreter.acancel_execution()

    assert signals == 2
    interpreter._akill_sandbox.assert_not_awaited()


@pytest.mark.asyncio
async def test_aexecute_skips_post_hooks_after_fatal_failure(interpreter: JspiBackend):
    hook = AsyncMock()
    interpreter.add_post_execute_hook(hook)
    interpreter._aexecute_inner = AsyncMock(  # type: ignore[method-assign]
        side_effect=SandboxFatalError("fatal")
    )

    with pytest.raises(SandboxFatalError, match="fatal"):
        await interpreter.aexecute("raise SystemExit")

    hook.assert_not_awaited()


@pytest.mark.asyncio
async def test_aexecute_skips_post_hooks_after_cancellation(interpreter: JspiBackend):
    started = asyncio.Event()
    hook = AsyncMock()
    interpreter.add_post_execute_hook(hook)

    async def block(code, variables):
        started.set()
        await asyncio.Future()

    interpreter._aexecute_inner = block  # type: ignore[method-assign]
    task = asyncio.create_task(interpreter.aexecute("await work()"))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    hook.assert_not_awaited()


@pytest.mark.asyncio
async def test_aexecute_preserves_primary_error_when_post_hook_fails(
    interpreter: JspiBackend,
):
    async def fail_hook(_backend):
        raise OSError("sync failed")

    interpreter.add_post_execute_hook(fail_hook)
    interpreter._aexecute_inner = AsyncMock(  # type: ignore[method-assign]
        side_effect=ValueError("primary")
    )

    with pytest.raises(ValueError, match="primary") as raised:
        await interpreter.aexecute("bad code")

    assert isinstance(raised.value.post_execute_error, OSError)


@pytest.mark.asyncio
async def test_recoverable_timeout_owns_stubborn_async_tool_until_next_iteration():
    interpreter = JspiBackend.__new__(JspiBackend)
    interpreter._active_tool_count = 0
    interpreter._pending_file_ops = {}
    interpreter._quarantined_async_tool_calls = set()
    cancellation_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    next_iteration_started = asyncio.Event()
    tool_task: asyncio.Task | None = None
    sent_tool_call = False

    async def execute_tool(name, params, request_id):
        nonlocal tool_task
        tool_task = asyncio.current_task()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancellation_started.set()
            await release_cleanup.wait()
        return {"value": "done", "type": "string"}

    async def read_with_timeout(timeout):
        nonlocal sent_tool_call
        if not sent_tool_call:
            sent_tool_call = True
            return (
                '{"jsonrpc":"2.0","id":"tool-1","method":"tool_call",'
                '"params":{"name":"slow","args":[],"kwargs":{}}}'
            )
        if cancellation_started.is_set():
            next_iteration_started.set()
            return '{"jsonrpc":"2.0","id":2,"result":{"output":"next"}}'
        await asyncio.sleep(timeout)
        return None

    interpreter._execute_tool_async = execute_tool  # type: ignore[method-assign]
    interpreter._read_with_timeout_async = read_with_timeout  # type: ignore[method-assign]
    interpreter._send_completed_responses = AsyncMock()  # type: ignore[method-assign]
    interpreter._write_stdin_async = AsyncMock()  # type: ignore[method-assign]
    interpreter._wait_and_send_all_responses = AsyncMock()  # type: ignore[method-assign]
    interpreter._async_files = AsyncMock()  # type: ignore[method-assign]

    try:
        result = await interpreter._execute_async(
            1,
            timeout_seconds=0.01,
            timeout_failure_class=ITERATION_TIMEOUT_FAILURE_CLASS,
        )

        assert "[Timeout]" in result
        await asyncio.wait_for(cancellation_started.wait(), timeout=0.1)
        assert tool_task is not None and not tool_task.done()

        next_iteration = asyncio.create_task(interpreter._execute_async(2))
        await asyncio.sleep(0.02)
        assert not next_iteration_started.is_set()

        release_cleanup.set()
        assert await asyncio.wait_for(next_iteration, timeout=0.2) == "next"
    finally:
        release_cleanup.set()
        if tool_task is not None:
            await asyncio.gather(tool_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_owned_jspi_release_waits_for_async_tool_cancellation(monkeypatch):
    interpreter = JspiBackend.__new__(JspiBackend)
    interpreter._host_tool_tasks = set()
    interpreter._pending_file_ops = {}
    cleanup_finished = asyncio.Event()
    shutdown_saw_cleanup = False

    async def tool_task():
        try:
            await asyncio.Future()
        finally:
            cleanup_finished.set()

    async def shutdown():
        nonlocal shutdown_saw_cleanup
        shutdown_saw_cleanup = cleanup_finished.is_set()

    interpreter.ashutdown = shutdown  # type: ignore[method-assign]
    monkeypatch.setattr(
        "predict_rlm.backends.jspi.execution.JspiBackend",
        lambda **kwargs: interpreter,
    )
    backend = JspiExecutionBackend()
    task: asyncio.Task | None = None
    try:
        context = SimpleNamespace(session=None, ownership=None)
        async with backend.start(ExecutionSpec(), context):
            task = asyncio.create_task(tool_task())
            interpreter._track_host_tool_task(task)
            await asyncio.sleep(0)

        assert cleanup_finished.is_set()
        assert shutdown_saw_cleanup
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_structured_timeout_retains_async_tool_ownership():
    interpreter = JspiBackend.__new__(JspiBackend)
    interpreter._host_tool_tasks = set()
    interpreter._active_tool_count = 0
    interpreter._pending_file_ops = {}
    release_cleanup = asyncio.Event()
    tool_started = asyncio.Event()
    reads = 0

    async def execute_tool(name, params, request_id):
        tool_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            await release_cleanup.wait()
        return {"value": "done", "type": "string"}

    async def read_with_timeout(timeout):
        nonlocal reads
        reads += 1
        if reads == 1:
            return (
                '{"jsonrpc":"2.0","id":"tool-1","method":"tool_call",'
                '"params":{"name":"slow","args":[],"kwargs":{}}}'
            )
        await tool_started.wait()
        return (
            '{"jsonrpc":"2.0","id":1,"result":'
            '{"timeout":{"seconds":0.1},"stdout":"","stderr":""}}'
        )

    interpreter._execute_tool_async = execute_tool  # type: ignore[method-assign]
    interpreter._read_with_timeout_async = read_with_timeout  # type: ignore[method-assign]
    interpreter._send_completed_responses = AsyncMock()  # type: ignore[method-assign]
    interpreter._write_stdin_async = AsyncMock()  # type: ignore[method-assign]
    interpreter._wait_and_send_all_responses = AsyncMock()  # type: ignore[method-assign]

    result = await interpreter._execute_async(1)
    idle = asyncio.create_task(interpreter.await_host_work())
    await asyncio.sleep(0)
    try:
        assert "[Timeout]" in result
        assert not idle.done()
    finally:
        release_cleanup.set()
        await asyncio.wait_for(idle, timeout=0.1)
