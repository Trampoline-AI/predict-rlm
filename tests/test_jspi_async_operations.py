from __future__ import annotations

import asyncio
import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from predict_rlm.backends.base import SandboxFatalError
from predict_rlm.backends.jspi import JspiBackend
from predict_rlm.backends.jspi.execution import JspiExecutionBackend
from predict_rlm.execution_timeout import ITERATION_TIMEOUT_FAILURE_CLASS
from predict_rlm.runtime import ExecutionSpec
from predict_rlm.workspace import WorkspaceFileInfo


def _fail_sync_helper(*args, **kwargs):
    raise AssertionError("async JSPI path called a synchronous helper")


@pytest.fixture
def interpreter() -> JspiBackend:
    backend = JspiBackend(preinstall_packages=False)
    backend._ensure_deno_process = _fail_sync_helper  # type: ignore[method-assign]
    backend._send_request = _fail_sync_helper  # type: ignore[method-assign]
    backend._mount_files = _fail_sync_helper  # type: ignore[method-assign]
    backend._register_tools = _fail_sync_helper  # type: ignore[method-assign]
    return backend


@pytest.mark.asyncio
async def test_aexecute_uses_only_async_setup_helpers(interpreter: JspiBackend):
    interpreter._aensure_deno_process = AsyncMock()  # type: ignore[method-assign]
    interpreter._amount_files = AsyncMock()  # type: ignore[method-assign]
    interpreter._aregister_tools = AsyncMock()  # type: ignore[method-assign]
    interpreter._write_stdin_async = AsyncMock()  # type: ignore[method-assign]
    interpreter._execute_with_timeout = AsyncMock(return_value="ok")  # type: ignore[method-assign]

    result = await interpreter._aexecute_inner("print('ok')", {})

    assert result == "ok"
    interpreter._aensure_deno_process.assert_awaited_once_with()
    interpreter._amount_files.assert_awaited_once_with()
    interpreter._aregister_tools.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_aexecute_broken_pipe_uses_async_kill(interpreter: JspiBackend):
    interpreter._aensure_deno_process = AsyncMock()  # type: ignore[method-assign]
    interpreter._amount_files = AsyncMock()  # type: ignore[method-assign]
    interpreter._aregister_tools = AsyncMock()  # type: ignore[method-assign]
    interpreter._write_stdin_async = AsyncMock(  # type: ignore[method-assign]
        side_effect=BrokenPipeError
    )
    interpreter._kill_sandbox = _fail_sync_helper  # type: ignore[method-assign]
    interpreter._akill_sandbox = AsyncMock()  # type: ignore[method-assign]

    with pytest.raises(SandboxFatalError, match="BrokenPipeError"):
        await interpreter._aexecute_inner("print('ok')", {})

    interpreter._akill_sandbox.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_async_control_operations_use_async_rpc(interpreter: JspiBackend):
    interpreter._aensure_deno_process = AsyncMock()  # type: ignore[method-assign]
    interpreter._asend_request = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            {"result": {"mounted": "/sandbox/input.txt"}},
            {"result": {"created": "/sandbox/output"}},
            {"result": {"files": ["/sandbox/output/result.txt"]}},
            {
                "result": {
                    "files": {
                        "result.txt": {
                            "type": "file",
                            "sha256": "abc",
                            "size": 3,
                        }
                    }
                }
            },
            {"result": {"ok": True}},
        ]
    )

    await interpreter.amount_file_at("/host/input.txt", "/sandbox/input.txt")
    await interpreter.amkdir_p("/sandbox/output")
    files = await interpreter.alist_dir("/sandbox/output")
    manifest = await interpreter.aworkspace_manifest("/sandbox/output")
    await interpreter.async_file_to("/sandbox/output/result.txt", "/host/result.txt")

    assert files == ["/sandbox/output/result.txt"]
    assert manifest == {
        "result.txt": WorkspaceFileInfo(type="file", sha256="abc", size=3)
    }
    assert [call.args[0] for call in interpreter._asend_request.await_args_list] == [
        "mount_file",
        "mkdir_p",
        "list_dir",
        "workspace_manifest",
        "sync_file",
    ]


@pytest.mark.asyncio
async def test_async_package_setup_uses_async_rpc(interpreter: JspiBackend):
    interpreter._aensure_deno_process = AsyncMock()  # type: ignore[method-assign]
    interpreter._asend_request = AsyncMock(  # type: ignore[method-assign]
        return_value={"result": {"installed": ["openpyxl"]}}
    )

    await interpreter.aensure_skill_packages(["openpyxl", "openpyxl"])

    interpreter._asend_request.assert_awaited_once_with(
        "install_packages",
        {"packages": ["openpyxl"]},
        "installing skill packages",
    )


@pytest.mark.asyncio
async def test_async_ready_uses_native_async_startup(interpreter: JspiBackend):
    interpreter._aensure_deno_process = AsyncMock()  # type: ignore[method-assign]

    await interpreter.aensure_ready()
    await interpreter.astart()

    assert interpreter._aensure_deno_process.await_count == 2


def test_async_lifecycle_methods_do_not_use_to_thread():
    methods = (
        JspiBackend._aensure_deno_process,
        JspiBackend._ahealth_check,
        JspiBackend._asend_request,
        JspiBackend._amount_files,
        JspiBackend._aregister_tools,
        JspiBackend._async_files,
        JspiBackend._akill_sandbox,
        JspiBackend.aensure_ready,
        JspiBackend.astart,
        JspiBackend.aensure_skill_packages,
        JspiBackend.amount_file_at,
        JspiBackend.amkdir_p,
        JspiBackend.alist_dir,
        JspiBackend.aworkspace_manifest,
        JspiBackend.async_file_to,
        JspiBackend.aexecute,
        JspiBackend.ainterrupt,
        JspiBackend.ashutdown,
    )

    for method in methods:
        source = inspect.getsource(method)
        assert "to_thread" not in source, method.__qualname__


@pytest.mark.asyncio
async def test_ainterrupt_does_not_call_sync_interrupt(interpreter: JspiBackend):
    interpreter.interrupt = _fail_sync_helper  # type: ignore[method-assign]
    interpreter._akill_sandbox = AsyncMock()  # type: ignore[method-assign]

    await interpreter.ainterrupt()

    interpreter._akill_sandbox.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_ashutdown_does_not_call_sync_shutdown(interpreter: JspiBackend):
    interpreter.shutdown = _fail_sync_helper  # type: ignore[method-assign]
    interpreter.deno_process = type(
        "Process",
        (),
        {"poll": lambda self: None, "stdin": None},
    )()
    interpreter._write_stdin_async = AsyncMock()  # type: ignore[method-assign]
    interpreter._await_process_exit = AsyncMock(return_value=True)  # type: ignore[method-assign]

    await asyncio.wait_for(interpreter.ashutdown(), timeout=0.1)

    message = json.loads(interpreter._write_stdin_async.await_args.args[0])
    assert message["method"] == "shutdown"
    assert interpreter.deno_process is None


@pytest.mark.asyncio
async def test_async_rpc_uses_only_async_fd_primitives(interpreter: JspiBackend):
    interpreter._write_stdin_async = AsyncMock()  # type: ignore[method-assign]
    interpreter._read_with_timeout_async = AsyncMock(  # type: ignore[method-assign]
        side_effect=lambda timeout: json.dumps(
            {"jsonrpc": "2.0", "id": interpreter._request_id, "result": {"ok": True}}
        )
    )

    response = await interpreter._asend_request("mkdir_p", {"path": "/sandbox/x"}, "test")

    assert response["result"] == {"ok": True}
    interpreter._write_stdin_async.assert_awaited_once()
    interpreter._read_with_timeout_async.assert_awaited_once()


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
async def test_sync_worker_quarantine_defers_jspi_shutdown(interpreter: JspiBackend):
    release = __import__("threading").Event()

    worker = interpreter._start_sync_worker(lambda: release.wait())
    shutdown = MagicMock()
    interpreter.shutdown = shutdown  # type: ignore[method-assign]

    retired = interpreter.retire_when_sync_workers_finish()
    await asyncio.sleep(0)

    assert retired is True
    shutdown.assert_not_called()

    release.set()
    await asyncio.wait_for(worker.wait(), timeout=1)
    for _ in range(100):
        if shutdown.called:
            break
        await asyncio.sleep(0.01)

    shutdown.assert_called_once_with()


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
