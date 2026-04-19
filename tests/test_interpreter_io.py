import asyncio
import errno
import json
import os
import subprocess
import types

import pytest
from dspy.primitives.code_interpreter import CodeInterpreterError

import predict_rlm.interpreter as rlm_interpreter
from predict_rlm.interpreter import JSONRPC_APP_ERRORS, JspiInterpreter


class _BlockingStdin:
    def write(self, data):
        raise BlockingIOError(errno.EAGAIN, "pipe full")

    def flush(self):
        raise AssertionError("flush should not run")


class _SilentStderr:
    def read(self):
        return ""


class _BufferingStdin:
    def __init__(self):
        self.data = []
        self.flushed = False

    def write(self, data):
        self.data.append(data)

    def flush(self):
        self.flushed = True


class _BufferingStdout:
    def __init__(self, lines: list[str], fd: int = 123):
        self.lines = list(lines)
        self._fd = fd

    def fileno(self):
        return self._fd

    def readline(self):
        if self.lines:
            return self.lines.pop(0)
        return ""


def test_execute_recovers_when_stdin_blocks(monkeypatch):
    interpreter = JspiInterpreter(preinstall_packages=False)

    read_fd, write_fd = os.pipe()
    try:
        interpreter.deno_process = types.SimpleNamespace(
            stdin=_BlockingStdin(),
            stderr=_SilentStderr(),
            poll=lambda: None,
        )
        interpreter._stdin_fd = write_fd
        interpreter._stdout_fd = read_fd
        interpreter._ensure_deno_process = lambda: None
        interpreter._mount_files = lambda: None
        interpreter._register_tools = lambda: None
        interpreter._tools_registered = True
        interpreter._mounted_files = True

        real_os_write = rlm_interpreter.os.write
        state = {"blocked": True, "calls": 0}

        def fake_os_write(fd, data):
            assert fd == write_fd
            state["calls"] += 1
            if state["blocked"]:
                state["blocked"] = False
                raise BlockingIOError(errno.EAGAIN, "would block")
            return real_os_write(fd, data)

        monkeypatch.setattr(rlm_interpreter.os, "write", fake_os_write)

        select_calls = {"count": 0}

        def fake_select(rlist, wlist, xlist, timeout=None):
            select_calls["count"] += 1
            return ([], wlist, [])

        monkeypatch.setattr(rlm_interpreter.select, "select", fake_select)

        async def fake_execute_async(self, request_id):
            return "ok"

        interpreter._execute_async = types.MethodType(fake_execute_async, interpreter)

        loop = asyncio.new_event_loop()
        monkeypatch.setattr(asyncio, "get_event_loop", lambda: loop)
        try:
            result = interpreter.execute("print('hi')")
        finally:
            loop.close()

        assert result == "ok"
        assert state["calls"] == 2
        assert select_calls["count"] == 1
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_send_request_falls_back_before_fd_ready(monkeypatch):
    interpreter = JspiInterpreter(preinstall_packages=False)
    stdin = _BufferingStdin()

    interpreter.deno_process = types.SimpleNamespace(
        stdin=stdin,
        stderr=None,
        poll=lambda: None,
    )
    interpreter._stdin_fd = -1

    def fake_read(timeout=None):
        return json.dumps({"id": interpreter._request_id, "result": {"output": "ok"}})

    interpreter._read_with_timeout = fake_read  # type: ignore[assignment]

    response = interpreter._send_request("execute", {"code": "print(1)"}, "during test")

    assert stdin.data  # ensure blocking fallback wrote data
    assert stdin.flushed
    assert response["result"]["output"] == "ok"


def test_read_with_timeout_falls_back(monkeypatch):
    interpreter = JspiInterpreter(preinstall_packages=False)
    expected_line = json.dumps({"result": {"output": "hello"}, "id": 1}) + "\n"
    stdout = _BufferingStdout([expected_line])

    interpreter.deno_process = types.SimpleNamespace(
        stdin=None,
        stdout=stdout,
        poll=lambda: None,
    )
    interpreter._stdout_fd = -1
    interpreter._request_id = 1

    def fake_select(rlist, wlist, xlist, timeout=None):
        return (rlist, [], [])

    monkeypatch.setattr(rlm_interpreter.select, "select", fake_select)

    line = interpreter._read_with_timeout(timeout=0.1)
    assert line == expected_line.strip()


# ---------------------------------------------------------------------------
# _get_semaphore classmethod
# ---------------------------------------------------------------------------


def test_get_semaphore_creates_on_first_call():
    """First call creates an asyncio.Semaphore; second call returns the same one."""
    # Reset class-level state so the test is isolated
    JspiInterpreter._sandbox_semaphore = None
    try:
        sem1 = JspiInterpreter._get_semaphore()
        sem2 = JspiInterpreter._get_semaphore()
        assert isinstance(sem1, asyncio.Semaphore)
        assert sem1 is sem2
    finally:
        JspiInterpreter._sandbox_semaphore = None


# ---------------------------------------------------------------------------
# shutdown() with timeout
# ---------------------------------------------------------------------------


def test_shutdown_clean_exit():
    """shutdown() sends shutdown message and waits; process exits cleanly."""
    interpreter = JspiInterpreter(preinstall_packages=False)
    wait_calls = []
    close_called = {"value": False}

    class _ClosableBufferingStdin(_BufferingStdin):
        def close(self):
            close_called["value"] = True

    stdin = _ClosableBufferingStdin()

    interpreter.deno_process = types.SimpleNamespace(
        stdin=stdin,
        stderr=_SilentStderr(),
        poll=lambda: None,  # process is alive
        wait=lambda timeout=None: wait_calls.append(timeout),
        kill=lambda: (_ for _ in ()).throw(AssertionError("kill should not be called")),
    )
    interpreter._stdin_fd = -1  # use blocking fallback for _write_stdin

    interpreter.shutdown()

    assert any('"method": "shutdown"' in d or '"method":"shutdown"' in d for d in stdin.data)
    assert close_called["value"]
    assert interpreter.deno_process is None
    assert wait_calls == [5]


def test_shutdown_timeout_kills_process():
    """shutdown() kills process when wait() raises TimeoutExpired."""
    interpreter = JspiInterpreter(preinstall_packages=False)
    stdin = _BufferingStdin()
    killed = {"value": False}
    final_wait_calls = []

    def fake_wait(timeout=None):
        if timeout is not None:
            raise subprocess.TimeoutExpired("deno", timeout)
        final_wait_calls.append(True)

    # stdin.close() needs to be a callable
    close_called = {"value": False}

    class _ClosableStdin(_BufferingStdin):
        def close(self):
            close_called["value"] = True

    stdin = _ClosableStdin()

    interpreter.deno_process = types.SimpleNamespace(
        stdin=stdin,
        stderr=_SilentStderr(),
        poll=lambda: None,
        wait=fake_wait,
        kill=lambda: killed.__setitem__("value", True),
    )
    interpreter._stdin_fd = -1

    interpreter.shutdown()

    assert killed["value"] is True
    assert len(final_wait_calls) == 1  # wait() after kill, no timeout
    assert interpreter.deno_process is None


def test_shutdown_noop_when_process_already_exited():
    """shutdown() is a no-op if deno_process is None."""
    interpreter = JspiInterpreter(preinstall_packages=False)
    interpreter.deno_process = None
    interpreter.shutdown()  # should not raise
    assert interpreter.deno_process is None


# ---------------------------------------------------------------------------
# _sync_files
# ---------------------------------------------------------------------------


def test_sync_files_sends_messages():
    """_sync_files sends a sync_file JSON-RPC message for each write path."""
    interpreter = JspiInterpreter(preinstall_packages=False)
    stdin = _BufferingStdin()

    interpreter.deno_process = types.SimpleNamespace(
        stdin=stdin,
        stderr=None,
        poll=lambda: None,
    )
    interpreter._stdin_fd = -1
    interpreter.enable_write_paths = ["/host/data/output.csv", "/host/data/report.json"]
    interpreter.sync_files = True

    interpreter._sync_files()

    assert len(stdin.data) == 2
    for i, path in enumerate(interpreter.enable_write_paths):
        msg = json.loads(stdin.data[i].rstrip("\n"))
        assert msg["method"] == "sync_file"
        assert msg["params"]["host_path"] == str(path)
        assert msg["params"]["virtual_path"] == f"/sandbox/{os.path.basename(path)}"


def test_sync_files_skips_when_disabled():
    """_sync_files is a no-op when sync_files is False."""
    interpreter = JspiInterpreter(preinstall_packages=False)
    stdin = _BufferingStdin()

    interpreter.deno_process = types.SimpleNamespace(
        stdin=stdin, stderr=None, poll=lambda: None
    )
    interpreter._stdin_fd = -1
    interpreter.enable_write_paths = ["/host/data/output.csv"]
    interpreter.sync_files = False

    interpreter._sync_files()

    assert stdin.data == []


def test_sync_files_skips_when_no_write_paths():
    """_sync_files is a no-op when enable_write_paths is empty."""
    interpreter = JspiInterpreter(preinstall_packages=False)
    stdin = _BufferingStdin()

    interpreter.deno_process = types.SimpleNamespace(
        stdin=stdin, stderr=None, poll=lambda: None
    )
    interpreter._stdin_fd = -1
    interpreter.enable_write_paths = []
    interpreter.sync_files = True

    interpreter._sync_files()

    assert stdin.data == []


# ---------------------------------------------------------------------------
# _send_request error paths
# ---------------------------------------------------------------------------


def test_send_request_deno_exit_detection():
    """_send_request raises CodeInterpreterError when Deno has exited."""
    interpreter = JspiInterpreter(preinstall_packages=False)
    stdin = _BufferingStdin()

    # poll() returns None during _write_stdin (process alive), then 1 after
    # _read_with_timeout returns empty (process died mid-execution).
    poll_results = iter([None, 1])

    interpreter.deno_process = types.SimpleNamespace(
        stdin=stdin,
        stderr=types.SimpleNamespace(read=lambda: "segfault"),
        poll=lambda: next(poll_results),
    )
    interpreter._stdin_fd = -1

    # _read_with_timeout returns empty string (Deno died)
    interpreter._read_with_timeout = lambda timeout=None: ""  # type: ignore[assignment]

    with pytest.raises(CodeInterpreterError, match=r"Deno exited \(code 1\).*segfault"):
        interpreter._send_request("execute", {"code": "1+1"}, "during test")


def test_send_request_response_id_mismatch():
    """_send_request raises CodeInterpreterError when no matching response
    ever arrives. The resync loop discards mismatched-id frames (stale
    responses from prior timed-out requests) and only raises after
    exhausting its safety cap — ensuring that a runaway wrong-id stream
    doesn't hang the caller forever.
    """
    interpreter = JspiInterpreter(preinstall_packages=False)
    stdin = _BufferingStdin()

    interpreter.deno_process = types.SimpleNamespace(
        stdin=stdin, stderr=None, poll=lambda: None
    )
    interpreter._stdin_fd = -1

    # Return a response with mismatched id forever — the resync cap
    # ensures we bail instead of spinning.
    interpreter._read_with_timeout = lambda timeout=None: json.dumps(  # type: ignore[assignment]
        {"id": 9999, "result": {"output": "ok"}}
    )

    with pytest.raises(CodeInterpreterError, match="stale|resync"):
        interpreter._send_request("execute", {"code": "1+1"}, "during test")


def test_send_request_error_in_response():
    """_send_request raises CodeInterpreterError when response contains an error."""
    interpreter = JspiInterpreter(preinstall_packages=False)
    stdin = _BufferingStdin()

    interpreter.deno_process = types.SimpleNamespace(
        stdin=stdin, stderr=None, poll=lambda: None
    )
    interpreter._stdin_fd = -1

    def fake_read(timeout=None):
        return json.dumps(
            {"id": interpreter._request_id, "error": {"message": "tool failed"}}
        )

    interpreter._read_with_timeout = fake_read  # type: ignore[assignment]

    with pytest.raises(CodeInterpreterError, match="tool failed"):
        interpreter._send_request("execute", {"code": "1+1"}, "during test")


# ---------------------------------------------------------------------------
# aexecute semaphore
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aexecute_acquires_and_releases_semaphore():
    """aexecute acquires the semaphore before running and releases it after."""
    JspiInterpreter._sandbox_semaphore = None
    try:
        interpreter = JspiInterpreter(preinstall_packages=False)
        sem = JspiInterpreter._get_semaphore()

        acquire_count = {"value": 0}
        release_count = {"value": 0}
        original_acquire = sem.acquire
        original_release = sem.release

        async def tracking_acquire():
            acquire_count["value"] += 1
            return await original_acquire()

        def tracking_release():
            release_count["value"] += 1
            return original_release()

        sem.acquire = tracking_acquire  # type: ignore[assignment]
        sem.release = tracking_release  # type: ignore[assignment]

        # Mock _aexecute_inner to avoid real Deno subprocess
        async def fake_inner(code, variables):
            return "result"

        interpreter._aexecute_inner = fake_inner  # type: ignore[assignment]

        result = await interpreter.aexecute("print(1)")
        assert result == "result"
        assert acquire_count["value"] == 1
        assert release_count["value"] == 1
    finally:
        JspiInterpreter._sandbox_semaphore = None


@pytest.mark.asyncio
async def test_aexecute_releases_semaphore_on_error():
    """aexecute releases the semaphore even when _aexecute_inner raises."""
    JspiInterpreter._sandbox_semaphore = None
    try:
        interpreter = JspiInterpreter(preinstall_packages=False)
        sem = JspiInterpreter._get_semaphore()

        release_count = {"value": 0}
        original_release = sem.release

        def tracking_release():
            release_count["value"] += 1
            return original_release()

        sem.release = tracking_release  # type: ignore[assignment]

        async def failing_inner(code, variables):
            raise RuntimeError("boom")

        interpreter._aexecute_inner = failing_inner  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="boom"):
            await interpreter.aexecute("print(1)")

        assert release_count["value"] == 1
    finally:
        JspiInterpreter._sandbox_semaphore = None


# ---------------------------------------------------------------------------
# _write_stdin_async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_stdin_async_normal_write(monkeypatch):
    """_write_stdin_async writes data via os.write in a single pass."""
    interpreter = JspiInterpreter(preinstall_packages=False)
    written_chunks = []

    interpreter.deno_process = types.SimpleNamespace(poll=lambda: None)
    interpreter._stdin_fd = 42

    def fake_write(fd, data):
        assert fd == 42
        written_chunks.append(data)
        return len(data)

    monkeypatch.setattr(rlm_interpreter.os, "write", fake_write)

    await interpreter._write_stdin_async("hello\n")

    assert b"".join(written_chunks) == b"hello\n"


@pytest.mark.asyncio
async def test_write_stdin_async_blocking_retry(monkeypatch):
    """_write_stdin_async retries via add_writer when BlockingIOError occurs."""
    interpreter = JspiInterpreter(preinstall_packages=False)

    interpreter.deno_process = types.SimpleNamespace(poll=lambda: None)
    interpreter._stdin_fd = 42

    state = {"blocked": True}
    written_chunks = []

    def fake_write(fd, data):
        if state["blocked"]:
            state["blocked"] = False
            raise BlockingIOError(errno.EAGAIN, "pipe full")
        written_chunks.append(data)
        return len(data)

    monkeypatch.setattr(rlm_interpreter.os, "write", fake_write)

    # We need a real event loop with add_writer support.
    # The default event loop on macOS uses kqueue which requires real fds,
    # so we mock add_writer/remove_writer to immediately signal writable.
    loop = asyncio.get_running_loop()

    def fake_add_writer(fd, callback, *args):
        # Immediately schedule the callback so the write can proceed
        loop.call_soon(callback, *args)

    def fake_remove_writer(fd):
        pass

    monkeypatch.setattr(loop, "add_writer", fake_add_writer)
    monkeypatch.setattr(loop, "remove_writer", fake_remove_writer)

    await interpreter._write_stdin_async("data\n")

    assert b"".join(written_chunks) == b"data\n"


@pytest.mark.asyncio
async def test_write_stdin_async_raises_when_process_dead():
    """_write_stdin_async raises CodeInterpreterError when process has exited."""
    interpreter = JspiInterpreter(preinstall_packages=False)
    interpreter.deno_process = types.SimpleNamespace(poll=lambda: 1)
    interpreter._stdin_fd = 42

    with pytest.raises(CodeInterpreterError, match="no longer running"):
        await interpreter._write_stdin_async("hello\n")


# ---------------------------------------------------------------------------
# _send_completed_responses
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_completed_responses_sends_one_result(monkeypatch):
    """_send_completed_responses sends at most one response for a completed task."""
    interpreter = JspiInterpreter(preinstall_packages=False)
    sent_messages = []

    async def fake_write(data):
        sent_messages.append(data)

    interpreter._write_stdin_async = fake_write  # type: ignore[assignment]

    # Create two completed tasks
    task1 = asyncio.get_running_loop().create_future()
    task1.set_result({"value": "result1", "type": "string"})
    task2 = asyncio.get_running_loop().create_future()
    task2.set_result({"value": "result2", "type": "string"})

    pending = {"req1": task1, "req2": task2}

    await interpreter._send_completed_responses(pending)

    # Only one sent (at most one per call)
    assert len(sent_messages) == 1
    msg = json.loads(sent_messages[0].rstrip("\n"))
    assert "result" in msg
    # One task was popped
    assert len(pending) == 1


@pytest.mark.asyncio
async def test_send_completed_responses_error_task(monkeypatch):
    """_send_completed_responses sends a JSON-RPC error for a failed task."""
    interpreter = JspiInterpreter(preinstall_packages=False)
    sent_messages = []

    async def fake_write(data):
        sent_messages.append(data)

    interpreter._write_stdin_async = fake_write  # type: ignore[assignment]

    # Task that returned an error dict
    task = asyncio.get_running_loop().create_future()
    task.set_result({"error": "tool exploded"})

    pending = {"req1": task}

    await interpreter._send_completed_responses(pending)

    assert len(sent_messages) == 1
    msg = json.loads(sent_messages[0].rstrip("\n"))
    assert "error" in msg
    assert msg["error"]["message"] == "tool exploded"
    assert msg["error"]["code"] == JSONRPC_APP_ERRORS["RuntimeError"]
    assert len(pending) == 0


@pytest.mark.asyncio
async def test_send_completed_responses_exception_in_task():
    """_send_completed_responses handles an exception raised by the task."""
    interpreter = JspiInterpreter(preinstall_packages=False)
    sent_messages = []

    async def fake_write(data):
        sent_messages.append(data)

    interpreter._write_stdin_async = fake_write  # type: ignore[assignment]

    task = asyncio.get_running_loop().create_future()
    task.set_exception(ValueError("unexpected"))

    pending = {"req1": task}

    await interpreter._send_completed_responses(pending)

    assert len(sent_messages) == 1
    msg = json.loads(sent_messages[0].rstrip("\n"))
    assert "error" in msg
    assert "unexpected" in msg["error"]["message"]


# ---------------------------------------------------------------------------
# _wait_and_send_all_responses
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_and_send_all_responses_sends_all():
    """_wait_and_send_all_responses waits for all tasks and sends responses."""
    interpreter = JspiInterpreter(preinstall_packages=False)
    sent_messages = []

    async def fake_write(data):
        sent_messages.append(data)

    interpreter._write_stdin_async = fake_write  # type: ignore[assignment]

    task1 = asyncio.get_running_loop().create_future()
    task1.set_result({"value": "r1", "type": "string"})
    task2 = asyncio.get_running_loop().create_future()
    task2.set_result({"value": "r2", "type": "json"})

    pending = {"req1": task1, "req2": task2}

    await interpreter._wait_and_send_all_responses(pending)

    assert len(sent_messages) == 2
    assert len(pending) == 0  # cleared


@pytest.mark.asyncio
async def test_wait_and_send_all_responses_error_path():
    """_wait_and_send_all_responses sends error for a task that raises."""
    interpreter = JspiInterpreter(preinstall_packages=False)
    sent_messages = []

    async def fake_write(data):
        sent_messages.append(data)

    interpreter._write_stdin_async = fake_write  # type: ignore[assignment]

    task = asyncio.get_running_loop().create_future()
    task.set_exception(RuntimeError("crash"))

    pending = {"req1": task}

    await interpreter._wait_and_send_all_responses(pending)

    assert len(sent_messages) == 1
    msg = json.loads(sent_messages[0].rstrip("\n"))
    assert "error" in msg
    assert "crash" in msg["error"]["message"]
    assert len(pending) == 0


# ---------------------------------------------------------------------------
# SyntaxError detail formatting in _execute_async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_async_syntax_error_formatting():
    """_execute_async formats SyntaxError with line/col/text from args tuple."""
    interpreter = JspiInterpreter(preinstall_packages=False)

    # Build a JSON-RPC error response with SyntaxError args format
    error_response = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "error": {
            "code": -32000,
            "message": "invalid syntax",
            "data": {
                "type": "SyntaxError",
                "args": [
                    "invalid syntax",
                    ["<code>", 5, 10, "x = (1 +\n"],
                ],
            },
        },
    })

    call_count = {"value": 0}

    async def fake_read(timeout=None):
        call_count["value"] += 1
        if call_count["value"] == 1:
            return error_response
        return None

    interpreter._read_with_timeout_async = fake_read  # type: ignore[assignment]

    # _send_completed_responses and _wait_and_send_all_responses need to be no-ops
    async def noop_responses(pending):
        pass

    interpreter._send_completed_responses = noop_responses  # type: ignore[assignment]
    interpreter._wait_and_send_all_responses = noop_responses  # type: ignore[assignment]

    with pytest.raises(SyntaxError) as exc_info:
        await interpreter._execute_async(1)

    detail = str(exc_info.value)
    assert "line 5" in detail
    assert "col 10" in detail
    assert "x = (1 +" in detail


@pytest.mark.asyncio
async def test_execute_async_syntax_error_minimal_args():
    """_execute_async handles SyntaxError with minimal args (just message)."""
    interpreter = JspiInterpreter(preinstall_packages=False)

    error_response = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "error": {
            "code": -32000,
            "message": "unexpected EOF",
            "data": {
                "type": "SyntaxError",
                "args": ["unexpected EOF"],
            },
        },
    })

    call_count = {"value": 0}

    async def fake_read(timeout=None):
        call_count["value"] += 1
        if call_count["value"] == 1:
            return error_response
        return None

    interpreter._read_with_timeout_async = fake_read  # type: ignore[assignment]

    async def noop_responses(pending):
        pass

    interpreter._send_completed_responses = noop_responses  # type: ignore[assignment]
    interpreter._wait_and_send_all_responses = noop_responses  # type: ignore[assignment]

    with pytest.raises(SyntaxError, match="unexpected EOF"):
        await interpreter._execute_async(1)


# ---------------------------------------------------------------------------
# _read_with_timeout fallback when fd < 0 and stdout.fileno() is used
# ---------------------------------------------------------------------------


def test_read_with_timeout_uses_stdout_fileno_when_fd_negative(monkeypatch):
    """_read_with_timeout uses stdout.fileno() when _stdout_fd < 0."""
    interpreter = JspiInterpreter(preinstall_packages=False)
    expected_line = json.dumps({"result": "ok", "id": 1}) + "\n"

    # _BufferingStdout.fileno() returns 999 by default
    stdout = _BufferingStdout([expected_line], fd=999)

    interpreter.deno_process = types.SimpleNamespace(
        stdin=None,
        stdout=stdout,
        poll=lambda: None,
    )
    interpreter._stdout_fd = -1
    interpreter._request_id = 1

    fileno_calls = {"count": 0}
    original_fileno = stdout.fileno

    def tracking_fileno():
        fileno_calls["count"] += 1
        return original_fileno()

    stdout.fileno = tracking_fileno

    def fake_select(rlist, wlist, xlist, timeout=None):
        # Verify select was called with the fileno() result
        assert rlist == [999]
        return (rlist, [], [])

    monkeypatch.setattr(rlm_interpreter.select, "select", fake_select)

    line = interpreter._read_with_timeout(timeout=0.1)
    assert line == expected_line.strip()
    assert fileno_calls["count"] == 1


def test_read_with_timeout_returns_none_when_no_stdout(monkeypatch):
    """_read_with_timeout returns None when stdout is None and fd < 0."""
    interpreter = JspiInterpreter(preinstall_packages=False)

    interpreter.deno_process = types.SimpleNamespace(
        stdin=None,
        stdout=None,
        poll=lambda: None,
    )
    interpreter._stdout_fd = -1

    result = interpreter._read_with_timeout(timeout=0.1)
    assert result is None
