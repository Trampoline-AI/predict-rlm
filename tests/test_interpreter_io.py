import asyncio
import errno
import json
import os
import types

import predict_rlm.interpreter as rlm_interpreter
from predict_rlm.interpreter import JspiInterpreter


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
