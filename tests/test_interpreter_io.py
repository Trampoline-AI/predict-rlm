import asyncio
import os
import threading
from types import SimpleNamespace

import pytest
from dspy.primitives.code_interpreter import CodeInterpreterError

from predict_rlm.backends import JspiBackend
from predict_rlm.backends.base import SandboxFatalError


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_pipe_backpressure_does_not_drop_bytes(asynchronous):
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    backend = JspiBackend.__new__(JspiBackend)
    backend._stdin_fd = write_fd
    backend.deno_process = SimpleNamespace(poll=lambda: None, stdin=object())
    payload = "é" * 100_000 + "\n"
    received = bytearray()

    def drain():
        while chunk := os.read(read_fd, 4096):
            received.extend(chunk)

    reader = threading.Thread(target=drain)
    reader.start()
    try:
        if asynchronous:
            await asyncio.wait_for(backend._write_stdin_async(payload), timeout=5)
        else:
            await asyncio.wait_for(asyncio.to_thread(backend._write_stdin, payload), timeout=5)
    finally:
        os.close(write_fd)
        reader.join(timeout=5)
        os.close(read_fd)
    assert not reader.is_alive()
    assert received.decode() == payload


def test_request_deadline_survives_partial_stdout_and_preserves_buffer(monkeypatch):
    import predict_rlm.backends.jspi.backend as backend_module

    monkeypatch.setattr(backend_module, "DENO_REQUEST_TIMEOUT_SEC", 0.1)
    read_fd, write_fd = os.pipe()
    backend = JspiBackend.__new__(JspiBackend)
    backend._stdout_fd = read_fd
    backend._stdin_fd = -1
    backend._read_buf = ""
    backend._request_id = 0
    backend.deno_process = SimpleNamespace(
        stdin=SimpleNamespace(write=lambda data: None, flush=lambda: None),
        stdout=SimpleNamespace(fileno=lambda: read_fd),
        stderr=None,
        poll=lambda: None,
    )
    errors = []

    def request():
        try:
            backend._send_request("health_check", {}, context="partial stdout")
        except BaseException as exc:
            errors.append(exc)

    os.write(write_fd, b"partial")
    worker = threading.Thread(target=request, daemon=True)
    worker.start()
    try:
        worker.join(timeout=2)
        assert not worker.is_alive(), "partial stdout bypassed the request deadline"
        assert len(errors) == 1 and isinstance(errors[0], CodeInterpreterError)
        os.write(write_fd, b"-completion\n")
        assert backend._read_line_raw(timeout=1) == "partial-completion"
    finally:
        os.close(write_fd)
        worker.join(timeout=2)
        os.close(read_fd)


def test_stdout_eof_does_not_drain_live_stderr():
    backend = JspiBackend.__new__(JspiBackend)

    class BlockingStderr:
        def read(self):
            raise AssertionError("stderr.read() would block while its writer remains alive")

    async def no_completed_responses(_pending_tasks):
        return None

    async def stdout_eof(_timeout):
        return ""

    backend.deno_process = SimpleNamespace(
        stderr=BlockingStderr(),
        kill=lambda: None,
        poll=lambda: 0,
    )
    backend._pending_file_ops = {}
    backend._send_completed_responses = no_completed_responses
    backend._read_with_timeout_async = stdout_eof
    with pytest.raises(SandboxFatalError):
        asyncio.run(backend._execute_async(execute_request_id=1))
