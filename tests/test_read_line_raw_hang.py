"""RED-GREEN repro for the deno-stdout blocking-read hang.

Background:
    ``JspiInterpreter._read_with_timeout`` is called inline from
    ``_send_request``, which is used by the synchronous health-check
    path (``_health_check`` → ``_ensure_deno_process``). When that path
    runs inside an asyncio coroutine (via ``aexecute`` → ``_aexecute_inner``),
    a blocking read on the deno stdout fd freezes the **entire event loop**.

    The outer ``_read_with_timeout`` uses ``select.select()`` with a
    timeout — but only to check *whether there is data to read*. Once
    select says ready it calls ``_read_line_raw()``, which loops on
    ``os.read(fd, 65536)`` until a newline arrives. If the first ``os.read``
    returns partial bytes without a newline, the next ``os.read`` iteration
    has **no timeout** and blocks indefinitely waiting for more data.

    This caused a production stall on 2026-04-18: a gemini+medium eval
    seized at 319/400 tasks with the main asyncio event loop parked in
    ``os.read`` inside ``_read_line_raw``. ``kill -USR1`` via the
    faulthandler committed that morning confirmed the exact frame.

RED (pre-fix): partial bytes (no newline) followed by an unbounded wait
    makes ``_read_with_timeout(timeout=0.2)`` hang forever. The test's
    outer thread-join timeout of 2s catches the hang and fails the
    assertion — proves the bug exists.

GREEN (post-fix): ``_read_line_raw`` re-checks the deadline between
    each ``os.read`` iteration and raises ``TimeoutError`` (or returns
    ``None`` via ``_read_with_timeout``) when the budget is exhausted.
"""

from __future__ import annotations

import os
import threading
import time
import types

import pytest

from predict_rlm.interpreter import JspiInterpreter


def _make_interp_with_partial_pipe():
    """Build a JspiInterpreter pointed at a real OS pipe fd, with
    some partial bytes (no newline) already sitting in the kernel
    buffer. The write end is left open and silent, simulating a deno
    process that wrote a partial line then stopped producing output
    (crashed, hung on GC, blocked on its own I/O — doesn't matter).
    """
    read_fd, write_fd = os.pipe()
    # Write partial data (no '\n'). Writer stays open so read won't get EOF.
    os.write(write_fd, b"partial-line-no-newline-here")

    interp = JspiInterpreter.__new__(JspiInterpreter)
    interp._stdout_fd = read_fd
    interp._read_buf = ""
    interp.deno_process = types.SimpleNamespace(
        stdout=types.SimpleNamespace(fileno=lambda: read_fd),
        poll=lambda: None,
    )
    return interp, write_fd


def test_read_with_timeout_does_not_hang_on_partial_line():
    """Hand ``_read_with_timeout`` an fd that emits partial bytes then
    goes silent. It must return within the configured timeout (plus a
    small safety margin) — not block forever.

    Run in a background thread so a hang in the code under test
    doesn't freeze pytest itself.
    """
    interp, write_fd = _make_interp_with_partial_pipe()
    try:
        result = {"returned": False, "value": None, "elapsed": 0.0}

        def _call():
            t0 = time.monotonic()
            result["value"] = interp._read_with_timeout(timeout=0.2)
            result["elapsed"] = time.monotonic() - t0
            result["returned"] = True

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        # 2s safety net: if the bug is present the thread hangs here
        # and we fail the assertion below with a clean message.
        t.join(timeout=2.0)

        assert not t.is_alive(), (
            "JspiInterpreter._read_with_timeout hung on a partial-line "
            "stdout — deno-stdout deadlock is present"
        )
        assert result["returned"], "thread ended without returning a value"
        # The caller configured timeout=0.2s, so the call should return
        # within roughly that window plus a small safety margin.
        assert result["elapsed"] < 1.0, (
            f"returned but took {result['elapsed']:.2f}s — longer than "
            "the 0.2s caller-requested timeout, fix may be too loose"
        )
    finally:
        os.close(write_fd)
        try:
            os.close(interp._stdout_fd)
        except OSError:
            pass


def test_read_line_raw_directly_respects_timeout_when_given_one():
    """``_read_line_raw`` itself must accept a timeout (the fix's API
    addition) and raise when exceeded. Narrower anchor than the
    integration test above: fails if someone re-removes the timeout
    parameter or short-circuits its check.
    """
    interp, write_fd = _make_interp_with_partial_pipe()
    try:
        t0 = time.monotonic()
        with pytest.raises((TimeoutError, OSError)):
            interp._read_line_raw(timeout=0.1)
        elapsed = time.monotonic() - t0
        assert elapsed < 0.5, (
            f"_read_line_raw(timeout=0.1) took {elapsed:.2f}s to fail — "
            "timeout not being enforced tightly"
        )
    finally:
        os.close(write_fd)
        try:
            os.close(interp._stdout_fd)
        except OSError:
            pass


def test_send_request_does_not_hang_on_silent_deno():
    """The real-world trigger: ``_send_request`` calls
    ``_read_with_timeout`` and must NOT pass ``timeout=None``. If it
    does, a deno process that never replies freezes the caller (and,
    when the caller is on the asyncio event loop, every sibling
    coroutine with it).

    We mock the deno process's stdin (absorb writes silently) and stdout
    (real pipe that emits partial bytes then stalls). ``_send_request``
    must raise within a bounded window — the previous hang behavior
    would block forever and fail the 3s thread-join safety net.
    """
    import predict_rlm.interpreter as rlm_interpreter

    # Shrink the request-read budget so the test doesn't wait 30s for
    # the default fix to fire. Reusing the module-level knob the fix
    # introduces keeps this test honest against future tuning.
    if hasattr(rlm_interpreter, "DENO_REQUEST_TIMEOUT_SEC"):
        original = rlm_interpreter.DENO_REQUEST_TIMEOUT_SEC
        rlm_interpreter.DENO_REQUEST_TIMEOUT_SEC = 0.3
    else:
        original = None  # RED state — attribute doesn't exist yet

    try:
        interp, write_fd = _make_interp_with_partial_pipe()
        # Silence writes to stdin so _write_stdin doesn't raise.
        class _QuietStdin:
            def write(self, _data): pass
            def flush(self): pass
            def close(self): pass
        interp.deno_process = types.SimpleNamespace(
            stdin=_QuietStdin(),
            stdout=types.SimpleNamespace(fileno=lambda: interp._stdout_fd),
            stderr=None,
            poll=lambda: None,
        )
        interp._request_id = 0
        interp._use_jspi = False
        interp._stdin_fd = -1  # force the stdin.write() fallback path
        interp._loop = None

        result = {"returned": False, "exc": None, "elapsed": 0.0}

        def _call():
            t0 = time.monotonic()
            try:
                interp._send_request("health_check", {}, context="test")
            except BaseException as e:
                result["exc"] = e
            result["elapsed"] = time.monotonic() - t0
            result["returned"] = True

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=3.0)

        assert not t.is_alive(), (
            "JspiInterpreter._send_request hung on a silent deno stdout — "
            "_send_request is still passing timeout=None to _read_with_timeout"
        )
        assert result["exc"] is not None, (
            "_send_request returned without raising — the silent-stdout "
            "case should produce a CodeInterpreterError, not a silent success"
        )
        assert result["elapsed"] < 2.0, (
            f"_send_request took {result['elapsed']:.2f}s to fail — "
            "timeout enforcement is too loose"
        )
    finally:
        if original is not None:
            rlm_interpreter.DENO_REQUEST_TIMEOUT_SEC = original
        try:
            os.close(write_fd)
        except OSError:
            pass
        try:
            os.close(interp._stdout_fd)
        except OSError:
            pass


def test_read_line_raw_returns_full_line_when_newline_arrives():
    """Healthy-path guardrail: when the writer eventually emits a newline,
    ``_read_line_raw`` must return the full line. Guards against the fix
    regressing to "always raise/return None prematurely".
    """
    interp, write_fd = _make_interp_with_partial_pipe()
    try:
        # Finish the partial line; writer then closes.
        os.write(write_fd, b"-completion\nleftover-bytes")
        os.close(write_fd)
        write_fd = -1

        line = interp._read_line_raw()
        assert line == "partial-line-no-newline-here-completion"
    finally:
        if write_fd >= 0:
            try:
                os.close(write_fd)
            except OSError:
                pass
        try:
            os.close(interp._stdout_fd)
        except OSError:
            pass
