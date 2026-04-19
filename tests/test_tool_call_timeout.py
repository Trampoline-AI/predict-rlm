"""RED-GREEN repro for unbounded host-side tool calls.

Background:
    Sandbox code calling ``await recalculate(path)`` triggers a host
    tool dispatch in ``JspiInterpreter._execute_tool_async``. If the
    tool is slow (e.g. LibreOffice on a whole-column formula → 2-3
    minutes), the overall ``execute`` round-trip blows past the
    ``_exec_timeout`` ceiling. That timeout **kills the Deno
    subprocess**, raises ``SandboxFatalError``, and turns what should
    have been a recoverable tool error into a cascade of
    ``[Errno 9] Bad file descriptor`` retries. A 2026-04-18 gemini
    eval lost 19 cases to this failure mode — every one of them
    scored 0 by the time ``task_timeout=600s`` finally fired.

    The fix: give each tool call its own wall-clock budget
    (``TOOL_CALL_TIMEOUT_SEC``, default 180s) via ``asyncio.wait_for``.
    If the tool exceeds its budget, return a clean error response to
    the sandbox — deno's ``await tool()`` resumes with the error,
    exec continues, the RLM can see "[Error] tool timed out" and
    rewrite its code using a different approach. The sandbox stays
    alive, the case stays recoverable.

RED: a mock tool that sleeps forever hangs ``_execute_tool_async``.
GREEN: it returns an error response with a timeout message within
    ~TOOL_CALL_TIMEOUT_SEC.
"""

from __future__ import annotations

import asyncio
import time

import pytest

import predict_rlm.interpreter as rlm_interpreter
from predict_rlm.interpreter import JspiInterpreter


def _build_interp_with_tool(tool_fn, tool_name: str = "slow_tool"):
    interp = JspiInterpreter.__new__(JspiInterpreter)
    interp.tools = {tool_name: tool_fn}
    interp._debug = False
    interp._executor = None  # only used for sync tools
    # Bypass the SyncedFile param-scanner by setting no tools use it.
    # The get_synced_file_params() helper returns {} for a plain fn.
    return interp


async def _async_never_returns(*args, **kwargs):
    """Async tool that hangs forever — the pathological case our fix
    must bound."""
    await asyncio.sleep(3600)
    return "unreachable"  # pragma: no cover


def test_async_tool_that_hangs_times_out_cleanly(monkeypatch):
    """A tool that never returns must surface as an error response
    within ~TOOL_CALL_TIMEOUT_SEC, not hang the caller indefinitely.
    """
    monkeypatch.setattr(rlm_interpreter, "TOOL_CALL_TIMEOUT_SEC", 0.3)

    interp = _build_interp_with_tool(_async_never_returns)
    t0 = time.monotonic()

    async def _run():
        # Outer safety net: if the fix isn't in, asyncio.wait_for here
        # turns the hang into a TimeoutError so pytest can fail
        # cleanly instead of running forever.
        return await asyncio.wait_for(
            interp._execute_tool_async("slow_tool", {"args": [], "kwargs": {}}),
            timeout=5.0,
        )

    try:
        response = asyncio.run(_run())
    except asyncio.TimeoutError:
        pytest.fail(
            "_execute_tool_async hung past the test's 5s safety timeout — "
            "the per-tool TOOL_CALL_TIMEOUT_SEC bound isn't enforced"
        )

    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, (
        f"tool returned but took {elapsed:.2f}s — expected ~0.3s based on "
        f"the monkeypatched timeout"
    )
    assert "error" in response, (
        f"expected error response after timeout, got {response!r}"
    )
    err = str(response.get("error") or "")
    assert "timed out" in err.lower() or "timeout" in err.lower(), (
        f"error message should mention the timeout; got {err!r}"
    )


def test_async_tool_that_completes_quickly_is_not_affected(monkeypatch):
    """Guardrail: normal fast tools must continue to return their
    results unchanged — the timeout is a ceiling, not a delay.
    """
    monkeypatch.setattr(rlm_interpreter, "TOOL_CALL_TIMEOUT_SEC", 1.0)

    async def _fast_tool(**_kwargs):
        return "ok"

    interp = _build_interp_with_tool(_fast_tool)
    response = asyncio.run(
        interp._execute_tool_async("slow_tool", {"args": [], "kwargs": {}})
    )
    assert response.get("value") == "ok"
    assert "error" not in response


def test_tool_exception_still_routes_through_error_path(monkeypatch):
    """If a tool raises (e.g. ValueError inside the tool), the existing
    ``except Exception`` in _execute_tool_async captures it and returns
    ``{"error": ...}``. The timeout wrap must not change this behaviour.
    """
    monkeypatch.setattr(rlm_interpreter, "TOOL_CALL_TIMEOUT_SEC", 1.0)

    async def _raising_tool(**_kwargs):
        raise ValueError("tool blew up")

    interp = _build_interp_with_tool(_raising_tool)
    response = asyncio.run(
        interp._execute_tool_async("slow_tool", {"args": [], "kwargs": {}})
    )
    assert "error" in response
    assert "blew up" in str(response["error"])
