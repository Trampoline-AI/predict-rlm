import asyncio
import json
import threading

import pytest
from aiohttp import web
from dspy_codex_lm.lm import CodexStreamError, CodexWSLM

pytestmark = pytest.mark.codex_lm


def _events():
    return [
        {"type": "response.output_text.delta", "delta": "hello"},
        {
            "type": "response.completed",
            "response": {
                "id": "resp-1",
                "model": "gpt-5.3-codex",
                "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
            },
        },
    ]


@pytest.mark.asyncio
async def test_codex_wslm_response_lite_prewarms_before_generating(unused_tcp_port):
    frames = []

    async def ws_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        first = await ws.receive_json()
        assert first["client_metadata"]["existing"] == "keep"
        assert (
            first["client_metadata"]["ws_request_header_x_openai_internal_codex_responses_lite"]
            == "true"
        )
        frames.append(first)
        if first.get("generate") is not False:
            for event in _events():
                await ws.send_str(json.dumps(event))
            await ws.close()
            return ws

        await ws.send_json(
            {
                "type": "response.completed",
                "response": {"id": "resp-prewarm"},
            }
        )
        second = await ws.receive_json()
        frames.append(second)
        for event in _events():
            await ws.send_str(json.dumps(event))
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/backend-api/codex/responses", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    port = unused_tcp_port
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()

    try:
        lm = CodexWSLM(
            model="gpt-5.6-sol",
            access_token="fake-token",
            account_id="fake-account",
            ws_base="http://127.0.0.1:%d/backend-api/codex" % port,
            cache=False,
            ws_fallback=False,
        )

        result = await lm.aforward(
            prompt="hello", cache=False, client_metadata={"existing": "keep"}
        )
    finally:
        await runner.cleanup()

    assert frames[0]["type"] == "response.create"
    assert frames[0].get("generate") is False
    assert frames[1]["type"] == "response.create"
    assert frames[1]["previous_response_id"] == "resp-prewarm"
    assert frames[1]["input"][0]["content"][0] == {
        "type": "input_text",
        "text": "hello",
    }
    assert result.output[0].content[0].text == "hello"


@pytest.mark.asyncio
async def test_codex_wslm_response_lite_prewarm_preserves_server_error_details(
    monkeypatch,
    unused_tcp_port,
):
    monkeypatch.setattr("dspy_codex_lm.lm.CODEX_STREAM_MAX_ATTEMPTS", 1)

    async def ws_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive_json()
        await ws.send_json(
            {
                "type": "error",
                "code": "rate_limit_exceeded",
                "message": "prewarm capacity exhausted",
                "retry_after": 2.5,
            }
        )
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/backend-api/codex/responses", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    port = unused_tcp_port
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()

    try:
        lm = CodexWSLM(
            model="gpt-5.6-sol",
            access_token="fake-token",
            account_id="fake-account",
            ws_base="http://127.0.0.1:%d/backend-api/codex" % port,
            cache=False,
            ws_fallback=False,
        )

        with pytest.raises(CodexStreamError) as exc_info:
            await lm.aforward(prompt="hello", cache=False)
    finally:
        await runner.cleanup()

    assert (
        exc_info.value.failure_kind,
        exc_info.value.failure_code,
        exc_info.value.retry_after_seconds,
    ) == (
        "error",
        "rate_limit_exceeded",
        2.5,
    )
    assert "prewarm capacity exhausted" in str(exc_info.value)


@pytest.mark.asyncio
async def test_codex_wslm_uses_fresh_turn_state_per_forward(unused_tcp_port):
    handshakes = []

    async def ws_handler(request):
        handshakes.append(dict(request.headers))
        ws = web.WebSocketResponse()
        ws.headers["x-codex-turn-state"] = f"sticky-{len(handshakes)}"
        await ws.prepare(request)
        await ws.receive_json()
        for event in _events():
            await ws.send_str(json.dumps(event))
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/backend-api/codex/responses", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    port = unused_tcp_port
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()

    try:
        lm = CodexWSLM(
            model="gpt-5.3-codex",
            access_token="fake-token",
            account_id="fake-account",
            ws_base="http://127.0.0.1:%d/backend-api/codex" % port,
            cache=False,
            ws_fallback=False,
        )

        await lm.aforward(prompt="first", cache=False)
        await lm.aforward(prompt="second", cache=False)
    finally:
        await runner.cleanup()

    assert len(handshakes) == 2
    assert "x-codex-turn-state" not in {k.lower(): v for k, v in handshakes[0].items()}
    assert "x-codex-turn-state" not in {k.lower(): v for k, v in handshakes[1].items()}


def test_codex_wslm_sync_forward_returns_when_server_keeps_websocket_open(
    monkeypatch,
    unused_tcp_port,
):
    monkeypatch.setattr("dspy_codex_lm.lm.CODEX_STREAM_HEARTBEAT_SEC", 0.2)
    monkeypatch.setattr("dspy_codex_lm.lm.CODEX_STREAM_MAX_ATTEMPTS", 1)
    server_ready = threading.Event()
    events_sent = threading.Event()
    release = threading.Event()
    holder = {}

    async def ws_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive_json()
        for event in _events():
            await ws.send_str(json.dumps(event))
        events_sent.set()
        await asyncio.to_thread(release.wait)
        await ws.close()
        return ws

    async def run_server():
        app = web.Application()
        app.router.add_get("/backend-api/codex/responses", ws_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        port = unused_tcp_port
        site = web.TCPSite(runner, "127.0.0.1", port)
        await site.start()
        holder["runner"] = runner
        holder["loop"] = asyncio.get_running_loop()
        server_ready.set()
        await asyncio.to_thread(release.wait)
        await runner.cleanup()

    thread = threading.Thread(target=lambda: asyncio.run(run_server()))
    thread.start()
    assert server_ready.wait(timeout=5)

    lm = CodexWSLM(
        model="gpt-5.3-codex",
        access_token="fake-token",
        account_id="fake-account",
        ws_base="http://127.0.0.1:%d/backend-api/codex" % unused_tcp_port,
        cache=False,
        ws_fallback=False,
    )

    try:
        result = lm.forward(prompt="hello", cache=False)
    finally:
        release.set()
        thread.join(timeout=5)

    assert events_sent.is_set()
    assert result.output[0].content[0].text == "hello"


@pytest.mark.asyncio
async def test_codex_wslm_websocket_401_is_codex_lm_auth_expired(
    monkeypatch,
    unused_tcp_port,
):
    monkeypatch.setattr("dspy_codex_lm.lm.CODEX_STREAM_MAX_ATTEMPTS", 1)

    async def auth_failed(_request):
        return web.Response(status=401, text="expired")

    app = web.Application()
    app.router.add_get("/backend-api/codex/responses", auth_failed)
    runner = web.AppRunner(app)
    await runner.setup()
    port = unused_tcp_port
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()

    try:
        lm = CodexWSLM(
            model="gpt-5.3-codex",
            access_token="expired-token",
            account_id="fake-account",
            ws_base="http://127.0.0.1:%d/backend-api/codex" % port,
            cache=False,
            ws_fallback=False,
        )

        with pytest.raises(CodexStreamError) as exc_info:
            await lm.aforward(prompt="hello", cache=False)
    finally:
        await runner.cleanup()

    assert exc_info.value.failure_kind == "codex_lm_auth_expired"
    assert exc_info.value.failure_code == 401
    assert "auth expired" in str(exc_info.value)


@pytest.mark.asyncio
async def test_codex_wslm_websocket_503_is_retryable_stream_error(
    monkeypatch,
    unused_tcp_port,
):
    monkeypatch.setattr("dspy_codex_lm.lm.CODEX_STREAM_MAX_ATTEMPTS", 1)

    async def service_unavailable(_request):
        return web.Response(status=503, text="temporarily unavailable")

    app = web.Application()
    app.router.add_get("/backend-api/codex/responses", service_unavailable)
    runner = web.AppRunner(app)
    await runner.setup()
    port = unused_tcp_port
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()

    try:
        lm = CodexWSLM(
            model="gpt-5.3-codex",
            access_token="fake-token",
            account_id="fake-account",
            ws_base="http://127.0.0.1:%d/backend-api/codex" % port,
            cache=False,
            ws_fallback=False,
        )

        with pytest.raises(CodexStreamError) as exc_info:
            await lm.aforward(prompt="hello", cache=False)
    finally:
        await runner.cleanup()

    assert exc_info.value.failure_kind == "codex_lm_ws_handshake"
    assert exc_info.value.failure_code == 503
    assert "503" in str(exc_info.value)
