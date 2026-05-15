import asyncio
import logging
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

import dspy
import litellm
from dspy.clients.lm import _convert_chat_request_to_responses_request
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from openai.types.responses import ResponseOutputMessage, ResponseOutputText
from tenacity import (
    AsyncRetrying,
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from dspy_codex_lm.auth import (
    list_enabled_auth_profiles,
    load_codex_auth,
    profile_auth_path,
    resolve_auth_source,
)
from dspy_codex_lm.cost import compute_cost, register_known_model_costs

CODEX_BASE = "https://chatgpt.com/backend-api/codex"
CODEX_AUTH_CONFIG_REFRESH_SECONDS: float = 60.0

register_known_model_costs()

# Register models that litellm doesn't know about yet so that
# supports_native_streaming() returns True and litellm doesn't
# downgrade to fake_stream (which strips "stream": true from
# the request body — fatal for the Codex backend).
_GPT54_ENTRY = litellm.model_cost.get("gpt-5.4", {})
for _slug in ("gpt-5.3-codex-spark", "gpt-5.5"):
    if _slug not in litellm.model_cost and _GPT54_ENTRY:
        litellm.model_cost[_slug] = {**_GPT54_ENTRY}
        litellm.model_cost[f"openai/{_slug}"] = {**_GPT54_ENTRY}

# Retry config for transient Codex stream failures (rate limits,
# incomplete streams, dropped connections). 4 total attempts (initial +
# 3 retries) with jittered exponential backoff — each retry waits a
# uniform random duration in [0, min(multiplier·2^(n-1), max)] so N
# concurrent callers hitting the same rate-limit window fan out across
# the backoff window instead of retrying in lockstep (which just
# re-concentrates load on the endpoint that was already struggling).
# Exposed as module-level knobs so callers / tests can override.
CODEX_STREAM_MAX_ATTEMPTS: int = 4
CODEX_STREAM_WAIT_MULTIPLIER: float = 2.0
CODEX_STREAM_WAIT_MAX: float = 8.0
# Max wall-clock between consecutive SSE events before we consider the
# stream hung and raise CodexStreamError (caught by the tenacity retry).
# Without this ceiling, a Codex connection that received HTTP 200 headers
# but then goes silent leaves ``async for event in stream:`` blocked
# indefinitely — asyncio.wait_for on the caller cannot cancel an inner
# socket read that never hits an await point, and TCP keepalive is 2
# hours by default on macOS. 30s comfortably covers slow first-token
# latency and normal inter-token gaps while keeping stalls observable.
CODEX_STREAM_HEARTBEAT_SEC: float = 30.0

logger = logging.getLogger(__name__)


def _codex_retry_kwargs() -> dict[str, Any]:
    """Build tenacity retry kwargs from the module-level knobs.

    Read at call time (not decorator time) so tests can monkeypatch
    the module-level constants without having to re-import lm.py.
    """
    return dict(
        retry=retry_if_exception_type(CodexStreamError),
        stop=stop_after_attempt(CODEX_STREAM_MAX_ATTEMPTS),
        wait=wait_random_exponential(
            multiplier=CODEX_STREAM_WAIT_MULTIPLIER,
            max=CODEX_STREAM_WAIT_MAX,
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


async def _aiter_stream_with_heartbeat(stream: Any):
    """Yield events from an async SSE stream, bounding each read.

    If no event arrives for ``CODEX_STREAM_HEARTBEAT_SEC`` seconds, raise
    ``CodexStreamError`` so the outer ``AsyncRetrying`` can recover with
    a fresh connection. This is the only structural guarantee that a
    hung socket read terminates in bounded time — an ``asyncio.wait_for``
    around the caller cannot cancel an inner read that never reaches an
    ``await`` point, and the kernel's TCP keepalive is 2 hours by default.
    """
    aiter = stream.__aiter__()
    while True:
        try:
            event = await asyncio.wait_for(
                aiter.__anext__(),
                timeout=CODEX_STREAM_HEARTBEAT_SEC,
            )
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError:
            raise CodexStreamError(
                f"Codex stream stalled — no SSE event received within "
                f"{CODEX_STREAM_HEARTBEAT_SEC}s heartbeat window"
            )
        yield event


_STREAM_END_SENTINEL = object()


def _iter_stream_with_heartbeat(stream: Any):
    """Sync variant: bound each event read using a ``ThreadPoolExecutor``
    watchdog. Mirrors the async version's contract — a stream that goes
    silent for ``CODEX_STREAM_HEARTBEAT_SEC`` seconds raises
    ``CodexStreamError`` so the outer ``Retrying`` loop can retry.

    We use a thread rather than ``signal.alarm`` because alarms are
    unix-only, don't compose with other signal handlers, and fire at
    process scope rather than per-stream.
    """
    import concurrent.futures

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        aiter = iter(stream)
        while True:
            future = executor.submit(next, aiter, _STREAM_END_SENTINEL)
            try:
                event = future.result(timeout=CODEX_STREAM_HEARTBEAT_SEC)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise CodexStreamError(
                    f"Codex stream stalled — no SSE event received within "
                    f"{CODEX_STREAM_HEARTBEAT_SEC}s heartbeat window"
                )
            if event is _STREAM_END_SENTINEL:
                return
            yield event
    finally:
        executor.shutdown(wait=False)


def _install_pydantic_warning_filter() -> None:
    """Silence ``PydanticSerializationUnexpectedValue(... ResponseAPIUsage ...)``.

    When a downstream pydantic model declares a field as
    ``ResponseAPIUsage`` and receives our enriched ``ResponseAPIUsage``
    (which carries Chat-Completions-shaped aliases ``prompt_tokens`` and
    ``completion_tokens`` so downstream consumers like DSPy's history
    and cost trackers see populated token counts), pydantic's serializer
    warns that the value "may not be as expected". The extras are
    intentional and load-bearing — the warning is just noise.
    """
    # `warnings.filterwarnings` compiles the message via ``re.match`` (no
    # ``re.DOTALL``) so ``.*`` won't cross the newline in pydantic's
    # multi-line message. Use ``(?s)`` to enable DOTALL inline so the
    # pattern matches from the start through "ResponseAPIUsage" regardless
    # of embedded newlines.
    warnings.filterwarnings(
        "ignore",
        message=r"(?s).*ResponseAPIUsage.*",
        category=UserWarning,
    )


_install_pydantic_warning_filter()


class CodexStreamError(RuntimeError):
    """Raised when the Codex response stream emits a failure event (rate
    limit, backend error, incomplete response) or ends without ever
    completing. Contains the upstream error code/message when available.
    """


def _g(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@dataclass(frozen=True)
class _AuthCredentials:
    access_token: str
    account_id: str


@dataclass(frozen=True)
class _AuthConfigSnapshot:
    credentials: tuple[_AuthCredentials, ...] = ()
    error: Exception | None = None


class CodexLM(dspy.LM):
    """DSPy LM that routes calls through the ChatGPT subscription backend.

    Reads the OAuth access token from ``~/.codex/auth.json`` (or a provided
    path / explicit token). Streams the Responses API, assembles a full
    ``ResponsesAPIResponse``, stamps it with real-rate cost from
    ``litellm.model_cost``, and routes everything through DSPy's cache layer.

    Sync (``forward``) and async (``aforward``) are both supported.
    """

    def __init__(
        self,
        model: str = "gpt-5.3-codex",
        instructions: str = "You are a helpful assistant.",
        access_token: str | None = None,
        account_id: str | None = None,
        auth_path: str | Path | None = None,
        auth_profile: str | None = None,
        auth_config_refresh_seconds: float = CODEX_AUTH_CONFIG_REFRESH_SECONDS,
        **kwargs,
    ):
        self._auth_path = auth_path
        self._auth_profile = auth_profile
        self._pinned_access_token = access_token
        self._pinned_account_id = account_id
        self._auth_config_refresh_seconds = max(
            0.0,
            float(auth_config_refresh_seconds),
        )
        self._auth_config_snapshot: _AuthConfigSnapshot | None = None
        self._auth_config_snapshot_at: float | None = None
        self._auth_config_cursor = 0
        if access_token is None or account_id is None:
            if auth_path is not None or auth_profile is not None:
                access_token, account_id = load_codex_auth(
                    auth_path,
                    profile=auth_profile,
                )
                self._pinned_access_token = access_token
                self._pinned_account_id = account_id
            else:
                access_token = "codex-lm-runtime-auth"
                account_id = "codex-lm-runtime-auth"
        super().__init__(
            model=f"openai/{model}",
            model_type="responses",
            api_base=CODEX_BASE,
            api_key=access_token,
            **kwargs,
        )
        self._instructions = instructions

    # ------- request / event plumbing -------

    def _build_request(
        self,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        messages = messages or [{"role": "user", "content": prompt}]
        merged = {**self.kwargs, **kwargs}
        merged.pop("rollout_id", None)
        merged.pop("cache", None)

        # DSPy's converter blindly wraps reasoning_effort into
        # ``reasoning: {effort: ..., summary: "auto"}``. If the user passed
        # ``None`` explicitly (or left the default unset), that becomes
        # ``effort: null`` which Codex 400s on. Drop it so the field is
        # absent from the request and Codex uses its default.
        if merged.get("reasoning_effort") is None:
            merged.pop("reasoning_effort", None)

        request = dict(model=self.model, messages=messages, **merged)
        request = _convert_chat_request_to_responses_request(request)
        if request["model"].startswith("openai/"):
            request["model"] = request["model"].split("/", 1)[1]

        access_token, account_id = self._request_auth()
        request["api_key"] = access_token

        request["store"] = False
        request["stream"] = True
        request["custom_llm_provider"] = "openai"
        request.setdefault("instructions", self._instructions)

        headers = dict(request.pop("headers", None) or {})
        headers["ChatGPT-Account-Id"] = account_id
        headers["originator"] = "opencode"
        headers["session_id"] = str(uuid.uuid4())
        return request, headers

    def _request_auth(self) -> tuple[str, str]:
        if self._pinned_access_token is not None and self._pinned_account_id is not None:
            return self._pinned_access_token, self._pinned_account_id
        snapshot = self._cached_auth_config_snapshot()
        if snapshot.error is not None:
            raise snapshot.error
        if not snapshot.credentials:
            raise FileNotFoundError("no codex-lm auth credentials available")
        index = self._auth_config_cursor % len(snapshot.credentials)
        credentials = snapshot.credentials[index]
        self._auth_config_cursor = (index + 1) % len(snapshot.credentials)
        return credentials.access_token, credentials.account_id

    def _cached_auth_config_snapshot(self) -> _AuthConfigSnapshot:
        now = monotonic()
        if (
            self._auth_config_snapshot is None
            or self._auth_config_snapshot_at is None
            or now - self._auth_config_snapshot_at >= self._auth_config_refresh_seconds
        ):
            self._refresh_auth_config_snapshot(now)
        if self._auth_config_snapshot is None:
            raise RuntimeError("auth config snapshot was not initialized")
        return self._auth_config_snapshot

    def _refresh_auth_config_snapshot(self, now: float) -> None:
        try:
            snapshot, cursor = self._load_auth_config_snapshot()
        except Exception as exc:
            snapshot = _AuthConfigSnapshot(error=exc)
            cursor = 0
        self._auth_config_snapshot = snapshot
        self._auth_config_snapshot_at = now
        self._auth_config_cursor = cursor

    def _load_auth_config_snapshot(self) -> tuple[_AuthConfigSnapshot, int]:
        source = resolve_auth_source(
            self._auth_path,
            profile=self._auth_profile,
            advance_rotation=False,
        )
        if source.source != "rotation":
            access_token, account_id = load_codex_auth(source.path)
            return (
                _AuthConfigSnapshot(credentials=(_AuthCredentials(access_token, account_id),)),
                0,
            )

        credentials = []
        cursor = 0
        for index, profile in enumerate(list_enabled_auth_profiles()):
            access_token, account_id = load_codex_auth(profile_auth_path(profile))
            credentials.append(_AuthCredentials(access_token, account_id))
            if profile == source.profile:
                cursor = index
        if not credentials:
            raise ValueError(
                "auth profile rotation is enabled but no enabled auth profiles "
                "are available; run `codex-lm auth enable NAME` or "
                "`codex-lm rotation off`"
            )
        return _AuthConfigSnapshot(credentials=tuple(credentials)), cursor

    def _fresh_state(self) -> dict[str, Any]:
        return {
            "text_parts": [],
            "usage_raw": None,
            "response_id": "codex-stream",
            "model_name": self.model,
            "completed": False,
            "failure": None,
        }

    def _handle_event(self, event: Any, state: dict[str, Any]) -> None:
        etype = _g(event, "type")
        if etype == "response.output_text.delta":
            delta = _g(event, "delta")
            if delta:
                state["text_parts"].append(delta)
        elif etype == "response.completed":
            raw = _g(event, "response")
            state["usage_raw"] = _g(raw, "usage") or state["usage_raw"]
            state["response_id"] = _g(raw, "id", state["response_id"])
            state["model_name"] = _g(raw, "model", state["model_name"])
            state["completed"] = True
        elif etype == "response.failed":
            raw = _g(event, "response")
            err = _g(raw, "error")
            state["failure"] = {
                "kind": "failed",
                "code": _g(err, "code"),
                "message": _g(err, "message") or "response.failed (no message)",
            }
        elif etype == "response.incomplete":
            raw = _g(event, "response")
            details = _g(raw, "incomplete_details")
            state["failure"] = {
                "kind": "incomplete",
                "code": _g(details, "reason"),
                "message": _g(details, "reason") or "response.incomplete",
            }
        elif etype == "error":
            state["failure"] = {
                "kind": "error",
                "code": _g(event, "code"),
                "message": _g(event, "message") or "error (no message)",
            }

    def _raise_if_failed(self, state: dict[str, Any]) -> None:
        failure = state["failure"]
        if failure is not None:
            msg = (
                f"Codex stream {failure['kind']}"
                + (f" ({failure['code']})" if failure.get("code") else "")
                + f": {failure['message']}"
            )
            logger.warning("codex stream failure: %s", msg)
            raise CodexStreamError(msg)
        if not state["completed"]:
            msg = (
                "Codex stream ended without a response.completed event "
                "(likely rate-limited, throttled, or dropped upstream)"
            )
            logger.warning(msg)
            raise CodexStreamError(msg)

    def _assemble(
        self,
        text_parts: list[str],
        usage_raw: Any,
        response_id: str,
        model_name: str,
    ) -> ResponsesAPIResponse:
        text = "".join(text_parts)
        output_items = [
            ResponseOutputMessage(
                id="msg-1",
                type="message",
                role="assistant",
                status="completed",
                content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
            )
        ]

        if usage_raw is None:
            usage_dict: dict[str, Any] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }
        elif hasattr(usage_raw, "model_dump"):
            usage_dict = usage_raw.model_dump()
        elif isinstance(usage_raw, dict):
            usage_dict = dict(usage_raw)
        else:
            usage_dict = dict(usage_raw)

        # Chat Completions-style aliases so downstream consumers that read
        # LiteLLM's Chat Completions usage shape (DSPy history, cost trackers,
        # per-role summaries) see non-zero token counts alongside the
        # Responses-API-native input_tokens/output_tokens fields.
        usage_dict["prompt_tokens"] = usage_dict.get("input_tokens", 0)
        usage_dict["completion_tokens"] = usage_dict.get("output_tokens", 0)

        # For pricing lookup, prefer the requested model slug (self.model —
        # e.g. "openai/gpt-5.4-mini") over the Codex backend's reported
        # model name (e.g. "gpt-5.4-mini-2026-03-17"), which is a dated
        # snapshot identifier and is typically NOT in LiteLLM's price
        # registry. Falling back to model_name preserves behaviour for
        # any caller that constructed CodexLM without self.model set.
        pricing_model = getattr(self, "model", None) or model_name
        cost = compute_cost(pricing_model, usage_dict)
        usage_dict["cost"] = cost
        usage_obj = ResponseAPIUsage.model_validate(usage_dict)

        response = ResponsesAPIResponse(
            id=response_id,
            created_at=0,
            object="response",
            output=output_items,
            model=model_name,
            status="completed",
            usage=usage_obj,
        )
        response._hidden_params = {"response_cost": cost}
        return response

    # ------- sync path -------

    def forward(self, prompt=None, messages=None, **kwargs):
        request, headers = self._build_request(prompt, messages, kwargs)
        cache = kwargs.pop("cache", self.cache)

        def _completion(request, num_retries, cache):
            for attempt in Retrying(**_codex_retry_kwargs()):
                with attempt:
                    stream = litellm.responses(
                        headers=headers,
                        num_retries=num_retries,
                        **request,
                    )
                    state = self._fresh_state()
                    for event in _iter_stream_with_heartbeat(stream):
                        self._handle_event(event, state)
                    self._raise_if_failed(state)
                    return self._assemble(
                        text_parts=state["text_parts"],
                        usage_raw=state["usage_raw"],
                        response_id=state["response_id"],
                        model_name=state["model_name"],
                    )

        completion_fn, litellm_cache_args = self._get_cached_completion_fn(_completion, cache)
        results = completion_fn(
            request=request,
            num_retries=self.num_retries,
            cache=litellm_cache_args,
        )
        _fire_usage_tracker_hook(self.model, results)
        return results

    # ------- async path -------

    async def aforward(self, prompt=None, messages=None, **kwargs):
        request, headers = self._build_request(prompt, messages, kwargs)
        cache = kwargs.pop("cache", self.cache)

        async def _acompletion(request, num_retries, cache):
            async for attempt in AsyncRetrying(**_codex_retry_kwargs()):
                with attempt:
                    stream = await litellm.aresponses(
                        headers=headers,
                        num_retries=num_retries,
                        **request,
                    )
                    state = self._fresh_state()
                    async for event in _aiter_stream_with_heartbeat(stream):
                        self._handle_event(event, state)
                    self._raise_if_failed(state)
                    return self._assemble(
                        text_parts=state["text_parts"],
                        usage_raw=state["usage_raw"],
                        response_id=state["response_id"],
                        model_name=state["model_name"],
                    )

        completion_fn, litellm_cache_args = self._get_cached_completion_fn(_acompletion, cache)
        results = await completion_fn(
            request=request,
            num_retries=self.num_retries,
            cache=litellm_cache_args,
        )
        _fire_usage_tracker_hook(self.model, results)
        return results


def _fire_usage_tracker_hook(model: str, results: Any) -> None:
    """Replicate ``dspy.clients.lm.LM.(a)forward``'s usage_tracker hook.

    DSPy's LM base class fires this at lines 167 and 205 of ``clients/lm.py``
    after each call so ``dspy.track_usage()`` can attribute tokens to the
    prediction. CodexLM overrides the whole method, which historically
    bypassed the hook and left ``pred.get_lm_usage()`` empty — the reason
    every downstream cost tracker saw $0 after routing through CodexLM.
    Mirror DSPy's conditions exactly: skip cache hits, require a usage
    tracker in context, and require the response to have a ``usage`` field.
    """
    if getattr(results, "cache_hit", False):
        return
    tracker = getattr(dspy.settings, "usage_tracker", None)
    if tracker is None:
        return
    if not hasattr(results, "usage"):
        return
    try:
        tracker.add_usage(model, dict(results.usage))
    except Exception as e:
        logger.debug("codex-lm usage_tracker.add_usage failed: %s", e)
