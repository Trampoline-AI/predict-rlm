import asyncio
import json
import logging
import os
import random
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any
from urllib.parse import urlparse, urlunparse

import aiohttp
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
from tenacity.wait import wait_base

from dspy_codex_lm.auth import (
    list_enabled_auth_profiles,
    load_codex_auth,
    profile_auth_path,
    resolve_auth_source,
)
from dspy_codex_lm.cost import compute_cost, register_known_model_costs

CODEX_BASE = "https://chatgpt.com/backend-api/codex"
CODEX_WS_BETA_HEADER = "responses_websockets=2026-02-06"
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
# incomplete streams, dropped connections). 5 total attempts (initial +
# 4 retries). When Codex includes a retry-after delay on a stream error,
# honor that server-requested delay; otherwise use jittered exponential
# backoff so concurrent callers fan out instead of retrying in lockstep.
# Exposed as module-level knobs so callers / tests can override.
DEFAULT_CODEX_STREAM_MAX_ATTEMPTS: int = 5
CODEX_STREAM_MAX_ATTEMPTS: int = DEFAULT_CODEX_STREAM_MAX_ATTEMPTS
CODEX_STREAM_WAIT_MULTIPLIER: float = 2.0
CODEX_STREAM_WAIT_MAX: float = 8.0
# Max wall-clock between consecutive SSE events before we consider the
# stream hung and raise CodexStreamError (caught by the tenacity retry).
# Without this ceiling, a Codex connection that received HTTP 200 headers
# but then goes silent leaves ``async for event in stream:`` blocked
# indefinitely — asyncio.wait_for on the caller cannot cancel an inner
# socket read that never hits an await point, and TCP keepalive is 2
# hours by default on macOS. Match upstream Codex's 300s default idle
# timeout while still making the watchdog configurable for focused tests
# and more aggressive caller-specific deployments.
CODEX_STREAM_HEARTBEAT_SEC: float = float(
    os.environ.get("CODEX_STREAM_HEARTBEAT_SEC", "300.0")
)
DEFAULT_CODEX_MODEL = "gpt-5.5"

logger = logging.getLogger(__name__)

_DEBUG_TRUE_VALUES = {"1", "true", "yes", "on"}


@contextmanager
def _scoped_proxy_env(proxy_url: str | None):
    if proxy_url is None:
        yield
        return

    names = ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = proxy_url
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _debug_env_enabled() -> bool:
    return any(
        os.environ.get(name, "").strip().lower() in _DEBUG_TRUE_VALUES
        for name in (
            "PREDICT_RLM_DEBUG",
            "RLM_DEBUG",
            "CODEX_LM_DEBUG",
            "DSPY_CODEX_LM_DEBUG",
        )
    )


def _codex_debug_event(event: str, **metadata: Any) -> None:
    if not _debug_env_enabled():
        return

    dedicated_log_path = os.environ.get("CODEX_LM_DEBUG_LOG")
    try:
        from predict_rlm.debug import debug_event, sanitize_metadata
    except Exception:
        sanitize_metadata = None
    else:
        if (
            os.environ.get("PREDICT_RLM_DEBUG") or os.environ.get("RLM_DEBUG")
        ) and not dedicated_log_path:
            debug_event(event, **metadata)
            return

    payload = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "event": event,
        **(sanitize_metadata(metadata) if sanitize_metadata is not None else metadata),
    }
    line = json.dumps(payload, sort_keys=True, default=str)
    log_path = dedicated_log_path or os.environ.get("PREDICT_RLM_DEBUG_LOG")
    if log_path:
        try:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            return
        except Exception:
            pass
    logger.debug(line)


def _elapsed_ms(start: float, end: float | None) -> float | None:
    if end is None:
        return None
    return round((end - start) * 1000.0, 3)


def _prompt_cache_stats(usage: Any) -> dict[str, Any]:
    prompt_tokens = _coerce_int(_g(usage, "prompt_tokens"))
    if prompt_tokens is None:
        prompt_tokens = _coerce_int(_g(usage, "input_tokens"))
    details = _g(usage, "prompt_tokens_details") or _g(usage, "input_tokens_details")
    cached_prompt_tokens = _coerce_int(_g(details, "cached_tokens"))
    stats: dict[str, Any] = {}
    if prompt_tokens is not None:
        stats["prompt_tokens"] = prompt_tokens
    if cached_prompt_tokens is not None:
        stats["cached_prompt_tokens"] = cached_prompt_tokens
        stats["prompt_cache_read_ratio"] = (
            cached_prompt_tokens / prompt_tokens if prompt_tokens else 0.0
        )
    return stats


def _request_debug_metadata(request: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    reasoning = request.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort") is not None:
        metadata["reasoning_effort"] = reasoning["effort"]
    if request.get("service_tier") is not None:
        metadata["service_tier"] = request["service_tier"]
    return metadata


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_output_text_delta_type(event_type: Any) -> bool:
    if event_type == "response.output_text.delta":
        return True
    return str(event_type).endswith("OUTPUT_TEXT_DELTA")


@dataclass
class _StreamTiming:
    model: str
    attempt_number: int | None
    start_at: float
    transport: str
    auth_metadata: dict[str, Any] | None = None
    request_metadata: dict[str, Any] | None = None
    first_event_at: float | None = None
    first_text_delta_at: float | None = None
    stream_end_at: float | None = None
    output_text_chars: int = 0
    events_before_first_text: int = 0
    max_inter_event_gap_ms: float | None = None
    last_event_type: str | None = None
    last_event_at: float | None = None
    non_text_event_counts: dict[str, int] | None = None

    def emit_start(self) -> None:
        _codex_debug_event(
            "codex_lm.stream.start",
            model=self.model,
            attempt_number=self.attempt_number,
            transport=self.transport,
            **(self.request_metadata or {}),
            **(self.auth_metadata or {}),
        )

    def observe_event(self, event: Any) -> None:
        now = monotonic()
        etype = str(_g(event, "type") or "unknown")
        self._record_event_gap(now)
        self.last_event_type = etype
        self.last_event_at = now
        if self.first_event_at is None:
            self.first_event_at = now
            _codex_debug_event(
                "codex_lm.stream.first_event",
                model=self.model,
                attempt_number=self.attempt_number,
                transport=self.transport,
                first_event_type=etype,
                tt_first_event_ms=_elapsed_ms(self.start_at, now),
                **(self.request_metadata or {}),
                **(self.auth_metadata or {}),
            )
        if _is_output_text_delta_type(etype):
            delta = _g(event, "delta")
            if delta:
                self.output_text_chars += len(delta)
                if self.first_text_delta_at is None:
                    self.first_text_delta_at = now
                    _codex_debug_event(
                        "codex_lm.stream.first_text_delta",
                        model=self.model,
                        attempt_number=self.attempt_number,
                        transport=self.transport,
                        tt_first_event_ms=_elapsed_ms(
                            self.start_at,
                            self.first_event_at,
                        ),
                        ttft_ms=_elapsed_ms(self.start_at, now),
                        first_delta_chars=len(delta),
                        **(self.request_metadata or {}),
                        **(self.auth_metadata or {}),
                    )
            elif self.first_text_delta_at is None:
                self.events_before_first_text += 1
        else:
            if self.non_text_event_counts is None:
                self.non_text_event_counts = {}
            self.non_text_event_counts[etype] = self.non_text_event_counts.get(etype, 0) + 1
            if self.first_text_delta_at is None:
                self.events_before_first_text += 1

    def _record_event_gap(self, now: float) -> None:
        if self.last_event_at is None:
            return
        gap_ms = _elapsed_ms(self.last_event_at, now)
        if gap_ms is None:
            return
        if self.max_inter_event_gap_ms is None or gap_ms > self.max_inter_event_gap_ms:
            self.max_inter_event_gap_ms = gap_ms

    def _event_gap_metadata(self, now: float | None = None) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "events_before_first_text": self.events_before_first_text,
            "max_inter_event_gap_ms": self.max_inter_event_gap_ms,
            "last_event_type": self.last_event_type,
            "non_text_event_counts": self.non_text_event_counts or {},
        }
        if now is not None and self.last_event_at is not None:
            metadata["last_event_age_ms"] = _elapsed_ms(self.last_event_at, now)
        else:
            metadata["last_event_age_ms"] = None
        return metadata

    def mark_stream_end(self) -> None:
        self.stream_end_at = monotonic()

    def emit_end(self, state: dict[str, Any], *, finished_at: float) -> None:
        _codex_debug_event(
            "codex_lm.stream.end",
            model=self.model,
            attempt_number=self.attempt_number,
            transport=self.transport,
            tt_first_event_ms=_elapsed_ms(self.start_at, self.first_event_at),
            ttft_ms=_elapsed_ms(self.start_at, self.first_text_delta_at),
            stream_total_ms=_elapsed_ms(self.start_at, self.stream_end_at),
            parse_overhead_ms=_elapsed_ms(self.stream_end_at, finished_at)
            if self.stream_end_at is not None
            else None,
            output_text_chars=self.output_text_chars,
            completed=bool(state.get("completed")),
            **(self.request_metadata or {}),
            **(self.auth_metadata or {}),
            **self._event_gap_metadata(finished_at),
            **_prompt_cache_stats(state.get("usage_raw")),
        )

    def emit_error(self, state: dict[str, Any], exc: BaseException) -> None:
        if self.stream_end_at is None:
            self.mark_stream_end()
        failure = state.get("failure") or {}
        _codex_debug_event(
            "codex_lm.stream.error",
            model=self.model,
            attempt_number=self.attempt_number,
            transport=self.transport,
            tt_first_event_ms=_elapsed_ms(self.start_at, self.first_event_at),
            ttft_ms=_elapsed_ms(self.start_at, self.first_text_delta_at),
            stream_total_ms=_elapsed_ms(self.start_at, self.stream_end_at),
            output_text_chars=self.output_text_chars,
            completed=bool(state.get("completed")),
            failure_kind=(
                failure.get("kind") or getattr(exc, "failure_kind", None) or type(exc).__name__
            ),
            failure_code=failure.get("code") or getattr(exc, "failure_code", None),
            exception_type=type(exc).__name__,
            **(self.request_metadata or {}),
            **(self.auth_metadata or {}),
            **self._event_gap_metadata(self.stream_end_at),
        )


def _codex_retry_kwargs() -> dict[str, Any]:
    """Build tenacity retry kwargs from the module-level knobs.

    Read at call time (not decorator time) so tests can monkeypatch
    the module-level constants without having to re-import lm.py.
    """
    return dict(
        retry=retry_if_exception_type(CodexStreamError),
        stop=stop_after_attempt(CODEX_STREAM_MAX_ATTEMPTS),
        wait=_WaitServerRetryAfterOrRandomExponential(),
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


def _disable_litellm_stream_logging(stream: Any) -> Any:
    """Keep LiteLLM stream logging from reassembling Codex responses.

    CodexLM consumes the stream, assembles the final ResponsesAPIResponse, and
    fires DSPy's usage tracker itself. Recent LiteLLM versions try to do a
    second private logging pass on ``response.completed`` and can receive a
    dict-shaped ``response`` from the Codex backend, which raises before
    CodexLM can assemble the result.
    """

    def noop(*args, **kwargs):
        return None

    for name in (
        "_handle_logging_completed_response",
        "_handle_logging_failed_response",
    ):
        if hasattr(stream, name):
            try:
                setattr(stream, name, noop)
            except Exception:
                pass
    return stream


def _coerce_retry_after_seconds(value: Any) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        return None
    return seconds


def _first_present(obj: Any, *keys: str) -> Any:
    for key in keys:
        value = _g(obj, key)
        if value is not None:
            return value
    return None


def _retry_after_seconds_from_error(error: Any) -> float | None:
    seconds = _coerce_retry_after_seconds(
        _first_present(
            error,
            "retry_after_seconds",
            "retryAfterSeconds",
            "retry_after",
            "retryAfter",
        )
    )
    if seconds is not None:
        return seconds
    milliseconds = _coerce_retry_after_seconds(
        _first_present(error, "retry_after_ms", "retryAfterMs")
    )
    if milliseconds is None:
        return None
    return milliseconds / 1000.0


class CodexStreamError(RuntimeError):
    """Raised when the Codex response stream fails before completion."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        failure_kind: str | None = None,
        failure_code: str | int | None = None,
    ):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
        self.failure_kind = failure_kind
        self.failure_code = failure_code


class _WaitServerRetryAfterOrRandomExponential(wait_base):
    def __init__(self):
        self._fallback = wait_random_exponential(
            multiplier=CODEX_STREAM_WAIT_MULTIPLIER,
            max=CODEX_STREAM_WAIT_MAX,
        )

    def __call__(self, retry_state: Any) -> float:
        if retry_state.outcome is not None:
            exception = retry_state.outcome.exception()
            retry_after = getattr(exception, "retry_after_seconds", None)
            if retry_after is not None:
                return retry_after
        return self._fallback(retry_state)


def _g(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _codex_stream_error_from_state(state: dict[str, Any]) -> CodexStreamError | None:
    failure = state["failure"]
    if failure is None:
        return None
    msg = (
        f"Codex stream {failure['kind']}"
        + (f" ({failure['code']})" if failure.get("code") else "")
        + f": {failure['message']}"
    )
    return CodexStreamError(
        msg,
        retry_after_seconds=failure.get("retry_after_seconds"),
        failure_kind=failure.get("kind"),
        failure_code=failure.get("code"),
    )


@dataclass(frozen=True)
class _AuthCredentials:
    access_token: str
    account_id: str
    profile: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class _AuthConfigSnapshot:
    credentials: tuple[_AuthCredentials, ...] = ()
    error: Exception | None = None
    rotating: bool = False


class CodexHTTPLM(dspy.LM):
    """DSPy LM that routes calls through the ChatGPT subscription HTTP/SSE backend.

    Reads the OAuth access token from ``~/.codex/auth.json`` (or a provided
    path / explicit token). Streams the Responses API, assembles a full
    ``ResponsesAPIResponse``, stamps it with real-rate cost from
    ``litellm.model_cost``, and routes everything through DSPy's cache layer.

    Sync (``forward``) and async (``aforward``) are both supported.
    """

    def __init__(
        self,
        model: str = DEFAULT_CODEX_MODEL,
        instructions: str = "You are a helpful assistant.",
        access_token: str | None = None,
        account_id: str | None = None,
        auth_path: str | Path | None = None,
        auth_profile: str | None = None,
        auth_config_refresh_seconds: float = CODEX_AUTH_CONFIG_REFRESH_SECONDS,
        **kwargs,
    ):
        self._proxy_url = kwargs.pop("proxy_url", None)
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
        proxy_url = merged.pop("proxy_url", self._proxy_url)

        request = dict(model=self.model, messages=messages, **merged)
        request = _convert_chat_request_to_responses_request(request)
        if request["model"].startswith("openai/"):
            request["model"] = request["model"].split("/", 1)[1]

        request["store"] = False
        request["stream"] = True
        request["custom_llm_provider"] = "openai"
        request.setdefault("instructions", self._instructions)

        headers = dict(request.pop("headers", None) or {})
        self._apply_request_auth(request, headers)
        headers["originator"] = "opencode"
        headers["session_id"] = str(uuid.uuid4())
        request["_codex_lm_proxy_url"] = proxy_url
        return request, headers

    def _request_auth(
        self,
        *,
        exclude_profile: str | None = None,
        exclude_account_id: str | None = None,
    ) -> _AuthCredentials:
        if self._pinned_access_token is not None and self._pinned_account_id is not None:
            return _AuthCredentials(
                self._pinned_access_token,
                self._pinned_account_id,
                profile=self._auth_profile,
                source="profile" if self._auth_profile is not None else "pinned",
            )
        snapshot = self._cached_auth_config_snapshot()
        if snapshot.error is not None:
            raise snapshot.error
        if not snapshot.credentials:
            raise FileNotFoundError("no codex-lm auth credentials available")
        credentials = snapshot.credentials
        if not snapshot.rotating:
            return credentials[0]
        if len(credentials) > 1 and (exclude_profile is not None or exclude_account_id is not None):
            alternates = tuple(
                credential
                for credential in credentials
                if credential.profile != exclude_profile
                and credential.account_id != exclude_account_id
            )
            if len(alternates) == 1:
                return alternates[0]
            if alternates:
                credentials = alternates
        return random.choice(credentials)

    def _apply_request_auth(
        self,
        request: dict[str, Any],
        headers: dict[str, str],
        *,
        exclude_profile: str | None = None,
        exclude_account_id: str | None = None,
    ) -> dict[str, Any]:
        credentials = self._request_auth(
            exclude_profile=exclude_profile,
            exclude_account_id=exclude_account_id,
        )
        request["api_key"] = credentials.access_token
        headers["ChatGPT-Account-Id"] = credentials.account_id
        metadata = self._auth_debug_metadata(credentials)
        request["_codex_lm_auth_metadata"] = metadata
        return metadata

    def _auth_debug_metadata(self, credentials: _AuthCredentials) -> dict[str, Any]:
        metadata: dict[str, Any] = {"auth_source": credentials.source}
        if credentials.profile is not None:
            metadata["auth_profile"] = credentials.profile
        return metadata

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
            snapshot = self._load_auth_config_snapshot()
        except Exception as exc:
            snapshot = _AuthConfigSnapshot(error=exc)
        self._auth_config_snapshot = snapshot
        self._auth_config_snapshot_at = now

    def _load_auth_config_snapshot(self) -> _AuthConfigSnapshot:
        source = resolve_auth_source(
            self._auth_path,
            profile=self._auth_profile,
            advance_rotation=False,
        )
        if source.source != "rotation":
            access_token, account_id = load_codex_auth(source.path)
            return _AuthConfigSnapshot(
                credentials=(
                    _AuthCredentials(
                        access_token,
                        account_id,
                        profile=source.profile,
                        source=source.source,
                    ),
                ),
            )

        credentials = []
        for profile in list_enabled_auth_profiles():
            access_token, account_id = load_codex_auth(profile_auth_path(profile))
            credentials.append(
                _AuthCredentials(
                    access_token,
                    account_id,
                    profile=profile,
                    source="rotation",
                )
            )
        if not credentials:
            raise ValueError(
                "auth profile rotation is enabled but no enabled auth profiles "
                "are available; run `codex-lm auth enable NAME` or "
                "`codex-lm rotation off`"
            )
        return _AuthConfigSnapshot(credentials=tuple(credentials), rotating=True)

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
        if _is_output_text_delta_type(etype):
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
                "retry_after_seconds": _retry_after_seconds_from_error(err),
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
        error = _codex_stream_error_from_state(state)
        if error is not None:
            logger.warning("codex stream failure: %s", error)
            raise error

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
        auth_metadata = request.pop("_codex_lm_auth_metadata", {})
        proxy_url = request.pop("_codex_lm_proxy_url", None)
        cache = kwargs.pop("cache", self.cache)

        def _completion(request, num_retries, cache):
            nonlocal auth_metadata
            for attempt in Retrying(**_codex_retry_kwargs()):
                with attempt:
                    if attempt.retry_state.attempt_number > 1:
                        auth_metadata = self._apply_request_auth(
                            request,
                            headers,
                            exclude_profile=auth_metadata.get("auth_profile"),
                        )
                        request.pop("_codex_lm_auth_metadata", None)
                    timing = _StreamTiming(
                        model=str(request.get("model") or self.model),
                        attempt_number=attempt.retry_state.attempt_number,
                        start_at=monotonic(),
                        transport="http_sse",
                        request_metadata=_request_debug_metadata(request),
                        auth_metadata=auth_metadata,
                    )
                    state = self._fresh_state()
                    timing.emit_start()
                    try:
                        with _scoped_proxy_env(proxy_url):
                            stream = litellm.responses(
                                headers=headers,
                                num_retries=num_retries,
                                **request,
                            )
                        stream = _disable_litellm_stream_logging(stream)
                        for event in _iter_stream_with_heartbeat(stream):
                            timing.observe_event(event)
                            self._handle_event(event, state)
                            if state["completed"] or state["failure"] is not None:
                                break
                        timing.mark_stream_end()
                        self._raise_if_failed(state)
                        result = self._assemble(
                            text_parts=state["text_parts"],
                            usage_raw=state["usage_raw"],
                            response_id=state["response_id"],
                            model_name=state["model_name"],
                        )
                        timing.emit_end(state, finished_at=monotonic())
                        return result
                    except Exception as exc:
                        timing.emit_error(state, exc)
                        raise

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
        auth_metadata = request.pop("_codex_lm_auth_metadata", {})
        proxy_url = request.pop("_codex_lm_proxy_url", None)
        cache = kwargs.pop("cache", self.cache)

        async def _acompletion(request, num_retries, cache):
            nonlocal auth_metadata
            async for attempt in AsyncRetrying(**_codex_retry_kwargs()):
                with attempt:
                    if attempt.retry_state.attempt_number > 1:
                        auth_metadata = self._apply_request_auth(
                            request,
                            headers,
                            exclude_profile=auth_metadata.get("auth_profile"),
                        )
                        request.pop("_codex_lm_auth_metadata", None)
                    timing = _StreamTiming(
                        model=str(request.get("model") or self.model),
                        attempt_number=attempt.retry_state.attempt_number,
                        start_at=monotonic(),
                        transport="http_sse",
                        request_metadata=_request_debug_metadata(request),
                        auth_metadata=auth_metadata,
                    )
                    state = self._fresh_state()
                    timing.emit_start()
                    try:
                        with _scoped_proxy_env(proxy_url):
                            stream = await litellm.aresponses(
                                headers=headers,
                                num_retries=num_retries,
                                **request,
                            )
                        stream = _disable_litellm_stream_logging(stream)
                        async for event in _aiter_stream_with_heartbeat(stream):
                            timing.observe_event(event)
                            self._handle_event(event, state)
                            if state["completed"] or state["failure"] is not None:
                                break
                        timing.mark_stream_end()
                        self._raise_if_failed(state)
                        result = self._assemble(
                            text_parts=state["text_parts"],
                            usage_raw=state["usage_raw"],
                            response_id=state["response_id"],
                            model_name=state["model_name"],
                        )
                        timing.emit_end(state, finished_at=monotonic())
                        return result
                    except Exception as exc:
                        timing.emit_error(state, exc)
                        raise

        completion_fn, litellm_cache_args = self._get_cached_completion_fn(_acompletion, cache)
        results = await completion_fn(
            request=request,
            num_retries=self.num_retries,
            cache=litellm_cache_args,
        )
        _fire_usage_tracker_hook(self.model, results)
        return results


class _UnavailableCodexWSTransport:
    def stream_turn(self, **_: Any):
        raise CodexStreamError("Codex WebSocket transport is not configured")

    async def astream_turn(self, **_: Any):
        raise CodexStreamError("Codex WebSocket transport is not configured")


def _responses_ws_url(base: str) -> str:
    parsed = urlparse(base.rstrip("/"))
    scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme, parsed.scheme)
    path = parsed.path.rstrip("/")
    if not path.endswith("/responses"):
        path = f"{path}/responses"
    return urlunparse(parsed._replace(scheme=scheme, path=path))


def _ws_request_payload(request: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "api_base",
        "api_key",
        "custom_llm_provider",
        "_codex_lm_auth_metadata",
        "_codex_lm_proxy_url",
    }
    payload = {
        key: value for key, value in request.items() if key not in excluded and value is not None
    }
    payload["type"] = "response.create"
    return payload


class _AiohttpCodexWSTransport:
    def __init__(self, base_url: str = CODEX_BASE):
        self.url = _responses_ws_url(base_url)

    def stream_turn(self, **kwargs: Any):
        async def collect() -> list[Any]:
            events = []
            stream = self.astream_turn(**kwargs)
            async for event in _aiter_stream_with_heartbeat(stream):
                events.append(event)
                event_type = _g(event, "type")
                if event_type in {"response.completed", "response.failed"}:
                    break
            return events

        return iter(asyncio.run(collect()))

    async def astream_turn(
        self,
        *,
        request: dict[str, Any],
        headers: dict[str, str],
        request_id: str,
        sticky_state: dict[str, Any],
    ):
        ws_headers = dict(headers)
        ws_headers["Authorization"] = f"Bearer {request['api_key']}"
        ws_headers["OpenAI-Beta"] = CODEX_WS_BETA_HEADER
        ws_headers["x-client-request-id"] = request_id
        if sticky_state.get("turn_state"):
            ws_headers["x-codex-turn-state"] = sticky_state["turn_state"]

        async with aiohttp.ClientSession() as session:
            try:
                async with session.ws_connect(self.url, headers=ws_headers, compress=15) as ws:
                    response = getattr(ws, "_response", None)
                    turn_state = getattr(response, "headers", {}).get("x-codex-turn-state")
                    if turn_state:
                        sticky_state["turn_state"] = turn_state
                    await ws.send_str(json.dumps(_ws_request_payload(request)))
                    async for message in ws:
                        if message.type == aiohttp.WSMsgType.TEXT:
                            yield json.loads(message.data)
                        elif message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE):
                            break
                        elif message.type == aiohttp.WSMsgType.ERROR:
                            raise CodexStreamError(f"Codex WebSocket error: {ws.exception()}")
            except aiohttp.WSServerHandshakeError as exc:
                if exc.status == 401:
                    raise CodexStreamError(
                        "CodexLM auth expired: WebSocket handshake returned 401 Unauthorized",
                        failure_kind="codex_lm_auth_expired",
                        failure_code=401,
                    ) from exc
                raise


@dataclass
class _CodexWSTurnSession:
    transport: Any
    request: dict[str, Any]
    headers: dict[str, str]
    request_id: str
    sticky_state: dict[str, Any]

    def stream(self):
        return self.transport.stream_turn(
            request=self.request,
            headers=self.headers,
            request_id=self.request_id,
            sticky_state=self.sticky_state,
        )

    async def astream(self):
        stream = self.transport.astream_turn(
            request=self.request,
            headers=self.headers,
            request_id=self.request_id,
            sticky_state=self.sticky_state,
        )
        if hasattr(stream, "__await__"):
            return await stream
        return stream


class CodexWSLM(CodexHTTPLM):
    """Codex LM variant with an injectable WebSocket turn transport.

    The live Codex WebSocket wire protocol is deliberately not hard-coded here.
    Tests and callers can inject a transport implementing ``stream_turn`` and
    ``astream_turn``. The default transport speaks the live Codex WebSocket
    protocol; ``ws_fallback`` controls whether exhausted WS turns fall back to
    the HTTP/SSE ``CodexHTTPLM`` path.
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
        ws_transport: Any | None = None,
        ws_base: str = CODEX_BASE,
        ws_fallback: bool = True,
        fallback_lm: Any | None = None,
        **kwargs,
    ):
        fallback_kwargs = dict(kwargs)
        super().__init__(
            model=model,
            instructions=instructions,
            access_token=access_token,
            account_id=account_id,
            auth_path=auth_path,
            auth_profile=auth_profile,
            auth_config_refresh_seconds=auth_config_refresh_seconds,
            **kwargs,
        )
        self._ws_transport = ws_transport or _AiohttpCodexWSTransport(ws_base)
        self._ws_fallback_enabled = ws_fallback
        self._ws_fallback_active = False
        self._fallback_lm = fallback_lm or CodexHTTPLM(
            model=model,
            instructions=instructions,
            access_token=access_token,
            account_id=account_id,
            auth_path=auth_path,
            auth_profile=auth_profile,
            auth_config_refresh_seconds=auth_config_refresh_seconds,
            **fallback_kwargs,
        )

    def _make_ws_turn(
        self,
        request: dict[str, Any],
        headers: dict[str, str],
    ) -> _CodexWSTurnSession:
        return _CodexWSTurnSession(
            transport=self._ws_transport,
            request=request,
            headers=headers,
            request_id=str(uuid.uuid4()),
            sticky_state={},
        )

    def forward(self, prompt=None, messages=None, **kwargs):
        if self._ws_fallback_active:
            return self._fallback_lm.forward(prompt=prompt, messages=messages, **kwargs)

        fallback_kwargs = dict(kwargs)
        request, headers = self._build_request(prompt, messages, kwargs)
        auth_metadata = request.pop("_codex_lm_auth_metadata", {})
        request.pop("_codex_lm_proxy_url", None)
        cache = kwargs.pop("cache", self.cache)

        def _completion(request, headers, num_retries, cache):
            nonlocal auth_metadata
            turn = self._make_ws_turn(request, headers)
            for attempt in Retrying(**_codex_retry_kwargs()):
                with attempt:
                    if attempt.retry_state.attempt_number > 1:
                        auth_metadata = self._apply_request_auth(
                            request,
                            headers,
                            exclude_profile=auth_metadata.get("auth_profile"),
                        )
                        request.pop("_codex_lm_auth_metadata", None)
                    timing = _StreamTiming(
                        model=str(request.get("model") or self.model),
                        attempt_number=attempt.retry_state.attempt_number,
                        start_at=monotonic(),
                        transport="websocket",
                        request_metadata=_request_debug_metadata(request),
                        auth_metadata=auth_metadata,
                    )
                    state = self._fresh_state()
                    timing.emit_start()
                    try:
                        stream = turn.stream()
                        for event in _iter_stream_with_heartbeat(stream):
                            timing.observe_event(event)
                            self._handle_event(event, state)
                            if state["completed"] or state["failure"] is not None:
                                break
                        timing.mark_stream_end()
                        self._raise_if_failed(state)
                        result = self._assemble(
                            text_parts=state["text_parts"],
                            usage_raw=state["usage_raw"],
                            response_id=state["response_id"],
                            model_name=state["model_name"],
                        )
                        timing.emit_end(state, finished_at=monotonic())
                        return result
                    except Exception as exc:
                        timing.emit_error(state, exc)
                        raise

        completion_fn, litellm_cache_args = self._get_cached_completion_fn(_completion, cache)
        try:
            results = completion_fn(
                request=request,
                headers=headers,
                num_retries=self.num_retries,
                cache=litellm_cache_args,
            )
        except CodexStreamError:
            if not self._ws_fallback_enabled:
                raise
            self._ws_fallback_active = True
            return self._fallback_lm.forward(prompt=prompt, messages=messages, **fallback_kwargs)
        _fire_usage_tracker_hook(self.model, results)
        return results

    async def aforward(self, prompt=None, messages=None, **kwargs):
        if self._ws_fallback_active:
            return await self._fallback_lm.aforward(prompt=prompt, messages=messages, **kwargs)

        fallback_kwargs = dict(kwargs)
        request, headers = self._build_request(prompt, messages, kwargs)
        auth_metadata = request.pop("_codex_lm_auth_metadata", {})
        request.pop("_codex_lm_proxy_url", None)
        cache = kwargs.pop("cache", self.cache)

        async def _acompletion(request, headers, num_retries, cache):
            nonlocal auth_metadata
            turn = self._make_ws_turn(request, headers)
            async for attempt in AsyncRetrying(**_codex_retry_kwargs()):
                with attempt:
                    if attempt.retry_state.attempt_number > 1:
                        auth_metadata = self._apply_request_auth(
                            request,
                            headers,
                            exclude_profile=auth_metadata.get("auth_profile"),
                        )
                        request.pop("_codex_lm_auth_metadata", None)
                    timing = _StreamTiming(
                        model=str(request.get("model") or self.model),
                        attempt_number=attempt.retry_state.attempt_number,
                        start_at=monotonic(),
                        transport="websocket",
                        request_metadata=_request_debug_metadata(request),
                        auth_metadata=auth_metadata,
                    )
                    state = self._fresh_state()
                    timing.emit_start()
                    try:
                        stream = await turn.astream()
                        async for event in _aiter_stream_with_heartbeat(stream):
                            timing.observe_event(event)
                            self._handle_event(event, state)
                            if state["completed"] or state["failure"] is not None:
                                break
                        timing.mark_stream_end()
                        self._raise_if_failed(state)
                        result = self._assemble(
                            text_parts=state["text_parts"],
                            usage_raw=state["usage_raw"],
                            response_id=state["response_id"],
                            model_name=state["model_name"],
                        )
                        timing.emit_end(state, finished_at=monotonic())
                        return result
                    except Exception as exc:
                        timing.emit_error(state, exc)
                        raise

        completion_fn, litellm_cache_args = self._get_cached_completion_fn(_acompletion, cache)
        try:
            results = await completion_fn(
                request=request,
                headers=headers,
                num_retries=self.num_retries,
                cache=litellm_cache_args,
            )
        except CodexStreamError:
            if not self._ws_fallback_enabled:
                raise
            self._ws_fallback_active = True
            return await self._fallback_lm.aforward(
                prompt=prompt,
                messages=messages,
                **fallback_kwargs,
            )
        _fire_usage_tracker_hook(self.model, results)
        return results


CodexLM = CodexWSLM


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
