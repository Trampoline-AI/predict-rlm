from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from dspy_codex_lm.auth import load_codex_auth

CODEX_USAGE_ENDPOINT = "https://chatgpt.com/backend-api/wham/usage"
DEFAULT_TIMEOUT = 10.0

_SENSITIVE_KEYS = {
    "access_token",
    "account",
    "account_id",
    "authorization",
    "email",
    "id_token",
    "refresh_token",
    "token",
    "tokens",
    "user",
}

_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_BEARER_RE = re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_ACCOUNT_RE = re.compile(r"\bacct[-_A-Za-z0-9]*\b", re.IGNORECASE)

_LIMIT_KEYS = (
    "limit",
    "max",
    "max_value",
    "quota",
    "total",
    "granted",
    "total_granted",
)
_USED_KEYS = (
    "used",
    "current",
    "current_value",
    "usage",
    "consumed",
    "spent",
    "total_used",
)
_REMAINING_KEYS = (
    "remaining",
    "remaining_value",
    "available",
    "balance",
    "left",
    "credits_remaining",
)
_USED_PERCENT_KEY = "used_percent"
_RESET_AT_KEYS = (
    "reset_at",
    "resets_at",
    "reset_time",
    "next_reset_at",
    "expires_at",
    "expiry",
    "expiration",
    "renewal_at",
)
_RESET_AFTER_KEYS = (
    "reset_after_seconds",
    "resets_in_seconds",
    "seconds_until_reset",
    "reset_in_seconds",
    "retry_after_seconds",
)


@dataclass(frozen=True)
class UsageWindow:
    label: str
    remaining: float | int | None
    limit: float | int | None
    used: float | int | None
    percent_remaining: float | None
    reset: str | None
    window_seconds: int | None = None


@dataclass(frozen=True)
class DisabledProfileUsage:
    reason: str = "Disabled; live usage fetch skipped."


def format_disabled_profile_usage_entry(
    profile_name: str,
) -> tuple[str, DisabledProfileUsage]:
    return profile_name, DisabledProfileUsage()


def fetch_codex_usage(
    *,
    access_token: str | None = None,
    account_id: str | None = None,
    auth_path: str | Path | None = None,
    endpoint: str = CODEX_USAGE_ENDPOINT,
    timeout: float = DEFAULT_TIMEOUT,
    transport: Callable[..., Any] | None = None,
) -> Any:
    """Fetch the ChatGPT/Codex usage payload with Codex CLI auth.

    When token fields are not passed explicitly, auth resolution is delegated
    to :func:`load_codex_auth`. The returned value is the decoded JSON payload.
    """
    if access_token is None or account_id is None:
        loaded_token, loaded_account = load_codex_auth(
            Path(auth_path) if auth_path is not None else None
        )
        access_token = access_token or loaded_token
        account_id = account_id or loaded_account

    headers = {
        "Authorization": f"Bearer {access_token}",
        "ChatGPT-Account-ID": account_id,
        "Accept": "application/json",
    }

    raw = (transport or _urllib_transport)(endpoint, headers=headers, timeout=timeout)
    return _decode_json(raw)


def summarize_usage(payload: Any) -> list[UsageWindow]:
    rows: list[UsageWindow] = []
    _collect_usage_windows(payload, [], rows)
    return sorted(rows, key=lambda row: row.label)


_ANSI_RESET = "\x1b[0m"
_ANSI_BOLD_CYAN = "\x1b[1;36m"
_ANSI_GREEN = "\x1b[32m"
_ANSI_YELLOW = "\x1b[33m"
_ANSI_RED = "\x1b[31m"
_ANSI_DIM = "\x1b[2m"
USAGE_SEPARATOR = "-" * 60


def format_usage_summary(payload: Any, *, color: bool = False) -> str:
    rows = summarize_usage(payload)
    lines = [_ansi("Codex usage", _ANSI_BOLD_CYAN, color)]
    if isinstance(payload, Mapping):
        plan_type = payload.get("plan_type")
        if plan_type:
            lines.append(f"Plan: {_redact_text(str(plan_type))}")
        credits = _format_credits(payload.get("credits"))
        if credits:
            lines.append(credits)
    if not rows:
        lines.append("No usage windows found.")
        return _frame_usage_lines(lines)
    if _has_live_rate_limit_rows(rows):
        lines.extend(_format_live_usage_rows(rows, color=color))
        return _frame_usage_lines(lines)
    lines.extend(_format_row(row, color=color) for row in rows)
    return _frame_usage_lines(lines)


def format_profile_usage_summaries(
    items: list[tuple[str, Any]],
    *,
    color: bool = False,
    default_profile: str | None = None,
) -> str:
    lines: list[str] = []
    for index, (profile_name, payload) in enumerate(items):
        if index:
            lines.append("")
        heading = _redact_profile_display_name(profile_name)
        if isinstance(payload, DisabledProfileUsage):
            heading = f"{heading} (disabled)"
        if profile_name == default_profile:
            heading = f"{heading} (default)"
        if isinstance(payload, Mapping):
            plan_type = payload.get("plan_type")
            if plan_type:
                heading = f"{heading} ({_redact_text(str(plan_type))})"
        lines.append(_ansi(f"{heading}:", _ANSI_BOLD_CYAN, color))
        if isinstance(payload, DisabledProfileUsage):
            lines.append(f"  {payload.reason}")
            continue
        rows = summarize_usage(payload)
        if not rows:
            lines.append("  No usage windows found.")
        elif _has_live_rate_limit_rows(rows):
            lines.extend(_format_live_usage_rows(rows, indent="  ", color=color))
        else:
            lines.extend(f"  {_format_row(row, color=color)}" for row in rows)
    return _frame_usage_lines(lines)


def _frame_usage_lines(lines: list[str]) -> str:
    return "\n".join([USAGE_SEPARATOR, *lines, USAGE_SEPARATOR])


def _format_credits(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    if value.get("unlimited") is True:
        return "Credits: unlimited"
    balance = value.get("balance")
    if balance is not None:
        has_credits = value.get("has_credits")
        suffix = ""
        if isinstance(has_credits, bool):
            suffix = f"; has_credits={str(has_credits).lower()}"
        return f"Credits: balance {_redact_text(str(balance))}{suffix}"
    return None


def _urllib_transport(url: str, *, headers: Mapping[str, str], timeout: float) -> bytes:
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _decode_json(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, bytes):
        return json.loads(raw.decode("utf-8"))
    if isinstance(raw, str):
        return json.loads(raw)
    if hasattr(raw, "json"):
        return raw.json()
    return raw


def _collect_usage_windows(
    value: Any,
    path: list[str],
    rows: list[UsageWindow],
) -> None:
    if isinstance(value, Mapping):
        limit_name = value.get("limit_name")
        rate_limit = value.get("rate_limit")
        if limit_name is not None and isinstance(rate_limit, Mapping):
            _collect_usage_windows(rate_limit, [str(limit_name)], rows)
            return

        if _is_usage_window(value):
            label = ".".join(_redact_segment(part) for part in path) or "usage"
            rows.append(_usage_window(label, value))

        for key, child in value.items():
            key_text = str(key)
            if _is_sensitive_key(key_text):
                continue
            _collect_usage_windows(child, [*path, key_text], rows)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _collect_usage_windows(child, [*path, str(index)], rows)


def _is_usage_window(value: Mapping[str, Any]) -> bool:
    if _number_for_key(value, _USED_PERCENT_KEY) is not None:
        return True
    return _first_number(value, _LIMIT_KEYS) is not None and (
        _first_number(value, _USED_KEYS) is not None
        or _first_number(value, _REMAINING_KEYS) is not None
    )


def _usage_window(label: str, value: Mapping[str, Any]) -> UsageWindow:
    limit = _first_number(value, _LIMIT_KEYS)
    used = _first_number(value, _USED_KEYS)
    remaining = _first_number(value, _REMAINING_KEYS)
    used_percent = _number_for_key(value, _USED_PERCENT_KEY)

    if remaining is None and limit is not None and used is not None:
        remaining = limit - used
    if used is None and limit is not None and remaining is not None:
        used = limit - remaining

    percent = None
    if used_percent is not None:
        percent = round(max(0.0, min(100.0, 100.0 - used_percent)), 1)
    elif limit not in (None, 0) and remaining is not None:
        percent = round((remaining / limit) * 100, 1)

    return UsageWindow(
        label=label,
        remaining=_clean_number(remaining),
        limit=_clean_number(limit),
        used=_clean_number(used),
        percent_remaining=percent,
        reset=_reset_text(value),
        window_seconds=_window_seconds(value),
    )


def _window_seconds(value: Mapping[str, Any]) -> int | None:
    seconds = _number_for_key(value, "limit_window_seconds")
    if seconds is None:
        return None
    return int(seconds)


def _first_number(value: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        number = _number_for_key(value, key)
        if number is not None:
            return number
    return None


def _number_for_key(value: Mapping[str, Any], key: str) -> float | None:
    lowered = {str(raw_key).lower(): raw for raw_key, raw in value.items()}
    return _as_number(lowered.get(key.lower()))


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _clean_number(value: float | None) -> float | int | None:
    if value is None:
        return None
    if value.is_integer():
        return int(value)
    return value


def _reset_text(value: Mapping[str, Any]) -> str | None:
    lowered = {str(key).lower(): raw for key, raw in value.items()}
    for key in _RESET_AT_KEYS:
        raw = lowered.get(key)
        if raw is not None:
            return _format_reset_at(raw)
    for key in _RESET_AFTER_KEYS:
        seconds = _as_number(lowered.get(key))
        if seconds is not None:
            return f"in {_duration(seconds)}"
    return None


def _format_reset_at(value: Any) -> str:
    timestamp = _as_number(value)
    if timestamp is None:
        return _redact_text(str(value))
    dt = datetime.fromtimestamp(timestamp)
    return f"{dt:%H:%M} on {dt.day} {dt:%b}"


def _duration(seconds: float) -> str:
    seconds_i = int(round(seconds))
    if seconds_i < 60:
        return f"{seconds_i}s"
    minutes, rem_seconds = divmod(seconds_i, 60)
    if minutes < 60:
        return f"{minutes}m" if rem_seconds == 0 else f"{minutes}m {rem_seconds}s"
    hours, rem_minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h" if rem_minutes == 0 else f"{hours}h {rem_minutes}m"
    days, rem_hours = divmod(hours, 24)
    return f"{days}d" if rem_hours == 0 else f"{days}d {rem_hours}h"


def _format_row(row: UsageWindow, *, color: bool = False) -> str:
    remaining = _format_number(row.remaining)
    limit = _format_number(row.limit)
    if row.remaining is not None and row.limit is not None:
        usage = f"{remaining}/{limit} remaining"
    elif row.remaining is not None:
        usage = f"{remaining} remaining"
    elif row.used is not None and row.limit is not None:
        usage = f"{_format_number(row.used)}/{limit} used"
    elif row.used is not None:
        usage = f"{_format_number(row.used)} used"
    else:
        usage = "usage available"

    if row.percent_remaining is not None:
        usage = f"{usage} ({row.percent_remaining:.1f}% remaining)"
    if row.reset:
        usage = f"{usage}; resets {row.reset}"
    return f"{_ansi(row.label, _ANSI_BOLD_CYAN, color)}: {usage}"


def _has_live_rate_limit_rows(rows: list[UsageWindow]) -> bool:
    return any(row.window_seconds is not None for row in rows)


def _format_live_usage_rows(
    rows: list[UsageWindow],
    *,
    indent: str = "",
    color: bool = False,
) -> list[str]:
    lines: list[str] = []
    general = [row for row in rows if row.label.startswith("rate_limit.")]
    additional = [row for row in rows if not row.label.startswith("rate_limit.")]

    if general:
        lines.append(f"{indent}{_ansi('General usage limits:', _ANSI_BOLD_CYAN, color)}")
        for row in sorted(general, key=_live_row_sort_key):
            lines.append(_format_live_row(row, indent=f"{indent}  ", color=color))

    grouped: dict[str, list[UsageWindow]] = {}
    for row in additional:
        group, _ = _split_live_label(row.label)
        grouped.setdefault(group, []).append(row)

    for group in sorted(grouped, key=str.casefold):
        heading = f"{_redact_text(group)} limit:"
        lines.append(f"{indent}{_ansi(heading, _ANSI_BOLD_CYAN, color)}")
        for row in sorted(grouped[group], key=_live_row_sort_key):
            lines.append(_format_live_row(row, indent=f"{indent}  ", color=color))
    return lines


def _format_live_row(
    row: UsageWindow,
    *,
    indent: str = "",
    color: bool = False,
) -> str:
    label = _live_window_label(row)
    percent = row.percent_remaining
    percent_text = "?" if percent is None else f"{percent:g}%"
    usage = f"{_usage_bar(percent)} {percent_text} left"
    usage = _ansi(usage, _usage_color(percent), color)
    if row.reset:
        usage = f"{usage} {_ansi(f'(resets {row.reset})', _ANSI_DIM, color)}"
    return f"{indent}{label + ':':<29}{usage}"


def _usage_bar(percent: float | None) -> str:
    if percent is None:
        filled = 0
    else:
        filled = round(max(0.0, min(100.0, percent)) / 100.0 * 20)
    return f"[{'█' * filled}{'░' * (20 - filled)}]"


def _usage_color(percent: float | None) -> str:
    if percent is None:
        return _ANSI_DIM
    if percent >= 50:
        return _ANSI_GREEN
    if percent >= 20:
        return _ANSI_YELLOW
    return _ANSI_RED


def _ansi(value: str, code: str, enabled: bool) -> str:
    if not enabled:
        return value
    return f"{code}{value}{_ANSI_RESET}"


def _live_window_label(row: UsageWindow) -> str:
    if row.window_seconds == 18000:
        return "5h limit"
    if row.window_seconds == 604800:
        return "Weekly limit"
    if row.window_seconds is not None:
        return f"{_duration(row.window_seconds)} limit"
    _, leaf = _split_live_label(row.label)
    return leaf.replace("_", " ")


def _live_row_sort_key(row: UsageWindow) -> tuple[int, str]:
    _, leaf = _split_live_label(row.label)
    if leaf == "primary_window":
        order = 0
    elif leaf == "secondary_window":
        order = 1
    else:
        order = 2
    return order, row.label


def _split_live_label(label: str) -> tuple[str, str]:
    if "." not in label:
        return label, label
    group, leaf = label.rsplit(".", 1)
    return group, leaf


def _format_number(value: float | int | None) -> str:
    if value is None:
        return "?"
    if isinstance(value, int):
        return str(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def _is_sensitive_key(key: str) -> bool:
    key_lower = key.lower()
    return key_lower in _SENSITIVE_KEYS or key_lower.endswith("_token")


def _redact_segment(value: str) -> str:
    return _redact_text(value)


def _redact_profile_display_name(value: str) -> str:
    value = _BEARER_RE.sub("Bearer [redacted]", value)
    value = _ACCOUNT_RE.sub("[redacted-account]", value)
    return value


def _redact_text(value: str) -> str:
    value = _BEARER_RE.sub("Bearer [redacted]", value)
    value = _EMAIL_RE.sub("[redacted-email]", value)
    value = _ACCOUNT_RE.sub("[redacted-account]", value)
    return value
