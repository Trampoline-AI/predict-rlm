"""Opt-in debug logging for predict-rlm internals."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TRUE_VALUES = {"1", "true", "yes", "on"}
_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|authorization|bearer|credential|password|secret|token)",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(
    r"(?i)\b(bearer\s+[a-z0-9._~+/=-]+|sk-[a-z0-9_-]{8,}|"
    r"xox[a-z]-[a-z0-9-]{8,}|gh[pousr]_[a-z0-9_]{8,})\b"
)
_MAX_STRING_CHARS = 240
_LOGGER_NAME = "predict_rlm.debug"
_CONFIGURED_KEY: tuple[str | None, bool] | None = None


def is_enabled() -> bool:
    """Return whether predict-rlm debug logging is enabled by environment."""
    return _truthy_env("PREDICT_RLM_DEBUG") or _truthy_env("RLM_DEBUG")


def debug_event(event: str, **metadata: Any) -> None:
    """Emit one sanitized debug event if opt-in logging is enabled."""
    if not is_enabled():
        return
    try:
        logger = _logger()
        json_enabled = _truthy_env("PREDICT_RLM_DEBUG_JSON")
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
            **sanitize_metadata(metadata),
        }
        if json_enabled:
            logger.debug(json.dumps(payload, sort_keys=True, default=str))
        else:
            logger.debug(_format_plain(payload))
    except Exception:
        return


def sanitize_metadata(value: Any, *, key: str | None = None) -> Any:
    """Return a small log-safe value with obvious secrets redacted."""
    if key and _SECRET_KEY_RE.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(k): sanitize_metadata(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [sanitize_metadata(item, key=key) for item in value]
    if isinstance(value, str):
        if _SECRET_VALUE_RE.search(value):
            return _SECRET_VALUE_RE.sub("[REDACTED]", value)
        if len(value) > _MAX_STRING_CHARS:
            return f"{value[:_MAX_STRING_CHARS]}...<truncated {len(value)} chars>"
        return value
    return value


def safe_model_name(lm: Any) -> str | None:
    """Return a non-secret model label for debug metadata."""
    if lm is None:
        return None
    model = getattr(lm, "model", None) or getattr(lm, "name", None)
    if model is None:
        model = type(lm).__name__
    return str(sanitize_metadata(str(model)))


def reset_debug_logger_for_tests() -> None:
    """Reset cached handlers so tests can vary environment destinations."""
    global _CONFIGURED_KEY
    logger = logging.getLogger(_LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    _CONFIGURED_KEY = None


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def _logger() -> logging.Logger:
    global _CONFIGURED_KEY
    log_path = os.environ.get("PREDICT_RLM_DEBUG_LOG") or None
    json_enabled = _truthy_env("PREDICT_RLM_DEBUG_JSON")
    key = (log_path, json_enabled)

    logger = logging.getLogger(_LOGGER_NAME)
    if _CONFIGURED_KEY == key and logger.handlers:
        return logger

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(log_path, encoding="utf-8")
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    _CONFIGURED_KEY = key
    return logger


def _format_plain(payload: dict[str, Any]) -> str:
    ts = payload.pop("ts")
    event = payload.pop("event")
    parts = [f"{key}={_plain_value(value)}" for key, value in sorted(payload.items())]
    return " ".join([ts, event, *parts])


def _plain_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)
