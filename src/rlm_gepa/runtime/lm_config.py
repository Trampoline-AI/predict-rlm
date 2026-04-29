from __future__ import annotations

import logging
from typing import Any

import dspy
import litellm

_LITELLM_LOGGER_NAMES = ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy")


def configure_litellm_logging() -> None:
    litellm.turn_off_message_logging = True
    for logger_name in _LITELLM_LOGGER_NAMES:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)


def validate_lm_env(lm: str) -> None:
    check = litellm.validate_environment(model=lm)
    if check.get("keys_in_environment"):
        return
    missing_keys = check.get("missing_keys") or []
    if missing_keys:
        raise RuntimeError(
            f"{lm} is missing required environment variables: {', '.join(missing_keys)}"
        )
    raise RuntimeError(f"{lm} is missing required provider environment configuration")


def get_lm_config(
    lm: str,
    reasoning_effort: str | None = None,
    thinking_budget: int | None = None,
) -> dict[str, Any]:
    configure_litellm_logging()
    validate_lm_env(lm)

    config: dict[str, Any] = {"model": lm, "num_retries": 5}
    if reasoning_effort and reasoning_effort != "none":
        config["reasoning_effort"] = reasoning_effort

    if thinking_budget is not None:
        config["thinking_budget"] = thinking_budget
    return config


def get_sub_lm_config(lm: str, reasoning_effort: str | None = "none") -> dict[str, Any]:
    return get_lm_config(lm, reasoning_effort=reasoning_effort)


def build_lm(model_or_lm: Any, *, reasoning_effort: str | None = None, cache: bool = False) -> Any:
    if not isinstance(model_or_lm, str):
        return model_or_lm
    return dspy.LM(**get_lm_config(model_or_lm, reasoning_effort), cache=cache)


configure_litellm_logging()
