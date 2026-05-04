from __future__ import annotations

import logging
import os
from typing import Any

import dspy
import litellm

_LITELLM_LOGGER_NAMES = ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy")
_MERCURY_MODELS = {"mercury-2", "mercury-edit-2"}
_INCEPTION_API_BASE = "https://api.inceptionlabs.ai/v1"


def configure_litellm_logging() -> None:
    litellm.turn_off_message_logging = True
    for logger_name in _LITELLM_LOGGER_NAMES:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)


def validate_lm_env(lm: str) -> None:
    if _is_mercury_model(lm):
        if os.environ.get("INCEPTION_API_KEY"):
            return
        raise RuntimeError(f"{lm} is missing required environment variables: INCEPTION_API_KEY")

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
    if _is_mercury_model(lm):
        config["api_base"] = _INCEPTION_API_BASE
        config["api_key"] = os.environ["INCEPTION_API_KEY"]

    if reasoning_effort and reasoning_effort != "none":
        config["reasoning_effort"] = reasoning_effort
        if _is_mercury_model(lm):
            config["allowed_openai_params"] = ["reasoning_effort"]

    if thinking_budget is not None:
        config["thinking_budget"] = thinking_budget
    return config


def get_sub_lm_config(lm: str, reasoning_effort: str | None = "none") -> dict[str, Any]:
    return get_lm_config(lm, reasoning_effort=reasoning_effort)


def _is_mercury_model(lm: str) -> bool:
    return lm.split("/")[-1] in _MERCURY_MODELS


def build_lm(model_or_lm: Any, *, reasoning_effort: str | None = None, cache: bool = False) -> Any:
    if not isinstance(model_or_lm, str):
        return model_or_lm
    return dspy.LM(**get_lm_config(model_or_lm, reasoning_effort), cache=cache)


configure_litellm_logging()
