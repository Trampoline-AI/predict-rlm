from __future__ import annotations

import logging
from typing import Any

import dspy
import litellm
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

_LITELLM_LOGGER_NAMES = ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy")
_RATE_LIMIT_RETRY_ATTEMPTS = 10
_RATE_LIMIT_RETRY_WAIT = wait_exponential_jitter(initial=1, max=60, jitter=1)


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
    if _is_moonshot_kimi_model(lm):
        if reasoning_effort == "none":
            config["extra_body"] = {"thinking": {"type": "disabled"}}
        elif reasoning_effort:
            config["extra_body"] = {"thinking": {"type": "enabled"}}
    elif reasoning_effort and reasoning_effort != "none":
        config["reasoning_effort"] = reasoning_effort

    if thinking_budget is not None:
        config["thinking_budget"] = thinking_budget
    return config


def _is_moonshot_kimi_model(lm: str) -> bool:
    model = lm.lower()
    return "moonshot/" in model and "kimi" in model


def get_sub_lm_config(lm: str, reasoning_effort: str | None = "none") -> dict[str, Any]:
    return get_lm_config(lm, reasoning_effort=reasoning_effort)


def _forward_with_rate_limit_retry(self: dspy.LM, *args: Any, **kwargs: Any) -> Any:
    retryer = Retrying(
        retry=retry_if_exception_type(litellm.RateLimitError),
        wait=_RATE_LIMIT_RETRY_WAIT,
        stop=stop_after_attempt(_RATE_LIMIT_RETRY_ATTEMPTS),
        reraise=True,
    )
    for attempt in retryer:
        with attempt:
            return super(type(self), self).forward(*args, **kwargs)


class RateLimitRetryLM(dspy.LM):
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return _forward_with_rate_limit_retry(self, *args, **kwargs)


def _rate_limit_retry_lm_class() -> type[dspy.LM]:
    if RateLimitRetryLM.__mro__[1] is dspy.LM:
        return RateLimitRetryLM
    return type(
        "RateLimitRetryLM",
        (dspy.LM,),
        {
            "__module__": __name__,
            "forward": _forward_with_rate_limit_retry,
        },
    )


def build_lm(model_or_lm: Any, *, reasoning_effort: str | None = None, cache: bool = False) -> Any:
    if not isinstance(model_or_lm, str):
        return model_or_lm
    lm_class = _rate_limit_retry_lm_class()
    return lm_class(**get_lm_config(model_or_lm, reasoning_effort), cache=cache)


configure_litellm_logging()
