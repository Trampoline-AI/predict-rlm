from __future__ import annotations

import os
from typing import Any

import dspy
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

_RATE_LIMIT_RETRY_ATTEMPTS = 10
_RATE_LIMIT_RETRY_WAIT = wait_exponential_jitter(initial=1, max=60, jitter=1)


def validate_lm_env(lm: str) -> None:
    providers = tuple(provider.definition for provider in dspy.lm15.registered_providers())
    with dspy.lm15.LMRouter(dspy.lm15.RouterConfig(providers=providers)) as router:
        resolution = router.resolve_openai_chat(lm)
    if (
        resolution.credential_policy == "key"
        and resolution.placeholder_key is None
        and resolution.env_key
        and not os.environ.get(resolution.env_key)
    ):
        raise RuntimeError(
            f"{lm} is missing required environment variables: {resolution.env_key}"
        )


def get_lm_config(
    lm: str,
    reasoning_effort: str | None = None,
    thinking_budget: int | None = None,
) -> dict[str, Any]:
    validate_lm_env(lm)
    if thinking_budget is not None:
        raise dspy.LMUnsupportedFeatureError(
            "thinking_budget is not supported by DSPy 3.4's keyword bridge; "
            "use reasoning_effort or supply a native LM with canonical request configuration.",
            model=lm,
        )

    config: dict[str, Any] = {"model": lm, "engine": "lm15", "num_retries": 5}
    if _is_moonshot_kimi_model(lm):
        if reasoning_effort == "none":
            config["thinking"] = {"type": "disabled"}
        elif reasoning_effort:
            config["reasoning_effort"] = reasoning_effort
    elif reasoning_effort and reasoning_effort != "none":
        config["reasoning_effort"] = reasoning_effort

    return config


def _is_moonshot_kimi_model(lm: str) -> bool:
    model = lm.lower()
    return "moonshot/" in model and "kimi" in model


def get_sub_lm_config(lm: str, reasoning_effort: str | None = "none") -> dict[str, Any]:
    return get_lm_config(lm, reasoning_effort=reasoning_effort)


def _forward_with_rate_limit_retry(self: dspy.LM, *args: Any, **kwargs: Any) -> Any:
    retryer = Retrying(
        retry=retry_if_exception_type(dspy.LMRateLimitError),
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
