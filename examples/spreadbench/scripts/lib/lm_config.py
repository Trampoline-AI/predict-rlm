"""Provider-aware ``dspy.LM`` configuration.

Single entry point that builds the kwargs for any LM we evaluate:
gpt-5.x, claude-*, mercury-2, or any other LiteLLM-addressable model.
The only one that needs special handling is mercury-2 because the
Inception Labs API is nonstandard — everyone else gets ``reasoning_effort``
as a top-level kwarg and lets LiteLLM route it to the provider's
extended-thinking / effort field. We also validate required API key env vars
for common providers so misconfiguration fails fast.
"""

from __future__ import annotations

import logging
import os

import litellm

_LITELLM_LOGGER_NAMES = ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy")


def configure_litellm_logging() -> None:
    """Suppress noisy LiteLLM background-worker logs.

    DSPy imports LiteLLM and can reset logger levels after this module
    is imported, so call this both at import time and when building LM
    configs.
    """
    try:
        litellm.turn_off_message_logging = True
    except Exception:
        pass
    for logger_name in _LITELLM_LOGGER_NAMES:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)


configure_litellm_logging()

SUB_LM: str = "openai/gpt-5.1"

_MERCURY_MODEL = "openai/mercury-2"

# Per-model price overrides as (input_usd_per_mtok, output_usd_per_mtok).
# LiteLLM's cost map doesn't cover non-OpenAI/Anthropic providers like
# Inception Labs, so any model listed here gets its cost recomputed from
# token counts via :func:`compute_lm_cost` instead of relying on
# ``lm.history[i]["cost"]`` (which is ``None``/``0`` for unknown models).
_PRICE_OVERRIDES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    _MERCURY_MODEL: (0.25, 0.75),
}


def compute_lm_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float | None:
    """Return USD cost for *model* if it has a price override, else None.

    Override entries are USD per 1M input/output tokens. Returning
    ``None`` signals to the caller that LiteLLM's per-call cost (from
    ``lm.history``) should be used instead.
    """
    override = _PRICE_OVERRIDES_USD_PER_MTOK.get(model)
    if override is None:
        return None
    in_price, out_price = override
    return (prompt_tokens / 1_000_000) * in_price + (
        completion_tokens / 1_000_000
    ) * out_price


def _require_env(var_name: str, lm: str) -> None:
    if not os.environ.get(var_name):
        raise RuntimeError(f"{lm} requires the {var_name} environment variable.")


def validate_lm_env(lm: str) -> None:
    """Raise if *lm* requires an API key env var that is missing."""
    if lm == _MERCURY_MODEL:
        _require_env("INCEPTION_API_KEY", lm)
        return

    check = litellm.validate_environment(model=lm)
    if check.get("keys_in_environment"):
        return

    missing_keys = check.get("missing_keys") or []
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise RuntimeError(f"{lm} is missing required environment variables: {missing}")

    raise RuntimeError(f"{lm} is missing required provider environment configuration.")


def get_lm_config(
    lm: str,
    reasoning_effort: str | None = None,
    thinking_budget: int | None = None,
) -> dict:
    """Return the kwargs for ``dspy.LM(**kwargs)``.

    Args:
        lm: The full LiteLLM model string (e.g. ``"openai/gpt-5.4"``,
            ``"anthropic/claude-opus-4-6"``, ``"openai/mercury-2"``).
        reasoning_effort: Optional effort label — usually ``"low"``,
            ``"medium"``, or ``"high"``. For mercury-2 this maps to the
            ``extra_body.reasoning_effort`` field and defaults to
            ``"instant"`` when omitted. For every other provider it's
            applied as a top-level kwarg only when explicitly set.
        thinking_budget: Optional explicit token budget for the reasoning
            phase. Passed through to LiteLLM's ``thinking_budget`` kwarg,
            which gemini's wrapper translates into the provider-native
            ``thinkingConfig.thinkingBudget`` field. ``0`` disables
            thinking where the model supports it (Gemini 3 non-flash
            cannot fully disable but maps 0 to the minimum tier).
            Overrides ``reasoning_effort`` when both are set.

    Raises:
        RuntimeError: if an API key env var required by ``lm`` is missing.
    """
    configure_litellm_logging()
    validate_lm_env(lm)

    cfg: dict = {"model": lm, "num_retries": 5}

    if lm == _MERCURY_MODEL:
        cfg.update(
            {
                "api_base": "https://api.inceptionlabs.ai/v1",
                "api_key": os.environ["INCEPTION_API_KEY"],
                "extra_body": {"reasoning_effort": reasoning_effort or "instant"},
            }
        )
    elif reasoning_effort:
        cfg["reasoning_effort"] = reasoning_effort

    if thinking_budget is not None:
        cfg["thinking_budget"] = thinking_budget

    return cfg


def get_sub_lm_config(lm: str) -> dict:
    """Return config for the sub-LM with provider-aware defaults."""
    cfg = get_lm_config(lm)
    if lm != _MERCURY_MODEL:
        cfg["reasoning_effort"] = "none"
    return cfg
