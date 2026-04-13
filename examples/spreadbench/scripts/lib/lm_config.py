"""Provider-aware ``dspy.LM`` configuration.

Single entry point that builds the kwargs for any LM we evaluate:
gpt-5.x, claude-*, mercury-2, or any other LiteLLM-addressable model.
The only one that needs special handling is mercury-2 because the
Inception Labs API is nonstandard — everyone else gets ``reasoning_effort``
as a top-level kwarg and lets LiteLLM route it to the provider's
extended-thinking / effort field.
"""

from __future__ import annotations

import os

SUB_LM: str = "openai/gpt-5.1"


def get_lm_config(lm: str, reasoning_effort: str | None = None) -> dict:
    """Return the kwargs for ``dspy.LM(**kwargs)``.

    Args:
        lm: The full LiteLLM model string (e.g. ``"openai/gpt-5.4"``,
            ``"anthropic/claude-opus-4-6"``, ``"openai/mercury-2"``).
        reasoning_effort: Optional effort label — usually ``"low"``,
            ``"medium"``, or ``"high"``. For mercury-2 this maps to the
            ``extra_body.reasoning_effort`` field and defaults to
            ``"instant"`` when omitted. For every other provider it's
            applied as a top-level kwarg only when explicitly set.

    Raises:
        RuntimeError: if ``lm`` is ``openai/mercury-2`` and the
            ``INCEPTION_API_KEY`` environment variable isn't set.
    """
    cfg: dict = {"model": lm, "num_retries": 5}

    if lm == "openai/mercury-2":
        key = os.environ.get("INCEPTION_API_KEY")
        if not key:
            raise RuntimeError(
                "openai/mercury-2 requires the INCEPTION_API_KEY "
                "environment variable."
            )
        cfg.update(
            {
                "api_base": "https://api.inceptionlabs.ai/v1",
                "api_key": key,
                "extra_body": {"reasoning_effort": reasoning_effort or "instant"},
            }
        )
    elif reasoning_effort:
        cfg["reasoning_effort"] = reasoning_effort

    return cfg
