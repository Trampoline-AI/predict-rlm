from __future__ import annotations

import litellm
import pytest
from tenacity import wait_none

from rlm_gepa.runtime import lm_config

pytestmark = pytest.mark.gepa


def _skip_env_validation(monkeypatch):
    monkeypatch.setattr(lm_config, "validate_lm_env", lambda _lm: None)


def test_build_lm_retries_litellm_rate_limits_with_tenacity(monkeypatch):
    _skip_env_validation(monkeypatch)
    monkeypatch.setattr(lm_config, "_RATE_LIMIT_RETRY_WAIT", wait_none(), raising=False)
    attempts = 0

    def flaky_forward(_self, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise litellm.RateLimitError(
                "Error code: 429 - {'error': 'Rate limit exceeded'}",
                llm_provider="openai",
                model="openai/gpt-5.4",
                num_retries=3,
            )
        return "ok"

    monkeypatch.setattr(lm_config.dspy.LM, "forward", flaky_forward)

    lm = lm_config.build_lm("openai/gpt-5.4")

    assert lm.forward(prompt="hi") == "ok"
    assert attempts == 3
