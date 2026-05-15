from __future__ import annotations

import litellm
from tenacity import wait_none

from rlm_gepa.runtime import lm_config


def _skip_env_validation(monkeypatch):
    monkeypatch.setattr(lm_config, "validate_lm_env", lambda _lm: None)


def test_kimi_reasoning_effort_none_disables_native_thinking(monkeypatch):
    _skip_env_validation(monkeypatch)

    config = lm_config.get_lm_config("moonshot/kimi-k2.6", reasoning_effort="none")

    assert config["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in config


def test_kimi_reasoning_effort_enables_native_thinking_without_fake_effort(monkeypatch):
    _skip_env_validation(monkeypatch)

    config = lm_config.get_lm_config("moonshot/kimi-k2.6", reasoning_effort="low")

    assert config["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "reasoning_effort" not in config


def test_kimi_unspecified_reasoning_leaves_provider_default(monkeypatch):
    _skip_env_validation(monkeypatch)

    config = lm_config.get_lm_config("moonshot/kimi-k2.6", reasoning_effort=None)

    assert "extra_body" not in config
    assert "reasoning_effort" not in config


def test_non_kimi_reasoning_effort_behavior_is_unchanged(monkeypatch):
    _skip_env_validation(monkeypatch)

    config = lm_config.get_lm_config("openai/gpt-5.4", reasoning_effort="low")
    disabled_config = lm_config.get_lm_config("openai/gpt-5.4", reasoning_effort="none")

    assert config["reasoning_effort"] == "low"
    assert "reasoning_effort" not in disabled_config
    assert "extra_body" not in config
    assert "extra_body" not in disabled_config


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
