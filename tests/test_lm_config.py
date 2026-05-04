from rlm_gepa.runtime import lm_config


def test_mercury_reasoning_effort_uses_inception_directly(monkeypatch):
    monkeypatch.setenv("INCEPTION_API_KEY", "inception-test-key")

    config = lm_config.get_lm_config("openai/mercury-2", reasoning_effort="low")

    assert config["api_base"] == "https://api.inceptionlabs.ai/v1"
    assert config["api_key"] == "inception-test-key"
    assert config["reasoning_effort"] == "low"
    assert config["allowed_openai_params"] == ["reasoning_effort"]


def test_non_mercury_reasoning_effort_unchanged(monkeypatch):
    monkeypatch.setattr(lm_config, "validate_lm_env", lambda _lm: None)

    config = lm_config.get_lm_config("openai/gpt-5.5", reasoning_effort="medium")

    assert config["reasoning_effort"] == "medium"
    assert "allowed_openai_params" not in config
