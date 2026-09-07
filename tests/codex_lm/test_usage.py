from dspy_codex_lm.usage import format_profile_usage_summaries, summarize_usage


def test_usage_derives_remaining_credit_and_nested_model_limits():
    payload = {
        "credits": {"granted": 1000, "used": 250},
        "rate_limit": {"primary": {"current_value": 9, "max_value": 10}},
        "models": {"codex": {"rate_limit": {"remaining": 3, "limit": 20}}},
    }
    rows = {row.label: row for row in summarize_usage(payload)}
    assert rows["credits"].remaining == 750
    assert rows["credits"].percent_remaining == 75
    assert rows["rate_limit.primary"].remaining == 1
    assert rows["rate_limit.primary"].percent_remaining == 10
    assert rows["models.codex.rate_limit"].percent_remaining == 15


def test_live_usage_windows_are_not_conflated_with_model_specific_limits():
    payload = {
        "rate_limit": {
            "primary_window": {"used_percent": 8, "limit_window_seconds": 18000},
            "secondary_window": {"used_percent": 35, "limit_window_seconds": 604800},
        },
        "additional_rate_limits": [
            {
                "limit_name": "GPT-5.3-Codex-Spark",
                "rate_limit": {
                    "primary_window": {"used_percent": 0, "limit_window_seconds": 18000},
                },
            }
        ],
    }
    rows = {row.label: row.percent_remaining for row in summarize_usage(payload)}
    assert rows == {
        "rate_limit.primary_window": 92,
        "rate_limit.secondary_window": 65,
        "GPT-5.3-Codex-Spark.primary_window": 100,
    }


def test_profile_summary_preserves_display_name_without_leaking_payload_secrets():
    payload = {
        "rate_limit": {"primary": {"used": 1, "limit": 4}},
        "user": {"email": "payload@example.com"},
        "account_id": "acct-secret",
        "access_token": "secret-token",
    }
    text = format_profile_usage_summaries([("profile@example.com", payload)])
    assert "profile@example.com" in text
    assert "75.0%" in text
    assert "payload@example.com" not in text
    assert "acct-secret" not in text
    assert "secret-token" not in text
