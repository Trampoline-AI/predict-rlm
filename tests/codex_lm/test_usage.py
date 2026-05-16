import json
from pathlib import Path

from dspy_codex_lm.usage import (
    CODEX_USAGE_ENDPOINT,
    fetch_codex_usage,
    format_disabled_profile_usage_entry,
    format_profile_usage_summaries,
    format_usage_summary,
    summarize_usage,
)


def _auth_file(tmp_path: Path) -> Path:
    path = tmp_path / "auth.json"
    path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "secret-token",
                    "account_id": "acct-secret",
                    "refresh_token": "secret-refresh",
                },
                "user": {"email": "person@example.com"},
            }
        )
    )
    return path


def test_fetch_codex_usage_builds_required_headers(tmp_path: Path):
    seen = {}

    def fake_transport(url, *, headers, timeout):
        seen["url"] = url
        seen["headers"] = headers
        seen["timeout"] = timeout
        return {"rate_limit": {}}

    payload = fetch_codex_usage(
        auth_path=_auth_file(tmp_path),
        transport=fake_transport,
    )

    assert payload == {"rate_limit": {}}
    assert seen == {
        "url": CODEX_USAGE_ENDPOINT,
        "headers": {
            "Authorization": "Bearer secret-token",
            "ChatGPT-Account-ID": "acct-secret",
            "Accept": "application/json",
        },
        "timeout": 10.0,
    }


def test_summarize_usage_parses_rate_limit_credit_and_model_windows():
    payload = {
        "rate_limit": {
            "primary": {
                "used": 12,
                "limit": 40,
                "remaining": 28,
                "reset_at": "2026-05-10T12:00:00Z",
            },
            "secondary": {
                "current_value": 9,
                "max_value": 10,
                "resets_in_seconds": 1800,
            },
        },
        "credits": {
            "granted": 1000,
            "used": 250,
            "expires_at": "2026-06-01T00:00:00Z",
        },
        "additional": {
            "models": {
                "gpt-5.3-codex": {
                    "rate_limit": {
                        "remaining": 3,
                        "limit": 20,
                        "reset_after_seconds": 600,
                    }
                }
            }
        },
        "user": {"email": "person@example.com"},
        "account_id": "acct-secret",
    }

    rows = summarize_usage(payload)

    assert [row.label for row in rows] == [
        "additional.models.gpt-5.3-codex.rate_limit",
        "credits",
        "rate_limit.primary",
        "rate_limit.secondary",
    ]
    assert rows[0].percent_remaining == 15.0
    assert rows[0].reset == "in 10m"
    assert rows[1].remaining == 750
    assert rows[1].percent_remaining == 75.0
    assert rows[1].reset == "2026-06-01T00:00:00Z"
    assert rows[2].remaining == 28
    assert rows[2].percent_remaining == 70.0
    assert rows[2].reset == "2026-05-10T12:00:00Z"
    assert rows[3].remaining == 1
    assert rows[3].percent_remaining == 10.0
    assert rows[3].reset == "in 30m"


def test_summarize_usage_parses_live_wham_shape():
    payload = {
        "plan_type": "pro",
        "rate_limit": {
            "allowed": True,
            "limit_reached": False,
            "primary_window": {
                "used_percent": 8,
                "limit_window_seconds": 18000,
                "reset_after_seconds": 5719,
                "reset_at": 1778387445,
            },
            "secondary_window": {
                "used_percent": 35,
                "limit_window_seconds": 604800,
                "reset_after_seconds": 157475,
                "reset_at": 1778539200,
            },
        },
        "additional_rate_limits": [
            {
                "limit_name": "GPT-5.3-Codex-Spark",
                "metered_feature": "codex.spark",
                "rate_limit": {
                    "primary_window": {
                        "used_percent": 0,
                        "limit_window_seconds": 18000,
                        "reset_after_seconds": 18000,
                    },
                    "secondary_window": {
                        "used_percent": 0,
                        "limit_window_seconds": 604800,
                        "reset_after_seconds": 604800,
                    },
                },
            }
        ],
        "credits": {
            "balance": "0",
            "has_credits": False,
            "unlimited": False,
            "approx_cloud_messages": [0, 0],
        },
        "user_id": "user-secret",
    }

    rows = summarize_usage(payload)

    assert [row.label for row in rows] == [
        "GPT-5.3-Codex-Spark.primary_window",
        "GPT-5.3-Codex-Spark.secondary_window",
        "rate_limit.primary_window",
        "rate_limit.secondary_window",
    ]
    assert rows[0].percent_remaining == 100.0
    assert rows[0].reset == "in 5h"
    assert rows[1].percent_remaining == 100.0
    assert rows[1].reset == "in 7d"
    assert rows[2].percent_remaining == 92.0
    assert rows[2].reset == "04:30 on 10 May"
    assert rows[3].percent_remaining == 65.0
    assert rows[3].reset == "22:40 on 11 May"

    text = format_usage_summary(payload)
    assert "Plan: pro" in text
    assert "Credits: balance 0; has_credits=false" in text
    assert (
        "5h limit:                    [██████████████████░░] 92% left (resets 04:30 on 10 May)"
    ) in text
    assert (
        "Weekly limit:                [█████████████░░░░░░░] 65% left (resets 22:40 on 11 May)"
    ) in text
    assert "GPT-5.3-Codex-Spark limit:" in text
    assert (
        "  5h limit:                    [████████████████████] 100% left (resets in 5h)"
    ) in text
    assert (
        "  Weekly limit:                [████████████████████] 100% left (resets in 7d)"
    ) in text


def test_format_usage_summary_groups_top_level_live_windows_under_general_header():
    payload = {
        "rate_limit": {
            "primary_window": {
                "used_percent": 12,
                "limit_window_seconds": 18000,
                "reset_after_seconds": 300,
            },
            "secondary_window": {
                "used_percent": 35,
                "limit_window_seconds": 604800,
                "reset_after_seconds": 604800,
            },
        },
        "additional_rate_limits": [
            {
                "limit_name": "GPT-5.3-Codex-Spark",
                "rate_limit": {
                    "primary_window": {
                        "used_percent": 0,
                        "limit_window_seconds": 18000,
                        "reset_after_seconds": 18000,
                    },
                },
            }
        ],
    }

    text = format_usage_summary(payload)

    lines = text.splitlines()
    assert lines[0] == "-" * 60
    assert lines[-1] == "-" * 60
    assert lines[1:-1] == [
        "Codex usage",
        "General usage limits:",
        "  5h limit:                    [██████████████████░░] 88% left (resets in 5m)",
        "  Weekly limit:                [█████████████░░░░░░░] 65% left (resets in 7d)",
        "GPT-5.3-Codex-Spark limit:",
        "  5h limit:                    [████████████████████] 100% left (resets in 5h)",
    ]


def test_format_profile_usage_summaries_groups_general_and_model_limits_per_profile():
    work_payload = {
        "rate_limit": {
            "primary_window": {
                "used_percent": 12,
                "limit_window_seconds": 18000,
                "reset_after_seconds": 300,
            },
        },
        "additional_rate_limits": [
            {
                "limit_name": "GPT-5.3-Codex-Spark",
                "rate_limit": {
                    "primary_window": {
                        "used_percent": 0,
                        "limit_window_seconds": 18000,
                        "reset_after_seconds": 18000,
                    },
                },
            }
        ],
    }
    personal_payload = {
        "rate_limit": {
            "secondary_window": {
                "used_percent": 35,
                "limit_window_seconds": 604800,
                "reset_after_seconds": 604800,
            },
        },
        "additional_rate_limits": [
            {
                "limit_name": "GPT-5.4",
                "rate_limit": {
                    "primary_window": {
                        "used_percent": 20,
                        "limit_window_seconds": 18000,
                        "reset_after_seconds": 60,
                    },
                },
            }
        ],
    }

    text = format_profile_usage_summaries(
        [("work", work_payload), ("personal", personal_payload)],
        default_profile="personal",
    )

    lines = text.splitlines()
    assert lines[0] == "-" * 60
    assert lines[-1] == "-" * 60
    assert lines[1:-1] == [
        "work:",
        "  General usage limits:",
        "    5h limit:                    [██████████████████░░] 88% left (resets in 5m)",
        "  GPT-5.3-Codex-Spark limit:",
        "    5h limit:                    [████████████████████] 100% left (resets in 5h)",
        "",
        "personal (default):",
        "  General usage limits:",
        "    Weekly limit:                [█████████████░░░░░░░] 65% left (resets in 7d)",
        "  GPT-5.4 limit:",
        "    5h limit:                    [████████████████░░░░] 80% left (resets in 1m)",
    ]


def test_format_profile_usage_summaries_preserves_profile_email_display_name():
    payload = {
        "rate_limit": {
            "primary_window": {
                "used_percent": 12,
                "limit_window_seconds": 18000,
                "reset_after_seconds": 300,
            },
        },
        "user": {"email": "payload@example.com"},
    }

    text = format_profile_usage_summaries([("gabriel@example.com", payload)])
    colored = format_profile_usage_summaries(
        [("gabriel@example.com", payload)],
        color=True,
    )

    assert "gabriel@example.com:" in text
    assert "[redacted-email]:" not in text
    assert "payload@example.com" not in text
    assert "\x1b[1;36mgabriel@example.com:\x1b[0m" in colored
    assert "[redacted-email]:" not in colored
    assert "payload@example.com" not in colored


def test_format_disabled_profile_usage_entry_is_labeled_and_redacted():
    text = format_profile_usage_summaries(
        [
            ("work", {"rate_limit": {"primary": {"used": 1, "limit": 2}}}),
            format_disabled_profile_usage_entry("acct-secret"),
        ],
        default_profile="acct-secret",
    )

    assert "work:" in text
    assert "[redacted-account] (disabled) (default):" in text
    assert "  Disabled; live usage fetch skipped." in text
    assert "acct-secret" not in text


def test_format_usage_summary_color_can_be_enabled_and_disabled():
    payload = {
        "rate_limit": {
            "primary_window": {
                "used_percent": 12,
                "limit_window_seconds": 18000,
                "reset_after_seconds": 300,
            },
        },
    }

    plain = format_usage_summary(payload, color=False)
    colored = format_usage_summary(payload, color=True)

    assert "\x1b[" not in plain
    assert "\x1b[" in colored
    assert "General usage limits:" in colored
    assert "[██████████████████░░] 88% left" in colored


def test_format_usage_summary_is_redacted_and_stable():
    payload = {
        "rate_limit": {
            "primary": {
                "used": 1,
                "limit": 4,
                "remaining": 3,
                "reset_at": "2026-05-10T12:00:00Z",
            }
        },
        "account_id": "acct-secret",
        "email": "person@example.com",
        "access_token": "secret-token",
    }

    text = format_usage_summary(payload)

    lines = text.splitlines()
    assert lines[0] == "-" * 60
    assert lines[-1] == "-" * 60
    assert lines[1:-1] == [
        "Codex usage",
        "rate_limit.primary: 3/4 remaining (75.0% remaining); resets 2026-05-10T12:00:00Z",
    ]
    assert "secret-token" not in text
    assert "acct-secret" not in text
    assert "person@example.com" not in text
