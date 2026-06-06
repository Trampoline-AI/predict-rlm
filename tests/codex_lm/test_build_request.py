def test_build_request_from_prompt(lm):
    request, headers = lm._build_request(prompt="hello world", messages=None, kwargs={})
    assert request["store"] is False
    assert request["stream"] is True
    assert request["instructions"]  # non-empty
    # Input was converted to list form (Responses API)
    assert isinstance(request["input"], list)
    assert request["input"][0]["role"] == "user"
    assert request["input"][0]["content"][0]["type"] == "input_text"
    assert request["input"][0]["content"][0]["text"] == "hello world"


def test_build_request_from_messages(lm):
    request, _ = lm._build_request(
        prompt=None,
        messages=[{"role": "user", "content": "hi"}],
        kwargs={},
    )
    assert request["input"][0]["content"][0]["text"] == "hi"


def test_build_request_headers_include_account_id(lm):
    _, headers = lm._build_request(prompt="x", messages=None, kwargs={})
    assert headers["ChatGPT-Account-Id"] == "fake-account"
    assert headers["originator"] == "opencode"
    assert "session_id" in headers


def test_each_call_gets_unique_session_id(lm):
    _, h1 = lm._build_request(prompt="x", messages=None, kwargs={})
    _, h2 = lm._build_request(prompt="x", messages=None, kwargs={})
    assert h1["session_id"] != h2["session_id"]


def test_build_request_drops_rollout_id_and_cache(lm):
    request, _ = lm._build_request(
        prompt="x", messages=None, kwargs={"rollout_id": "r1", "cache": True}
    )
    assert "rollout_id" not in request
    assert "cache" not in request


def test_build_request_canonicalizes_model_for_codex_backend(lm):
    request, _ = lm._build_request(prompt="x", messages=None, kwargs={})
    assert request["model"] == "gpt-5.3-codex"
    assert request["custom_llm_provider"] == "openai"


def test_build_request_canonicalizes_spark_model():
    from dspy_codex_lm import CodexHTTPLM as CodexLM
    from litellm.utils import supports_native_streaming

    spark = CodexLM(
        model="gpt-5.3-codex-spark",
        access_token="fake",
        account_id="acct",
    )
    request, _ = spark._build_request(prompt="x", messages=None, kwargs={})
    assert request["model"] == "gpt-5.3-codex-spark"
    assert request["stream"] is True
    assert request["custom_llm_provider"] == "openai"
    assert supports_native_streaming("gpt-5.3-codex-spark", "openai") is True


def test_custom_instructions_via_ctor(lm):
    from dspy_codex_lm import CodexHTTPLM as CodexLM

    custom = CodexLM(
        model="gpt-5.3-codex",
        instructions="You are Ada.",
        access_token="fake",
        account_id="acct",
    )
    request, _ = custom._build_request(prompt="x", messages=None, kwargs={})
    assert request["instructions"] == "You are Ada."


def test_ctor_can_pin_auth_profile(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from dspy_codex_lm import CodexHTTPLM as CodexLM
    from dspy_codex_lm.auth import import_auth_profile

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    source = tmp_path / "source-auth.json"
    source.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "profile-token",
                    "account_id": "acct-profile",
                }
            }
        ),
        encoding="utf-8",
    )
    import_auth_profile("pro", source)

    custom = CodexLM(model="gpt-5.3-codex", auth_profile="pro")

    _, headers = custom._build_request(prompt="x", messages=None, kwargs={})
    assert headers["ChatGPT-Account-Id"] == "acct-profile"


def test_long_lived_lm_randomly_selects_accounts_per_request(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from dspy_codex_lm import CodexHTTPLM as CodexLM
    from dspy_codex_lm.auth import import_auth_profile
    from dspy_codex_lm.cli import main

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    def write_auth(path: Path, *, access_token: str, account_id: str) -> Path:
        path.write_text(
            json.dumps(
                {
                    "tokens": {
                        "access_token": access_token,
                        "account_id": account_id,
                    }
                }
            ),
            encoding="utf-8",
        )
        return path

    import_auth_profile(
        "alpha",
        write_auth(
            tmp_path / "alpha.json",
            access_token="alpha-token",
            account_id="acct-alpha",
        ),
    )
    import_auth_profile(
        "beta",
        write_auth(
            tmp_path / "beta.json",
            access_token="beta-token",
            account_id="acct-beta",
        ),
    )
    assert main(["codex-lm", "rotation", "on"]) == 0

    seen_headers = []
    seen_api_keys = []

    def fake_responses(*, headers, api_key, **kwargs):
        seen_headers.append(dict(headers))
        seen_api_keys.append(api_key)
        return iter(
            [
                SimpleNamespace(type="response.output_text.delta", delta="ok"),
                SimpleNamespace(
                    type="response.completed",
                    response={
                        "id": "resp",
                        "model": "gpt-5.3-codex",
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 1,
                            "total_tokens": 2,
                        },
                    },
                ),
            ]
        )

    monkeypatch.setattr("dspy_codex_lm.lm.litellm.responses", fake_responses)

    random_choices = []
    selected_accounts = iter(["acct-beta", "acct-alpha"])

    def choose_credentials(credentials):
        credentials = tuple(credentials)
        random_choices.append([credential.account_id for credential in credentials])
        selected = next(selected_accounts)
        return next(
            credential for credential in credentials if credential.account_id == selected
        )

    monkeypatch.setattr("dspy_codex_lm.lm.random.choice", choose_credentials)

    lm = CodexLM(model="gpt-5.3-codex")
    lm.forward(prompt="one", cache=False)
    lm.forward(prompt="two", cache=False)

    assert [item["ChatGPT-Account-Id"] for item in seen_headers] == [
        "acct-beta",
        "acct-alpha",
    ]
    assert seen_api_keys == ["beta-token", "alpha-token"]
    assert random_choices == [["acct-alpha", "acct-beta"], ["acct-alpha", "acct-beta"]]


def test_long_lived_lm_resyncs_disabled_and_reenabled_profiles(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from dspy_codex_lm import CodexHTTPLM as CodexLM
    from dspy_codex_lm.auth import enable_auth_profile, import_auth_profile
    from dspy_codex_lm.cli import main

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    now = 0.0
    monkeypatch.setattr("dspy_codex_lm.lm.monotonic", lambda: now)

    def write_auth(path: Path, *, access_token: str, account_id: str) -> Path:
        path.write_text(
            json.dumps(
                {
                    "tokens": {
                        "access_token": access_token,
                        "account_id": account_id,
                    }
                }
            ),
            encoding="utf-8",
        )
        return path

    import_auth_profile(
        "alpha",
        write_auth(
            tmp_path / "alpha.json",
            access_token="alpha-token",
            account_id="acct-alpha",
        ),
    )
    import_auth_profile(
        "beta",
        write_auth(
            tmp_path / "beta.json",
            access_token="beta-token",
            account_id="acct-beta",
        ),
    )
    assert main(["codex-lm", "rotation", "on"]) == 0

    selected_accounts = iter(
        [
            "acct-alpha",
            "acct-beta",
            "acct-alpha",
            "acct-beta",
            "acct-beta",
            "acct-alpha",
        ]
    )

    def choose_credentials(credentials):
        credentials = tuple(credentials)
        selected = next(selected_accounts)
        return next(
            credential for credential in credentials if credential.account_id == selected
        )

    monkeypatch.setattr("dspy_codex_lm.lm.random.choice", choose_credentials)

    lm = CodexLM(model="gpt-5.3-codex", auth_config_refresh_seconds=60.0)
    _, headers = lm._build_request(prompt="one", messages=None, kwargs={})
    enable_auth_profile("alpha", enabled=False)
    _, stale_headers = lm._build_request(prompt="two", messages=None, kwargs={})
    _, still_stale_headers = lm._build_request(
        prompt="three",
        messages=None,
        kwargs={},
    )
    now = 60.0
    _, refreshed_headers = lm._build_request(prompt="four", messages=None, kwargs={})
    enable_auth_profile("alpha", enabled=True)
    _, stale_reenabled_headers = lm._build_request(
        prompt="five",
        messages=None,
        kwargs={},
    )
    now = 120.0
    _, reenabled_headers = lm._build_request(prompt="six", messages=None, kwargs={})

    assert [
        item["ChatGPT-Account-Id"]
        for item in [
            headers,
            stale_headers,
            still_stale_headers,
            refreshed_headers,
            stale_reenabled_headers,
            reenabled_headers,
        ]
    ] == [
        "acct-alpha",
        "acct-beta",
        "acct-alpha",
        "acct-beta",
        "acct-beta",
        "acct-alpha",
    ]


def test_long_lived_lm_randomly_selects_cached_snapshot_without_auth_reload(
    tmp_path,
    monkeypatch,
):
    import json
    from pathlib import Path

    from dspy_codex_lm import CodexHTTPLM as CodexLM
    from dspy_codex_lm.auth import import_auth_profile
    from dspy_codex_lm.cli import main

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr("dspy_codex_lm.lm.monotonic", lambda: 0.0)

    def write_auth(path: Path, *, access_token: str, account_id: str) -> Path:
        path.write_text(
            json.dumps(
                {
                    "tokens": {
                        "access_token": access_token,
                        "account_id": account_id,
                    }
                }
            ),
            encoding="utf-8",
        )
        return path

    import_auth_profile(
        "alpha",
        write_auth(
            tmp_path / "alpha.json",
            access_token="alpha-token",
            account_id="acct-alpha",
        ),
    )
    import_auth_profile(
        "beta",
        write_auth(
            tmp_path / "beta.json",
            access_token="beta-token",
            account_id="acct-beta",
        ),
    )
    assert main(["codex-lm", "rotation", "on"]) == 0

    import dspy_codex_lm.lm as lm_module

    original_load_codex_auth = lm_module.load_codex_auth
    load_calls = []

    def counting_load_codex_auth(*args, **kwargs):
        load_calls.append((args, kwargs))
        return original_load_codex_auth(*args, **kwargs)

    monkeypatch.setattr(lm_module, "load_codex_auth", counting_load_codex_auth)

    random_choices = []
    selected_accounts = iter(["acct-beta", "acct-alpha", "acct-beta"])

    def choose_credentials(credentials):
        credentials = tuple(credentials)
        random_choices.append([credential.account_id for credential in credentials])
        selected = next(selected_accounts)
        return next(
            credential for credential in credentials if credential.account_id == selected
        )

    monkeypatch.setattr("dspy_codex_lm.lm.random.choice", choose_credentials)

    lm = CodexLM(model="gpt-5.3-codex", auth_config_refresh_seconds=60.0)
    first, first_headers = lm._build_request(prompt="one", messages=None, kwargs={})
    second, second_headers = lm._build_request(prompt="two", messages=None, kwargs={})
    third, third_headers = lm._build_request(prompt="three", messages=None, kwargs={})

    assert [
        item["ChatGPT-Account-Id"]
        for item in [
            first_headers,
            second_headers,
            third_headers,
        ]
    ] == [
        "acct-beta",
        "acct-alpha",
        "acct-beta",
    ]
    assert [item["api_key"] for item in [first, second, third]] == [
        "beta-token",
        "alpha-token",
        "beta-token",
    ]
    assert random_choices == [
        ["acct-alpha", "acct-beta"],
        ["acct-alpha", "acct-beta"],
        ["acct-alpha", "acct-beta"],
    ]
    assert len(load_calls) == 2


def test_pinned_access_token_bypasses_rotation_random_choice(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from dspy_codex_lm import CodexHTTPLM as CodexLM
    from dspy_codex_lm.auth import import_auth_profile
    from dspy_codex_lm.cli import main

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    def write_auth(path: Path, *, access_token: str, account_id: str) -> Path:
        path.write_text(
            json.dumps(
                {
                    "tokens": {
                        "access_token": access_token,
                        "account_id": account_id,
                    }
                }
            ),
            encoding="utf-8",
        )
        return path

    import_auth_profile(
        "alpha",
        write_auth(
            tmp_path / "alpha.json",
            access_token="alpha-token",
            account_id="acct-alpha",
        ),
    )
    assert main(["codex-lm", "rotation", "on"]) == 0

    def fail_if_random_choice_used(credentials):
        raise AssertionError(f"unexpected random rotation from {credentials}")

    monkeypatch.setattr("dspy_codex_lm.lm.random.choice", fail_if_random_choice_used)

    lm = CodexLM(
        model="gpt-5.3-codex",
        access_token="pinned-token",
        account_id="acct-pinned",
    )
    request, headers = lm._build_request(prompt="one", messages=None, kwargs={})

    assert request["api_key"] == "pinned-token"
    assert headers["ChatGPT-Account-Id"] == "acct-pinned"
