import json
import stat
from pathlib import Path

import pytest
from dspy_codex_lm.auth import (
    CODEX_LM_AUTH_PROFILE_ENV,
    auth_status_metadata,
    enable_auth_profile,
    get_active_profile,
    import_auth_profile,
    is_auth_profile_disabled,
    list_auth_profiles,
    list_enabled_auth_profiles,
    load_codex_auth,
    remove_auth_profile,
    set_active_profile,
    validate_profile_name,
)
from dspy_codex_lm.cli import main

LEGACY_AUTH_FALLBACK_ENV = "CODEX_LM_ENABLE_LEGACY_AUTH_FALLBACK"


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolate_auth_home) -> Path:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv(CODEX_LM_AUTH_PROFILE_ENV, raising=False)
    monkeypatch.delenv(LEGACY_AUTH_FALLBACK_ENV, raising=False)
    return tmp_path


def _write_auth(
    path: Path,
    *,
    access_token: str = "abc",
    account_id: str = "acct-1234567890",
    refresh_token: str = "ref",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": access_token,
                    "account_id": account_id,
                    "refresh_token": refresh_token,
                    "id_token": "id-secret",
                },
                "user": {"email": "person@example.com"},
            }
        )
    )
    return path


def test_legacy_credentials_require_explicit_opt_in(fake_home, monkeypatch):
    _write_auth(
        fake_home / ".codex" / "auth.json",
        access_token="fallback-token",
        account_id="acct-fallback",
    )

    with pytest.raises(FileNotFoundError) as excinfo:
        load_codex_auth()

    message = str(excinfo.value)
    assert "fallback-token" not in message
    assert "acct-fallback" not in message

    monkeypatch.setenv(LEGACY_AUTH_FALLBACK_ENV, "1")

    assert load_codex_auth() == ("fallback-token", "acct-fallback")


def test_profile_import_persists_private_credentials_and_removal_clears_selection(
    fake_home: Path,
    tmp_path: Path,
):
    source = _write_auth(
        tmp_path / "source-auth.json",
        access_token="profile-token",
        account_id="acct-profile",
    )
    display_name = "gabriel@trampoline.ai"

    dest = import_auth_profile(display_name, source)
    assert stat.S_IMODE(dest.stat().st_mode) == 0o600

    assert list_auth_profiles() == [display_name]
    assert get_active_profile() == display_name

    assert load_codex_auth() == ("profile-token", "acct-profile")

    remove_auth_profile(display_name)
    assert not dest.parent.exists()
    assert list_auth_profiles() == []
    assert get_active_profile() is None


def test_disable_enable_profile_persists_state_without_deleting_auth(
    fake_home: Path,
    tmp_path: Path,
):
    dest = import_auth_profile(
        "work",
        _write_auth(
            tmp_path / "work.json",
            access_token="work-token",
            account_id="acct-work",
        ),
    )

    enable_auth_profile("work", enabled=False)

    assert dest.is_file()
    assert is_auth_profile_disabled("work") is True
    assert list_auth_profiles() == ["work"]
    assert list_enabled_auth_profiles() == []
    metadata = json.loads((dest.parent / "profile.json").read_text(encoding="utf-8"))
    assert metadata["disabled"] is True

    enable_auth_profile("work", enabled=True)

    assert is_auth_profile_disabled("work") is False
    assert list_enabled_auth_profiles() == ["work"]
    metadata = json.loads((dest.parent / "profile.json").read_text(encoding="utf-8"))
    assert metadata["disabled"] is False


def test_import_profile_rejects_slug_collision(fake_home: Path, tmp_path: Path):
    import_auth_profile(
        "a@b",
        _write_auth(tmp_path / "first.json", access_token="first-token"),
    )

    with pytest.raises(ValueError, match="maps to an existing auth profile slug"):
        import_auth_profile(
            "a#b",
            _write_auth(tmp_path / "second.json", access_token="second-token"),
        )

    assert load_codex_auth(profile="a@b") == ("first-token", "acct-1234567890")


def test_disabled_explicit_env_and_active_profiles_fail_clearly(
    fake_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import_auth_profile(
        "work",
        _write_auth(
            tmp_path / "work.json",
            access_token="work-token",
            account_id="acct-work",
        ),
    )
    set_active_profile("work")
    enable_auth_profile("work", enabled=False)

    with pytest.raises(ValueError, match="auth profile 'work' is disabled"):
        load_codex_auth(profile="work")

    monkeypatch.setenv(CODEX_LM_AUTH_PROFILE_ENV, "work")
    with pytest.raises(ValueError, match="auth profile 'work' is disabled"):
        load_codex_auth()
    monkeypatch.delenv(CODEX_LM_AUTH_PROFILE_ENV)

    with pytest.raises(ValueError, match="active auth profile 'work' is disabled"):
        load_codex_auth()


def test_rotation_skips_disabled_profiles_and_all_disabled_fails(
    fake_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
):
    import_auth_profile(
        "alpha",
        _write_auth(
            tmp_path / "alpha.json",
            access_token="alpha-token",
            account_id="acct-alpha",
        ),
    )
    import_auth_profile(
        "beta",
        _write_auth(
            tmp_path / "beta.json",
            access_token="beta-token",
            account_id="acct-beta",
        ),
    )
    enable_auth_profile("alpha", enabled=False)

    assert main(["codex-lm", "rotation", "on"]) == 0
    capsys.readouterr()

    def choose_only_enabled(profiles):
        return profiles[0]

    monkeypatch.setattr("dspy_codex_lm.auth.random.choice", choose_only_enabled)

    assert load_codex_auth() == ("beta-token", "acct-beta")

    enable_auth_profile("beta", enabled=False)

    with pytest.raises(ValueError, match="rotation is enabled but no enabled"):
        load_codex_auth()


def test_rotation_is_bypassed_by_explicit_and_env_profile(
    fake_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
):
    import_auth_profile(
        "alpha",
        _write_auth(
            tmp_path / "alpha.json",
            access_token="alpha-token",
            account_id="acct-alpha",
        ),
    )
    import_auth_profile(
        "beta",
        _write_auth(
            tmp_path / "beta.json",
            access_token="beta-token",
            account_id="acct-beta",
        ),
    )

    assert main(["codex-lm", "rotation", "on"]) == 0
    capsys.readouterr()

    random_choices = []

    def choose_alpha(profiles):
        random_choices.append([profile.name for profile in profiles])
        return next(profile for profile in profiles if profile.name == "alpha")

    monkeypatch.setattr("dspy_codex_lm.auth.random.choice", choose_alpha)

    explicit = _write_auth(
        tmp_path / "explicit.json",
        access_token="explicit-token",
        account_id="acct-explicit",
    )

    assert load_codex_auth(explicit) == ("explicit-token", "acct-explicit")
    assert load_codex_auth(profile="beta") == ("beta-token", "acct-beta")
    monkeypatch.setenv(CODEX_LM_AUTH_PROFILE_ENV, "beta")
    assert load_codex_auth() == ("beta-token", "acct-beta")
    monkeypatch.delenv(CODEX_LM_AUTH_PROFILE_ENV)
    assert random_choices == []
    assert load_codex_auth() == ("alpha-token", "acct-alpha")
    assert random_choices == [["alpha", "beta"]]


def test_status_metadata_redacts_secrets(fake_home: Path, tmp_path: Path):
    import_auth_profile(
        "work",
        _write_auth(
            tmp_path / "source.json",
            access_token="secret-token",
            account_id="acct-very-secret-account",
        ),
    )
    set_active_profile("work")

    metadata = auth_status_metadata()

    assert "secret-token" not in json.dumps(metadata)
    assert "person@example.com" not in json.dumps(metadata)
    assert "acct-very-secret-account" not in json.dumps(metadata)
    assert "id-secret" not in json.dumps(metadata)
    assert metadata["access_token"] == "present"


@pytest.mark.parametrize("name", ["", "../work", "work/name"])
def test_invalid_profile_names_rejected(name: str):
    with pytest.raises(ValueError):
        validate_profile_name(name)


def test_long_lived_lm_refreshes_disabled_and_reenabled_accounts(fake_home, monkeypatch):
    from conftest import build_stream_events
    from dspy_codex_lm import CodexHTTPLM

    for name in ("alpha", "beta"):
        import_auth_profile(
            name,
            _write_auth(
                fake_home / f"{name}.json",
                access_token=f"{name}-token",
                account_id=f"acct-{name}",
            ),
        )
    assert main(["codex-lm", "rotation", "on"]) == 0
    now = 0.0
    monkeypatch.setattr("dspy_codex_lm.lm.monotonic", lambda: now)
    monkeypatch.setattr(
        "dspy_codex_lm.lm.random.choice",
        lambda credentials: min(credentials, key=lambda credential: credential.account_id),
    )

    def transport(*, headers, api_key, **_):
        account = headers["ChatGPT-Account-Id"]
        assert api_key == account.removeprefix("acct-") + "-token"
        return iter(build_stream_events(account))

    monkeypatch.setattr("dspy_codex_lm.lm.litellm.responses", transport)
    lm = CodexHTTPLM(model="gpt-5.3-codex", auth_config_refresh_seconds=60.0)
    assert lm.forward(prompt="one", cache=False).output[0].content[0].text == "acct-alpha"
    enable_auth_profile("alpha", enabled=False)
    now = 60.0
    assert lm.forward(prompt="two", cache=False).output[0].content[0].text == "acct-beta"
    enable_auth_profile("alpha", enabled=True)
    now = 120.0
    assert lm.forward(prompt="three", cache=False).output[0].content[0].text == "acct-alpha"
