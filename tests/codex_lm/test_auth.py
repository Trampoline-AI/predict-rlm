import json
import stat
from pathlib import Path

import pytest
from dspy_codex_lm.auth import (
    CODEX_LM_AUTH_PROFILE_ENV,
    auth_status_metadata,
    clear_active_profile,
    enable_auth_profile,
    get_active_profile,
    import_auth_profile,
    is_auth_profile_disabled,
    list_auth_profiles,
    list_enabled_auth_profiles,
    load_codex_auth,
    profile_auth_path,
    remove_auth_profile,
    set_active_profile,
    validate_profile_name,
)
from dspy_codex_lm.cli import main

LEGACY_AUTH_FALLBACK_ENV = "CODEX_LM_ENABLE_LEGACY_AUTH_FALLBACK"


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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


def test_loads_access_and_account_id(tmp_path: Path):
    auth = tmp_path / "auth.json"
    _write_auth(auth, account_id="acct-1")
    assert load_codex_auth(auth) == ("abc", "acct-1")


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_codex_auth(tmp_path / "nope.json")


def test_missing_fields_raises(tmp_path: Path):
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"tokens": {"access_token": "abc"}}))
    with pytest.raises(KeyError):
        load_codex_auth(auth)


def test_load_without_selection_does_not_fall_back_by_default(fake_home: Path):
    _write_auth(
        fake_home / ".codex" / "auth.json",
        access_token="fallback-token",
        account_id="acct-fallback",
    )

    with pytest.raises(FileNotFoundError) as excinfo:
        load_codex_auth()

    message = str(excinfo.value)
    assert "codex-lm auth login NAME" in message
    assert "codex-lm auth import NAME" in message
    assert "codex-lm auth use NAME" in message
    assert LEGACY_AUTH_FALLBACK_ENV in message
    assert "fallback-token" not in message
    assert "acct-fallback" not in message


def test_load_falls_back_to_codex_cli_auth_when_enabled(
    fake_home: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _write_auth(
        fake_home / ".codex" / "auth.json",
        access_token="fallback-token",
        account_id="acct-fallback",
    )
    monkeypatch.setenv(LEGACY_AUTH_FALLBACK_ENV, "1")

    assert load_codex_auth() == ("fallback-token", "acct-fallback")


def test_import_list_use_remove_profile(fake_home: Path, tmp_path: Path):
    source = _write_auth(
        tmp_path / "source-auth.json",
        access_token="profile-token",
        account_id="acct-profile",
    )

    dest = import_auth_profile("work.1", source)

    assert dest == profile_auth_path("work.1")
    assert list_auth_profiles() == ["work.1"]
    assert get_active_profile() == "work.1"
    assert stat.S_IMODE(dest.stat().st_mode) == 0o600

    set_active_profile("work.1")
    assert get_active_profile() == "work.1"
    assert load_codex_auth() == ("profile-token", "acct-profile")

    remove_auth_profile("work.1")
    assert list_auth_profiles() == []
    assert get_active_profile() is None


def test_import_profile_sets_default_only_when_none(
    fake_home: Path,
    tmp_path: Path,
):
    import_auth_profile(
        "first",
        _write_auth(tmp_path / "first.json", access_token="first-token"),
    )
    import_auth_profile(
        "second",
        _write_auth(tmp_path / "second.json", access_token="second-token"),
    )

    assert get_active_profile() == "first"
    assert load_codex_auth() == ("first-token", "acct-1234567890")


def test_import_display_profile_uses_safe_slug_and_metadata(
    fake_home: Path,
    tmp_path: Path,
):
    source = _write_auth(
        tmp_path / "source-auth.json",
        access_token="profile-token",
        account_id="acct-profile",
    )
    display_name = "gabriel@trampoline.ai"
    slug = "gabriel-trampoline.ai"

    dest = import_auth_profile(display_name, source)

    assert dest == fake_home / ".codex-lm" / "auth" / slug / "auth.json"
    assert profile_auth_path(display_name) == dest
    assert list_auth_profiles() == [display_name]
    assert json.loads((dest.parent / "profile.json").read_text(encoding="utf-8")) == {
        "name": display_name,
        "slug": slug,
    }
    assert get_active_profile() == display_name

    set_active_profile(display_name)
    assert get_active_profile() == display_name
    assert load_codex_auth() == ("profile-token", "acct-profile")

    remove_auth_profile(display_name)
    assert not dest.parent.exists()
    assert list_auth_profiles() == []
    assert get_active_profile() is None


def test_legacy_profile_dirs_without_metadata_still_work(
    fake_home: Path,
):
    auth = _write_auth(
        fake_home / ".codex-lm" / "auth" / "legacy" / "auth.json",
        access_token="legacy-token",
        account_id="acct-legacy",
    )

    assert list_auth_profiles() == ["legacy"]
    assert profile_auth_path("legacy") == auth

    set_active_profile("legacy")
    assert get_active_profile() == "legacy"
    assert load_codex_auth() == ("legacy-token", "acct-legacy")


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


def test_legacy_profile_dirs_are_enabled_by_default(fake_home: Path):
    _write_auth(
        fake_home / ".codex-lm" / "auth" / "legacy" / "auth.json",
        access_token="legacy-token",
        account_id="acct-legacy",
    )

    assert is_auth_profile_disabled("legacy") is False
    assert list_enabled_auth_profiles() == ["legacy"]


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


def test_env_profile_overrides_active_without_changing_marker(
    fake_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import_auth_profile(
        "active",
        _write_auth(
            tmp_path / "active.json",
            access_token="active-token",
            account_id="acct-active",
        ),
    )
    import_auth_profile(
        "env",
        _write_auth(
            tmp_path / "env.json",
            access_token="env-token",
            account_id="acct-env",
        ),
    )
    set_active_profile("active")

    monkeypatch.setenv(CODEX_LM_AUTH_PROFILE_ENV, "env")

    assert load_codex_auth() == ("env-token", "acct-env")
    assert get_active_profile() == "active"


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


def test_explicit_path_ignores_profile_disabled_state(
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
    set_active_profile("work")
    enable_auth_profile("work", enabled=False)

    assert load_codex_auth(dest) == ("work-token", "acct-work")


def test_rotation_round_robins_saved_profiles_and_wraps(
    fake_home: Path,
    tmp_path: Path,
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
    set_active_profile("beta")

    assert main(["codex-lm", "rotation", "on"]) == 0
    capsys.readouterr()

    assert load_codex_auth() == ("alpha-token", "acct-alpha")
    assert load_codex_auth() == ("beta-token", "acct-beta")
    assert load_codex_auth() == ("alpha-token", "acct-alpha")
    assert get_active_profile() == "beta"


def test_rotation_skips_disabled_profiles_and_all_disabled_fails(
    fake_home: Path,
    tmp_path: Path,
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

    assert load_codex_auth() == ("beta-token", "acct-beta")
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

    assert load_codex_auth(profile="beta") == ("beta-token", "acct-beta")
    monkeypatch.setenv(CODEX_LM_AUTH_PROFILE_ENV, "beta")
    assert load_codex_auth() == ("beta-token", "acct-beta")
    monkeypatch.delenv(CODEX_LM_AUTH_PROFILE_ENV)
    assert load_codex_auth() == ("alpha-token", "acct-alpha")


def test_rotation_status_metadata_does_not_advance_cursor(
    fake_home: Path,
    tmp_path: Path,
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

    metadata = auth_status_metadata()

    assert metadata["source"] == "rotation"
    assert metadata["profile"] == "alpha"
    assert load_codex_auth() == ("alpha-token", "acct-alpha")


def test_explicit_path_overrides_env_and_active(
    fake_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import_auth_profile(
        "active",
        _write_auth(
            tmp_path / "active.json",
            access_token="active-token",
            account_id="acct-active",
        ),
    )
    set_active_profile("active")
    monkeypatch.setenv(CODEX_LM_AUTH_PROFILE_ENV, "active")
    explicit = _write_auth(
        tmp_path / "explicit.json",
        access_token="explicit-token",
        account_id="acct-explicit",
    )

    assert load_codex_auth(explicit) == ("explicit-token", "acct-explicit")


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

    assert metadata["profile"] == "work"
    assert metadata["source"] == "active"
    assert metadata["access_token"] == "present"
    assert metadata["refresh_token"] == "present"
    assert metadata["id_token"] == "present"
    assert metadata["account_id"] == "acct-v...ount"
    assert "secret-token" not in json.dumps(metadata)
    assert "person@example.com" not in json.dumps(metadata)


@pytest.mark.parametrize("name", ["", ".", "..", "../work", "work/name", "work name"])
def test_invalid_profile_names_rejected(name: str):
    with pytest.raises(ValueError):
        validate_profile_name(name)


def test_clear_active_profile_is_idempotent(fake_home: Path):
    clear_active_profile()
    assert get_active_profile() is None
