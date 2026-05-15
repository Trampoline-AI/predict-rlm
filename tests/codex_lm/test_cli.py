from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import dspy
import pytest
from dspy_codex_lm.auth import CODEX_LM_AUTH_PROFILE_ENV, load_codex_auth
from dspy_codex_lm.cli import (
    CODEX_SUPPORTED_MODELS,
    CodexLMUnsupportedModelError,
    install_monkeypatch,
    is_openai_family,
    main,
    resolve_codex_model,
    restore_monkeypatch,
)


@pytest.fixture
def restore_lm():
    original_top = dspy.LM
    try:
        import dspy.clients.lm as mod

        original_inner = mod.LM
    except Exception:
        mod = None
        original_inner = None
    yield
    dspy.LM = original_top
    if mod is not None:
        mod.LM = original_inner


@pytest.fixture(autouse=True)
def fake_auth():
    with mock.patch("dspy_codex_lm.lm.load_codex_auth", return_value=("fake", "fake-acct")):
        yield


def _write_script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "user_script.py"
    path.write_text(body)
    return path


def _write_auth(
    path: Path,
    *,
    access_token: str = "secret-token",
    account_id: str = "acct-secret-account",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            '{"tokens": {"access_token": "%s", "account_id": "%s", '
            '"refresh_token": "secret-refresh"}, '
            '"user": {"email": "person@example.com"}}'
        )
        % (access_token, account_id)
    )
    return path


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv(CODEX_LM_AUTH_PROFILE_ENV, raising=False)
    monkeypatch.delenv("CODEX_LM_ENABLE_LEGACY_AUTH_FALLBACK", raising=False)
    return home


# ---- pattern matcher unit tests ----


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-4o",
        "openai/o3",
        "gpt-4o",
        "gpt-4.1-mini",
        "gpt-5",
        "o1-mini",
        "o3",
        "o4-mini",
        "chatgpt-4o-latest",
    ],
)
def test_intercept_matches_openai_family(model: str):
    assert is_openai_family(model) is True


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-3-5-sonnet",
        "claude-3-opus",
        "google/gemini-2.0-flash",
        "gemini-pro",
        "cohere/command-r",
        "together_ai/meta-llama/Llama-3",
        "ollama/llama3",
    ],
)
def test_intercept_skips_other_providers(model: str):
    assert is_openai_family(model) is False


# ---- resolver unit tests ----


@pytest.mark.parametrize("slug", sorted(CODEX_SUPPORTED_MODELS))
def test_resolve_returns_codex_slug_unchanged(slug: str):
    assert resolve_codex_model(slug) == slug


def test_resolve_codex_model_strips_openai_prefix():
    assert resolve_codex_model("openai/gpt-5.3-codex") == "gpt-5.3-codex"


def test_resolve_codex_model_accepts_spark_slug():
    assert resolve_codex_model("openai/gpt-5.3-codex-spark") == "gpt-5.3-codex-spark"


@pytest.mark.parametrize(
    "slug",
    [
        "gpt-4o",
        "openai/gpt-4o",
        "gpt-4.1",
        "gpt-5",
        "gpt-5-mini",
        "o3",
        "o4-mini",
        "openai/o3",
    ],
)
def test_resolve_raises_for_unsupported(slug: str):
    with pytest.raises(CodexLMUnsupportedModelError) as excinfo:
        resolve_codex_model(slug)
    msg = str(excinfo.value)
    assert slug in msg
    # Error message includes the supported set so users can self-serve
    assert "gpt-5.3-codex" in msg


# ---- install / restore ----


def test_install_and_restore(restore_lm):
    original = dspy.LM
    install_monkeypatch(verbose=False)
    assert dspy.LM is not original
    restore_monkeypatch(original, original)
    assert dspy.LM is original


def test_install_intercepts_codex_slug(restore_lm):
    install_monkeypatch(verbose=False)
    lm = dspy.LM(model="openai/gpt-5.3-codex", api_key="sk-fake")
    from dspy_codex_lm import CodexLM

    assert isinstance(lm, CodexLM)
    assert lm.model == "openai/gpt-5.3-codex"


def test_lm_copy_does_not_trigger_intercept(restore_lm):
    """``lm.copy()`` uses ``copy.deepcopy``, which calls
    ``cls.__new__(cls)`` with NO model argument. Before the sentinel-
    default fix, that kicked into the ``model="gpt-4o-mini"`` default
    and raised ``CodexLMUnsupportedModelError``. Pin that deepcopy goes
    through cleanly regardless of what the default is.
    """
    install_monkeypatch(verbose=False)
    lm = dspy.LM(model="anthropic/claude-opus-4-6", api_key="sk-fake")
    # This triggers copy.deepcopy under the hood
    lm_copy = lm.copy()
    # No exception; copy behaves like the original
    assert type(lm_copy) is type(lm)
    assert lm_copy.model == lm.model
    assert lm_copy.history == []  # DSPy's copy resets history


def test_codex_lm_copy_does_not_trigger_intercept(restore_lm):
    """Same guarantee for an already-intercepted CodexLM instance."""
    install_monkeypatch(verbose=False)
    lm = dspy.LM(model="openai/gpt-5.3-codex", api_key="sk-fake")
    lm_copy = lm.copy()
    assert type(lm_copy) is type(lm)
    assert lm_copy.model == "openai/gpt-5.3-codex"


def test_intercepted_instance_tagged_with_source_model(restore_lm):
    install_monkeypatch(verbose=False)
    lm = dspy.LM(model="openai/gpt-5.3-codex", api_key="sk-fake")
    assert lm.codex_intercepted_from == "openai/gpt-5.3-codex"


def test_passthrough_has_no_intercept_attr(restore_lm):
    install_monkeypatch(verbose=False)
    lm = dspy.LM(model="anthropic/claude-3-5-sonnet", api_key="sk-fake")
    assert not hasattr(lm, "codex_intercepted_from")


def test_intercept_emits_info_log(restore_lm, caplog):
    import logging as _logging

    caplog.set_level(_logging.INFO, logger="dspy_codex_lm.cli")
    install_monkeypatch(verbose=False)
    dspy.LM(model="openai/gpt-5.3-codex", api_key="sk-fake")
    records = [r for r in caplog.records if r.name == "dspy_codex_lm.cli"]
    assert len(records) == 1
    assert records[0].levelname == "INFO"
    assert "intercept" in records[0].getMessage()
    assert "openai/gpt-5.3-codex" in records[0].getMessage()
    assert "gpt-5.3-codex" in records[0].getMessage()


def test_no_log_for_passthrough(restore_lm, caplog):
    import logging as _logging

    caplog.set_level(_logging.INFO, logger="dspy_codex_lm.cli")
    install_monkeypatch(verbose=False)
    dspy.LM(model="anthropic/claude-3-5-sonnet", api_key="sk-fake")
    records = [r for r in caplog.records if r.name == "dspy_codex_lm.cli"]
    assert records == []


def test_install_raises_on_unsupported_openai_model(restore_lm):
    install_monkeypatch(verbose=False)
    with pytest.raises(CodexLMUnsupportedModelError):
        dspy.LM(model="openai/gpt-4o", api_key="sk-fake")


def test_install_passes_through_non_openai(restore_lm):
    install_monkeypatch(verbose=False)
    lm = dspy.LM(model="anthropic/claude-3-5-sonnet", api_key="sk-fake")
    from dspy_codex_lm import CodexLM

    assert not isinstance(lm, CodexLM)
    assert lm.model == "anthropic/claude-3-5-sonnet"


def test_install_drops_conflicting_kwargs(restore_lm):
    install_monkeypatch(verbose=False)
    # If we didn't strip api_key/api_base/model_type, CodexLM would get duplicate kwargs
    lm = dspy.LM(
        model="openai/gpt-5.3-codex",
        api_key="sk-fake",
        api_base="https://should-be-dropped",
        model_type="chat",
    )
    from dspy_codex_lm import CodexLM

    assert isinstance(lm, CodexLM)
    # model_type should be "responses" (CodexLM's own value), not "chat"
    assert lm.model_type == "responses"


# ---- main() end-to-end ----


def test_main_intercepts_in_child_script(tmp_path, restore_lm, capsys):
    script = _write_script(
        tmp_path,
        """
import dspy
from dspy_codex_lm import CodexLM
lm = dspy.LM(model="openai/gpt-5.3-codex", api_key="sk-fake")
assert isinstance(lm, CodexLM), f"expected CodexLM, got {type(lm).__name__}"
print("OK")
""",
    )
    rc = main(["codex-lm", str(script)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "OK" in captured.out
    assert "intercept" in captured.err


def test_child_script_can_inspect_intercept_attr(tmp_path, restore_lm, capsys):
    script = _write_script(
        tmp_path,
        """
import dspy
lm = dspy.LM(model="openai/gpt-5.3-codex", api_key="sk-fake")
assert lm.codex_intercepted_from == "openai/gpt-5.3-codex"
print(f"came_from={lm.codex_intercepted_from}")
""",
    )
    rc = main(["codex-lm", str(script)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "came_from=openai/gpt-5.3-codex" in captured.out


def test_child_script_can_configure_log_capture(tmp_path, restore_lm, capsys):
    script = _write_script(
        tmp_path,
        """
import logging
logs = []

class Grab(logging.Handler):
    def emit(self, record):
        logs.append(record.getMessage())

logging.getLogger("dspy_codex_lm.cli").addHandler(Grab())
logging.getLogger("dspy_codex_lm.cli").setLevel(logging.INFO)

import dspy
dspy.LM(model="openai/gpt-5.3-codex", api_key="sk-fake")
assert any("intercept" in m for m in logs), f"no intercept log captured: {logs}"
print(f"logged={logs[0]}")
""",
    )
    rc = main(["codex-lm", str(script)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "logged=intercept" in captured.out


def test_main_exits_3_on_unsupported_model(tmp_path, restore_lm, capsys):
    script = _write_script(
        tmp_path,
        "import dspy; dspy.LM(model='openai/gpt-4o', api_key='sk-fake')",
    )
    rc = main(["codex-lm", str(script)])
    captured = capsys.readouterr()
    assert rc == 3
    assert "cannot route" in captured.err
    assert "gpt-4o" in captured.err


def test_main_passes_through_non_openai_in_child(tmp_path, restore_lm, capsys):
    script = _write_script(
        tmp_path,
        """
import dspy
from dspy_codex_lm import CodexLM
lm = dspy.LM(model="anthropic/claude-3-5-sonnet", api_key="sk-fake")
assert not isinstance(lm, CodexLM), "should not be intercepted"
print("PASSTHROUGH")
""",
    )
    rc = main(["codex-lm", str(script)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "PASSTHROUGH" in captured.out
    assert "intercept" not in captured.err


def test_main_forwards_argv_to_child(tmp_path, restore_lm, capsys):
    script = _write_script(
        tmp_path,
        "import sys; print('argv=' + '|'.join(sys.argv))",
    )
    main(["codex-lm", str(script), "--flag", "value", "pos1"])
    captured = capsys.readouterr()
    assert "--flag|value|pos1" in captured.out
    # argv[0] should be the script path, not codex-lm
    assert captured.out.startswith(f"argv={script}")


def test_main_forwards_exit_code(tmp_path, restore_lm):
    script = _write_script(tmp_path, "import sys; sys.exit(42)")
    assert main(["codex-lm", str(script)]) == 42


def test_main_quiet_env_suppresses_banner(tmp_path, restore_lm, monkeypatch, capsys):
    monkeypatch.setenv("CODEX_LM_QUIET", "1")
    script = _write_script(
        tmp_path,
        "import dspy; dspy.LM(model='openai/gpt-5.3-codex')",
    )
    main(["codex-lm", str(script)])
    captured = capsys.readouterr()
    assert "intercept" not in captured.err


def test_main_missing_args_returns_usage(restore_lm, capsys):
    assert main(["codex-lm"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "usage: codex-lm" in captured.out
    assert "codex-lm rotation" in captured.out
    assert "Supported Codex models:" not in captured.out


def test_smoke_test_help_is_clean(capsys):
    assert main(["codex-lm", "smoke-test", "--help"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == (
        "usage: codex-lm smoke-test [PROFILE] [--model MODEL] [--prompt TEXT]"
    )
    assert captured.err == ""


def test_main_usage_command_prints_redacted_summary(
    fake_home: Path,
    monkeypatch,
    capsys,
):
    def fake_fetch():
        return {
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

    monkeypatch.setenv("CODEX_LM_ENABLE_LEGACY_AUTH_FALLBACK", "1")
    monkeypatch.setattr("dspy_codex_lm.cli.fetch_codex_usage", fake_fetch)

    assert main(["codex-lm", "usage"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert lines[0] == "-" * 60
    assert lines[-1] == "-" * 60
    assert lines[1:-1] == [
        "Codex usage",
        "rate_limit.primary: 3/4 remaining (75.0% remaining); resets 2026-05-10T12:00:00Z",
    ]
    assert "secret-token" not in captured.out
    assert "acct-secret" not in captured.out
    assert "person@example.com" not in captured.out


def test_main_usage_command_fetches_all_saved_profiles(
    fake_home: Path,
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    first = _write_auth(
        tmp_path / "first.json",
        access_token="first-token",
        account_id="acct-first-secret",
    )
    second = _write_auth(
        tmp_path / "second.json",
        access_token="second-token",
        account_id="acct-second-secret",
    )
    assert main(["codex-lm", "auth", "import", "work", "--from", str(first)]) == 0
    assert main(["codex-lm", "auth", "import", "personal", "--from", str(second)]) == 0
    capsys.readouterr()

    calls = []

    def fake_fetch(*, auth_path=None):
        calls.append(Path(auth_path))
        return {
            "rate_limit": {
                "primary_window": {
                    "used_percent": 25 if "work" in str(auth_path) else 50,
                    "limit_window_seconds": 18000,
                    "reset_after_seconds": 60,
                }
            },
            "access_token": "secret-token",
            "account_id": "acct-secret",
            "user": {"email": "person@example.com"},
        }

    monkeypatch.setattr("dspy_codex_lm.cli.fetch_codex_usage", fake_fetch)

    assert main(["codex-lm", "usage"]) == 0
    captured = capsys.readouterr()

    assert calls == [
        fake_home / ".codex-lm" / "auth" / "personal" / "auth.json",
        fake_home / ".codex-lm" / "auth" / "work" / "auth.json",
    ]
    assert "personal:" in captured.out
    assert "work (default):" in captured.out
    assert captured.out.count("5h limit:") == 2
    assert "secret-token" not in captured.out
    assert "acct-secret" not in captured.out
    assert "person@example.com" not in captured.out


def test_main_usage_skips_live_fetch_for_disabled_profiles(
    fake_home: Path,
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    first = _write_auth(
        tmp_path / "first.json",
        access_token="first-token",
        account_id="acct-first-secret",
    )
    second = _write_auth(
        tmp_path / "second.json",
        access_token="second-token",
        account_id="acct-second-secret",
    )
    assert main(["codex-lm", "auth", "import", "work", "--from", str(first)]) == 0
    assert main(["codex-lm", "auth", "import", "personal", "--from", str(second)]) == 0
    assert main(["codex-lm", "auth", "disable", "personal"]) == 0
    capsys.readouterr()

    calls = []

    def fake_fetch(*, auth_path=None):
        calls.append(Path(auth_path))
        return {
            "rate_limit": {
                "primary_window": {
                    "used_percent": 25,
                    "limit_window_seconds": 18000,
                    "reset_after_seconds": 60,
                }
            },
            "access_token": "secret-token",
            "account_id": "acct-secret",
        }

    monkeypatch.setattr("dspy_codex_lm.cli.fetch_codex_usage", fake_fetch)

    assert main(["codex-lm", "usage"]) == 0
    captured = capsys.readouterr()

    assert calls == [fake_home / ".codex-lm" / "auth" / "work" / "auth.json"]
    assert "personal (disabled):" in captured.out
    assert "  Disabled; live usage fetch skipped." in captured.out
    assert "work (default):" in captured.out
    assert "secret-token" not in captured.out
    assert "acct-secret" not in captured.out


def test_main_usage_shows_rotation_on_for_saved_profiles(
    fake_home: Path,
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    first = _write_auth(tmp_path / "first.json")
    second = _write_auth(tmp_path / "second.json")
    assert main(["codex-lm", "auth", "import", "work", "--from", str(first)]) == 0
    assert main(["codex-lm", "auth", "import", "personal", "--from", str(second)]) == 0
    assert main(["codex-lm", "auth", "use", "work"]) == 0
    assert main(["codex-lm", "rotation", "on"]) == 0
    capsys.readouterr()
    rotation_state = fake_home / ".codex-lm" / "rotation.json"
    state_before = rotation_state.read_text(encoding="utf-8")

    def fake_fetch(*, auth_path=None):
        return {
            "rate_limit": {
                "primary_window": {
                    "used_percent": 25 if "work" in str(auth_path) else 50,
                    "limit_window_seconds": 18000,
                    "reset_after_seconds": 60,
                }
            },
            "access_token": "secret-token",
            "account_id": "acct-secret",
            "user": {"email": "person@example.com"},
        }

    monkeypatch.setattr("dspy_codex_lm.cli.fetch_codex_usage", fake_fetch)

    assert main(["codex-lm", "usage"]) == 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()

    assert lines[0] == "-" * 60
    assert lines[1:3] == ["Rotation: on (round robin)", ""]
    assert "personal:" in captured.out
    assert "work (default):" in captured.out
    assert captured.out.count("5h limit:") == 2
    assert rotation_state.read_text(encoding="utf-8") == state_before
    assert "secret-token" not in captured.out
    assert "acct-secret" not in captured.out
    assert "person@example.com" not in captured.out


def test_main_usage_color_can_be_forced_and_disabled(
    fake_home: Path,
    monkeypatch,
    capsys,
):
    def fake_fetch():
        return {
            "rate_limit": {
                "primary_window": {
                    "used_percent": 12,
                    "limit_window_seconds": 18000,
                    "reset_after_seconds": 300,
                }
            }
        }

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr("dspy_codex_lm.cli.fetch_codex_usage", fake_fetch)

    assert main(["codex-lm", "--color=always", "usage"]) == 0
    captured = capsys.readouterr()
    assert "\x1b[" in captured.out
    assert "General usage limits:" in captured.out

    assert main(["codex-lm", "--no-color", "usage"]) == 0
    captured = capsys.readouterr()
    assert "\x1b[" not in captured.out
    assert "General usage limits:" in captured.out


def test_main_usage_respects_no_color_env(fake_home: Path, monkeypatch, capsys):
    def fake_fetch():
        return {
            "rate_limit": {
                "primary_window": {
                    "used_percent": 12,
                    "limit_window_seconds": 18000,
                    "reset_after_seconds": 300,
                }
            }
        }

    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr("dspy_codex_lm.cli.fetch_codex_usage", fake_fetch)

    assert main(["codex-lm", "--color=always", "usage"]) == 0
    captured = capsys.readouterr()
    assert "\x1b[" not in captured.out
    assert "General usage limits:" in captured.out


def test_auth_import_list_use_status_remove(fake_home: Path, tmp_path: Path, capsys):
    source = _write_auth(
        tmp_path / "auth.json",
        access_token="profile-token",
        account_id="acct-profile-secret",
    )

    assert main(["codex-lm", "auth", "import", "work", "--from", str(source)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "work" in captured.out

    assert main(["codex-lm", "auth", "list"]) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["* work"]

    assert main(["codex-lm", "auth", "use", "work"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "Default auth profile: work"

    assert main(["codex-lm", "auth", "list"]) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["* work"]

    assert main(["codex-lm", "auth", "status"]) == 0
    captured = capsys.readouterr()
    assert "Active profile: work" in captured.out
    assert "Selected profile: work (active profile)" in captured.out
    assert "Access token: present" in captured.out
    assert "Refresh token: present" in captured.out
    assert "acct-p...cret" in captured.out
    assert "profile-token" not in captured.out
    assert "acct-profile-secret" not in captured.out
    assert "person@example.com" not in captured.out

    assert main(["codex-lm", "auth", "remove", "work"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "Removed auth profile: work"

    assert main(["codex-lm", "auth", "list"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "No auth profiles saved."


def test_auth_disable_enable_and_list_marker(fake_home: Path, tmp_path: Path, capsys):
    source = _write_auth(
        tmp_path / "auth.json",
        access_token="profile-token",
        account_id="acct-profile-secret",
    )

    assert main(["codex-lm", "auth", "import", "work", "--from", str(source)]) == 0
    capsys.readouterr()

    assert main(["codex-lm", "auth", "disable", "work"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "Disabled auth profile: work"
    assert "profile-token" not in captured.out

    assert main(["codex-lm", "auth", "list"]) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["* work (disabled)"]

    assert main(["codex-lm", "auth", "enable", "work"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "Enabled auth profile: work"

    assert main(["codex-lm", "auth", "list"]) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["* work"]


def test_auth_disable_missing_name_prints_usage(fake_home: Path, capsys):
    assert main(["codex-lm", "auth", "disable"]) == 2
    captured = capsys.readouterr()

    assert "auth disable requires exactly one profile name" in captured.err
    assert "usage: codex-lm auth disable NAME" in captured.err


def test_auth_use_and_default_alias_switch_load_codex_auth(
    fake_home: Path,
    tmp_path: Path,
    capsys,
):
    work = _write_auth(
        tmp_path / "work.json",
        access_token="work-token",
        account_id="acct-work",
    )
    personal = _write_auth(
        tmp_path / "personal.json",
        access_token="personal-token",
        account_id="acct-personal",
    )

    assert main(["codex-lm", "auth", "import", "work", "--from", str(work)]) == 0
    assert main(["codex-lm", "auth", "import", "personal", "--from", str(personal)]) == 0
    capsys.readouterr()

    assert main(["codex-lm", "auth", "use", "work"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "Default auth profile: work"
    assert load_codex_auth() == ("work-token", "acct-work")

    assert main(["codex-lm", "auth", "default", "personal"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "Default auth profile: personal"
    assert load_codex_auth() == ("personal-token", "acct-personal")


def test_rotation_on_off_status(
    fake_home: Path,
    tmp_path: Path,
    capsys,
):
    first = _write_auth(tmp_path / "first.json")
    assert main(["codex-lm", "auth", "import", "work", "--from", str(first)]) == 0
    capsys.readouterr()

    assert main(["codex-lm", "rotation"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.strip() == "Rotation: off"

    assert main(["codex-lm", "rotation", "on"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.strip() == "Rotation: on (round robin)"

    assert main(["codex-lm", "rotation", "status"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.strip() == "Rotation: on (round robin)"

    assert main(["codex-lm", "rotation", "off"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.strip() == "Rotation: off"


def test_rotation_on_without_profiles_errors(
    fake_home: Path,
    capsys,
):
    assert main(["codex-lm", "rotation", "on"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no auth profiles saved" in captured.err
    assert "secret-token" not in captured.err
    assert "person@example.com" not in captured.err


def test_auth_list_shows_rotation_on_and_keeps_default_marker(
    fake_home: Path,
    tmp_path: Path,
    capsys,
):
    first = _write_auth(tmp_path / "first.json")
    second = _write_auth(tmp_path / "second.json")
    assert main(["codex-lm", "auth", "import", "work", "--from", str(first)]) == 0
    assert main(["codex-lm", "auth", "import", "personal", "--from", str(second)]) == 0
    assert main(["codex-lm", "auth", "use", "work"]) == 0
    capsys.readouterr()

    assert main(["codex-lm", "auth", "list"]) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["  personal", "* work"]
    assert "Rotation:" not in captured.out

    assert main(["codex-lm", "rotation", "on"]) == 0
    capsys.readouterr()
    rotation_state = fake_home / ".codex-lm" / "rotation.json"
    state_before = rotation_state.read_text(encoding="utf-8")

    assert main(["codex-lm", "auth", "list"]) == 0
    captured = capsys.readouterr()

    assert captured.out.splitlines() == [
        "Rotation: on (round robin)",
        "",
        "  personal",
        "* work",
    ]
    assert rotation_state.read_text(encoding="utf-8") == state_before


def test_auth_list_and_status_color_can_be_forced(
    fake_home: Path,
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    source = _write_auth(tmp_path / "auth.json")

    assert main(["codex-lm", "auth", "import", "work", "--from", str(source)]) == 0
    assert main(["codex-lm", "auth", "use", "work"]) == 0
    capsys.readouterr()

    assert main(["codex-lm", "--color=always", "auth", "list"]) == 0
    captured = capsys.readouterr()
    assert "\x1b[" in captured.out
    assert "* " in captured.out
    assert "work" in captured.out

    assert main(["codex-lm", "--color=always", "auth", "status"]) == 0
    captured = capsys.readouterr()
    assert "\x1b[" in captured.out
    assert "Codex auth status" in captured.out
    assert "secret-token" not in captured.out


def test_auth_status_shows_env_override(
    fake_home: Path,
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    source = _write_auth(tmp_path / "auth.json", account_id="acct-env-secret")
    assert main(["codex-lm", "auth", "import", "envprofile", "--from", str(source)]) == 0
    capsys.readouterr()
    monkeypatch.setenv(CODEX_LM_AUTH_PROFILE_ENV, "envprofile")

    assert main(["codex-lm", "auth", "status"]) == 0
    captured = capsys.readouterr()

    assert "Selected profile: envprofile (CODEX_LM_AUTH_PROFILE)" in captured.out
    assert "acct-e...cret" in captured.out
    assert "secret-token" not in captured.out
    assert "person@example.com" not in captured.out


def test_auth_login_uses_isolated_codex_home(fake_home: Path, monkeypatch, capsys):
    seen = {}

    def fake_run(cmd, *, env):
        seen["cmd"] = cmd
        seen["codex_home"] = env["CODEX_HOME"]
        _write_auth(
            Path(env["CODEX_HOME"]) / "auth.json",
            access_token="login-token",
            account_id="acct-login-secret",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("dspy_codex_lm.cli.subprocess.run", fake_run)

    assert main(["codex-lm", "auth", "login", "login-profile"]) == 0
    captured = capsys.readouterr()

    assert seen["cmd"] == ["codex", "login", "--device-auth"]
    assert Path(seen["codex_home"]).name.startswith("codex-lm-auth-")
    assert str(fake_home) not in seen["codex_home"]
    assert "login-profile" in captured.out

    assert main(["codex-lm", "auth", "use", "login-profile"]) == 0
    capsys.readouterr()
    assert main(["codex-lm", "auth", "status"]) == 0
    captured = capsys.readouterr()
    assert "acct-l...cret" in captured.out
    assert "login-token" not in captured.out


def test_auth_login_missing_name_prints_usage(fake_home: Path, capsys):
    assert main(["codex-lm", "auth", "login"]) == 2
    captured = capsys.readouterr()

    assert "auth login requires a profile name" in captured.err
    assert "usage: codex-lm auth login NAME [--device-auth]" in captured.err


def test_auth_login_accepts_display_name_and_preserves_it(
    fake_home: Path,
    monkeypatch,
    capsys,
):
    seen = {}

    def fake_run(cmd, *, env):
        seen["cmd"] = cmd
        _write_auth(
            Path(env["CODEX_HOME"]) / "auth.json",
            access_token="login-token",
            account_id="acct-login-secret",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("dspy_codex_lm.cli.subprocess.run", fake_run)

    display_name = "gabriel@trampoline.ai"
    slug = "gabriel-trampoline.ai"

    assert main(["codex-lm", "auth", "login", display_name]) == 0
    captured = capsys.readouterr()

    assert seen["cmd"] == ["codex", "login", "--device-auth"]
    assert display_name in captured.out
    profile_dir = fake_home / ".codex-lm" / "auth" / slug
    assert (profile_dir / "auth.json").is_file()
    assert (profile_dir / "profile.json").read_text(encoding="utf-8") == (
        '{\n  "name": "gabriel@trampoline.ai",\n  "slug": "gabriel-trampoline.ai"\n}\n'
    )

    assert main(["codex-lm", "auth", "list"]) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [f"* {display_name}"]

    assert main(["codex-lm", "auth", "use", display_name]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == f"Default auth profile: {display_name}"

    assert main(["codex-lm", "auth", "status"]) == 0
    captured = capsys.readouterr()
    assert f"Active profile: {display_name}" in captured.out
    assert f"Selected profile: {display_name} (active profile)" in captured.out
    assert "login-token" not in captured.out
    assert "acct-login-secret" not in captured.out

    assert main(["codex-lm", "auth", "remove", display_name]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == f"Removed auth profile: {display_name}"
    assert not profile_dir.exists()


def test_auth_login_returns_codex_exit_code(fake_home: Path, monkeypatch, capsys):
    def fake_run(cmd, *, env):
        return SimpleNamespace(returncode=17)

    monkeypatch.setattr("dspy_codex_lm.cli.subprocess.run", fake_run)

    assert main(["codex-lm", "auth", "login", "work", "--device-auth"]) == 17
    captured = capsys.readouterr()
    assert captured.out == ""


def test_auth_login_sets_default_when_none(fake_home: Path, monkeypatch, capsys):
    def fake_run(cmd, *, env):
        _write_auth(
            Path(env["CODEX_HOME"]) / "auth.json",
            access_token="login-token",
            account_id="acct-login-secret",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("dspy_codex_lm.cli.subprocess.run", fake_run)

    assert main(["codex-lm", "auth", "login", "first-login"]) == 0
    capsys.readouterr()

    assert load_codex_auth() == ("login-token", "acct-login-secret")


def test_smoke_test_checks_each_saved_profile(
    fake_home: Path,
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    first = _write_auth(
        tmp_path / "first.json",
        access_token="first-token",
        account_id="acct-first-secret",
    )
    second = _write_auth(
        tmp_path / "second.json",
        access_token="second-token",
        account_id="acct-second-secret",
    )
    assert main(["codex-lm", "auth", "import", "work", "--from", str(first)]) == 0
    assert main(["codex-lm", "auth", "import", "personal", "--from", str(second)]) == 0
    capsys.readouterr()

    calls = []

    class FakeCodexLM:
        def __init__(self, *, model, auth_profile=None):
            calls.append(("init", model, auth_profile))

        def forward(self, *, prompt, cache):
            calls.append(("forward", prompt, cache))
            return SimpleNamespace()

    monkeypatch.setattr("dspy_codex_lm.cli.CodexLM", FakeCodexLM)

    assert main(["codex-lm", "smoke-test", "--model", "gpt-5.4-mini"]) == 0
    captured = capsys.readouterr()

    assert calls == [
        ("init", "gpt-5.4-mini", "personal"),
        ("forward", "Reply with OK.", False),
        ("init", "gpt-5.4-mini", "work"),
        ("forward", "Reply with OK.", False),
    ]
    assert captured.err == ""
    assert "personal: ok" in captured.out
    assert "work: ok" in captured.out


def test_smoke_test_checks_only_requested_profile(
    fake_home: Path,
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    first = _write_auth(tmp_path / "first.json")
    second = _write_auth(tmp_path / "second.json")
    assert main(["codex-lm", "auth", "import", "work", "--from", str(first)]) == 0
    assert main(["codex-lm", "auth", "import", "personal", "--from", str(second)]) == 0
    capsys.readouterr()

    calls = []

    class FakeCodexLM:
        def __init__(self, *, model, auth_profile=None):
            calls.append(("init", model, auth_profile))

        def forward(self, *, prompt, cache):
            calls.append(("forward", prompt, cache))
            return SimpleNamespace()

    monkeypatch.setattr("dspy_codex_lm.cli.CodexLM", FakeCodexLM)

    assert main(["codex-lm", "smoke-test", "work", "--prompt", "ping"]) == 0
    captured = capsys.readouterr()

    assert calls == [
        ("init", "gpt-5.3-codex", "work"),
        ("forward", "ping", False),
    ]
    assert "work: ok" in captured.out
    assert "personal" not in captured.out


def test_auth_rejects_invalid_profile_name(fake_home: Path, capsys):
    assert main(["codex-lm", "auth", "use", "../bad"]) == 2
    captured = capsys.readouterr()
    assert "invalid auth profile name" in captured.err
    assert "usage: codex-lm auth use NAME" in captured.err


@pytest.mark.parametrize(
    ("argv", "usage"),
    [
        (
            ["codex-lm", "auth", "import", "../bad"],
            "usage: codex-lm auth import NAME [--from PATH]",
        ),
        (
            ["codex-lm", "auth", "login", "../bad"],
            "usage: codex-lm auth login NAME [--device-auth]",
        ),
        (
            ["codex-lm", "auth", "remove", "../bad"],
            "usage: codex-lm auth remove NAME",
        ),
    ],
)
def test_auth_invalid_profile_input_prints_command_usage(
    fake_home: Path,
    monkeypatch,
    capsys,
    argv,
    usage,
):
    run = mock.Mock()
    monkeypatch.setattr("dspy_codex_lm.cli.subprocess.run", run)

    assert main(argv) == 2
    captured = capsys.readouterr()

    assert "invalid auth profile name" in captured.err
    assert usage in captured.err
    run.assert_not_called()
