import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import dspy
import pytest
from dspy_codex_lm.auth import CODEX_LM_AUTH_PROFILE_ENV, load_codex_auth
from dspy_codex_lm.cli import (
    main,
)


@pytest.fixture
def restore_lm(monkeypatch):
    import dspy.clients.lm as lm_module

    monkeypatch.setattr(dspy, "LM", dspy.LM)
    monkeypatch.setattr(lm_module, "LM", lm_module.LM)
    monkeypatch.setattr(sys, "argv", sys.argv.copy())
    monkeypatch.setattr(sys, "path", sys.path.copy())


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


@pytest.fixture(autouse=True)
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolate_auth_home) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv(CODEX_LM_AUTH_PROFILE_ENV, raising=False)
    monkeypatch.delenv("CODEX_LM_ENABLE_LEGACY_AUTH_FALLBACK", raising=False)
    return home


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


def test_main_exits_3_on_unsupported_model(tmp_path, restore_lm, capsys):
    script = _write_script(
        tmp_path,
        "import dspy; dspy.LM(model='openai/gpt-4o', api_key='sk-fake')",
    )
    rc = main(["codex-lm", str(script)])
    captured = capsys.readouterr()
    assert rc == 3
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
    assert "work (default):" in captured.out
    assert "secret-token" not in captured.out
    assert "acct-secret" not in captured.out


def test_auth_login_uses_isolated_codex_home(fake_home: Path, monkeypatch, capsys):
    seen = {}
    existing = _write_auth(fake_home / ".codex" / "auth.json")
    before = existing.read_bytes()

    def fake_run(cmd, *, env):
        seen["codex_home"] = env["CODEX_HOME"]
        _write_auth(
            Path(env["CODEX_HOME"]) / "auth.json",
            access_token="login-token",
            account_id="acct-login-secret",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("dspy_codex_lm.cli.subprocess.run", fake_run)

    assert main(["codex-lm", "auth", "login", "login-profile"]) == 0
    capsys.readouterr()
    assert existing.read_bytes() == before
    assert not Path(seen["codex_home"]).exists()
    assert load_codex_auth(profile="login-profile") == ("login-token", "acct-login-secret")

    assert main(["codex-lm", "auth", "use", "login-profile"]) == 0
    capsys.readouterr()
    assert main(["codex-lm", "auth", "status"]) == 0
    captured = capsys.readouterr()
    assert "login-token" not in captured.out
    assert "acct-login-secret" not in captured.out
    assert "secret-refresh" not in captured.out


def test_auth_login_returns_codex_exit_code(fake_home: Path, monkeypatch, capsys):
    def fake_run(cmd, *, env):
        return SimpleNamespace(returncode=17)

    monkeypatch.setattr("dspy_codex_lm.cli.subprocess.run", fake_run)

    assert main(["codex-lm", "auth", "login", "work", "--device-auth"]) == 17
    captured = capsys.readouterr()
    assert captured.out == ""
    assert not (fake_home / ".codex-lm" / "auth" / "work").exists()
