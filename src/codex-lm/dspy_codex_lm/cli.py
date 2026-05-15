"""CLI wrapper: ``codex-lm my_script.py [args]``.

Monkeypatches ``dspy.LM`` in the child script's namespace so that any LM
construction targeting an OpenAI-family model is routed through
:class:`CodexLM`, using the Codex CLI's OAuth token.

Non-OpenAI models (anthropic/, google/, cohere/, etc.) are left alone.

Routing is **strict**: an OpenAI-family model is only routed if its slug
matches one of the known Codex models. Anything else (``gpt-4o``, ``o3``,
``gpt-5-mini``, …) raises :class:`CodexLMUnsupportedModelError`.

Silence the intercept banner with ``CODEX_LM_QUIET=1``.
"""

from __future__ import annotations

import logging
import os
import re
import runpy
import subprocess
import sys
import tempfile
from typing import Iterable, Literal

import dspy

from dspy_codex_lm.auth import (
    CODEX_LM_AUTH_PROFILE_ENV,
    CODEX_LM_ENABLE_LEGACY_AUTH_FALLBACK_ENV,
    auth_profile_display_name,
    auth_rotation_status,
    auth_status_metadata,
    enable_auth_profile,
    get_active_profile,
    import_auth_profile,
    list_auth_profile_states,
    list_auth_profiles,
    profile_auth_path,
    remove_auth_profile,
    set_active_profile,
    set_auth_rotation_enabled,
)
from dspy_codex_lm.lm import CodexLM
from dspy_codex_lm.usage import (
    fetch_codex_usage,
    format_disabled_profile_usage_entry,
    format_profile_usage_summaries,
    format_usage_summary,
)

logger = logging.getLogger(__name__)

OPENAI_MODEL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^openai/", re.IGNORECASE),
    re.compile(r"^gpt-\d", re.IGNORECASE),
    re.compile(r"^o[1-9](?:-|$)", re.IGNORECASE),
    re.compile(r"^chatgpt-", re.IGNORECASE),
]


# Models the Codex backend actually knows about (per opencode's allowlist in
# packages/opencode/src/plugin/codex.ts). Not all are entitled to every
# ChatGPT plan — the backend will 400 on models your plan doesn't cover.
CODEX_SUPPORTED_MODELS: frozenset[str] = frozenset(
    {
        "gpt-5.1-codex",
        "gpt-5.1-codex-max",
        "gpt-5.1-codex-mini",
        "gpt-5.2",
        "gpt-5.2-codex",
        "gpt-5.3-codex",
        "gpt-5.3-codex-spark",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.5",
    }
)

# Kwargs that CodexLM owns — drop them if passed to the intercepted LM.
DROPPED_KWARGS = ("api_key", "api_base", "base_url", "model_type", "provider")

AUTH_COMMAND_USAGE = {
    "list": "usage: codex-lm auth list",
    "status": "usage: codex-lm auth status",
    "use": "usage: codex-lm auth use NAME",
    "default": "usage: codex-lm auth default NAME",
    "import": "usage: codex-lm auth import NAME [--from PATH]",
    "login": "usage: codex-lm auth login NAME [--device-auth]",
    "disable": "usage: codex-lm auth disable NAME",
    "enable": "usage: codex-lm auth enable NAME",
    "remove": "usage: codex-lm auth remove NAME",
}
SMOKE_TEST_USAGE = "usage: codex-lm smoke-test [PROFILE] [--model MODEL] [--prompt TEXT]"

ColorMode = Literal["auto", "always", "never"]
_ANSI_RESET = "\x1b[0m"
_ANSI_BOLD_CYAN = "\x1b[1;36m"


class CodexLMUnsupportedModelError(RuntimeError):
    """Raised when a script requests an OpenAI-family model that has no
    direct Codex equivalent and no ``CODEX_LM_MODEL`` override is set."""


def is_openai_family(model: str) -> bool:
    return any(p.search(model) for p in OPENAI_MODEL_PATTERNS)


def _strip_provider_prefix(model: str) -> str:
    return model.split("/")[-1] if "/" in model else model


def resolve_codex_model(requested: str) -> str:
    """Return the Codex model slug for a requested OpenAI-family slug.

    The requested slug (stripped of any provider prefix) must appear in the
    Codex allowlist. Otherwise, raise :class:`CodexLMUnsupportedModelError`
    with a helpful message listing the supported Codex models.
    """
    slug = _strip_provider_prefix(requested)
    if slug in CODEX_SUPPORTED_MODELS:
        return slug
    supported = ", ".join(sorted(CODEX_SUPPORTED_MODELS))
    raise CodexLMUnsupportedModelError(
        f"codex-lm: cannot route {requested!r} to Codex — no corresponding "
        f"Codex model.\n\n"
        f"Supported Codex models: {supported}\n\n"
        f"Update the script to use one of the supported Codex model slugs."
    )


def install_monkeypatch(
    verbose: bool = True,
    patterns: Iterable[re.Pattern[str]] | None = None,
    exclude: Iterable[str] = (),
) -> tuple[type, type | None]:
    """Replace ``dspy.LM`` with a subclass that reroutes OpenAI models to CodexLM.

    Routing is strict: OpenAI-family models without a direct Codex equivalent
    raise :class:`CodexLMUnsupportedModelError` at instantiation time.

    Args:
        verbose: print a banner on each intercept.
        patterns: override the OpenAI family match patterns (for tests).
        exclude: model substrings to skip (case-insensitive). Any model
            whose name contains one of these strings will NOT be
            intercepted, even if it matches OPENAI_MODEL_PATTERNS.
            Pass e.g. ``["mercury"]`` to let ``openai/mercury-2`` through
            to LiteLLM unmodified.

    Returns the original (``dspy.LM``, ``dspy.clients.lm.LM``) pair so tests
    can restore them.
    """
    original_top = dspy.LM
    try:
        import dspy.clients.lm as dspy_lm_module

        original_inner: type | None = dspy_lm_module.LM
    except Exception:
        dspy_lm_module = None
        original_inner = None

    match_patterns = list(patterns) if patterns is not None else OPENAI_MODEL_PATTERNS
    exclude_lower = [s.lower() for s in exclude]

    def _matches(model: str) -> bool:
        if exclude_lower and any(ex in model.lower() for ex in exclude_lower):
            return False
        return any(p.search(model) for p in match_patterns)

    # Sentinel so we can distinguish "caller didn't pass model" (e.g.
    # ``copy.deepcopy`` which calls ``cls.__new__(cls)`` with no args)
    # from "caller explicitly passed something". Without this, deepcopy
    # would hit the default and get intercepted as a phantom gpt-4o-mini
    # construction — fatal when gpt-4o-mini isn't in the Codex allowlist.
    _model_unset = object()

    class InterceptedLM(original_top):  # type: ignore[misc,valid-type]
        def __new__(cls, model=_model_unset, *args, **kwargs):
            if model is _model_unset:
                # Called without an explicit model — almost certainly the
                # ``copy.deepcopy`` path, which creates a blank instance
                # via ``cls.__new__(cls)`` and then copies ``__dict__``.
                # Do NOT intercept; just allocate a blank instance and
                # let deepcopy populate it.
                return super().__new__(cls)
            if isinstance(model, str) and _matches(model):
                codex_model = resolve_codex_model(model)
                logger.info("intercept dspy.LM(%r) -> CodexLM(%r)", model, codex_model)
                if verbose:
                    print(
                        f"[codex-lm] intercept dspy.LM({model!r}) -> CodexLM({codex_model!r})",
                        file=sys.stderr,
                    )
                for k in DROPPED_KWARGS:
                    kwargs.pop(k, None)
                instance = CodexLM(model=codex_model, **kwargs)
                instance.codex_intercepted_from = model  # type: ignore[attr-defined]
                return instance
            return super().__new__(cls)

    InterceptedLM.__name__ = original_top.__name__
    InterceptedLM.__qualname__ = original_top.__qualname__

    dspy.LM = InterceptedLM  # type: ignore[misc,assignment]
    if dspy_lm_module is not None:
        dspy_lm_module.LM = InterceptedLM  # type: ignore[assignment]
    return original_top, original_inner


def restore_monkeypatch(original_top: type, original_inner: type | None) -> None:
    dspy.LM = original_top  # type: ignore[assignment]
    if original_inner is not None:
        try:
            import dspy.clients.lm as dspy_lm_module

            dspy_lm_module.LM = original_inner  # type: ignore[assignment]
        except Exception:
            pass


def _print_auth_help(file=None) -> None:
    file = file or sys.stderr
    print(
        "usage: codex-lm auth <command> [args]\n"
        "\n"
        "Commands:\n"
        "  list                         list saved auth profiles\n"
        "  status                       show selected auth source and redacted "
        "metadata\n"
        "  use NAME                     select a saved profile as the default\n"
        "  default NAME                 alias for `auth use NAME`\n"
        "  import NAME [--from PATH]    import an existing Codex auth.json\n"
        "  login NAME [--device-auth]   run Codex CLI device login for a profile\n"
        "  disable NAME                 mark a profile disabled without deleting it\n"
        "  enable NAME                  re-enable a disabled profile\n"
        "  remove NAME                  delete a saved profile\n"
        "\n"
        f"env:\n"
        f"  {CODEX_LM_AUTH_PROFILE_ENV}=NAME  use a profile without changing "
        "active state\n"
        f"  {CODEX_LM_ENABLE_LEGACY_AUTH_FALLBACK_ENV}=1  allow legacy "
        "~/.codex/auth.json fallback\n",
        file=file,
    )


def _print_main_help(*, file=None, full: bool = True) -> None:
    file = file or sys.stderr
    supported = ", ".join(sorted(CODEX_SUPPORTED_MODELS))
    if not full:
        print(
            "usage: codex-lm <script.py> [args...]\n"
            "       codex-lm usage\n"
            "       codex-lm smoke-test [PROFILE] [--model MODEL] [--prompt TEXT]\n"
            "       codex-lm rotation [on|off|status]\n"
            "       codex-lm auth <command> [args]\n"
            "\n"
            "Commands:\n"
            "  usage       show redacted usage for saved auth profiles\n"
            "  smoke-test  send a tiny request for saved auth profiles\n"
            "  rotation    round-robin ordinary requests across saved profiles\n"
            "  auth        manage saved Codex auth profiles\n"
            "\n"
            "Run `codex-lm --help` for full options.",
            file=file,
        )
        return
    print(
        "usage: codex-lm [--exclude=SUBSTR ...] <script.py> [args...]\n"
        "       codex-lm [--color=auto|always|never] usage\n"
        "       codex-lm smoke-test [PROFILE] [--model MODEL] [--prompt TEXT]\n"
        "       codex-lm rotation [on|off|status]\n"
        "       codex-lm auth <command> [args]\n"
        "\n"
        "Runs the given Python script with dspy.LM monkeypatched so that\n"
        "OpenAI-family LM constructions are routed through CodexLM.\n"
        "Use `codex-lm usage` to print a redacted ChatGPT/Codex usage summary.\n"
        "Use `codex-lm smoke-test` to test saved auth profiles.\n"
        "Use `codex-lm rotation on` to round-robin ordinary requests across "
        "saved profiles.\n"
        "Use `codex-lm auth` to manage Codex auth profiles.\n"
        "\n"
        "Routing is strict: only OpenAI-family models whose slug matches\n"
        "a known Codex model are routed; others raise an error.\n"
        "\n"
        "Options:\n"
        "  --exclude=SUBSTR  skip interception for models containing SUBSTR\n"
        "                    (case-insensitive, repeatable). E.g.\n"
        "                    --exclude=mercury lets openai/mercury-2 through\n"
        "                    to LiteLLM unmodified.\n"
        "  --color=MODE      colorize CLI output: auto, always, or never\n"
        "  --no-color        disable ANSI color output\n"
        "\n"
        f"Supported Codex models: {supported}\n"
        "\n"
        "env:\n"
        "  CODEX_LM_QUIET            if set, suppress intercept banner\n"
        f"  {CODEX_LM_AUTH_PROFILE_ENV}=NAME  use an auth profile override\n"
        f"  {CODEX_LM_ENABLE_LEGACY_AUTH_FALLBACK_ENV}=1  allow legacy "
        "~/.codex/auth.json fallback\n",
        file=file,
    )


def _main_auth(argv: list[str], *, color: bool = False) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        _print_auth_help()
        return 0 if argv and argv[0] in ("-h", "--help") else 2

    command, args = argv[0], argv[1:]
    try:
        if command == "list":
            if args:
                raise ValueError("auth list takes no arguments")
            return _auth_list(color=color)
        if command == "status":
            if args:
                raise ValueError("auth status takes no arguments")
            return _auth_status(color=color)
        if command in {"use", "default"}:
            if len(args) != 1:
                raise ValueError(f"auth {command} requires exactly one profile name")
            display_name = auth_profile_display_name(args[0])
            set_active_profile(args[0])
            print(f"Default auth profile: {display_name}")
            return 0
        if command == "import":
            name, source = _parse_auth_import_args(args)
            dest = import_auth_profile(name, source)
            print(f"Imported auth profile {name!r} -> {dest}")
            return 0
        if command == "login":
            name = _parse_auth_login_args(args)
            return _auth_login(name)
        if command in {"disable", "enable"}:
            if len(args) != 1:
                raise ValueError(f"auth {command} requires exactly one profile name")
            display_name = auth_profile_display_name(args[0])
            enabled = command == "enable"
            enable_auth_profile(args[0], enabled=enabled)
            verb = "Enabled" if enabled else "Disabled"
            print(f"{verb} auth profile: {display_name}")
            return 0
        if command == "remove":
            if len(args) != 1:
                raise ValueError("auth remove requires exactly one profile name")
            display_name = auth_profile_display_name(args[0])
            remove_auth_profile(args[0])
            print(f"Removed auth profile: {display_name}")
            return 0
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        _print_auth_error(command, exc)
        return 2

    print(f"codex-lm auth: unknown command {command!r}", file=sys.stderr)
    _print_auth_help()
    return 2


def _main_rotation(argv: list[str]) -> int:
    command = argv[0] if argv else "status"
    args = argv[1:] if argv else []
    if command in {"-h", "--help"}:
        print("usage: codex-lm rotation [on|off|status]")
        return 0
    if args or command not in {"on", "off", "status"}:
        print(
            "codex-lm rotation: usage: codex-lm rotation [on|off|status]",
            file=sys.stderr,
        )
        return 2
    try:
        if command == "on":
            status = set_auth_rotation_enabled(True)
        elif command == "off":
            status = set_auth_rotation_enabled(False)
        else:
            status = auth_rotation_status()
    except (FileNotFoundError, ValueError) as exc:
        print(f"codex-lm rotation: {exc}", file=sys.stderr)
        return 2
    print(_format_rotation_status(status))
    return 0


def _format_rotation_status(status: dict[str, object]) -> str:
    if not status.get("enabled"):
        return "Rotation: off"
    return "Rotation: on (round robin)"


def _enabled_rotation_display_line(*, color: bool = False) -> str | None:
    status = auth_rotation_status()
    if not status.get("enabled"):
        return None
    return _ansi(_format_rotation_status(status), _ANSI_BOLD_CYAN, color)


def _prepend_framed_summary_line(summary: str, line: str) -> str:
    lines = summary.splitlines()
    if not lines:
        return line
    return "\n".join([lines[0], line, "", *lines[1:]])


def _auth_list(*, color: bool = False) -> int:
    profiles = list_auth_profile_states()
    active = get_active_profile()
    if not profiles:
        print("No auth profiles saved.")
        return 0
    rotation_line = _enabled_rotation_display_line(color=color)
    if rotation_line is not None:
        print(rotation_line)
        print()
    for name, disabled in profiles:
        marker = "*" if name == active else " "
        display = _ansi(name, _ANSI_BOLD_CYAN, color and name == active)
        if disabled:
            display = f"{display} (disabled)"
        print(f"{marker} {display}")
    return 0


def _auth_status(*, color: bool = False) -> int:
    metadata = auth_status_metadata()
    profile = metadata["profile"]
    source = metadata["source"]
    print(_ansi("Codex auth status", _ANSI_BOLD_CYAN, color))
    print(f"Auth path: {metadata['path']}")
    print(f"Auth file: {'present' if metadata['exists'] else 'missing'}")
    print(f"Active profile: {metadata['active_profile'] or 'none'}")
    if profile:
        print(f"Selected profile: {profile} ({_auth_source_label(source)})")
    else:
        print(f"Selected profile: none ({_auth_source_label(source)})")
    if metadata["exists"]:
        print(f"Account ID: {metadata.get('account_id') or 'missing'}")
        print(f"Access token: {metadata.get('access_token', 'missing')}")
        print(f"Refresh token: {metadata.get('refresh_token', 'missing')}")
        print(f"ID token: {metadata.get('id_token', 'missing')}")
    return 0


def _auth_source_label(source: str) -> str:
    if source == "env":
        return CODEX_LM_AUTH_PROFILE_ENV
    if source == "active":
        return "active profile"
    if source == "fallback":
        return f"fallback ~/.codex/auth.json ({CODEX_LM_ENABLE_LEGACY_AUTH_FALLBACK_ENV}=1)"
    return source


def _ansi(value: str, code: str, enabled: bool) -> str:
    if not enabled:
        return value
    return f"{code}{value}{_ANSI_RESET}"


def _print_auth_error(command: str, exc: Exception) -> None:
    message = str(exc)
    print(f"codex-lm auth: {message}", file=sys.stderr)
    usage = AUTH_COMMAND_USAGE.get(command)
    if usage and usage not in message:
        print(usage, file=sys.stderr)


def _parse_auth_import_args(args: list[str]) -> tuple[str, str | None]:
    if not args:
        raise ValueError("auth import requires a profile name")
    name = args[0]
    source = None
    rest = args[1:]
    while rest:
        flag = rest.pop(0)
        if flag != "--from" or not rest:
            raise ValueError("auth import expected --from PATH")
        source = rest.pop(0)
    return name, source


def _parse_auth_login_args(args: list[str]) -> str:
    if not args:
        raise ValueError("auth login requires a profile name")
    name = args[0]
    for flag in args[1:]:
        if flag != "--device-auth":
            raise ValueError(f"auth login unknown option {flag!r}")
    return name


def _auth_login(name: str) -> int:
    profile_auth_path(name)
    with tempfile.TemporaryDirectory(prefix="codex-lm-auth-") as codex_home:
        env = os.environ.copy()
        env["CODEX_HOME"] = codex_home
        result = subprocess.run(["codex", "login", "--device-auth"], env=env)
        if result.returncode != 0:
            return result.returncode
        source = os.path.join(codex_home, "auth.json")
        dest = import_auth_profile(name, source)
    print(f"Imported auth profile {name!r} -> {dest}")
    return 0


def _main_smoke_test(argv: list[str]) -> int:
    if argv and argv[0] in {"-h", "--help"}:
        print(SMOKE_TEST_USAGE)
        return 0
    try:
        profile, model, prompt = _parse_smoke_test_args(argv)
        if profile is None:
            profiles = list_auth_profiles()
            targets = [(item, item) for item in profiles] or [(None, "default auth")]
        else:
            display_name = auth_profile_display_name(profile)
            targets = [(profile, display_name)]
    except (FileNotFoundError, ValueError) as exc:
        print(f"codex-lm smoke-test: {exc}", file=sys.stderr)
        print(SMOKE_TEST_USAGE, file=sys.stderr)
        return 2

    failed = False
    for auth_profile, label in targets:
        try:
            lm = CodexLM(model=model, auth_profile=auth_profile)
            lm.forward(prompt=prompt, cache=False)
        except Exception as exc:
            failed = True
            print(f"{label}: failed: {exc}", file=sys.stderr)
        else:
            print(f"{label}: ok")
    return 1 if failed else 0


def _parse_smoke_test_args(args: list[str]) -> tuple[str | None, str, str]:
    profile = None
    model = "gpt-5.3-codex"
    prompt = "Reply with OK."
    rest = list(args)
    if rest and not rest[0].startswith("-"):
        profile = rest.pop(0)
    while rest:
        flag = rest.pop(0)
        if flag == "--model" and rest:
            model = rest.pop(0)
        elif flag == "--prompt" and rest:
            prompt = rest.pop(0)
        elif flag in {"-h", "--help"}:
            raise ValueError(SMOKE_TEST_USAGE)
        else:
            raise ValueError(f"unknown option {flag!r}")
    return profile, model, prompt


def _parse_color_mode(value: str) -> ColorMode:
    if value == "auto":
        return "auto"
    if value == "always":
        return "always"
    if value == "never":
        return "never"
    raise ValueError("--color must be one of: auto, always, never")


def _should_color(mode: ColorMode, *, stream=None) -> bool:
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR"):
        return False
    if mode == "always":
        return True
    if mode == "never":
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else list(sys.argv)
    # Parse codex-lm's own flags before the script path.
    exclude: list[str] = []
    color_mode: ColorMode = "auto"
    rest = argv[1:]
    while rest:
        if rest[0].startswith("--exclude="):
            exclude.append(rest.pop(0).split("=", 1)[1])
            continue
        if rest[0].startswith("--color="):
            try:
                color_mode = _parse_color_mode(rest.pop(0).split("=", 1)[1])
            except ValueError as exc:
                print(f"codex-lm: {exc}", file=sys.stderr)
                return 2
            continue
        if rest[0] == "--no-color":
            color_mode = "never"
            rest.pop(0)
            continue
        break
    color = _should_color(color_mode)

    if rest and rest[0] == "usage":
        usage_args = rest[1:]
        while usage_args:
            arg = usage_args.pop(0)
            if arg.startswith("--color="):
                try:
                    color_mode = _parse_color_mode(arg.split("=", 1)[1])
                except ValueError as exc:
                    print(f"codex-lm usage: {exc}", file=sys.stderr)
                    return 2
            elif arg == "--no-color":
                color_mode = "never"
            else:
                print(f"codex-lm usage: unknown option {arg!r}", file=sys.stderr)
                return 2
        color = _should_color(color_mode)
        try:
            profiles = list_auth_profiles()
            if profiles:
                items = []
                for profile, disabled in list_auth_profile_states():
                    if disabled:
                        items.append(format_disabled_profile_usage_entry(profile))
                    else:
                        items.append(
                            (
                                profile,
                                fetch_codex_usage(auth_path=profile_auth_path(profile)),
                            )
                        )
                print(
                    _format_profile_usage_cli_output(
                        items,
                        color=color,
                        default_profile=get_active_profile(),
                    )
                )
            else:
                payload = fetch_codex_usage()
                print(format_usage_summary(payload, color=color))
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            print(f"codex-lm usage: {exc}", file=sys.stderr)
            return 2
        return 0

    if rest and rest[0] == "auth":
        return _main_auth(rest[1:], color=color)

    if rest and rest[0] == "rotation":
        return _main_rotation(rest[1:])

    if rest and rest[0] == "smoke-test":
        return _main_smoke_test(rest[1:])

    if not rest:
        _print_main_help(file=sys.stdout, full=False)
        return 0

    if rest[0] in ("-h", "--help"):
        _print_main_help()
        return 0

    script = rest[0]
    if not os.path.isfile(script):
        print(f"codex-lm: script not found: {script}", file=sys.stderr)
        return 2
    verbose = not os.environ.get("CODEX_LM_QUIET")
    install_monkeypatch(verbose=verbose, exclude=exclude)
    script_dir = os.path.dirname(os.path.abspath(script))
    sys.argv = [script, *rest[1:]]
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0
    except CodexLMUnsupportedModelError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    return 0


def _format_profile_usage_cli_output(
    items: list[tuple[str, object]],
    *,
    color: bool,
    default_profile: str | None,
) -> str:
    summary = format_profile_usage_summaries(
        items,
        color=color,
        default_profile=default_profile,
    )
    rotation_line = _enabled_rotation_display_line(color=color)
    if rotation_line is None:
        return summary
    return _prepend_framed_summary_line(summary, rotation_line)


if __name__ == "__main__":
    sys.exit(main())
