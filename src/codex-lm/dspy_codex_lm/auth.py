from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

CODEX_LM_AUTH_PROFILE_ENV = "CODEX_LM_AUTH_PROFILE"
CODEX_LM_ENABLE_LEGACY_AUTH_FALLBACK_ENV = "CODEX_LM_ENABLE_LEGACY_AUTH_FALLBACK"
PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
PROFILE_SLUG_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
PROFILE_METADATA = "profile.json"


@dataclass(frozen=True)
class AuthSource:
    path: Path
    profile: str | None
    source: Literal["path", "profile", "env", "rotation", "active", "fallback"]


@dataclass(frozen=True)
class AuthProfile:
    name: str
    slug: str
    disabled: bool = False

    @property
    def auth_path(self) -> Path:
        return auth_profiles_dir() / self.slug / "auth.json"

    @property
    def metadata_path(self) -> Path:
        return auth_profiles_dir() / self.slug / PROFILE_METADATA


def codex_auth_path() -> Path:
    return Path.home() / ".codex" / "auth.json"


def codex_lm_home() -> Path:
    return Path.home() / ".codex-lm"


def auth_profiles_dir() -> Path:
    return codex_lm_home() / "auth"


def active_profile_path() -> Path:
    return codex_lm_home() / "active_profile"


def rotation_state_path() -> Path:
    return codex_lm_home() / "rotation.json"


def rotation_lock_path() -> Path:
    return codex_lm_home() / "rotation.lock"


def validate_profile_name(name: str) -> str:
    if not name or name in {".", ".."} or not PROFILE_NAME_RE.fullmatch(name):
        raise ValueError(
            "invalid auth profile name; use only letters, numbers, dots, "
            "underscores, and dashes"
        )
    return name


def validate_profile_display_name(name: str) -> str:
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or any(ord(char) < 32 for char in name)
    ):
        raise ValueError(
            "invalid auth profile name; use a non-empty name without path "
            "separators or control characters"
        )
    return name


def slugify_profile_name(name: str) -> str:
    validate_profile_display_name(name)
    slug = PROFILE_SLUG_SAFE_RE.sub("-", name).strip(".-_")
    if not slug:
        raise ValueError("invalid auth profile name; could not derive a filesystem-safe slug")
    return validate_profile_name(slug)


def profile_auth_path(name: str) -> Path:
    return _profile_for_path(name).auth_path


def auth_profile_display_name(name: str) -> str:
    return _resolve_profile_reference(name).name


def list_auth_profiles() -> list[str]:
    return [profile.name for profile in _list_auth_profile_objects()]


def list_enabled_auth_profiles() -> list[str]:
    return [profile.name for profile in _list_enabled_auth_profile_objects()]


def list_auth_profile_states() -> list[tuple[str, bool]]:
    return [(profile.name, profile.disabled) for profile in _list_auth_profile_objects()]


def is_auth_profile_disabled(name: str) -> bool:
    return _resolve_profile_reference(name).disabled


def enable_auth_profile(name: str, *, enabled: bool) -> None:
    profile = _resolve_profile_reference(name)
    _write_profile_metadata(
        AuthProfile(profile.name, profile.slug, disabled=not enabled),
        include_disabled=True,
    )


def _list_auth_profile_objects() -> list[AuthProfile]:
    root = auth_profiles_dir()
    if not root.exists():
        return []
    profiles = []
    for child in root.iterdir():
        if child.is_dir() and (child / "auth.json").is_file():
            try:
                profile = _profile_from_slug(child.name)
            except (TypeError, ValueError):
                continue
            if profile is not None:
                profiles.append(profile)
    return sorted(profiles, key=lambda profile: profile.name.casefold())


def _list_enabled_auth_profile_objects() -> list[AuthProfile]:
    return [profile for profile in _list_auth_profile_objects() if not profile.disabled]


def is_auth_rotation_enabled() -> bool:
    return bool(_read_rotation_state().get("enabled", False))


def auth_rotation_status() -> dict[str, Any]:
    state = _read_rotation_state()
    profiles = _list_auth_profile_objects()
    enabled_profiles = [profile for profile in profiles if not profile.disabled]
    return {
        "enabled": bool(state.get("enabled", False)),
        "profile_count": len(enabled_profiles),
        "total_profile_count": len(profiles),
    }


def set_auth_rotation_enabled(enabled: bool) -> dict[str, Any]:
    with _rotation_lock():
        profiles = _list_auth_profile_objects()
        if enabled and not profiles:
            raise FileNotFoundError(
                "no auth profiles saved; run `codex-lm auth login NAME` "
                "or `codex-lm auth import NAME` before enabling rotation"
            )
        enabled_profiles = [profile for profile in profiles if not profile.disabled]
        if enabled and not enabled_profiles:
            raise ValueError(
                "no enabled auth profiles available; run "
                "`codex-lm auth enable NAME` before enabling rotation"
            )
        state = _read_rotation_state()
        cursor = _coerce_cursor(state.get("cursor", 0))
        next_state = {
            "enabled": enabled,
            "cursor": cursor % len(enabled_profiles) if enabled_profiles else 0,
        }
        _write_rotation_state(next_state)
        return {
            "enabled": enabled,
            "profile_count": len(enabled_profiles),
            "total_profile_count": len(profiles),
        }


def next_rotation_auth_source() -> AuthSource | None:
    return _rotation_auth_source(advance=True)


def peek_rotation_auth_source() -> AuthSource | None:
    return _rotation_auth_source(advance=False)


def _rotation_auth_source(*, advance: bool) -> AuthSource | None:
    with _rotation_lock():
        state = _read_rotation_state()
        if not state.get("enabled", False):
            return None
        profiles = _list_enabled_auth_profile_objects()
        if not profiles:
            raise ValueError(
                "auth profile rotation is enabled but no enabled auth profiles "
                "are available; run `codex-lm auth enable NAME` or "
                "`codex-lm rotation off`"
            )
        index = _coerce_cursor(state.get("cursor", 0)) % len(profiles)
        profile = profiles[index]
        if advance:
            _write_rotation_state(
                {
                    "enabled": True,
                    "cursor": (index + 1) % len(profiles),
                }
            )
        return AuthSource(profile.auth_path, profile.name, "rotation")


def _get_active_profile_slug() -> str | None:
    marker = active_profile_path()
    if not marker.exists():
        return None
    slug = marker.read_text(encoding="utf-8").strip()
    if not slug:
        return None
    return validate_profile_name(slug)


def get_active_profile() -> str | None:
    slug = _get_active_profile_slug()
    if slug is None:
        return None
    profile = _profile_from_slug(slug)
    return profile.name if profile is not None else slug


def set_active_profile(name: str) -> None:
    profile = _resolve_profile_reference(name)
    marker = active_profile_path()
    _ensure_private_dir(marker.parent)
    marker.write_text(f"{profile.slug}\n", encoding="utf-8")
    _chmod_private_file(marker)


def clear_active_profile() -> None:
    marker = active_profile_path()
    if marker.exists():
        marker.unlink()


def import_auth_profile(name: str, source_path: str | Path | None = None) -> Path:
    profile = _profile_for_import(name)
    source = Path(source_path) if source_path is not None else codex_auth_path()
    data = _read_auth_json(source)
    _tokens_from_json(data)
    dest = profile.auth_path
    _write_auth_json(dest, data)
    _write_profile_metadata(profile)
    if _get_active_profile_slug() is None:
        set_active_profile(profile.name)
    return dest


def remove_auth_profile(name: str) -> None:
    profile = _resolve_profile_reference(name)
    root = auth_profiles_dir() / profile.slug
    shutil.rmtree(root)
    if _get_active_profile_slug() == profile.slug:
        clear_active_profile()


def resolve_auth_source(
    path: str | Path | None = None,
    profile: str | None = None,
    *,
    advance_rotation: bool = True,
) -> AuthSource:
    if path is not None:
        return AuthSource(Path(path), None, "path")
    if profile is not None:
        resolved = _resolve_profile_reference(profile)
        if resolved.disabled:
            _raise_disabled_profile(resolved)
        return AuthSource(resolved.auth_path, resolved.name, "profile")
    env_profile = os.environ.get(CODEX_LM_AUTH_PROFILE_ENV)
    if env_profile:
        resolved = _resolve_profile_reference(env_profile)
        if resolved.disabled:
            _raise_disabled_profile(resolved)
        return AuthSource(resolved.auth_path, resolved.name, "env")
    rotation = next_rotation_auth_source() if advance_rotation else peek_rotation_auth_source()
    if rotation is not None:
        return rotation
    active_slug = _get_active_profile_slug()
    if active_slug:
        active = _profile_from_slug(active_slug)
        name = active.name if active is not None else active_slug
        if active is not None and active.disabled:
            _raise_disabled_profile(active, prefix="active ")
        return AuthSource(
            auth_profiles_dir() / active_slug / "auth.json",
            name,
            "active",
        )
    if os.environ.get(CODEX_LM_ENABLE_LEGACY_AUTH_FALLBACK_ENV) == "1":
        return AuthSource(codex_auth_path(), None, "fallback")
    raise FileNotFoundError(
        "no codex-lm auth profile selected; run "
        "`codex-lm auth login NAME`, `codex-lm auth import NAME`, or "
        "`codex-lm auth use NAME`, or set "
        f"{CODEX_LM_ENABLE_LEGACY_AUTH_FALLBACK_ENV}=1 to use legacy "
        "~/.codex/auth.json fallback"
    )


def load_codex_auth(
    path: str | Path | None = None,
    *,
    profile: str | None = None,
) -> tuple[str, str]:
    """Read the access token and ChatGPT account id from Codex auth.

    Resolution order, unless ``path`` or ``profile`` is explicit:
    ``CODEX_LM_AUTH_PROFILE``, saved-profile rotation, and the codex-lm
    active profile marker. The Codex CLI's legacy ``~/.codex/auth.json`` file
    is only used when ``CODEX_LM_ENABLE_LEGACY_AUTH_FALLBACK=1`` is set.

    Returns:
        (access_token, account_id)
    """
    source = resolve_auth_source(path, profile)
    return _tokens_from_json(_read_auth_json(source.path))


def auth_status_metadata(
    path: str | Path | None = None,
    *,
    profile: str | None = None,
) -> dict[str, Any]:
    source = resolve_auth_source(path, profile, advance_rotation=False)
    exists = source.path.is_file()
    metadata: dict[str, Any] = {
        "path": str(source.path),
        "profile": source.profile,
        "source": source.source,
        "exists": exists,
        "active_profile": get_active_profile(),
    }
    if not exists:
        return metadata

    data = _read_auth_json(source.path)
    tokens = data.get("tokens") if isinstance(data, dict) else None
    if isinstance(tokens, dict):
        metadata["access_token"] = _presence(tokens.get("access_token"))
        metadata["refresh_token"] = _presence(tokens.get("refresh_token"))
        metadata["id_token"] = _presence(tokens.get("id_token"))
        account_id = tokens.get("account_id")
        metadata["account_id"] = _redact_account_id(account_id)
    return metadata


def _profile_for_path(name: str) -> AuthProfile:
    if _is_valid_slug(name):
        profile = _profile_from_slug(name)
        if profile is not None:
            return profile

    display_name = validate_profile_display_name(name)
    slug = slugify_profile_name(display_name)
    profile = _profile_from_slug(slug)
    if profile is not None and profile.name != display_name:
        _raise_profile_slug_collision()
    return AuthProfile(display_name, slug)


def _profile_for_import(name: str) -> AuthProfile:
    if _is_valid_slug(name):
        profile = _profile_from_slug(name)
        if profile is not None:
            return profile

    display_name = validate_profile_display_name(name)
    slug = slugify_profile_name(display_name)
    profile = _profile_from_slug(slug)
    if profile is not None:
        if profile.name != display_name:
            _raise_profile_slug_collision()
        return profile
    return AuthProfile(display_name, slug)


def _resolve_profile_reference(name: str) -> AuthProfile:
    if _is_valid_slug(name):
        profile = _profile_from_slug(name)
        if profile is not None:
            return profile

    display_name = validate_profile_display_name(name)
    slug = slugify_profile_name(display_name)
    profile = _profile_from_slug(slug)
    if profile is None:
        raise FileNotFoundError(f"auth profile {display_name!r} does not exist")
    if profile.name != display_name:
        _raise_profile_slug_collision()
    return profile


def _profile_from_slug(slug: str) -> AuthProfile | None:
    validate_profile_name(slug)
    root = auth_profiles_dir() / slug
    if not (root / "auth.json").is_file():
        return None

    metadata = root / PROFILE_METADATA
    if not metadata.is_file():
        return AuthProfile(slug, slug)

    data = _read_json_object(metadata)
    name = data.get("name")
    stored_slug = data.get("slug")
    disabled = data.get("disabled", False)
    if not isinstance(name, str) or not isinstance(stored_slug, str):
        raise TypeError(f"profile metadata {metadata} must contain name and slug")
    if stored_slug != slug:
        raise ValueError(f"profile metadata {metadata} has mismatched slug")
    return AuthProfile(
        validate_profile_display_name(name),
        slug,
        disabled=bool(disabled),
    )


def _is_valid_slug(name: str) -> bool:
    try:
        validate_profile_name(name)
    except ValueError:
        return False
    return True


def _raise_profile_slug_collision() -> None:
    raise ValueError(
        "auth profile name maps to an existing auth profile slug with a different display name"
    )


def _write_profile_metadata(
    profile: AuthProfile,
    *,
    include_disabled: bool = False,
) -> None:
    _ensure_private_dir(profile.metadata_path.parent)
    metadata: dict[str, Any] = {"name": profile.name, "slug": profile.slug}
    if include_disabled or profile.disabled:
        metadata["disabled"] = profile.disabled
    text = json.dumps(
        metadata,
        indent=2,
        sort_keys=True,
    )
    profile.metadata_path.write_text(f"{text}\n", encoding="utf-8")
    _chmod_private_file(profile.metadata_path)


def _read_auth_json(path: Path) -> dict[str, Any]:
    return _read_json_object(path)


def _raise_disabled_profile(profile: AuthProfile, *, prefix: str = "") -> None:
    raise ValueError(
        f"{prefix}auth profile {profile.name!r} is disabled; run "
        f"`codex-lm auth enable {profile.name}` to re-enable it"
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"auth file {path} must contain a JSON object")
    return value


def _tokens_from_json(data: dict[str, Any]) -> tuple[str, str]:
    tokens = data["tokens"]
    return tokens["access_token"], tokens["account_id"]


def _write_auth_json(path: Path, data: dict[str, Any]) -> None:
    _ensure_private_dir(path.parent)
    text = json.dumps(data, indent=2, sort_keys=True)
    path.write_text(f"{text}\n", encoding="utf-8")
    _chmod_private_file(path)


def _read_rotation_state() -> dict[str, Any]:
    path = rotation_state_path()
    if not path.is_file():
        return {"enabled": False, "cursor": 0}
    try:
        data = _read_json_object(path)
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return {"enabled": False, "cursor": 0}
    return {
        "enabled": bool(data.get("enabled", False)),
        "cursor": _coerce_cursor(data.get("cursor", 0)),
    }


def _write_rotation_state(state: dict[str, Any]) -> None:
    path = rotation_state_path()
    _ensure_private_dir(path.parent)
    text = json.dumps(
        {
            "enabled": bool(state.get("enabled", False)),
            "cursor": _coerce_cursor(state.get("cursor", 0)),
        },
        indent=2,
        sort_keys=True,
    )
    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".rotation.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_file = Path(handle.name)
            handle.write(f"{text}\n")
            handle.flush()
            os.fsync(handle.fileno())
        _chmod_private_file(tmp_file)
        os.replace(tmp_file, path)
    finally:
        if tmp_file is not None and tmp_file.exists():
            tmp_file.unlink()
    _chmod_private_file(path)


def _coerce_cursor(value: Any) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


@contextmanager
def _rotation_lock() -> Iterator[None]:
    path = rotation_lock_path()
    _ensure_private_dir(path.parent)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        try:
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
    _chmod_private_file(path)


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass


def _chmod_private_file(path: Path) -> None:
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _presence(value: Any) -> str:
    return "present" if isinstance(value, str) and value else "missing"


def _redact_account_id(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if len(value) <= 8:
        return f"{value[:2]}...{value[-2:]}"
    return f"{value[:6]}...{value[-4:]}"
