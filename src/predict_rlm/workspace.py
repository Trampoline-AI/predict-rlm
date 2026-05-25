"""Workspace directory mirroring and sync-back for PredictRLM sandboxes."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_WORKSPACE_EXCLUDES = [
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".DS_Store",
]


class WorkspaceMode(str, Enum):
    """How a Workspace is made available inside the sandbox."""

    MIRROR = "mirror"
    DIRECT = "direct"


class Workspace(BaseModel):
    """A mutable host directory mounted into the sandbox for coding-agent RLMs."""

    path: str = Field(description="Host directory to make available in the sandbox.")
    mount_path: str = Field(
        default="/sandbox/workspace",
        description="Sandbox directory where the workspace is mounted.",
    )
    mode: WorkspaceMode = Field(
        default=WorkspaceMode.MIRROR,
        description=(
            "Workspace mounting strategy. 'mirror' copies files into the sandbox "
            "and syncs changes back; 'direct' uses an SBX passthrough mount."
        ),
    )
    sync_back: bool = Field(
        default=True,
        description=(
            "Whether mirror-mode sandbox changes are synced back to the host "
            "after each code block."
        ),
    )
    exclude: list[str] = Field(
        default_factory=lambda: list(DEFAULT_WORKSPACE_EXCLUDES),
        description="Directory or file names to exclude from sandbox mirroring and sync-back.",
    )
    max_file_bytes: int | None = Field(
        default=5_000_000,
        description="Maximum file size to mirror. Larger files are skipped.",
    )


class WorkspaceSyncConflictError(RuntimeError):
    """Raised when host and sandbox both changed a workspace path."""


@dataclass(frozen=True)
class DirectWorkspaceMount:
    host_path: str
    sandbox_path: str


@dataclass
class WorkspaceFileInfo:
    type: str
    sha256: str
    size: int


@dataclass
class WorkspaceSyncConflict:
    path: str
    reason: str


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _workspace_rel_path(root: str, path: str) -> str:
    rel = os.path.relpath(path, root)
    if rel == ".":
        return ""
    return rel.replace(os.sep, "/")


def _validate_workspace_relpath(rel_path: str) -> None:
    path = Path(rel_path)
    if path.is_absolute() or ".." in path.parts or rel_path in ("", "."):
        raise ValueError(f"Invalid workspace relative path: {rel_path!r}")


def _is_excluded_relpath(rel_path: str, excludes: list[str]) -> bool:
    parts = Path(rel_path).parts
    return any(part in excludes for part in parts)


def _host_workspace_manifest(workspace: Workspace) -> dict[str, WorkspaceFileInfo]:
    root = os.path.abspath(workspace.path)
    manifest: dict[str, WorkspaceFileInfo] = {}

    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = _workspace_rel_path(root, dirpath)
        dirnames[:] = [
            d
            for d in sorted(dirnames)
            if not _is_excluded_relpath(
                f"{rel_dir}/{d}" if rel_dir else d,
                workspace.exclude,
            )
            and not os.path.islink(os.path.join(dirpath, d))
        ]
        for filename in sorted(filenames):
            rel_path = f"{rel_dir}/{filename}" if rel_dir else filename
            if _is_excluded_relpath(rel_path, workspace.exclude):
                continue
            host_path = os.path.join(root, *Path(rel_path).parts)
            try:
                file_stat = os.lstat(host_path)
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(file_stat.st_mode):
                continue
            size = file_stat.st_size
            if workspace.max_file_bytes is not None and size > workspace.max_file_bytes:
                continue
            manifest[rel_path] = WorkspaceFileInfo(
                type="file",
                sha256=_sha256_file(host_path),
                size=size,
            )
    return manifest


class WorkspaceSyncState:
    """Tracks host/sandbox manifests and syncs one workspace safely."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.root = os.path.abspath(workspace.path)
        if os.path.islink(self.root):
            raise ValueError(f"Workspace path cannot be a symlink: {workspace.path}")
        self.real_root = os.path.realpath(self.root)
        self.original_host_manifest: dict[str, WorkspaceFileInfo] = {}
        self.last_host_manifest: dict[str, WorkspaceFileInfo] = {}
        self.last_sandbox_manifest: dict[str, WorkspaceFileInfo] = {}

    def iter_mounts(self) -> list[tuple[str, str]]:
        self.original_host_manifest = _host_workspace_manifest(self.workspace)
        self.last_host_manifest = dict(self.original_host_manifest)
        self.last_sandbox_manifest = dict(self.original_host_manifest)
        return [
            (
                os.path.join(self.root, *Path(rel_path).parts),
                f"{self.workspace.mount_path}/{rel_path}",
            )
            for rel_path in sorted(self.original_host_manifest)
        ]

    def host_path(self, rel_path: str) -> str:
        _validate_workspace_relpath(rel_path)
        if _is_excluded_relpath(rel_path, self.workspace.exclude):
            raise ValueError(f"Excluded workspace path cannot be synced: {rel_path!r}")
        path = os.path.abspath(os.path.join(self.root, *Path(rel_path).parts))
        root_prefix = self.root + os.sep
        if path != self.root and not path.startswith(root_prefix):
            raise ValueError(f"Workspace path escapes root: {rel_path!r}")
        real_parent = os.path.realpath(os.path.dirname(path))
        real_root_prefix = self.real_root + os.sep
        if real_parent != self.real_root and not real_parent.startswith(real_root_prefix):
            raise ValueError(f"Workspace path escapes root: {rel_path!r}")
        return path

    def current_host_manifest(self) -> dict[str, WorkspaceFileInfo]:
        return _host_workspace_manifest(self.workspace)

    def _host_path_conflict_reason(
        self,
        rel_path: str,
        host_path: str,
        tracked_host_path: bool,
    ) -> str | None:
        parts = Path(rel_path).parts
        current = self.root
        for part in parts[:-1]:
            current = os.path.join(current, part)
            try:
                file_stat = os.lstat(current)
            except FileNotFoundError:
                break
            if stat.S_ISLNK(file_stat.st_mode):
                return "host parent path is a symlink"
            if not stat.S_ISDIR(file_stat.st_mode):
                return "host parent path is not a directory"

        if os.path.lexists(host_path):
            try:
                file_stat = os.lstat(host_path)
            except FileNotFoundError:
                return None
            if stat.S_ISLNK(file_stat.st_mode):
                return "host path is a symlink"
            if not tracked_host_path:
                return "host path exists but was not mounted"
        return None

    def sync_from_sandbox(self, repl: Any) -> list[WorkspaceSyncConflict]:
        if not self.workspace.sync_back:
            return []

        try:
            raw_sandbox_manifest = repl.workspace_manifest(self.workspace.mount_path)
        except Exception as exc:
            raise WorkspaceSyncConflictError(
                "Workspace sync conflict: workspace mount could not be inspected "
                f"at {self.workspace.mount_path!r}: {exc}"
            ) from exc

        sandbox_manifest = {
            rel: info
            for rel, info in raw_sandbox_manifest.items()
            if not _is_excluded_relpath(rel, self.workspace.exclude)
        }
        host_manifest = self.current_host_manifest()
        conflicts: list[WorkspaceSyncConflict] = []
        writes: list[tuple[str, str, WorkspaceFileInfo]] = []
        deletes: list[str] = []

        candidate_paths = set(self.last_sandbox_manifest) | set(sandbox_manifest)
        for rel_path in sorted(candidate_paths):
            _validate_workspace_relpath(rel_path)
            old_sandbox = self.last_sandbox_manifest.get(rel_path)
            new_sandbox = sandbox_manifest.get(rel_path)
            old_host = self.last_host_manifest.get(rel_path)
            current_host = host_manifest.get(rel_path)

            sandbox_changed = old_sandbox != new_sandbox
            if not sandbox_changed:
                continue

            if (
                new_sandbox is not None
                and self.workspace.max_file_bytes is not None
                and new_sandbox.size > self.workspace.max_file_bytes
            ):
                conflicts.append(
                    WorkspaceSyncConflict(
                        path=rel_path,
                        reason=(
                            "sandbox file exceeds max_file_bytes "
                            f"({new_sandbox.size} > {self.workspace.max_file_bytes})"
                        ),
                    )
                )
                continue

            host_changed = current_host != old_host
            if host_changed:
                conflicts.append(
                    WorkspaceSyncConflict(
                        path=rel_path,
                        reason="host changed concurrently while sandbox also changed",
                    )
                )
                continue

            tracked_host_path = rel_path in self.last_host_manifest or rel_path in host_manifest
            try:
                host_path = self.host_path(rel_path)
                conflict_reason = self._host_path_conflict_reason(
                    rel_path,
                    host_path,
                    tracked_host_path,
                )
            except ValueError as e:
                conflict_reason = str(e)
            if conflict_reason:
                conflicts.append(WorkspaceSyncConflict(path=rel_path, reason=conflict_reason))
                continue

            if new_sandbox is None:
                deletes.append(host_path)
                continue

            writes.append((rel_path, host_path, new_sandbox))

        if conflicts:
            details = ", ".join(f"{c.path} ({c.reason})" for c in conflicts)
            raise WorkspaceSyncConflictError(f"Workspace sync conflict: {details}")

        for host_path in deletes:
            if os.path.lexists(host_path):
                os.remove(host_path)

        for rel_path, host_path, new_sandbox in writes:
            os.makedirs(os.path.dirname(host_path), exist_ok=True)
            repl.sync_file_to(f"{self.workspace.mount_path}/{rel_path}", host_path)
            host_manifest[rel_path] = new_sandbox

        self.last_sandbox_manifest = dict(sandbox_manifest)
        self.last_host_manifest = self.current_host_manifest()
        return []
