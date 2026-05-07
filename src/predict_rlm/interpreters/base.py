"""Shared interpreter backend types."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field

DEFAULT_SBX_TEMPLATE = "docker.io/docker/sandbox-templates:shell"


class SandboxBackend(str, Enum):
    """Named sandbox backends supported by PredictRLM."""

    JSPI = "jspi"
    SBX = "sbx"


class SbxConfig(BaseModel):
    """Configuration for the Docker Sandboxes backend."""

    name: str | None = None
    cpus: int | None = None
    memory: str | None = None
    template: str | None = DEFAULT_SBX_TEMPLATE
    kit: str | None = None
    branch: str | None = None
    persist: bool = False
    remove_on_shutdown: bool = True
    extra_workspaces: list[str] = Field(default_factory=list)
    workspace_read_only: bool = False
    create_timeout: float = 120.0
    exec_timeout: float = 300.0


class PredictRLMInterpreter(Protocol):
    """Runtime methods PredictRLM needs from a sandbox interpreter."""

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any: ...

    async def aexecute(
        self, code: str, variables: dict[str, Any] | None = None
    ) -> Any: ...

    def mount_file_at(self, host_path: str, virtual_path: str) -> None: ...

    def mkdir_p(self, virtual_path: str) -> None: ...

    def list_dir(self, virtual_path: str) -> list[str]: ...

    def sync_file_to(self, virtual_path: str, host_path: str) -> None: ...

    def shutdown(self) -> None: ...
