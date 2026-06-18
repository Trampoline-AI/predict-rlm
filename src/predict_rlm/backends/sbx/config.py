"""Docker Sandboxes backend configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

DEFAULT_SBX_TEMPLATE = "docker.io/docker/sandbox-templates:shell"


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
    reuse: bool = False
    stop_on_shutdown: bool = False
    extra_workspaces: list[str] = Field(default_factory=list)
    workspace_read_only: bool = False
    create_timeout: float = 120.0
    exec_timeout: float = 300.0
    websocket_port: int = 0
    websocket_startup_timeout: float = 30.0
    websocket_max_message_bytes: int = 32 * 1024 * 1024

    @model_validator(mode="after")
    def _apply_reuse_semantics(self) -> "SbxConfig":
        if self.reuse:
            if not self.name:
                raise ValueError("SbxConfig.reuse=True requires a non-empty `name`.")
            self.persist = True
            self.remove_on_shutdown = False
        return self
