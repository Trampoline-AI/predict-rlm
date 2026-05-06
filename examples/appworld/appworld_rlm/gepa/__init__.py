from .config import APPWORLD_SPEC, AppWorldGepaConfig, default_config
from .project import AppWorldGepaProject, build_project


def main() -> int:
    from .cli import main as run_cli

    return run_cli()


__all__ = [
    "APPWORLD_SPEC",
    "AppWorldGepaConfig",
    "AppWorldGepaProject",
    "build_project",
    "default_config",
    "main",
]
