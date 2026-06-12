"""Docker Sandboxes execution backend package."""

from .config import DEFAULT_SBX_TEMPLATE, SbxConfig

__all__ = [
    "DEFAULT_SBX_TEMPLATE",
    "SbxBackend",
    "SbxConfig",
    "SbxPool",
]


def __getattr__(name: str):
    if name == "SbxBackend":
        from .backend import SbxBackend

        return SbxBackend
    if name == "SbxPool":
        from .pool import SbxPool

        return SbxPool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
