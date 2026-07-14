"""Docker Sandboxes execution backend package."""

from .config import DEFAULT_SBX_TEMPLATE, SbxConfig

__all__ = [
    "DEFAULT_SBX_TEMPLATE",
    "SbxBackend",
    "SbxConfig",
    "SbxPool",
    "SbxExecutionBackend",
    "SbxPoolExecutionBackend",
]


def __getattr__(name: str):
    if name == "SbxBackend":
        from .backend import SbxBackend

        return SbxBackend
    if name == "SbxPool":
        from .pool import SbxPool

        return SbxPool
    if name in {"SbxExecutionBackend", "SbxPoolExecutionBackend"}:
        from .execution import SbxExecutionBackend, SbxPoolExecutionBackend

        return {
            "SbxExecutionBackend": SbxExecutionBackend,
            "SbxPoolExecutionBackend": SbxPoolExecutionBackend,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
