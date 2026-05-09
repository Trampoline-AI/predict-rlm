from .service import AppWorldRLM
from .signature import SolveAppWorldTask
from .skills import get_appworld_skill


def __getattr__(name: str):
    if name == "appworld_skill":
        return get_appworld_skill()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AppWorldRLM", "SolveAppWorldTask", "get_appworld_skill"]
