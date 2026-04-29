from __future__ import annotations

import logging

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only when optional dep is missing
    tqdm = None

RLM_LOGGER_NAME = "dspy.predict.rlm"


def progress_write(message: str) -> None:
    writer = getattr(tqdm, "write", None) if tqdm is not None else None
    if writer is not None:
        writer(message)
        return
    print(message)


class ProgressLogHandler(logging.Handler):
    def __init__(self, prefix: str):
        super().__init__(level=logging.INFO)
        self.prefix = prefix

    def emit(self, record: logging.LogRecord) -> None:
        try:
            progress_write(f"[{self.prefix}] {self.format(record)}")
        except Exception:
            self.handleError(record)


def install_rlm_log_stream(prefix: str) -> tuple[logging.Logger, int, bool, logging.Handler]:
    logger = logging.getLogger(RLM_LOGGER_NAME)
    old_level = logger.level
    old_propagate = logger.propagate
    handler = ProgressLogHandler(prefix)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    return logger, old_level, old_propagate, handler


def restore_rlm_log_stream(state: tuple[logging.Logger, int, bool, logging.Handler]) -> None:
    logger, old_level, old_propagate, handler = state
    logger.removeHandler(handler)
    logger.setLevel(old_level)
    logger.propagate = old_propagate
