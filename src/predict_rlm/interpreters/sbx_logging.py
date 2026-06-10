"""Logging helpers for SBX client adapter backends."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from predict_rlm._logging import format_log_fields

logger = logging.getLogger("predict_rlm.interpreters.sbx")


def log_interpreter_lifecycle(
    *,
    enabled: bool,
    event: str,
    sandbox_name: str | None,
    process_pid: int | None,
    staging_root: str | Path | None,
    **fields: Any,
) -> None:
    if not enabled:
        return
    logger.debug(
        "%s%s",
        event,
        format_log_fields(
            {
                "backend": "sbx",
                "sandbox_name": sandbox_name,
                "process_pid": process_pid,
                "staging_root": str(staging_root) if staging_root else None,
                **fields,
            }
        ),
    )


def log_partial_output(
    *,
    enabled: bool,
    output: str,
    sandbox_name: str | None,
    process_pid: int | None,
    staging_root: str | Path | None,
    **fields: Any,
) -> None:
    if not enabled or not output:
        return
    logger.debug(
        "sandbox.partial_output%s\n%s",
        format_log_fields(
            {
                "backend": "sbx",
                "sandbox_name": sandbox_name,
                "process_pid": process_pid,
                "staging_root": str(staging_root) if staging_root else None,
                "chars": len(output),
                **fields,
            }
        ),
        output.rstrip(),
    )


def log_pool_lifecycle(
    *,
    enabled: bool,
    event: str,
    pool_name: str,
    pool_size: int,
    **fields: Any,
) -> None:
    if not enabled:
        return
    logger.debug(
        "%s%s",
        event,
        format_log_fields(
            {
                "backend": "sbx",
                "pool": pool_name,
                "pool_size": pool_size,
                **fields,
            }
        ),
    )
