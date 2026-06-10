from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .tbench_agent import (
    DAYTONA_REMOTE_RESULT_SENTINEL,
    _build_lm,
    _build_submit_confirmation,
    _coerce_answer,
    _coerce_submit_confirmation_mode,
    _install_codex_lm_monkeypatch,
    _predict_rlm_class,
    _signature_with_task_instruction,
    _with_terminal_bench_skill,
    _write_trace,
)


def _read_payload(arg: str) -> dict[str, Any]:
    path = Path(arg)
    if path.exists():
        raw = path.read_text(encoding="utf-8")
    else:
        raw = arg
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("remote controller payload must be a JSON object")
    return payload


def _set_debug_environment(payload: dict[str, Any]) -> None:
    if payload.get("codex_lm_debug"):
        os.environ["CODEX_LM_DEBUG"] = "1"
    codex_lm_debug_log = payload.get("codex_lm_debug_log")
    if codex_lm_debug_log:
        os.environ["CODEX_LM_DEBUG_LOG"] = str(codex_lm_debug_log)
    if payload.get("predict_rlm_debug"):
        os.environ["PREDICT_RLM_DEBUG"] = "1"
    if payload.get("predict_rlm_debug_json"):
        os.environ["PREDICT_RLM_DEBUG_JSON"] = "1"
    predict_rlm_debug_log = payload.get("predict_rlm_debug_log")
    if predict_rlm_debug_log:
        os.environ["PREDICT_RLM_DEBUG_LOG"] = str(predict_rlm_debug_log)


def _install_verbose_rlm_log_stream(payload: dict[str, Any]) -> tuple[logging.Logger, int, bool, logging.Handler] | None:
    if not dict(payload.get("predict_rlm_kwargs") or {}).get("verbose"):
        return None
    logger = logging.getLogger("predict_rlm.trace")
    old_level = logger.level
    old_propagate = logger.propagate
    log_path = os.environ.get("PREDICT_RLM_DEBUG_LOG")
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(log_path, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    return logger, old_level, old_propagate, handler


def _restore_verbose_rlm_log_stream(state: tuple[logging.Logger, int, bool, logging.Handler] | None) -> None:
    if state is None:
        return
    logger, old_level, old_propagate, handler = state
    logger.removeHandler(handler)
    handler.close()
    logger.setLevel(old_level)
    logger.propagate = old_propagate



def _local_process_interpreter_class() -> Any:
    from predict_rlm.interpreters import DirectProcessRunnerClientAdapter

    return DirectProcessRunnerClientAdapter


def _logging_dir(payload: dict[str, Any]) -> Path | None:
    logging_dir = payload.get("logging_dir")
    return Path(logging_dir) if logging_dir else None


def _write_run_status(logging_dir: Path | None, status: str, **fields: Any) -> None:
    if logging_dir is None:
        return
    logging_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        **fields,
    }
    (logging_dir / "predict_rlm_status.json").write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )


async def _run_predict_rlm_async(payload: dict[str, Any]) -> str:
    _set_debug_environment(payload)
    verbose_log_state = _install_verbose_rlm_log_stream(payload)
    if payload.get("codex_lm"):
        _install_codex_lm_monkeypatch(tuple(payload.get("codex_lm_exclude") or ()))

    interpreter_kwargs = dict(payload.get("interpreter_kwargs") or {})
    interpreter = _local_process_interpreter_class()(**interpreter_kwargs)
    logging_dir = _logging_dir(payload)
    trace_export_path = logging_dir / "predict_rlm_trace.json" if logging_dir else None
    _write_run_status(logging_dir, "running")
    try:
        rlm_kwargs = dict(payload.get("predict_rlm_kwargs") or {})
        if trace_export_path is not None:
            rlm_kwargs.setdefault("trace_export_path", trace_export_path)
        if "lm" in rlm_kwargs:
            rlm_kwargs["lm"] = _build_lm(
                rlm_kwargs["lm"],
                payload.get("lm_reasoning_effort"),
                payload.get("lm_service_tier"),
            )
        if "sub_lm" in rlm_kwargs:
            rlm_kwargs["sub_lm"] = _build_lm(
                rlm_kwargs["sub_lm"],
                payload.get("sub_lm_reasoning_effort"),
                payload.get("sub_lm_service_tier"),
            )
        rlm_kwargs["interpreter"] = interpreter
        _with_terminal_bench_skill(rlm_kwargs, payload.get("skill_instructions"))
        if "max_iterations" in rlm_kwargs:
            rlm_kwargs["max_iterations"] = int(rlm_kwargs["max_iterations"])
        submit_confirmation_mode = _coerce_submit_confirmation_mode(
            payload.get("submit_confirmation_mode")
        )
        submit_confirmation = _build_submit_confirmation(
            submit_confirmation_mode,
            str(payload.get("instruction", "")),
        )
        if submit_confirmation is not None:
            if "submit_confirmation" in rlm_kwargs:
                raise ValueError(
                    "submit_confirmation_mode cannot be combined with a custom "
                    "submit_confirmation callback."
                )
            rlm_kwargs["submit_confirmation"] = submit_confirmation
        signature = _signature_with_task_instruction(
            payload.get("signature", "instruction -> answer"),
            str(payload.get("instruction", "")),
        )
        rlm = _predict_rlm_class()(signature, **rlm_kwargs)
        result = await rlm.acall()
        _write_trace(getattr(result, "trace", None), logging_dir, path=trace_export_path)
        _write_run_status(logging_dir, "completed", has_trace=getattr(result, "trace", None) is not None)
        return _coerce_answer(result)
    except BaseException as exc:
        _write_trace(getattr(exc, "trace", None), logging_dir, path=trace_export_path)
        _write_run_status(
            logging_dir,
            "failed",
            error_type=type(exc).__name__,
            error=str(exc),
            has_trace=getattr(exc, "trace", None) is not None,
        )
        raise
    finally:
        _restore_verbose_rlm_log_stream(verbose_log_state)
        await asyncio.to_thread(interpreter.shutdown)


def _run_predict_rlm(payload: dict[str, Any]) -> str:
    return asyncio.run(_run_predict_rlm_async(payload))


def _print_result(payload: dict[str, Any]) -> None:
    print(
        DAYTONA_REMOTE_RESULT_SENTINEL + json.dumps(payload, sort_keys=True),
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        _print_result(
            {
                "ok": False,
                "error_type": "UsageError",
                "error": "remote controller requires one payload JSON argument",
            }
        )
        return 2
    try:
        payload = _read_payload(args[0])
        answer = _run_predict_rlm(payload)
    except BaseException as exc:
        _print_result(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return 1
    _print_result({"ok": True, "answer": answer})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
