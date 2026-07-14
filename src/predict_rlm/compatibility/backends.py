"""Legacy constructor options compiled into final execution backends."""

from __future__ import annotations

from typing import Any, Callable

from predict_rlm.backends import BackendName
from predict_rlm.backends.adapters import InterpreterBackendAdapter
from predict_rlm.backends.jspi import JspiExecutionBackend
from predict_rlm.runtime import ExecutionBackend, SessionOwnership


def execution_from_legacy_options(
    *,
    owner: Any,
    interpreter: Any,
    sandbox_backend: BackendName,
    sbx_config: Any,
    sbx_pool: Any,
    allowed_domains: list[str] | None,
    runtime_hooks: list[Any],
    on_runtime_hook_event: Callable[..., Any] | None,
) -> ExecutionBackend:
    if interpreter is not None:
        return InterpreterBackendAdapter(
            sandbox_backend.value,
            owner._acquire_runtime_interpreter,
            ownership=SessionOwnership.INJECTED,
            supports_host_directory_mounts=callable(
                getattr(interpreter, "configure_direct_workspace_mounts", None)
            ),
        )

    pool_kwargs = getattr(sbx_pool, "_interpreter_kwargs", None)
    legacy_pool_transport = sbx_pool is not None and (
        not isinstance(pool_kwargs, dict)
        or pool_kwargs.get("_supervisor_command") is not None
    )
    if legacy_pool_transport:
        return InterpreterBackendAdapter(
            sandbox_backend.value,
            owner._acquire_runtime_interpreter,
            ownership=SessionOwnership.POOLED,
        )

    if sbx_pool is not None:
        from predict_rlm.backends.sbx import SbxPoolExecutionBackend

        return SbxPoolExecutionBackend(
            sbx_pool,
            runtime_hooks=runtime_hooks,
            on_runtime_hook_event=on_runtime_hook_event,
        )
    if sandbox_backend is BackendName.SBX:
        from predict_rlm.backends.sbx import SbxExecutionBackend

        return SbxExecutionBackend(
            config=sbx_config,
            runtime_hooks=runtime_hooks,
            on_runtime_hook_event=on_runtime_hook_event,
        )
    return JspiExecutionBackend(
        allowed_domains=allowed_domains,
        telemetry_context=lambda: owner._current_telemetry_context,
    )
