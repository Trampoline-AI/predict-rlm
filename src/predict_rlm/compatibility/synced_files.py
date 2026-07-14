"""Portable SyncedFile tool operation."""

from __future__ import annotations

import functools
import inspect
import os
import shutil
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from predict_rlm.files import get_synced_file_params
from predict_rlm.runtime import (
    Artifact,
    CallableTool,
    RuntimeTool,
    ToolOperation,
    current_run_context,
    invoke_host_callable,
    preserve_sync_leaf,
)


class SyncedFileToolOperation(ToolOperation):
    name = "synced-files"

    def apply(self, tool: RuntimeTool) -> RuntimeTool:
        function = tool.function if isinstance(tool, CallableTool) else tool
        synced_params = get_synced_file_params(function)
        if not synced_params:
            return tool

        signature = inspect.signature(function)

        @functools.wraps(function)
        async def dispatch(*args: Any, **kwargs: Any) -> Any:
            ctx = current_run_context()
            if ctx is None or ctx.session is None:
                raise RuntimeError("SyncedFile tool calls require an active execution session")
            bound = signature.bind_partial(*args, **kwargs)
            temporary_root: str | None = None
            synced: list[tuple[str, str, bool]] = []
            try:
                for parameter, annotation in synced_params.items():
                    sandbox_path = bound.arguments.get(parameter)
                    if not isinstance(sandbox_path, str) or not sandbox_path:
                        continue
                    if annotation.host_dir is None:
                        if temporary_root is None:
                            temporary_root = tempfile.mkdtemp(prefix="tool-file-sync-")
                        host_dir = temporary_root
                    else:
                        host_dir = annotation.host_dir
                        Path(host_dir).mkdir(parents=True, exist_ok=True)
                    host_path = os.path.join(host_dir, os.path.basename(sandbox_path))
                    artifact = Artifact(
                        id=f"synced-file-{uuid.uuid4().hex}",
                        kind="compat.file",
                        metadata={
                            "sandbox_path": sandbox_path,
                            "destination_path": host_path,
                        },
                    )
                    await ctx.session.collect(artifact)
                    bound.arguments[parameter] = host_path
                    synced.append((sandbox_path, host_path, annotation.writeback))

                result = await invoke_host_callable(
                    function,
                    *bound.args,
                    **bound.kwargs,
                )
                for sandbox_path, host_path, writeback in synced:
                    if not writeback or not os.path.isfile(host_path):
                        continue
                    await ctx.session.mount(
                        Artifact(
                            id=f"synced-file-writeback-{uuid.uuid4().hex}",
                            kind="compat.file",
                            metadata={
                                "source_path": host_path,
                                "sandbox_path": sandbox_path,
                            },
                        )
                    )
                return result
            except BaseException as exc:
                worker = getattr(exc, "sync_worker", None)
                if temporary_root is not None and worker is not None and not worker.done:
                    deferred_root = temporary_root
                    worker.add_done_callback(
                        lambda _worker: shutil.rmtree(deferred_root, ignore_errors=True)
                    )
                    temporary_root = None
                raise
            finally:
                if temporary_root is not None:
                    shutil.rmtree(temporary_root, ignore_errors=True)

        dispatch.__predict_rlm_synced_file_operation__ = True
        preserve_sync_leaf(dispatch, function)

        host_dirs = tuple(
            dict.fromkeys(
                annotation.host_dir
                for annotation in synced_params.values()
                if annotation.host_dir is not None
            )
        )
        if isinstance(tool, CallableTool):
            return replace(
                tool,
                function=dispatch,
                extra_read_paths=(*tool.extra_read_paths, *host_dirs),
                extra_write_paths=(*tool.extra_write_paths, *host_dirs),
            )
        return CallableTool(
            name=tool.name,
            function=dispatch,
            description=tool.description,
            schema=tool.schema,
            extra_read_paths=host_dirs,
            extra_write_paths=host_dirs,
        )
