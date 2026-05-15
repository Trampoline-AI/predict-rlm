from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


class AppWorldRunnerError(RuntimeError):
    pass


_COMPLETE_TASK_FEEDBACK = "Use SUBMIT(answer=value) or SUBMIT() to finish the task."
_DROP_COMPLETE_TASK = object()


class AppWorldSessionClient:
    def __init__(
        self,
        python: str | None = None,
        data_root: str | Path | None = None,
        experiment_name: str = "predict_rlm",
    ):
        self.python = python or os.environ.get("APPWORLD_PYTHON") or _default_appworld_python()
        self.data_root = Path(data_root or os.environ.get("APPWORLD_DATA_ROOT", "data"))
        self.experiment_name = experiment_name
        self.worker = Path(__file__).with_name("appworld_worker.py")
        self._proc: subprocess.Popen[str] | None = None
        self._started_tasks: set[str] = set()
        atexit.register(self.close)

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        self._started_tasks.clear()
        if proc is None:
            return
        if proc.stdin is not None and proc.poll() is None:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    def __del__(self) -> None:
        self.close()

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        proc = self._ensure_process()
        if proc.stdin is None or proc.stdout is None:
            raise AppWorldRunnerError("AppWorld JSONL worker pipes are unavailable")
        proc.stdin.write(json.dumps(payload, sort_keys=True) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            self.close()
            raise AppWorldRunnerError("AppWorld JSONL worker exited without a response")
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise AppWorldRunnerError(f"AppWorld JSONL worker returned non-JSON: {line[:500]}") from exc

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        self._proc = subprocess.Popen(
            [self.python, str(self.worker), "--jsonl"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return self._proc

    def _ensure_task(self, task_id: str) -> None:
        if task_id in self._started_tasks:
            return
        response = self.request(
            {
                "op": "start_task",
                "task_id": task_id,
                "session_id": task_id,
                "experiment_name": self.experiment_name,
                "data_root": str(self.data_root),
            }
        )
        if not response.get("success"):
            raise AppWorldRunnerError(str(response.get("feedback") or response))
        self._started_tasks.add(task_id)

    def _tool_response(self, task_id: str, op: str, **kwargs: Any) -> dict[str, Any]:
        self._ensure_task(task_id)
        return self.request({"op": op, "task_id": task_id, "session_id": task_id, **kwargs})

    def _tool_request(self, task_id: str, op: str, **kwargs: Any) -> str:
        return _tool_text(self._tool_response(task_id, op, **kwargs))

    def list_appworld_apps(self, task_id: str) -> str:
        """List AppWorld apps and short app descriptions for a task."""
        return self._tool_request(task_id, "list_apps")

    def show_appworld_api_descriptions(self, task_id: str, app_name: str) -> str:
        """Show available AppWorld APIs for one app."""
        response = self._tool_response(task_id, "show_api_descriptions", app_name=app_name)
        if _is_supervisor_app(app_name):
            response = _filter_model_facing_completion_docs(response, in_supervisor=True)
        return _tool_text(response)

    def show_appworld_api_doc(self, task_id: str, app_name: str, api_name: str) -> str:
        """Show detailed AppWorld API documentation for one app API."""
        if _is_completion_api(app_name, api_name):
            return _blocked_completion_tool_text(task_id)
        return self._tool_request(task_id, "show_api_doc", app_name=app_name, api_name=api_name)

    def search_appworld_api_docs(self, task_id: str, query: str) -> str:
        """Search AppWorld API documentation for relevant apps and API names."""
        response = self._tool_response(task_id, "search_api_docs", query=query)
        return _tool_text(_filter_model_facing_completion_docs(response))

    def call_appworld_api(
        self,
        task_id: str,
        app_name: str,
        api_name: str,
        kwargs_json: str,
    ) -> str:
        """Call one typed AppWorld API with JSON keyword arguments.

        Args:
            task_id: AppWorld task id such as "82e2fac_1".
            app_name: AppWorld app namespace, for example "venmo" or "supervisor".
            api_name: API method name within the app namespace.
            kwargs_json: JSON object string containing keyword arguments for the API.

        Returns:
            JSON string with success, result/output, stdout, stderr, and feedback.
        """
        return self._call_appworld_api(
            task_id,
            app_name,
            api_name,
            kwargs_json,
            allow_complete_task=False,
        )

    def complete_appworld_task(self, task_id: str, kwargs_json: str) -> str:
        """Internal host-side AppWorld completion path."""
        return self._call_appworld_api(
            task_id,
            "supervisor",
            "complete_task",
            kwargs_json,
            allow_complete_task=True,
        )

    def _call_appworld_api(
        self,
        task_id: str,
        app_name: str,
        api_name: str,
        kwargs_json: str,
        *,
        allow_complete_task: bool,
    ) -> str:
        if _is_completion_api(app_name, api_name) and not allow_complete_task:
            return _blocked_completion_tool_text(task_id)
        try:
            kwargs = json.loads(kwargs_json or "{}")
        except json.JSONDecodeError as exc:
            return _tool_text(
                {
                    "task_id": task_id,
                    "session_id": task_id,
                    "operation": "call_api",
                    "success": False,
                    "feedback": f"kwargs_json is not valid JSON: {exc}",
                    "error": str(exc),
                    "result": None,
                    "output": None,
                }
            )
        if not isinstance(kwargs, dict):
            return _tool_text(
                {
                    "task_id": task_id,
                    "session_id": task_id,
                    "operation": "call_api",
                    "success": False,
                    "feedback": "kwargs_json must decode to a JSON object",
                    "result": None,
                    "output": None,
                }
            )
        return self._tool_request(
            task_id,
            "call_api",
            app_name=app_name,
            api_name=api_name,
            kwargs=kwargs,
        )

    def evaluate_appworld_task(self, task_id: str) -> str:
        """Evaluate the current persistent AppWorld task state and return the evaluator score."""
        return self._tool_request(task_id, "evaluate_task")

    def close_appworld_task(self, task_id: str) -> str:
        """Close a persistent AppWorld task session and release worker-side resources."""
        if task_id not in self._started_tasks:
            return _tool_text(
                {
                    "task_id": task_id,
                    "session_id": task_id,
                    "operation": "close_task",
                    "success": True,
                    "feedback": "task was not active",
                    "result": None,
                    "output": None,
                }
            )
        response = self.request({"op": "close_task", "task_id": task_id, "session_id": task_id})
        self._started_tasks.discard(task_id)
        return _tool_text(response)


def _default_appworld_python() -> str:
    for appworld_python in (
        Path.cwd() / ".appworld-venv" / "bin" / "python",
        Path.cwd() / "examples" / "appworld" / ".appworld-venv" / "bin" / "python",
    ):
        if appworld_python.is_file():
            # Do not resolve the venv interpreter symlink: on macOS/uv venvs the
            # symlink target can be the base Python binary, and executing the
            # resolved target loses the venv site-packages (including appworld).
            return str(appworld_python)
    return sys.executable


def _tool_text(payload: dict[str, Any]) -> str:
    text = {
        "success": bool(payload.get("success", False)),
        "stdout": str(payload.get("stdout") or "")[-4000:],
        "stderr": str(payload.get("stderr") or "")[-4000:],
        "feedback": str(payload.get("feedback") or "")[-4000:],
    }
    for field in ("score", "error", "result", "output"):
        if field in payload and payload[field] is not None:
            text[field] = payload[field]
    return json.dumps(text, sort_keys=True)


def _is_supervisor_app(app_name: Any) -> bool:
    return str(app_name).strip().lower() == "supervisor"


def _is_completion_api(app_name: Any, api_name: Any) -> bool:
    return _is_supervisor_app(app_name) and str(api_name).strip().lower() == "complete_task"


def _blocked_completion_tool_text(task_id: str) -> str:
    return _tool_text(
        {
            "task_id": task_id,
            "session_id": task_id,
            "operation": "call_api",
            "success": False,
            "feedback": _COMPLETE_TASK_FEEDBACK,
            "result": None,
            "output": None,
        }
    )


def _filter_model_facing_completion_docs(
    response: dict[str, Any],
    *,
    in_supervisor: bool = False,
) -> dict[str, Any]:
    filtered = dict(response)
    for field in ("result", "output"):
        if field in filtered:
            value = _filter_completion_doc_value(filtered[field], in_supervisor=in_supervisor)
            filtered[field] = [] if value is _DROP_COMPLETE_TASK else value
    return filtered


def _filter_completion_doc_value(value: Any, *, in_supervisor: bool = False) -> Any:
    if isinstance(value, dict):
        if _dict_is_completion_doc(value, in_supervisor=in_supervisor):
            return _DROP_COMPLETE_TASK
        result = {}
        next_in_supervisor = in_supervisor or _dict_names_supervisor(value)
        for key, item in value.items():
            key_text = str(key)
            key_in_supervisor = next_in_supervisor or key_text.lower() == "supervisor"
            if _is_completion_doc_key(key_text, in_supervisor=key_in_supervisor):
                continue
            filtered = _filter_completion_doc_value(item, in_supervisor=key_in_supervisor)
            if filtered is not _DROP_COMPLETE_TASK:
                result[key] = filtered
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            filtered = _filter_completion_doc_value(item, in_supervisor=in_supervisor)
            if filtered is not _DROP_COMPLETE_TASK:
                result.append(filtered)
        return result
    if isinstance(value, str) and _mentions_completion_doc(value):
        return "\n".join(
            line for line in value.splitlines() if not _mentions_completion_doc(line)
        )
    return value


def _dict_names_supervisor(value: dict[Any, Any]) -> bool:
    for field in ("app_name", "app", "namespace"):
        if _is_supervisor_app(value.get(field)):
            return True
    return False


def _dict_is_completion_doc(value: dict[Any, Any], *, in_supervisor: bool) -> bool:
    app_name = None
    for field in ("app_name", "app", "namespace"):
        if field in value:
            app_name = value[field]
            break
    api_name = None
    for field in ("api_name", "api", "name"):
        if field in value:
            api_name = value[field]
            break
    if api_name is None:
        return False
    return _is_completion_api(app_name or ("supervisor" if in_supervisor else ""), api_name)


def _is_completion_doc_key(key: str, *, in_supervisor: bool) -> bool:
    normalized = key.strip().lower()
    if normalized in {"supervisor.complete_task", "supervisor__complete_task"}:
        return True
    return in_supervisor and normalized == "complete_task"


def _mentions_completion_doc(text: str) -> bool:
    normalized = text.lower()
    return "supervisor.complete_task" in normalized or "supervisor__complete_task" in normalized
