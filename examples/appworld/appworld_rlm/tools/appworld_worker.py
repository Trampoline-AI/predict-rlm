from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import traceback
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--jsonl":
        return run_jsonl_worker()
    request = json.loads(input())
    result = run_request(request)
    print(json.dumps(result, sort_keys=True))
    return 0


def run_request(request: dict[str, Any]) -> dict[str, Any]:
    task_id = str(request["task_id"])
    program = str(request["program"])
    data_root = str(request.get("data_root") or "")
    experiment_name = str(request.get("experiment_name") or "predict_rlm")
    if data_root:
        os.environ.setdefault("APPWORLD_ROOT", _appworld_root_from_data_root(data_root))

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr:
            try:
                from appworld import AppWorld

                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    with AppWorld(task_id=task_id, experiment_name=experiment_name) as world:
                        output = world.execute(program)
                        score = 0.0
                        feedback = "program executed"
                        success = False
                        if hasattr(world, "evaluate"):
                            evaluation = world.evaluate()
                            score, feedback, success = _parse_evaluation(evaluation)
                        return {
                            "task_id": task_id,
                            "session_id": task_id,
                            "operation": "run_appworld_program",
                            "success": success,
                            "score": score,
                            "stdout": _read_tempfile(stdout),
                            "stderr": _read_tempfile(stderr),
                            "feedback": feedback,
                            "output": to_jsonable(output),
                        }
            except Exception:
                return {
                    "task_id": task_id,
                    "session_id": task_id,
                    "operation": "run_appworld_program",
                    "success": False,
                    "score": 0.0,
                    "stdout": _read_tempfile(stdout),
                    "stderr": _read_tempfile(stderr),
                    "feedback": traceback.format_exc(),
                    "output": None,
                }


class JsonlAppWorldWorker:
    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._task_sessions: dict[str, str] = {}

    def close_all(self) -> None:
        for session_id in list(self._sessions):
            self._close_session(session_id)

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = str(request.get("op") or request.get("operation") or "")
        task_id = str(request.get("task_id") or "")
        session_id = str(request.get("session_id") or task_id or "")
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout:
            with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr:
                try:
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        result = self._handle_operation(operation, request, session_id, task_id)
                    result.setdefault("stdout", _read_tempfile(stdout))
                    result.setdefault("stderr", _read_tempfile(stderr))
                    return result
                except Exception:
                    return {
                        "task_id": task_id,
                        "session_id": session_id,
                        "operation": operation,
                        "success": False,
                        "score": 0.0 if operation == "evaluate_task" else None,
                        "stdout": _read_tempfile(stdout),
                        "stderr": _read_tempfile(stderr),
                        "feedback": traceback.format_exc(),
                        "error": traceback.format_exc(),
                        "result": None,
                        "output": None,
                    }

    def _handle_operation(
        self,
        operation: str,
        request: dict[str, Any],
        session_id: str,
        task_id: str,
    ) -> dict[str, Any]:
        if operation == "start_task":
            return self._start_task(request)
        if operation == "close_task":
            return self._close_task(session_id, task_id, operation)

        world, resolved_session_id, resolved_task_id = self._get_world(session_id, task_id)
        result: Any
        score: float | None = None
        feedback = ""
        success = True
        if operation == "list_apps":
            result = world.apis.api_docs.show_app_descriptions()
        elif operation == "show_api_descriptions":
            result = world.apis.api_docs.show_api_descriptions(
                app_name=str(request["app_name"])
            )
        elif operation == "show_api_doc":
            result = world.apis.api_docs.show_api_doc(
                app_name=str(request["app_name"]),
                api_name=str(request["api_name"]),
            )
        elif operation == "search_api_docs":
            result = world.apis.api_docs.search_api_docs(query=str(request["query"]))
        elif operation == "call_api":
            kwargs = request.get("kwargs") or {}
            if not isinstance(kwargs, dict):
                raise TypeError("call_api kwargs must be a JSON object")
            api = getattr(getattr(world.apis, str(request["app_name"])), str(request["api_name"]))
            result = api(**kwargs)
            _save_world_state(world)
        elif operation == "evaluate_task":
            evaluation = world.evaluate()
            score, feedback, success = _parse_evaluation(evaluation)
            result = evaluation
        else:
            raise ValueError(f"unsupported op: {operation}")

        payload = {
            "task_id": resolved_task_id,
            "session_id": resolved_session_id,
            "operation": operation,
            "success": success,
            "feedback": feedback,
            "result": to_jsonable(result),
            "output": to_jsonable(result),
        }
        if score is not None:
            payload["score"] = score
        return payload

    def _start_task(self, request: dict[str, Any]) -> dict[str, Any]:
        from appworld import AppWorld

        task_id = str(request["task_id"])
        session_id = str(request.get("session_id") or self._task_sessions.get(task_id) or task_id)
        data_root = str(request.get("data_root") or "")
        experiment_name = str(request.get("experiment_name") or "predict_rlm")
        if data_root:
            os.environ["APPWORLD_ROOT"] = _appworld_root_from_data_root(data_root)
        if session_id in self._sessions:
            return {
                "task_id": task_id,
                "session_id": session_id,
                "operation": "start_task",
                "success": True,
                "feedback": "task already started",
                "result": {"task_id": task_id, "session_id": session_id},
                "output": {"task_id": task_id, "session_id": session_id},
            }

        world = AppWorld(task_id=task_id, experiment_name=experiment_name)
        entered_world = world.__enter__()
        self._sessions[session_id] = {
            "task_id": task_id,
            "world": entered_world,
            "manager": world,
        }
        self._task_sessions[task_id] = session_id
        return {
            "task_id": task_id,
            "session_id": session_id,
            "operation": "start_task",
            "success": True,
            "feedback": "task started",
            "result": {"task_id": task_id, "session_id": session_id},
            "output": {"task_id": task_id, "session_id": session_id},
        }

    def _get_world(self, session_id: str, task_id: str) -> tuple[Any, str, str]:
        resolved_session_id = session_id or self._task_sessions.get(task_id, "")
        if resolved_session_id not in self._sessions and task_id in self._task_sessions:
            resolved_session_id = self._task_sessions[task_id]
        if resolved_session_id not in self._sessions:
            raise KeyError(f"no active AppWorld task for session_id={session_id!r} task_id={task_id!r}")
        session = self._sessions[resolved_session_id]
        return session["world"], resolved_session_id, str(session["task_id"])

    def _close_task(self, session_id: str, task_id: str, operation: str) -> dict[str, Any]:
        resolved_session_id = session_id or self._task_sessions.get(task_id, "")
        if resolved_session_id not in self._sessions and task_id in self._task_sessions:
            resolved_session_id = self._task_sessions[task_id]
        if resolved_session_id not in self._sessions:
            return {
                "task_id": task_id,
                "session_id": resolved_session_id,
                "operation": operation,
                "success": True,
                "feedback": "task was not active",
                "result": None,
                "output": None,
            }
        closed_task_id = str(self._sessions[resolved_session_id]["task_id"])
        self._close_session(resolved_session_id)
        return {
            "task_id": closed_task_id,
            "session_id": resolved_session_id,
            "operation": operation,
            "success": True,
            "feedback": "task closed",
            "result": None,
            "output": None,
        }

    def _close_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        task_id = str(session["task_id"])
        if self._task_sessions.get(task_id) == session_id:
            self._task_sessions.pop(task_id, None)
        manager = session["manager"]
        manager.__exit__(None, None, None)


def run_jsonl_worker() -> int:
    worker = JsonlAppWorldWorker()
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                response = worker.handle(request)
            except Exception:
                response = {
                    "task_id": "",
                    "session_id": "",
                    "operation": "",
                    "success": False,
                    "score": 0.0,
                    "stdout": "",
                    "stderr": "",
                    "feedback": traceback.format_exc(),
                    "error": traceback.format_exc(),
                    "result": None,
                    "output": None,
                }
            print(json.dumps(to_jsonable(response), sort_keys=True), flush=True)
    finally:
        worker.close_all()
    return 0


def _parse_evaluation(evaluation: Any) -> tuple[float, str, bool]:
    if hasattr(evaluation, "to_dict"):
        return _parse_evaluation(evaluation.to_dict())
    if hasattr(evaluation, "model_dump"):
        return _parse_evaluation(evaluation.model_dump())
    if isinstance(evaluation, dict):
        score = float(evaluation.get("score", evaluation.get("success", 0.0)) or 0.0)
        success = bool(evaluation.get("success", score >= 1.0))
        feedback_value = evaluation.get("feedback")
        if feedback_value is None and hasattr(evaluation, "get"):
            feedback_value = evaluation.get("report")
        feedback = str(feedback_value if feedback_value is not None else evaluation)
        return score, feedback, success
    if isinstance(evaluation, bool):
        return (1.0 if evaluation else 0.0), str(evaluation), evaluation
    if isinstance(evaluation, int | float):
        score = float(evaluation)
        return score, str(evaluation), score >= 1.0
    return 0.0, str(evaluation), False


def _save_world_state(world: Any) -> None:
    save_state = getattr(world, "_save_state", None)
    output_db_home_path = getattr(world, "output_db_home_path_on_disk", None)
    if callable(save_state) and output_db_home_path is not None:
        save_state(output_db_home_path)
    save_logs = getattr(world, "save_logs", None)
    if callable(save_logs):
        save_logs()


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(to_jsonable(key)): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [to_jsonable(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if hasattr(value, "dict") and callable(value.dict):
        return to_jsonable(value.dict())
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return to_jsonable(value.to_dict())
    if hasattr(value, "__dict__"):
        public = {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_") and not callable(item)
        }
        if public:
            return to_jsonable(public)
    return str(value)


def _read_tempfile(file: io.TextIOBase) -> str:
    file.seek(0)
    return file.read()


def _appworld_root_from_data_root(data_root: str) -> str:
    path = Path(data_root)
    if path.name == "data":
        parent = path.parent
        return str(parent if str(parent) else Path("."))
    return data_root


if __name__ == "__main__":
    raise SystemExit(main())
