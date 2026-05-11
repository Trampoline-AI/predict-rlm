from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import dspy

from predict_rlm import PredictRLM, Skill

from ..tools.runner import AppWorldSessionClient
from .signature import build_solve_appworld_task_signature
from .skills import get_appworld_skill


class AppWorldRLM(dspy.Module):
    def __init__(
        self,
        lm: dspy.LM | str | None = None,
        sub_lm: dspy.LM | str | None = None,
        max_iterations: int = 50,
        verbose: bool = False,
        debug: bool = False,
        skill: Skill | None = None,
        data_root: str | Path | None = None,
        appworld_client: AppWorldSessionClient | None = None,
    ):
        self.lm = lm
        self.sub_lm = sub_lm
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.debug = debug
        self.skill = skill
        self.appworld_client = appworld_client or AppWorldSessionClient(data_root=data_root)

    def build_predictor(
        self,
        skill: Skill | None = None,
        task_id: str = "__current_task__",
        supervisor_name: str = "",
        supervisor_email: str = "",
        supervisor_phone_number: str = "",
    ) -> PredictRLM:
        base_skill = skill or self.skill or get_appworld_skill()
        skill = base_skill.model_copy(update={"tools": self._current_task_tools(task_id)})
        return PredictRLM(
            build_solve_appworld_task_signature(
                supervisor_name=supervisor_name,
                supervisor_email=supervisor_email,
                supervisor_phone_number=supervisor_phone_number,
            ),
            lm=self.lm,
            sub_lm=self.sub_lm,
            skills=[skill],
            max_iterations=self.max_iterations,
            verbose=self.verbose,
            debug=self.debug,
        )

    async def aforward(
        self,
        task_id: str,
        instruction: str,
        supervisor_name: str = "",
        supervisor_email: str = "",
        supervisor_phone_number: str = "",
    ):
        predictor = self.build_predictor(
            task_id=task_id,
            supervisor_name=supervisor_name,
            supervisor_email=supervisor_email,
            supervisor_phone_number=supervisor_phone_number,
        )
        prediction = await predictor.acall(instruction=instruction)
        if not _trace_has_successful_complete_task(prediction):
            self._complete_task_from_prediction(task_id, prediction)
        return prediction

    def _current_task_tools(self, task_id: str) -> dict[str, Callable[..., dict[str, Any]]]:
        def list_appworld_apps() -> dict[str, Any]:
            """List AppWorld apps available for the current task."""
            return _decode_tool_result(self.appworld_client.list_appworld_apps(task_id))

        def show_appworld_api_descriptions(app_name: str) -> dict[str, Any]:
            """Show API names and short descriptions for one AppWorld app."""
            return _decode_tool_result(
                self.appworld_client.show_appworld_api_descriptions(task_id, app_name)
            )

        def show_appworld_api_doc(app_name: str, api_name: str) -> dict[str, Any]:
            """Show full documentation for one AppWorld API."""
            return _decode_tool_result(
                self.appworld_client.show_appworld_api_doc(task_id, app_name, api_name)
            )

        def search_appworld_api_docs(query: str) -> dict[str, Any]:
            """Search AppWorld API documentation for the current task."""
            return _decode_tool_result(self.appworld_client.search_appworld_api_docs(task_id, query))

        def call_appworld_api(
            app_name: str,
            api_name: str,
            kwargs: dict[str, Any],
        ) -> dict[str, Any]:
            """Call one AppWorld API for the current task with Python-dict kwargs."""
            return _decode_tool_result(
                self.appworld_client.call_appworld_api(
                    task_id,
                    app_name,
                    api_name,
                    json.dumps(kwargs),
                )
            )

        return {
            "list_appworld_apps": list_appworld_apps,
            "show_appworld_api_descriptions": show_appworld_api_descriptions,
            "show_appworld_api_doc": show_appworld_api_doc,
            "search_appworld_api_docs": search_appworld_api_docs,
            "call_appworld_api": call_appworld_api,
        }

    def _complete_task_from_prediction(self, task_id: str, prediction: Any) -> None:
        kwargs_json = json.dumps(_completion_payload_from_prediction(prediction))
        complete_task = getattr(self.appworld_client, "complete_appworld_task", None)
        if callable(complete_task):
            result = complete_task(task_id, kwargs_json)
        else:
            result = self.appworld_client.call_appworld_api(
                task_id,
                "supervisor",
                "complete_task",
                kwargs_json,
            )
        payload = _parse_tool_result(result)
        if payload is None:
            raise RuntimeError(f"AppWorld auto complete_task returned unsupported payload: {result!r}")
        if not payload.get("success", False):
            feedback = payload.get("feedback") or payload.get("error") or result
            raise RuntimeError(f"AppWorld auto complete_task failed: {feedback}")


def _completion_payload_from_prediction(prediction: Any) -> dict[str, Any]:
    if hasattr(prediction, "answer"):
        return _completion_payload_from_answer(getattr(prediction, "answer"))
    return {}


def _completion_payload_from_answer(answer: Any) -> dict[str, Any]:
    if answer is None:
        return {}
    return {"answer": answer}


def _trace_has_successful_complete_task(prediction: Any) -> bool:
    trace = getattr(prediction, "trace", None)
    for step in getattr(trace, "steps", []) or []:
        for call in getattr(step, "tool_calls", []) or []:
            if _is_successful_complete_task_call(call):
                return True
    return False


def _is_successful_complete_task_call(call: Any) -> bool:
    if getattr(call, "name", None) != "call_appworld_api":
        return False
    if getattr(call, "error", None):
        return False
    args = list(getattr(call, "args", []) or [])
    kwargs = dict(getattr(call, "kwargs", {}) or {})
    if "app_name" in kwargs or "api_name" in kwargs:
        app_name = kwargs.get("app_name")
        api_name = kwargs.get("api_name")
    elif len(args) >= 4:
        app_name = args[1]
        api_name = args[2]
    else:
        app_name = args[0] if len(args) > 0 else None
        api_name = args[1] if len(args) > 1 else None
    if app_name != "supervisor" or api_name != "complete_task":
        return False
    payload = _parse_tool_result(getattr(call, "result", None))
    return payload is not None and bool(payload.get("success", False))


def _parse_tool_result(result: Any) -> dict[str, Any] | None:
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            payload = json.loads(result)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            return payload
    return None


def _decode_tool_result(result: Any) -> dict[str, Any]:
    payload = _parse_tool_result(result)
    if payload is None:
        raise RuntimeError(f"AppWorld tool returned unsupported payload: {result!r}")
    return payload
