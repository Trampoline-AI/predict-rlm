from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class AppWorldRunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunnerResult:
    task_id: str
    success: bool
    score: float
    stdout: str
    stderr: str
    feedback: str
    output: Any | None = None

    @classmethod
    def from_json(cls, payload: str) -> "RunnerResult":
        data = json.loads(payload)
        return cls(
            task_id=str(data.get("task_id", "")),
            success=bool(data.get("success", False)),
            score=float(data.get("score", 0.0)),
            stdout=str(data.get("stdout", "")),
            stderr=str(data.get("stderr", "")),
            feedback=str(data.get("feedback", "")),
            output=data.get("output"),
        )

    def to_tool_text(self) -> str:
        return json.dumps(
            {
                "task_id": self.task_id,
                "success": self.success,
                "score": self.score,
                "stdout": self.stdout[-4000:],
                "stderr": self.stderr[-4000:],
                "feedback": self.feedback[-4000:],
                "output": self.output,
            },
            sort_keys=True,
        )


class AppWorldRunnerClient:
    def __init__(
        self,
        python: str | None = None,
        data_root: str | Path | None = None,
        timeout: int = 300,
    ):
        self.python = python or os.environ.get("APPWORLD_PYTHON") or _default_appworld_python()
        self.data_root = Path(data_root or os.environ.get("APPWORLD_DATA_ROOT", "data"))
        self.timeout = timeout
        self.worker = Path(__file__).with_name("appworld_worker.py")

    def run(self, task_id: str, program: str, experiment_name: str = "predict_rlm") -> RunnerResult:
        request = {
            "task_id": task_id,
            "program": program,
            "experiment_name": experiment_name,
            "data_root": str(self.data_root),
        }
        proc = subprocess.run(
            [self.python, str(self.worker)],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            timeout=self.timeout,
            check=False,
        )
        if proc.returncode != 0:
            raise AppWorldRunnerError(
                f"AppWorld runner exited {proc.returncode}: {proc.stderr.strip()}"
            )
        try:
            return RunnerResult.from_json(proc.stdout)
        except json.JSONDecodeError as exc:
            raise AppWorldRunnerError(f"AppWorld runner returned non-JSON: {proc.stdout[:500]}") from exc

    def run_appworld_program(self, task_id: str, program: str) -> str:
        """Execute a self-contained Python program against an AppWorld task.

        Args:
            task_id: AppWorld task id such as "82e2fac_1".
            program: Python source to run inside an isolated AppWorld environment.

        Returns:
            JSON string with success, normalized score, stdout, stderr, feedback,
            and optional evaluator output. The program runs in a fresh environment
            for each call.
        """
        return self.run(task_id=task_id, program=program).to_tool_text()


class AppWorldSessionClient(AppWorldRunnerClient):
    def __init__(
        self,
        python: str | None = None,
        data_root: str | Path | None = None,
        timeout: int = 300,
        experiment_name: str = "predict_rlm",
    ):
        super().__init__(python=python, data_root=data_root, timeout=timeout)
        self.experiment_name = experiment_name
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
            stderr = proc.stderr.read() if proc.stderr is not None else ""
            self.close()
            raise AppWorldRunnerError(f"AppWorld JSONL worker exited without a response: {stderr[-1000:]}")
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

    def _tool_request(self, task_id: str, op: str, **kwargs: Any) -> str:
        self._ensure_task(task_id)
        response = self.request({"op": op, "task_id": task_id, "session_id": task_id, **kwargs})
        return _tool_text(response)

    def list_appworld_apps(self, task_id: str) -> str:
        """List AppWorld apps and short app descriptions for a task."""
        return self._tool_request(task_id, "list_apps")

    def show_appworld_api_descriptions(self, task_id: str, app_name: str) -> str:
        """Show available AppWorld APIs for one app."""
        return self._tool_request(task_id, "show_api_descriptions", app_name=app_name)

    def show_appworld_api_doc(self, task_id: str, app_name: str, api_name: str) -> str:
        """Show detailed AppWorld API documentation for one app API."""
        return self._tool_request(task_id, "show_api_doc", app_name=app_name, api_name=api_name)

    def search_appworld_api_docs(self, task_id: str, query: str) -> str:
        """Search AppWorld API documentation for relevant apps and API names."""
        return self._tool_request(task_id, "search_api_docs", query=query)

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
            JSON string with success, operation, result/output, stdout, stderr, and feedback.
        """
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
    local_appworld_python = Path.cwd() / ".appworld-venv" / "bin" / "python"
    if local_appworld_python.is_file():
        # Do not resolve the venv interpreter symlink: on macOS/uv venvs the
        # symlink target can be the base Python binary, and executing the
        # resolved target loses the venv site-packages (including appworld).
        return str(local_appworld_python)
    return sys.executable


def _tool_text(payload: dict[str, Any]) -> str:
    return json.dumps(
        {
            "task_id": payload.get("task_id", ""),
            "session_id": payload.get("session_id", ""),
            "operation": payload.get("operation", ""),
            "success": bool(payload.get("success", False)),
            "score": payload.get("score"),
            "stdout": str(payload.get("stdout") or "")[-4000:],
            "stderr": str(payload.get("stderr") or "")[-4000:],
            "feedback": str(payload.get("feedback") or "")[-4000:],
            "error": payload.get("error"),
            "result": payload.get("result"),
            "output": payload.get("output"),
        },
        sort_keys=True,
    )
