import json
import os
from pathlib import Path
from importlib import resources
from typing import Any

from predict_rlm import Skill

_ICL_DEMO_TASK_IDS_ASSET = "assets/appworld_icl_demo_task_ids.json"


def load_official_icl_demo_task_ids() -> dict[str, Any]:
    asset_path = resources.files(__package__) / _ICL_DEMO_TASK_IDS_ASSET
    return json.loads(asset_path.read_text())


def _default_appworld_data_root() -> Path | None:
    for env_name in ("APPWORLD_DATA_ROOT",):
        value = os.environ.get(env_name)
        if value:
            return Path(value)
    appworld_root = os.environ.get("APPWORLD_ROOT")
    if appworld_root:
        candidate = Path(appworld_root) / "data"
        if candidate.exists():
            return candidate
    for candidate in (Path.cwd() / "data", Path.cwd() / "examples" / "appworld" / "data"):
        if candidate.exists():
            return candidate
    return None


def _load_local_task_instruction(data_root: Path, task_id: str) -> str | None:
    specs_path = data_root / "tasks" / task_id / "specs.json"
    if not specs_path.is_file():
        return None
    specs = json.loads(specs_path.read_text())
    instruction = specs.get("instruction")
    return instruction if isinstance(instruction, str) and instruction.strip() else None


def _load_local_required_apis(data_root: Path, task_id: str) -> list[str]:
    """Best-effort local-only train/dev ground-truth API hints for demo tasks.

    AppWorld/HALO load these demo tasks with full ground truth for API predictor
    and full-code demos. We do not source-control extracted ground truth; this
    helper only reads it from the user's local AppWorld installation if the
    appworld package exposes it in this Python environment.
    """
    try:
        appworld_python = os.environ.get("APPWORLD_PYTHON")
        if appworld_python:
            # Avoid importing host-incompatible AppWorld if the configured runner
            # uses a separate pydantic-v1 venv. The prompt still gets local task
            # instructions from specs.json.
            return []
        from appworld.task import Task  # type: ignore

        previous_root = os.environ.get("APPWORLD_ROOT")
        os.environ.setdefault("APPWORLD_ROOT", str(data_root.parent))
        try:
            task = Task.load(task_id, load_ground_truth=True, ground_truth_mode="full")
        finally:
            if previous_root is None:
                os.environ.pop("APPWORLD_ROOT", None)
            else:
                os.environ["APPWORLD_ROOT"] = previous_root
        ground_truth = getattr(task, "ground_truth", None)
        required_apis = getattr(ground_truth, "required_apis", None)
        if isinstance(required_apis, list):
            return [str(api) for api in required_apis]
    except Exception:
        return []
    return []


def render_official_icl_demos(
    demo_manifest: dict[str, Any] | None = None, data_root: Path | None = None
) -> str:
    manifest = demo_manifest or load_official_icl_demo_task_ids()
    task_ids = [str(task_id) for task_id in manifest["demo_task_ids"]]
    data_root = data_root or _default_appworld_data_root()

    lines = [
        "## Stock task-ID examples, adapted to this RLM wrapper",
        "",
        "The official AppWorld/HALO API-predictor and full-code configs use the",
        "train/dev demo task IDs below. This repo stores only the IDs; task text and",
        "any released train/dev ground-truth metadata are read from the local AppWorld",
        "download at runtime, not committed to git.",
        "",
        "Use these examples as behavioral patterns, not as benchmark answers. Do not",
        "memorize task IDs, split files, private data, or reference answers.",
        "",
    ]
    if data_root is None:
        lines.extend(
            [
                "Local AppWorld data root was not found when this skill was imported.",
                f"Official demo task IDs: {', '.join(task_ids)}.",
                "When data is available, load each task's instruction from",
                "`data/tasks/<task_id>/specs.json` and solve using fresh API calls.",
                "",
            ]
        )
        return "\n".join(lines)

    for index, task_id in enumerate(task_ids, start=1):
        instruction = _load_local_task_instruction(data_root, task_id)
        required_apis = _load_local_required_apis(data_root, task_id)
        lines.extend([f"Example {index}: official demo task ID `{task_id}`.", ""])
        if instruction:
            lines.append(f"Task instruction loaded locally: `{instruction}`")
        else:
            lines.append("Task instruction was not available from local AppWorld data.")
        if required_apis:
            lines.extend(
                [
                    "Released train/dev required API hints loaded locally:",
                    ", ".join(f"`{api}`" for api in required_apis),
                ]
            )
        lines.extend(
            [
                "Pattern:",
                "1. Discover relevant apps and API docs through the wrapper tools.",
                "2. Retrieve exact IDs, credentials, contacts, dates, and records from APIs.",
                "3. Perform only the requested read or mutation, with no collateral changes.",
                "4. Finish with `SUBMIT(final_answer=...)`; use `null` for no-answer tasks.",
                "",
            ]
        )
    return "\n".join(lines)


APPWORLD_SKILL_BASE_INSTRUCTIONS = """# AppWorld skill

You are an AI Assistant whose job is to complete the supervisor's day-to-day
AppWorld tasks fully autonomously. Your job is not to chat about the instruction;
your job is to make the correct API calls in the persistent AppWorld environment.

## Available RLM tools

For real tasks, this wrapper exposes exactly five lightweight AppWorld tools:

- `list_appworld_apps()`
- `show_appworld_api_descriptions(app_name)`
- `show_appworld_api_doc(app_name, api_name)`
- `search_appworld_api_docs(query)`
- `call_appworld_api(app_name, api_name, kwargs_json)`

The harness binds every tool call to the current task. Do not pass or invent a
task ID, and do not attempt benchmark cleanup yourself. Discover API names and
argument schemas through the documentation tools, then call APIs through
`call_appworld_api` with `kwargs_json` set to a JSON object string. For example,
use `call_appworld_api("spotify", "login", "{\"username\": \"...\", \"password\": \"...\"}")`.

## Stock AppWorld instructions, adapted

A. General instructions:

- Act fully on your own. Make all decisions yourself. Never ask the supervisor or
  anyone else to confirm or clarify. Your role is to solve the task, not to give
  directions to the user.
- You have full permission to operate across the connected AppWorld accounts and
  services needed for the task.
- Never invent or guess values. If a task requires an ID, username, date, song,
  product, person, address, or any other value, retrieve it through the relevant
  API before using it.
- Never leave placeholders such as `your_username`, `TODO`, or guessed IDs.
  Always fill real values by retrieving them from AppWorld APIs.
- When the instruction omits a detail, choose any valid value from available data
  unless the task implies a specific choice.
- Avoid collateral damage. Only perform what was explicitly requested. Do not
  delete, return, cancel, message, buy, follow, unfollow, or mutate unrelated
  records.
- Minimize turns. Inspect docs only as needed, and batch independent tool calls
  when possible.

B. App-specific instructions:

- All supervisor personal information, including biographical details,
  credentials, addresses, and payment cards, is stored in the Supervisor app.
  Use documented Supervisor APIs, especially `show_profile` and
  `show_account_passwords`, rather than inventing credential APIs.
- Any reference to friends, family, or another person/relation refers to people
  in the phone contacts list. Use phone/contact APIs to resolve them.
- Always obtain current date or time from the phone app's
  `get_current_date_and_time` API, never from your internal clock.
- All requests use a single default timezone.
- For temporal requests, use complete time boundaries. For example, "yesterday"
  means 00:00:00 through 23:59:59.
- References to "file system" mean the AppWorld file system app, not the host OS.
  Do not use OS modules or host filesystem assumptions.
- Paginated APIs: process all pages by looping through `page_index`; do not stop
  at the first page unless the API docs guarantee there are no more results.

C. Solving strategy:

1. Use `list_appworld_apps()` to identify relevant apps, then read API
   descriptions/docs before calling unfamiliar APIs.
2. For mutating APIs or APIs with unclear argument names/types, inspect
   `show_appworld_api_doc(app_name, api_name)` before calling.
3. Get credentials through documented Supervisor APIs when login is required,
   then log in through the target app's login API.
4. Inspect current state with read/search/list APIs before mutating anything.
5. Use exact IDs and values observed from API results. Pass arguments through
   `call_appworld_api(app_name, api_name, kwargs_json)` by their documented
   parameter names.
6. Read returned `success`, `result`/`output`, `feedback`, `stdout`, `stderr`, and
   errors carefully. After a failed call, change only the part implicated by the
   error or traceback.
7. Do not call evaluator, reference-answer, fallback-program, cleanup, or
   direct `app__api` tools. Evaluation and cleanup are harness-side only.

D. Task-completion instructions:

- Stock AppWorld requires `supervisor.complete_task` after completion. In this
  RLM wrapper, returning `final_answer`/`SUBMIT(final_answer=...)` automatically
  calls `supervisor.complete_task` behind the scenes unless you already called it
  successfully.
- If an answer is needed, return only the exact answer value requested.
- If no answer is required, return `null` exactly. Do not return `None`, `none`,
  `Done`, `success`, or a prose summary for state-change-only tasks.
- Do not call `supervisor.complete_task` yourself. Use `SUBMIT(final_answer=...)`;
  the wrapper handles AppWorld completion.
- The task is designed to be doable. Only report failure if you have exhausted
  the relevant documented APIs.

When an answer is given:

- Keep answers minimal. Return only the entity, number, or direct value requested,
  not a full sentence.
- Numbers must be numeric, not words. Return `10`, not `ten`.
"""

APPWORLD_SKILL_INSTRUCTIONS = (
    APPWORLD_SKILL_BASE_INSTRUCTIONS + "\n" + render_official_icl_demos()
)

appworld_skill = Skill(
    name="appworld",
    instructions=APPWORLD_SKILL_INSTRUCTIONS,
)

__all__ = [
    "APPWORLD_SKILL_INSTRUCTIONS",
    "appworld_skill",
    "load_official_icl_demo_task_ids",
    "render_official_icl_demos",
]
