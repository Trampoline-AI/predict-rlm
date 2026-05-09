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


def render_official_icl_demos(
    demo_manifest: dict[str, Any] | None = None, data_root: Path | None = None
) -> str:
    manifest = demo_manifest or load_official_icl_demo_task_ids()
    task_ids = [str(task_id) for task_id in manifest["demo_task_ids"]]
    data_root = data_root or _default_appworld_data_root()

    separator = "----------------------------------------------------------------------------"
    lines = []
    if data_root is None:
        lines.extend(
            [
                "Local tutorial task instructions are unavailable in this environment.",
                "Proceed by reading API documentation and solving the provided task directly.",
                "",
            ]
        )
        return "\n".join(lines)

    tutorial_blocks = []
    for index, task_id in enumerate(task_ids, start=1):
        instruction = _load_local_task_instruction(data_root, task_id)
        if not instruction:
            continue
        tutorial_blocks.extend(
            [
                separator,
                f"# Tutorial Task Instruction {index}",
                instruction,
                "Disclaimer: This is not a real task, only a tutorial with fake data values.",
                "",
            ]
        )
    if tutorial_blocks:
        lines.extend(
            [
                "Next, I will show you some worked-out examples as a tutorial before we proceed with the real task instruction.",
                "",
                *tutorial_blocks,
            ]
        )
    else:
        lines.extend(
            [
                "Local tutorial task instructions are unavailable in this environment.",
                "Proceed by reading API documentation and solving the provided task directly.",
                "",
            ]
        )
    return "\n".join(lines)


APPWORLD_SKILL_BASE_INSTRUCTIONS = """# App API usage rules

## Available tools

You have access to exactly five lightweight tools:

- `list_appworld_apps()`
- `show_appworld_api_descriptions(app_name)`
- `show_appworld_api_doc(app_name, api_name)`
- `search_appworld_api_docs(query)`
- `call_appworld_api(app_name, api_name, kwargs_json)`

The functions correspond to APIs from various apps you have access to. The
environment already knows which app data to use. No extra routing argument is
needed. Discover API names and argument schemas through the documentation tools,
then call APIs through `call_appworld_api` with `kwargs_json` set to a JSON object
string. For example, use
`call_appworld_api("spotify", "login", "{\"username\": \"...\", \"password\": \"...\"}")`.

A. General instructions:

- Act fully on your own. Make all decisions yourself. Never ask the supervisor or
  anyone else to confirm or clarify. Your role is to solve the task, not to give
  directions to the user.
- You have full permission to operate across the connected accounts and services
  needed for the task.
- Never invent or guess values. If a task requires an ID, username, date, song,
  product, person, address, or any other value, retrieve it through the relevant
  API before using it.
- Never leave placeholders such as `your_username`, `TODO`, or guessed IDs.
  Always fill real values by retrieving them from APIs.
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
- References to "file system" mean the file system app, not the host OS.
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
7. Use only the five tools listed above.

D. Task-completion instructions:

- Make the terminal submission call. For answer tasks, use
  `SUBMIT(answer=<exact answer value>)`, for example `SUBMIT(answer=10)`.
  For no-answer or state-change-only tasks, use `SUBMIT()`, which leaves the
  optional answer unset.
- The task is designed to be doable. Only report failure if you have exhausted
  the relevant documented APIs.

When an answer is given:

- Keep answers minimal. Return only the entity, number, or direct value requested,
  not a full sentence.
- Numbers must be numeric, not words. Return `10`, not `ten`.
"""

def get_appworld_skill_instructions(data_root: Path | str | None = None) -> str:
    return APPWORLD_SKILL_BASE_INSTRUCTIONS + "\n" + render_official_icl_demos(data_root=data_root)


def get_appworld_skill(data_root: Path | str | None = None) -> Skill:
    return Skill(
        name="appworld",
        instructions=get_appworld_skill_instructions(data_root=data_root),
    )


def __getattr__(name: str) -> Any:
    if name == "APPWORLD_SKILL_INSTRUCTIONS":
        return get_appworld_skill_instructions()
    if name == "appworld_skill":
        return get_appworld_skill()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "APPWORLD_SKILL_BASE_INSTRUCTIONS",
    "get_appworld_skill",
    "get_appworld_skill_instructions",
    "load_official_icl_demo_task_ids",
    "render_official_icl_demos",
]
