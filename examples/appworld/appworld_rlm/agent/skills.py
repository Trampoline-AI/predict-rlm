import ast
import json
import os
from importlib import resources
from pathlib import Path
from typing import Any

from predict_rlm import Skill

_ICL_DEMO_TASK_IDS_ASSET = "assets/appworld_icl_demo_task_ids.json"


def load_official_icl_demo_task_ids() -> dict[str, Any]:
    asset_path = resources.files(__package__) / _ICL_DEMO_TASK_IDS_ASSET
    return json.loads(asset_path.read_text())


def _default_appworld_data_root() -> Path | None:
    value = os.environ.get("APPWORLD_DATA_ROOT")
    if value:
        return Path(value)

    appworld_root = os.environ.get("APPWORLD_ROOT")
    if appworld_root:
        return Path(appworld_root) / "data"

    for candidate in (Path.cwd() / "data", Path.cwd() / "examples" / "appworld" / "data"):
        if candidate.exists():
            return candidate
    return None


def _missing_appworld_data_error(data_root: Path | None) -> FileNotFoundError:
    checked = [str(data_root)] if data_root is not None else [
        "APPWORLD_DATA_ROOT",
        "APPWORLD_ROOT/data",
        "./data",
        "./examples/appworld/data",
    ]
    return FileNotFoundError(
        "Official AppWorld demo task data is required to render ICL worked examples. "
        "The repo stores only the official demo task ID manifest, then loads the "
        "corresponding demo instructions and ground-truth solutions from local AppWorld "
        "data at runtime. Run examples/appworld/scripts/setup_appworld_data.sh or set "
        "APPWORLD_DATA_ROOT. Checked: "
        + ", ".join(checked)
    )


def render_official_icl_demos(
    demo_manifest: dict[str, Any] | None = None,
    data_root: Path | str | None = None,
) -> str:
    manifest = demo_manifest or load_official_icl_demo_task_ids()
    task_ids = [str(task_id) for task_id in manifest["demo_task_ids"]]
    resolved_data_root = Path(data_root) if data_root is not None else _default_appworld_data_root()
    if resolved_data_root is None or not resolved_data_root.is_dir():
        raise _missing_appworld_data_error(resolved_data_root)

    separator = "----------------------------------------------------------------------------"
    lines = [
        "Next, I will show you some worked-out examples as a tutorial before we proceed with the real task instruction.",
        "These tutorial examples are rendered from the official AppWorld demo task IDs and their local reference solution files.",
        "",
    ]
    for index, task_id in enumerate(task_ids, start=1):
        lines.extend(
            [
                separator,
                f"# Tutorial Task Instruction {index}",
                _load_demo_instruction(resolved_data_root, task_id),
                "Disclaimer: This is not a real task, only a tutorial with fake data values.",
                "",
                "Worked solution sketch:",
                *_python_block(_adapt_compiled_solution(resolved_data_root, task_id)),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _load_demo_instruction(data_root: Path, task_id: str) -> str:
    specs_path = data_root / "tasks" / task_id / "specs.json"
    if not specs_path.is_file():
        raise FileNotFoundError(f"missing AppWorld demo specs: {specs_path}")
    specs = json.loads(specs_path.read_text())
    instruction = specs.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"AppWorld demo specs has no instruction: {specs_path}")
    return instruction.strip()


def _adapt_compiled_solution(data_root: Path, task_id: str) -> list[str]:
    solution_path = data_root / "tasks" / task_id / "ground_truth" / "compiled_solution.py"
    if not solution_path.is_file():
        raise FileNotFoundError(f"missing AppWorld demo compiled solution: {solution_path}")

    module = ast.parse(solution_path.read_text())
    solution = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "solution"
    )
    body = [_DemoSolutionTransformer().visit(statement) for statement in solution.body]
    body = [statement for statement in body if statement is not None]
    ast.fix_missing_locations(module)
    code = ast.unparse(ast.Module(body=body, type_ignores=[]))
    code = _replace_appworld_datetime(code)
    code = _strip_empty_lines(code)
    return [
        "import json",
        "",
        "async def appworld_api(app_name, api_name, **kwargs):",
        "    response_text = await call_appworld_api(app_name, api_name, json.dumps(kwargs))",
        "    response = json.loads(response_text)",
        "    return response.get('result', response.get('output', response))",
        "",
        *code.splitlines(),
    ]


class _DemoSolutionTransformer(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        app_api = _appworld_api_call(node)
        if app_api is None:
            return node
        app_name, api_name = app_api
        if app_name == "supervisor" and api_name == "complete_task":
            return _submit_call(node)
        return ast.Await(
            value=ast.Call(
                func=ast.Name(id="appworld_api", ctx=ast.Load()),
                args=[ast.Constant(app_name), ast.Constant(api_name)],
                keywords=node.keywords,
            )
        )


def _appworld_api_call(node: ast.Call) -> tuple[str, str] | None:
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    api_name = func.attr
    app_obj = func.value
    if not isinstance(app_obj, ast.Attribute):
        return None
    if not isinstance(app_obj.value, ast.Name) or app_obj.value.id != "apis":
        return None
    return app_obj.attr, api_name


def _submit_call(node: ast.Call) -> ast.Call:
    keywords = [keyword for keyword in node.keywords if keyword.arg == "answer"]
    return ast.Call(func=ast.Name(id="SUBMIT", ctx=ast.Load()), args=[], keywords=keywords)


def _replace_appworld_datetime(code: str) -> str:
    return code.replace(
        "DateTime.now().set(month=month).set(day=1).strftime('%Y-%m-%d')",
        "'2023-01-01'",
    )


def _strip_empty_lines(code: str) -> str:
    lines = code.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _adapt_function_call(name: str, arguments: str) -> str:
    return _adapt_function_call_code(name, arguments, {})[0]


def _adapt_function_call_code(
    name: str,
    arguments: str,
    variable_counts: dict[str, int],
) -> tuple[str, str | None]:
    app_name, api_name = _split_app_api_name(name)
    kwargs = _json_object(arguments)
    if app_name == "supervisor" and api_name == "complete_task":
        if "answer" in kwargs:
            return f"SUBMIT(answer={kwargs['answer']!r})", None
        return "SUBMIT()", None
    kwargs_literal = json.dumps(kwargs, ensure_ascii=False, sort_keys=True)
    variable_name = _response_variable_name(api_name, variable_counts)
    return (
        f'{variable_name} = await call_appworld_api('
        f'"{app_name}", "{api_name}", json.dumps({kwargs_literal})'
        ")",
        variable_name,
    )


def _response_variable_name(api_name: str, variable_counts: dict[str, int]) -> str:
    base = "".join(ch.lower() if ch.isalnum() else "_" for ch in api_name).strip("_")
    base = base or "api_call"
    variable_name = f"{base}_response"
    count = variable_counts.get(variable_name, 0) + 1
    variable_counts[variable_name] = count
    return variable_name if count == 1 else f"{variable_name}_{count}"


def _split_app_api_name(name: str) -> tuple[str, str]:
    if "__" in name:
        app_name, api_name = name.split("__", 1)
    elif "." in name:
        app_name, api_name = name.split(".", 1)
    else:
        raise ValueError(f"AppWorld demo function name must include an app/API separator: {name}")
    return app_name, api_name


def _json_object(raw: str) -> dict[str, Any]:
    payload = json.loads(raw or "{}")
    if not isinstance(payload, dict):
        raise ValueError("AppWorld demo function arguments must decode to a JSON object")
    return payload


def _python_block(code_lines: list[str]) -> list[str]:
    return ["```python", *code_lines, "```"]


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
string. For example, use:

```python
import json
await call_appworld_api("spotify", "login", json.dumps({"username": "...", "password": "..."}))
```

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
