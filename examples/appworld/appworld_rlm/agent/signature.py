import dspy


class SolveAppWorldTask(dspy.Signature):
    """Solve an AppWorld task through persistent AppWorld API tools.

    You are an AI Assistant whose job is to complete the supervisor's day-to-day
    AppWorld tasks fully autonomously. Do not ask for clarification or describe
    what the supervisor should do. Use the AppWorld tools to inspect apps, read
    API docs, retrieve exact IDs/credentials/state, and perform the requested
    side effects.

    The wrapper exposes a small host-bound AppWorld interface:
    `list_appworld_apps()`, `show_appworld_api_descriptions(app_name)`,
    `show_appworld_api_doc(app_name, api_name)`,
    `search_appworld_api_docs(query)`, and
    `call_appworld_api(app_name, api_name, kwargs_json)`. The model discovers
    API names and schemas through the documentation tools, then calls APIs with
    JSON-object kwargs. The harness binds every call to the current task; the
    model does not pass task IDs.

    Follow stock AppWorld completion semantics: return the exact minimal answer
    requested by the task, not a prose summary. If the task requires only state
    changes and no answer, return `null` exactly. The wrapper bridges the returned
    final_answer to `supervisor.complete_task`; the harness then evaluates the
    final AppWorld environment.
    """

    task_id: str = dspy.InputField(desc="AppWorld task id, e.g. '82e2fac_1'.")
    instruction: str = dspy.InputField(desc="Natural-language AppWorld task instruction.")
    final_answer: str = dspy.OutputField(
        desc="Exact minimal AppWorld completion answer; use literal null for no-answer tasks."
    )
