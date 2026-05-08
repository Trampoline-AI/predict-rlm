import dspy


Answer = int | float | str | None


_SIGNATURE_TEMPLATE = """Complete the supervisor's task through app API tools.

I am your supervisor, and you are an AI Assistant whose job is to complete
my day-to-day tasks fully autonomously.

My name is: {name}. My personal email is {email} and phone number is {phone}.

Do not ask for clarification or describe what I should do. Use API documentation
to understand how to interact with the apps, retrieve exact IDs/credentials/state,
and perform the requested side effects.

You have access to five tools:
`list_appworld_apps()`, `show_appworld_api_descriptions(app_name)`,
`show_appworld_api_doc(app_name, api_name)`, `search_appworld_api_docs(query)`,
and `call_appworld_api(app_name, api_name, kwargs_json)`. The functions
correspond to APIs from various apps you have access to. Discover API names and
schemas through the documentation tools, then call APIs with JSON-object kwargs.
No extra routing argument is needed.

When the task is complete, make the terminal submission call. For answer tasks,
use `SUBMIT(answer=<exact minimal answer>)`. For no-answer or state-change-only
tasks, use `SUBMIT()`, which leaves the optional answer unset.
"""


def build_solve_appworld_task_signature(
    supervisor_name: str = "",
    supervisor_email: str = "",
    supervisor_phone_number: str = "",
) -> type[dspy.Signature]:
    return dspy.Signature(
        {
            "instruction": dspy.InputField(
                desc="Natural-language task instruction from the supervisor."
            ),
            "answer": (
                Answer,
                dspy.OutputField(
                    default=None,
                    desc="Exact minimal answer when the task requests one; leave unset for no-answer/state-change-only tasks.",
                ),
            ),
        },
        _SIGNATURE_TEMPLATE.format(
            name=supervisor_name.strip(),
            email=supervisor_email.strip(),
            phone=supervisor_phone_number.strip(),
        ),
    )


SolveAppWorldTask = build_solve_appworld_task_signature()
