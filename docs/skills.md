# Skills

predict-rlm supports [Agent Skills](https://agentskills.io): folders with a
`SKILL.md` that describe how to do a task, optionally with reference files,
Python helpers, or data. The format is shared with Claude Code, the Claude Agent
SDK, OpenAI Codex, and other agent frameworks, so the same skill folder works
across runtimes.

Skills differ from [capabilities](capabilities.md) in when their content reaches
the model:

|                         | Capability                              | Skill                                    |
| ----------------------- | --------------------------------------- | ---------------------------------------- |
| In the prompt           | Full instructions                       | Name and description only                |
| Loaded on demand        | —                                       | Instructions and files via `load_skill()` |
| PyPI packages           | Installed in the sandbox                | —                                        |
| Host tools              | Exposed alongside `predict()`           | —                                        |
| Files                   | `.py` modules on `/sandbox/lib`         | Whole folder at `/sandbox/skills/<name>/` |

Use a capability when the RLM always needs the guidance or a package. Use skills
when you have many specialized playbooks and the RLM should pick the relevant
ones itself.

## Skill folder layout

```
invoice-totals/
├── SKILL.md
├── helper.py
└── reference/
    └── currencies.json
```

`SKILL.md` starts with YAML frontmatter, followed by the instructions in Markdown:

```markdown
---
name: invoice-totals
description: Reconcile invoice line items with the stated total. Use when an invoice total needs checking.
---

# Invoice totals

1. Extract the line items with `predict()`.
2. Load `reference/currencies.json` for the decimal places of each currency.
3. Sum with `helper.total(items)` and compare with the stated total.
```

- `name` — required. Lowercase letters, digits, and single hyphens, up to 64
  characters. It must match the folder name.
- `description` — required, 1–1024 characters. It is the only text the RLM sees
  before loading the skill, so say what the skill does **and when to use it**.
- Other spec fields (`license`, `compatibility`, `metadata`, `allowed-tools`)
  are accepted and ignored.

## Using skills

```python
from predict_rlm import PredictRLM, Skill

rlm = PredictRLM(
    "invoice: File -> total_matches: bool",
    lm="openai/gpt-5.4",
    sub_lm="openai/gpt-5.1",
    skills=[Skill.from_dir("skills/invoice-totals")],
)
```

You can also define a skill inline when it has no files:

```python
Skill(
    name="tone-check",
    description="Check that a reply is polite and concise. Use before submitting customer-facing text.",
    instructions="Rewrite any sentence longer than 30 words...",
)
```

## How progressive disclosure works

1. The prompt gets a **Skills** section listing each skill's name and
   description.
2. When the task matches a description, the RLM calls
   `print(await load_skill("invoice-totals"))`. The tool returns the `SKILL.md`
   body and the list of the skill's files; like any value in the sandbox, the
   RLM only sees it once printed.
3. Skill files are copied into the sandbox at `/sandbox/skills/<name>/`, so the
   host folder is never modified. The RLM reads them with `open()` and can
   import Python helpers by adding the folder to `sys.path`.

`load_skill` is only registered when at least one skill is configured. Its
calls appear in the run trace like any other tool call, so you can see which
skills a run used.

## Sandbox limits

The sandbox runs [Pyodide](https://pyodide.org/), so:

- Python and data files in a skill work.
- Shell scripts and native binaries in a skill **cannot run**.
- A skill cannot install packages. If its helpers need a PyPI package, add a
  [capability](capabilities.md) that installs it.
