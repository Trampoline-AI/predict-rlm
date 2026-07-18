# Repo Map

`predict-rlm` extends DSPy's RLM with a built-in `predict()` tool. It has a
two-level execution model:

1. The outer LLM writes and executes Python in a sandbox.
2. The sub-LM handles perception and extraction through `predict()` calls.

## Key Modules

- `src/predict_rlm/predict_rlm.py`: `PredictRLM`, `predict()` tool creation,
  action/extract signatures, LM contexts, and file I/O orchestration.
- `src/predict_rlm/backends/jspi/backend.py`: default Deno/Pyodide backend.
- `src/predict_rlm/backends/sbx/backend.py`: Docker Sandboxes backend.
- `src/predict_rlm/backends/supervisor/`: shared sandbox runner process
  supervision.
- `src/predict_rlm/rlm_skills.py`: `Skill` dataclass and `merge_skills()`.
- `src/predict_rlm/_shared.py`: action/extract signature construction and tool
  doc formatting.
- `src/predict_rlm/skills/`: built-in `pdf`, `spreadsheet`, and `docx` skills.
- `src/rlm_gepa/`: RLM-GEPA optimizer integration.
- `.agents/skills/`: repo-scoped agent skills for downstream users and
  contributors.

## Example Structure

Examples generally follow:

```text
schema.py -> signature.py -> tools.py -> skills.py -> service.py -> run.py
```

Keep generated or example RLM packages grouped under `agent/`, with optional
`tools/`, `bench/`, and `gepa/` packages only when needed.

## Common Commands

```bash
uv sync
uv sync --extra examples
make test-unit
make test-integration
uv run pytest tests/test_predict_rlm.py::TestPredictTool::test_predict_returns_dict_response -v
uv run ruff check src/ tests/
git diff --check
```

Use targeted checks for narrow changes. Run broader suites when touching shared
interfaces, sandbox execution, optimizer behavior, or examples.
