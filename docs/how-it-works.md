# How it works

1. You define **inputs**, **outputs**, and **tools** — what the RLM receives, what it should produce, and what actions it can take
2. The outer LLM writes Python code in a sandboxed Pyodide/WASM REPL
3. Inside the sandbox, it calls `await predict(signature, **kwargs)` to invoke the sub-LM for understanding and extraction
4. It iterates — exploring data, calling tools, building up intermediate results, and handling errors
5. When done, it calls `SUBMIT()` with the final structured output

Each iteration is a REPL turn: the LLM sees the output of its previous code, decides what to do next, and writes more code. State persists between iterations, so it can accumulate findings across many steps.

## Signatures and file I/O

The DSPy signature defines the **inputs**, **outputs**, and **strategy** (via the docstring). Use `File` for file-typed fields — input files are mounted into the sandbox, output files are synced back (see [API](api.md#file) for details).

```python
from predict_rlm import File, PredictRLM, Skill

class AnalyzeDocuments(dspy.Signature):
    """Analyze documents and produce a structured report.

    1. Survey the documents — file names, page counts, document types
    2. Render pages as images and use predict() to extract content
    3. Produce the report following the criteria's format
    """
    documents: list[File] = dspy.InputField()
    analysis: DocumentAnalysis = dspy.OutputField()

pdf_skill = Skill(
    name="pdf",
    instructions="Use pymupdf to open and render PDF pages...",
    packages=["pymupdf"],
)

rlm = PredictRLM(
    AnalyzeDocuments,
    lm="openai/gpt-5.4",
    sub_lm="openai/gpt-5.1",
    skills=[pdf_skill],
)

documents = [File(path="report.pdf"), File(path="appendix.pdf")]
result = rlm(documents=documents)
```

Inside the sandbox, the RLM autonomously decides which pages to load and when:

```python
# The RLM writes code like this — you don't write this, the LLM does:
import pymupdf, base64, asyncio

doc = pymupdf.open(documents[0])
images = [
    "data:image/png;base64,"
    + base64.b64encode(
        doc[i].get_pixmap(dpi=200).tobytes("png")
    ).decode()
    for i in range(3)
]
results = await asyncio.gather(*[
    predict("page: dspy.Image -> dates: list[str]", page=img)
    for img in images
])
```

## Mutable workspaces

Use `Workspace` when the RLM should edit a host directory, such as fixing code, updating a project, or producing several related files whose names are not known up front. A workspace is an input-only field. By default, predict-rlm mirrors eligible host files into the sandbox, gives the RLM the sandbox path, and syncs sandbox changes back to the host after each code block.

```python
from predict_rlm import PredictRLM, Workspace

class FixBug(dspy.Signature):
    """Fix the bug in the project and summarize the change."""
    workspace: Workspace = dspy.InputField()
    bug_report: str = dspy.InputField()
    summary: str = dspy.OutputField()

rlm = PredictRLM(FixBug, lm="openai/gpt-5.4")
result = rlm(
    workspace=Workspace(path="/path/to/project"),
    bug_report="The CSV importer drops rows with empty optional columns.",
)
```

Inside the sandbox, the `workspace` input is a string path such as `/sandbox/workspace`, so the RLM can use ordinary Python file APIs:

```python
from pathlib import Path

root = Path(workspace)
source = root / "src" / "importer.py"
text = source.read_text()
source.write_text(text.replace("if not value: continue", "if value is None: continue"))
```

Default workspace sync is conservative:

- Excluded names such as `.git`, `.venv`, `node_modules`, caches, `dist`, and `build` are not mirrored or synced back.
- Files larger than `max_file_bytes` are skipped on the way in. If a sandbox edit grows a file beyond that limit, sync raises a conflict instead of deleting or overwriting the host file.
- Symlinks are not followed. Host symlink paths and path escapes are treated as sync conflicts.
- If the host and sandbox both change the same path since the last sync, sync raises `WorkspaceSyncConflictError` rather than clobbering the host change.

Use `File` outputs when the RLM should return specific generated artifacts. Use `Workspace` when the output is a set of edits to an existing directory.

### Direct SBX workspaces

For local coding-agent workflows, mirror mode can be too isolated: project commands often need the real lockfiles, generated files, local toolchains, and normal subprocess behavior. With the Docker Sandboxes backend, `Workspace(mode="direct")` mounts the host directory directly into the sandbox:

```python
rlm = PredictRLM(FixBug, sandbox_backend="sbx")
result = rlm(
    workspace=Workspace(
        path="/path/to/project",
        mount_path="/workspace",
        mode="direct",
    ),
    bug_report="The CSV importer drops rows with empty optional columns.",
)
```

In direct mode, the RLM receives the mounted sandbox path, Python code and child subprocesses can use that path, and file edits are real host workspace edits immediately. predict-rlm does not create a mirror, does not apply `exclude` or `max_file_bytes`, and does not run post-execute workspace sync.

Direct mode requires `sandbox_backend="sbx"`. JSPI/Pyodide rejects direct workspaces because it cannot directly mount host directories. `SbxPool` also rejects direct workspaces in this version because pooled sandboxes are created before per-call workspace inputs are known.
