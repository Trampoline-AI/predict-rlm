# How it works

1. You define **inputs**, **outputs**, and **tools** — what the RLM receives,
   what it should produce, and what actions it can take
2. The outer LLM writes Python code in a sandboxed Pyodide/WASM REPL
3. Inside the sandbox, it calls `await predict(signature, **kwargs)` to invoke
   the sub-LM for understanding and extraction
4. It iterates — exploring data, calling tools, building up intermediate
   results, and handling errors
5. When done, it calls `SUBMIT()` with the final structured output

Each iteration is a REPL turn: the LLM sees the output of its previous code,
decides what to do next, and writes more code. State persists between
iterations, so it can accumulate findings across many steps. See
[predict-rlm Architecture](../ARCHITECTURE.md) for the backend component model
and timeout/state guarantees.

## Observability

`PredictRLM` exposes three complementary views into a run:

- `prediction.trace` is the structured artifact. It records iterations, code,
  output, tool calls, `predict()` calls, token usage, timings, and errors.
- Verbose output is enabled by default and prints the same kind of execution
  story for humans as colored stderr blocks: reasoning, generated code, output,
  tool calls, errors, and `SUBMIT`. Pass `verbose=False` for quiet execution.
- `debug=True` prints timestamped lifecycle diagnostics for the RLM and sandbox:
  process startup, requests, timeouts, shutdown, and captured partial output
  before failures. Error-like debug records are colored red.

If sandbox code prints output and then raises, the printed output is preserved
before the formatted `[Error] ...` line in both the verbose stream and the
structured run trace.

## Signatures, file I/O, and in-context inputs

The DSPy signature defines the **inputs**, **outputs**, and **strategy** (via
the docstring). Use `File` for file-typed fields — input files are mounted into
the sandbox, output files are synced back (see [API](api.md#file) for details).
Use `CtxStr` for string inputs like criteria or rubrics whose adapter-prepared
value should be visible in the outer RLM prompt for the invocation.

```python
from predict_rlm import CtxStr, File, PredictRLM, Skill

class AnalyzeDocuments(dspy.Signature):
    """Analyze documents and produce a structured report.

    1. Survey the documents — file names, page counts, document types
    2. Render pages as images and use predict() to extract content
    3. Produce the report following the criteria's format
    """
    documents: list[File] = dspy.InputField()
    criteria: CtxStr = dspy.InputField(desc="Report criteria to follow")
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
result = rlm(documents=documents, criteria="Cover deadlines, fees, and risks.")
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
