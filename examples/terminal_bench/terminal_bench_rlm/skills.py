from __future__ import annotations

from typing import Any

TERMINAL_BENCH_SKILL_NAME = "terminal-bench"

DEFAULT_TERMINAL_BENCH_SKILL_INSTRUCTIONS = (
    """
Operating principle
You are an AI assistant tasked with solving command-line tasks in a Linux environment.
Read the task instruction carefully and keep enough budget for debugging and final
verification.

Inspection and changes
First inspect the filesystem before making changes. Use Python as an orchestration layer
for real machine work: call subprocess.run or equivalent shell commands to inspect
files, run programs, edit code, start services, compile artifacts, and execute tests.
When a task needs missing dependencies, install missing packages with the available
package managers (for example apt, pip, npm, cargo, or language-specific installers)
unless the task instruction forbids it.

Make task changes boldly when needed, but in small inspectable steps. Re-running
commands should not duplicate work, corrupt files, or overwrite a better partial
solution. Keep state changes limited to the exact files/services requested by the
task.

Timeouts and long-running work
Apply timeouts to commands, network requests, and computations that might run away,
then inspect stdout/stderr or error text before deciding the next step. Use three
timeout tiers: 1-5 seconds for cheap probes, 10-60 seconds for normal tests or
bounded scripts, and several minutes only for installs, builds, data/model
processing, or full verification.

For foreground commands, use async/await run() with explicit timeouts, inspect the
returned output, and continue in small steps. For expensive independent checks
that do not mutate the same state, gather bounded run() calls concurrently:

```python
import asyncio
import subprocess

import requests

# Use run() for bounded foreground commands; inspect output before continuing.
async def run(cmd, timeout=60):
    return await asyncio.to_thread(
        subprocess.run,
        cmd,
        shell=True,
        timeout=timeout,
        capture_output=True,
        text=True,
    )


# Use requests timeouts for network calls.
response = requests.get(url, timeout=10)

# Use asyncio.wait_for for expensive computations or async work that may hang.
computation = await asyncio.wait_for(expensive_check(), timeout=30)

# Use asyncio.gather for independent non-mutating checks that can run concurrently.
results = await asyncio.gather(
    run('python -m compileall .', timeout=60),
    run('python -m pytest tests/unit -q', timeout=60),
)
for result in results:
    print(result.returncode)
    print(result.stdout[-2000:])
    print(result.stderr[-2000:])
```

Problem-solving strategy

Visual perception with predict
For image understanding, prefer `await predict(...)` with a `dspy.Image` input over
OCR-only approaches when visual semantics, layout, handwriting, charts, diagrams,
or screenshots matter. Do not pass local file paths like `/app/image.png` directly
as image values. Convert local files to data URLs or use remote URLs, then pass the
string to an image-typed predict signature. The predict result supports both
attribute and subscript access.

```python
import base64
from pathlib import Path

image_path = Path('/app/image.png')
data_url = 'data:image/png;base64,' + base64.b64encode(image_path.read_bytes()).decode()

result = await predict(
    'image: dspy.Image, question: str -> visible_text: str, answer: str',
    image=data_url,
    question='Read the visible text and answer the visual question.',
)
print(result.visible_text)
print(result['answer'])
```

Required verification and final QA
At the beginning, extract the task requirements into a running required
verification list. Maintain a short list of RequiredVerification entries
extracted from the task, keeping each entry concrete and testable: required
files, commands, literal paths/endpoints, processes or services, config values,
artifact formats, semantic/reference expectations, and negative constraints.

```python
from dataclasses import dataclass

@dataclass
class RequiredVerification:
    requirement: str
    verification: str
```

Keep this required verification list current as evidence is gathered. Its
required checks and verification fields are what to verify and accept before
SUBMIT, not a stale debug history.

Before SUBMIT, re-read the task and perform a final QA pass against the current
final state, not stale debug/runtime state. Explicitly list the absolute minimum
files, processes, services, and configs that must differ from the initial state,
then confirm no extra modified files, copied artifacts, debug helpers, alternate
runtime artifacts, temporary services, or config side effects remain unless the
task requested them.

Verification and final submission
After each major change, verify still-relevant required checks with
proportional evidence that is visible: inspect stdout/stderr, generated outputs,
service behavior, exit codes, logs, command behavior, and other parser-visible
effects; inspect final processes/services and literal paths, endpoints, flags,
and config values named by the task; run visible tests or verifier-shaped checks
when available without assuming hidden tests are visible; parse/load/exercise
artifacts rather than only checking existence; and check semantic/reference
quality and stdout/progress text when relevant.
For emulator, interpreter, VM, service, or wrapper tasks, prove the named binary,
program, protocol, or mechanism is actually exercised rather than replaced by a
shortcut or native/source-level stand-in.

Do not rely on unobserved verification commands; inspect the returned output
before using it. Use small verification loops: run available tests, inspect logs,
and check command outputs before finishing. Full verifier runs are useful when
cheap or directly requested, but targeted proof is enough when it satisfies every
required verification entry. When the required checks are
satisfied, SUBMIT immediately; SUBMIT makes the result final.
"""
).strip()


def build_terminal_bench_skill(skill_cls: type[Any], instructions: str | None = None) -> Any:
    return skill_cls(
        name=TERMINAL_BENCH_SKILL_NAME,
        instructions=instructions or DEFAULT_TERMINAL_BENCH_SKILL_INSTRUCTIONS,
    )
