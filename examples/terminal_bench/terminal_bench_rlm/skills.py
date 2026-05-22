from __future__ import annotations

from typing import Any

TERMINAL_BENCH_SKILL_NAME = "terminal-bench"

DEFAULT_TERMINAL_BENCH_SKILL_INSTRUCTIONS = (
    """
Operating principle
You are solving Terminal-Bench tasks inside a Linux task container. Read the task
instruction carefully and keep enough budget for debugging and final verification.
"""
    "As soon as the task is solved and verified, stop improving and submit, "
    "premature optimization is the root of all evil.\n"
    """

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

For expensive independent checks that do not mutate the same state, use a small
helper like this. For long commands, start a job, wait briefly, print tails, and
call wait(job, seconds=5) again in a later iteration instead of blocking blindly:

```python
import asyncio
import subprocess
from pathlib import Path
import requests

def tail(path, n=2000):
    return Path(path).read_text(errors='replace')[-n:] if Path(path).exists() else ''

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

# Use start()/wait() for longer jobs; poll briefly, return to loop, inspect tails later.
async def start(cmd):
    stdout_path = Path('/tmp/job-stdout.log')
    stderr_path = Path('/tmp/job-stderr.log')
    stdout = stdout_path.open('w')
    stderr = stderr_path.open('w')
    proc = subprocess.Popen(cmd, shell=True, stdout=stdout, stderr=stderr, text=True)
    return {'proc': proc, 'stdout_path': stdout_path, 'stderr_path': stderr_path}

async def wait(job, seconds=5):
    try:
        returncode = await asyncio.to_thread(job['proc'].wait, timeout=seconds)
        status = 'completed'
    except subprocess.TimeoutExpired:
        returncode = None
        status = 'running'
    return {
        'status': status,
        'returncode': returncode,
        'stdout_tail': tail(job['stdout_path']),
        'stderr_tail': tail(job['stderr_path']),
    }

# Use requests timeouts for network calls.
response = requests.get(url, timeout=10)

# Use asyncio.wait_for for expensive computations or async work that may hang.
computation = await asyncio.wait_for(asyncio.to_thread(expensive_fn), timeout=60)

job = await start('pytest -q')
progress = await wait(job, seconds=5)
print(progress)

# Use asyncio.gather for independent non-mutating checks that can run concurrently.
results = await asyncio.gather(
    run('python -m compileall .', timeout=60),
    run('python -m pytest tests/unit -q', timeout=60),
)
```

Problem-solving strategy
Avoid brute-force searches when a direct, sampled, analytical, or tool-assisted
approach can solve the problem, and choose elegant, smart, effective strategies
over exhaustive loops. Use programmatic tools for binary, image, audio, video,
archive, or other non-text inputs rather than guessing.

Work iteratively: establish a faithful partial solution early, keep the best-so-far
behavior in place, and improve it step by step. When the objective is unbounded,
first satisfy the bounded contract.

Verification and final submission
Verify the observable task contract after each major change: commands, interfaces,
stdout/stderr, generated outputs, service behavior, exit codes, and other
parser-visible effects. Do not submit based on an unobserved verification command;
run the verification in one iteration, inspect the printed output, then submit in a
separate later iteration because SUBMIT makes the result final. Use small
verification loops: run available tests, inspect logs, and check command outputs
before finishing.

Verify the observable task contract, then submit; do not keep optimizing after success.
"""
).strip()


def build_terminal_bench_skill(skill_cls: type[Any], instructions: str | None = None) -> Any:
    return skill_cls(
        name=TERMINAL_BENCH_SKILL_NAME,
        instructions=instructions or DEFAULT_TERMINAL_BENCH_SKILL_INSTRUCTIONS,
    )
