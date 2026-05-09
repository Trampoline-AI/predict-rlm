# Long-Running Local Runs

Project-local guidance for evals, benchmarks, training-like jobs, and release checks
that are likely to outlive one assistant turn.

## Default stance

Choose the observability mode before launch:
- Use `tmux` when the user wants inspectability, live shells, or reruns in place.
- Use a Hermes-tracked background process with `notify_on_complete=true` when the
  user cares more about completion notification than shell visibility.
- Do not hide the actual long-running command behind detached `tmux` if the goal
  is Hermes completion notification; Hermes only sees the launcher exit.

## Tmux pattern

For inspectable runs, use one session per run family with separate windows:

1. `run` — executes the job and leaves a shell open after exit.
2. `stats` — shows canonical result stats once artifacts exist, otherwise tails
   recent logs.

Before creating or reusing a session, inspect existing sessions and avoid
clobbering attached or unrelated work.

When relaunching the same run family after a fix, prefer reusing the existing
`run` and `stats` windows if that keeps artifacts understandable. Create a fresh
session when reuse would confuse materially different runs.

Always give the user the attach command:

```bash
tmux attach -t <session>
```

## Run artifacts

Prefer a small checked-or-temporary run script over sending a long opaque command
into `tmux`. The script should:

- `set -euo pipefail`.
- `cd` to the project root.
- write to a timestamped output directory and matching log path.
- echo start time, output directory, log path, and important environment-derived
  paths.
- pipe stdout/stderr through `tee`.
- summarize results after success when the canonical result artifact exists.

The stats/watch command should be useful before and after result artifacts exist:
show canonical stats when possible, otherwise tail the log and print what it is
waiting for.

## Preflight

Before launching expensive or long runs:

- verify required credentials and provider-specific environment variables when
  switching model families.
- verify the intended semantic/code change actually landed.
- ensure artifacts will not overwrite previous results unless explicitly intended.
- confirm destructive or high-impact operations; ordinary local eval launches
  scoped to new output directories are normally OK.

## Verification

Immediately after launch, verify:

- the `run` and `stats` windows/processes exist.
- the first log lines show the expected command, environment, and output path.
- failures remain visible instead of closing the shell/window.
- the user has the attach command or the Hermes background process id.

If recurring progress updates are promised, verify the first interval actually
emits fresh progress. Do not rely on cron-only progress monitoring unless the
scheduler has been observed firing.
