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

For inspectable runs, use one session per run family and make the default user
entry point a split-pane `dashboard` window:

1. `dashboard` — two panes visible at once in a side-by-side vertical split:
   live job log/output on the left and stats/artifact watcher on the right. Use
   `tmux split-window -h` and `tmux select-layout even-horizontal`; avoid making
   these separate tmux windows/tabs.
2. Optional raw windows such as `run` or `stats` are fine for recovery and
   debugging, but do not make the user switch between windows just to understand
   progress.

Before creating or reusing a session, inspect existing sessions and avoid
clobbering attached or unrelated work.

When relaunching the same run family after a fix, prefer reusing the existing
session and dashboard if that keeps artifacts understandable. Create a fresh
session when reuse would confuse materially different runs.

After launch, select the dashboard window so attaching users land on the useful
view. Always give the user the attach command:

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

- the `dashboard` window exists and has both panes running.
- attaching lands on the dashboard, not a raw helper window.
- the first log lines show the expected command, environment, and output path.
- failures remain visible instead of closing the shell/window.
- the user has the attach command or the Hermes background process id.

If recurring progress updates are promised, verify the first interval actually
emits fresh progress. Do not rely on cron-only progress monitoring unless the
scheduler has been observed firing.
