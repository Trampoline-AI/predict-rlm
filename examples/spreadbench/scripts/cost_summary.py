"""Summarize a ``cost_log.jsonl`` file from an optimize or eval run.

Every row records one (event, role, model) tuple with its own input /
output tokens and USD cost, pulled directly from a
``predict_rlm.trace.RunTrace`` at the time of the invocation. Rows are
point-in-time and never mutated — cumulative totals are just sums over
the log.

Schema per row:
    {
        "ts": "...",
        "event": "minibatch" | "valset" | "evaluate" | "proposer_call"
                 | "proposer_error" | "startup",
        "role": "main" | "sub" | "proposer" | "proposer_sub" | None,
        "model": "...",
        "input_tokens": int, "output_tokens": int, "cost_usd": float,
        ...per-event extras (proposer_call_idx, cases, iterations, ...)
    }

Note on event names: optimize runs emit ``minibatch`` (GEPA's
reflective-dataset eval, capture_traces=True) and ``valset`` (full
held-out val, capture_traces=False) as separate events so filenames
match (``task_traces/minibatch_NNNN.jsonl`` vs
``task_traces/valset_NNNN.jsonl``). Standalone eval.py still emits
``evaluate`` — it's always a valset-style run, no minibatch concept.

Usage:
    python cost_summary.py <run_dir>
    cat cost_log.jsonl | python cost_summary.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_events(path_or_stdin) -> list[dict]:
    if path_or_stdin == "-":
        lines = sys.stdin.readlines()
    else:
        p = Path(path_or_stdin)
        if p.is_dir():
            p = p / "cost_log.jsonl"
        if not p.is_file():
            raise SystemExit(f"cost_log.jsonl not found at {p}")
        lines = p.read_text().splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def totals_by_role_model(events: list[dict]) -> dict[tuple[str, str], dict]:
    agg: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "events": 0}
    )
    for e in events:
        if e.get("role") is None or e.get("model") is None:
            continue  # startup / session boundary marker
        k = (e["role"], e["model"])
        agg[k]["input_tokens"] += int(e.get("input_tokens", 0))
        agg[k]["output_tokens"] += int(e.get("output_tokens", 0))
        agg[k]["cost_usd"] += float(e.get("cost_usd", 0))
        agg[k]["events"] += 1
    return agg


def print_totals(events: list[dict]) -> None:
    agg = totals_by_role_model(events)
    print("=" * 86)
    print("Cumulative totals across entire log")
    print("=" * 86)
    print(
        f"{'role':<14} {'model':<34} {'events':>6} {'tok_in':>12} "
        f"{'tok_out':>10} {'cost':>10}"
    )
    print("-" * 86)
    grand_total = 0.0
    for (role, model), v in sorted(agg.items()):
        print(
            f"{role:<14} {model:<34} {v['events']:>6} {v['input_tokens']:>12,} "
            f"{v['output_tokens']:>10,} ${v['cost_usd']:>9.2f}"
        )
        grand_total += v["cost_usd"]
    print("-" * 86)
    print(f"{'TOTAL':<14} {'':<34} {'':>6} {'':>12} {'':>10} ${grand_total:>9.2f}")


def print_by_event(events: list[dict]) -> None:
    by_event: dict[str, float] = defaultdict(float)
    by_event_count: dict[str, int] = defaultdict(int)
    for e in events:
        by_event[e["event"]] += float(e.get("cost_usd", 0))
        by_event_count[e["event"]] += 1
    print()
    print("By event type:")
    for ev, cost in sorted(by_event.items(), key=lambda x: -x[1]):
        print(f"  {ev:<16} {by_event_count[ev]:>4} rows   ${cost:>8.2f}")


def print_sessions(events: list[dict]) -> None:
    print()
    print("Per-session subtotals (startup event = session boundary):")
    session_cost = 0.0
    session_start = events[0]["ts"] if events else "?"
    session_idx = 0
    for e in events:
        if e["event"] == "startup":
            if session_cost > 0 or session_idx > 0:
                print(
                    f"  session {session_idx:>2}: started {session_start}  "
                    f"→ ${session_cost:.2f}"
                )
            session_idx += 1
            session_start = e["ts"]
            session_cost = 0.0
        else:
            session_cost += float(e.get("cost_usd", 0))
    print(
        f"  session {session_idx:>2}: started {session_start}  "
        f"→ ${session_cost:.2f}  (current)"
    )


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else "-"
    events = load_events(arg)
    if not events:
        print("(empty log)")
        return 0
    print_totals(events)
    print_by_event(events)
    print_sessions(events)
    print(f"\n({len(events)} events in log)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
