"""Generate PNG visualization of a GEPA optimize run.

Reads gepa_state.bin and optimization_summary.json from a run directory
and produces two charts:
  1. Score vs Rollouts — candidate scores against cumulative metric calls,
     with a best-so-far trend line and seed baseline.
  2. Candidate Lineage — parent-child tree with depth-based layout.

Usage:
    uv run --with plotly --with kaleido python scripts/plot_optimize.py \\
        runs/optimize_20260421_135558_gpt-5.4-mini__sub-gpt-5.4-mini__prop-claude-sonnet-4-6

    # Custom output path:
    uv run --with plotly --with kaleido python scripts/plot_optimize.py \\
        runs/optimize_20260421_... -o /tmp/my_chart.png
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -- Color palette (dark theme, consistent across both charts) --------
C_BEST = "#2ecc71"
C_SEED = "#e2b44d"
C_PARETO = "#3498db"
C_GRAY = "#888"
C_EDGE = "#555"
C_RED = "#e74c3c"
C_TEXT = "#ddd"
C_PLOT_BG = "#2a2a2a"
C_PAPER_BG = "#1e1e1e"
FONT_FAMILY = "Inter, Helvetica, Arial, sans-serif"


def load_run(run_dir: Path) -> dict:
    with open(run_dir / "gepa_state.bin", "rb") as f:
        state = pickle.load(f)
    summary = json.loads((run_dir / "optimization_summary.json").read_text())
    return {
        "n": len(state["parent_program_for_candidate"]),
        "scores": summary["val_aggregate_scores"],
        "subscores": state["prog_candidate_val_subscores"],
        "parents": state["parent_program_for_candidate"],
        "eval_counts": state["num_metric_calls_by_discovery"],
        "pareto_map": state["program_at_pareto_front_valset"],
        "best_idx": summary["best_idx"],
        "run_dir": run_dir.name,
        "total_cost": summary.get("total_cost_usd"),
        "config": summary.get("config", {}),
    }


def build_pareto_set(pareto_map: dict) -> set[int]:
    s: set[int] = set()
    for front in pareto_map.values():
        s |= front
    return s


def classify_candidate(
    i: int, best_idx: int, pareto_set: set[int]
) -> tuple[str, str, int]:
    """Return (color, outline_color, marker_size) for candidate i."""
    if i == best_idx:
        return C_BEST, "#fff", 12
    if i == 0:
        return C_SEED, C_SEED, 12
    if i in pareto_set:
        return C_PARETO, C_TEXT, 10
    return C_GRAY, C_GRAY, 7


# =====================================================================
# Chart 1: Score vs Rollouts
# =====================================================================

def make_score_vs_rollouts(data: dict) -> go.Figure:
    n = data["n"]
    scores = data["scores"]
    eval_counts = data["eval_counts"]
    best_idx = data["best_idx"]
    pareto_set = build_pareto_set(data["pareto_map"])
    subscores = data["subscores"]

    sorted_idx = sorted(range(n), key=lambda i: eval_counts[i])
    sorted_rollouts = [eval_counts[i] for i in sorted_idx]
    sorted_scores = [scores[i] for i in sorted_idx]

    # Best-so-far line
    best_so_far = []
    running = 0.0
    for i in sorted_idx:
        running = max(running, scores[i])
        best_so_far.append(running)

    # Per-dot styling
    colors, sizes = [], []
    for i in sorted_idx:
        c, _, s = classify_candidate(i, best_idx, pareto_set)
        colors.append(c)
        sizes.append(s)

    labels = [
        "" if sorted_idx[j] in (0, best_idx)
        else f"#{sorted_idx[j]} ({sorted_scores[j]:.3f})"
        for j in range(len(sorted_idx))
    ]

    fig = go.Figure()

    # Candidate dots
    fig.add_trace(go.Scatter(
        x=sorted_rollouts, y=sorted_scores,
        mode="markers+text",
        marker=dict(color=colors, size=sizes, line=dict(color="#555", width=1)),
        text=labels, textposition="top center",
        textfont=dict(size=7, color="#aaa"),
        name="Candidate",
        hovertext=[
            f"#{sorted_idx[j]}: {sorted_scores[j]:.4f} "
            f"({eval_counts[sorted_idx[j]]} rollouts)"
            for j in range(len(sorted_idx))
        ],
        hoverinfo="text",
    ))

    # Best-so-far trend
    fig.add_trace(go.Scatter(
        x=sorted_rollouts, y=best_so_far,
        mode="lines", line=dict(color=C_BEST, width=2),
        name="Best so far",
    ))

    # Seed baseline
    fig.add_trace(go.Scatter(
        x=[sorted_rollouts[0], sorted_rollouts[-1]],
        y=[scores[0], scores[0]],
        mode="lines", line=dict(color=C_RED, width=1, dash="dash"),
        name="Seed baseline",
    ))

    # Seed annotation card
    seed_lines = [f"<b>Candidate 0 (seed)</b>", f"Val avg: {scores[0]:.4f}"]
    fig.add_annotation(
        x=eval_counts[0], y=scores[0],
        ax=110, ay=30,
        text="<br>".join(seed_lines),
        showarrow=True, arrowhead=0, arrowwidth=1,
        arrowcolor="rgba(226,180,77,0.6)",
        standoff=6,
        font=dict(size=10, color=C_TEXT, family="monospace"),
        bgcolor="rgba(40,40,40,0.9)",
        bordercolor=C_SEED, borderwidth=1, borderpad=6, align="left",
    )

    # Best annotation card
    pct_avg = (scores[best_idx] - scores[0]) / scores[0] * 100
    best_lines = [
        f"<b>Candidate {best_idx} (best)</b>",
        f"Val avg: {scores[best_idx]:.4f} <b>({pct_avg:+.1f}%)</b>",
    ]
    fig.add_annotation(
        x=eval_counts[best_idx], y=scores[best_idx],
        ax=-15, ay=-80,
        text="<br>".join(best_lines),
        showarrow=True, arrowhead=0, arrowwidth=1,
        arrowcolor="rgba(46,204,113,0.6)",
        standoff=6,
        font=dict(size=10, color=C_TEXT, family="monospace"),
        bgcolor="rgba(40,40,40,0.9)",
        bordercolor=C_BEST, borderwidth=1, borderpad=6, align="left",
    )

    fig.update_layout(
        title=dict(text="Score vs Rollouts", x=0.5, xanchor="center",
                   font=dict(size=16)),
        plot_bgcolor=C_PLOT_BG, paper_bgcolor=C_PAPER_BG,
        font=dict(color=C_TEXT, family=FONT_FAMILY),
        xaxis=dict(
            title="Number of Rollouts",
            gridcolor="#444", rangemode="tozero",
            linecolor="#666", mirror=True,
            ticks="outside", tickcolor="#666",
        ),
        yaxis=dict(
            title="Val Score",
            gridcolor="#444",
            range=[min(sorted_scores) - 0.02, max(sorted_scores) + 0.04],
            linecolor="#666", mirror=True,
            ticks="outside", tickcolor="#666",
        ),
        legend=dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom",
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="#555", borderwidth=1),
        height=480, width=1200,
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig


# =====================================================================
# Chart 2: Candidate Lineage
# =====================================================================

def make_lineage(data: dict) -> go.Figure:
    n = data["n"]
    scores = data["scores"]
    parents_raw = data["parents"]
    best_idx = data["best_idx"]
    pareto_set = build_pareto_set(data["pareto_map"])

    parents = [p[0] for p in parents_raw]

    # Build children map and depth
    children: dict[int, list[int]] = {i: [] for i in range(n)}
    depth = [0] * n
    for i in range(n):
        if parents[i] is not None:
            children[parents[i]].append(i)

    def compute_depth(node: int, d: int) -> None:
        depth[node] = d
        for c in children[node]:
            compute_depth(c, d + 1)

    roots = [i for i in range(n) if parents[i] is None]
    for r in roots:
        compute_depth(r, 0)

    max_depth = max(depth) if depth else 0

    # Post-order traversal for x positioning
    x_pos: dict[int, float] = {}
    next_x = [0]

    def layout(node: int) -> None:
        kids = children[node]
        if not kids:
            x_pos[node] = next_x[0]
            next_x[0] += 1
        else:
            for k in kids:
                layout(k)
            x_pos[node] = sum(x_pos[k] for k in kids) / len(kids)

    for r in roots:
        layout(r)

    y_pos = {i: -depth[i] for i in range(n)}

    # Edge traces
    edge_x, edge_y = [], []
    for i in range(n):
        if parents[i] is not None:
            p = parents[i]
            edge_x += [x_pos[p], x_pos[i], None]
            edge_y += [y_pos[p], y_pos[i], None]

    fig = go.Figure()

    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines", line=dict(color=C_EDGE, width=1.5),
        hoverinfo="skip", showlegend=False,
    ))

    # Nodes
    node_colors, node_outlines = [], []
    for i in range(n):
        c, o, _ = classify_candidate(i, best_idx, pareto_set)
        node_colors.append(c)
        node_outlines.append(o)

    node_labels = [f"{i}<br>({scores[i]:.3f})" for i in range(n)]
    hover_texts = [
        f"Candidate {i}<br>"
        f"Val avg: {scores[i]:.4f}<br>"
        f"Parent: {'seed' if parents[i] is None else parents[i]}"
        + ("<br><b>★ BEST</b>" if i == best_idx
           else "<br>◆ Pareto front" if i in pareto_set
           else "")
        for i in range(n)
    ]

    fig.add_trace(go.Scatter(
        x=[x_pos[i] for i in range(n)],
        y=[y_pos[i] for i in range(n)],
        mode="markers+text",
        marker=dict(color=node_colors, size=40,
                    line=dict(color=node_outlines, width=2)),
        text=node_labels,
        textfont=dict(size=9, color="#fff"),
        textposition="middle center",
        hovertext=hover_texts, hoverinfo="text",
        showlegend=False,
    ))

    # Annotation cards for seed and best
    x_mid = (min(x_pos.values()) + max(x_pos.values())) / 2
    for ci, border, suffix in [(0, C_SEED, "seed"), (best_idx, C_BEST, "best")]:
        lines = [f"<b>Candidate {ci} ({suffix})</b>"]
        if ci == best_idx and ci != 0:
            pct = (scores[ci] - scores[0]) / scores[0] * 100
            lines.append(f"Val avg: {scores[ci]:.4f} <b>({pct:+.1f}%)</b>")
        else:
            lines.append(f"Val avg: {scores[ci]:.4f}")

        place_right = x_pos[ci] <= x_mid
        fig.add_annotation(
            x=x_pos[ci], y=y_pos[ci],
            xanchor="left" if place_right else "right",
            yanchor="middle",
            xshift=30 if place_right else -30,
            text="<br>".join(lines),
            showarrow=False,
            font=dict(size=10, color=C_TEXT, family="monospace"),
            bgcolor="rgba(50,50,50,0.95)",
            bordercolor=border, borderwidth=1, borderpad=6, align="left",
        )

    xmin, xmax = min(x_pos.values()), max(x_pos.values())
    ymin, ymax = min(y_pos.values()), max(y_pos.values())
    xpad = max(1.5, (xmax - xmin) * 0.08)

    fig.update_layout(
        title=dict(text="Candidate Lineage", x=0.5, xanchor="center",
                   font=dict(size=16)),
        plot_bgcolor="#333", paper_bgcolor=C_PLOT_BG,
        font=dict(color=C_TEXT, family=FONT_FAMILY),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[xmin - xpad, xmax + xpad]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[ymin - 0.8, ymax + 0.6]),
        height=480,
        width=1200,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig


# =====================================================================
# Combined output
# =====================================================================

def make_combined(data: dict) -> go.Figure:
    """Stack Score-vs-Rollouts on top, Lineage on bottom, in one figure."""
    fig1 = make_score_vs_rollouts(data)
    fig2 = make_lineage(data)

    n = data["n"]
    max_depth = 0
    parents_flat = [p[0] for p in data["parents"]]
    depth = [0] * n
    children: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        if parents_flat[i] is not None:
            children[parents_flat[i]].append(i)
    def _depth(node: int, d: int) -> None:
        nonlocal max_depth
        depth[node] = d
        max_depth = max(max_depth, d)
        for c in children[node]:
            _depth(c, d + 1)
    for i in range(n):
        if parents_flat[i] is None:
            _depth(i, 0)

    lineage_h = max(500, (max_depth + 1) * 90 + 100)
    score_h = 500
    total_h = score_h + lineage_h + 80

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[score_h / total_h, lineage_h / total_h],
        vertical_spacing=0.08,
        subplot_titles=["Score vs Rollouts", "Candidate Lineage"],
    )

    for trace in fig1.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in fig2.data:
        fig.add_trace(trace, row=2, col=1)

    # Copy annotations with yref adjustment
    for ann in fig1.layout.annotations:
        ann_dict = ann.to_plotly_json()
        ann_dict["xref"] = "x1"
        ann_dict["yref"] = "y1"
        if "axref" not in ann_dict:
            ann_dict["axref"] = "pixel"
        if "ayref" not in ann_dict:
            ann_dict["ayref"] = "pixel"
        fig.add_annotation(ann_dict)

    for ann in fig2.layout.annotations:
        ann_dict = ann.to_plotly_json()
        ann_dict["xref"] = "x2"
        ann_dict["yref"] = "y2"
        fig.add_annotation(ann_dict)

    # Axis styling
    fig.update_xaxes(
        title_text="Number of Rollouts", gridcolor="#444", rangemode="tozero",
        linecolor="#666", mirror=True, ticks="outside", tickcolor="#666",
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Val Score", gridcolor="#444",
        linecolor="#666", mirror=True, ticks="outside", tickcolor="#666",
        row=1, col=1,
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False, showticklabels=False,
        row=2, col=1,
    )
    fig.update_yaxes(
        showgrid=False, zeroline=False, showticklabels=False,
        row=2, col=1,
    )

    run_name = data["run_dir"]
    config = data["config"]
    lm_slug = config.get("lm", "?")
    cost = data.get("total_cost")
    subtitle = f"{lm_slug} · {n} candidates"
    if cost:
        subtitle += f" · ${cost:.0f}"

    fig.update_layout(
        title=dict(
            text=f"{run_name}<br><sup>{subtitle}</sup>",
            x=0.5, xanchor="center", font=dict(size=14),
        ),
        plot_bgcolor=C_PLOT_BG, paper_bgcolor=C_PAPER_BG,
        font=dict(color=C_TEXT, family=FONT_FAMILY),
        showlegend=True,
        legend=dict(x=0.98, y=0.55, xanchor="right", yanchor="bottom",
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="#555", borderwidth=1),
        height=total_h, width=1100,
        margin=dict(l=60, r=40, t=80, b=30),
    )

    # Override subplot title styling
    for ann in fig.layout.annotations:
        if hasattr(ann, "text") and ann.text in ("Score vs Rollouts",
                                                   "Candidate Lineage"):
            ann.font = dict(size=14, color=C_TEXT)

    return fig


def trim_candidates(data: dict, n: int) -> dict:
    """Keep only the first *n* candidates (by discovery order)."""
    if n >= data["n"]:
        return data
    keep = set(range(n))
    best_idx = data["best_idx"]
    if best_idx >= n:
        best_idx = max(range(n), key=lambda i: data["scores"][i])
    pareto_map = {}
    for k, front in data["pareto_map"].items():
        trimmed = front & keep
        if trimmed:
            pareto_map[k] = trimmed
    return {
        "n": n,
        "scores": data["scores"][:n],
        "subscores": data["subscores"][:n],
        "parents": data["parents"][:n],
        "eval_counts": data["eval_counts"][:n],
        "pareto_map": pareto_map,
        "best_idx": best_idx,
        "run_dir": data["run_dir"],
        "total_cost": data.get("total_cost"),
        "config": data.get("config", {}),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Plot GEPA optimize run as PNG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("run_dir", type=Path,
                   help="path to the optimize run directory")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="output PNG path (default: <run_dir>/optimize_plot.png)")
    p.add_argument("--separate", action="store_true",
                   help="emit two separate PNGs instead of one combined chart")
    p.add_argument("-n", "--max_candidates", type=int, default=None,
                   help="show only the first N candidates (by discovery order)")
    p.add_argument("--scale", type=float, default=4.0,
                   help="PNG export scale factor (4.0 = high DPI)")
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    if not (run_dir / "gepa_state.bin").exists():
        print(f"Error: {run_dir}/gepa_state.bin not found", file=sys.stderr)
        return 1

    data = load_run(run_dir)
    if args.max_candidates is not None:
        data = trim_candidates(data, args.max_candidates)
    print(f"Loaded {data['n']} candidates from {run_dir.name}")
    print(f"  Best: candidate {data['best_idx']} "
          f"(score {data['scores'][data['best_idx']]:.4f})")

    if args.separate:
        out_score = args.output or run_dir / "score_vs_rollouts.png"
        out_lineage = (
            args.output.with_stem(args.output.stem + "_lineage")
            if args.output else run_dir / "candidate_lineage.png"
        )
        fig1 = make_score_vs_rollouts(data)
        fig1.write_image(str(out_score), scale=args.scale)
        print(f"  → {out_score}")
        fig2 = make_lineage(data)
        fig2.write_image(str(out_lineage), scale=args.scale)
        print(f"  → {out_lineage}")
    else:
        out = args.output or run_dir / "optimize_plot.png"
        fig = make_combined(data)
        fig.write_image(str(out), scale=args.scale)
        print(f"  → {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
