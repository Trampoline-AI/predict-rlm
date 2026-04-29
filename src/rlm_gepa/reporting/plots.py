from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

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


def write_plots(run_dir: str | Path, output: str | Path | None = None) -> list[Path]:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional extra path
        raise RuntimeError("plotting requires the gepa-viz extra: plotly and kaleido") from exc

    run_path = Path(run_dir)
    data = load_plot_data(run_path)
    score_path, lineage_path = resolve_plot_output_paths(run_path, output)

    score_fig = make_score_vs_rollouts(data, go)
    lineage_fig = make_lineage(data, go)

    return [*_write_figure(score_fig, score_path), *_write_figure(lineage_fig, lineage_path)]


def resolve_plot_output_paths(run_dir: Path, output: str | Path | None = None) -> tuple[Path, Path]:
    if output is None:
        plot_dir = run_dir / "plots"
        return plot_dir / "score_vs_rollouts.png", plot_dir / "candidate_lineage.png"

    out = Path(output)
    if out.suffix:
        return (
            out.with_name(f"{out.stem}_score_vs_rollouts{out.suffix}"),
            out.with_name(f"{out.stem}_candidate_lineage{out.suffix}"),
        )
    return out / "score_vs_rollouts.png", out / "candidate_lineage.png"


def _write_figure(fig: Any, path: Path) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path), scale=4.0)
    return [path]


def load_plot_data(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "gepa_state.bin").open("rb") as f:
        state = pickle.load(f)
    summary_path = run_dir / "optimization_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if not isinstance(state, dict):
        state = dict(getattr(state, "__dict__", {}))
    scores = list(summary.get("val_aggregate_scores") or state.get("program_full_scores_val_set") or [])
    parents = list(state.get("parent_program_for_candidate") or [])
    eval_counts = list(state.get("num_metric_calls_by_discovery") or range(len(scores)))
    return {
        "n": len(scores),
        "scores": scores,
        "parents": parents,
        "eval_counts": eval_counts,
        "best_idx": summary.get("best_idx", max(range(len(scores)), key=scores.__getitem__) if scores else 0),
        "pareto_map": state.get("program_at_pareto_front_valset") or {},
    }


def make_score_vs_rollouts(data: dict[str, Any], go: Any) -> Any:
    n = data["n"]
    scores = data["scores"]
    eval_counts = data["eval_counts"]
    best_idx = data["best_idx"]
    pareto_set = _pareto_set(data["pareto_map"])
    if n == 0:
        return go.Figure()

    sorted_idx = sorted(range(n), key=lambda index: _eval_count(eval_counts, index))
    sorted_rollouts = [_eval_count(eval_counts, index) for index in sorted_idx]
    sorted_scores = [scores[index] for index in sorted_idx]

    best_so_far: list[float] = []
    running = sorted_scores[0]
    for index in sorted_idx:
        running = max(running, scores[index])
        best_so_far.append(running)

    colors: list[str] = []
    sizes: list[int] = []
    for index in sorted_idx:
        color, _, size = _classify_candidate(index, best_idx, pareto_set)
        colors.append(color)
        sizes.append(size)

    labels = [
        "" if index in (0, best_idx) else f"#{index} ({scores[index]:.3f})"
        for index in sorted_idx
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sorted_rollouts,
            y=sorted_scores,
            mode="markers+text",
            marker={"color": colors, "size": sizes, "line": {"color": "#555", "width": 1}},
            text=labels,
            textposition="top center",
            textfont={"size": 7, "color": "#aaa"},
            hovertext=[
                f"#{index}: {scores[index]:.4f} ({_eval_count(eval_counts, index)} rollouts)"
                for index in sorted_idx
            ],
            hoverinfo="text",
            name="Candidate",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=sorted_rollouts,
            y=best_so_far,
            mode="lines",
            line={"color": C_BEST, "width": 2},
            name="Best so far",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=[sorted_rollouts[0], sorted_rollouts[-1]],
            y=[scores[0], scores[0]],
            mode="lines",
            line={"color": C_RED, "width": 1, "dash": "dash"},
            name="Seed baseline",
        ),
    )

    _add_score_callout(
        fig,
        x=_eval_count(eval_counts, 0),
        y=scores[0],
        text=f"<b>Candidate 0 (seed)</b><br>Val avg: {scores[0]:.4f}",
        bordercolor=C_SEED,
        arrowcolor="rgba(226,180,77,0.6)",
        ax=110,
        ay=30,
    )
    pct_avg = (scores[best_idx] - scores[0]) / scores[0] * 100 if scores[0] else 0.0
    _add_score_callout(
        fig,
        x=_eval_count(eval_counts, best_idx),
        y=scores[best_idx],
        text=(
            f"<b>Candidate {best_idx} (best)</b><br>"
            f"Val avg: {scores[best_idx]:.4f} <b>({pct_avg:+.1f}%)</b>"
        ),
        bordercolor=C_BEST,
        arrowcolor="rgba(46,204,113,0.6)",
        ax=-15,
        ay=-80,
    )

    fig.update_layout(
        title={"text": "Score vs Rollouts", "x": 0.5, "xanchor": "center", "font": {"size": 16}},
        plot_bgcolor=C_PLOT_BG,
        paper_bgcolor=C_PAPER_BG,
        font={"color": C_TEXT, "family": FONT_FAMILY},
        xaxis={
            "title": "Number of Rollouts",
            "gridcolor": "#444",
            "rangemode": "tozero",
            "linecolor": "#666",
            "mirror": True,
            "ticks": "outside",
            "tickcolor": "#666",
        },
        yaxis={
            "title": "Val Score",
            "gridcolor": "#444",
            "range": [min(sorted_scores) - 0.02, max(sorted_scores) + 0.04],
            "linecolor": "#666",
            "mirror": True,
            "ticks": "outside",
            "tickcolor": "#666",
        },
        legend={
            "x": 0.98,
            "y": 0.02,
            "xanchor": "right",
            "yanchor": "bottom",
            "bgcolor": "rgba(0,0,0,0.5)",
            "bordercolor": "#555",
            "borderwidth": 1,
        },
        height=480,
        width=1200,
        margin={"l": 60, "r": 30, "t": 60, "b": 50},
    )
    return fig


def make_lineage(data: dict[str, Any], go: Any) -> Any:
    n = data["n"]
    scores = data["scores"]
    parents = [_primary_parent(_parent_ids(data["parents"][i]), i) if i < len(data["parents"]) else None for i in range(n)]
    best_idx = data["best_idx"]
    pareto_set = _pareto_set(data["pareto_map"])

    children: dict[int, list[int]] = {index: [] for index in range(n)}
    for index, parent in enumerate(parents):
        if parent is not None:
            children[parent].append(index)

    depth = [0] * n

    def compute_depth(node: int, current_depth: int) -> None:
        depth[node] = current_depth
        for child in children[node]:
            compute_depth(child, current_depth + 1)

    roots = [index for index, parent in enumerate(parents) if parent is None]
    for root in roots:
        compute_depth(root, 0)

    x_pos: dict[int, float] = {}
    next_x = 0

    def layout(node: int) -> None:
        nonlocal next_x
        kids = children[node]
        if not kids:
            x_pos[node] = float(next_x)
            next_x += 1
            return
        for child in kids:
            layout(child)
        x_pos[node] = sum(x_pos[child] for child in kids) / len(kids)

    for root in roots:
        layout(root)

    y_pos = {index: -depth[index] for index in range(n)}
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for index, parent in enumerate(parents):
        if parent is not None:
            edge_x.extend([x_pos[parent], x_pos[index], None])
            edge_y.extend([y_pos[parent], y_pos[index], None])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"color": C_EDGE, "width": 1.5},
            hoverinfo="skip",
            showlegend=False,
        ),
    )

    node_colors: list[str] = []
    node_outlines: list[str] = []
    for index in range(n):
        color, outline, _ = _classify_candidate(index, best_idx, pareto_set)
        node_colors.append(color)
        node_outlines.append(outline)

    fig.add_trace(
        go.Scatter(
            x=[x_pos[index] for index in range(n)],
            y=[y_pos[index] for index in range(n)],
            mode="markers+text",
            marker={"color": node_colors, "size": 40, "line": {"color": node_outlines, "width": 2}},
            text=[f"{index}<br>({scores[index]:.3f})" for index in range(n)],
            textposition="middle center",
            textfont={"size": 9, "color": "#fff"},
            hovertext=[
                f"Candidate {index}<br>Val avg: {scores[index]:.4f}<br>"
                f"Parent: {'seed' if parents[index] is None else parents[index]}"
                + ("<br><b>BEST</b>" if index == best_idx else "<br>Pareto front" if index in pareto_set else "")
                for index in range(n)
            ],
            hoverinfo="text",
            showlegend=False,
        ),
    )

    x_mid = (min(x_pos.values()) + max(x_pos.values())) / 2 if x_pos else 0.0
    for candidate_idx, border, suffix in ((0, C_SEED, "seed"), (best_idx, C_BEST, "best")):
        if candidate_idx >= n:
            continue
        lines = [f"<b>Candidate {candidate_idx} ({suffix})</b>"]
        if candidate_idx == best_idx and candidate_idx != 0:
            pct = (scores[candidate_idx] - scores[0]) / scores[0] * 100 if scores[0] else 0.0
            lines.append(f"Val avg: {scores[candidate_idx]:.4f} <b>({pct:+.1f}%)</b>")
        else:
            lines.append(f"Val avg: {scores[candidate_idx]:.4f}")
        place_right = x_pos[candidate_idx] <= x_mid
        fig.add_annotation(
            x=x_pos[candidate_idx],
            y=y_pos[candidate_idx],
            xanchor="left" if place_right else "right",
            yanchor="middle",
            xshift=30 if place_right else -30,
            text="<br>".join(lines),
            showarrow=False,
            font={"size": 10, "color": C_TEXT, "family": "monospace"},
            bgcolor="rgba(50,50,50,0.95)",
            bordercolor=border,
            borderwidth=1,
            borderpad=6,
            align="left",
        )

    xmin = min(x_pos.values()) if x_pos else 0.0
    xmax = max(x_pos.values()) if x_pos else 1.0
    ymin = min(y_pos.values()) if y_pos else -1.0
    ymax = max(y_pos.values()) if y_pos else 0.0
    xpad = max(1.5, (xmax - xmin) * 0.08)
    fig.update_layout(
        title={"text": "Candidate Lineage", "x": 0.5, "xanchor": "center", "font": {"size": 16}},
        plot_bgcolor="#333",
        paper_bgcolor=C_PLOT_BG,
        font={"color": C_TEXT, "family": FONT_FAMILY},
        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False, "range": [xmin - xpad, xmax + xpad]},
        yaxis={"showgrid": False, "zeroline": False, "showticklabels": False, "range": [ymin - 0.8, ymax + 0.6]},
        height=480,
        width=1200,
        margin={"l": 30, "r": 30, "t": 60, "b": 30},
    )
    return fig


def _primary_parent(parent_ids: list[int], child: int) -> int | None:
    for parent in parent_ids:
        if 0 <= parent < child:
            return parent
    return None


def _add_score_callout(
    fig: Any,
    *,
    x: float,
    y: float,
    text: str,
    bordercolor: str,
    arrowcolor: str,
    ax: int,
    ay: int,
) -> None:
    fig.add_annotation(
        x=x,
        y=y,
        text=text,
        ax=ax,
        ay=ay,
        showarrow=True,
        arrowhead=0,
        arrowwidth=1,
        arrowcolor=arrowcolor,
        standoff=6,
        align="left",
        bgcolor="rgba(40,40,40,0.9)",
        bordercolor=bordercolor,
        borderwidth=1,
        borderpad=6,
        font={"color": C_TEXT, "family": "monospace", "size": 10},
    )


def _parent_ids(raw: Any) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, list | tuple):
        return [int(value) for value in raw if value is not None]
    return [int(raw)]


def _pareto_set(pareto_map: dict[Any, Any]) -> set[int]:
    out: set[int] = set()
    for value in pareto_map.values():
        out.update(int(item) for item in value)
    return out


def _eval_count(eval_counts: list[Any], index: int) -> int | float:
    if index < len(eval_counts):
        return eval_counts[index]
    return index


def _classify_candidate(index: int, best_idx: int, pareto_set: set[int]) -> tuple[str, str, int]:
    if index == best_idx:
        return C_BEST, "#fff", 12
    if index == 0:
        return C_SEED, C_SEED, 12
    if index in pareto_set:
        return C_PARETO, C_TEXT, 10
    return C_GRAY, C_GRAY, 7
