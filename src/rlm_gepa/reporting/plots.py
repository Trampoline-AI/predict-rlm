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
LINEAGE_X_GAP = 0.75
LINEAGE_ANNOTATION_X_FOOTPRINT = 1.5
LINEAGE_EDGE_NODE_RADIUS = 0.25


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

    live_full_scores = list(state.get("program_full_scores_val_set") or [])
    live_subscores = list(state.get("prog_candidate_val_subscores") or [])
    parents = list(state.get("parent_program_for_candidate") or [])
    live_n = max(
        len(list(state.get("program_candidates") or [])),
        len(live_subscores),
        len(live_full_scores),
        len(parents),
    )
    if _has_live_score_data(live_full_scores, live_subscores):
        scores = [_candidate_score(index, live_full_scores, live_subscores) for index in range(live_n)]
        best_idx = max(range(len(scores)), key=scores.__getitem__) if scores else 0
    else:
        scores = [float(score) for score in summary.get("val_aggregate_scores") or []]
        best_idx = summary.get("best_idx", max(range(len(scores)), key=scores.__getitem__) if scores else 0)

    parents = list(state.get("parent_program_for_candidate") or [])
    eval_counts = _eval_counts(state.get("num_metric_calls_by_discovery"), len(scores))
    return {
        "n": len(scores),
        "scores": scores,
        "parents": parents,
        "eval_counts": eval_counts,
        "best_idx": best_idx,
        "pareto_map": state.get("program_at_pareto_front_valset") or {},
    }


def _has_live_score_data(full_scores: list[Any], subscores: list[Any]) -> bool:
    return any(_is_number(score) for score in full_scores) or any(_score_values(score) for score in subscores)


def _candidate_score(index: int, full_scores: list[Any], subscores: list[Any]) -> float:
    if index < len(full_scores) and _is_number(full_scores[index]):
        return float(full_scores[index])
    if index < len(subscores):
        values = _score_values(subscores[index])
        if values:
            return sum(values) / len(values)
    return 0.0


def _score_values(scores: Any) -> list[float]:
    if isinstance(scores, dict):
        raw_values = scores.values()
    elif isinstance(scores, list | tuple):
        raw_values = scores
    else:
        raw_values = [scores]
    return [float(value) for value in raw_values if _is_number(value)]


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _eval_counts(raw_counts: Any, n: int) -> list[Any]:
    counts = list(raw_counts or [])
    if len(counts) >= n:
        return counts[:n]
    if not counts:
        return list(range(n))
    last_count = counts[-1]
    original_len = len(counts)
    for index in range(original_len, n):
        try:
            counts.append(last_count + (index - original_len + 1))
        except TypeError:
            counts.append(index)
    return counts


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
    all_parents = [
        _valid_parent_ids(data["parents"][i] if i < len(data["parents"]) else None, child=i) for i in range(n)
    ]
    parents = [_primary_parent(parent_ids, i) for i, parent_ids in enumerate(all_parents)]
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

    x_pos = _reflow_lineage_x_positions(x_pos, depth, all_parents)
    x_pos = _separate_lineage_nodes_from_edges(x_pos, depth, all_parents)
    best_annotation_place_right: bool | None = None
    if 0 <= best_idx < n:
        x_pos, best_annotation_place_right = _reserve_lineage_annotation_space(x_pos, depth, best_idx)

    y_pos = {index: -depth[index] for index in range(n)}
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for index, parent_ids in enumerate(all_parents):
        for parent in parent_ids:
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
                f"{_parent_hover_label(all_parents[index])}"
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
        place_right = (
            best_annotation_place_right
            if candidate_idx == best_idx and best_annotation_place_right is not None
            else x_pos[candidate_idx] <= x_mid
        )
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


def _separate_lineage_nodes_from_edges(
    x_pos: dict[int, float],
    depth: list[int],
    all_parents: list[list[int]],
) -> dict[int, float]:
    separated = dict(x_pos)
    y_pos = {index: float(-candidate_depth) for index, candidate_depth in enumerate(depth)}
    for _ in range(20):
        shift = _next_lineage_node_edge_shift(separated, y_pos, all_parents)
        if shift is None:
            return separated
        candidate, new_x = shift
        separated[candidate] = new_x
    return separated


def _next_lineage_node_edge_shift(
    x_pos: dict[int, float],
    y_pos: dict[int, float],
    all_parents: list[list[int]],
) -> tuple[int, float] | None:
    for child, parent_ids in enumerate(all_parents):
        for parent in parent_ids:
            start = (x_pos[parent], y_pos[parent])
            end = (x_pos[child], y_pos[child])
            for candidate, candidate_x in x_pos.items():
                if candidate in {parent, child}:
                    continue
                point = (candidate_x, y_pos[candidate])
                if not _point_lies_between_y(point, start, end):
                    continue
                edge_x = _lineage_edge_x_at_y(start, end, y_pos[candidate])
                if abs(candidate_x - edge_x) < LINEAGE_EDGE_NODE_RADIUS:
                    return candidate, _shift_lineage_node_x(candidate, candidate_x, edge_x, x_pos, y_pos)
    return None


def _shift_lineage_node_x(
    candidate: int,
    candidate_x: float,
    edge_x: float,
    x_pos: dict[int, float],
    y_pos: dict[int, float],
) -> float:
    options = []
    preferred_direction = 1.0 if candidate_x >= edge_x else -1.0
    for direction in (preferred_direction, -preferred_direction):
        shifted_x = edge_x + direction * LINEAGE_X_GAP
        shifted_x = _space_lineage_x_from_layer(candidate, shifted_x, direction, x_pos, y_pos)
        options.append((abs(shifted_x - candidate_x), shifted_x))
    return min(options)[1]


def _space_lineage_x_from_layer(
    candidate: int,
    shifted_x: float,
    direction: float,
    x_pos: dict[int, float],
    y_pos: dict[int, float],
) -> float:
    same_layer = [
        other_x
        for other, other_x in x_pos.items()
        if other != candidate and y_pos[other] == y_pos[candidate]
    ]
    while any(abs(shifted_x - other_x) < LINEAGE_X_GAP for other_x in same_layer):
        blockers = [other_x for other_x in same_layer if abs(shifted_x - other_x) < LINEAGE_X_GAP]
        shifted_x = (max(blockers) if direction > 0 else min(blockers)) + direction * LINEAGE_X_GAP
    return shifted_x


def _point_lies_between_y(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]
) -> bool:
    return min(start[1], end[1]) < point[1] < max(start[1], end[1])


def _lineage_edge_x_at_y(start: tuple[float, float], end: tuple[float, float], y: float) -> float:
    if start[1] == end[1]:
        return start[0]
    ratio = (y - start[1]) / (end[1] - start[1])
    return start[0] + ratio * (end[0] - start[0])


def _valid_parent_ids(raw: Any, *, child: int) -> list[int]:
    parents: list[int] = []
    seen: set[int] = set()
    for parent in _parent_ids(raw):
        if parent in seen or not 0 <= parent < child:
            continue
        seen.add(parent)
        parents.append(parent)
    return parents


def _parent_hover_label(parent_ids: list[int]) -> str:
    if not parent_ids:
        return "Parent: seed"
    if len(parent_ids) == 1:
        return f"Parent: {parent_ids[0]}"
    return f"Parents: {', '.join(str(parent) for parent in parent_ids)}"


def _reflow_lineage_x_positions(
    initial_x: dict[int, float],
    depth: list[int],
    all_parents: list[list[int]],
) -> dict[int, float]:
    layers: dict[int, list[int]] = {}
    for candidate, candidate_depth in enumerate(depth):
        layers.setdefault(candidate_depth, []).append(candidate)
    if not layers:
        return initial_x

    orders = {
        candidate_depth: sorted(candidates, key=lambda candidate: (initial_x[candidate], candidate))
        for candidate_depth, candidates in layers.items()
    }
    all_children: dict[int, list[int]] = {candidate: [] for candidate in range(len(depth))}
    for child, parent_ids in enumerate(all_parents):
        for parent in parent_ids:
            all_children[parent].append(child)

    for candidate_depth in sorted(orders, reverse=True):
        positions = _rank_positions(orders)
        orders[candidate_depth] = _sort_layer_by_neighbors(
            orders[candidate_depth],
            all_children,
            positions,
        )

    for candidate_depth in sorted(orders):
        positions = _rank_positions(orders)
        parent_neighbors = {candidate: all_parents[candidate] for candidate in orders[candidate_depth]}
        orders[candidate_depth] = _sort_layer_by_neighbors(
            orders[candidate_depth],
            parent_neighbors,
            positions,
        )

    return _layered_barycenter_x_positions(orders, all_parents, initial_x)


def _sort_layer_by_neighbors(
    candidates: list[int],
    neighbors_by_candidate: dict[int, list[int]],
    positions: dict[int, float],
) -> list[int]:
    current_rank = {candidate: rank for rank, candidate in enumerate(candidates)}

    def sort_key(candidate: int) -> tuple[float, int, int]:
        neighbor_positions = [positions[neighbor] for neighbor in neighbors_by_candidate[candidate] if neighbor in positions]
        if not neighbor_positions:
            return float(current_rank[candidate]), current_rank[candidate], candidate
        return sum(neighbor_positions) / len(neighbor_positions), current_rank[candidate], candidate

    return sorted(candidates, key=sort_key)


def _rank_positions(orders: dict[int, list[int]]) -> dict[int, float]:
    return {
        candidate: float(rank)
        for candidates in orders.values()
        for rank, candidate in enumerate(candidates)
    }


def _layered_barycenter_x_positions(
    orders: dict[int, list[int]],
    all_parents: list[list[int]],
    initial_x: dict[int, float],
) -> dict[int, float]:
    x_pos: dict[int, float] = {}
    for candidate_depth in sorted(orders):
        candidates = orders[candidate_depth]
        targets = [
            _candidate_x_target(candidate, all_parents[candidate], x_pos, initial_x)
            for candidate in candidates
        ]
        for candidate, x in zip(candidates, _spread_targets(targets), strict=True):
            x_pos[candidate] = x
    return x_pos


def _candidate_x_target(
    candidate: int,
    parent_ids: list[int],
    x_pos: dict[int, float],
    initial_x: dict[int, float],
) -> float:
    parent_positions = [x_pos[parent] for parent in parent_ids if parent in x_pos]
    if parent_positions:
        return sum(parent_positions) / len(parent_positions)
    return initial_x[candidate]


def _spread_targets(targets: list[float]) -> list[float]:
    if not targets:
        return []
    x_values = list(targets)
    for index in range(1, len(x_values)):
        x_values[index] = max(x_values[index], x_values[index - 1] + LINEAGE_X_GAP)
    target_center = sum(targets) / len(targets)
    x_center = sum(x_values) / len(x_values)
    shift = target_center - x_center
    return [x + shift for x in x_values]


def _reserve_lineage_annotation_space(
    x_pos: dict[int, float],
    depth: list[int],
    candidate_idx: int,
) -> tuple[dict[int, float], bool]:
    if candidate_idx not in x_pos:
        return x_pos, True
    same_layer_nodes = [
        index
        for index, candidate_depth in enumerate(depth)
        if index != candidate_idx and candidate_depth == depth[candidate_idx] and index in x_pos
    ]
    if not same_layer_nodes:
        x_mid = (min(x_pos.values()) + max(x_pos.values())) / 2 if x_pos else 0.0
        return x_pos, x_pos[candidate_idx] <= x_mid

    x_mid = (min(x_pos.values()) + max(x_pos.values())) / 2
    prefer_right = x_pos[candidate_idx] <= x_mid
    right_cost = _annotation_space_reservation_cost(x_pos, same_layer_nodes, candidate_idx, place_right=True)
    left_cost = _annotation_space_reservation_cost(x_pos, same_layer_nodes, candidate_idx, place_right=False)
    place_right = prefer_right if right_cost == left_cost else right_cost < left_cost
    return _shift_same_layer_nodes_out_of_annotation(x_pos, same_layer_nodes, candidate_idx, place_right), place_right


def _annotation_space_reservation_cost(
    x_pos: dict[int, float],
    same_layer_nodes: list[int],
    candidate_idx: int,
    *,
    place_right: bool,
) -> float:
    shifted = _shift_same_layer_nodes_out_of_annotation(x_pos, same_layer_nodes, candidate_idx, place_right)
    return sum(abs(shifted[index] - x_pos[index]) for index in same_layer_nodes)


def _shift_same_layer_nodes_out_of_annotation(
    x_pos: dict[int, float],
    same_layer_nodes: list[int],
    candidate_idx: int,
    place_right: bool,
) -> dict[int, float]:
    shifted = dict(x_pos)
    anchor = x_pos[candidate_idx]
    if place_right:
        next_allowed = anchor + LINEAGE_ANNOTATION_X_FOOTPRINT
        for index in sorted((node for node in same_layer_nodes if x_pos[node] > anchor), key=x_pos.__getitem__):
            shifted[index] = max(shifted[index], next_allowed)
            next_allowed = shifted[index] + LINEAGE_X_GAP
    else:
        next_allowed = anchor - LINEAGE_ANNOTATION_X_FOOTPRINT
        for index in sorted((node for node in same_layer_nodes if x_pos[node] < anchor), key=x_pos.__getitem__, reverse=True):
            shifted[index] = min(shifted[index], next_allowed)
            next_allowed = shifted[index] - LINEAGE_X_GAP
    return shifted


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
    if isinstance(raw, bool):
        return []
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, list | tuple):
        parent_ids: list[int] = []
        for value in raw:
            parent_ids.extend(_parent_ids(value))
        return parent_ids
    try:
        return [int(raw)]
    except (TypeError, ValueError):
        return []


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
