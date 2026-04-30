"""Pair-selection logic for evidence-backed RLM merge patching."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

ProgramIdx = int


@dataclass(frozen=True)
class PatchMergePair:
    parent_a_id: int
    parent_b_id: int
    base_parent_id: int
    patch_source_parent_id: int
    ancestor: int
    common_ancestors: tuple[int, ...]
    oracle_score: float
    oracle_gain: float
    base_wins: tuple[Any, ...]
    patch_source_wins: tuple[Any, ...]
    weight: float


def walk_ancestors(
    parent_list: Sequence[Sequence[int | None]],
    node: int,
) -> set[int]:
    """Return all transitive ancestors of ``node`` in GEPA's parent list."""
    seen: set[int] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        parents = parent_list[current] if 0 <= current < len(parent_list) else None
        if not parents:
            continue
        for parent in parents:
            if parent is not None and parent not in seen:
                seen.add(parent)
                stack.append(parent)
    return seen


def pick_patch_merge_pair(
    *,
    merge_candidates: Sequence[int],
    program_candidates: Sequence[dict[str, str]],
    parent_program_for_candidate: Sequence[Sequence[int | None]],
    prog_candidate_val_subscores: Sequence[dict[Any, float]],
    tracked_scores: Sequence[float],
    merges_performed: list[tuple[int, int, int]],
    rng: random.Random,
    component_name: str,
    min_each: int = 3,
    eps: float = 1e-6,
) -> PatchMergePair | None:
    """Select a base parent plus patch-source parent using oracle upside.

    The common ancestor is lineage/debug metadata only. Retry dedupe is
    pair-level because patch prompts do not receive ancestor text.
    """
    if len(merge_candidates) < 2:
        return None
    if len(parent_program_for_candidate) < 3 or not tracked_scores:
        return None

    attempted_pairs = {
        (min(id1, id2), max(id1, id2))
        for id1, id2, _ancestor in merges_performed
    }
    current_best_score = max(tracked_scores)

    eligible: list[PatchMergePair] = []
    for a, b in combinations(sorted(set(merge_candidates)), 2):
        if program_candidates[a].get(component_name) == program_candidates[b].get(component_name):
            continue
        if (a, b) in attempted_pairs:
            continue

        ancestors_a = walk_ancestors(parent_program_for_candidate, a)
        ancestors_b = walk_ancestors(parent_program_for_candidate, b)
        if a in ancestors_b or b in ancestors_a:
            continue

        common = ancestors_a & ancestors_b
        if not common:
            continue

        scores_a = prog_candidate_val_subscores[a]
        scores_b = prog_candidate_val_subscores[b]
        overlap_ids = set(scores_a) & set(scores_b)
        if not overlap_ids:
            continue

        ordered_overlap_ids = sorted(overlap_ids, key=repr)
        a_wins = tuple(
            task_id
            for task_id in ordered_overlap_ids
            if scores_a[task_id] > scores_b[task_id] + eps
        )
        b_wins = tuple(
            task_id
            for task_id in ordered_overlap_ids
            if scores_b[task_id] > scores_a[task_id] + eps
        )
        if len(a_wins) < min_each or len(b_wins) < min_each:
            continue

        oracle_score = sum(
            max(scores_a[task_id], scores_b[task_id]) for task_id in ordered_overlap_ids
        ) / len(ordered_overlap_ids)
        if oracle_score <= current_best_score:
            continue

        score_a = _tracked_score(tracked_scores, a)
        score_b = _tracked_score(tracked_scores, b)
        if score_a > score_b:
            base_parent_id = a
        elif score_b > score_a:
            base_parent_id = b
        else:
            base_parent_id = min(a, b)
        patch_source_parent_id = b if base_parent_id == a else a
        base_wins = a_wins if base_parent_id == a else b_wins
        patch_source_wins = b_wins if base_parent_id == a else a_wins

        common_ancestors = tuple(sorted(common))
        ancestor = sorted(common_ancestors, key=lambda idx: (-_tracked_score(tracked_scores, idx), idx))[0]
        oracle_gain = oracle_score - current_best_score
        balance = min(len(a_wins), len(b_wins)) / max(len(a_wins), len(b_wins))
        weight = max(oracle_gain, eps) * balance
        eligible.append(
            PatchMergePair(
                parent_a_id=a,
                parent_b_id=b,
                base_parent_id=base_parent_id,
                patch_source_parent_id=patch_source_parent_id,
                ancestor=ancestor,
                common_ancestors=common_ancestors,
                oracle_score=oracle_score,
                oracle_gain=oracle_gain,
                base_wins=base_wins,
                patch_source_wins=patch_source_wins,
                weight=weight,
            )
        )

    if not eligible:
        return None
    weights = [item.weight for item in eligible]
    if max(weights) - min(weights) <= eps:
        return min(
            eligible,
            key=lambda item: (
                item.parent_a_id,
                item.parent_b_id,
                item.base_parent_id,
                item.patch_source_parent_id,
                item.ancestor,
            ),
        )
    return rng.choices(
        eligible,
        weights=weights,
        k=1,
    )[0]


def _tracked_score(tracked_scores: Sequence[float], candidate_id: int) -> float:
    if 0 <= candidate_id < len(tracked_scores):
        return tracked_scores[candidate_id]
    return float("-inf")
