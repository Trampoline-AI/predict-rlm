"""Pair-selection logic for RLM merge synthesis.

The merge proposer needs two sibling candidates and their common ancestor. The
siblings must each win some validation examples the other loses; otherwise the
case is a normal reflective-improvement problem, not a merge problem.

Eligibility requires:
- both parents are merge candidates selected from the Pareto/frontier machinery,
- neither parent is an ancestor of the other,
- they share an ancestor that both descendants dominate on aggregate score,
- both parents have distinct text on the target component,
- each parent strictly wins at least ``min_each`` overlapping validation tasks,
- the normalized ``(min(parent_a, parent_b), max(...), ancestor)`` triplet has
  not already been attempted.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

ProgramIdx = int


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


def pick_merge_triplet(
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
    max_attempts: int = 20,
) -> tuple[int, int, int] | None:
    """Select ``(id1, id2, ancestor)`` for RLM merge synthesis.

    Eligible sampled pairs are weighted by ``len(a_wins) * len(b_wins)`` so
    balanced complementary siblings are preferred over barely-divergent pairs.
    """
    if len(merge_candidates) < 2:
        return None
    if len(parent_program_for_candidate) < 3:
        return None

    merges_performed_set = set(merges_performed)

    eligible: list[tuple[tuple[int, int, int], int]] = []
    for _ in range(max_attempts):
        a, b = rng.sample(list(merge_candidates), 2)
        if a == b:
            continue
        if b < a:
            a, b = b, a

        if program_candidates[a].get(component_name) == program_candidates[b].get(component_name):
            continue

        ancestors_a = walk_ancestors(parent_program_for_candidate, a)
        ancestors_b = walk_ancestors(parent_program_for_candidate, b)
        if a in ancestors_b or b in ancestors_a:
            continue

        common = ancestors_a & ancestors_b
        if not common:
            continue

        eligible_ancestors = [
            ancestor
            for ancestor in common
            if tracked_scores[ancestor] <= tracked_scores[a]
            and tracked_scores[ancestor] <= tracked_scores[b]
        ]
        if not eligible_ancestors:
            continue

        scores_a = prog_candidate_val_subscores[a]
        scores_b = prog_candidate_val_subscores[b]
        overlap_ids = set(scores_a) & set(scores_b)
        if len(overlap_ids) < 2 * min_each:
            continue

        a_wins = [task_id for task_id in overlap_ids if scores_a[task_id] > scores_b[task_id] + eps]
        b_wins = [task_id for task_id in overlap_ids if scores_b[task_id] > scores_a[task_id] + eps]
        if len(a_wins) < min_each or len(b_wins) < min_each:
            continue

        if len(eligible_ancestors) == 1:
            ancestor = eligible_ancestors[0]
        else:
            weights = [max(tracked_scores[ancestor], 1e-9) for ancestor in eligible_ancestors]
            ancestor = rng.choices(eligible_ancestors, weights=weights, k=1)[0]

        if (a, b, ancestor) in merges_performed_set:
            continue

        eligible.append(((a, b, ancestor), len(a_wins) * len(b_wins)))

    if not eligible:
        return None
    triplets, weights = zip(*eligible, strict=True)
    return rng.choices(list(triplets), weights=list(weights), k=1)[0]
