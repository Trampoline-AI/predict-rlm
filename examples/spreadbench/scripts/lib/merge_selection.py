"""Pair-selection logic for RlmMergeProposer.

# What the merge proposer is actually doing

The merge proposer asks a specific question: given TWO sibling
candidates A and B that both evolved from a common ancestor but took
different paths (winning different val tasks), can an LM synthesize
their strengths into ONE combined skill?

That question fundamentally needs THREE things in the pool:
  - Candidate A
  - Candidate B
  - Their common ancestor — something BOTH descended from and BOTH
    outperform

This is structurally like a git three-way merge: without the
common-ancestor baseline, you can't tell which parts are A's unique
contribution vs B's.

# Why early iterations almost always hit pair_skipped

If the pool has only {seed, cand 1}, there's only one possible pair,
and seed IS cand 1's ancestor (same line of descent, not diverged
siblings). Rule #2 below rejects it: "neither id1 nor id2 can be an
ancestor of the other." No other pairs → return None → pair_skipped.

To get merge-eligible structure, the optimizer needs this shape:

      seed (cand 0)
      /     \\
   cand 1   cand 2     <- both Pareto-dominators, neither is ancestor
                          of the other, share seed as common ancestor

GEPA's round-robin module_selector + Pareto candidate_selector
naturally produce this, but only once ≥2 non-seed candidates exist
AND at least one descends from a DIFFERENT parent than the others.
A strictly linear chain (seed → cand 1 → cand 2 → ...) has no
siblings, so no merge-eligible triplet.

# This module replaces GEPA's stock pair selector

``gepa.proposer.merge.sample_and_attempt_merge_programs_by_common_predictors``
(at ``gepa/proposer/merge.py:112``) uses a per-component filter
``does_triplet_have_desirable_predictors`` that requires, for each
component, one descendant to match the ancestor verbatim while the
other differs. For single-component skills where BOTH descendants
have edited the only component (our case), no component satisfies
this, and the stock filter returns False. That's why stock merges
are no-ops on our setup — the case we care about (both diverged
from ancestor) is exactly the case stock rejects.

# Eligibility criteria (all must hold)

  1. Both parents are Pareto-front dominators (matches stock behavior,
     narrows to high-quality candidates).
  2. Neither is an ancestor of the other; they share a common ancestor
     via the parent chain (identical to stock).
  3. The common ancestor is dominated by both descendants on aggregate
     score (same as stock's ``filter_ancestors`` check — ancestor
     score ≤ both descendant scores).
  4. MUTUAL divergence on val: each parent strictly wins
     ``>= min_each`` tasks the other loses (the novel criterion; stock
     requires per-component-predictor structure instead). This
     blocks the "one side strictly dominates" case — that's a
     reflective-improvement case, not a merge case.
  5. The two candidates have different text on the target component
     (skip text-identical pairs even if they're aggregate-Pareto-tied).
  6. Normalized ``(min(id1,id2), max(id1,id2), ancestor)`` tuple is not
     already in ``merges_performed`` (attempt-level dedup).

Normalization is applied BEFORE the membership check so the same
unordered pair can't be reselected in reverse order after a prior
rejection.
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
    """BFS over ``parent_list`` starting at ``node``, returning the set of
    all transitive ancestor indices.

    Extracted from GEPA's ``find_common_ancestor_pair`` inner helper
    (``get_ancestors``) so we don't pull in its surrounding
    ``filter_ancestors`` which calls the per-component filter we're
    replacing. Standalone + iterative.
    """
    seen: set[int] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        parents = parent_list[n] if 0 <= n < len(parent_list) else None
        if not parents:
            continue
        for p in parents:
            if p is not None and p not in seen:
                seen.add(p)
                stack.append(p)
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

    Returns ``None`` when no eligible triplet exists in up to
    ``max_attempts`` rng draws. The tuple is always returned with
    ``id1 <= id2`` so callers can dedup-check against ``merges_performed``
    directly without re-normalizing.

    Weighting: when multiple eligible pairs exist, a random draw is
    accepted with probability proportional to
    ``len(a_wins) * len(b_wins)``. This favors pairs with BALANCED mutual
    advantage over asymmetric pairs (where one side strictly dominates
    and the RLM merge has less to synthesize).
    """
    if len(merge_candidates) < 2:
        return None
    if len(parent_program_for_candidate) < 3:
        return None

    dominator_set = set(merge_candidates)
    merges_performed_set = set(merges_performed)

    for _ in range(max_attempts):
        a, b = rng.sample(list(merge_candidates), 2)
        if a == b:
            continue
        # Canonical order
        if b < a:
            a, b = b, a

        # Skip text-identical pairs on the target component.
        if (
            program_candidates[a].get(component_name)
            == program_candidates[b].get(component_name)
        ):
            continue

        # Neither is an ancestor of the other
        anc_a = walk_ancestors(parent_program_for_candidate, a)
        anc_b = walk_ancestors(parent_program_for_candidate, b)
        if a in anc_b or b in anc_a:
            continue

        common = anc_a & anc_b
        if not common:
            continue

        # Filter: ancestor must be dominated by both descendants on
        # aggregate score (matches stock filter_ancestors).
        eligible_ancestors = [
            anc for anc in common
            if tracked_scores[anc] <= tracked_scores[a]
            and tracked_scores[anc] <= tracked_scores[b]
        ]
        if not eligible_ancestors:
            continue

        # Mutual divergence on val
        scores_a = prog_candidate_val_subscores[a]
        scores_b = prog_candidate_val_subscores[b]
        overlap_ids = set(scores_a.keys()) & set(scores_b.keys())
        if len(overlap_ids) < 2 * min_each:
            continue

        a_wins = [
            tid for tid in overlap_ids
            if scores_a[tid] > scores_b[tid] + eps
        ]
        b_wins = [
            tid for tid in overlap_ids
            if scores_b[tid] > scores_a[tid] + eps
        ]
        if len(a_wins) < min_each or len(b_wins) < min_each:
            continue

        # Pick an ancestor (weighted by ancestor tracked score, matching
        # stock behavior — prefer higher-quality ancestors when multiple
        # are eligible).
        if len(eligible_ancestors) == 1:
            ancestor = eligible_ancestors[0]
        else:
            weights = [max(tracked_scores[anc], 1e-9) for anc in eligible_ancestors]
            ancestor = rng.choices(eligible_ancestors, weights=weights, k=1)[0]

        # Dedup on normalized triplet — a was already <= b above.
        if (a, b, ancestor) in merges_performed_set:
            continue

        # Accept with probability weighted by mutual-wins product.
        # Normalize against a reasonable scale so the gate isn't punishingly
        # low on small val sets. Accept outright when both sides hit
        # min_each + a little margin.
        accept_weight = len(a_wins) * len(b_wins)
        # Expected scale: with min_each=3, baseline is 9. Accept any pair
        # deterministically once found — we've already gated hard on all
        # eligibility criteria, so further weighting is diminishing
        # returns and would increase the miss rate.
        _ = accept_weight  # kept for future use / postmortem
        return a, b, ancestor

    return None
