from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import comb


@dataclass(frozen=True)
class AcceptanceDecision:
    accepted: bool
    reason: str
    dense_delta: float
    hard_wins: int
    hard_losses: int
    hard_flip_p_value: float


def should_accept_reflective_candidate(
    *,
    before_scores: Sequence[float],
    after_scores: Sequence[float],
    dense_loss_floor: float = -0.01,
    hard_flip_p_threshold: float = 0.40,
    perfect_score: float = 1.0,
    eps: float = 1e-9,
) -> AcceptanceDecision:
    if len(before_scores) != len(after_scores):
        raise ValueError("before_scores and after_scores must have the same length")

    dense_delta = _mean(after_scores) - _mean(before_scores)
    hard_wins, hard_losses = _hard_flips(before_scores, after_scores, perfect_score, eps)
    p_value = _two_sided_sign_test_p_value(hard_wins, hard_losses)

    if dense_delta > 0.0:
        return AcceptanceDecision(
            accepted=True,
            reason="dense_improved",
            dense_delta=dense_delta,
            hard_wins=hard_wins,
            hard_losses=hard_losses,
            hard_flip_p_value=p_value,
        )

    accepted = dense_delta >= dense_loss_floor and hard_wins > hard_losses and p_value <= hard_flip_p_threshold
    return AcceptanceDecision(
        accepted=accepted,
        reason="hard_flip_signal" if accepted else "not_improved",
        dense_delta=dense_delta,
        hard_wins=hard_wins,
        hard_losses=hard_losses,
        hard_flip_p_value=p_value,
    )


def _mean(scores: Sequence[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0


def _hard_flips(
    before_scores: Sequence[float],
    after_scores: Sequence[float],
    perfect_score: float,
    eps: float,
) -> tuple[int, int]:
    wins = 0
    losses = 0
    for before, after in zip(before_scores, after_scores, strict=True):
        before_hard = before >= perfect_score - eps
        after_hard = after >= perfect_score - eps
        if after_hard and not before_hard:
            wins += 1
        elif before_hard and not after_hard:
            losses += 1
    return wins, losses


def _two_sided_sign_test_p_value(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    less_frequent_flips = min(wins, losses)
    one_tail = sum(comb(trials, k) for k in range(less_frequent_flips + 1)) / (2**trials)
    return min(1.0, 2 * one_tail)
