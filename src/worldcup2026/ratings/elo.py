"""Elo ratings with international-football adjustments.

See METHODOLOGY.md §3.1 for the design (K-weights by competition, goal-difference
multiplier, confederation prior, neutral-venue handling).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EloConfig:
    base_k: float = 30.0
    home_advantage: float = 65.0
    initial_rating: float = 1500.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Standard Elo expected score for A vs B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_ratings(*args, **kwargs):
    """Update ratings from a batch of match results. Implementation lands in v1."""
    raise NotImplementedError
