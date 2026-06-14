"""Elo ratings with international-football adjustments.

See METHODOLOGY.md §3.1 for the design (K-weights by competition, GD multiplier,
neutral-venue handling).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

MatchImportance = Literal["friendly", "qualifier", "confederation", "world_cup"]


@dataclass
class EloConfig:
    base_k: float = 30.0
    home_advantage: float = 65.0
    initial_rating: float = 1500.0
    k_weights: dict[str, float] = field(
        default_factory=lambda: {
            "friendly": 1.0,
            "qualifier": 1.5,
            "confederation": 2.0,
            "world_cup": 2.5,
        }
    )


def expected_score(rating_a: float, rating_b: float) -> float:
    """Standard Elo expected score for A vs B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def goal_difference_multiplier(home_goals: int, away_goals: int) -> float:
    """World Football Elo Ratings GD multiplier — diminishing returns above 2 goals."""
    gd = abs(home_goals - away_goals)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


class EloRatings:
    """Mutable Elo ratings store. Pure-Python; no pandas required."""

    def __init__(self, config: EloConfig | None = None):
        self.config = config or EloConfig()
        self._ratings: dict[str, float] = defaultdict(lambda: self.config.initial_rating)

    def get(self, team: str) -> float:
        return self._ratings[team]

    def set(self, team: str, rating: float) -> None:
        self._ratings[team] = rating

    def update(
        self,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
        importance: MatchImportance = "friendly",
        neutral: bool = False,
    ) -> tuple[float, float]:
        """Apply a single match result; return the new (home, away) ratings."""
        hr = self._ratings[home]
        ar = self._ratings[away]
        ha = 0.0 if neutral else self.config.home_advantage
        exp_home = expected_score(hr + ha, ar)

        if home_goals > away_goals:
            score_home = 1.0
        elif home_goals < away_goals:
            score_home = 0.0
        else:
            score_home = 0.5

        k = self.config.base_k * self.config.k_weights.get(importance, 1.0)
        g = goal_difference_multiplier(home_goals, away_goals)
        delta = k * g * (score_home - exp_home)

        self._ratings[home] = hr + delta
        self._ratings[away] = ar - delta
        return self._ratings[home], self._ratings[away]

    def update_batch(self, matches: Iterable[dict]) -> None:
        """Apply a series of matches in order.

        Each dict must contain: home, away, home_goals, away_goals.
        Optional: importance, neutral.
        """
        for m in matches:
            self.update(
                home=m["home"],
                away=m["away"],
                home_goals=m["home_goals"],
                away_goals=m["away_goals"],
                importance=m.get("importance", "friendly"),
                neutral=m.get("neutral", False),
            )

    def as_dict(self) -> dict[str, float]:
        return dict(self._ratings)
