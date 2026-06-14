"""Tournament simulation. See METHODOLOGY.md §3.3.

v0: provides match-level sampling and group-stage simulation only. The full 48-team
knockout bracket with FIFA tiebreakers lands in v1.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationConfig:
    n_runs: int = 20_000
    seed: int = 42
    max_goals: int = 10


@dataclass
class GroupStanding:
    team: str
    played: int = 0
    won: int = 0
    drawn: int = 0
    lost: int = 0
    goals_for: int = 0
    goals_against: int = 0

    @property
    def points(self) -> int:
        return 3 * self.won + self.drawn

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


ScoreSampler = Callable[[str, str, np.random.Generator], tuple[int, int]]


def sample_score_from_matrix(matrix: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    """Sample (home_goals, away_goals) from a joint score-matrix distribution."""
    n = matrix.shape[0]
    flat = matrix.ravel()
    idx = rng.choice(flat.size, p=flat / flat.sum())
    return int(idx // n), int(idx % n)


def simulate_group(
    teams: list[str],
    fixtures: list[tuple[str, str]],
    sampler: ScoreSampler,
    rng: np.random.Generator,
) -> list[GroupStanding]:
    """Simulate one group stage.

    Tiebreakers in v0: points → goal difference → goals for. Head-to-head and the
    rest of the FIFA cascade are deferred to v1.
    """
    standings = {t: GroupStanding(team=t) for t in teams}
    for home, away in fixtures:
        hg, ag = sampler(home, away, rng)
        h = standings[home]
        a = standings[away]
        h.played += 1
        a.played += 1
        h.goals_for += hg
        h.goals_against += ag
        a.goals_for += ag
        a.goals_against += hg
        if hg > ag:
            h.won += 1
            a.lost += 1
        elif hg < ag:
            a.won += 1
            h.lost += 1
        else:
            h.drawn += 1
            a.drawn += 1
    return sorted(
        standings.values(),
        key=lambda s: (s.points, s.goal_difference, s.goals_for),
        reverse=True,
    )


def monte_carlo_group(
    teams: list[str],
    fixtures: list[tuple[str, str]],
    sampler: ScoreSampler,
    n_runs: int = 20_000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Run N group-stage simulations; return per-team finish-position probabilities."""
    rng = np.random.default_rng(seed)
    counts: dict[str, list[int]] = {t: [0] * len(teams) for t in teams}
    for _ in range(n_runs):
        standings = simulate_group(teams, fixtures, sampler, rng)
        for pos, s in enumerate(standings):
            counts[s.team][pos] += 1
    return {
        t: {f"finish_{pos + 1}": c / n_runs for pos, c in enumerate(positions)}
        for t, positions in counts.items()
    }


def simulate_tournament(*args, **kwargs):
    """Full 48-team simulator with knockout bracket. Lands in v1."""
    raise NotImplementedError


def simulate_knockout(*args, **kwargs):
    """Knockout sim with ET and penalties. Lands in v1."""
    raise NotImplementedError
