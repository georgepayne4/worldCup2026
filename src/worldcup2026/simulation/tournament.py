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


ROUND_LABELS = (
    "round_of_32",
    "round_of_16",
    "quarter_final",
    "semi_final",
    "final",
    "champion",
)


def default_group_fixtures(teams: list[str]) -> list[tuple[str, str]]:
    """Single round-robin: 6 fixtures for 4 teams."""
    return [(teams[i], teams[j]) for i in range(len(teams)) for j in range(i + 1, len(teams))]


def _bracket_seed_order(n: int) -> list[int]:
    """Standard 1-indexed tournament bracket order (1 vs n, 2 vs n-1, ...) arranged
    so higher seeds meet later rounds. n must be a power of two."""
    order = [1]
    while len(order) < n:
        size = 2 * len(order)
        new = []
        for s in order:
            new.append(s)
            new.append(size + 1 - s)
        order = new
    return order


def simulate_knockout_match(
    home: str,
    away: str,
    sampler: ScoreSampler,
    rng: np.random.Generator,
    et_sampler: ScoreSampler | None = None,
) -> tuple[str, str]:
    """Return (winner, loser).

    If `et_sampler` is supplied it is called once on a drawn 90 minutes —
    typically the same Dixon-Coles rates scaled by ~1/3 for 30 extra-time
    minutes. If still drawn after ET (or no ET sampler is provided) the tie
    is broken by an unbiased coin flip — v0 proxy for penalties; v1 will
    weight by strength.
    """
    hg, ag = sampler(home, away, rng)
    if hg > ag:
        return home, away
    if ag > hg:
        return away, home
    if et_sampler is not None:
        et_hg, et_ag = et_sampler(home, away, rng)
        if et_hg > et_ag:
            return home, away
        if et_ag > et_hg:
            return away, home
    return (home, away) if rng.random() < 0.5 else (away, home)


def _best_third_placed(
    group_results: dict[str, list[GroupStanding]], k: int = 8
) -> list[GroupStanding]:
    thirds = [standings[2] for standings in group_results.values()]
    thirds.sort(key=lambda s: (s.points, s.goal_difference, s.goals_for), reverse=True)
    return thirds[:k]


def simulate_world_cup(
    groups: dict[str, list[str]],
    sampler: ScoreSampler,
    rng: np.random.Generator,
    fixtures_fn: Callable[[list[str]], list[tuple[str, str]]] = default_group_fixtures,
    et_sampler: ScoreSampler | None = None,
) -> dict[str, str]:
    """Run one full 48-team simulation; return team -> furthest round reached.

    Format: 12 groups of 4, top 2 plus the 8 best third-placed teams advance to a
    32-team knockout.

    v0 simplifications: knockout pairings use a generic 1-vs-32 seeded bracket
    from overall group-stage performance rather than FIFA's published pairing
    rules; drawn knockout matches resolve via coin flip.
    """
    if len(groups) != 12:
        raise ValueError(f"expected 12 groups, got {len(groups)}")

    reached: dict[str, str] = {t: "group_stage" for ts in groups.values() for t in ts}
    group_standings: dict[str, list[GroupStanding]] = {
        name: simulate_group(teams, fixtures_fn(teams), sampler, rng)
        for name, teams in groups.items()
    }

    advancing: list[GroupStanding] = []
    for standings in group_standings.values():
        advancing.append(standings[0])
        advancing.append(standings[1])
    advancing.extend(_best_third_placed(group_standings, k=8))

    seeded = sorted(
        advancing,
        key=lambda s: (s.points, s.goal_difference, s.goals_for),
        reverse=True,
    )
    teams_in_r32 = [s.team for s in seeded]
    for t in teams_in_r32:
        reached[t] = "round_of_32"

    order = _bracket_seed_order(len(teams_in_r32))
    current = [teams_in_r32[s - 1] for s in order]

    for round_name in ROUND_LABELS[1:]:
        winners = []
        for i in range(0, len(current), 2):
            winner, _ = simulate_knockout_match(
                current[i], current[i + 1], sampler, rng, et_sampler=et_sampler
            )
            winners.append(winner)
        for w in winners:
            reached[w] = round_name
        current = winners

    return reached


def monte_carlo_world_cup(
    groups: dict[str, list[str]],
    sampler: ScoreSampler,
    n_runs: int = 20_000,
    seed: int = 42,
    fixtures_fn: Callable[[list[str]], list[tuple[str, str]]] = default_group_fixtures,
    et_sampler: ScoreSampler | None = None,
) -> dict[str, dict[str, float]]:
    """Run N simulations; return P(furthest round = X) per team."""
    rng = np.random.default_rng(seed)
    all_teams = [t for ts in groups.values() for t in ts]
    labels = ("group_stage", *ROUND_LABELS)
    counts: dict[str, dict[str, int]] = {t: dict.fromkeys(labels, 0) for t in all_teams}
    for _ in range(n_runs):
        for t, r in simulate_world_cup(
            groups, sampler, rng, fixtures_fn, et_sampler=et_sampler
        ).items():
            counts[t][r] += 1
    return {
        t: {r: c / n_runs for r, c in round_counts.items()}
        for t, round_counts in counts.items()
    }
