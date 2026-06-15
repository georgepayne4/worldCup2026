"""Tournament simulation. See METHODOLOGY.md §3.3.

v0: provides match-level sampling and group-stage simulation only. The full 48-team
knockout bracket with FIFA tiebreakers lands in v1.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from worldcup2026.simulation.bracket import (
    GROUP_LETTERS,
    assign_thirds,
    bracket_order,
)


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

# Already-played fixtures, keyed by (home, away) exactly as they appear in the
# fixture list, mapping to the observed (home_goals, away_goals). Used to
# condition a simulation on results so far ("re-sim from current state").
KnownResults = dict[tuple[str, str], tuple[int, int]]

# P(home wins a penalty shootout) given the two teams. Used to break knockout
# ties by strength instead of a coin flip.
ShootoutModel = Callable[[str, str], float]


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
    known_results: KnownResults | None = None,
) -> list[GroupStanding]:
    """Simulate one group stage.

    Fixtures present in `known_results` (keyed by the same ``(home, away)``
    tuple) use the observed score instead of being sampled — this is how a
    re-simulation is conditioned on matches already played. The rest are
    sampled as usual.

    Tiebreakers in v0: points → goal difference → goals for. Head-to-head and the
    rest of the FIFA cascade are deferred to v1.
    """
    standings = {t: GroupStanding(team=t) for t in teams}
    for home, away in fixtures:
        if known_results is not None and (home, away) in known_results:
            hg, ag = known_results[(home, away)]
        else:
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


def simulate_knockout_match(
    home: str,
    away: str,
    sampler: ScoreSampler,
    rng: np.random.Generator,
    et_sampler: ScoreSampler | None = None,
    shootout_p: ShootoutModel | None = None,
) -> tuple[str, str]:
    """Return (winner, loser).

    If `et_sampler` is supplied it is called once on a drawn 90 minutes —
    typically the same Dixon-Coles rates scaled by ~1/3 for 30 extra-time
    minutes. If still drawn after ET (or no ET sampler is provided) the tie is
    decided by a penalty shootout: `shootout_p(home, away)` gives P(home wins);
    absent it, an unbiased coin flip (shootouts are close to random, with only a
    mild edge to the stronger side).
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
    p_home = 0.5 if shootout_p is None else shootout_p(home, away)
    return (home, away) if rng.random() < p_home else (away, home)


def _qualifying_thirds(group_standings: dict[str, list[GroupStanding]]) -> list[str]:
    """Group letters of the 8 best third-placed teams, ranked across all groups."""
    ranked = sorted(
        group_standings.items(),
        key=lambda kv: (
            kv[1][2].points,
            kv[1][2].goal_difference,
            kv[1][2].goals_for,
        ),
        reverse=True,
    )
    return [letter for letter, _ in ranked[:8]]


def simulate_world_cup(
    groups: dict[str, list[str]],
    sampler: ScoreSampler,
    rng: np.random.Generator,
    fixtures_fn: Callable[[list[str]], list[tuple[str, str]]] = default_group_fixtures,
    et_sampler: ScoreSampler | None = None,
    known_results: KnownResults | None = None,
    shootout_p: ShootoutModel | None = None,
) -> dict[str, str]:
    """Run one full 48-team simulation; return team -> furthest round reached.

    Format: 12 groups of 4 (labelled A-L), top 2 plus the 8 best third-placed
    teams advance to a 32-team knockout. The knockout uses FIFA's fixed bracket
    (`simulation.bracket`): group winners/runners-up occupy their published R32
    slots and the third-placed teams are routed under Annex C's constraints.
    `known_results` fixes already-played group fixtures (see `simulate_group`)
    so the run reflects the tournament's current state. `shootout_p` decides
    drawn knockout ties by strength (else a coin flip).
    """
    if set(groups) != set(GROUP_LETTERS):
        raise ValueError(
            f"fixed bracket requires the 12 groups labelled A-L; got {sorted(groups)}"
        )

    reached: dict[str, str] = {t: "group_stage" for ts in groups.values() for t in ts}
    group_standings: dict[str, list[GroupStanding]] = {
        name: simulate_group(teams, fixtures_fn(teams), sampler, rng, known_results)
        for name, teams in groups.items()
    }

    group_winners = {letter: s[0].team for letter, s in group_standings.items()}
    runners_up = {letter: s[1].team for letter, s in group_standings.items()}
    qualifying = _qualifying_thirds(group_standings)
    thirds = {letter: group_standings[letter][2].team for letter in qualifying}
    third_assignment = assign_thirds(qualifying)

    current = bracket_order(group_winners, runners_up, thirds, third_assignment)
    for t in current:
        reached[t] = "round_of_32"

    for round_name in ROUND_LABELS[1:]:
        winners = []
        for i in range(0, len(current), 2):
            winner, _ = simulate_knockout_match(
                current[i], current[i + 1], sampler, rng,
                et_sampler=et_sampler, shootout_p=shootout_p,
            )
            winners.append(winner)
        for w in winners:
            reached[w] = round_name
        current = winners

    return reached


def cumulative_round_probabilities(
    probs: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Transform P(furthest round = X) into P(reached round X or further).

    Same keys, same teams. P(group_stage) is always 1.0; P(champion) equals
    the original 'champion' value. This is the form bet markets price —
    "to reach the final" is P(reach final or further) = P(final) + P(champion).
    """
    labels = ("group_stage", *ROUND_LABELS)
    return {
        team: {
            label: sum(team_probs[r] for r in labels[i:])
            for i, label in enumerate(labels)
        }
        for team, team_probs in probs.items()
    }


def monte_carlo_world_cup(
    groups: dict[str, list[str]],
    sampler: ScoreSampler,
    n_runs: int = 20_000,
    seed: int = 42,
    fixtures_fn: Callable[[list[str]], list[tuple[str, str]]] = default_group_fixtures,
    et_sampler: ScoreSampler | None = None,
    known_results: KnownResults | None = None,
    shootout_p: ShootoutModel | None = None,
) -> dict[str, dict[str, float]]:
    """Run N simulations; return P(furthest round = X) per team.

    `known_results` conditions every run on the same already-played fixtures —
    the basis for "re-sim from current state". `shootout_p` decides drawn
    knockout ties by strength.
    """
    rng = np.random.default_rng(seed)
    all_teams = [t for ts in groups.values() for t in ts]
    labels = ("group_stage", *ROUND_LABELS)
    counts: dict[str, dict[str, int]] = {t: dict.fromkeys(labels, 0) for t in all_teams}
    for _ in range(n_runs):
        for t, r in simulate_world_cup(
            groups, sampler, rng, fixtures_fn, et_sampler=et_sampler,
            known_results=known_results, shootout_p=shootout_p,
        ).items():
            counts[t][r] += 1
    return {
        t: {r: c / n_runs for r, c in round_counts.items()}
        for t, round_counts in counts.items()
    }
