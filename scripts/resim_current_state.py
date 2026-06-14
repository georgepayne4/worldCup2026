"""Re-simulate the 2026 World Cup from its current state.

Pipeline (all real data — see METHODOLOGY.md):

1. Load historical international results (``data/raw/international_results.csv``).
2. Fit Dixon-Coles by MLE with exponential time-decay weighting and
   neutral-venue handling.
3. Load the WC2026 fixtures + groups and the results played so far.
4. Condition the Monte Carlo on those played results and simulate the rest of
   the tournament from neutral-venue score distributions.
5. Print updated champion / round-reached / group-advancement probabilities.

Run:  python scripts/resim_current_state.py [--n-runs N] [--since YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd

from worldcup2026.data.loaders import (
    PROCESSED_DIR,
    derive_groups,
    load_fixtures_2026,
    load_international_results,
)
from worldcup2026.models.dixon_coles import (
    fit,
    match_rates,
    score_matrix,
    time_decay_weights,
)
from worldcup2026.simulation.tournament import (
    cumulative_round_probabilities,
    monte_carlo_world_cup,
)

# "Today" — the tournament is mid-group-stage. Used as the time-decay reference
# and to date the report.
TODAY = date(2026, 6, 14)

# 2026 co-hosts (dataset spellings). They get a small home-style residual even
# though World Cup venues are otherwise treated as neutral.
HOSTS = frozenset({"Mexico", "United States", "Canada"})


def fit_ratings(since: str, half_life_days: float, max_goals: int):
    """Fit Dixon-Coles on historical results up to TODAY with time decay."""
    results = load_international_results()
    window = results[
        (results["date"] >= since) & (results["date"] <= str(TODAY))
    ].copy()
    matches = [
        {
            "home": r.home,
            "away": r.away,
            "home_goals": int(r.home_goals),
            "away_goals": int(r.away_goals),
            "neutral": bool(r.neutral),
        }
        for r in window.itertuples(index=False)
    ]
    weights = time_decay_weights(
        window["date"].to_numpy(), np.datetime64(TODAY, "D"), half_life_days
    )
    print(
        f"Fitting Dixon-Coles on {len(matches):,} matches "
        f"({since} -> {TODAY}, half-life {half_life_days:.0f}d)..."
    )
    teams = sorted(set(window["home"]) | set(window["away"]))
    fitted = fit(matches, teams=teams, weights=weights)
    print(
        f"  teams={len(teams)}  rho={fitted.rho:+.3f}  "
        f"home_advantage={fitted.home_advantage:+.3f}"
    )
    return fitted


def build_samplers(fitted, teams, rho, max_goals, hosts, host_advantage):
    """Pre-compute a sampler over every ordered pair of WC teams.

    Matches are simulated at neutral venues, except that a host nation
    (`hosts`) carries a `host_advantage` log-rate residual whenever it plays —
    a small home-style edge for crowd/familiarity that the methodology zeroes
    for everyone else (see METHODOLOGY §3.1, §8).

    Each fixture's score matrix is flattened to a cumulative distribution once;
    sampling is then a single ``random()`` + ``searchsorted`` — far cheaper than
    ``rng.choice`` per call across millions of draws.
    """
    n = max_goals + 1
    normal_cdf: dict[tuple[str, str], np.ndarray] = {}
    et_cdf: dict[tuple[str, str], np.ndarray] = {}
    for home in teams:
        for away in teams:
            if home == away:
                continue
            home_boost = host_advantage if home in hosts else 0.0
            away_boost = host_advantage if away in hosts else 0.0
            lam, mu = match_rates(
                fitted, home, away, neutral=True,
                home_boost=home_boost, away_boost=away_boost,
            )
            m = score_matrix(lam, mu, rho, max_goals)
            normal_cdf[(home, away)] = np.cumsum(m.ravel())
            et = score_matrix(lam / 3.0, mu / 3.0, rho, max_goals)
            et_cdf[(home, away)] = np.cumsum(et.ravel())

    def _draw(cdf, r):
        idx = int(np.searchsorted(cdf, r.random() * cdf[-1], side="right"))
        return idx // n, idx % n

    def sampler(home, away, r):
        return _draw(normal_cdf[(home, away)], r)

    def et_sampler(home, away, r):
        return _draw(et_cdf[(home, away)], r)

    return sampler, et_sampler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-runs", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--window-years",
        type=float,
        default=10.0,
        help="training window length (tuned via scripts/backtest_window.py)",
    )
    ap.add_argument(
        "--since",
        default=None,
        help="explicit earliest fit date; overrides --window-years",
    )
    ap.add_argument("--half-life", type=float, default=1095.0, help="decay half-life (days)")
    ap.add_argument("--max-goals", type=int, default=8)
    ap.add_argument(
        "--host-advantage",
        type=float,
        default=0.15,
        help="host log-rate residual for Mexico/USA/Canada (0 disables)",
    )
    ap.add_argument("--top", type=int, default=20, help="rows to print")
    args = ap.parse_args()

    since = args.since or str(
        (pd.Timestamp(TODAY) - pd.Timedelta(days=round(args.window_years * 365.25))).date()
    )
    fitted = fit_ratings(since, args.half_life, args.max_goals)

    fixtures = load_fixtures_2026()
    groups = derive_groups(fixtures)
    all_teams = [t for ts in groups.values() for t in ts]
    missing = [t for t in all_teams if t not in fitted.attack]
    if missing:
        raise SystemExit(f"no fitted rating for: {missing} — widen --since")

    # Real per-group fixtures (with the actual home/away orientation).
    group_of = {t: name for name, ts in groups.items() for t in ts}
    group_fixtures: dict[str, list[tuple[str, str]]] = {name: [] for name in groups}
    for home, away in zip(fixtures["home"], fixtures["away"], strict=True):
        group_fixtures[group_of[home]].append((home, away))
    fixtures_by_teamset = {
        frozenset(ts): group_fixtures[name] for name, ts in groups.items()
    }

    def fixtures_fn(teams):
        return fixtures_by_teamset[frozenset(teams)]

    # Already-played results -> condition the simulation on them.
    played = fixtures[fixtures["played"]]
    known_results = {
        (r.home, r.away): (int(r.home_goals), int(r.away_goals))
        for r in played.itertuples(index=False)
    }
    print(f"\nGroup stage: {len(played)} of {len(fixtures)} matches played.")

    print(
        f"Host residual: +{args.host_advantage:.2f} log-rate for "
        f"{', '.join(sorted(HOSTS))}."
    )
    sampler, et_sampler = build_samplers(
        fitted, all_teams, fitted.rho, args.max_goals, HOSTS, args.host_advantage
    )

    print(f"Running {args.n_runs:,} Monte Carlo tournaments from current state...")
    probs = monte_carlo_world_cup(
        groups,
        sampler,
        n_runs=args.n_runs,
        seed=args.seed,
        fixtures_fn=fixtures_fn,
        et_sampler=et_sampler,
        known_results=known_results,
    )
    cum = cumulative_round_probabilities(probs)

    leaderboard = sorted(
        all_teams, key=lambda t: probs[t]["champion"], reverse=True
    )

    # Persist the full table (all 48 teams) for downstream/betting use.
    out_path = PROCESSED_DIR / f"wc2026_resim_{TODAY}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rounds = ["round_of_32", "round_of_16", "quarter_final", "semi_final", "final"]
    rows = []
    for t in leaderboard:
        row = {"team": t, "group": group_of[t], "champion": probs[t]["champion"]}
        row.update({f"{r}_plus": cum[t][r] for r in rounds})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nWrote full 48-team table -> {out_path.relative_to(PROCESSED_DIR.parents[1])}")
    print(f"\nUpdated probabilities as of {TODAY} ({args.n_runs:,} sims)\n")
    hdr = f"{'Team':<24}{'Grp':>4}{'Champ':>8}{'Final+':>8}{'SF+':>8}{'QF+':>8}{'Adv':>8}"
    print(hdr)
    print("-" * len(hdr))
    for t in leaderboard[: args.top]:
        print(
            f"{t:<24}{group_of[t]:>4}"
            f"{probs[t]['champion']:>8.1%}"
            f"{cum[t]['final']:>8.1%}"
            f"{cum[t]['semi_final']:>8.1%}"
            f"{cum[t]['quarter_final']:>8.1%}"
            f"{cum[t]['round_of_32']:>8.1%}"
        )


if __name__ == "__main__":
    main()
