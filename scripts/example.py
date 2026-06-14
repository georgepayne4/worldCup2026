"""End-to-end demo: synthetic match history -> Dixon-Coles fit -> Monte Carlo WC.

Run:  python scripts/example.py

Everything is synthetic — no external data. Useful as a smoke check that the
modules compose, and as a template for the real pipeline once data lands.
"""

from __future__ import annotations

import numpy as np

from worldcup2026.models.dixon_coles import (
    DixonColesParams,
    fit,
    match_rates,
    score_matrix,
)
from worldcup2026.simulation.tournament import (
    monte_carlo_world_cup,
    sample_score_from_matrix,
)

N_TEAMS = 48
N_HISTORY_ROUNDS = 1            # round-robin repeats for synthetic history
N_MONTE_CARLO_RUNS = 500


def main() -> None:
    rng = np.random.default_rng(42)
    teams = [f"T{i:02d}" for i in range(N_TEAMS)]

    true_params = DixonColesParams(
        attack={t: float(s) for t, s in zip(teams, rng.normal(0.0, 0.3, N_TEAMS))},
        defence={t: float(s) for t, s in zip(teams, rng.normal(0.0, 0.3, N_TEAMS))},
        home_advantage=0.0,
        rho=-0.08,
    )

    print(f"Generating {N_HISTORY_ROUNDS} round-robin(s) of synthetic history...")
    matches = []
    for _ in range(N_HISTORY_ROUNDS):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                lam, mu = match_rates(true_params, home, away, neutral=True)
                m = score_matrix(lam, mu, true_params.rho, max_goals=8)
                hg, ag = sample_score_from_matrix(m, rng)
                matches.append(
                    {"home": home, "away": away, "home_goals": hg, "away_goals": ag}
                )
    print(f"  {len(matches)} matches")

    print("Fitting Dixon-Coles by MLE...")
    fitted = fit(matches, teams=teams)
    print(f"  rho={fitted.rho:+.3f}  home_advantage={fitted.home_advantage:+.3f}")

    print("Pre-computing per-fixture score matrices...")
    normal_mats: dict[tuple[str, str], np.ndarray] = {}
    et_mats: dict[tuple[str, str], np.ndarray] = {}
    for home in teams:
        for away in teams:
            if home == away:
                continue
            lam, mu = match_rates(fitted, home, away, neutral=True)
            normal_mats[(home, away)] = score_matrix(lam, mu, fitted.rho)
            et_mats[(home, away)] = score_matrix(lam / 3.0, mu / 3.0, fitted.rho)

    def sampler(home: str, away: str, r: np.random.Generator) -> tuple[int, int]:
        return sample_score_from_matrix(normal_mats[(home, away)], r)

    def et_sampler(home: str, away: str, r: np.random.Generator) -> tuple[int, int]:
        return sample_score_from_matrix(et_mats[(home, away)], r)

    groups = {chr(65 + g): teams[g * 4 : (g + 1) * 4] for g in range(12)}

    print(f"Running {N_MONTE_CARLO_RUNS} Monte Carlo tournaments...")
    probs = monte_carlo_world_cup(
        groups, sampler, n_runs=N_MONTE_CARLO_RUNS, seed=1, et_sampler=et_sampler
    )

    leaderboard = sorted(probs.items(), key=lambda kv: kv[1]["champion"], reverse=True)
    print(f"\n{'Team':<6} {'Champ':>7} {'Final+':>7} {'SF+':>7} {'QF+':>7}")
    print("-" * 39)
    for team, p in leaderboard[:10]:
        champ = p["champion"]
        final_plus = champ + p["final"]
        sf_plus = final_plus + p["semi_final"]
        qf_plus = sf_plus + p["quarter_final"]
        print(f"{team:<6} {champ:>7.1%} {final_plus:>7.1%} {sf_plus:>7.1%} {qf_plus:>7.1%}")


if __name__ == "__main__":
    main()
