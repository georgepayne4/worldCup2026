"""Suggest high-accuracy accumulators across the remaining World Cup fixtures.

Fits the tuned Dixon-Coles model, prices every still-to-play fixture, and builds
a few non-overlapping accas (one leg per match) from the model's most confident
selections. Pure model probabilities — no market feed needed — so it answers
"give me a safe 2–3 game acca to place".

Each leg shows the model's win probability and **fair odds**. The edge rule: only
back a leg/acca where the bookmaker's offered odds are *longer* than fair.

Run:  python scripts/build_accas.py [--legs 3] [--accas 3] [--min-prob 0.6]
"""

from __future__ import annotations

import argparse

import pandas as pd

from worldcup2026.betting.acca import suggest_accas
from worldcup2026.betting.markets import match_market_table
from worldcup2026.data.loaders import (
    load_fixtures_2026,
    load_international_results,
)
from worldcup2026.evaluation.backtest import fit_window
from worldcup2026.models.dixon_coles import match_rates, score_matrix

TODAY = "2026-06-15"
HOSTS = frozenset({"Mexico", "United States", "Canada"})
HOST_ADVANTAGE = 0.15


def describe(row: pd.Series) -> str:
    """Human-readable leg description."""
    home, away, sel = row["home_team"], row["away_team"], row["selection"]
    market = row["market"]
    if market == "h2h":
        return {"Home": f"{home} to win", "Draw": "Draw", "Away": f"{away} to win"}[sel]
    if market == "double_chance":
        return {
            "1X": f"{home} or Draw",
            "12": f"{home} or {away} (not Draw)",
            "X2": f"Draw or {away}",
        }[sel]
    if market == "totals":
        return f"{sel} {row['line']:.1f} goals"
    if market == "btts":
        return f"Both teams to score: {sel}"
    return f"{market} {sel}"


def candidate_legs(fitted, fixtures, max_goals: int) -> pd.DataFrame:
    """Model market table for every remaining fixture, stacked."""
    remaining = fixtures[~fixtures["played"]]
    tables = []
    for fx in remaining.itertuples(index=False):
        if fx.home not in fitted.attack or fx.away not in fitted.attack:
            continue
        home_boost = HOST_ADVANTAGE if fx.home in HOSTS else 0.0
        away_boost = HOST_ADVANTAGE if fx.away in HOSTS else 0.0
        lam, mu = match_rates(
            fitted, fx.home, fx.away, neutral=True,
            home_boost=home_boost, away_boost=away_boost,
        )
        matrix = score_matrix(lam, mu, fitted.rho, max_goals)
        tables.append(
            match_market_table(
                matrix, match_id=f"{fx.home} v {fx.away}",
                home_team=fx.home, away_team=fx.away,
            )
        )
    return pd.concat(tables, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legs", type=int, default=3, help="legs per acca")
    ap.add_argument("--accas", type=int, default=3, help="how many accas to suggest")
    ap.add_argument("--min-prob", type=float, default=0.60, help="min per-leg win prob")
    ap.add_argument(
        "--max-prob",
        type=float,
        default=0.90,
        help="max per-leg win prob (trims near-locks that add no return)",
    )
    ap.add_argument("--window-years", type=float, default=10.0)
    ap.add_argument("--half-life", type=float, default=1095.0)
    ap.add_argument("--max-goals", type=int, default=8)
    args = ap.parse_args()

    print(f"Fitting model (window {args.window_years:.0f}y, half-life {args.half_life:.0f}d)...")
    results = load_international_results()
    fitted = fit_window(results, TODAY, args.window_years, args.half_life)

    fixtures = load_fixtures_2026()
    n_remaining = int((~fixtures["played"]).sum())
    print(f"Pricing {n_remaining} remaining fixtures and building accas...\n")

    candidates = candidate_legs(fitted, fixtures, args.max_goals)
    accas = suggest_accas(
        candidates,
        n_accas=args.accas,
        n_legs=args.legs,
        mode="accuracy",
        min_prob=args.min_prob,
        max_prob=args.max_prob,
    )

    if not accas:
        print("No legs cleared the confidence filter — lower --min-prob.")
        return

    for n, acca in enumerate(accas, 1):
        print(f"Acca {n}: {acca.summary()}")
        for leg in acca.legs.itertuples(index=False):
            row = pd.Series(leg._asdict())
            print(
                f"   {row['match_id']:<28} {describe(row):<28} "
                f"{row['prob']:>5.0%}  (fair {row['fair_odds']:.2f})"
            )
        print()

    print(
        "Edge rule: back a leg only if the bookmaker price is LONGER than fair. "
        "Combined fair odds is the break-even price for the whole acca."
    )


if __name__ == "__main__":
    main()
