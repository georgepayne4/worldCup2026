"""Find same-game-multi value the market misprices via independence.

This is the project's actual edge thesis. Each fixture's grid is blended to the
**market consensus** marginals, so the only thing left that is *ours* is the
correlation between legs. Soft books price same-game-multi legs as if
independent; where the true joint probability exceeds that independence product
(``corr_ratio > 1``), backing the multi at such a book is +EV. ``corr_edge =
corr_ratio - 1`` is that edge before the book's margin.

Because marginals are the market's, this is stripped of the model's 1X2
miscalibration (Gate G1) — the surviving signal is pure correlation.

Run:  python scripts/find_multis.py --odds data/raw/odds/odds_*.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from worldcup2026.betting.blend import blend_to_market
from worldcup2026.betting.markets import rank_same_game_multis, same_game_multi_pairs
from worldcup2026.data.loaders import load_fixtures_2026, load_international_results
from worldcup2026.data.odds import (
    apply_team_aliases,
    consensus_probabilities,
    load_snapshot,
    market_targets,
    normalize_h2h_selections,
)
from worldcup2026.evaluation.backtest import fit_window
from worldcup2026.models.dixon_coles import match_rates, score_matrix

TODAY = "2026-06-15"
HOSTS = frozenset({"Mexico", "United States", "Canada"})
HOST_ADVANTAGE = 0.15


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--odds", required=True, help="normalized odds snapshot CSV")
    ap.add_argument("--total-line", type=float, default=2.5)
    ap.add_argument("--min-edge", type=float, default=0.03, help="min correlation edge to show")
    ap.add_argument("--min-prob", type=float, default=0.10, help="skip very unlikely multis")
    ap.add_argument("--max-goals", type=int, default=8)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    print("Fitting model and blending each fixture to market consensus...")
    results = load_international_results()
    fitted = fit_window(results, TODAY, 10.0, 1095.0)
    fixtures = load_fixtures_2026()
    odds = normalize_h2h_selections(apply_team_aliases(load_snapshot(args.odds)))
    consensus = consensus_probabilities(odds)

    pairs = same_game_multi_pairs(args.total_line)
    tables = []
    for fx in fixtures[~fixtures["played"]].itertuples(index=False):
        if fx.home not in fitted.attack or fx.away not in fitted.attack:
            continue
        h2h, totals = market_targets(consensus, fx.home, fx.away)
        if h2h is None:  # need market marginals to isolate correlation
            continue
        hb = HOST_ADVANTAGE if fx.home in HOSTS else 0.0
        ab = HOST_ADVANTAGE if fx.away in HOSTS else 0.0
        lam, mu = match_rates(fitted, fx.home, fx.away, neutral=True, home_boost=hb, away_boost=ab)
        grid = blend_to_market(score_matrix(lam, mu, fitted.rho, args.max_goals), h2h=h2h, totals=totals)
        tables.append(rank_same_game_multis(grid, pairs, match_id=f"{fx.home} v {fx.away}"))

    if not tables:
        print("No fixtures with market marginals found.")
        return

    allm = pd.concat(tables, ignore_index=True)
    value = allm[(allm["corr_edge"] >= args.min_edge) & (allm["joint_prob"] >= args.min_prob)]
    value = value.sort_values("corr_edge", ascending=False).head(args.top)

    if value.empty:
        print("No same-game multis cleared the correlation-edge filter.")
        return

    show = value.copy()
    show["joint"] = (show["joint_prob"] * 100).round(1)
    show["edge"] = (show["corr_edge"] * 100).round(1)
    show["fair_odds"] = show["fair_odds"].round(2)
    print(f"\nTop {len(show)} correlated same-game multis (independence-priced books):\n")
    print(show[["match_id", "legs", "joint", "fair_odds", "edge"]].to_string(index=False))
    print(
        "\nedge % ~ value vs a book pricing the legs independently, before its "
        "margin. Back these at books that build same-game multis by multiplying "
        "leg prices (most soft books). 'joint' is the model's true combined prob."
    )


if __name__ == "__main__":
    main()
