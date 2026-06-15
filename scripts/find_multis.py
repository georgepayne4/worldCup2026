"""Find same-game-multi value with REAL EV vs bookmaker prices.

The project's edge thesis, made concrete. For each remaining fixture:

* the model's grid is blended to the **market consensus** marginals, so the only
  thing left that is ours is the *correlation* between legs;
* the **bookmaker SGM price** is reconstructed as the best single-book product of
  its own leg prices (`markets.book_multi_prices`) — what a book that builds
  same-game multis by multiplying legs would actually offer;
* **real EV = model_joint_prob × book_SGM_price − 1**.

Positively-correlated legs (e.g. a favourite winning *and* unders, in a tight
game) are underpriced by such independence-multiplying books — that's the edge.
We restrict to result×total combos, which we can price from h2h+totals leg odds
(and which are *less obvious* than Over+BTTS, so likelier to be mispriced).

**Assumption:** the book offers the SGM at the product of its leg prices (true of
many soft books). Books with proper correlation engines will offer less; treat
EV as an upper bound and confirm against the actual bet-builder price.

Run:  python scripts/find_multis.py --odds data/raw/odds/odds_*.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from worldcup2026.betting.blend import blend_to_market
from worldcup2026.betting.markets import (
    book_multi_prices,
    rank_same_game_multis,
    same_game_multi_pairs,
)
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
    ap.add_argument("--min-ev", type=float, default=0.02, help="min real EV to show")
    ap.add_argument("--min-prob", type=float, default=0.10, help="skip unlikely multis")
    ap.add_argument("--min-books", type=int, default=3, help="require this many books quoting the multi")
    ap.add_argument("--max-goals", type=int, default=8)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    print("Fitting model and blending each fixture to market consensus...")
    results = load_international_results()
    fitted = fit_window(results, TODAY, 10.0, 1095.0)
    fixtures = load_fixtures_2026()
    odds = normalize_h2h_selections(apply_team_aliases(load_snapshot(args.odds)))
    consensus = consensus_probabilities(odds)

    pairs = same_game_multi_pairs(args.total_line, include_btts=False)

    model_tables = []
    for fx in fixtures[~fixtures["played"]].itertuples(index=False):
        if fx.home not in fitted.attack or fx.away not in fitted.attack:
            continue
        h2h, totals = market_targets(consensus, fx.home, fx.away)
        if h2h is None:
            continue
        hb = HOST_ADVANTAGE if fx.home in HOSTS else 0.0
        ab = HOST_ADVANTAGE if fx.away in HOSTS else 0.0
        lam, mu = match_rates(fitted, fx.home, fx.away, neutral=True, home_boost=hb, away_boost=ab)
        grid = blend_to_market(score_matrix(lam, mu, fitted.rho, args.max_goals), h2h=h2h, totals=totals)
        t = rank_same_game_multis(grid, pairs)
        t["home_team"], t["away_team"] = fx.home, fx.away
        model_tables.append(t)

    if not model_tables:
        print("No fixtures with market marginals found.")
        return

    model = pd.concat(model_tables, ignore_index=True)
    book = book_multi_prices(odds, pairs)
    merged = model.merge(book, on=["home_team", "away_team", "legs"], how="inner")
    merged["ev"] = merged["joint_prob"] * merged["sgm_price"] - 1.0

    value = merged[
        (merged["ev"] >= args.min_ev)
        & (merged["joint_prob"] >= args.min_prob)
        & (merged["n_books"] >= args.min_books)
    ].sort_values("ev", ascending=False).head(args.top)

    if value.empty:
        print("\nNo same-game multis cleared the real-EV filter.")
        return

    show = value.copy()
    show["match"] = show["home_team"] + " v " + show["away_team"]
    show["joint"] = (show["joint_prob"] * 100).round(1)
    show["ev"] = (show["ev"] * 100).round(1)
    show["sgm_price"] = show["sgm_price"].round(2)
    show["fair"] = show["fair_odds"].round(2)
    print(f"\nTop {len(show)} same-game multis by REAL EV vs book SGM price:\n")
    print(
        show[["match", "legs", "joint", "fair", "sgm_price", "bookmaker", "n_books", "ev"]]
        .to_string(index=False)
    )
    print(
        "\nfair = model's true joint fair odds; sgm_price = best book's leg-PRODUCT "
        "price; ev = joint*sgm_price-1."
    )
    print(
        "CAVEAT: ev is real ONLY if the book offers the multi at the leg product. "
        "For strongly-correlated combos (e.g. Draw+Under) competent bet-builders "
        "bake in the correlation and offer far less, so these large EVs are a "
        "SCREENING signal, not bankable edge. Plug a real bet-builder quote into "
        "ev = joint*quote-1 to get true EV; edge is likeliest at soft books that "
        "naively multiply, on less-obvious correlations."
    )


if __name__ == "__main__":
    main()
