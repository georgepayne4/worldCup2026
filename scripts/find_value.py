"""Find +EV bets: model prices vs market odds -> staked bet sheet.

End-to-end MVP pipeline: fit the model, price every remaining World Cup fixture,
compare to bookmaker odds, keep the +EV selections, and size them with
fractional Kelly under exposure caps (``betting.edge``).

Odds source:
* ``--odds PATH`` reads a normalized snapshot (see ``data.odds``). Provider team
  names must match the dataset's; a name map is a known follow-up.
* ``--demo`` fabricates a plausible market from the model (with margin + noise)
  so the pipeline runs without a live feed — for illustration only.

Run:  python scripts/find_value.py --demo
      python scripts/find_value.py --odds data/raw/odds/odds_*.csv --bankroll 500
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from worldcup2026.betting.edge import (
    StakingConfig,
    bet_sheet_summary,
    build_bet_sheet,
    log_bets,
)
from worldcup2026.betting.markets import match_market_table
from worldcup2026.data.loaders import load_fixtures_2026, load_international_results
from worldcup2026.data.odds import apply_team_aliases, best_prices, load_snapshot
from worldcup2026.evaluation.backtest import fit_window
from worldcup2026.models.dixon_coles import match_rates, score_matrix

TODAY = "2026-06-15"
HOSTS = frozenset({"Mexico", "United States", "Canada"})
HOST_ADVANTAGE = 0.15
# Markets to bet (exclude double_chance: it overlaps h2h and would double-count).
BET_MARKETS = ("h2h", "totals", "btts")


def model_candidates(fitted, fixtures, max_goals: int) -> pd.DataFrame:
    """Model probabilities for every remaining fixture's bettable selections."""
    tables = []
    for fx in fixtures[~fixtures["played"]].itertuples(index=False):
        if fx.home not in fitted.attack or fx.away not in fitted.attack:
            continue
        hb = HOST_ADVANTAGE if fx.home in HOSTS else 0.0
        ab = HOST_ADVANTAGE if fx.away in HOSTS else 0.0
        lam, mu = match_rates(fitted, fx.home, fx.away, neutral=True, home_boost=hb, away_boost=ab)
        matrix = score_matrix(lam, mu, fitted.rho, max_goals)
        tables.append(
            match_market_table(
                matrix, match_id=f"{fx.home} v {fx.away}",
                home_team=fx.home, away_team=fx.away,
            )
        )
    df = pd.concat(tables, ignore_index=True).rename(columns={"prob": "model_prob"})
    return df[df["market"].isin(BET_MARKETS)].reset_index(drop=True)


def synth_market(
    candidates: pd.DataFrame, seed: int, margin: float = 0.06, noise: float = 0.05
) -> pd.DataFrame:
    """Fabricate market odds from the model for --demo.

    Perturbs *fair odds* multiplicatively: shorten by a margin (the book's edge)
    times lognormal noise. EV then depends only on the noise, not on whether the
    selection is a favourite or a longshot — so the demo doesn't manufacture
    absurd value on 50/1 shots.
    """
    rng = np.random.default_rng(seed)
    fair = 1.0 / candidates["model_prob"].to_numpy()
    market_odds = fair * (1.0 - margin) * np.exp(rng.normal(0.0, noise, len(candidates)))
    out = candidates.copy()
    out["market_odds"] = np.clip(market_odds, 1.01, 26.0)
    return out


def _remap_h2h(odds: pd.DataFrame) -> pd.DataFrame:
    """Map provider h2h outcome names (team names) to Home/Draw/Away."""
    odds = odds.copy()
    is_h2h = odds["market"] == "h2h"
    sel = odds["selection"]
    odds.loc[is_h2h & (sel == odds["home_team"]), "selection"] = "Home"
    odds.loc[is_h2h & (sel == odds["away_team"]), "selection"] = "Away"
    return odds


def join_market(candidates: pd.DataFrame, odds_path: str) -> pd.DataFrame:
    """Join real snapshot odds (best price per selection) onto model candidates."""
    odds = _remap_h2h(apply_team_aliases(best_prices(load_snapshot(odds_path))))
    keys = ["home_team", "away_team", "market", "selection", "line"]
    cand = candidates.copy()
    # h2h/btts have no line; use a sentinel so NaN keys join cleanly (and dtypes match).
    cand["line"] = pd.to_numeric(cand["line"], errors="coerce").fillna(-1.0)
    odds = odds.copy()
    odds["line"] = pd.to_numeric(odds["line"], errors="coerce").fillna(-1.0)
    merged = cand.merge(
        odds[[*keys, "price"]].rename(columns={"price": "market_odds"}),
        on=keys, how="inner",
    )
    merged["line"] = merged["line"].replace(-1.0, np.nan)
    print(f"  matched {len(merged)} of {len(candidates)} model selections to odds")
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--odds", help="normalized odds snapshot CSV")
    src.add_argument("--demo", action="store_true", help="fabricate a market from the model")
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--kelly", type=float, default=0.25, help="fractional-Kelly multiplier")
    ap.add_argument("--min-ev", type=float, default=0.03)
    ap.add_argument("--max-goals", type=int, default=8)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--log", help="append the bet sheet to this CSV")
    args = ap.parse_args()

    print(f"Fitting model... (host residual on {', '.join(sorted(HOSTS))})")
    results = load_international_results()
    fitted = fit_window(results, TODAY, 10.0, 1095.0)
    fixtures = load_fixtures_2026()

    candidates = model_candidates(fitted, fixtures, args.max_goals)
    if args.demo:
        print("Using SYNTHETIC market (demo) - not real prices.")
        candidates = synth_market(candidates, seed=1)
    else:
        candidates = join_market(candidates, args.odds)

    cfg = StakingConfig(bankroll=args.bankroll, kelly_fraction=args.kelly, min_ev=args.min_ev)
    sheet = build_bet_sheet(candidates, cfg)

    if sheet.empty:
        print("\nNo +EV bets cleared the threshold.")
        return

    show = sheet.head(args.top)[
        ["match_id", "market", "selection", "line", "model_prob", "fair_odds", "market_odds", "ev", "stake"]
    ].copy()
    for col in ("model_prob", "ev"):
        show[col] = (show[col] * 100).round(1)
    show[["fair_odds", "market_odds"]] = show[["fair_odds", "market_odds"]].round(2)
    show["stake"] = show["stake"].round(2)
    print(f"\nTop {len(show)} value bets (model_prob % | ev %):\n")
    print(show.to_string(index=False))

    rep = bet_sheet_summary(sheet, cfg)
    print(
        f"\n{rep['n_bets']} bets | staked {rep['total_staked']:.0f} "
        f"({rep['exposure_pct']:.0%} of bankroll) | "
        f"expected profit {rep['expected_profit']:+.1f} (ROI {rep['expected_roi']:+.1%})"
    )
    if args.log:
        log_bets(sheet, args.log)
        print(f"Logged {len(sheet)} bets -> {args.log}")
    if args.demo:
        print("\n(Demo market is synthetic; EVs are illustrative, not real edges.)")


if __name__ == "__main__":
    main()
