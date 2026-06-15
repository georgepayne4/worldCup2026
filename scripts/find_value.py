"""Find +EV bets: model prices vs market odds -> staked bet sheet.

With ``--odds`` the model's per-fixture marginals are **blended to the market
consensus** (vig-removed mean across books) before pricing, and bets are then
checked against the **best available** price. So a single 1X2/totals bet only
shows value where a book is offering longer than the consensus fair price (real
line-shopping value) — the model's raw 1X2 disagreement (which Gate G1 showed is
miscalibration, not edge) is removed. The structural edge lives in correlated
same-game multis instead — see ``scripts/find_multis.py``.

``--demo`` fabricates a market from the model (no blend) just to exercise the
staking pipeline offline.

Run:  python scripts/find_value.py --odds data/raw/odds/odds_*.csv
      python scripts/find_value.py --demo
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from worldcup2026.betting.blend import blend_to_market, sharpen_1x2
from worldcup2026.betting.edge import (
    StakingConfig,
    bet_sheet_summary,
    build_bet_sheet,
    log_bets,
)
from worldcup2026.betting.markets import match_market_table
from worldcup2026.data.loaders import load_fixtures_2026, load_international_results
from worldcup2026.data.odds import (
    apply_team_aliases,
    best_prices,
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
# Markets to bet (exclude double_chance: it overlaps h2h and would double-count).
BET_MARKETS = ("h2h", "totals", "btts")


def model_candidates(
    fitted, fixtures, max_goals: int, temperature: float = 1.0, consensus=None
) -> pd.DataFrame:
    """Bettable selections per remaining fixture, with model probabilities.

    If `consensus` is given, each fixture's grid is blended to the market
    marginals (so single-bet prices reflect the market, not raw model error);
    otherwise the calibration `temperature` is applied.
    """
    tables = []
    for fx in fixtures[~fixtures["played"]].itertuples(index=False):
        if fx.home not in fitted.attack or fx.away not in fitted.attack:
            continue
        hb = HOST_ADVANTAGE if fx.home in HOSTS else 0.0
        ab = HOST_ADVANTAGE if fx.away in HOSTS else 0.0
        lam, mu = match_rates(fitted, fx.home, fx.away, neutral=True, home_boost=hb, away_boost=ab)
        grid = score_matrix(lam, mu, fitted.rho, max_goals)
        if consensus is not None:
            h2h, totals = market_targets(consensus, fx.home, fx.away)
            grid = blend_to_market(grid, h2h=h2h, totals=totals) if (h2h or totals) \
                else sharpen_1x2(grid, temperature)
        else:
            grid = sharpen_1x2(grid, temperature)
        tables.append(
            match_market_table(grid, match_id=f"{fx.home} v {fx.away}",
                               home_team=fx.home, away_team=fx.away)
        )
    df = pd.concat(tables, ignore_index=True).rename(columns={"prob": "model_prob"})
    return df[df["market"].isin(BET_MARKETS)].reset_index(drop=True)


def synth_market(candidates: pd.DataFrame, seed: int, margin: float = 0.06, noise: float = 0.05) -> pd.DataFrame:
    """Fabricate market odds from the model for --demo (multiplicative noise)."""
    rng = np.random.default_rng(seed)
    fair = 1.0 / candidates["model_prob"].to_numpy()
    market_odds = fair * (1.0 - margin) * np.exp(rng.normal(0.0, noise, len(candidates)))
    out = candidates.copy()
    out["market_odds"] = np.clip(market_odds, 1.01, 26.0)
    return out


def join_best(candidates: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    """Attach the best available decimal price to each model candidate."""
    keys = ["home_team", "away_team", "market", "selection", "line"]
    cand = candidates.copy()
    cand["line"] = pd.to_numeric(cand["line"], errors="coerce").fillna(-1.0)
    best = best.copy()
    best["line"] = pd.to_numeric(best["line"], errors="coerce").fillna(-1.0)
    merged = cand.merge(
        best[[*keys, "price"]].rename(columns={"price": "market_odds"}), on=keys, how="inner"
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
    ap.add_argument("--min-ev", type=float, default=0.02)
    ap.add_argument(
        "--min-prob",
        type=float,
        default=0.05,
        help="drop longshots below this prob (best-vs-consensus is noisy there)",
    )
    ap.add_argument("--max-goals", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0, help="calibration (matches without odds)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--log", help="append the bet sheet to this CSV")
    args = ap.parse_args()

    print(f"Fitting model... (host residual on {', '.join(sorted(HOSTS))})")
    results = load_international_results()
    fitted = fit_window(results, TODAY, 10.0, 1095.0)
    fixtures = load_fixtures_2026()

    if args.demo:
        candidates = model_candidates(fitted, fixtures, args.max_goals, args.temperature)
        print("Using SYNTHETIC market (demo) - not real prices.")
        candidates = synth_market(candidates, seed=1)
    else:
        odds = normalize_h2h_selections(apply_team_aliases(load_snapshot(args.odds)))
        consensus = consensus_probabilities(odds)
        print("Blending model marginals to market consensus; pricing vs best book.")
        candidates = model_candidates(fitted, fixtures, args.max_goals, args.temperature, consensus)
        candidates = join_best(candidates, best_prices(odds))

    candidates = candidates[candidates["model_prob"] >= args.min_prob]
    cfg = StakingConfig(bankroll=args.bankroll, kelly_fraction=args.kelly, min_ev=args.min_ev)
    sheet = build_bet_sheet(candidates, cfg)

    if sheet.empty:
        print("\nNo +EV bets cleared the threshold (expected after blending — that's the point).")
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
    else:
        print(
            "\nWith marginals blended to market, single-bet EV is just line-shopping "
            "(best vs consensus). The real edge is correlation: scripts/find_multis.py."
        )


if __name__ == "__main__":
    main()
