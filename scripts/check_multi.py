"""Check a real same-game-multi quote for true EV.

`find_multis.py` screens *which* combos are correlated; this tells you whether a
**specific bet-builder price you've been quoted** is actually +EV. It prices the
multi off the model (marginals blended to market when a snapshot is given, so
only correlation is ours) and compares your offered price to the model's true
joint.

Legs are ``market:selection[:line]`` — e.g. ``h2h:Home``, ``totals:Under:2.5``,
``btts:Yes``, ``double_chance:1X``.

Run:
  python scripts/check_multi.py --home Germany --away "Ivory Coast" \\
      --legs h2h:Draw,totals:Under:2.5 --price 9.0 \\
      --odds data/raw/odds/odds_*.csv
"""

from __future__ import annotations

import argparse

from worldcup2026.betting.blend import blend_to_market, sharpen_1x2
from worldcup2026.betting.markets import same_game_multi
from worldcup2026.data.loaders import load_international_results
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


def parse_legs(spec: str) -> list[tuple]:
    legs = []
    for chunk in spec.split(","):
        parts = chunk.split(":")
        if len(parts) == 2:
            legs.append((parts[0], parts[1], None))
        elif len(parts) == 3:
            legs.append((parts[0], parts[1], float(parts[2])))
        else:
            raise SystemExit(f"bad leg spec: {chunk!r} (use market:selection[:line])")
    return legs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--legs", required=True, help="comma-separated market:selection[:line]")
    ap.add_argument("--price", type=float, required=True, help="offered decimal odds for the multi")
    ap.add_argument("--odds", help="snapshot to blend marginals to market (recommended)")
    ap.add_argument("--temperature", type=float, default=0.914, help="used if no --odds")
    ap.add_argument("--max-goals", type=int, default=8)
    args = ap.parse_args()

    results = load_international_results()
    fitted = fit_window(results, TODAY, 10.0, 1095.0)
    for team in (args.home, args.away):
        if team not in fitted.attack:
            raise SystemExit(f"no rating for {team!r} — check spelling against the dataset")

    hb = HOST_ADVANTAGE if args.home in HOSTS else 0.0
    ab = HOST_ADVANTAGE if args.away in HOSTS else 0.0
    lam, mu = match_rates(fitted, args.home, args.away, neutral=True, home_boost=hb, away_boost=ab)
    grid = score_matrix(lam, mu, fitted.rho, args.max_goals)

    if args.odds:
        consensus = consensus_probabilities(
            normalize_h2h_selections(apply_team_aliases(load_snapshot(args.odds)))
        )
        h2h, totals = market_targets(consensus, args.home, args.away)
        if h2h is None:
            print("(no market marginals for this fixture; using model + temperature)")
            grid = sharpen_1x2(grid, args.temperature)
        else:
            grid = blend_to_market(grid, h2h=h2h, totals=totals)
    else:
        grid = sharpen_1x2(grid, args.temperature)

    legs = parse_legs(args.legs)
    q = same_game_multi(grid, legs)
    ev = q.joint_prob * args.price - 1.0
    book_implied = 1.0 / args.price

    leg_str = " + ".join(f"{m}:{s}" + (f":{ln}" if ln else "") for m, s, ln in legs)
    print(f"\n{args.home} v {args.away} | {leg_str}")
    print(f"  model true joint prob : {q.joint_prob:.1%}  (fair odds {q.fair_odds:.2f})")
    print(f"  naive-independent     : {q.independent_prob:.1%}  (odds {q.naive_odds:.2f}) - what a multiplying book gives")
    print(f"  correlation ratio     : {q.correlation_ratio:.2f}x")
    print(f"  offered price         : {args.price:.2f}  (book-implied {book_implied:.1%})")
    print(f"  EV                    : {ev:+.1%}  ->  {'+EV, value' if ev > 0 else 'no value'}")


if __name__ == "__main__":
    main()
