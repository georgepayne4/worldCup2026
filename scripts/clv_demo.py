"""Demo of the CLV / P&L harness on a synthetic bet ledger.

No live odds needed — this fabricates a handful of matches with closing lines
and a bettor who, on average, takes prices a shade better than the close. It
shows the harness turning (bets, closing lines, results) into the headline
metrics the MVP is judged on: mean CLV-EV, beat-close rate, and ROI.

In production the *taken* price comes from `data.odds.best_prices` (line
shopping), the *closing* line from the final pre-kickoff snapshot, and the bet
selection from the edge filter (MVP-3). Run:  python scripts/clv_demo.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from worldcup2026.betting.clv import attach_clv, attach_settlement, clv_report
from worldcup2026.betting.odds import remove_vig

SELECTIONS = ["Home", "Draw", "Away"]


def main() -> None:
    rng = np.random.default_rng(7)
    n_events = 12
    margin = 0.05  # bookmaker overround on the closing line

    closing_rows = []
    bet_rows = []
    result_rows = []
    for i in range(n_events):
        event = f"E{i:02d}"
        true_p = rng.dirichlet([5.0, 3.0, 4.0])  # home-ish skew
        fair_price = 1.0 / true_p
        # Closing line: fair odds shortened by the margin (vig spread evenly) —
        # so the *closing* price is worse than fair.
        closing_price = fair_price / (1.0 + margin)
        for sel, price in zip(SELECTIONS, closing_price, strict=True):
            closing_rows.append((event, "h2h", sel, np.nan, round(float(price), 3)))

        # A sharp bettor: takes a price ~2% better than *fair* (e.g. by line
        # shopping earlier), so they genuinely beat the no-vig close on average.
        pick = int(rng.integers(0, 3))
        taken = round(float(fair_price[pick]) * (1.0 + rng.normal(0.02, 0.025)), 3)
        bet_rows.append((event, "h2h", SELECTIONS[pick], np.nan, taken, 1.0))

        outcome = int(rng.choice(3, p=true_p))
        for s_idx, sel in enumerate(SELECTIONS):
            result_rows.append((event, "h2h", sel, np.nan, s_idx == outcome))

    closing = pd.DataFrame(
        closing_rows, columns=["event_id", "market", "selection", "line", "price"]
    )
    bets = pd.DataFrame(
        bet_rows,
        columns=["event_id", "market", "selection", "line", "price_taken", "stake"],
    )
    results = pd.DataFrame(
        result_rows, columns=["event_id", "market", "selection", "line", "won"]
    )

    ledger = attach_settlement(attach_clv(bets, closing), results)

    show = ledger[
        ["event_id", "selection", "price_taken", "closing_price", "clv_ev", "beat_close", "profit"]
    ].copy()
    show["clv_ev"] = (show["clv_ev"] * 100).round(2)
    print(show.to_string(index=False))

    rep = clv_report(ledger)
    print("\n--- CLV / P&L report ---")
    print(f"  bets            : {rep['n_bets']}")
    print(f"  mean CLV-EV     : {rep['mean_clv_ev']:+.2%}")
    print(f"  beat-close rate : {rep['beat_close_rate']:.0%}")
    print(f"  ROI             : {rep['roi']:+.2%}  ({rep['total_profit']:+.2f}u staked {rep['total_staked']:.0f}u)")
    print(
        "\nNote: CLV-EV (vs the *no-vig* close) is the honest edge measure -- "
        "beating the raw closing price by a hair can still be -EV once the "
        "bookmaker margin is removed. ROI is noisy over 12 bets; CLV leads it. "
        f"Sample close -> fair probs {np.round(remove_vig(closing.iloc[:3]['price'].tolist()), 3)}."
    )


if __name__ == "__main__":
    main()
