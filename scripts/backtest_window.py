"""Grid-search the Dixon-Coles training window and time-decay half-life.

Scores 1X2 log-loss / RPS on two holdouts:

* **WC2022** — fit on data before kickoff (2022-11-20), predict the 64 finals
  matches. The closest analogue to predicting WC2026 (tournament, neutral,
  strong teams).
* **Recent competitive** — fit on data before 2024-01-01, predict competitive
  internationals (qualifiers / confederation finals / WC) from 2024 up to the
  2026 tournament. A larger, noisier sample.

Recommends the (window, half-life) minimising mean log-loss across both.

Run:  python scripts/backtest_window.py
"""

from __future__ import annotations

import argparse
from itertools import product

import pandas as pd

from worldcup2026.data.loaders import load_international_results
from worldcup2026.evaluation.backtest import evaluate

WC2022_KICKOFF = "2022-11-20"
RECENT_CUTOFF = "2024-01-01"
WC2026_KICKOFF = "2026-06-11"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=float, nargs="+", default=[4.0, 6.0, 8.0])
    ap.add_argument("--half-lives", type=float, nargs="+", default=[365.0, 547.0, 730.0])
    ap.add_argument("--max-goals", type=int, default=8)
    args = ap.parse_args()

    results = load_international_results()

    wc2022 = results[
        (results["tournament"] == "FIFA World Cup") & (results["date"].dt.year == 2022)
    ]
    recent = results[
        (results["importance"] != "friendly")
        & (results["date"] >= RECENT_CUTOFF)
        & (results["date"] < WC2026_KICKOFF)
    ]
    holdouts = {
        "WC2022": (WC2022_KICKOFF, wc2022),
        "recent": (RECENT_CUTOFF, recent),
    }
    print(
        f"Holdouts: WC2022={len(wc2022)} matches, "
        f"recent competitive={len(recent)} matches\n"
    )

    rows = []
    for w, hl in product(args.windows, args.half_lives):
        rec = {"window_y": w, "half_life_d": hl}
        for name, (ref, eval_df) in holdouts.items():
            r = evaluate(results, eval_df, ref, w, hl, args.max_goals)
            rec[f"{name}_logloss"] = r.log_loss
            rec[f"{name}_rps"] = r.rps
            rec[f"{name}_n"] = r.n_eval
        rec["mean_logloss"] = (rec["WC2022_logloss"] + rec["recent_logloss"]) / 2
        rows.append(rec)
        print(
            f"  window={w:>3.0f}y half-life={hl:>4.0f}d  "
            f"WC2022 LL={rec['WC2022_logloss']:.4f} RPS={rec['WC2022_rps']:.4f}  "
            f"recent LL={rec['recent_logloss']:.4f} RPS={rec['recent_rps']:.4f}"
        )

    table = pd.DataFrame(rows).sort_values("mean_logloss").reset_index(drop=True)
    pd.set_option("display.width", 160, "display.max_columns", None)
    print("\nRanked by mean log-loss (lower is better):\n")
    print(
        table[
            ["window_y", "half_life_d", "WC2022_logloss", "recent_logloss", "mean_logloss"]
        ].to_string(index=False)
    )
    best = table.iloc[0]
    print(
        f"\nRecommended: window={best.window_y:.0f}y  "
        f"half-life={best.half_life_d:.0f}d  "
        f"(mean log-loss {best.mean_logloss:.4f})"
    )


if __name__ == "__main__":
    main()
