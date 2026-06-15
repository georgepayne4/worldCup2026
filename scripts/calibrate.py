"""Fit a calibration temperature and re-verify it (Post-MVP P1).

Step 1 — fit. Refit the model as of 2024-01-01, predict 1X2 for competitive
internationals played since (out-of-sample, real outcomes), and fit the
temperature that minimises log-loss. T < 1 confirms the under-confidence Gate G1
flagged.

Step 2 — re-verify. Refit the live model, price the fixtures we have market odds
for, and show the favourite/underdog bias vs the market collapses once the
temperature is applied.

Run:  python scripts/calibrate.py
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from worldcup2026.data.loaders import load_fixtures_2026, load_international_results
from worldcup2026.data.odds import apply_team_aliases, best_prices, load_snapshot
from worldcup2026.evaluation.backtest import fit_window, outcomes_1x2, predict_1x2
from worldcup2026.evaluation.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
)
from worldcup2026.evaluation.metrics import log_loss
from worldcup2026.models.dixon_coles import match_probabilities, match_rates, score_matrix

RECENT_CUTOFF = "2024-01-01"
WC_KICKOFF = "2026-06-11"
TODAY = "2026-06-15"
HOSTS = frozenset({"Mexico", "United States", "Canada"})
HOST_ADVANTAGE = 0.15


def holdout_predictions(results: pd.DataFrame):
    """Out-of-sample 1X2 forecasts + outcomes on recent competitive matches."""
    holdout = results[
        (results["importance"] != "friendly")
        & (results["date"] >= RECENT_CUTOFF)
        & (results["date"] < WC_KICKOFF)
    ]
    fitted = fit_window(results, RECENT_CUTOFF, 10.0, 1095.0)
    known = set(fitted.attack)
    usable = holdout[holdout["home"].isin(known) & holdout["away"].isin(known)]
    matches = [
        {"home": r.home, "away": r.away, "home_goals": int(r.home_goals),
         "away_goals": int(r.away_goals), "neutral": bool(r.neutral)}
        for r in usable.itertuples(index=False)
    ]
    return predict_1x2(fitted, matches), outcomes_1x2(matches)


def market_bias(results: pd.DataFrame, temperature: float) -> pd.DataFrame | None:
    """Favourite/underdog bias vs market, raw model vs temperature-calibrated."""
    snaps = sorted(glob.glob("data/raw/odds/odds_*.csv"))
    if not snaps:
        return None
    fitted = fit_window(results, TODAY, 10.0, 1095.0)
    fx = load_fixtures_2026()
    rows = []
    for f in fx[~fx["played"]].itertuples(index=False):
        if f.home not in fitted.attack or f.away not in fitted.attack:
            continue
        hb = HOST_ADVANTAGE if f.home in HOSTS else 0.0
        ab = HOST_ADVANTAGE if f.away in HOSTS else 0.0
        lam, mu = match_rates(fitted, f.home, f.away, neutral=True, home_boost=hb, away_boost=ab)
        p_raw = np.array(match_probabilities(score_matrix(lam, mu, fitted.rho, 8)))
        p_cal = apply_temperature(p_raw[None, :], temperature)[0]
        for sel, pr, pc in zip(("Home", "Draw", "Away"), p_raw, p_cal, strict=True):
            rows.append({"home_team": f.home, "away_team": f.away, "selection": sel,
                         "raw": pr, "cal": pc})
    model = pd.DataFrame(rows)

    od = apply_team_aliases(best_prices(load_snapshot(snaps[-1])))
    od = od[od["market"] == "h2h"].copy()
    od.loc[od["selection"] == od["home_team"], "selection"] = "Home"
    od.loc[od["selection"] == od["away_team"], "selection"] = "Away"
    m = model.merge(od[["home_team", "away_team", "selection", "price"]],
                    on=["home_team", "away_team", "selection"], how="inner")
    m["raw_mkt"] = 1.0 / m["price"]
    grp = m.groupby(["home_team", "away_team"])
    m = m[grp["selection"].transform("count") == 3].copy()
    m["mkt"] = m.groupby(["home_team", "away_team"])["raw_mkt"].transform(lambda s: s / s.sum())

    def bucket(p):
        return "favourite (>50%)" if p > 0.5 else ("underdog (<20%)" if p < 0.2 else "mid (20-50%)")

    m["bucket"] = m["mkt"].map(bucket)
    return m.groupby("bucket").apply(
        lambda d: pd.Series({
            "n": len(d),
            "bias_raw": (d["raw"] - d["mkt"]).mean(),
            "bias_calibrated": (d["cal"] - d["mkt"]).mean(),
        }),
        include_groups=False,
    )


def main() -> None:
    results = load_international_results()

    print("Fitting temperature on out-of-sample competitive matches...")
    probs, outcomes = holdout_predictions(results)
    temperature = fit_temperature(probs, outcomes)
    cal = apply_temperature(probs, temperature)
    print(f"  holdout matches : {len(outcomes)}")
    print(f"  temperature T   : {temperature:.3f}  ({'sharpen' if temperature < 1 else 'soften'})")
    print(f"  log-loss        : {log_loss(probs, outcomes):.4f} -> {log_loss(cal, outcomes):.4f}")
    print(
        f"  ECE             : {expected_calibration_error(probs, outcomes):.4f} -> "
        f"{expected_calibration_error(cal, outcomes):.4f}"
    )

    print("\nRe-verifying against live market odds (bias = model - market):")
    bias = market_bias(results, temperature)
    if bias is None:
        print("  no odds snapshot found — skipping market re-verify.")
    else:
        print(bias.round(4).to_string())
        print("\nCalibrated bias should be closer to 0 across buckets (esp. favourites).")
    print(f"\nUse it:  python scripts/find_value.py --odds <snap> --temperature {temperature:.3f}")


if __name__ == "__main__":
    main()
