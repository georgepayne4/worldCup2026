"""Backtesting helpers for choosing fit hyper-parameters.

The two knobs that matter most for the Dixon-Coles fit are the **training
window** (how far back to look) and the **time-decay half-life** (how fast old
matches lose weight). This module fits on data before a cutoff and scores 1X2
predictions on a held-out set, so a grid search can pick the window/half-life
that generalise best (METHODOLOGY §6).

Outcome encoding for 1X2 is ordinal: 0 = home win, 1 = draw, 2 = away win, so
the ranked probability score penalises being further from the truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from worldcup2026.evaluation.metrics import (
    log_loss,
    ranked_probability_score,
)
from worldcup2026.models.dixon_coles import (
    DixonColesParams,
    fit,
    match_probabilities,
    match_rates,
    score_matrix,
    time_decay_weights,
)


def _records(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "home": r.home,
            "away": r.away,
            "home_goals": int(r.home_goals),
            "away_goals": int(r.away_goals),
            "neutral": bool(r.neutral),
        }
        for r in df.itertuples(index=False)
    ]


def fit_window(
    results: pd.DataFrame,
    reference: str | np.datetime64,
    window_years: float,
    half_life_days: float,
) -> DixonColesParams:
    """Fit Dixon-Coles on matches in [reference - window_years, reference)."""
    ref = pd.Timestamp(np.datetime64(reference, "D"))
    since = ref - pd.Timedelta(days=round(window_years * 365.25))
    window = results[(results["date"] >= since) & (results["date"] < ref)]
    matches = _records(window)
    weights = time_decay_weights(
        window["date"].to_numpy(), np.datetime64(ref.date(), "D"), half_life_days
    )
    teams = sorted(set(window["home"]) | set(window["away"]))
    return fit(matches, teams=teams, weights=weights)


def predict_1x2(
    fitted: DixonColesParams, matches: list[dict], max_goals: int = 8
) -> np.ndarray:
    """(n, 3) array of [P(home win), P(draw), P(away win)] per match."""
    rows = []
    for m in matches:
        lam, mu = match_rates(fitted, m["home"], m["away"], neutral=m.get("neutral", False))
        rows.append(match_probabilities(score_matrix(lam, mu, fitted.rho, max_goals)))
    return np.array(rows)


def outcomes_1x2(matches: list[dict]) -> np.ndarray:
    """Ordinal outcome per match: 0 home win, 1 draw, 2 away win."""
    out = []
    for m in matches:
        diff = m["home_goals"] - m["away_goals"]
        out.append(0 if diff > 0 else (1 if diff == 0 else 2))
    return np.array(out)


@dataclass
class BacktestResult:
    window_years: float
    half_life_days: float
    n_eval: int
    log_loss: float
    rps: float


def evaluate(
    results: pd.DataFrame,
    eval_matches: pd.DataFrame,
    reference: str | np.datetime64,
    window_years: float,
    half_life_days: float,
    max_goals: int = 8,
) -> BacktestResult:
    """Fit on the window before `reference`, score on `eval_matches`.

    Eval matches whose teams never appear in the training window are dropped
    (no rating to predict with).
    """
    fitted = fit_window(results, reference, window_years, half_life_days)
    known = set(fitted.attack)
    usable = eval_matches[
        eval_matches["home"].isin(known) & eval_matches["away"].isin(known)
    ]
    matches = _records(usable)
    probs = predict_1x2(fitted, matches, max_goals)
    outcomes = outcomes_1x2(matches)
    return BacktestResult(
        window_years=window_years,
        half_life_days=half_life_days,
        n_eval=len(matches),
        log_loss=log_loss(probs, outcomes),
        rps=ranked_probability_score(probs, outcomes),
    )
