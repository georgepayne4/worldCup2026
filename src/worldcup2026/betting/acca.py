"""Build accumulators (parlays) across *different* fixtures.

Legs from different matches are treated as independent — reasonable, unlike legs
within one match (use ``markets.same_game_multi`` for those). So an acca's win
probability is the product of its legs' probabilities, and its fair odds the
product of fair odds.

Two selection modes:

* ``accuracy`` — pick the highest-probability legs (one per match) above a
  confidence floor. The "give me a safe 2–3 game acca" use case.
* ``value`` — pick the highest positive-EV legs versus supplied market odds.

The trade-off is explicit in the output: more legs ⇒ longer odds but lower
chance of landing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_MATCH_COL_CANDIDATES = ("match_id", "fixture", "home_team")


@dataclass
class Acca:
    """A built accumulator."""

    legs: pd.DataFrame
    n_legs: int
    combined_prob: float
    combined_fair_odds: float
    combined_market_odds: float | None
    ev: float | None

    def summary(self) -> str:
        lines = [
            f"{self.n_legs}-leg acca | win prob {self.combined_prob:.1%} | "
            f"fair odds {self.combined_fair_odds:.2f}"
        ]
        if self.combined_market_odds is not None:
            lines[0] += (
                f" | market {self.combined_market_odds:.2f} | EV {self.ev:+.1%}"
            )
        return lines[0]


def _match_col(df: pd.DataFrame) -> str:
    for col in _MATCH_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        f"candidates need a match identifier column (one of {_MATCH_COL_CANDIDATES})"
    )


def build_acca(
    candidates: pd.DataFrame,
    *,
    n_legs: int = 3,
    mode: str = "accuracy",
    min_prob: float = 0.60,
    max_prob: float = 0.97,
    one_per_match: bool = True,
    market_odds_col: str = "market_odds",
) -> Acca:
    """Assemble one accumulator from a table of candidate legs.

    `candidates` has at least ``prob`` and a match id column (``match_id`` /
    ``fixture`` / ``home_team``); for ``value`` mode it also needs
    `market_odds_col`. ``min_prob``/``max_prob`` bound per-leg confidence so the
    acca avoids coin-flips and no-value near-certainties. Returns whatever legs
    qualify (up to `n_legs`).
    """
    if "prob" not in candidates.columns:
        raise ValueError("candidates need a 'prob' column")
    match_col = _match_col(candidates)
    df = candidates[(candidates["prob"] >= min_prob) & (candidates["prob"] <= max_prob)].copy()

    if mode == "value":
        if market_odds_col not in df.columns:
            raise ValueError(f"value mode needs a '{market_odds_col}' column")
        df["ev"] = df["prob"] * df[market_odds_col] - 1.0
        df = df[df["ev"] > 0].sort_values("ev", ascending=False)
    elif mode == "accuracy":
        df = df.sort_values("prob", ascending=False)
    else:
        raise ValueError("mode must be 'accuracy' or 'value'")

    chosen: list[pd.Series] = []
    used: set = set()
    for _, row in df.iterrows():
        if one_per_match and row[match_col] in used:
            continue
        chosen.append(row)
        used.add(row[match_col])
        if len(chosen) >= n_legs:
            break

    legs = pd.DataFrame(chosen).reset_index(drop=True)
    combined_prob = float(np.prod(legs["prob"])) if len(legs) else 0.0
    combined_fair = 1.0 / combined_prob if combined_prob > 0 else float("inf")

    combined_market: float | None = None
    ev: float | None = None
    if market_odds_col in legs.columns and len(legs):
        combined_market = float(np.prod(legs[market_odds_col]))
        ev = combined_prob * combined_market - 1.0

    return Acca(
        legs=legs,
        n_legs=len(legs),
        combined_prob=combined_prob,
        combined_fair_odds=combined_fair,
        combined_market_odds=combined_market,
        ev=ev,
    )


def suggest_accas(
    candidates: pd.DataFrame,
    *,
    n_accas: int = 3,
    n_legs: int = 3,
    mode: str = "accuracy",
    **kwargs,
) -> list[Acca]:
    """Return several non-overlapping accas (no fixture reused across them).

    Greedy: build one acca, drop its fixtures, build the next. Good for offering
    a few independent slips rather than one.
    """
    match_col = _match_col(candidates)
    pool = candidates.copy()
    accas: list[Acca] = []
    for _ in range(n_accas):
        acca = build_acca(pool, n_legs=n_legs, mode=mode, **kwargs)
        if acca.n_legs == 0:
            break
        accas.append(acca)
        pool = pool[~pool[match_col].isin(set(acca.legs[match_col]))]
        if pool.empty:
            break
    return accas
