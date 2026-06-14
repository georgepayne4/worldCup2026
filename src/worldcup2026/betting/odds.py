"""Convert bookmaker odds to probabilities and evaluate bets.

Small, market-agnostic primitives — the rest of the betting layer (METHODOLOGY
§9) builds on these.
"""

from __future__ import annotations

import numpy as np


def odds_to_probability(decimal_odds: float) -> float:
    """Implied probability of a decimal-odds quote (no vig removed)."""
    return 1.0 / decimal_odds


def probability_to_odds(probability: float) -> float:
    """Fair decimal odds for a given probability."""
    return 1.0 / probability


def remove_vig(odds: list[float]) -> list[float]:
    """Strip the bookmaker margin from a set of mutually-exclusive prices.

    Returns implied probabilities normalised to sum to 1 by the proportional
    method (each implied prob divided by the overround). Works for any market
    whose legs cover all outcomes (1X2, over/under, BTTS, etc.).
    """
    implied = np.array([1.0 / o for o in odds], dtype=float)
    overround = implied.sum()
    return (implied / overround).tolist()


def expected_value(probability: float, decimal_odds: float, stake: float = 1.0) -> float:
    """Expected profit per stake unit. >0 means +EV at this price."""
    return stake * (probability * decimal_odds - 1.0)


def kelly_fraction(probability: float, decimal_odds: float) -> float:
    """Full-Kelly bankroll fraction.

    Returns 0 when the bet is -EV. In practice apply a fractional multiplier
    (e.g. 0.25x) to dampen variance — full Kelly is provably growth-optimal
    only when the probability estimate is exactly right.
    """
    edge = probability * decimal_odds - 1.0
    if edge <= 0.0:
        return 0.0
    return edge / (decimal_odds - 1.0)
