"""Build the market's joint score distribution from Correct Score prices.

A Correct Score market prices every scoreline, so it *is* the market's full joint
distribution of a match — exactly the object our model approximates. Turn it into
a score grid and any correlated combo (Draw+Under, Home+Over, ...) can be priced
straight off the market with no independence assumption and no bet-builder widget:
``same_game_multi(grid, legs)`` on this grid is the market-true joint.

Betfair lists explicit scorelines up to ~3-3 plus three tail buckets — "Any Other
Home Win", "Any Other Away Win", "Any Other Draw" — for everything beyond. We
spread each bucket uniformly over the unlisted cells of its result region (a mild
simplification; refine with a Poisson tail later) and normalise the whole grid to
1, which removes the market's margin.
"""

from __future__ import annotations

import re

import numpy as np

_SCORE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")


def parse_score(name: str) -> tuple[int, int] | None:
    """`"2 - 1"` -> ``(2, 1)``; non-scoreline labels (tail buckets) -> ``None``."""
    m = _SCORE_RE.match(name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def correct_score_grid(
    scores: dict[tuple[int, int], float],
    *,
    other_home: float = 0.0,
    other_away: float = 0.0,
    other_draw: float = 0.0,
    max_goals: int = 10,
) -> np.ndarray:
    """Vig-removed joint score grid from explicit scoreline weights + tail buckets.

    `scores` maps ``(home, away) -> weight`` (implied probs or any positive
    weights). The ``other_*`` masses are the Any-Other-Home/Away/Draw buckets,
    spread over the unlisted cells of each result region. The grid is normalised
    to sum to 1 (so the market margin drops out).
    """
    n = max_goals + 1
    grid = np.zeros((n, n))
    listed = np.zeros((n, n), dtype=bool)
    for (h, a), w in scores.items():
        if 0 <= h <= max_goals and 0 <= a <= max_goals:
            grid[h, a] += w
            listed[h, a] = True

    i, j = np.indices((n, n))
    regions = [(other_home, i > j), (other_away, i < j), (other_draw, i == j)]
    for mass, region in regions:
        if mass <= 0:
            continue
        tail = region & ~listed
        target = tail if tail.any() else region
        grid[target] += mass / int(target.sum())

    total = grid.sum()
    if total <= 0:
        raise ValueError("correct-score weights sum to zero")
    return grid / total


def grid_from_prices(prices: dict[str, float], max_goals: int = 10) -> np.ndarray:
    """Build a market joint grid from a ``{selection_name: decimal_odds}`` map.

    Scoreline names like ``"2 - 1"`` become cells; names containing "home win" /
    "away win" / "draw" are treated as the tail buckets. Implied probabilities
    (1/odds) are the weights; the grid normalisation removes the overround.
    """
    scores: dict[tuple[int, int], float] = {}
    other_home = other_away = other_draw = 0.0
    for name, odds in prices.items():
        if odds is None or odds <= 0:
            continue
        implied = 1.0 / odds
        cell = parse_score(name)
        if cell is not None:
            scores[cell] = scores.get(cell, 0.0) + implied
            continue
        low = name.lower()
        if "home" in low:
            other_home += implied
        elif "away" in low:
            other_away += implied
        elif "draw" in low or "tie" in low:
            other_draw += implied
    return correct_score_grid(
        scores, other_home=other_home, other_away=other_away,
        other_draw=other_draw, max_goals=max_goals,
    )
