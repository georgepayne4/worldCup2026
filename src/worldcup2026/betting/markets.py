"""Match markets read off a Dixon-Coles joint score matrix.

The score matrix ``P(home_goals=i, away_goals=j)`` already contains the full
joint distribution of a match, so every standard market is just a sum over the
right cells — no extra modelling. Crucially, a **same-game multi** (legs from the
*same* match) is also a sum over the intersection of cells, which captures the
true correlation between legs. Soft books price these as if the legs were
independent; the gap is the edge (see ``same_game_multi``).

Selections are addressed by ``(market, selection, line)``:

* ``h2h``           — ``Home`` / ``Draw`` / ``Away``
* ``double_chance`` — ``1X`` / ``12`` / ``X2``
* ``totals``        — ``Over`` / ``Under`` at a goals ``line`` (e.g. 2.5)
* ``btts``          — ``Yes`` / ``No``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_TOTAL_LINES = (1.5, 2.5, 3.5)

_LegSpec = tuple  # (market, selection) or (market, selection, line)


def _indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.indices((n, n))


def selection_mask(market: str, selection: str, line: float | None, n: int) -> np.ndarray:
    """Boolean (n, n) grid mask for a market selection (rows=home, cols=away)."""
    i, j = _indices(n)
    total = i + j
    m = market.lower()
    s = str(selection).lower()
    if m in ("h2h", "1x2", "result"):
        if s in ("home", "1"):
            return i > j
        if s in ("draw", "x"):
            return i == j
        if s in ("away", "2"):
            return i < j
    elif m in ("double_chance", "dc"):
        if s in ("1x", "home_or_draw"):
            return i >= j
        if s in ("12", "home_or_away"):
            return i != j
        if s in ("x2", "draw_or_away"):
            return i <= j
    elif m in ("totals", "ou", "over_under"):
        if line is None:
            raise ValueError("totals market requires a line")
        if s == "over":
            return total > line
        if s == "under":
            return total < line
    elif m == "btts":
        both = (i >= 1) & (j >= 1)
        if s in ("yes", "y"):
            return both
        if s in ("no", "n"):
            return ~both
    raise ValueError(f"unknown market/selection: {market}/{selection}")


def probability(matrix: np.ndarray, market: str, selection: str, line: float | None = None) -> float:
    """Model probability of a single market selection."""
    return float(matrix[selection_mask(market, selection, line, matrix.shape[0])].sum())


def match_market_table(
    matrix: np.ndarray,
    *,
    match_id: str | None = None,
    home_team: str | None = None,
    away_team: str | None = None,
    total_lines=DEFAULT_TOTAL_LINES,
) -> pd.DataFrame:
    """All supported single-selection markets for one fixture, with fair odds.

    Columns: ``[match_id, home_team, away_team,] market, selection, line, prob,
    fair_odds``.
    """
    rows: list[dict] = []

    def add(market: str, selection: str, line: float | None = None) -> None:
        p = probability(matrix, market, selection, line)
        rows.append(
            {
                "market": market,
                "selection": selection,
                "line": line,
                "prob": p,
                "fair_odds": (1.0 / p if p > 0 else float("inf")),
            }
        )

    for sel in ("Home", "Draw", "Away"):
        add("h2h", sel)
    for sel in ("1X", "12", "X2"):
        add("double_chance", sel)
    for line in total_lines:
        add("totals", "Over", line)
        add("totals", "Under", line)
    add("btts", "Yes")
    add("btts", "No")

    df = pd.DataFrame(rows)
    meta = {"match_id": match_id, "home_team": home_team, "away_team": away_team}
    for col, val in reversed(meta.items()):
        if val is not None:
            df.insert(0, col, val)
    return df


@dataclass
class MultiQuote:
    """Pricing for a same-game multi (legs within one match)."""

    legs: list[_LegSpec]
    joint_prob: float
    independent_prob: float
    fair_odds: float
    naive_odds: float
    correlation_ratio: float  # joint / independent; >1 = positively correlated


def _normalise_leg(leg: _LegSpec) -> tuple[str, str, float | None]:
    if len(leg) == 2:
        return leg[0], leg[1], None
    return leg[0], leg[1], leg[2]


def same_game_multi(matrix: np.ndarray, legs: list[_LegSpec]) -> MultiQuote:
    """Price a multi of legs from the *same* match using the joint grid.

    Returns the correctly-correlated ``joint_prob`` alongside the
    ``independent_prob`` a leg-multiplying book assumes. ``correlation_ratio``
    > 1 means the legs are positively correlated, so an independence-pricing book
    offers odds that are too generous — the value case.
    """
    if not legs:
        raise ValueError("need at least one leg")
    n = matrix.shape[0]
    combined = np.ones((n, n), dtype=bool)
    independent = 1.0
    normalised: list[_LegSpec] = []
    for leg in legs:
        market, selection, line = _normalise_leg(leg)
        mask = selection_mask(market, selection, line, n)
        combined &= mask
        independent *= float(matrix[mask].sum())
        normalised.append((market, selection, line))
    joint = float(matrix[combined].sum())
    fair = 1.0 / joint if joint > 0 else float("inf")
    naive = 1.0 / independent if independent > 0 else float("inf")
    ratio = joint / independent if independent > 0 else float("nan")
    return MultiQuote(normalised, joint, independent, fair, naive, ratio)


def _leg_label(leg: tuple) -> str:
    market, selection, line = _normalise_leg(leg)
    if market == "totals":
        return f"{selection} {line:.1f}"
    if market == "btts":
        return f"BTTS {selection}"
    return str(selection)


def same_game_multi_pairs(total_line: float = 2.5) -> list[tuple]:
    """Sensible 2-leg same-game combos across different market families.

    Pairs one selection each from {result, total, BTTS} families (never two from
    the same family, which would be nested or mutually exclusive).
    """
    families = [
        [("h2h", "Home"), ("h2h", "Draw"), ("h2h", "Away")],
        [("totals", "Over", total_line), ("totals", "Under", total_line)],
        [("btts", "Yes"), ("btts", "No")],
    ]
    pairs: list[tuple] = []
    for a in range(len(families)):
        for b in range(a + 1, len(families)):
            for leg_a in families[a]:
                for leg_b in families[b]:
                    pairs.append((leg_a, leg_b))
    return pairs


def rank_same_game_multis(
    matrix: np.ndarray, pairs: list[tuple] | None = None, *, match_id: str | None = None
) -> pd.DataFrame:
    """Price candidate same-game multis and rank by correlation premium.

    On a grid whose marginals are the market's (see ``blend.blend_to_market``),
    ``corr_ratio = joint / independent`` is the *pure* correlation effect, and
    ``corr_edge = corr_ratio - 1`` approximates the edge of backing the multi at
    a book that prices its legs independently (before that book's margin).
    Positive = the multi is underpriced by an independence-pricing book.
    """
    pairs = pairs if pairs is not None else same_game_multi_pairs()
    rows = []
    for legs in pairs:
        q = same_game_multi(matrix, list(legs))
        rows.append(
            {
                "match_id": match_id,
                "legs": " + ".join(_leg_label(leg) for leg in legs),
                "joint_prob": q.joint_prob,
                "independent_prob": q.independent_prob,
                "corr_ratio": q.correlation_ratio,
                "corr_edge": q.correlation_ratio - 1.0,
                "fair_odds": q.fair_odds,
            }
        )
    out = pd.DataFrame(rows)
    if match_id is None:
        out = out.drop(columns=["match_id"])
    return out.sort_values("corr_edge", ascending=False).reset_index(drop=True)
