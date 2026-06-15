"""Anchor a model score matrix's marginals to the market.

The closing line is the best single predictor of a football result, but our raw
Dixon-Coles output isn't calibrated to it. So before pricing derived markets we
*blend*: keep the model's joint shape (its correlation structure) but rescale it
so chosen marginals — 1X2, totals — match vig-removed market probabilities.

This is iterative proportional fitting (IPF / Sinkhorn): for each market, which
partitions the grid into regions with target probabilities, scale each region to
its target; alternate across markets until all margins are hit. The result is a
market-anchored joint distribution we can read same-game multis off — so the
*marginals* are the market's, but the *correlations* are the model's. That split
is the whole edge thesis (see ROADMAP.md).
"""

from __future__ import annotations

import numpy as np

from worldcup2026.betting.markets import selection_mask

# A partition is a list of (mask, target_prob) whose masks tile the grid and
# whose targets sum to 1.
_Partition = list[tuple[np.ndarray, float]]


def blend_grid(
    matrix: np.ndarray,
    partitions: list[_Partition],
    *,
    iters: int = 200,
    tol: float = 1e-10,
) -> np.ndarray:
    """IPF a score matrix so every partition's region sums hit their targets.

    Each partition must tile the grid (regions mutually exclusive, exhaustive)
    with targets summing to 1 — true for 1X2, over/under, BTTS. With a single
    partition the result is exact; with several it alternates to a joint fit.
    """
    g = matrix.astype(float).copy()
    g /= g.sum()
    for _ in range(iters):
        worst = 0.0
        for partition in partitions:
            for mask, target in partition:
                current = g[mask].sum()
                worst = max(worst, abs(current - target))
                if current > 0:
                    g[mask] *= target / current
        if worst < tol:
            break
    g /= g.sum()
    return g


def _h2h_partition(n: int, p_home: float, p_draw: float, p_away: float) -> _Partition:
    return [
        (selection_mask("h2h", "Home", None, n), p_home),
        (selection_mask("h2h", "Draw", None, n), p_draw),
        (selection_mask("h2h", "Away", None, n), p_away),
    ]


def _totals_partition(n: int, line: float, p_over: float, p_under: float) -> _Partition:
    return [
        (selection_mask("totals", "Over", line, n), p_over),
        (selection_mask("totals", "Under", line, n), p_under),
    ]


def blend_to_market(
    matrix: np.ndarray,
    *,
    h2h: tuple[float, float, float] | None = None,
    totals: dict[float, tuple[float, float]] | None = None,
    iters: int = 200,
    tol: float = 1e-10,
) -> np.ndarray:
    """Blend toward market marginals.

    `h2h` is a ``(P_home, P_draw, P_away)`` target (already vig-removed). `totals`
    maps a goals line to a ``(P_over, P_under)`` target. Either or both may be
    given; passing neither returns a copy. Targets should be vig-free — use
    ``betting.odds.remove_vig`` on raw prices first.
    """
    n = matrix.shape[0]
    partitions: list[_Partition] = []
    if h2h is not None:
        partitions.append(_h2h_partition(n, *h2h))
    if totals is not None:
        for line, (p_over, p_under) in totals.items():
            partitions.append(_totals_partition(n, line, p_over, p_under))
    if not partitions:
        return matrix.astype(float).copy()
    return blend_grid(matrix, partitions, iters=iters, tol=tol)
