"""Dixon-Coles bivariate Poisson model.

Reference: Dixon, M. J., & Coles, S. G. (1997). Modelling association football scores
and inefficiencies in the football betting market. JRSS-C, 46(2), 265-280.

See METHODOLOGY.md §3.2 for the parameterisation used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson


@dataclass
class DixonColesParams:
    attack: dict[str, float]
    defence: dict[str, float]
    home_advantage: float = 0.25
    rho: float = -0.1


def tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Low-score correction term for (X=x, Y=y)."""
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def match_rates(
    params: DixonColesParams,
    home_team: str,
    away_team: str,
    neutral: bool = False,
) -> tuple[float, float]:
    """Expected goals (lambda for home, mu for away)."""
    ha = 0.0 if neutral else params.home_advantage
    log_lambda = params.attack[home_team] - params.defence[away_team] + ha
    log_mu = params.attack[away_team] - params.defence[home_team]
    return float(np.exp(log_lambda)), float(np.exp(log_mu))


def score_matrix(lam: float, mu: float, rho: float = -0.1, max_goals: int = 10) -> np.ndarray:
    """Return P(X=i, Y=j) as a (max_goals+1) x (max_goals+1) matrix.

    Rows index home goals, columns index away goals.
    """
    home_probs = poisson.pmf(np.arange(max_goals + 1), lam)
    away_probs = poisson.pmf(np.arange(max_goals + 1), mu)
    matrix = np.outer(home_probs, away_probs)
    for x in (0, 1):
        for y in (0, 1):
            matrix[x, y] *= tau(x, y, lam, mu, rho)
    matrix /= matrix.sum()
    return matrix


def match_probabilities(matrix: np.ndarray) -> tuple[float, float, float]:
    """Return (P_home_win, P_draw, P_away_win) from a score matrix."""
    p_home = float(np.tril(matrix, -1).sum())
    p_draw = float(np.diag(matrix).sum())
    p_away = float(np.triu(matrix, 1).sum())
    return p_home, p_draw, p_away


def over_under_probability(matrix: np.ndarray, line: float = 2.5) -> tuple[float, float]:
    """Return (P_over, P_under) for the given totals line."""
    n = matrix.shape[0]
    i, j = np.indices((n, n))
    totals = i + j
    p_over = float(matrix[totals > line].sum())
    p_under = float(matrix[totals < line].sum())
    return p_over, p_under


def btts_probability(matrix: np.ndarray) -> float:
    """P(both teams score)."""
    return float(matrix[1:, 1:].sum())


def expected_goals(matrix: np.ndarray) -> tuple[float, float]:
    """Return realised E[home_goals], E[away_goals] from the (post-DC-correction) grid."""
    n = matrix.shape[0]
    goals = np.arange(n)
    home_eg = float((matrix.sum(axis=1) * goals).sum())
    away_eg = float((matrix.sum(axis=0) * goals).sum())
    return home_eg, away_eg


def fit(*args, **kwargs):
    """Fit DC params by MLE with time-decay weighting. Lands in next session."""
    raise NotImplementedError
