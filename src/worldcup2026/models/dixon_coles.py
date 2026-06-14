"""Dixon-Coles bivariate Poisson model.

Reference: Dixon, M. J., & Coles, S. G. (1997). Modelling association football scores
and inefficiencies in the football betting market. JRSS-C, 46(2), 265-280.

See METHODOLOGY.md §3.2 for the parameterisation used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
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


@dataclass
class MatchPrediction:
    home_team: str
    away_team: str
    expected_home_goals: float
    expected_away_goals: float
    p_home_win: float
    p_draw: float
    p_away_win: float
    p_over_2_5: float
    p_under_2_5: float
    p_btts: float


def predict_match(
    params: DixonColesParams,
    home_team: str,
    away_team: str,
    neutral: bool = False,
    max_goals: int = 10,
) -> MatchPrediction:
    """One-call fixture prediction: 1X2, O/U 2.5, BTTS, and expected goals."""
    lam, mu = match_rates(params, home_team, away_team, neutral=neutral)
    matrix = score_matrix(lam, mu, params.rho, max_goals=max_goals)
    p_h, p_d, p_a = match_probabilities(matrix)
    p_over, p_under = over_under_probability(matrix, line=2.5)
    eh, ea = expected_goals(matrix)
    return MatchPrediction(
        home_team=home_team,
        away_team=away_team,
        expected_home_goals=eh,
        expected_away_goals=ea,
        p_home_win=p_h,
        p_draw=p_d,
        p_away_win=p_a,
        p_over_2_5=p_over,
        p_under_2_5=p_under,
        p_btts=btts_probability(matrix),
    )


def _negative_log_likelihood(
    params_flat: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    hg: np.ndarray,
    ag: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
) -> float:
    attack = params_flat[:n_teams]
    defence = params_flat[n_teams : 2 * n_teams]
    home_advantage = params_flat[2 * n_teams]
    rho = params_flat[2 * n_teams + 1]

    log_lambda = attack[home_idx] - defence[away_idx] + home_advantage
    log_mu = attack[away_idx] - defence[home_idx]
    lam = np.exp(log_lambda)
    mu = np.exp(log_mu)

    log_p = (
        hg * log_lambda - lam - gammaln(hg + 1)
        + ag * log_mu - mu - gammaln(ag + 1)
    )

    tau_arr = np.ones_like(log_p)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    tau_arr[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau_arr[m01] = 1.0 + lam[m01] * rho
    tau_arr[m10] = 1.0 + mu[m10] * rho
    tau_arr[m11] = 1.0 - rho
    tau_arr = np.maximum(tau_arr, 1e-15)
    log_p = log_p + np.log(tau_arr)

    return float(-(weights * log_p).sum())


def fit(
    matches: list[dict],
    teams: list[str] | None = None,
    weights: np.ndarray | None = None,
    init: DixonColesParams | None = None,
) -> DixonColesParams:
    """Fit Dixon-Coles params by maximum likelihood.

    `matches` is a list of dicts with keys: home, away, home_goals, away_goals.
    `weights` lets the caller supply per-match weights (e.g. time-decay — see
    `time_decay_weights`). `init` seeds the optimiser; absent it, we start
    from a neutral guess.

    Identifiability: the model is invariant to a common shift of attack and
    defence vectors. After fitting we re-centre so mean(attack) == 0.
    """
    if teams is None:
        teams = sorted({m["home"] for m in matches} | {m["away"] for m in matches})
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    home_idx = np.array([team_idx[m["home"]] for m in matches], dtype=int)
    away_idx = np.array([team_idx[m["away"]] for m in matches], dtype=int)
    hg = np.array([m["home_goals"] for m in matches], dtype=float)
    ag = np.array([m["away_goals"] for m in matches], dtype=float)

    if weights is None:
        weights = np.ones(len(matches))
    weights = np.asarray(weights, dtype=float)

    if init is None:
        x0 = np.concatenate(
            [np.zeros(n_teams), np.zeros(n_teams), np.array([0.25]), np.array([-0.1])]
        )
    else:
        x0 = np.concatenate(
            [
                np.array([init.attack[t] for t in teams]),
                np.array([init.defence[t] for t in teams]),
                np.array([init.home_advantage]),
                np.array([init.rho]),
            ]
        )

    bounds = (
        [(-3.0, 3.0)] * n_teams        # attack
        + [(-3.0, 3.0)] * n_teams      # defence
        + [(-1.0, 1.0)]                # home_advantage
        + [(-0.4, 0.4)]                # rho — wider risks tau going negative
    )

    result = minimize(
        _negative_log_likelihood,
        x0,
        args=(home_idx, away_idx, hg, ag, weights, n_teams),
        method="L-BFGS-B",
        bounds=bounds,
    )

    attack_flat = result.x[:n_teams]
    defence_flat = result.x[n_teams : 2 * n_teams]
    shift = attack_flat.mean()
    attack_flat = attack_flat - shift
    defence_flat = defence_flat - shift

    return DixonColesParams(
        attack={t: float(attack_flat[i]) for i, t in enumerate(teams)},
        defence={t: float(defence_flat[i]) for i, t in enumerate(teams)},
        home_advantage=float(result.x[2 * n_teams]),
        rho=float(result.x[2 * n_teams + 1]),
    )


def time_decay_weights(
    match_dates,
    reference_date,
    half_life_days: float = 540.0,
) -> np.ndarray:
    """Exponential time-decay weights w_t = exp(-ln 2 * age_days / half_life).

    `match_dates` and `reference_date` accept anything numpy can coerce to
    `datetime64[D]` (ISO strings, datetime, pandas timestamps).
    """
    dates = np.asarray(match_dates, dtype="datetime64[D]")
    ref = np.datetime64(reference_date, "D")
    days_ago = (ref - dates).astype("timedelta64[D]").astype(float)
    xi = np.log(2.0) / half_life_days
    return np.exp(-xi * days_ago)
