"""Dixon-Coles bivariate Poisson model.

Reference: Dixon, M. J., & Coles, S. G. (1997). Modelling association football scores
and inefficiencies in the football betting market. JRSS-C, 46(2), 265-280.

See METHODOLOGY.md §3.2 for the parameterisation used here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DixonColesParams:
    attack: dict[str, float]
    defence: dict[str, float]
    home_advantage: float
    rho: float


def fit(*args, **kwargs) -> DixonColesParams:
    """Fit Dixon-Coles params by MLE with time-decay weighting. v1."""
    raise NotImplementedError


def score_matrix(*args, **kwargs):
    """Return the joint score distribution for a fixture. v1."""
    raise NotImplementedError


def match_probabilities(*args, **kwargs):
    """Return (P_home, P_draw, P_away) from a score matrix. v1."""
    raise NotImplementedError
