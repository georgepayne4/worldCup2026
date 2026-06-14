"""Scoring rules for probabilistic forecasts. See METHODOLOGY.md §6."""

from __future__ import annotations

import numpy as np


def log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> float:
    """Multinomial log loss.

    probs: (n, k) probability matrix.
    outcomes: (n,) array of class indices in {0, ..., k-1}.
    """
    probs = np.clip(np.asarray(probs, dtype=float), eps, 1.0)
    outcomes = np.asarray(outcomes)
    return float(-np.log(probs[np.arange(len(outcomes)), outcomes]).mean())


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Multinomial Brier score (lower is better)."""
    probs = np.asarray(probs, dtype=float)
    n = probs.shape[0]
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), outcomes] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def ranked_probability_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """RPS for ordinal outcomes (lower is better).

    Penalises being further from the true ordinal class — the natural score for
    1X2 markets (away/draw/home as 0/1/2).
    """
    probs = np.asarray(probs, dtype=float)
    n, k = probs.shape
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), outcomes] = 1.0
    cum_probs = np.cumsum(probs, axis=1)
    cum_onehot = np.cumsum(onehot, axis=1)
    return float(((cum_probs - cum_onehot) ** 2).sum(axis=1).mean() / (k - 1))
