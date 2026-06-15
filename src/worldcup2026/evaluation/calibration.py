"""Probability calibration — temperature scaling and reliability diagnostics.

Gate G1 showed the model is *under-confident*: it shaves favourites and inflates
underdogs (probabilities compressed toward the middle), which manufactures fake
"value" on longshots. Temperature scaling fixes this with a single parameter
``T`` fit on held-out realised outcomes:

    q_i ∝ p_i ** (1 / T)

``T < 1`` sharpens (more confident), ``T > 1`` softens, ``T = 1`` is a no-op.
Fitting ``T`` to minimise out-of-sample log-loss is the standard, principled
recalibration; we then verify it also pulls the model toward the market.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from worldcup2026.evaluation.metrics import log_loss


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Power-scale each row of a probability matrix and renormalise.

    `temperature` < 1 sharpens, > 1 softens. Rows stay valid distributions.
    """
    p = np.clip(np.asarray(probs, dtype=float), 1e-15, 1.0)
    scaled = p ** (1.0 / temperature)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(
    probs: np.ndarray, outcomes: np.ndarray, bounds: tuple[float, float] = (0.3, 3.0)
) -> float:
    """Temperature minimising log-loss on (probs, outcomes). <1 ⇒ under-confident."""
    result = minimize_scalar(
        lambda t: log_loss(apply_temperature(probs, t), outcomes),
        bounds=bounds,
        method="bounded",
    )
    return float(result.x)


def _binary_pairs(probs: np.ndarray, outcomes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten an (n, k) forecast + class labels into pooled one-vs-rest pairs."""
    probs = np.asarray(probs, dtype=float)
    n, k = probs.shape
    onehot = np.zeros((n, k))
    onehot[np.arange(n), np.asarray(outcomes)] = 1.0
    return probs.ravel(), onehot.ravel()


def reliability_table(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Binned reliability (pooled one-vs-rest): predicted vs observed frequency."""
    pred, win = _binary_pairs(probs, outcomes)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(pred, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            rows.append(
                {
                    "bin_lo": edges[b],
                    "bin_hi": edges[b + 1],
                    "mean_pred": float(pred[mask].mean()),
                    "obs_freq": float(win[mask].mean()),
                    "count": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def expected_calibration_error(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> float:
    """Count-weighted mean |predicted - observed| across reliability bins."""
    table = reliability_table(probs, outcomes, n_bins)
    if table.empty:
        return float("nan")
    weight = table["count"] / table["count"].sum()
    return float((weight * (table["mean_pred"] - table["obs_freq"]).abs()).sum())
