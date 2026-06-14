"""Scoring rules for probabilistic forecasts.

Log loss, Brier score, ranked probability score (RPS), and reliability curves.
See METHODOLOGY.md §6.
"""

from __future__ import annotations

import numpy as np


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Multinomial log loss for 1X2 forecasts. v1."""
    raise NotImplementedError


def ranked_probability_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """RPS for ordinal outcomes. v1."""
    raise NotImplementedError
