"""Tests for probability calibration (evaluation.calibration, blend.sharpen_1x2)."""

import numpy as np

from worldcup2026.betting.blend import sharpen_1x2
from worldcup2026.evaluation.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    reliability_table,
)
from worldcup2026.evaluation.metrics import log_loss
from worldcup2026.models.dixon_coles import match_probabilities, score_matrix


def test_apply_temperature_identity_and_sharpening():
    p = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
    assert np.allclose(apply_temperature(p, 1.0), p)
    sharp = apply_temperature(p, 0.5)
    assert np.allclose(sharp.sum(axis=1), 1.0)
    assert sharp[:, 0].max() > p[:, 0].max()  # most-likely class gets more mass
    soft = apply_temperature(p, 2.0)
    assert soft[0].max() < p[0].max()  # softening flattens


def test_fit_temperature_recovers_sharpening():
    rng = np.random.default_rng(0)
    sharp = rng.dirichlet([6.0, 3.0, 2.0], size=600)
    outcomes = np.array([rng.choice(3, p=row) for row in sharp])
    blunt = apply_temperature(sharp, 2.5)  # under-confident
    t = fit_temperature(blunt, outcomes)
    assert t < 1.0  # needs sharpening
    assert log_loss(apply_temperature(blunt, t), outcomes) <= log_loss(blunt, outcomes) + 1e-9


def test_ece_zero_for_calibrated_forecasts():
    # A two-class forecast whose predicted prob equals the observed frequency.
    probs = np.array([[0.5, 0.5]] * 100)
    outcomes = np.array([0, 1] * 50)
    assert expected_calibration_error(probs, outcomes) < 0.05


def test_reliability_table_columns():
    probs = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])
    outcomes = np.array([0, 1, 0])
    table = reliability_table(probs, outcomes, n_bins=5)
    assert {"mean_pred", "obs_freq", "count"} <= set(table.columns)
    assert table["count"].sum() == probs.size


def test_sharpen_1x2_raises_favourite_and_preserves_mass():
    matrix = score_matrix(1.7, 1.0, rho=-0.08, max_goals=8)
    p_home0 = match_probabilities(matrix)[0]
    sharp = sharpen_1x2(matrix, 0.5)
    assert abs(sharp.sum() - 1.0) < 1e-9
    assert match_probabilities(sharp)[0] > p_home0  # favourite sharpened up
    assert np.allclose(sharpen_1x2(matrix, 1.0), matrix)  # T=1 no-op
