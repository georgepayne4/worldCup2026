"""Smoke tests — verifies the package imports cleanly. Real tests land with v1."""

import worldcup2026
from worldcup2026.ratings.elo import expected_score


def test_version():
    assert worldcup2026.__version__ == "0.0.1"


def test_elo_expected_score_symmetric():
    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1600, 1400) > 0.5
    assert expected_score(1400, 1600) < 0.5
