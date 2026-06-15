"""Tests for Correct Score -> joint grid and the Betfair price parser."""

import numpy as np

from worldcup2026.betting.correct_score import (
    correct_score_grid,
    grid_from_prices,
    parse_score,
)
from worldcup2026.betting.markets import same_game_multi
from worldcup2026.data.betfair import (
    correct_score_grid_from_betfair,
    selection_prices,
)


def test_parse_score():
    assert parse_score("2 - 1") == (2, 1)
    assert parse_score("0 - 0") == (0, 0)
    assert parse_score("Any Other Home Win") is None


def test_correct_score_grid_normalises_and_places_cells():
    scores = {(0, 0): 0.2, (1, 0): 0.1, (0, 1): 0.1, (1, 1): 0.1, (2, 1): 0.1}
    grid = correct_score_grid(scores, max_goals=4)
    assert abs(grid.sum() - 1.0) < 1e-12
    assert abs(grid[0, 0] - 0.2 / 0.6) < 1e-9
    assert abs(grid[2, 1] - 0.1 / 0.6) < 1e-9


def test_tail_buckets_land_in_their_result_region():
    # Only 0-0 explicit (a draw), plus an "Any Other Home Win" bucket.
    grid = correct_score_grid({(0, 0): 0.5}, other_home=0.5, max_goals=2)
    assert abs(grid.sum() - 1.0) < 1e-12
    i, j = np.indices(grid.shape)
    assert abs(grid[i > j].sum() - 0.5) < 1e-9  # all home-win mass in home region
    assert abs(grid[0, 0] - 0.5) < 1e-9
    assert grid[i < j].sum() < 1e-12  # nothing leaked to away region


def test_grid_from_prices_and_combo_pricing():
    prices = {"0 - 0": 2.0, "1 - 0": 4.0, "0 - 1": 4.0, "Any Other Draw": 4.0}
    grid = grid_from_prices(prices, max_goals=6)
    assert abs(grid.sum() - 1.0) < 1e-9
    # price a combo straight off the market joint
    q = same_game_multi(grid, [("totals", "Under", 1.5)])
    assert 0.0 < q.joint_prob < 1.0
    # 0-0 (implied 0.5 of 1.25 total = 0.4) is the only Under-1.5 + draw cell here
    assert q.joint_prob > 0.0


def test_betfair_selection_prices_and_grid():
    catalogue = {
        "runners": [
            {"selectionId": 1, "runnerName": "0 - 0"},
            {"selectionId": 2, "runnerName": "1 - 0"},
            {"selectionId": 3, "runnerName": "Any Other Home Win"},
        ]
    }
    book = {
        "runners": [
            {"selectionId": 1, "ex": {"availableToBack": [{"price": 11.0, "size": 5}]}},
            {"selectionId": 2, "ex": {"availableToBack": [{"price": 8.0, "size": 5}]}},
            {"selectionId": 3, "ex": {"availableToBack": [{"price": 6.0, "size": 5}]}},
            {"selectionId": 9, "ex": {"availableToBack": []}},  # no price -> skipped
        ]
    }
    prices = selection_prices(catalogue, book)
    assert prices == {"0 - 0": 11.0, "1 - 0": 8.0, "Any Other Home Win": 6.0}
    grid = correct_score_grid_from_betfair(catalogue, book, max_goals=6)
    assert abs(grid.sum() - 1.0) < 1e-9
