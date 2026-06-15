"""Tests for MVP-2: derived markets, market-blend, and the acca builder."""

import numpy as np
import pandas as pd
import pytest

from worldcup2026.betting.acca import build_acca, suggest_accas
from worldcup2026.betting.blend import blend_to_market
from worldcup2026.betting.markets import (
    book_multi_prices,
    match_market_table,
    probability,
    rank_same_game_multis,
    same_game_multi,
    same_game_multi_pairs,
    selection_mask,
)
from worldcup2026.models.dixon_coles import (
    match_probabilities,
    over_under_probability,
    score_matrix,
)

MATRIX = score_matrix(1.6, 1.1, rho=-0.08, max_goals=8)


# --- markets ---

def test_h2h_masks_match_dixon_coles_partition():
    p_home, p_draw, p_away = match_probabilities(MATRIX)
    assert abs(probability(MATRIX, "h2h", "Home") - p_home) < 1e-12
    assert abs(probability(MATRIX, "h2h", "Draw") - p_draw) < 1e-12
    assert abs(probability(MATRIX, "h2h", "Away") - p_away) < 1e-12
    assert abs(p_home + p_draw + p_away - 1.0) < 1e-12


def test_double_chance_is_union_of_results():
    p_home = probability(MATRIX, "h2h", "Home")
    p_draw = probability(MATRIX, "h2h", "Draw")
    assert abs(probability(MATRIX, "double_chance", "1X") - (p_home + p_draw)) < 1e-12


def test_totals_and_btts_partition():
    over, under = (
        probability(MATRIX, "totals", "Over", 2.5),
        probability(MATRIX, "totals", "Under", 2.5),
    )
    assert abs(over + under - 1.0) < 1e-12
    assert abs(probability(MATRIX, "btts", "Yes") + probability(MATRIX, "btts", "No") - 1.0) < 1e-12


def test_market_table_fair_odds_reciprocal():
    table = match_market_table(MATRIX, match_id="M1", home_team="A", away_team="B")
    assert {"match_id", "market", "selection", "prob", "fair_odds"} <= set(table.columns)
    row = table.iloc[0]
    assert abs(row["fair_odds"] - 1.0 / row["prob"]) < 1e-9


def test_same_game_multi_nested_legs_equal_subset_prob():
    # Home ⊂ 1X, so the joint equals P(Home) and the legs are positively correlated.
    q = same_game_multi(MATRIX, [("h2h", "Home"), ("double_chance", "1X")])
    assert abs(q.joint_prob - probability(MATRIX, "h2h", "Home")) < 1e-12
    assert q.correlation_ratio > 1.0
    assert q.fair_odds < q.naive_odds  # independence over-prices the multi


def test_same_game_multi_general_bounds():
    q = same_game_multi(MATRIX, [("h2h", "Home"), ("totals", "Over", 2.5)])
    assert 0.0 < q.joint_prob <= probability(MATRIX, "h2h", "Home")
    assert q.joint_prob <= probability(MATRIX, "totals", "Over", 2.5)


def test_selection_mask_rejects_unknown():
    with pytest.raises(ValueError):
        selection_mask("h2h", "Nonsense", None, MATRIX.shape[0])


def test_same_game_multi_pairs_are_cross_family():
    pairs = same_game_multi_pairs(2.5)
    assert len(pairs) == 16  # 3x2 + 3x2 + 2x2
    for a, b in pairs:
        assert a[0] != b[0]  # legs from different market families


def test_book_multi_prices_best_single_book_product():
    rows = []
    def add(book, market, selection, line, price):
        rows.append(dict(home_team="A", away_team="B", bookmaker=book,
                         market=market, selection=selection, line=line, price=price))
    # book1: Home 2.0, Under 1.8 -> 3.60 ; book2: Home 2.1, Under 1.7 -> 3.57
    add("book1", "h2h", "Home", float("nan"), 2.0)
    add("book1", "totals", "Under", 2.5, 1.8)
    add("book2", "h2h", "Home", float("nan"), 2.1)
    add("book2", "totals", "Under", 2.5, 1.7)
    add("book3", "h2h", "Home", float("nan"), 2.5)  # no totals -> can't form the multi
    odds = pd.DataFrame(rows)
    out = book_multi_prices(odds, [(("h2h", "Home"), ("totals", "Under", 2.5))])
    assert len(out) == 1
    row = out.iloc[0]
    assert row["legs"] == "Home + Under 2.5"
    assert abs(row["sgm_price"] - 3.60) < 1e-9  # best (longest) single-book product
    assert row["bookmaker"] == "book1"
    assert row["n_books"] == 2  # book3 lacks a leg


def test_rank_same_game_multis_flags_positive_correlation():
    table = rank_same_game_multis(MATRIX)
    assert table["corr_edge"].iloc[0] == (table["corr_ratio"].iloc[0] - 1.0)
    assert table["corr_edge"].is_monotonic_decreasing  # sorted by edge
    # Over 2.5 and BTTS Yes both need goals -> positively correlated.
    row = table[table["legs"] == "Over 2.5 + BTTS Yes"].iloc[0]
    assert row["corr_ratio"] > 1.0


# --- market-blend (IPF) ---

def test_blend_hits_h2h_target():
    target = (0.60, 0.25, 0.15)
    blended = blend_to_market(MATRIX, h2h=target)
    assert abs(blended.sum() - 1.0) < 1e-9
    assert np.allclose(match_probabilities(blended), target, atol=1e-9)


def test_blend_hits_totals_target():
    blended = blend_to_market(MATRIX, totals={2.5: (0.40, 0.60)})
    over, under = over_under_probability(blended, 2.5)
    assert abs(over - 0.40) < 1e-9 and abs(under - 0.60) < 1e-9


def test_blend_joint_fit_hits_both_margins():
    target = (0.55, 0.27, 0.18)
    blended = blend_to_market(MATRIX, h2h=target, totals={2.5: (0.45, 0.55)})
    assert np.allclose(match_probabilities(blended), target, atol=1e-6)
    over, _ = over_under_probability(blended, 2.5)
    assert abs(over - 0.45) < 1e-6


def test_blend_preserves_correlation_shape_within_region():
    # Rescaling a region shouldn't change the *relative* correct-score shape in it.
    blended = blend_to_market(MATRIX, h2h=(0.6, 0.25, 0.15))
    home = selection_mask("h2h", "Home", None, MATRIX.shape[0])
    ratio = blended[home] / MATRIX[home]
    assert np.allclose(ratio, ratio.flat[0])  # constant scale factor across the region


# --- acca builder ---

def _candidates():
    return pd.DataFrame(
        [
            {"match_id": "M1", "market": "h2h", "selection": "Home", "prob": 0.70, "market_odds": 1.60},
            {"match_id": "M1", "market": "totals", "selection": "Over", "prob": 0.80, "market_odds": 1.30},
            {"match_id": "M2", "market": "h2h", "selection": "Home", "prob": 0.65, "market_odds": 1.70},
            {"match_id": "M3", "market": "double_chance", "selection": "1X", "prob": 0.75, "market_odds": 1.40},
        ]
    )


def test_build_acca_accuracy_one_per_match():
    acca = build_acca(_candidates(), n_legs=3, mode="accuracy", min_prob=0.6)
    assert acca.n_legs == 3
    assert set(acca.legs["match_id"]) == {"M1", "M2", "M3"}  # no fixture reused
    assert abs(acca.combined_prob - (0.80 * 0.75 * 0.65)) < 1e-9
    assert abs(acca.combined_fair_odds - 1.0 / acca.combined_prob) < 1e-9


def test_build_acca_value_mode_positive_ev():
    acca = build_acca(_candidates(), n_legs=3, mode="value", min_prob=0.6)
    assert acca.ev is not None and acca.ev > 0
    assert acca.combined_market_odds > 1.0


def test_suggest_accas_non_overlapping():
    accas = suggest_accas(_candidates(), n_accas=2, n_legs=1, mode="accuracy", min_prob=0.6)
    assert len(accas) == 2
    picked = [a.legs["match_id"].iloc[0] for a in accas]
    assert len(set(picked)) == 2  # different fixtures
