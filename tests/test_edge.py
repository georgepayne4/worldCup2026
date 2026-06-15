"""Tests for the edge/staking engine (betting.edge)."""

import pandas as pd

from worldcup2026.betting.edge import (
    StakingConfig,
    bet_sheet_summary,
    build_bet_sheet,
    log_bets,
)


def _candidates():
    return pd.DataFrame(
        [
            {"match_id": "M1", "selection": "A", "model_prob": 0.60, "market_odds": 2.0},
            {"match_id": "M1", "selection": "B", "model_prob": 0.55, "market_odds": 2.0},
            {"match_id": "M2", "selection": "C", "model_prob": 0.40, "market_odds": 2.0},
        ]
    )


def test_filters_below_min_ev_and_computes_fields():
    cfg = StakingConfig(bankroll=1000, kelly_fraction=0.5, min_ev=0.0,
                        max_stake_frac=1.0, max_match_exposure_frac=1.0,
                        max_total_exposure_frac=1.0)
    sheet = build_bet_sheet(_candidates(), cfg)
    # C is -EV (0.4*2-1 = -0.2) -> dropped; A and B remain
    assert set(sheet["selection"]) == {"A", "B"}
    a = sheet[sheet["selection"] == "A"].iloc[0]
    assert abs(a["ev"] - 0.20) < 1e-9
    assert abs(a["fair_odds"] - 1 / 0.60) < 1e-9
    # full Kelly for A = 0.2/(2-1) = 0.2; half-Kelly stake = 0.1*1000 = 100
    assert abs(a["stake"] - 100.0) < 1e-9
    # sorted by EV descending
    assert list(sheet["selection"]) == ["A", "B"]


def test_per_bet_stake_cap():
    cfg = StakingConfig(bankroll=1000, kelly_fraction=1.0, min_ev=0.0,
                        max_stake_frac=0.05, max_match_exposure_frac=1.0,
                        max_total_exposure_frac=1.0)
    # huge edge -> full Kelly large, but capped at 5% = 50
    cand = pd.DataFrame([{"match_id": "M1", "model_prob": 0.9, "market_odds": 3.0}])
    sheet = build_bet_sheet(cand, cfg)
    assert abs(sheet.iloc[0]["stake"] - 50.0) < 1e-9


def test_per_match_exposure_cap_scales_group():
    cfg = StakingConfig(bankroll=1000, kelly_fraction=1.0, min_ev=0.0,
                        max_stake_frac=1.0, max_match_exposure_frac=0.10,
                        max_total_exposure_frac=1.0)
    # two +EV legs on the same match; their stakes must scale to <= 10% = 100
    cand = pd.DataFrame(
        [
            {"match_id": "M1", "model_prob": 0.6, "market_odds": 2.0},
            {"match_id": "M1", "model_prob": 0.6, "market_odds": 2.0},
        ]
    )
    sheet = build_bet_sheet(cand, cfg)
    assert abs(sheet["stake"].sum() - 100.0) < 1e-6


def test_total_exposure_cap_scales_all():
    cfg = StakingConfig(bankroll=1000, kelly_fraction=0.5, min_ev=0.0,
                        max_stake_frac=1.0, max_match_exposure_frac=1.0,
                        max_total_exposure_frac=0.10)
    sheet = build_bet_sheet(_candidates(), cfg)
    assert abs(sheet["stake"].sum() - 100.0) < 1e-6  # total capped at 10%


def test_summary_and_log(tmp_path):
    cfg = StakingConfig(bankroll=1000, kelly_fraction=0.5, min_ev=0.0)
    sheet = build_bet_sheet(_candidates(), cfg)
    rep = bet_sheet_summary(sheet, cfg)
    assert rep["n_bets"] == 2
    assert rep["expected_profit"] > 0
    assert 0 < rep["exposure_pct"] <= 1
    path = log_bets(sheet, tmp_path / "bets.csv")
    assert path.exists()
    logged = pd.read_csv(path)
    assert "placed_at" in logged.columns and len(logged) == 2


def test_empty_sheet_when_no_value():
    cfg = StakingConfig(min_ev=0.5)  # nothing clears +50% EV
    sheet = build_bet_sheet(_candidates(), cfg)
    assert sheet.empty
    assert {"ev", "stake"} <= set(sheet.columns)
