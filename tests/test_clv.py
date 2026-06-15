"""Tests for odds ingestion (data.odds) and the CLV harness (betting.clv)."""

import numpy as np
import pandas as pd

from worldcup2026.betting.clv import (
    attach_clv,
    attach_settlement,
    beat_close,
    clv_ev,
    clv_report,
    no_vig_probabilities,
    settle,
)
from worldcup2026.data.odds import (
    ODDS_COLUMNS,
    best_prices,
    consensus_probabilities,
    load_snapshot,
    market_targets,
    normalize_h2h_selections,
    normalize_odds,
    save_snapshot,
)

# --- odds ingestion ---

_RAW_EVENT = [
    {
        "id": "E1",
        "commence_time": "2026-06-20T18:00:00Z",
        "home_team": "Brazil",
        "away_team": "Serbia",
        "bookmakers": [
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Brazil", "price": 1.5},
                            {"name": "Draw", "price": 4.0},
                            {"name": "Serbia", "price": 7.0},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": 1.9, "point": 2.5},
                            {"name": "Under", "price": 1.95, "point": 2.5},
                        ],
                    },
                ],
            }
        ],
    }
]


def test_normalize_odds_flattens_to_schema():
    df = normalize_odds(_RAW_EVENT, captured_at="2026-06-19T12:00:00Z")
    assert list(df.columns) == ODDS_COLUMNS
    assert len(df) == 5  # 3 h2h + 2 totals
    h2h = df[df["market"] == "h2h"]
    assert set(h2h["selection"]) == {"Brazil", "Draw", "Serbia"}
    assert h2h["line"].isna().all()
    totals = df[df["market"] == "totals"]
    assert (totals["line"] == 2.5).all()
    assert df["price"].dtype.kind == "f"


def test_save_load_snapshot_roundtrip(tmp_path):
    df = normalize_odds(_RAW_EVENT)
    path = save_snapshot(df, tmp_path / "snap.csv")
    back = load_snapshot(path)
    assert list(back.columns) == ODDS_COLUMNS
    assert len(back) == len(df)
    assert back[back["market"] == "totals"]["line"].iloc[0] == 2.5


def test_best_prices_takes_max_per_selection():
    rows = [
        dict(captured_at="t", event_id="E1", commence_time="t", home_team="A",
             away_team="B", bookmaker="b1", market="h2h", selection="A", line=np.nan,
             price=2.0),
        dict(captured_at="t", event_id="E1", commence_time="t", home_team="A",
             away_team="B", bookmaker="b2", market="h2h", selection="A", line=np.nan,
             price=2.3),
    ]
    best = best_prices(pd.DataFrame(rows))
    assert len(best) == 1
    assert best["price"].iloc[0] == 2.3
    assert best["bookmaker"].iloc[0] == "b2"


def _two_book_h2h():
    rows = []
    for book, prices in [("b1", (2.0, 3.5, 4.0)), ("b2", (2.2, 3.4, 4.2))]:
        for sel, price in zip(("Brazil", "Draw", "Serbia"), prices, strict=True):
            rows.append(dict(captured_at="t", event_id="E1", commence_time="t",
                             home_team="Brazil", away_team="Serbia", bookmaker=book,
                             market="h2h", selection=sel, line=np.nan, price=price))
    return pd.DataFrame(rows)


def test_normalize_h2h_selections_maps_team_names():
    out = normalize_h2h_selections(_two_book_h2h())
    assert set(out["selection"]) == {"Home", "Draw", "Away"}


def test_consensus_probabilities_no_vig_and_ordered():
    cons = consensus_probabilities(normalize_h2h_selections(_two_book_h2h()))
    assert abs(cons["novig"].sum() - 1.0) < 1e-9  # one market -> sums to 1
    by_sel = dict(zip(cons["selection"], cons["novig"], strict=True))
    assert by_sel["Home"] > by_sel["Away"]  # Brazil favoured


def test_market_targets_extracts_h2h_and_totals():
    cons = pd.DataFrame(
        [
            ("A", "B", "h2h", "Home", np.nan, 0.5),
            ("A", "B", "h2h", "Draw", np.nan, 0.3),
            ("A", "B", "h2h", "Away", np.nan, 0.2),
            ("A", "B", "totals", "Over", 2.5, 0.55),
            ("A", "B", "totals", "Under", 2.5, 0.45),
        ],
        columns=["home_team", "away_team", "market", "selection", "line", "novig"],
    )
    h2h, totals = market_targets(cons, "A", "B")
    assert h2h == (0.5, 0.3, 0.2)
    assert totals == {2.5: (0.55, 0.45)}


# --- CLV scalar primitives ---

def test_no_vig_probabilities_sum_to_one():
    p = no_vig_probabilities([1.9, 3.5, 4.0])
    assert abs(sum(p) - 1.0) < 1e-12


def test_clv_ev_positive_when_price_beats_fair():
    # fair (no-vig) close = [0.5, 0.25, 0.25]; take 2.2 on the 0.5 shot
    assert abs(clv_ev(2.2, [2.0, 4.0, 4.0], 0) - 0.10) < 1e-9
    # taking exactly the fair price is zero-EV
    assert abs(clv_ev(2.0, [2.0, 4.0, 4.0], 0)) < 1e-9


def test_beat_close_and_settle():
    assert beat_close(2.1, 2.0) and not beat_close(1.9, 2.0)
    assert abs(settle(True, 2.5, stake=2.0) - 3.0) < 1e-9
    assert settle(False, 2.5, stake=2.0) == -2.0


# --- CLV on a bet ledger ---

def _closing():
    rows = [
        ("E1", "h2h", "Home", np.nan, 2.0),
        ("E1", "h2h", "Draw", np.nan, 4.0),
        ("E1", "h2h", "Away", np.nan, 4.0),
        ("E1", "totals", "Over", 2.5, 1.9),
        ("E1", "totals", "Under", 2.5, 1.9),
    ]
    return pd.DataFrame(rows, columns=["event_id", "market", "selection", "line", "price"])


def test_attach_clv_computes_expected_values():
    bets = pd.DataFrame(
        [
            ("E1", "h2h", "Home", np.nan, 2.2),
            ("E1", "totals", "Over", 2.5, 2.0),
            ("E1", "h2h", "Nonexistent", np.nan, 5.0),
        ],
        columns=["event_id", "market", "selection", "line", "price_taken"],
    )
    out = attach_clv(bets, _closing())
    assert abs(out.loc[0, "clv_ev"] - 0.10) < 1e-9
    assert out.loc[0, "beat_close"] is True or out.loc[0, "beat_close"] == True  # noqa: E712
    assert out.loc[0, "closing_price"] == 2.0
    # totals: no-vig of [1.9,1.9] -> 0.5 each; 0.5*2.0-1 = 0
    assert abs(out.loc[1, "clv_ev"]) < 1e-9
    # missing selection -> NaN
    assert np.isnan(out.loc[2, "clv_ev"])


def test_settlement_and_report():
    bets = pd.DataFrame(
        [
            ("E1", "h2h", "Home", np.nan, 2.2),
            ("E1", "totals", "Over", 2.5, 2.0),
        ],
        columns=["event_id", "market", "selection", "line", "price_taken"],
    )
    ledger = attach_clv(bets, _closing())
    results = pd.DataFrame(
        [
            ("E1", "h2h", "Home", np.nan, True),
            ("E1", "totals", "Over", 2.5, False),
        ],
        columns=["event_id", "market", "selection", "line", "won"],
    )
    ledger = attach_settlement(ledger, results)
    # Home won at 2.2 -> +1.2 ; Over lost -> -1.0  => net +0.2 on 2 staked
    assert abs(ledger["profit"].sum() - 0.2) < 1e-9
    rep = clv_report(ledger)
    assert rep["n_bets"] == 2
    assert abs(rep["total_staked"] - 2.0) < 1e-9
    assert abs(rep["roi"] - 0.1) < 1e-9
    assert 0.0 <= rep["beat_close_rate"] <= 1.0
