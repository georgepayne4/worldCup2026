"""Closing-line value (CLV) and P&L backtesting.

CLV — did the price you took beat the eventual closing price? — is the single
best *leading* indicator of long-run betting skill. Realised P&L is mostly noise
over the handful of matches a tournament offers; consistently positive CLV is
signal. So this is the harness the whole MVP is judged against (ROADMAP gate G1).

Two CLV flavours are computed:

* **beat-close** — a simple boolean: was your decimal price higher than the
  selection's closing price?
* **CLV-EV** — expected value of your bet using the **no-vig closing line** as
  the truth: ``fair_close_prob * your_price - 1``. This is the sharper, sized
  measure (a small price edge on a likely outcome is worth more than a big edge
  on a longshot).

Everything operates on the normalized odds schema from ``data.odds`` plus a
``price_taken`` (and optional ``stake``) per bet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from worldcup2026.betting.odds import remove_vig

_MARKET_KEY = ["event_id", "market", "line"]
_SELECTION_KEY = [*_MARKET_KEY, "selection"]
_NO_LINE = -999.0  # sentinel so NaN lines (h2h) group together


def no_vig_probabilities(prices) -> list[float]:
    """Vig-removed (fair) probabilities for a complete set of market prices."""
    return remove_vig(list(prices))


def clv_ev(price_taken: float, closing_prices, selection_idx: int) -> float:
    """EV per unit stake at `price_taken`, using the no-vig close as truth.

    `closing_prices` is the full set of closing decimal prices for the market;
    `selection_idx` indexes the backed selection within it. >0 means you took a
    price the closing line says is +EV.
    """
    fair = no_vig_probabilities(closing_prices)[selection_idx]
    return fair * price_taken - 1.0


def beat_close(price_taken: float, closing_price: float) -> bool:
    """True if the taken price is better (higher) than the closing price."""
    return price_taken > closing_price


def settle(won: bool, price_taken: float, stake: float = 1.0) -> float:
    """Profit for a settled bet (negative = loss of stake)."""
    return stake * (price_taken - 1.0) if won else -float(stake)


def _with_line_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["line"] = out["line"].fillna(_NO_LINE)
    return out


def attach_clv(bets: pd.DataFrame, closing: pd.DataFrame) -> pd.DataFrame:
    """Add `closing_price`, `clv_ev`, `beat_close` to each bet.

    `bets` needs columns ``event_id, market, selection, line, price_taken``.
    `closing` is a normalized odds table with one price per selection for the
    closing line (reduce a multi-book snapshot first, e.g. ``odds.best_prices``
    or a single sharp book). Bets whose market/selection is missing from the
    closing table get NaN CLV.
    """
    closing_s = _with_line_sentinel(closing)
    groups = {key: g for key, g in closing_s.groupby(_MARKET_KEY, dropna=False)}

    closing_price: list[float] = []
    clv: list[float] = []
    beat: list[float] = []
    for bet in _with_line_sentinel(bets).itertuples(index=False):
        group = groups.get((bet.event_id, bet.market, bet.line))
        sels = None if group is None else group["selection"].tolist()
        if group is None or bet.selection not in sels:
            closing_price.append(np.nan)
            clv.append(np.nan)
            beat.append(np.nan)
            continue
        prices = group["price"].tolist()
        idx = sels.index(bet.selection)
        cp = float(prices[idx])
        closing_price.append(cp)
        clv.append(clv_ev(bet.price_taken, prices, idx))
        beat.append(bool(bet.price_taken > cp))

    out = bets.copy()
    out["closing_price"] = closing_price
    out["clv_ev"] = clv
    out["beat_close"] = beat
    return out


def attach_settlement(bets: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Add `won` and `profit` from a results table.

    `results` has ``event_id, market, selection, line, won`` (bool). Stake comes
    from a `stake` column on `bets` if present, else 1.0 unit.
    """
    res = _with_line_sentinel(results)[[*_SELECTION_KEY, "won"]]
    out = _with_line_sentinel(bets).merge(res, on=_SELECTION_KEY, how="left")
    stake = out["stake"] if "stake" in out else pd.Series(1.0, index=out.index)
    out["profit"] = np.where(
        out["won"].fillna(False),
        stake * (out["price_taken"] - 1.0),
        -stake,
    )
    out.loc[out["won"].isna(), "profit"] = np.nan  # unsettled
    out["line"] = out["line"].replace(_NO_LINE, np.nan)
    return out


def clv_report(ledger: pd.DataFrame) -> dict[str, float]:
    """Aggregate a bet ledger into the headline CLV / P&L metrics."""
    n = int(len(ledger))
    report: dict[str, float] = {"n_bets": n}
    if "clv_ev" in ledger:
        clv = ledger["clv_ev"].dropna()
        report["mean_clv_ev"] = float(clv.mean()) if len(clv) else float("nan")
        report["beat_close_rate"] = (
            float(ledger["beat_close"].dropna().mean())
            if ledger["beat_close"].notna().any()
            else float("nan")
        )
    if "profit" in ledger:
        settled = ledger.dropna(subset=["profit"])
        stake = settled["stake"] if "stake" in settled else pd.Series(1.0, index=settled.index)
        staked = float(stake.sum())
        profit = float(settled["profit"].sum())
        report["n_settled"] = int(len(settled))
        report["total_staked"] = staked
        report["total_profit"] = profit
        report["roi"] = profit / staked if staked else float("nan")
    return report
