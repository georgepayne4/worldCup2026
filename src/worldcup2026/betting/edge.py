"""Turn model probabilities + market odds into a disciplined, staked bet sheet.

The pipeline: compute EV per selection, keep only bets clearing an EV threshold,
size them with **fractional Kelly**, then apply exposure caps (per bet, per match,
and across the whole sheet). Fractional Kelly + caps are deliberate: full Kelly on
uncertain probabilities is a fast route to ruin, and our probabilities *are*
uncertain (see ROADMAP P1). Bankroll management is most of what separates a
model that's "right" from one that survives.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class StakingConfig:
    bankroll: float = 1000.0
    kelly_fraction: float = 0.25            # fraction of full Kelly to stake
    min_ev: float = 0.02                    # skip bets below +2% EV
    max_stake_frac: float = 0.05            # cap any single stake at 5% bankroll
    max_match_exposure_frac: float = 0.10   # cap total stake on one match
    max_total_exposure_frac: float = 0.50   # cap total stake across the sheet
    match_col: str = "match_id"


_OUTPUT_COLS = ["fair_odds", "ev", "kelly_full", "stake", "stake_pct"]


def build_bet_sheet(
    candidates: pd.DataFrame, config: StakingConfig | None = None
) -> pd.DataFrame:
    """Rank and stake +EV bets from a candidate table.

    `candidates` needs ``model_prob`` and ``market_odds`` (decimal), plus a match
    id column (``config.match_col``) for the per-match cap. Returns the qualifying
    bets sorted by EV, with ``fair_odds, ev, kelly_full, stake, stake_pct`` added.
    Stakes already reflect fractional Kelly and all exposure caps.
    """
    config = config or StakingConfig()
    for col in ("model_prob", "market_odds"):
        if col not in candidates.columns:
            raise ValueError(f"candidates need a '{col}' column")

    df = candidates.copy()
    df["fair_odds"] = 1.0 / df["model_prob"]
    df["ev"] = df["model_prob"] * df["market_odds"] - 1.0
    df = df[df["ev"] >= config.min_ev].copy()
    if df.empty:
        for col in _OUTPUT_COLS:
            df[col] = pd.Series(dtype=float)
        return df

    # Full Kelly fraction, then scaled down and capped per bet.
    df["kelly_full"] = ((df["ev"]) / (df["market_odds"] - 1.0)).clip(lower=0.0)
    stake_frac = (df["kelly_full"] * config.kelly_fraction).clip(upper=config.max_stake_frac)
    df["stake"] = stake_frac * config.bankroll

    # Per-match exposure cap: scale a match's stakes down to the cap if exceeded.
    if config.match_col in df.columns:
        cap = config.max_match_exposure_frac * config.bankroll
        match_total = df.groupby(config.match_col)["stake"].transform("sum")
        df["stake"] *= np.where(match_total > cap, cap / match_total, 1.0)

    # Total exposure cap across the whole sheet.
    total = df["stake"].sum()
    total_cap = config.max_total_exposure_frac * config.bankroll
    if total > total_cap:
        df["stake"] *= total_cap / total

    df["stake_pct"] = df["stake"] / config.bankroll
    return df.sort_values("ev", ascending=False).reset_index(drop=True)


def bet_sheet_summary(sheet: pd.DataFrame, config: StakingConfig | None = None) -> dict:
    """Headline numbers for a bet sheet: count, exposure, expected profit/ROI."""
    config = config or StakingConfig()
    n = int(len(sheet))
    staked = float(sheet["stake"].sum()) if n else 0.0
    expected_profit = float((sheet["stake"] * sheet["ev"]).sum()) if n else 0.0
    return {
        "n_bets": n,
        "total_staked": staked,
        "exposure_pct": staked / config.bankroll if config.bankroll else float("nan"),
        "expected_profit": expected_profit,
        "expected_roi": expected_profit / staked if staked else float("nan"),
    }


def log_bets(
    sheet: pd.DataFrame, path: str | Path, placed_at: str | None = None
) -> Path:
    """Append a bet sheet to a CSV log (timestamped) for later CLV settlement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = sheet.copy()
    out.insert(0, "placed_at", placed_at or datetime.now(UTC).isoformat())
    out.to_csv(path, mode="a", header=not path.exists(), index=False)
    return path
