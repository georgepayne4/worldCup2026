"""Bookmaker odds ingestion.

A thin, provider-agnostic layer that turns a bookmaker feed into a single
**normalized long-format table** the rest of the betting layer consumes:

    captured_at, event_id, commence_time, home_team, away_team,
    bookmaker, market, selection, line, price

`market` is the provider's market key (``h2h`` = 1X2/moneyline, ``totals``,
``spreads``); `selection` is the outcome name (a team, ``Draw``, ``Over`` /
``Under``); `line` is the total/handicap (``NaN`` for ``h2h``); `price` is
decimal odds.

The default source is The Odds API (https://the-odds-api.com), whose free tier
is enough for snapshots. A key is required — set ``ODDS_API_KEY`` in ``.env``.
Historical/closing-line history needs a paid plan or forward-collected
snapshots; everything downstream (CLV, edge) is provider-agnostic and also runs
on snapshots loaded from disk.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from worldcup2026.data.loaders import RAW_DIR

ODDS_DIR = RAW_DIR / "odds"

ODDS_COLUMNS = [
    "captured_at",
    "event_id",
    "commence_time",
    "home_team",
    "away_team",
    "bookmaker",
    "market",
    "selection",
    "line",
    "price",
]

# The Odds API sport key for the World Cup.
WORLD_CUP_SPORT = "soccer_fifa_world_cup"
_ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Provider team spellings -> the dataset's (martj42) canonical names. Keep this
# the single place the two naming worlds are reconciled.
WC_TEAM_ALIASES = {
    "USA": "United States",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}


def apply_team_aliases(odds: pd.DataFrame, aliases: dict[str, str] | None = None) -> pd.DataFrame:
    """Canonicalise provider team names so odds join to dataset fixtures.

    Rewrites `home_team`, `away_team` and (team-name) `selection` values. Totals
    selections (``Over``/``Under``) are untouched.
    """
    aliases = WC_TEAM_ALIASES if aliases is None else aliases
    out = odds.copy()
    for col in ("home_team", "away_team", "selection"):
        if col in out.columns:
            out[col] = out[col].replace(aliases)
    return out


class OddsAPIError(RuntimeError):
    """Raised when the odds provider cannot be reached or is misconfigured."""


def fetch_odds(
    sport_key: str = WORLD_CUP_SPORT,
    *,
    regions: str = "uk,eu",
    markets: str = "h2h,totals",
    odds_format: str = "decimal",
    api_key: str | None = None,
    base_url: str = _ODDS_API_BASE,
) -> list[dict]:
    """Fetch raw event odds from The Odds API. Returns the parsed JSON list.

    `api_key` falls back to the ``ODDS_API_KEY`` environment variable (loaded
    from ``.env`` if present). Network access and a valid key are required; for
    offline work use `load_snapshot`.
    """
    if api_key is None and not os.environ.get("ODDS_API_KEY"):
        try:
            from dotenv import load_dotenv

            load_dotenv(RAW_DIR.parents[1] / ".env")  # repo-root .env
        except ModuleNotFoundError:  # pragma: no cover - dotenv is a dep
            pass
    key = api_key or os.environ.get("ODDS_API_KEY")
    if not key:
        raise OddsAPIError(
            "No odds API key. Set ODDS_API_KEY in .env or pass api_key=... "
            "(or work from a cached snapshot via load_snapshot)."
        )
    try:
        import requests
    except ModuleNotFoundError as exc:  # pragma: no cover - requests is a dep
        raise OddsAPIError("the `requests` package is required to fetch odds") from exc

    url = f"{base_url}/sports/{sport_key}/odds"
    params = {
        "apiKey": key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
    }
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        raise OddsAPIError(f"odds API returned {resp.status_code}: {resp.text[:200]}")
    return resp.json()


def normalize_odds(raw: list[dict], captured_at: str | None = None) -> pd.DataFrame:
    """Flatten a provider event list into the normalized long-format table.

    Accepts The Odds API v4 shape: a list of events, each with `bookmakers` ->
    `markets` -> `outcomes` (name, price, optional point).
    """
    captured_at = captured_at or datetime.now(UTC).isoformat()
    rows: list[dict] = []
    for event in raw:
        base = {
            "captured_at": captured_at,
            "event_id": event.get("id"),
            "commence_time": event.get("commence_time"),
            "home_team": event.get("home_team"),
            "away_team": event.get("away_team"),
        }
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                for outcome in market.get("outcomes", []):
                    rows.append(
                        {
                            **base,
                            "bookmaker": book.get("key"),
                            "market": market.get("key"),
                            "selection": outcome.get("name"),
                            "line": outcome.get("point"),
                            "price": outcome.get("price"),
                        }
                    )
    df = pd.DataFrame(rows, columns=ODDS_COLUMNS)
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True, errors="coerce")
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df


def save_snapshot(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Write a normalized odds table to a timestamped CSV under data/raw/odds/."""
    if path is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = ODDS_DIR / f"odds_{stamp}.csv"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_snapshot(path: str | Path) -> pd.DataFrame:
    """Read a normalized odds snapshot CSV back into the canonical schema."""
    df = pd.read_csv(path)
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True, errors="coerce")
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df[ODDS_COLUMNS]


def snapshot(
    sport_key: str = WORLD_CUP_SPORT,
    *,
    path: str | Path | None = None,
    **fetch_kwargs,
) -> pd.DataFrame:
    """Fetch -> normalize -> save in one call; returns the normalized table."""
    raw = fetch_odds(sport_key, **fetch_kwargs)
    df = normalize_odds(raw)
    save_snapshot(df, path)
    return df


def best_prices(odds: pd.DataFrame) -> pd.DataFrame:
    """Reduce a multi-bookmaker table to the best (max) price per selection.

    Returns one row per (event_id, market, selection, line) with the highest
    available decimal price and the bookmaker offering it — the price a
    line-shopping bettor could actually take.
    """
    key = ["event_id", "market", "selection", "line"]
    working = odds.copy()
    working["line"] = working["line"].fillna(-1.0)  # group NaN lines together
    idx = working.groupby(key, dropna=False)["price"].idxmax()
    out = working.loc[idx].copy()
    out["line"] = out["line"].replace(-1.0, pd.NA)
    return out.reset_index(drop=True)
