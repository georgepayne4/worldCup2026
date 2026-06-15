"""Betfair Exchange adapter (official API).

Pulls Correct Score (and Match Odds) prices from the Betfair Exchange so we can
treat the market's own Correct Score market as the joint score distribution (see
`betting.correct_score`). The Exchange is sharp and low-margin, and — unlike a
sportsbook bet-builder — it's a sanctioned API, so this is a legitimate source of
"true" correlated-market prices.

Auth: set ``BETFAIR_APP_KEY`` and either ``BETFAIR_SESSION_TOKEN`` or
``BETFAIR_USERNAME``/``BETFAIR_PASSWORD`` (interactive login) in ``.env``. Without
them the HTTP calls raise `BetfairError`; the parsing/grid logic is independent
and unit-tested.

Note: the live HTTP calls are written to Betfair's documented contract but are
**unverified here** (no credentials in this environment). The deterministic
parser (`selection_prices`) and grid construction are tested.
"""

from __future__ import annotations

import os

from worldcup2026.betting.correct_score import grid_from_prices
from worldcup2026.data.loaders import RAW_DIR

_IDENTITY_URL = "https://identitysso.betfair.com/api/login"
_BETTING_URL = "https://api.betfair.com/exchange/betting/rest/v1.0"
SOCCER_EVENT_TYPE = "1"
DEFAULT_MARKET_TYPES = ("CORRECT_SCORE", "MATCH_ODDS")


class BetfairError(RuntimeError):
    """Raised when Betfair credentials are missing or the API call fails."""


def _load_env() -> None:
    if not os.environ.get("BETFAIR_APP_KEY"):
        try:
            from dotenv import load_dotenv

            load_dotenv(RAW_DIR.parents[1] / ".env")
        except ModuleNotFoundError:  # pragma: no cover - dotenv is a dep
            pass


def _requests():
    try:
        import requests

        return requests
    except ModuleNotFoundError as exc:  # pragma: no cover - requests is a dep
        raise BetfairError("the `requests` package is required") from exc


def login(app_key: str | None = None, username: str | None = None, password: str | None = None) -> str:
    """Interactive login -> session token. Falls back to env vars."""
    _load_env()
    app_key = app_key or os.environ.get("BETFAIR_APP_KEY")
    username = username or os.environ.get("BETFAIR_USERNAME")
    password = password or os.environ.get("BETFAIR_PASSWORD")
    if not (app_key and username and password):
        raise BetfairError(
            "need BETFAIR_APP_KEY + BETFAIR_USERNAME + BETFAIR_PASSWORD to log in "
            "(or supply BETFAIR_SESSION_TOKEN directly)."
        )
    requests = _requests()
    resp = requests.post(
        _IDENTITY_URL,
        data={"username": username, "password": password},
        headers={"X-Application": app_key, "Accept": "application/json"},
        timeout=30,
    )
    payload = resp.json()
    if payload.get("status") != "SUCCESS":
        raise BetfairError(f"Betfair login failed: {payload.get('error', payload)}")
    return payload["token"]


def _session() -> tuple[str, str]:
    """Return (app_key, session_token), logging in if only credentials are set."""
    _load_env()
    app_key = os.environ.get("BETFAIR_APP_KEY")
    if not app_key:
        raise BetfairError("BETFAIR_APP_KEY not set — see .env / data/betfair.py docstring.")
    token = os.environ.get("BETFAIR_SESSION_TOKEN") or login(app_key)
    return app_key, token


def _post(endpoint: str, body: dict, app_key: str, token: str) -> list | dict:
    requests = _requests()
    resp = requests.post(
        f"{_BETTING_URL}/{endpoint}/",
        json=body,
        headers={
            "X-Application": app_key,
            "X-Authentication": token,
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise BetfairError(f"{endpoint} returned {resp.status_code}: {resp.text[:200]}")
    return resp.json()


def list_market_catalogue(
    *,
    text_query: str = "FIFA World Cup",
    market_type_codes=DEFAULT_MARKET_TYPES,
    max_results: int = 100,
) -> list[dict]:
    """List WC market catalogues (with runner names + event) for the given types."""
    app_key, token = _session()
    body = {
        "filter": {
            "eventTypeIds": [SOCCER_EVENT_TYPE],
            "textQuery": text_query,
            "marketTypeCodes": list(market_type_codes),
        },
        "maxResults": max_results,
        "marketProjection": ["RUNNER_DESCRIPTION", "EVENT", "MARKET_START_TIME"],
    }
    return _post("listMarketCatalogue", body, app_key, token)


def list_market_book(market_ids: list[str]) -> list[dict]:
    """Best back/lay prices for the given market ids."""
    app_key, token = _session()
    body = {"marketIds": market_ids, "priceProjection": {"priceData": ["EX_BEST_OFFERS"]}}
    return _post("listMarketBook", body, app_key, token)


def selection_prices(catalogue_market: dict, market_book: dict) -> dict[str, float]:
    """Map ``runner name -> best back price`` for one market.

    Pure transform of the two payloads (no network) — the tested core. Joins the
    catalogue's runner names to the book's best available-to-back prices by
    selection id.
    """
    names = {r["selectionId"]: r["runnerName"] for r in catalogue_market.get("runners", [])}
    out: dict[str, float] = {}
    for runner in market_book.get("runners", []):
        backs = runner.get("ex", {}).get("availableToBack", [])
        if not backs:
            continue
        name = names.get(runner["selectionId"])
        if name is not None:
            out[name] = float(backs[0]["price"])
    return out


def correct_score_grid_from_betfair(catalogue_market: dict, market_book: dict, max_goals: int = 10):
    """Market joint score grid from a Betfair Correct Score catalogue + book."""
    return grid_from_prices(selection_prices(catalogue_market, market_book), max_goals=max_goals)


def fetch_correct_score_grids(text_query: str = "FIFA World Cup", max_goals: int = 10) -> dict[str, object]:
    """Live: fetch every WC Correct Score market and build a grid per event.

    Returns ``{event_name: grid}``. Requires credentials (raises `BetfairError`
    otherwise). Unverified in this environment — see module docstring.
    """
    catalogues = [
        m for m in list_market_catalogue(market_type_codes=("CORRECT_SCORE",))
        if m.get("marketName", "").upper().startswith("CORRECT SCORE")
        or m.get("description", {}).get("marketType") == "CORRECT_SCORE"
    ]
    books = {b["marketId"]: b for b in list_market_book([m["marketId"] for m in catalogues])}
    grids: dict[str, object] = {}
    for cat in catalogues:
        book = books.get(cat["marketId"])
        if book is None:
            continue
        event = cat.get("event", {}).get("name", cat["marketId"])
        grids[event] = correct_score_grid_from_betfair(cat, book, max_goals)
    return grids
