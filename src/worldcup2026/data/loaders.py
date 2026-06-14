"""Loaders for raw match data, fixture lists, and reference tables.

Data source (v1): the community-maintained `martj42/international_results`
dataset — a single CSV of every international men's result since 1872. It now
also carries the 2026 World Cup fixture list, with unplayed matches scored as
``NA``. That gives us, from one file:

* the historical results we fit Elo / Dixon-Coles on, and
* the WC2026 group-stage fixtures plus whatever has been played so far.

See METHODOLOGY.md §4 for the source list. Download with::

    curl -sSL -o data/raw/international_results.csv \\
        https://raw.githubusercontent.com/martj42/international_results/master/results.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"

RESULTS_CSV = RAW_DIR / "international_results.csv"

# Confederation-championship finals (not their qualifiers). Anything ending in
# "qualification" is handled separately as a qualifier; the World Cup proper is
# its own top tier. Everything else falls back to "friendly".
_CONFEDERATION_FINALS = frozenset(
    {
        "UEFA Euro",
        "Copa América",
        "African Cup of Nations",
        "AFC Asian Cup",
        "Gold Cup",
        "CONCACAF Championship",
        "Oceania Nations Cup",
        "UEFA Nations League",
        "CONCACAF Nations League",
        "Confederations Cup",
    }
)


def classify_importance(tournament: str) -> str:
    """Map a raw `tournament` label to an Elo K-weight tier.

    Tiers (see ``EloConfig.k_weights``): ``world_cup`` > ``confederation`` >
    ``qualifier`` > ``friendly``. The mapping is intentionally coarse — the
    exact bucket for second-tier competitions matters little once time-decay
    weighting is applied.
    """
    if tournament == "FIFA World Cup":
        return "world_cup"
    if tournament.endswith("qualification"):
        return "qualifier"
    if tournament in _CONFEDERATION_FINALS:
        return "confederation"
    return "friendly"


def _read_raw(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else RESULTS_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it first:\n"
            "  curl -sSL -o data/raw/international_results.csv "
            "https://raw.githubusercontent.com/martj42/international_results/"
            "master/results.csv"
        )
    # home_score/away_score are "NA" for unplayed fixtures -> keep as nullable.
    df = pd.read_csv(path, na_values=["NA"])
    df = df.rename(
        columns={
            "home_team": "home",
            "away_team": "away",
            "home_score": "home_goals",
            "away_score": "away_goals",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = df["neutral"].astype(bool)
    df["importance"] = df["tournament"].map(classify_importance)
    return df


def load_international_results(
    path: Path | str | None = None,
    drop_unplayed: bool = True,
) -> pd.DataFrame:
    """Load the international match results table.

    Returns columns: ``date, home, away, home_goals, away_goals, tournament,
    neutral, importance``. By default only played matches are returned (scores
    present, integer-typed); pass ``drop_unplayed=False`` to keep future
    fixtures (their goal columns stay nullable).
    """
    df = _read_raw(path)
    if drop_unplayed:
        df = df.dropna(subset=["home_goals", "away_goals"]).copy()
        df["home_goals"] = df["home_goals"].astype(int)
        df["away_goals"] = df["away_goals"].astype(int)
    cols = [
        "date",
        "home",
        "away",
        "home_goals",
        "away_goals",
        "tournament",
        "neutral",
        "importance",
    ]
    return df[cols].reset_index(drop=True)


def load_fixtures_2026(path: Path | str | None = None) -> pd.DataFrame:
    """Load the 2026 World Cup group-stage fixture list.

    Returns one row per fixture with columns ``date, home, away, home_goals,
    away_goals, neutral, played``. Played fixtures carry integer scores; the
    rest have ``<NA>`` goals and ``played == False``.
    """
    df = _read_raw(path)
    mask = (df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == 2026)
    fixtures = df.loc[mask].copy()
    fixtures["played"] = fixtures["home_goals"].notna() & fixtures["away_goals"].notna()
    fixtures["home_goals"] = fixtures["home_goals"].astype("Int64")
    fixtures["away_goals"] = fixtures["away_goals"].astype("Int64")
    cols = ["date", "home", "away", "home_goals", "away_goals", "neutral", "played"]
    return (
        fixtures[cols]
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )


def derive_groups(fixtures: pd.DataFrame) -> dict[str, list[str]]:
    """Recover the 12 groups of 4 from the round-robin fixture pairings.

    The dataset doesn't label groups, but the group stage is a set of disjoint
    4-team round robins, so teams partition into connected components by who
    they play. Groups are labelled A-L in order of first appearance (Mexico's
    group is A, matching the official draw).
    """
    adjacency: dict[str, set[str]] = {}
    order: list[str] = []
    for home, away in zip(fixtures["home"], fixtures["away"], strict=True):
        for t in (home, away):
            if t not in adjacency:
                adjacency[t] = set()
                order.append(t)
        adjacency[home].add(away)
        adjacency[away].add(home)

    seen: set[str] = set()
    components: list[list[str]] = []
    for team in order:  # deterministic: first-appearance order
        if team in seen:
            continue
        stack = [team]
        component: list[str] = []
        seen.add(team)
        while stack:
            node = stack.pop()
            component.append(node)
            for nbr in adjacency[node]:
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        components.append(component)

    groups = {chr(65 + i): sorted(comp) for i, comp in enumerate(components)}
    sizes = {name: len(teams) for name, teams in groups.items()}
    if len(groups) != 12 or any(s != 4 for s in sizes.values()):
        raise ValueError(f"expected 12 groups of 4, got sizes {sizes}")
    return groups
