"""Loaders for raw match data, fixture lists, and reference tables.

Implementations land in v1 — see METHODOLOGY.md §4 for source list.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"


def load_international_results() -> pd.DataFrame:
    """Load the cleaned international match results table."""
    raise NotImplementedError


def load_fixtures_2026() -> pd.DataFrame:
    """Load the official 2026 World Cup fixture list."""
    raise NotImplementedError
