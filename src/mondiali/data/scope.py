"""Tier 3 scope: lista deterministica di nazionali da scrappare.

Output: union di
- WC2026_QUALIFIED (48 hardcoded, snapshot 2026-04-29)
- top-50 FIFA Elo per anno 2014-2025 (computato da matches.parquet)

Output finale ~70-80 nazionali. Salvato in data/processed/tier3_scope.json
per audit + uso del CLI tm-scrape.
"""
from __future__ import annotations

import pandas as pd

# 48 nazionali qualificate per WC2026 (stato 2026-04-29).
# Tre host (Canada, Mexico, US) + 45 sportive qualifiers.
WC2026_QUALIFIED: list[str] = [
    # Host (3)
    "Canada", "Mexico", "United States",
    # UEFA (16)
    "France", "England", "Spain", "Germany", "Portugal", "Netherlands",
    "Belgium", "Italy", "Croatia", "Switzerland", "Denmark", "Poland",
    "Serbia", "Austria", "Hungary", "Turkey",
    # CONMEBOL (6)
    "Argentina", "Brazil", "Uruguay", "Colombia", "Ecuador", "Peru",
    # CONCACAF (3 + hosts already counted above)
    "Costa Rica", "Panama", "Jamaica",
    # AFC (8)
    "Japan", "South Korea", "Australia", "Saudi Arabia", "Iran",
    "Qatar", "Iraq", "United Arab Emirates",
    # CAF (9)
    "Morocco", "Senegal", "Tunisia", "Algeria", "Egypt", "Nigeria",
    "Ghana", "Cameroon", "Ivory Coast",
    # OFC (1)
    "New Zealand",
    # Inter-confederation playoff potential (2)
    "Wales", "Scotland",
]
assert len(WC2026_QUALIFIED) == 48, f"expected 48, got {len(WC2026_QUALIFIED)}"


def compute_tier3_scope(matches: pd.DataFrame) -> list[str]:
    """Lista deterministica delle nazionali da scrappare per Tier 3.

    Union di WC2026_QUALIFIED + top-50 FIFA Elo per ogni anno 2014-2025.

    Args:
        matches: DataFrame con colonne `date`, `home_team`, `away_team`,
            `home_elo_before`, `away_elo_before`.

    Returns:
        Lista ordinata di nation names, ~70-80 entries.
    """
    df = matches[matches["date"] >= pd.Timestamp("2014-01-01")].copy()
    df["year"] = df["date"].dt.year

    top50_by_year: set[str] = set()
    for _year, grp in df.groupby("year"):
        max_elo_home = grp.groupby("home_team")["home_elo_before"].max()
        max_elo_away = grp.groupby("away_team")["away_elo_before"].max()
        team_elo = pd.concat([max_elo_home, max_elo_away]).groupby(level=0).max()
        top50_by_year.update(team_elo.nlargest(50).index.tolist())

    return sorted(set(WC2026_QUALIFIED) | top50_by_year)
