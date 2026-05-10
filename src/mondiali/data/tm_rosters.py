"""Player-level roster scraper for historical tournaments (Tier 4 enabler).

Scope: WC2018, Euro2020, WC2022, Euro2024.
Output: data/raw/transfermarkt/rosters.parquet (player, slug, position, value).
Reuses cache fast-path machinery from `transfermarkt.py`.
"""
from __future__ import annotations

from datetime import date

TOURNAMENT_META: dict[str, dict[str, object]] = {
    "wc2018":   {"start": date(2018, 6, 14), "end": date(2018, 7, 15), "saison_id": 2017},
    "euro2020": {"start": date(2021, 6, 11), "end": date(2021, 7, 11), "saison_id": 2020},
    "wc2022":   {"start": date(2022, 11, 20), "end": date(2022, 12, 18), "saison_id": 2022},
    "euro2024": {"start": date(2024, 6, 14), "end": date(2024, 7, 14), "saison_id": 2023},
}

TOURNAMENT_PARTICIPANTS: dict[str, list[str]] = {
    "wc2018": [
        "Russia", "Saudi Arabia", "Egypt", "Uruguay", "Portugal", "Spain",
        "Morocco", "Iran", "France", "Australia", "Argentina", "Iceland",
        "Peru", "Denmark", "Croatia", "Nigeria", "Costa Rica", "Serbia",
        "Germany", "Mexico", "Brazil", "Switzerland", "Sweden", "South Korea",
        "Belgium", "Panama", "Tunisia", "England", "Colombia", "Japan",
        "Poland", "Senegal",
    ],
    "euro2020": [
        "Italy", "Switzerland", "Turkey", "Wales", "Belgium", "Denmark",
        "Finland", "Russia", "Netherlands", "Ukraine", "Austria", "North Macedonia",
        "England", "Croatia", "Scotland", "Czech Republic", "Spain", "Sweden",
        "Poland", "Slovakia", "France", "Germany", "Hungary", "Portugal",
    ],
    "wc2022": [
        "Qatar", "Ecuador", "Senegal", "Netherlands", "England", "Iran",
        "United States", "Wales", "Argentina", "Saudi Arabia", "Mexico", "Poland",
        "France", "Australia", "Denmark", "Tunisia", "Spain", "Costa Rica",
        "Germany", "Japan", "Belgium", "Canada", "Morocco", "Croatia",
        "Brazil", "Serbia", "Switzerland", "Cameroon", "Portugal", "Ghana",
        "Uruguay", "South Korea",
    ],
    "euro2024": [
        "Germany", "Scotland", "Hungary", "Switzerland", "Spain", "Croatia",
        "Italy", "Albania", "Slovenia", "Denmark", "Serbia", "England",
        "Poland", "Netherlands", "Austria", "France", "Belgium", "Slovakia",
        "Romania", "Ukraine", "Turkey", "Georgia", "Portugal", "Czech Republic",
    ],
}

INJURIES_CSV_COLUMNS: list[str] = [
    "date_of_knowledge", "team", "tournament", "player_name",
    "player_url_slug", "market_value_eur", "status", "source",
]
INJURY_STATUS_DOMAIN: frozenset[str] = frozenset({"out", "doubtful", "available"})
INJURY_SOURCE_DOMAIN: frozenset[str] = frozenset({"wikipedia_squads", "manual"})
