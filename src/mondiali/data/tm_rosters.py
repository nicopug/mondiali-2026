"""Player-level roster scraper for historical tournaments (Tier 4 enabler).

Scope: WC2018, Euro2020, WC2022, Euro2024.
Output: data/raw/transfermarkt/rosters.parquet (player, slug, position, value).
Reuses cache fast-path machinery from `transfermarkt.py`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from bs4 import BeautifulSoup

from mondiali.data.transfermarkt import parse_value_eur

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


@dataclass(frozen=True)
class RosterPlayer:
    player_name: str
    player_url_slug: str
    position: str
    market_value_eur: int | None


_PLAYER_PROFILE_RE = re.compile(r"^/(?P<slug>[a-z0-9-]+)/profil/spieler/\d+")


def _parse_roster_html(html: str) -> list[RosterPlayer]:
    """Parse a TM kader page → list of RosterPlayer.

    Selects ``table.items`` (modern TM ≥2018) or ``table#kader`` (legacy).
    For each row, extracts name+slug from ``td.hauptlink a[title]``,
    position from the row's second non-hauptlink ``<td>``, and value from
    ``td.rechts.hauptlink`` via ``_parse_value_eur``.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.items") or soup.select_one("table#kader")
    if table is None:
        return []
    out: list[RosterPlayer] = []
    for row in table.select("tbody > tr"):
        link = row.select_one("td.hauptlink a[href*='/profil/spieler/']")
        if link is None:
            continue
        m = _PLAYER_PROFILE_RE.match(link.get("href", ""))
        if not m:
            continue
        slug = m.group("slug")
        name = link.get("title") or link.get_text(strip=True)
        tds = row.find_all("td", recursive=False)
        position = tds[1].get_text(strip=True) if len(tds) >= 2 else ""
        value_cell = row.select_one("td.rechts.hauptlink") or row.select_one("td.rechts")
        value: int | None = None
        if value_cell is not None:
            parsed = parse_value_eur(value_cell.get_text(strip=True))
            value = int(parsed) if parsed is not None else None
        out.append(RosterPlayer(
            player_name=name, player_url_slug=slug,
            position=position, market_value_eur=value,
        ))
    return out
