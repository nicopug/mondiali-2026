"""Player-level roster scraper for historical tournaments (Tier 4 enabler).

Scope: WC2018, Euro2020, WC2022, Euro2024.
Output: data/raw/transfermarkt/rosters.parquet (player, slug, position, value).
Reuses cache fast-path machinery from `transfermarkt.py`.
"""
from __future__ import annotations

import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import structlog
from bs4 import BeautifulSoup

from mondiali.data.tm_nations import NATION_TM_IDS
from mondiali.data.transfermarkt import RATE_LIMIT_SECONDS, parse_value_eur

log = structlog.get_logger(__name__)

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

ROSTER_URL_TEMPLATE = (
    "https://www.transfermarkt.com/{slug}/kader/verein/{tm_id}/saison_id/{saison}/plus/1"
)


def _build_roster_url(nation: str, tournament: str) -> str | None:
    """Build Transfermarkt roster URL for a nation + tournament.

    Returns None if nation or tournament is unknown.
    """
    entry = NATION_TM_IDS.get(nation)
    meta = TOURNAMENT_META.get(tournament)
    if entry is None or meta is None:
        return None
    slug, tm_id = entry
    return ROSTER_URL_TEMPLATE.format(slug=slug, tm_id=tm_id, saison=meta["saison_id"])


def _read_cached_roster(slug: str, tournament: str, cache_dir: Path) -> str | None:
    """Read cached roster HTML by {slug}__{tournament}.html pattern.

    Returns None if cache file does not exist.
    """
    p = cache_dir / f"{slug}__{tournament}.html"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return None


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


ROSTER_PARQUET_COLUMNS: list[str] = [
    "nation", "tournament", "tournament_start_date",
    "player_name", "player_url_slug", "position", "market_value_eur",
]


def _fetch_roster_html(url: str, slug: str, tournament: str, cache_dir: Path) -> str | None:
    """Fetch with rate limit + cache-write. Returns None on 4xx fatal or repeated failure."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    time.sleep(RATE_LIMIT_SECONDS)
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "mondiali-research/0.1"})
    except requests.RequestException as e:
        log.warning("roster_fetch_exception", url=url, error=str(e))
        return None
    if resp.status_code == 200:
        html = resp.text
        (cache_dir / f"{slug}__{tournament}.html").write_text(html, encoding="utf-8")
        return html
    log.warning("roster_fetch_non200", url=url, status=resp.status_code)
    return None


def _load_existing_roster_pairs(output_path: Path) -> set[tuple[str, str]]:
    if not output_path.exists():
        return set()
    df = pd.read_parquet(output_path)
    return set(zip(df["nation"].astype(str), df["tournament"].astype(str), strict=True))


def scrape_rosters_all(
    tournaments: Iterable[str],
    nations: Iterable[str] | None,
    cache_dir: Path,
    output_path: Path,
    *,
    resume: bool = True,
) -> int:
    """Scrape player-level rosters for given tournaments × nations.

    Args:
        tournaments: subset of TOURNAMENT_META keys.
        nations: if None, use TOURNAMENT_PARTICIPANTS for each tournament.
        cache_dir: dir for {slug}__{tournament}.html cache.
        output_path: rosters.parquet target.
        resume: skip pairs already in parquet.

    Returns:
        Number of (nation, tournament) pairs newly added to parquet.
    """
    existing = _load_existing_roster_pairs(output_path) if resume else set()
    new_rows: list[dict] = []

    for t in tournaments:
        meta = TOURNAMENT_META.get(t)
        if meta is None:
            log.warning("unknown_tournament", tournament=t)
            continue
        target_nations = (
            list(nations) if nations is not None else TOURNAMENT_PARTICIPANTS.get(t, [])
        )
        for nation in target_nations:
            if (nation, t) in existing:
                continue
            url = _build_roster_url(nation, t)
            if url is None:
                log.warning("no_url_for_nation", nation=nation, tournament=t)
                continue
            slug, _ = NATION_TM_IDS[nation]
            html = (
                _read_cached_roster(slug, t, cache_dir)
                or _fetch_roster_html(url, slug, t, cache_dir)
            )
            if html is None:
                continue
            players = _parse_roster_html(html)
            if not players:
                log.warning("roster_parse_empty", nation=nation, tournament=t)
                continue
            for p in players:
                new_rows.append({
                    "nation": nation,
                    "tournament": t,
                    "tournament_start_date": pd.Timestamp(meta["start"]),
                    "player_name": p.player_name,
                    "player_url_slug": p.player_url_slug,
                    "position": p.position,
                    "market_value_eur": p.market_value_eur,
                })

    if not new_rows:
        log.info("scrape_rosters_no_new_rows", existing_pairs=len(existing))
        return 0

    new_df = pd.DataFrame(new_rows, columns=ROSTER_PARQUET_COLUMNS)
    if output_path.exists():
        old_df = pd.read_parquet(output_path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined = new_df
    combined.to_parquet(output_path, index=False)

    n_pairs_added = combined[["nation", "tournament"]].drop_duplicates().shape[0] - len(existing)
    log.info("scrape_rosters_complete", n_pairs_added=n_pairs_added, n_rows_total=len(combined))
    return n_pairs_added
