"""Bootstrap injuries.csv from Wikipedia tournament-squads pages.

Parses the 'Withdrawals' / 'Replacements' / 'Pre-tournament withdrawals' sections.
Matches player_name -> rosters.parquet for slug + market_value. No fuzzy matching:
unmatched names are logged and skipped.
"""
from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import structlog
from bs4 import BeautifulSoup

from mondiali.data.tm_rosters import (
    INJURIES_CSV_COLUMNS,
    TOURNAMENT_META,
)

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class WithdrawalEntry:
    team: str
    player_name: str


WIKIPEDIA_URL_PATTERN = "https://en.wikipedia.org/wiki/{slug}"
TOURNAMENT_WIKIPEDIA_SLUG: dict[str, str] = {
    "wc2018": "2018_FIFA_World_Cup_squads",
    "euro2020": "UEFA_Euro_2020_squads",
    "wc2022": "2022_FIFA_World_Cup_squads",
    "euro2024": "UEFA_Euro_2024_squads",
}


_WITHDRAWAL_HEADLINE_IDS = {"Withdrawals", "Pre-tournament_withdrawals", "Replacements"}
_WITHDRAWAL_VERB_RE = re.compile(
    r"\b(withdrew|withdrawn|withdraw|out due to|injured)\b",
    re.IGNORECASE,
)


def parse_wikipedia_withdrawals(html: str) -> list[WithdrawalEntry]:
    """Extract (team, player_name) entries from the Withdrawals section.

    Strategy: locate ``<h2>`` headlines whose ``id`` is in _WITHDRAWAL_HEADLINE_IDS,
    walk siblings until next h2, parse ``<li>`` with bold team prefix + player link
    + a withdrawal verb in the same line.
    """
    soup = BeautifulSoup(html, "html.parser")
    out: list[WithdrawalEntry] = []
    for span in soup.find_all("span", class_="mw-headline"):
        if span.get("id") not in _WITHDRAWAL_HEADLINE_IDS:
            continue
        h2 = span.find_parent("h2")
        if h2 is None:
            continue
        for sib in h2.find_all_next():
            if sib.name == "h2":
                break
            if sib.name != "li":
                continue
            text = sib.get_text(" ", strip=True)
            if not _WITHDRAWAL_VERB_RE.search(text):
                continue
            bold = sib.find("b")
            link = sib.find("a", href=True)
            if bold is None or link is None:
                continue
            team = bold.get_text(strip=True)
            player_name = link.get_text(strip=True)
            out.append(WithdrawalEntry(team=team, player_name=player_name))
    return out


def _normalize_name(s: str) -> str:
    """Case-insensitive + accent-stripped normalization for exact matching only."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def bootstrap_injuries_for_tournament(
    tournament: str,
    wikipedia_html: str,
    rosters: pd.DataFrame,
    csv_path: Path,
) -> tuple[int, int]:
    """Parse Wikipedia HTML for ``tournament``, match against rosters, write to ``csv_path``.

    Returns ``(n_added, n_skipped_no_match)``.
    """
    meta = TOURNAMENT_META.get(tournament)
    if meta is None:
        raise ValueError(f"unknown tournament: {tournament}")
    start: date = meta["start"]  # type: ignore[assignment]
    date_of_knowledge = start - timedelta(days=1)

    entries = parse_wikipedia_withdrawals(wikipedia_html)

    roster_t = rosters[rosters["tournament"] == tournament].copy()
    roster_t["_team_norm"] = roster_t["nation"].map(_normalize_name)
    roster_t["_player_norm"] = roster_t["player_name"].map(_normalize_name)
    lookup = {
        (row["_team_norm"], row["_player_norm"]): row
        for _, row in roster_t.iterrows()
    }

    existing = (
        pd.read_csv(csv_path)
        if csv_path.exists()
        else pd.DataFrame(columns=INJURIES_CSV_COLUMNS)
    )
    existing_keys = set(
        zip(
            existing.get("team", pd.Series(dtype=str)).astype(str),
            existing.get("tournament", pd.Series(dtype=str)).astype(str),
            existing.get("player_url_slug", pd.Series(dtype=str)).astype(str),
            strict=True,
        )
    )

    new_rows: list[dict] = []
    n_skipped = 0
    for e in entries:
        key = (_normalize_name(e.team), _normalize_name(e.player_name))
        match = lookup.get(key)
        if match is None:
            log.warning(
                "injury_player_no_roster_match",
                team=e.team,
                player=e.player_name,
                tournament=tournament,
            )
            n_skipped += 1
            continue
        slug = match["player_url_slug"]
        dedup_key = (e.team, tournament, slug)
        if dedup_key in existing_keys:
            continue
        existing_keys.add(dedup_key)
        new_rows.append(
            {
                "date_of_knowledge": date_of_knowledge.isoformat(),
                "team": e.team,
                "tournament": tournament,
                "player_name": e.player_name,
                "player_url_slug": slug,
                "market_value_eur": (
                    int(match["market_value_eur"])
                    if pd.notna(match["market_value_eur"])
                    else ""
                ),
                "status": "out",
                "source": "wikipedia_squads",
            }
        )

    if new_rows:
        new_df = pd.DataFrame(new_rows, columns=INJURIES_CSV_COLUMNS)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(csv_path, index=False)

    log.info(
        "injuries_bootstrap_complete",
        tournament=tournament,
        n_parsed=len(entries),
        n_matched=len(new_rows),
        n_skipped_no_match=n_skipped,
    )
    return len(new_rows), n_skipped


def fetch_wikipedia_squads_html(tournament: str, cache_dir: Path) -> str | None:
    """Fetch the Wikipedia squads HTML for ``tournament``, with on-disk cache.

    Returns the page HTML, or ``None`` if the slug is unknown / fetch failed.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{tournament}.html"
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    slug = TOURNAMENT_WIKIPEDIA_SLUG.get(tournament)
    if slug is None:
        log.warning("no_wikipedia_slug", tournament=tournament)
        return None
    url = WIKIPEDIA_URL_PATTERN.format(slug=slug)
    time.sleep(1.0)
    try:
        resp = requests.get(
            url, timeout=30, headers={"User-Agent": "mondiali-research/0.1"}
        )
    except requests.RequestException as e:
        log.warning("wikipedia_fetch_exception", error=str(e), url=url)
        return None
    if resp.status_code != 200:
        log.warning("wikipedia_fetch_non200", status=resp.status_code, url=url)
        return None
    cached.write_text(resp.text, encoding="utf-8")
    return resp.text
