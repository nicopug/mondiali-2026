"""Bootstrap injuries.csv from Wikipedia tournament-squads pages.

Parses the 'Withdrawals' / 'Replacements' / 'Pre-tournament withdrawals' sections.
Matches player_name -> rosters.parquet for slug + market_value. No fuzzy matching:
unmatched names are logged and skipped.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import structlog
from bs4 import BeautifulSoup

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class WithdrawalEntry:
    team: str
    player_name: str


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
