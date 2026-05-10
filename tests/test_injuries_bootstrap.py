"""Tests for injuries_bootstrap.py."""
from __future__ import annotations

from pathlib import Path

from mondiali.data.injuries_bootstrap import parse_wikipedia_withdrawals

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_parse_wikipedia_withdrawals_section_extracts_entries() -> None:
    html = (FIXTURE_DIR / "wikipedia_wc2018_squads_excerpt.html").read_text(encoding="utf-8")
    entries = parse_wikipedia_withdrawals(html)
    teams_players = {(e.team, e.player_name) for e in entries}
    assert ("Spain", "Dani Carvajal") in teams_players
    assert ("France", "Laurent Koscielny") in teams_players
    # The "selected but Bernd Leno was called" entry must NOT be parsed as withdrawal.
    assert ("Germany", "Manuel Neuer") not in teams_players


def test_parse_wikipedia_returns_empty_when_no_section() -> None:
    html = "<html><body><h2>Squads</h2><p>Nothing here.</p></body></html>"
    assert parse_wikipedia_withdrawals(html) == []
