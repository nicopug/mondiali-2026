"""Tests for injuries_bootstrap.py."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from mondiali.data.injuries_bootstrap import (
    bootstrap_injuries_for_tournament,
    parse_wikipedia_withdrawals,
)
from mondiali.data.tm_rosters import INJURY_SOURCE_DOMAIN, INJURY_STATUS_DOMAIN

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


def _make_rosters_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[
        "nation", "tournament", "tournament_start_date",
        "player_name", "player_url_slug", "position", "market_value_eur",
    ])


def test_bootstrap_matches_player_via_slug_writes_csv(tmp_path: Path, caplog) -> None:
    rosters = _make_rosters_df([
        {
            "nation": "Spain", "tournament": "wc2018",
            "tournament_start_date": pd.Timestamp("2018-06-14"),
            "player_name": "Dani Carvajal", "player_url_slug": "dani-carvajal",
            "position": "Right-Back", "market_value_eur": 32_000_000,
        },
    ])
    html = (FIXTURE_DIR / "wikipedia_wc2018_squads_excerpt.html").read_text(encoding="utf-8")
    csv_path = tmp_path / "injuries.csv"
    csv_path.write_text(
        "date_of_knowledge,team,tournament,player_name,player_url_slug,market_value_eur,status,source\n",
        encoding="utf-8",
    )
    n_added, n_skipped = bootstrap_injuries_for_tournament(
        tournament="wc2018",
        wikipedia_html=html,
        rosters=rosters,
        csv_path=csv_path,
    )
    assert n_added == 1
    assert n_skipped >= 1  # Laurent Koscielny no-match
    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert df.iloc[0]["team"] == "Spain"
    assert df.iloc[0]["player_url_slug"] == "dani-carvajal"
    assert df.iloc[0]["status"] == "out"
    assert df.iloc[0]["source"] == "wikipedia_squads"
    assert df.iloc[0]["date_of_knowledge"] == "2018-06-13"
    assert int(df.iloc[0]["market_value_eur"]) == 32_000_000


def test_bootstrap_idempotent_no_duplicates(tmp_path: Path) -> None:
    rosters = _make_rosters_df([{
        "nation": "Spain", "tournament": "wc2018",
        "tournament_start_date": pd.Timestamp("2018-06-14"),
        "player_name": "Dani Carvajal", "player_url_slug": "dani-carvajal",
        "position": "Right-Back", "market_value_eur": 32_000_000,
    }])
    html = (FIXTURE_DIR / "wikipedia_wc2018_squads_excerpt.html").read_text(encoding="utf-8")
    csv_path = tmp_path / "injuries.csv"
    csv_path.write_text(
        "date_of_knowledge,team,tournament,player_name,player_url_slug,market_value_eur,status,source\n",
        encoding="utf-8",
    )
    bootstrap_injuries_for_tournament("wc2018", html, rosters, csv_path)
    bootstrap_injuries_for_tournament("wc2018", html, rosters, csv_path)
    df = pd.read_csv(csv_path)
    assert len(df) == 1


def test_status_enum_constants_complete() -> None:
    assert frozenset({"out", "doubtful", "available"}) == INJURY_STATUS_DOMAIN
    assert frozenset({"wikipedia_squads", "manual"}) == INJURY_SOURCE_DOMAIN


def test_bootstrap_date_of_knowledge_is_day_before_kickoff(tmp_path: Path) -> None:
    rosters = _make_rosters_df([{
        "nation": "Spain", "tournament": "wc2022",
        "tournament_start_date": pd.Timestamp("2022-11-20"),
        "player_name": "Dani Carvajal", "player_url_slug": "dani-carvajal",
        "position": "Right-Back", "market_value_eur": 30_000_000,
    }])
    html = """<html><body>
    <h2><span class="mw-headline" id="Withdrawals">Withdrawals</span></h2>
    <ul><li><b>Spain</b>: <a href="/wiki/x">Dani Carvajal</a> withdrew due to injury.</li></ul>
    </body></html>"""
    csv_path = tmp_path / "injuries.csv"
    csv_path.write_text(
        "date_of_knowledge,team,tournament,player_name,player_url_slug,market_value_eur,status,source\n",
        encoding="utf-8",
    )
    bootstrap_injuries_for_tournament("wc2022", html, rosters, csv_path)
    df = pd.read_csv(csv_path)
    assert df.iloc[0]["date_of_knowledge"] == "2022-11-19"
