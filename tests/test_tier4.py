"""Tests for features/tier4.py — top-5 absence count + value ratio."""
from __future__ import annotations

import pandas as pd
import pytest

from mondiali.features.tier4 import TIER4_COLUMNS, add_tier4_features


def _make_match(date_str: str, home: str, away: str) -> dict:
    return {"date": pd.Timestamp(date_str), "home_team": home, "away_team": away}


def _make_roster(
    nation: str, tournament: str, start: str, players: list[tuple[str, str, int]]
) -> list[dict]:
    rows = []
    for name, slug, value in players:
        rows.append({
            "nation": nation,
            "tournament": tournament,
            "tournament_start_date": pd.Timestamp(start),
            "player_name": name,
            "player_url_slug": slug,
            "position": "MID",
            "market_value_eur": value,
        })
    return rows


def _make_injury(
    date_of_knowledge: str,
    team: str,
    tournament: str,
    slug: str,
    value: int,
    status: str = "out",
) -> dict:
    return {
        "date_of_knowledge": pd.Timestamp(date_of_knowledge),
        "team": team,
        "tournament": tournament,
        "player_name": slug,
        "player_url_slug": slug,
        "market_value_eur": value,
        "status": status,
        "source": "wikipedia_squads",
    }


def test_top5_count_excludes_status_available() -> None:
    matches = pd.DataFrame([_make_match("2022-11-25", "France", "Denmark")])
    roster_rows = _make_roster("France", "wc2022", "2022-11-20", [
        ("M1", "m1", 100_000_000),
        ("M2", "m2", 80_000_000),
        ("M3", "m3", 60_000_000),
        ("M4", "m4", 40_000_000),
        ("M5", "m5", 20_000_000),
        ("M6", "m6", 10_000_000),
    ])
    rosters = pd.DataFrame(roster_rows)
    injuries = pd.DataFrame([
        _make_injury("2022-11-19", "France", "wc2022", "m1", 100_000_000, "out"),
        _make_injury("2022-11-19", "France", "wc2022", "m6", 10_000_000, "out"),
        _make_injury("2022-11-19", "France", "wc2022", "m2", 80_000_000, "available"),
    ])
    out = add_tier4_features(matches, rosters, injuries)
    assert out.loc[0, "home_top5_absent_count"] == 1
    # Denmark has no roster → NaN per invariant (covered separately).
    assert pd.isna(out.loc[0, "away_top5_absent_count"])


def test_top5_value_ratio_correct() -> None:
    matches = pd.DataFrame([_make_match("2022-11-25", "France", "Denmark")])
    rosters = pd.DataFrame(_make_roster("France", "wc2022", "2022-11-20", [
        ("M1", "m1", 100_000_000),
        ("M2", "m2", 80_000_000),
        ("M3", "m3", 60_000_000),
        ("M4", "m4", 40_000_000),
        ("M5", "m5", 20_000_000),
    ]))
    injuries = pd.DataFrame([
        _make_injury("2022-11-19", "France", "wc2022", "m1", 100_000_000, "out"),
    ])
    out = add_tier4_features(matches, rosters, injuries)
    expected_ratio = 100_000_000 / (100 + 80 + 60 + 40 + 20) / 1_000_000
    assert out.loc[0, "home_value_absent_ratio"] == pytest.approx(expected_ratio)


def test_value_ratio_zero_when_no_absences() -> None:
    matches = pd.DataFrame([_make_match("2022-11-25", "France", "Denmark")])
    rosters = pd.DataFrame(_make_roster("France", "wc2022", "2022-11-20", [
        ("M1", "m1", 100_000_000),
        ("M2", "m2", 80_000_000),
        ("M3", "m3", 60_000_000),
        ("M4", "m4", 40_000_000),
        ("M5", "m5", 20_000_000),
    ]))
    injuries = pd.DataFrame(columns=[
        "date_of_knowledge", "team", "tournament", "player_name",
        "player_url_slug", "market_value_eur", "status", "source",
    ])
    out = add_tier4_features(matches, rosters, injuries)
    assert out.loc[0, "home_top5_absent_count"] == 0
    assert out.loc[0, "home_value_absent_ratio"] == 0.0


def test_pre_2018_match_returns_nan() -> None:
    matches = pd.DataFrame([_make_match("2014-06-15", "Brazil", "Croatia")])
    rosters = pd.DataFrame(_make_roster("Brazil", "wc2022", "2022-11-20", [
        ("M1", "m1", 100_000_000),
        ("M2", "m2", 80_000_000),
        ("M3", "m3", 60_000_000),
        ("M4", "m4", 40_000_000),
        ("M5", "m5", 20_000_000),
    ]))
    injuries = pd.DataFrame(columns=[
        "date_of_knowledge", "team", "tournament", "player_name",
        "player_url_slug", "market_value_eur", "status", "source",
    ])
    out = add_tier4_features(matches, rosters, injuries)
    for col in TIER4_COLUMNS:
        assert pd.isna(out.loc[0, col])


def test_friendly_outside_tournament_returns_nan() -> None:
    """A friendly between WC2022 participants but on a non-tournament date → NaN."""
    matches = pd.DataFrame([_make_match("2023-03-10", "France", "Denmark")])
    rosters = pd.DataFrame(_make_roster("France", "wc2022", "2022-11-20", [
        ("M1", "m1", 100_000_000),
        ("M2", "m2", 80_000_000),
        ("M3", "m3", 60_000_000),
        ("M4", "m4", 40_000_000),
        ("M5", "m5", 20_000_000),
    ]))
    injuries = pd.DataFrame(columns=[
        "date_of_knowledge", "team", "tournament", "player_name",
        "player_url_slug", "market_value_eur", "status", "source",
    ])
    out = add_tier4_features(matches, rosters, injuries)
    for col in TIER4_COLUMNS:
        assert pd.isna(out.loc[0, col])


def test_strict_pre_match_anti_leakage() -> None:
    """An injury with date_of_knowledge == match.date must be IGNORED (strict <)."""
    matches = pd.DataFrame([_make_match("2022-11-25", "France", "Denmark")])
    rosters = pd.DataFrame(_make_roster("France", "wc2022", "2022-11-20", [
        ("M1", "m1", 100_000_000),
        ("M2", "m2", 80_000_000),
        ("M3", "m3", 60_000_000),
        ("M4", "m4", 40_000_000),
        ("M5", "m5", 20_000_000),
    ]))
    injuries = pd.DataFrame([
        _make_injury("2022-11-25", "France", "wc2022", "m1", 100_000_000, "out"),
    ])
    out = add_tier4_features(matches, rosters, injuries)
    assert out.loc[0, "home_top5_absent_count"] == 0


def test_missing_roster_returns_nan_for_that_side() -> None:
    """France roster missing → France-side features NaN; Denmark side OK."""
    matches = pd.DataFrame([_make_match("2022-11-25", "France", "Denmark")])
    rosters = pd.DataFrame(_make_roster("Denmark", "wc2022", "2022-11-20", [
        ("D1", "d1", 50_000_000),
        ("D2", "d2", 40_000_000),
        ("D3", "d3", 30_000_000),
        ("D4", "d4", 20_000_000),
        ("D5", "d5", 10_000_000),
    ]))
    injuries = pd.DataFrame(columns=[
        "date_of_knowledge", "team", "tournament", "player_name",
        "player_url_slug", "market_value_eur", "status", "source",
    ])
    out = add_tier4_features(matches, rosters, injuries)
    assert pd.isna(out.loc[0, "home_top5_absent_count"])
    assert out.loc[0, "away_top5_absent_count"] == 0
