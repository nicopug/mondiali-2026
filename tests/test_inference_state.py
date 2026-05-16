"""Tests for inference.state — Elo + form cache persistence."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mondiali.inference.state import (
    ELO_STATE_COLS,
    FORM_CACHE_COLS,
    FORM_WINDOW,
    load_state,
    save_state,
)


def _make_matches() -> pd.DataFrame:
    rows = []
    base_date = pd.Timestamp("2024-01-01")
    for i in range(10):
        rows.append({
            "date": base_date + pd.Timedelta(days=i * 7),
            "home_team": "France" if i % 2 == 0 else "Italy",
            "away_team": "Italy" if i % 2 == 0 else "France",
            "home_score": 1,
            "away_score": 0,
            "tournament": "Friendly",
            "neutral": False,
            "competition_importance": 30.0,
            "home_elo_before": 1800.0,
            "away_elo_before": 1790.0,
        })
    return pd.DataFrame(rows)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    matches = _make_matches()
    save_state(matches, tmp_path)
    elo, form = load_state(tmp_path)
    assert set(elo.columns) == set(ELO_STATE_COLS)
    assert set(form.columns) == set(FORM_CACHE_COLS)
    assert len(elo) == 2  # France + Italy
    assert form.groupby("nation").size().max() <= FORM_WINDOW


def test_form_cache_keeps_most_recent_per_nation(tmp_path: Path) -> None:
    matches = _make_matches()
    save_state(matches, tmp_path)
    _, form = load_state(tmp_path)
    # France's most recent match_date should be in the cache
    france_max = matches[
        (matches["home_team"] == "France") | (matches["away_team"] == "France")
    ]["date"].max()
    france_form_max = form[form["nation"] == "France"]["match_date"].max()
    assert pd.Timestamp(france_form_max) == pd.Timestamp(france_max)


def test_elo_state_has_one_row_per_nation(tmp_path: Path) -> None:
    matches = _make_matches()
    save_state(matches, tmp_path)
    elo, _ = load_state(tmp_path)
    assert elo["nation"].is_unique


def test_form_cache_score_for_perspective(tmp_path: Path) -> None:
    """For a home team, score_for == home_score; for away team, score_for == away_score."""
    matches = pd.DataFrame([{
        "date": pd.Timestamp("2024-06-01"),
        "home_team": "Spain", "away_team": "Germany",
        "home_score": 3, "away_score": 1,
        "tournament": "Friendly", "neutral": False,
        "competition_importance": 30.0,
        "home_elo_before": 1850.0, "away_elo_before": 1830.0,
    }])
    save_state(matches, tmp_path)
    _, form = load_state(tmp_path)
    spain = form[form["nation"] == "Spain"].iloc[0]
    germany = form[form["nation"] == "Germany"].iloc[0]
    assert int(spain["score_for"]) == 3
    assert int(spain["score_against"]) == 1
    assert bool(spain["is_home"]) is True
    assert int(germany["score_for"]) == 1
    assert int(germany["score_against"]) == 3
    assert bool(germany["is_home"]) is False
    assert float(spain["opponent_elo"]) == pytest.approx(1830.0)
    assert float(germany["opponent_elo"]) == pytest.approx(1850.0)
