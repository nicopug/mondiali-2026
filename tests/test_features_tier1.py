"""Test feature builder Tier 1."""
from __future__ import annotations

import pandas as pd
import pytest

from mondiali.features.tier1 import (
    add_days_rest,
    add_tier1_features,
    competition_importance_from_tournament,
)


@pytest.mark.parametrize(
    ("tournament", "expected"),
    [
        ("FIFA World Cup", 4),
        ("FIFA World Cup qualification", 2),
        ("UEFA Euro", 3),
        ("UEFA Euro qualification", 2),
        ("Copa América", 3),
        ("Friendly", 1),
        ("UEFA Nations League", 1),
        ("Random thing", 1),
    ],
)
def test_competition_importance_ordinal(tournament: str, expected: int) -> None:
    """Mappa il torneo a importanza ordinale 1-4."""
    assert competition_importance_from_tournament(tournament) == expected


def test_add_days_rest_first_match_per_team_is_nan() -> None:
    """Il primo match di una squadra non ha storia → days_rest_home/away = NaN."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-10"]),
            "home_team": ["A", "A"],
            "away_team": ["B", "C"],
        }
    )
    result = add_days_rest(df)
    assert pd.isna(result.iloc[0]["days_rest_home"])
    assert pd.isna(result.iloc[0]["days_rest_away"])
    assert result.iloc[1]["days_rest_home"] == 9.0
    assert pd.isna(result.iloc[1]["days_rest_away"])


def test_add_days_rest_tracks_each_team_separately() -> None:
    """A gioca 2020-01-01 e 2020-01-20 (sempre home): days_rest_home seconda riga = 19."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-20"]),
            "home_team": ["A", "A"],
            "away_team": ["B", "C"],
        }
    )
    result = add_days_rest(df)
    assert result.iloc[1]["days_rest_home"] == 19.0


def test_add_days_rest_counts_as_team_regardless_of_home_away() -> None:
    """A gioca 2020-01-01 in casa, 2020-01-10 in trasferta: days_rest_away seconda = 9."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-10"]),
            "home_team": ["A", "B"],
            "away_team": ["C", "A"],
        }
    )
    result = add_days_rest(df)
    assert result.iloc[1]["days_rest_away"] == 9.0


def test_add_days_rest_diff_column_present() -> None:
    """Colonna days_rest_diff = home - away (entrambi i lati NaN ok)."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-05", "2020-01-20"]),
            "home_team": ["A", "B", "A"],
            "away_team": ["B", "A", "B"],
        }
    )
    result = add_days_rest(df)
    assert "days_rest_diff" in result.columns
    assert result.iloc[2]["days_rest_diff"] == pytest.approx(15.0 - 15.0)


def test_add_tier1_features_adds_all_columns() -> None:
    """add_tier1_features aggiunge competition_importance + days_rest_*."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-10"]),
            "home_team": ["A", "A"],
            "away_team": ["B", "C"],
            "tournament": ["FIFA World Cup", "Friendly"],
        }
    )
    result = add_tier1_features(df)
    assert "competition_importance" in result.columns
    assert "days_rest_home" in result.columns
    assert "days_rest_away" in result.columns
    assert "days_rest_diff" in result.columns
    assert result.iloc[0]["competition_importance"] == 4
    assert result.iloc[1]["competition_importance"] == 1
