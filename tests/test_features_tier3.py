"""Test feature builder Tier 3."""
from __future__ import annotations

import pandas as pd

from mondiali.features.tier3 import TIER3_COLUMNS, add_tier3_features


def _make_matches(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _make_snapshots(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def test_tier3_columns_constant():
    assert TIER3_COLUMNS == [
        "home_market_value_total", "away_market_value_total",
        "home_market_value_top11", "away_market_value_top11",
        "home_tm_age_days", "away_tm_age_days",
    ]


def test_add_tier3_features_basic_lookup():
    matches = _make_matches([
        {"date": "2018-06-15", "home_team": "Italy", "away_team": "France"},
    ])
    snapshots = _make_snapshots([
        {"nation": "Italy", "year": 2018, "snapshot_date": "2018-05-01",
         "total_value_eur": 500_000_000.0, "top11_value_eur": 400_000_000.0},
        {"nation": "France", "year": 2018, "snapshot_date": "2018-05-01",
         "total_value_eur": 800_000_000.0, "top11_value_eur": 700_000_000.0},
        {"nation": "Italy", "year": 2017, "snapshot_date": "2017-05-01",
         "total_value_eur": 450_000_000.0, "top11_value_eur": 380_000_000.0},
        {"nation": "France", "year": 2017, "snapshot_date": "2017-05-01",
         "total_value_eur": 750_000_000.0, "top11_value_eur": 650_000_000.0},
    ])
    out = add_tier3_features(matches, snapshots)
    assert out.iloc[0]["home_market_value_total"] == 500_000_000.0
    assert out.iloc[0]["away_market_value_total"] == 800_000_000.0
    assert out.iloc[0]["home_tm_age_days"] == 45
    assert out.iloc[0]["away_tm_age_days"] == 45


def test_add_tier3_features_strict_pre_match():
    """Snapshot DOPO match → ignorato (no future leak)."""
    matches = _make_matches([
        {"date": "2018-06-15", "home_team": "Italy", "away_team": "France"},
    ])
    snapshots = _make_snapshots([
        {"nation": "Italy", "year": 2018, "snapshot_date": "2018-08-01",
         "total_value_eur": 500_000_000.0, "top11_value_eur": 400_000_000.0},
        {"nation": "Italy", "year": 2017, "snapshot_date": "2017-05-01",
         "total_value_eur": 450_000_000.0, "top11_value_eur": 380_000_000.0},
        {"nation": "France", "year": 2018, "snapshot_date": "2018-05-01",
         "total_value_eur": 800_000_000.0, "top11_value_eur": 700_000_000.0},
        {"nation": "France", "year": 2017, "snapshot_date": "2017-05-01",
         "total_value_eur": 750_000_000.0, "top11_value_eur": 650_000_000.0},
    ])
    out = add_tier3_features(matches, snapshots)
    assert out.iloc[0]["home_market_value_total"] == 450_000_000.0
    assert out.iloc[0]["home_tm_age_days"] == 410


def test_add_tier3_features_pre_2014_is_nan():
    matches = _make_matches([
        {"date": "2010-06-15", "home_team": "Italy", "away_team": "France"},
    ])
    snapshots = _make_snapshots([
        {"nation": "Italy", "year": 2014, "snapshot_date": "2014-05-01",
         "total_value_eur": 500_000_000.0, "top11_value_eur": 400_000_000.0},
        {"nation": "Italy", "year": 2015, "snapshot_date": "2015-05-01",
         "total_value_eur": 510_000_000.0, "top11_value_eur": 410_000_000.0},
        {"nation": "France", "year": 2014, "snapshot_date": "2014-05-01",
         "total_value_eur": 800_000_000.0, "top11_value_eur": 700_000_000.0},
        {"nation": "France", "year": 2015, "snapshot_date": "2015-05-01",
         "total_value_eur": 810_000_000.0, "top11_value_eur": 710_000_000.0},
    ])
    out = add_tier3_features(matches, snapshots)
    for col in TIER3_COLUMNS:
        assert pd.isna(out.iloc[0][col]), f"{col} should be NaN for pre-2014 match"


def test_add_tier3_features_age_clipping_540():
    """Snapshot più vecchio di 540 giorni → NaN."""
    matches = _make_matches([
        {"date": "2018-06-15", "home_team": "Italy", "away_team": "France"},
    ])
    snapshots = _make_snapshots([
        {"nation": "Italy", "year": 2014, "snapshot_date": "2014-05-01",
         "total_value_eur": 500_000_000.0, "top11_value_eur": 400_000_000.0},
        {"nation": "Italy", "year": 2015, "snapshot_date": "2015-05-01",
         "total_value_eur": 510_000_000.0, "top11_value_eur": 410_000_000.0},
        {"nation": "France", "year": 2018, "snapshot_date": "2018-05-01",
         "total_value_eur": 800_000_000.0, "top11_value_eur": 700_000_000.0},
        {"nation": "France", "year": 2017, "snapshot_date": "2017-05-01",
         "total_value_eur": 750_000_000.0, "top11_value_eur": 650_000_000.0},
    ])
    out = add_tier3_features(matches, snapshots)
    assert pd.isna(out.iloc[0]["home_market_value_total"])
    assert pd.isna(out.iloc[0]["home_tm_age_days"])
    assert out.iloc[0]["away_market_value_total"] == 800_000_000.0


def test_add_tier3_features_hard_floor_excludes_nation():
    """Nazionale con 1 solo snapshot → tutte le sue feature NaN."""
    matches = _make_matches([
        {"date": "2018-06-15", "home_team": "Italy", "away_team": "Eritrea"},
    ])
    snapshots = _make_snapshots([
        {"nation": "Italy", "year": 2018, "snapshot_date": "2018-05-01",
         "total_value_eur": 500_000_000.0, "top11_value_eur": 400_000_000.0},
        {"nation": "Italy", "year": 2017, "snapshot_date": "2017-05-01",
         "total_value_eur": 450_000_000.0, "top11_value_eur": 380_000_000.0},
        {"nation": "Eritrea", "year": 2018, "snapshot_date": "2018-05-01",
         "total_value_eur": 5_000_000.0, "top11_value_eur": 4_000_000.0},
    ])
    out = add_tier3_features(matches, snapshots)
    assert out.iloc[0]["home_market_value_total"] == 500_000_000.0
    assert pd.isna(out.iloc[0]["away_market_value_total"])
    assert pd.isna(out.iloc[0]["away_tm_age_days"])


def test_add_tier3_features_nation_not_in_snapshots():
    """Nazionale completamente assente da snapshots → NaN per quel lato."""
    matches = _make_matches([
        {"date": "2018-06-15", "home_team": "Italy", "away_team": "Anguilla"},
    ])
    snapshots = _make_snapshots([
        {"nation": "Italy", "year": 2018, "snapshot_date": "2018-05-01",
         "total_value_eur": 500_000_000.0, "top11_value_eur": 400_000_000.0},
        {"nation": "Italy", "year": 2017, "snapshot_date": "2017-05-01",
         "total_value_eur": 450_000_000.0, "top11_value_eur": 380_000_000.0},
    ])
    out = add_tier3_features(matches, snapshots)
    assert out.iloc[0]["home_market_value_total"] == 500_000_000.0
    assert pd.isna(out.iloc[0]["away_market_value_total"])
