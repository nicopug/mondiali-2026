"""Test scope generator per Tier 3."""
from __future__ import annotations

import pandas as pd

from mondiali.data.scope import WC2026_QUALIFIED, compute_tier3_scope


def test_wc2026_qualified_is_a_list_of_48():
    """48 nazionali qualificate al World Cup 2026 (3 host + 45 sportive qualifiers)."""
    assert isinstance(WC2026_QUALIFIED, list)
    assert len(WC2026_QUALIFIED) == 48
    assert all(isinstance(x, str) for x in WC2026_QUALIFIED)
    assert "United States" in WC2026_QUALIFIED  # host
    assert "Argentina" in WC2026_QUALIFIED  # defending champ


def test_compute_tier3_scope_includes_wc2026_qualified():
    """Lo scope finale deve contenere tutte le 48 WC2026 qualified."""
    matches = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2021-01-01"]),
        "home_team": ["Italy", "Brazil"],
        "away_team": ["France", "Argentina"],
        "home_elo_before": [1900.0, 2000.0],
        "away_elo_before": [1950.0, 1980.0],
    })
    scope = compute_tier3_scope(matches)
    for nation in WC2026_QUALIFIED:
        assert nation in scope


def test_compute_tier3_scope_includes_top_elo_per_year():
    """Una nazionale con Elo alto in un anno 2014+ deve entrare nel top-50 storico."""
    rows = []
    # Synthetic: Spain ha Elo molto alto nel 2017
    for d in pd.date_range("2017-01-01", "2017-12-31", periods=60):
        rows.append({
            "date": d,
            "home_team": "Spain",
            "away_team": "Random Team",
            "home_elo_before": 2100.0,
            "away_elo_before": 1500.0,
        })
    matches = pd.DataFrame(rows)
    scope = compute_tier3_scope(matches)
    assert "Spain" in scope


def test_compute_tier3_scope_excludes_pre_2014():
    """Una nazionale appare SOLO pre-2014: NON deve entrare via top-50 storico
    (ma può entrare comunque se è in WC2026_QUALIFIED)."""
    rows = []
    for d in pd.date_range("2010-01-01", "2010-12-31", periods=60):
        rows.append({
            "date": d,
            "home_team": "Galaxy United",  # nome fittizio non in WC2026
            "away_team": "Foobar FC",
            "home_elo_before": 2200.0,
            "away_elo_before": 1500.0,
        })
    matches = pd.DataFrame(rows)
    scope = compute_tier3_scope(matches)
    assert "Galaxy United" not in scope


def test_compute_tier3_scope_is_sorted_and_unique():
    """Output ordinato e senza duplicati (deterministico)."""
    matches = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01"]),
        "home_team": ["Italy"],
        "away_team": ["France"],
        "home_elo_before": [1900.0],
        "away_elo_before": [1950.0],
    })
    scope = compute_tier3_scope(matches)
    assert scope == sorted(scope)
    assert len(scope) == len(set(scope))
