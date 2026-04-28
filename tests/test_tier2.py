"""Test Tier 2 form features (rolling N=5)."""
from __future__ import annotations

import pandas as pd
import pytest

from mondiali.features.tier2 import TIER2_COLUMNS, add_tier2_features


def _build_synthetic_matches() -> pd.DataFrame:
    """6 match: A vs B per 6 date, alternando vincitori. Elo home_/away_before fisso."""
    rows = [
        ("2020-01-01", "A", "B", 2, 0, 1500.0, 1400.0),  # A wins
        ("2020-02-01", "B", "A", 1, 1, 1400.0, 1500.0),  # draw
        ("2020-03-01", "A", "B", 0, 2, 1500.0, 1400.0),  # A loses
        ("2020-04-01", "B", "A", 3, 1, 1400.0, 1500.0),  # A loses (B wins)
        ("2020-05-01", "A", "B", 1, 0, 1500.0, 1400.0),  # A wins
        ("2020-06-01", "B", "A", 0, 0, 1400.0, 1500.0),  # draw
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "date", "home_team", "away_team",
            "home_score", "away_score",
            "home_elo_before", "away_elo_before",
        ],
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_tier2_columns_added() -> None:
    df = _build_synthetic_matches()
    out = add_tier2_features(df)
    for col in TIER2_COLUMNS:
        assert col in out.columns
    assert len(out) == len(df)


def test_tier2_first_match_is_nan_for_team() -> None:
    df = _build_synthetic_matches()
    out = add_tier2_features(df)
    first = out.iloc[0]
    for col in [
        "home_form_5", "home_gd_5", "home_goals_scored_5",
        "home_goals_conceded_5", "home_avg_opp_elo_5",
    ]:
        assert pd.isna(first[col]), f"{col} should be NaN at first match"
    for col in [
        "away_form_5", "away_gd_5", "away_goals_scored_5",
        "away_goals_conceded_5", "away_avg_opp_elo_5",
    ]:
        assert pd.isna(first[col]), f"{col} should be NaN at first match"


def test_tier2_form_5_partial_window() -> None:
    """Al 3° match di team A: ha giocato 2 match precedenti (W e D). form_5 = 3 + 1 = 4."""
    df = _build_synthetic_matches()
    out = add_tier2_features(df)
    third = out.iloc[2]  # 2020-03-01: A home
    assert third["home_form_5"] == 4.0


def test_tier2_form_5_full_window_team_A() -> None:  # noqa: N802
    """Al 6° match (B vs A, 2020-06-01) team A ha 5 match precedenti.
    Sequenza A: W (2-0), D (1-1), L (0-2), L (1-3), W (1-0).
    form = 3+1+0+0+3 = 7. away_form_5 (A è away) = 7."""
    df = _build_synthetic_matches()
    out = add_tier2_features(df)
    sixth = out.iloc[5]
    assert sixth["away_form_5"] == 7.0


def test_tier2_avg_opp_elo_5() -> None:
    """Al 6° match team A ha sempre giocato contro B (Elo 1400). avg_opp_elo = 1400."""
    df = _build_synthetic_matches()
    out = add_tier2_features(df)
    sixth = out.iloc[5]
    assert sixth["away_avg_opp_elo_5"] == pytest.approx(1400.0)


def test_tier2_strict_anteriority() -> None:
    """home_form_5 alla data D non deve usare il match D stesso.
    Regression: closed='left' nel rolling.
    """
    df = _build_synthetic_matches()
    out = add_tier2_features(df)
    # 2° match (B vs A, 2020-02-01): A è away, ha giocato 1 match prima (W 2-0).
    # Atteso: solo W → form=3, NON 4.
    assert out.iloc[1]["away_form_5"] == 3.0


def test_tier2_same_date_ordering() -> None:
    """Two matches for the same team on the same date must use match_idx as tiebreaker:
    the second match's rolling window must include the first match's result.
    """
    rows = [
        ("2020-01-01", "A", "B", 2, 0, 1500.0, 1400.0),  # idx 0: A wins → A=3 pts
        ("2020-01-01", "A", "C", 1, 1, 1500.0, 1450.0),  # idx 1: A draws → A had 1 prior match
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "date", "home_team", "away_team",
            "home_score", "away_score",
            "home_elo_before", "away_elo_before",
        ],
    )
    df["date"] = pd.to_datetime(df["date"])
    out = add_tier2_features(df)
    # On idx 1, A has played idx 0 (W=3 pts) before
    assert out.iloc[1]["home_form_5"] == 3.0
    # idx 0: A has no prior matches → NaN
    assert pd.isna(out.iloc[0]["home_form_5"])
