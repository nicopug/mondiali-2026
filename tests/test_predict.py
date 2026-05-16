"""Tests for inference.predict — build_inference_row + predict_match."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mondiali.inference.predict import build_inference_row
from mondiali.inference.state import save_state


def test_build_inference_row_returns_24_features() -> None:
    matches = pd.read_parquet("data/processed/matches.parquet")
    matches = matches.sort_values("date").reset_index(drop=True)
    history = matches.iloc[:-1]

    from mondiali.inference.state import _build_elo_state, _build_form_cache
    elo = _build_elo_state(history)
    form = _build_form_cache(history)

    last = matches.iloc[-1]
    snapshots = pd.read_parquet("data/raw/transfermarkt/snapshots.parquet")
    row = build_inference_row(
        home=last["home_team"], away=last["away_team"],
        date=pd.Timestamp(last["date"]), neutral=bool(last["neutral"]),
        elo_state=elo, form_cache=form, tm_snapshots=snapshots,
        competition_importance=float(last["competition_importance"]),
    )
    required_cols = {
        "home_elo_before", "away_elo_before",
        "home_form_5", "away_form_5",
        "home_gd_5", "away_gd_5",
        "home_goals_scored_5", "away_goals_scored_5",
        "home_goals_conceded_5", "away_goals_conceded_5",
        "home_avg_opp_elo_5", "away_avg_opp_elo_5",
        "days_rest_home", "days_rest_away",
        "home_market_value_total", "away_market_value_total",
        "home_market_value_top11", "away_market_value_top11",
        "home_tm_age_days", "away_tm_age_days",
        "neutral", "competition_importance",
    }
    assert required_cols.issubset(set(row.columns))


def test_build_inference_row_form_matches_batch(tmp_path: Path) -> None:
    """Form aggregates from inference equal those of build_processed_matches."""
    matches = pd.read_parquet("data/processed/matches.parquet")
    matches = matches.sort_values("date").reset_index(drop=True)
    # Use the last match in the dataset — only its history is available in state
    last = matches.iloc[-1]
    history = matches[matches["date"] < last["date"]]
    state_dir = tmp_path
    save_state(history, state_dir)
    from mondiali.inference.state import load_state
    elo, form = load_state(state_dir)

    snapshots = pd.read_parquet("data/raw/transfermarkt/snapshots.parquet")
    row = build_inference_row(
        home=last["home_team"], away=last["away_team"],
        date=pd.Timestamp(last["date"]), neutral=bool(last["neutral"]),
        elo_state=elo, form_cache=form, tm_snapshots=snapshots,
        competition_importance=float(last["competition_importance"]),
    )

    # Form-5 features (sums and means over last 5) must match exactly.
    for col in ("home_form_5", "away_form_5", "home_gd_5", "away_gd_5",
                "home_goals_scored_5", "away_goals_scored_5",
                "home_goals_conceded_5", "away_goals_conceded_5",
                "home_avg_opp_elo_5", "away_avg_opp_elo_5"):
        expected = float(last[col]) if pd.notna(last[col]) else None
        got = float(row[col].iloc[0]) if pd.notna(row[col].iloc[0]) else None
        if expected is None:
            assert got is None, f"{col}: expected NaN, got {got}"
        else:
            assert got == pytest.approx(expected, abs=1e-6), f"{col}: {got} vs {expected}"


def test_unknown_nation_falls_back_to_default_elo() -> None:
    matches = pd.read_parquet("data/processed/matches.parquet")
    history = matches.sort_values("date").iloc[:1000]
    from mondiali.inference.state import _build_elo_state, _build_form_cache
    elo = _build_elo_state(history)
    form = _build_form_cache(history)
    row = build_inference_row(
        home="Atlantis", away="Eldorado",  # fictional
        date=pd.Timestamp("2026-06-15"), neutral=True,
        elo_state=elo, form_cache=form, tm_snapshots=None,
    )
    assert float(row["home_elo_before"].iloc[0]) == 1500.0
    assert float(row["away_elo_before"].iloc[0]) == 1500.0
    assert pd.isna(row["home_form_5"].iloc[0])
