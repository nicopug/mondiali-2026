"""Framework anti-data-leakage.

Ogni feature deve essere calcolata usando esclusivamente informazioni strettamente
anteriori a `match_date`. Questo file contiene:
1. Una sentinella che verifica l'invariante sull'Elo history (home_elo_before di
   un match alla data D deve essere l'Elo di prima di D, mai di D-stesso o dopo).
2. Hook futuri per Tier 2+ (form, market value, ecc.) — implementati negli STEP
   successivi.

Regola: se `log_loss < 0.92` in validation, questo test framework deve essere
eseguito prima di qualsiasi claim di miglioramento — log-loss troppo basso è
sintomo #1 di leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mondiali.config import CONFIG
from mondiali.features.elo import EloSystem
from mondiali.features.talent import add_talent_features
from mondiali.features.tier3 import TIER3_COLUMNS, add_tier3_features
from mondiali.features.tier4 import add_tier4_features


def _load_processed() -> pd.DataFrame | None:
    """Carica matches.parquet se esiste, altrimenti None."""
    path = CONFIG.data_processed / "matches.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def test_elo_before_is_strictly_pre_match() -> None:
    """Per ogni match, home_elo_before deve essere il rating PRIMA dell'update di
    quel match. Test: ri-simuliamo l'Elo history e confrontiamo.
    """
    df = _load_processed()
    if df is None:
        pytest.skip("data/processed/matches.parquet not found — run build_processed_matches first")

    elo = EloSystem()
    df_sorted = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    expected_home = []
    expected_away = []
    for row in df_sorted.itertuples(index=False):
        expected_home.append(elo.get(row.home_team))
        expected_away.append(elo.get(row.away_team))
        elo.update(
            home=row.home_team,
            away=row.away_team,
            home_goals=int(row.home_score),
            away_goals=int(row.away_score),
            k_factor=float(row.k_factor_used),
            neutral=bool(row.neutral),
        )

    assert df_sorted["home_elo_before"].tolist() == pytest.approx(expected_home, abs=1e-6)
    assert df_sorted["away_elo_before"].tolist() == pytest.approx(expected_away, abs=1e-6)


def test_no_future_matches_in_processed() -> None:
    """matches.parquet non deve contenere partite future (date > oggi)."""
    df = _load_processed()
    if df is None:
        pytest.skip("data/processed/matches.parquet not found")

    today = pd.Timestamp.now().normalize()
    future_rows = df[df["date"] > today]
    assert len(future_rows) == 0, (
        f"Found {len(future_rows)} future matches in processed set — "
        f"likely ingestion bug or unresolved fixtures slipped through"
    )


def test_days_rest_is_strictly_pre_match() -> None:
    """Per ogni match, days_rest_home/away riflette la storia PRIMA di quella data.
    Ri-simuliamo e confrontiamo.
    """
    df = _load_processed()
    if df is None:
        pytest.skip("data/processed/matches.parquet not found")

    df_sorted = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    last_seen: dict[str, pd.Timestamp] = {}
    expected_home: list[float] = []
    expected_away: list[float] = []
    for row in df_sorted.itertuples(index=False):
        prev_h = last_seen.get(row.home_team)
        prev_a = last_seen.get(row.away_team)
        expected_home.append(float("nan") if prev_h is None else (row.date - prev_h).days)
        expected_away.append(float("nan") if prev_a is None else (row.date - prev_a).days)
        last_seen[row.home_team] = row.date
        last_seen[row.away_team] = row.date

    # Confronto con NaN-aware
    h_obs = df_sorted["days_rest_home"].tolist()
    a_obs = df_sorted["days_rest_away"].tolist()
    for obs, exp in zip(h_obs, expected_home, strict=True):
        if pd.isna(exp):
            assert pd.isna(obs)
        else:
            assert obs == pytest.approx(exp)
    for obs, exp in zip(a_obs, expected_away, strict=True):
        if pd.isna(exp):
            assert pd.isna(obs)
        else:
            assert obs == pytest.approx(exp)


def test_tier2_form_5_is_strictly_pre_match() -> None:
    """Per ogni match, home_form_5 e away_form_5 devono usare solo match
    strettamente precedenti. Ri-simuliamo con la stessa logica del builder.
    """
    df = _load_processed()
    if df is None:
        pytest.skip("data/processed/matches.parquet not found")

    if "home_form_5" not in df.columns:
        pytest.skip("Tier 2 features not present in matches.parquet — run build_processed first")

    df_sorted = df.sort_values("date", kind="mergesort").reset_index(drop=True)

    history: dict[str, list[float]] = {}
    expected_home_form: list[float] = []
    expected_away_form: list[float] = []
    for row in df_sorted.itertuples(index=False):
        h_hist = history.get(row.home_team, [])
        a_hist = history.get(row.away_team, [])
        expected_home_form.append(sum(h_hist[-5:]) if h_hist else float("nan"))
        expected_away_form.append(sum(a_hist[-5:]) if a_hist else float("nan"))
        draw = row.home_score == row.away_score
        h_pts = 3.0 if row.home_score > row.away_score else (1.0 if draw else 0.0)
        a_pts = 3.0 if row.away_score > row.home_score else (1.0 if draw else 0.0)
        history.setdefault(row.home_team, []).append(h_pts)
        history.setdefault(row.away_team, []).append(a_pts)

    h_obs = df_sorted["home_form_5"].tolist()
    a_obs = df_sorted["away_form_5"].tolist()
    for obs, exp in zip(h_obs, expected_home_form, strict=True):
        if pd.isna(exp):
            assert pd.isna(obs)
        else:
            assert obs == pytest.approx(exp)
    for obs, exp in zip(a_obs, expected_away_form, strict=True):
        if pd.isna(exp):
            assert pd.isna(obs)
        else:
            assert obs == pytest.approx(exp)


def test_tier3_market_value_strict_pre_match() -> None:
    """Per ogni match con TM non-NaN, snapshot_date deve essere strictly < match_date.

    Re-simula da snapshots.parquet via add_tier3_features (no shortcuts via parquet
    già processato). Salta se snapshots.parquet non esiste (scraper non ancora eseguito).
    """
    parquet = CONFIG.data_processed / "matches.parquet"
    snapshots_path = CONFIG.data_raw / "transfermarkt" / "snapshots.parquet"
    if not parquet.exists() or not snapshots_path.exists():
        pytest.skip("matches.parquet o snapshots.parquet non disponibili")

    matches = pd.read_parquet(parquet)
    matches["date"] = pd.to_datetime(matches["date"])
    snapshots = pd.read_parquet(snapshots_path)
    snapshots["snapshot_date"] = pd.to_datetime(snapshots["snapshot_date"])

    rebuilt = add_tier3_features(
        matches.drop(columns=TIER3_COLUMNS, errors="ignore"),
        snapshots,
    )

    home_age = rebuilt["home_tm_age_days"].dropna()
    away_age = rebuilt["away_tm_age_days"].dropna()
    if len(home_age):
        assert (home_age >= 0).all(), (
            f"negative home_tm_age_days: {home_age[home_age < 0].head()}"
        )
    if len(away_age):
        assert (away_age >= 0).all(), (
            f"negative away_tm_age_days: {away_age[away_age < 0].head()}"
        )

    pre2014 = rebuilt[rebuilt["date"] < "2014-01-01"]
    if len(pre2014):
        assert pre2014["home_market_value_total"].isna().all()
        assert pre2014["away_market_value_total"].isna().all()


def test_tier4_strict_pre_match() -> None:
    """Same-date injury must NOT be counted (strict < on date_of_knowledge)."""
    matches = pd.DataFrame([
        {"date": pd.Timestamp("2022-11-25"), "home_team": "France", "away_team": "Denmark"},
        {"date": pd.Timestamp("2022-11-26"), "home_team": "Spain", "away_team": "Germany"},
    ])
    rosters = pd.DataFrame([
        {
            "nation": "France",
            "tournament": "wc2022",
            "tournament_start_date": pd.Timestamp("2022-11-20"),
            "player_name": f"M{i}",
            "player_url_slug": f"m{i}",
            "position": "MID",
            "market_value_eur": v,
        }
        for i, v in enumerate(
            [100_000_000, 80_000_000, 60_000_000, 40_000_000, 20_000_000], start=1
        )
    ])
    injuries = pd.DataFrame([{
        "date_of_knowledge": pd.Timestamp("2022-11-25"),
        "team": "France", "tournament": "wc2022",
        "player_name": "M1", "player_url_slug": "m1",
        "market_value_eur": 100_000_000, "status": "out", "source": "wikipedia_squads",
    }])
    out = add_tier4_features(matches, rosters, injuries)
    france_row = out[out["home_team"] == "France"].iloc[0]
    assert france_row["home_top5_absent_count"] == 0, (
        "Tier 4 leaked: same-date injury was counted (must be strict <)"
    )



def test_predict_match_strict_pre_form_cache(tmp_path) -> None:
    """build_inference_row must filter form_cache strictly (match_date < target_date).

    Setup: two form_cache states — one that INCLUDES a match on the target date,
    one that excludes it. Inference must produce identical form-5 features
    because the strict-pre filter removes the target-date row.
    """
    from mondiali.inference.predict import build_inference_row
    from mondiali.inference.state import _build_elo_state, _build_form_cache

    target_date = pd.Timestamp("2024-06-15")
    history = pd.DataFrame([{
        "date": pd.Timestamp("2024-05-01"),
        "home_team": "France", "away_team": "Italy",
        "home_score": 2, "away_score": 1,
        "tournament": "Friendly", "neutral": False,
        "competition_importance": 30.0,
        "home_elo_before": 1800.0, "away_elo_before": 1790.0,
    }])
    leaked = pd.concat([history, pd.DataFrame([{
        "date": target_date,
        "home_team": "France", "away_team": "Italy",
        "home_score": 5, "away_score": 0,  # would skew form if leaked
        "tournament": "Friendly", "neutral": False,
        "competition_importance": 30.0,
        "home_elo_before": 1810.0, "away_elo_before": 1800.0,
    }])], ignore_index=True)

    elo_clean = _build_elo_state(history)
    form_clean = _build_form_cache(history)
    elo_leaked = _build_elo_state(leaked)
    form_leaked = _build_form_cache(leaked)

    row_clean = build_inference_row(
        home="France", away="Italy", date=target_date, neutral=False,
        elo_state=elo_clean, form_cache=form_clean, tm_snapshots=None,
    )
    row_leaked = build_inference_row(
        home="France", away="Italy", date=target_date, neutral=False,
        elo_state=elo_leaked, form_cache=form_leaked, tm_snapshots=None,
    )
    for col in ("home_form_5", "home_gd_5", "home_goals_scored_5",
                "home_goals_conceded_5"):
        a = row_clean[col].iloc[0]
        b = row_leaked[col].iloc[0]
        # Either both NaN or equal
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == b, (
            f"Anti-leakage failed: {col} differs ({a} vs {b}) — "
            f"target-date row leaked into form cache"
        )


def test_talent_features_are_nan_or_derived_from_pre_match_values():
    """Talent columns derive ONLY from tier3 market-value cols (already strict
    pre-match). Assert: where talent_gap is non-NaN, both source values exist."""
    matches = _load_processed()
    if matches is None:
        pytest.skip("data/processed/matches.parquet not found")

    out = add_talent_features(matches)
    non_nan = out["talent_gap_top11"].notna()
    assert (out.loc[non_nan, "home_market_value_top11"].notna()).all()
    assert (out.loc[non_nan, "away_market_value_top11"].notna()).all()
    # exact identity with the differential (no extra transformation/leakage)
    recomputed = (
        out.loc[non_nan, "home_market_value_top11"]
        - out.loc[non_nan, "away_market_value_top11"]
    )
    assert np.allclose(out.loc[non_nan, "talent_gap_top11"], recomputed)
