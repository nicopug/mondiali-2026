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

import pandas as pd
import pytest

from mondiali.config import CONFIG
from mondiali.features.elo import EloSystem


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
