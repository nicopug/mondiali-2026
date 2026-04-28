"""Test symmetric row builder + XGBoost Poisson training."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mondiali.model.poisson_xgb import (
    SYMMETRIC_FEATURES,
    PoissonXGBModel,
    build_symmetric_rows,
)


def _sample_processed() -> pd.DataFrame:
    """Mini DataFrame con le colonne attese da build_processed_matches."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-10"]),
            "home_team": ["France", "Spain"],
            "away_team": ["Brazil", "France"],
            "home_score": [2, 1],
            "away_score": [1, 1],
            "neutral": [False, True],
            "tournament": ["Friendly", "FIFA World Cup"],
            "home_elo_before": [1900.0, 1850.0],
            "away_elo_before": [1950.0, 1920.0],
            "competition_importance": [1, 4],
            "days_rest_home": [5.0, 30.0],
            "days_rest_away": [7.0, 9.0],
            "days_rest_diff": [-2.0, 21.0],
            "home_form_5": [10.0, 8.0],
            "away_form_5": [6.0, 11.0],
            "home_gd_5": [3.0, 1.0],
            "away_gd_5": [-1.0, 4.0],
            "home_goals_scored_5": [2.0, 1.5],
            "away_goals_scored_5": [1.2, 2.2],
            "home_goals_conceded_5": [0.8, 1.0],
            "away_goals_conceded_5": [1.6, 0.6],
            "home_avg_opp_elo_5": [1500.0, 1480.0],
            "away_avg_opp_elo_5": [1520.0, 1510.0],
        }
    )


def test_build_symmetric_rows_doubles_dataframe() -> None:
    """Ogni match → 2 righe (home perspective, away perspective)."""
    df = _sample_processed()
    X, y = build_symmetric_rows(df)  # noqa: N806
    assert X.shape[0] == 2 * len(df)
    assert y.shape[0] == 2 * len(df)


def test_build_symmetric_rows_targets_are_goals_from_team_perspective() -> None:
    """Riga home-perspective → target = home_score; away-perspective → away_score."""
    df = _sample_processed()
    _, y = build_symmetric_rows(df)
    # Ordine atteso: riga0 home, riga1 away, riga2 home, riga3 away
    assert y.tolist() == [2, 1, 1, 1]


def test_build_symmetric_rows_is_home_flag_alternates() -> None:
    """Colonna is_home = 1 per le righe home-perspective non-neutral, 0 altrimenti.

    Match 0 (non-neutral): home=1, away=0. Match 1 (neutral): home=0, away=0
    perché il vantaggio casa reale non si applica in venue neutrale (vedi
    test_build_symmetric_rows_respects_neutral_flag).
    """
    df = _sample_processed()
    X, _ = build_symmetric_rows(df)  # noqa: N806
    is_home_col = SYMMETRIC_FEATURES.index("is_home")
    assert X[:, is_home_col].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_build_symmetric_rows_flips_elo_per_perspective() -> None:
    """Nella riga home-perspective team_elo = home_elo_before; in away-perspective opposto."""
    df = _sample_processed()
    X, _ = build_symmetric_rows(df)  # noqa: N806
    team_elo_col = SYMMETRIC_FEATURES.index("team_elo")
    opp_elo_col = SYMMETRIC_FEATURES.index("opponent_elo")
    # Match 0: France vs Brazil, France elo 1900, Brazil elo 1950
    assert X[0, team_elo_col] == 1900.0  # home-perspective: team = France
    assert X[0, opp_elo_col] == 1950.0
    assert X[1, team_elo_col] == 1950.0  # away-perspective: team = Brazil
    assert X[1, opp_elo_col] == 1900.0


def test_build_symmetric_rows_respects_neutral_flag() -> None:
    """is_home in venue neutral è 0 per entrambe le prospettive (no vantaggio casa reale)."""
    df = _sample_processed()
    X, _ = build_symmetric_rows(df)  # noqa: N806
    is_home_col = SYMMETRIC_FEATURES.index("is_home")
    # Match 1: FIFA WC, neutral=True → entrambe le righe devono avere is_home=0
    assert X[2, is_home_col] == 0.0
    assert X[3, is_home_col] == 0.0


def test_build_symmetric_rows_shape_width_matches_features() -> None:
    """X.shape[1] deve combaciare con len(SYMMETRIC_FEATURES) — pin contro reorder/drop."""
    df = _sample_processed()
    X, _ = build_symmetric_rows(df)  # noqa: N806
    assert X.shape == (2 * len(df), len(SYMMETRIC_FEATURES))


def test_symmetric_features_column_order_is_pinned() -> None:
    """Pin esplicito sull'ordine — Task 7+ (PoissonXGBModel) e Tasks downstream
    indicizzano per posizione, non per nome. Reorder è breaking change.
    """
    assert SYMMETRIC_FEATURES == [
        "team_elo",
        "opponent_elo",
        "elo_diff_signed",
        "is_home",
        "is_neutral",
        "competition_importance",
        "team_days_rest",
        "opponent_days_rest",
        "team_form_5",
        "opponent_form_5",
        "team_gd_5",
        "opponent_gd_5",
        "team_goals_scored_5",
        "opponent_goals_scored_5",
        "team_goals_conceded_5",
        "opponent_goals_conceded_5",
        "team_avg_opp_elo_5",
        "opponent_avg_opp_elo_5",
    ]


def test_build_symmetric_rows_empty_df_returns_zero_shaped_arrays() -> None:
    """DataFrame vuoto -> X (0, k), y (0,) — niente eccezioni."""
    empty = _sample_processed().iloc[0:0]
    X, y = build_symmetric_rows(empty)  # noqa: N806
    assert X.shape == (0, len(SYMMETRIC_FEATURES))
    assert y.shape == (0,)


def test_build_symmetric_rows_propagates_nan_in_days_rest() -> None:
    """Task 1 lascia NaN in days_rest_* per la prima partita di ogni team. XGBoost
    gestisce NaN nativamente; la funzione deve propagarli senza errori.
    """
    df = _sample_processed()
    df.loc[0, "days_rest_home"] = float("nan")
    X, _ = build_symmetric_rows(df)  # noqa: N806
    rest_col = SYMMETRIC_FEATURES.index("team_days_rest")
    opp_rest_col = SYMMETRIC_FEATURES.index("opponent_days_rest")
    assert np.isnan(X[0, rest_col])  # match0 home-perspective: team_days_rest = NaN
    assert np.isnan(X[1, opp_rest_col])  # match0 away-perspective: opponent_days_rest = NaN


def test_poisson_xgb_fit_returns_self() -> None:
    df = _sample_processed()
    # Duplica con rng per avere abbastanza dati
    df_big = pd.concat([df] * 100, ignore_index=True)
    model = PoissonXGBModel()
    result = model.fit(df_big)
    assert result is model


def test_poisson_xgb_predict_lambda_positive_and_shape() -> None:
    """predict_lambda ritorna (lambda_home, lambda_away) > 0 per ogni match."""
    df = _sample_processed()
    df_big = pd.concat([df] * 100, ignore_index=True)
    model = PoissonXGBModel().fit(df_big)

    lam_h, lam_a = model.predict_lambda(df)
    assert lam_h.shape == (len(df),)
    assert lam_a.shape == (len(df),)
    assert (lam_h > 0).all()
    assert (lam_a > 0).all()


def test_poisson_xgb_predict_before_fit_raises() -> None:
    df = _sample_processed()
    model = PoissonXGBModel()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_lambda(df)


def test_poisson_xgb_save_before_fit_raises(tmp_path: Path) -> None:
    """save() prima di fit() deve sollevare RuntimeError, non scrivere file vuoti."""
    model = PoissonXGBModel()
    with pytest.raises(RuntimeError, match="fit"):
        model.save(tmp_path / "noop.json")


def test_poisson_xgb_load_missing_file_raises(tmp_path: Path) -> None:
    """load() su path inesistente deve sollevare FileNotFoundError chiaro."""
    model = PoissonXGBModel()
    with pytest.raises(FileNotFoundError):
        model.load(tmp_path / "nope.json")


def test_poisson_xgb_fit_with_early_stopping_completes(tmp_path: Path) -> None:
    """Path early_stopping_val: training termina e booster_ è popolato.

    Task 11 (training pipeline) userà early stopping con walk-forward CV;
    questo test pinia che il branch eval_set non rompa l'API.
    """
    df = _sample_processed()
    df_big = pd.concat([df] * 100, ignore_index=True)
    df_val = pd.concat([df] * 20, ignore_index=True)
    model = PoissonXGBModel({"n_estimators": 50}).fit(
        df_big, early_stopping_val=df_val, early_stopping_rounds=5
    )
    assert model.booster_ is not None
    lam_h, lam_a = model.predict_lambda(df)
    assert lam_h.shape == (len(df),)
    assert (lam_h > 0).all()
    assert (lam_a > 0).all()


def test_poisson_xgb_json_serialization_roundtrip(tmp_path: Path) -> None:
    """Serializzazione JSON nativa: dopo save/load predict_lambda è identico."""
    df = _sample_processed()
    df_big = pd.concat([df] * 50, ignore_index=True)
    model = PoissonXGBModel().fit(df_big)
    lam_h_before, lam_a_before = model.predict_lambda(df)

    json_path = tmp_path / "model.json"
    model.save(json_path)

    loaded = PoissonXGBModel()
    loaded.load(json_path)
    lam_h_after, lam_a_after = loaded.predict_lambda(df)
    np.testing.assert_allclose(lam_h_before, lam_h_after, rtol=1e-6)
    np.testing.assert_allclose(lam_a_before, lam_a_after, rtol=1e-6)


def test_symmetric_features_has_18_columns_including_tier2() -> None:
    """SYMMETRIC_FEATURES include 8 Tier 0+1 + 10 Tier 2."""
    assert len(SYMMETRIC_FEATURES) == 18
    expected_tier2 = [
        "team_form_5", "opponent_form_5",
        "team_gd_5", "opponent_gd_5",
        "team_goals_scored_5", "opponent_goals_scored_5",
        "team_goals_conceded_5", "opponent_goals_conceded_5",
        "team_avg_opp_elo_5", "opponent_avg_opp_elo_5",
    ]
    for col in expected_tier2:
        assert col in SYMMETRIC_FEATURES


def test_build_symmetric_rows_with_tier2_columns() -> None:
    """build_symmetric_rows popola le 10 Tier 2 con simmetria home/away."""
    df = pd.DataFrame({
        "home_team": ["A"], "away_team": ["B"],
        "date": pd.to_datetime(["2020-01-01"]),
        "home_score": [2], "away_score": [1],
        "neutral": [False],
        "home_elo_before": [1500.0], "away_elo_before": [1400.0],
        "competition_importance": [3],
        "days_rest_home": [10.0], "days_rest_away": [20.0],
        "home_form_5": [12.0], "away_form_5": [4.0],
        "home_gd_5": [5.0], "away_gd_5": [-3.0],
        "home_goals_scored_5": [2.5], "away_goals_scored_5": [0.8],
        "home_goals_conceded_5": [0.5], "away_goals_conceded_5": [1.6],
        "home_avg_opp_elo_5": [1450.0], "away_avg_opp_elo_5": [1480.0],
    })
    X, y = build_symmetric_rows(df)  # noqa: N806
    assert X.shape == (2, 18)
    tf_idx = SYMMETRIC_FEATURES.index("team_form_5")
    of_idx = SYMMETRIC_FEATURES.index("opponent_form_5")
    # Home perspective: team=A, opponent=B
    assert X[0, tf_idx] == 12.0
    assert X[0, of_idx] == 4.0
    # Away perspective: team=B, opponent=A
    assert X[1, tf_idx] == 4.0
    assert X[1, of_idx] == 12.0
