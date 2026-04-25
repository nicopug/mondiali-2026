"""Test symmetric row builder + XGBoost Poisson training."""
from __future__ import annotations

import pandas as pd

from mondiali.model.poisson_xgb import SYMMETRIC_FEATURES, build_symmetric_rows


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
