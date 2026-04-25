"""Test del baseline Elo-only logistic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mondiali.model.elo_logistic import EloLogisticBaseline


def _make_df(n: int = 200) -> pd.DataFrame:
    """Genera df sintetico con elo_diff e outcome Bernoulli based on elo."""
    rng = np.random.default_rng(42)
    elo_diff = rng.normal(0, 150, n)
    # P(home_win) cresce con elo_diff
    logits = elo_diff / 200
    p_home = 1 / (1 + np.exp(-logits))
    r = rng.uniform(0, 1, n)
    home_score = np.where(r < p_home * 0.6, 2, np.where(r < p_home * 0.6 + 0.25, 1, 0))
    away_score = np.where(r < p_home * 0.6, 0, np.where(r < p_home * 0.6 + 0.25, 1, 2))
    return pd.DataFrame(
        {
            "home_elo_before": 1500 + elo_diff / 2,
            "away_elo_before": 1500 - elo_diff / 2,
            "neutral": [False] * n,
            "home_score": home_score,
            "away_score": away_score,
        }
    )


def test_fit_learns_positive_elo_diff_coefficient() -> None:
    """Coefficiente su elo_diff positivo → Elo alto vince di più."""
    df = _make_df(500)
    model = EloLogisticBaseline()
    model.fit(df)
    assert model.model_ is not None
    coef_elo_diff = model.model_.coef_[0, 0]  # classe 0 = home_win
    assert coef_elo_diff > 0


def test_predict_proba_shape_and_sum_to_one() -> None:
    """Shape (n, 3), ogni riga somma a 1."""
    df = _make_df(300)
    model = EloLogisticBaseline().fit(df)
    probs = model.predict_proba(df.head(50))
    assert probs.shape == (50, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)


def test_predict_proba_strong_home_favorite_gt_away() -> None:
    """Home con Elo molto più alto in casa → P(home_win) > P(away_win)."""
    df_train = _make_df(500)
    model = EloLogisticBaseline().fit(df_train)
    df_test = pd.DataFrame(
        {
            "home_elo_before": [2000.0],
            "away_elo_before": [1500.0],
            "neutral": [False],
            "home_score": [0],
            "away_score": [0],
        }
    )
    probs = model.predict_proba(df_test)
    assert probs[0, 0] > probs[0, 2]


def test_predict_before_fit_raises() -> None:
    model = EloLogisticBaseline()
    df = _make_df(5)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_proba(df)
