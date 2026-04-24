"""Test per baseline Tier 0 (prior costante 1/X/2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mondiali.training.baseline_prior import PriorBaseline


def _make_df(n_home_win: int, n_draw: int, n_away_win: int) -> pd.DataFrame:
    """Helper: df con esiti 1/X/2 in proporzioni richieste."""
    rows = (
        [{"home_score": 2, "away_score": 0}] * n_home_win
        + [{"home_score": 1, "away_score": 1}] * n_draw
        + [{"home_score": 0, "away_score": 2}] * n_away_win
    )
    return pd.DataFrame(rows)


def test_prior_fit_computes_frequencies() -> None:
    """fit() calcola le frequenze delle 3 classi dal training set."""
    df = _make_df(n_home_win=45, n_draw=25, n_away_win=30)
    model = PriorBaseline()
    model.fit(df)

    assert model.prior_ == pytest.approx([0.45, 0.25, 0.30], abs=0.001)


def test_prior_predict_proba_returns_constant_rows() -> None:
    """predict_proba restituisce lo stesso vettore di prior per ogni riga di input."""
    df_train = _make_df(50, 20, 30)
    model = PriorBaseline()
    model.fit(df_train)

    df_test = _make_df(1, 1, 1)
    probs = model.predict_proba(df_test)

    assert probs.shape == (3, 3)
    for row in probs:
        assert row == pytest.approx([0.50, 0.20, 0.30], abs=0.001)


def test_prior_proba_rows_sum_to_one() -> None:
    """Ogni riga di probabilità somma a 1."""
    df = _make_df(10, 20, 30)
    model = PriorBaseline()
    model.fit(df)
    probs = model.predict_proba(df.head(5))

    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)


def test_prior_raises_if_predict_before_fit() -> None:
    """predict_proba prima di fit() solleva."""
    model = PriorBaseline()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_proba(pd.DataFrame({"home_score": [1], "away_score": [0]}))
