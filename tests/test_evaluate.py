"""Test della funzione di evaluation (log-loss 1/X/2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mondiali.training.evaluate import compute_outcomes, log_loss_1x2


def test_compute_outcomes_encodes_1x2_correctly() -> None:
    """home_win=0, draw=1, away_win=2."""
    df = pd.DataFrame(
        {"home_score": [2, 1, 0, 3], "away_score": [0, 1, 2, 3]}
    )
    assert compute_outcomes(df).tolist() == [0, 1, 2, 1]


def test_log_loss_perfect_prediction_is_zero() -> None:
    """Predizione perfetta (probabilità 1.0 alla classe vera) → log-loss ~ 0."""
    df = pd.DataFrame({"home_score": [2, 1, 0], "away_score": [0, 1, 2]})
    probs = np.array(
        [
            [1 - 2e-15, 1e-15, 1e-15],
            [1e-15, 1 - 2e-15, 1e-15],
            [1e-15, 1e-15, 1 - 2e-15],
        ]
    )
    loss = log_loss_1x2(df, probs)
    assert loss < 1e-10


def test_log_loss_uniform_prediction_is_log3() -> None:
    """Predizione uniforme 1/3 per tutte le classi → log-loss = ln(3) ≈ 1.0986."""
    df = pd.DataFrame({"home_score": [2, 1, 0], "away_score": [0, 1, 2]})
    probs = np.full((3, 3), 1 / 3)
    loss = log_loss_1x2(df, probs)
    assert loss == pytest.approx(np.log(3), abs=0.001)


def test_log_loss_raises_on_shape_mismatch() -> None:
    """Probabilità con shape sbagliata → ValueError."""
    df = pd.DataFrame({"home_score": [1, 2], "away_score": [0, 0]})
    probs = np.array([[0.5, 0.3, 0.2]])
    with pytest.raises(ValueError, match="shape"):
        log_loss_1x2(df, probs)
