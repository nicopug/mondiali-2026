"""Tests for training.validate_markets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mondiali.training.validate_markets import (
    _binary_brier,
    _binary_log_loss,
    _market_baselines,
    _market_outcomes,
)


def test_market_outcomes_binary() -> None:
    df = pd.DataFrame({
        "home_score": [0, 1, 2, 3, 0],
        "away_score": [0, 1, 0, 2, 1],
    })
    out = _market_outcomes(df)
    assert list(out["over_under_1_5"]) == [0.0, 1.0, 1.0, 1.0, 0.0]
    assert list(out["over_under_2_5"]) == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert list(out["over_under_3_5"]) == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert list(out["btts"]) == [0.0, 1.0, 0.0, 1.0, 0.0]


def test_market_baselines_are_means() -> None:
    df = pd.DataFrame({
        "home_score": [0, 1, 2, 3, 0],
        "away_score": [0, 1, 0, 2, 1],
    })
    base = _market_baselines(df)
    assert base["over_under_2_5"] == 1.0 / 5.0  # only 3-2 is over
    assert base["btts"] == 2.0 / 5.0  # 1-1 and 3-2


def test_binary_log_loss_perfect() -> None:
    y = np.array([1.0, 0.0, 1.0])
    p = np.array([0.9999, 0.0001, 0.9999])
    assert _binary_log_loss(y, p) < 0.01


def test_binary_brier_zero_for_perfect() -> None:
    y = np.array([1.0, 0.0])
    assert _binary_brier(y, y) == 0.0
