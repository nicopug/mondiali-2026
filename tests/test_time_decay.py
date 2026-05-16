"""Tests for time_decay.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mondiali.training.time_decay import DEFAULT_HALF_LIFE_DAYS, time_decay_weights


def _matches(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(dates)})


def test_weight_at_target_date_is_one() -> None:
    m = _matches(["2024-01-01"])
    w = time_decay_weights(m, "2024-01-01", symmetric_expansion=False)
    assert w[0] == 1.0


def test_weight_at_one_halflife_is_half() -> None:
    m = _matches(["2021-01-01"])
    w = time_decay_weights(
        m, "2024-01-01", half_life_days=1095.0, symmetric_expansion=False,
    )
    np.testing.assert_allclose(w[0], 0.5, atol=1e-3)


def test_weight_decreases_monotonically() -> None:
    m = _matches(["2024-01-01", "2020-01-01", "2010-01-01", "2002-01-01"])
    w = time_decay_weights(m, "2024-06-01", symmetric_expansion=False)
    for i in range(len(w) - 1):
        assert w[i] > w[i + 1]


def test_symmetric_expansion_doubles_length() -> None:
    m = _matches(["2024-01-01", "2023-01-01"])
    w = time_decay_weights(m, "2024-06-01", symmetric_expansion=True)
    assert len(w) == 4
    assert w[0] == w[1]
    assert w[2] == w[3]


def test_future_matches_clipped_to_one() -> None:
    m = _matches(["2025-01-01"])
    w = time_decay_weights(m, "2024-01-01", symmetric_expansion=False)
    assert w[0] == 1.0


def test_default_half_life_is_three_years() -> None:
    assert abs(DEFAULT_HALF_LIFE_DAYS - 1095.0) < 1.0
