from __future__ import annotations

import numpy as np
import pandas as pd

from mondiali.features.talent import TALENT_COLUMNS, add_talent_features


def test_talent_gap_and_log_ratio_computed():
    df = pd.DataFrame({
        "home_market_value_top11": [100.0, 50.0],
        "away_market_value_top11": [40.0, 50.0],
    })
    out = add_talent_features(df)
    assert TALENT_COLUMNS == ["talent_gap_top11", "talent_log_ratio"]
    assert np.isclose(out["talent_gap_top11"].iloc[0], 60.0)
    assert np.isclose(out["talent_gap_top11"].iloc[1], 0.0)
    assert out["talent_log_ratio"].iloc[0] > 0
    assert np.isclose(out["talent_log_ratio"].iloc[1], 0.0)


def test_talent_features_preserve_nan_when_value_missing():
    df = pd.DataFrame({
        "home_market_value_top11": [np.nan, 50.0],
        "away_market_value_top11": [40.0, np.nan],
    })
    out = add_talent_features(df)
    assert out["talent_gap_top11"].isna().iloc[0]
    assert out["talent_gap_top11"].isna().iloc[1]
    assert out["talent_log_ratio"].isna().iloc[0]


def test_add_talent_features_does_not_mutate_input():
    df = pd.DataFrame({
        "home_market_value_top11": [100.0],
        "away_market_value_top11": [40.0],
    })
    _ = add_talent_features(df)
    assert "talent_gap_top11" not in df.columns
