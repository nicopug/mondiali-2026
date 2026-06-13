"""Talent differential primitive (Phase 1 feature + Phase 2 Elo anchor).

Derives per-match squad-talent differentials from the market-value columns
already present in matches.parquet (produced by add_tier3_features). NaN where
market value is missing; XGBoost handles NaN natively. No scraping, no merge —
inherits tier3's strict-pre-match anti-leakage guarantee.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TALENT_COLUMNS: list[str] = ["talent_gap_top11", "talent_log_ratio"]


def add_talent_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `matches` with talent differential columns added.

    - talent_gap_top11 = home_market_value_top11 - away_market_value_top11
    - talent_log_ratio = log1p(home_top11) - log1p(away_top11)

    NaN-preserving: if either side's top11 value is NaN, both outputs are NaN.
    """
    out = matches.copy()
    home = out["home_market_value_top11"].astype(float)
    away = out["away_market_value_top11"].astype(float)
    out["talent_gap_top11"] = home - away
    out["talent_log_ratio"] = np.log1p(home) - np.log1p(away)
    return out
