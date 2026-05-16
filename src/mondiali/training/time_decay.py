"""Time-decay sample weighting for Poisson training.

Weight = exp(-(target_date - match_date).days / HALF_LIFE_DAYS * ln(2))
       = 0.5 ** ((target_date - match_date) / HALF_LIFE_DAYS_)

This is exponential decay with HALF_LIFE_DAYS as the time after which a match's
weight halves. 3 years ≈ 1095 days is a common default for football modeling
(team strength changes meaningfully over multi-year horizons).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_HALF_LIFE_DAYS = 1095.0  # ~3 years


def time_decay_weights(
    matches: pd.DataFrame,
    target_date: pd.Timestamp | str,
    *,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    symmetric_expansion: bool = True,
) -> np.ndarray:
    """Return per-row weights for ``matches``.

    If ``symmetric_expansion=True`` (default for use with build_symmetric_rows),
    each match's weight is duplicated (home-perspective and away-perspective).
    """
    target = pd.Timestamp(target_date)
    dates = pd.to_datetime(matches["date"])
    age_days = (target - dates).dt.days.to_numpy(dtype=float)
    age_days = np.maximum(age_days, 0.0)
    w = np.power(0.5, age_days / half_life_days)
    if symmetric_expansion:
        # Duplicate each weight to align with build_symmetric_rows output
        # (2 rows per match, indices 0::2 = home, 1::2 = away).
        out = np.empty(2 * len(w), dtype=float)
        out[0::2] = w
        out[1::2] = w
        return out
    return w
