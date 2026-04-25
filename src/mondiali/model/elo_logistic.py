"""Baseline Elo-only logistic.

Features: [elo_diff, is_neutral_int].
Target: outcome 1/X/2 (0=home, 1=draw, 2=away).

Serve come comparatore obbligatorio per Tier 1: se XGBoost Poisson Tier 1 non
batte questo baseline in log-loss su validation di almeno 0.003, STOP — debug
features/leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mondiali.training.evaluate import compute_outcomes


class EloLogisticBaseline:
    """LogisticRegression multi-classe su [elo_diff, is_neutral_int]."""

    def __init__(self, *, C: float = 1.0, random_state: int = 42) -> None:  # noqa: N803
        self.C = C
        self.random_state = random_state
        self.model_: LogisticRegression | None = None

    def _design_matrix(self, matches: pd.DataFrame) -> np.ndarray:
        elo_diff = matches["home_elo_before"].to_numpy() - matches["away_elo_before"].to_numpy()
        is_neutral = matches["neutral"].astype(int).to_numpy()
        return np.column_stack([elo_diff, is_neutral])

    def fit(self, matches: pd.DataFrame) -> EloLogisticBaseline:
        """Fit su matches con home_elo_before, away_elo_before, neutral, home_score, away_score."""
        X = self._design_matrix(matches)  # noqa: N806
        y = compute_outcomes(matches)
        self.model_ = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state,
        ).fit(X, y)
        return self

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("EloLogisticBaseline must be fit() before predict_proba")
        X = self._design_matrix(matches)  # noqa: N806
        return np.asarray(self.model_.predict_proba(X))
