"""Baseline Tier 0 — prior costante 1/X/2.

Predice sempre le frequenze storiche del training set. Serve come floor di
riferimento: qualsiasi modello successivo deve batterlo in log-loss o è
indistinguibile dal rumore.

Classi (ordine fisso): 0 = home win, 1 = draw, 2 = away win.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class PriorBaseline:
    """Predice sempre le frequenze storiche 1/X/2 del training set."""

    def __init__(self) -> None:
        self.prior_: list[float] | None = None

    def fit(self, matches: pd.DataFrame) -> PriorBaseline:
        """Calcola le frequenze 1/X/2 dai match di training.

        Args:
            matches: DataFrame con colonne `home_score`, `away_score`.

        Returns:
            self (per chaining).
        """
        outcomes = _compute_outcomes(matches)
        counts = np.bincount(outcomes, minlength=3).astype(float)
        self.prior_ = (counts / counts.sum()).tolist()
        return self

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        """Ritorna shape (n, 3) con riga costante = prior."""
        if self.prior_ is None:
            raise RuntimeError("PriorBaseline must be fit() before predict_proba")
        n = len(matches)
        return np.tile(np.array(self.prior_), (n, 1))


def _compute_outcomes(matches: pd.DataFrame) -> np.ndarray:
    """0 = home win, 1 = draw, 2 = away win."""
    home = matches["home_score"].to_numpy()
    away = matches["away_score"].to_numpy()
    out = np.where(home > away, 0, np.where(home == away, 1, 2))
    return out.astype(np.int64)
