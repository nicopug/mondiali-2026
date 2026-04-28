"""Isotonic calibrator post-hoc per probabilita 1X2.

Architettura: 3 isotonic regressions indipendenti (P1, PX, P2) + rinormalizzazione
riga per riga.
"""
from __future__ import annotations

import numpy as np
import structlog
from sklearn.isotonic import IsotonicRegression

log = structlog.get_logger(__name__)


class IsotonicCalibrator1X2:
    """Tre isotonic regressions indipendenti (1, X, 2) + rinormalizzazione."""

    def __init__(self) -> None:
        self.iso_home_: IsotonicRegression | None = None
        self.iso_draw_: IsotonicRegression | None = None
        self.iso_away_: IsotonicRegression | None = None

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> IsotonicCalibrator1X2:
        if probs.shape[1] != 3:
            raise ValueError(f"probs must have 3 columns, got {probs.shape}")
        if probs.shape[0] != outcomes.shape[0]:
            raise ValueError("probs and outcomes length mismatch")

        self.iso_home_ = IsotonicRegression(
            out_of_bounds="clip", y_min=0.0, y_max=1.0,
        ).fit(probs[:, 0], (outcomes == 0).astype(float))
        self.iso_draw_ = IsotonicRegression(
            out_of_bounds="clip", y_min=0.0, y_max=1.0,
        ).fit(probs[:, 1], (outcomes == 1).astype(float))
        self.iso_away_ = IsotonicRegression(
            out_of_bounds="clip", y_min=0.0, y_max=1.0,
        ).fit(probs[:, 2], (outcomes == 2).astype(float))
        log.info("isotonic calibrator fit", n=len(outcomes))
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        if self.iso_home_ is None or self.iso_draw_ is None or self.iso_away_ is None:
            raise RuntimeError("Calibrator must be fit() before predict()")
        if probs.shape[1] != 3:
            raise ValueError(f"probs must have 3 columns, got {probs.shape}")

        p_home = self.iso_home_.predict(probs[:, 0])
        p_draw = self.iso_draw_.predict(probs[:, 1])
        p_away = self.iso_away_.predict(probs[:, 2])
        out = np.column_stack([p_home, p_draw, p_away])

        s = out.sum(axis=1, keepdims=True)
        zero_mask = (s.flatten() == 0)
        out[zero_mask] = probs[zero_mask]
        s_safe = out.sum(axis=1, keepdims=True)
        out = out / s_safe
        return np.asarray(out)
