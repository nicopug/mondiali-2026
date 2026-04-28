"""Isotonic calibrator post-hoc per probabilita 1X2.

Architettura: 3 isotonic regressions indipendenti (P1, PX, P2) + rinormalizzazione
riga per riga.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression

log = structlog.get_logger(__name__)


def _serialize_iso(iso: IsotonicRegression) -> dict[str, Any]:
    return {
        "X_thresholds": iso.X_thresholds_.tolist(),
        "y_thresholds": iso.y_thresholds_.tolist(),
        "X_min": float(iso.X_min_),
        "X_max": float(iso.X_max_),
        "increasing": bool(iso.increasing_),
    }


def _deserialize_iso(data: dict[str, Any]) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.X_thresholds_ = np.asarray(data["X_thresholds"], dtype=float)
    iso.y_thresholds_ = np.asarray(data["y_thresholds"], dtype=float)
    iso.X_min_ = float(data["X_min"])
    iso.X_max_ = float(data["X_max"])
    iso.increasing_ = bool(data["increasing"])
    iso.f_ = interp1d(
        iso.X_thresholds_,
        iso.y_thresholds_,
        kind="linear",
        bounds_error=False,
        fill_value=(iso.y_thresholds_[0], iso.y_thresholds_[-1]),
    )
    return iso


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

    def save(self, path: Path) -> None:
        if self.iso_home_ is None or self.iso_draw_ is None or self.iso_away_ is None:
            raise RuntimeError("Calibrator must be fit() before save()")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "iso_home": _serialize_iso(self.iso_home_),
            "iso_draw": _serialize_iso(self.iso_draw_),
            "iso_away": _serialize_iso(self.iso_away_),
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> IsotonicCalibrator1X2:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text())
        cal = cls()
        cal.iso_home_ = _deserialize_iso(data["iso_home"])
        cal.iso_draw_ = _deserialize_iso(data["iso_draw"])
        cal.iso_away_ = _deserialize_iso(data["iso_away"])
        return cal
