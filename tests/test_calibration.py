"""Test IsotonicCalibrator1X2."""
from __future__ import annotations

import numpy as np
import pytest

from mondiali.model.calibration import IsotonicCalibrator1X2


def test_calibrator_fit_predict_shape_and_rows_sum_to_one() -> None:
    """predict ritorna (n,3) con righe normalizzate."""
    rng = np.random.default_rng(42)
    n = 200
    raw = rng.dirichlet([1, 1, 1], size=n)
    outcomes = rng.integers(0, 3, size=n)
    cal = IsotonicCalibrator1X2().fit(raw, outcomes)
    out = cal.predict(raw)
    assert out.shape == (n, 3)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-10)
    assert (out >= 0).all()


def test_calibrator_brier_does_not_increase_on_fit_set() -> None:
    """Brier dopo calibration <= Brier prima sulla stessa split (no overfit oversimple)."""
    rng = np.random.default_rng(0)
    n = 1000
    outcomes = rng.choice([0, 1, 2], size=n, p=[0.5, 0.25, 0.25])
    probs = np.full((n, 3), 0.05)
    pred = rng.integers(0, 3, size=n)
    probs[np.arange(n), pred] = 0.9

    def brier(p: np.ndarray, y: np.ndarray) -> float:
        oh = np.zeros((len(y), 3))
        oh[np.arange(len(y)), y] = 1.0
        return float(((p - oh) ** 2).sum(axis=1).mean())

    before = brier(probs, outcomes)
    cal = IsotonicCalibrator1X2().fit(probs, outcomes)
    calibrated = cal.predict(probs)
    after = brier(calibrated, outcomes)
    assert after <= before


def test_calibrator_handles_zero_sum_row_with_fallback() -> None:
    """Se tutti e 3 gli isotonic mappano a 0, il floor previene la divisione per zero."""
    rng = np.random.default_rng(1)
    n = 100
    outcomes = np.zeros(n, dtype=int)
    raw = rng.dirichlet([1, 1, 1], size=n)
    cal = IsotonicCalibrator1X2().fit(raw, outcomes)
    edge = np.array([[0.0, 0.5, 0.5]])
    out = cal.predict(edge)
    assert out.shape == (1, 3)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-10)


def test_calibrator_never_emits_zero_probability() -> None:
    """Floor invariant: nessuna probability calibrata può essere esattamente 0.

    Senza il floor, isotonic predice 0 per inputs estremi → log-loss esplode
    quando l'outcome reale cade su quella classe.
    """
    from mondiali.model.calibration import MIN_PROB_FLOOR
    rng = np.random.default_rng(42)
    n = 1000
    raw = rng.dirichlet([1, 1, 1], size=n)
    outcomes = rng.choice([0, 1, 2], size=n, p=[0.5, 0.25, 0.25])
    cal = IsotonicCalibrator1X2().fit(raw, outcomes)
    extreme = np.array([
        [0.99, 0.005, 0.005],
        [0.005, 0.99, 0.005],
        [0.005, 0.005, 0.99],
        [0.0001, 0.0001, 0.9998],
    ])
    out = cal.predict(extreme)
    assert (out > 0.0).all(), f"Found zero prob after floor: {out}"
    # Each prob >= floor / (1 + floor*3) after renormalization
    min_expected = MIN_PROB_FLOOR / (1.0 + 3.0 * MIN_PROB_FLOOR) * 0.99
    assert out.min() >= min_expected, f"Min prob {out.min()} below floor"


def test_calibrator_idempotent_on_perfectly_calibrated() -> None:
    """Probs already perfectly calibrated -> predict ~ identity (within noise)."""
    rng = np.random.default_rng(2)
    n = 5000
    outcomes = rng.choice([0, 1, 2], size=n, p=[0.5, 0.25, 0.25])
    probs = np.tile([0.5, 0.25, 0.25], (n, 1))
    cal = IsotonicCalibrator1X2().fit(probs, outcomes)
    out = cal.predict(probs)
    np.testing.assert_allclose(out.mean(axis=0), [0.5, 0.25, 0.25], atol=0.05)


def test_calibrator_json_roundtrip(tmp_path) -> None:
    """save -> load -> predict identico al bit."""
    rng = np.random.default_rng(42)
    n = 500
    raw = rng.dirichlet([1, 1, 1], size=n)
    outcomes = rng.choice([0, 1, 2], size=n, p=[0.5, 0.25, 0.25])
    cal = IsotonicCalibrator1X2().fit(raw, outcomes)

    path = tmp_path / "calibrator.json"
    cal.save(path)

    loaded = IsotonicCalibrator1X2.load(path)
    assert loaded.predict(raw) == pytest.approx(cal.predict(raw), abs=0.0)


def test_calibrator_load_missing_file_raises(tmp_path) -> None:
    """load di file inesistente solleva FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        IsotonicCalibrator1X2.load(tmp_path / "nonexistent.json")
