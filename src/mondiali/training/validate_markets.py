"""Per-market validation gate (Brier + log-loss vs baseline naive).

Two-stage pipeline:
1. ``fit_market_calibrators(model, val_calib, rho)`` — fits a binary isotonic
   per secondary market (U/O 1.5/2.5/3.5, BTTS) on the calibration slice.
2. ``validate_all_markets(model, train, val_gate, rho, calibrators)`` — computes
   Brier + log-loss per market on the gate slice, using the calibrated probs
   when calibration improves Brier, otherwise raw.

``validated`` flag = ``model_brier < baseline_brier - BRIER_MARGIN``.
``calibrated`` flag (per market) = whether the binary calibrator was kept.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mondiali.model.calibration import BinaryMarketCalibrator
from mondiali.model.dixon_coles import dixon_coles_correct, joint_matrix
from mondiali.model.markets import prob_btts, prob_over_under
from mondiali.model.poisson_xgb import PoissonXGBModel

BRIER_MARGIN = 0.005
LOG_LOSS_EPS = 1e-9

MARKETS_UO_LINES: tuple[float, ...] = (1.5, 2.5, 3.5)
SECONDARY_MARKETS: tuple[str, ...] = (
    "over_under_1_5", "over_under_2_5", "over_under_3_5", "btts",
)


def _binary_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p_clipped = np.clip(p, LOG_LOSS_EPS, 1.0 - LOG_LOSS_EPS)
    return float(-np.mean(y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped)))


def _binary_brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _market_outcomes(matches: pd.DataFrame) -> dict[str, np.ndarray]:
    total = matches["home_score"].to_numpy() + matches["away_score"].to_numpy()
    btts = (
        (matches["home_score"].to_numpy() > 0)
        & (matches["away_score"].to_numpy() > 0)
    ).astype(float)
    return {
        "over_under_1_5": (total > 1.5).astype(float),
        "over_under_2_5": (total > 2.5).astype(float),
        "over_under_3_5": (total > 3.5).astype(float),
        "btts": btts,
    }


def _model_market_probs(
    model: PoissonXGBModel, matches: pd.DataFrame, rho: float,
) -> dict[str, np.ndarray]:
    lam_h, lam_a = model.predict_lambda(matches)
    n = len(matches)
    p = {market: np.zeros(n) for market in SECONDARY_MARKETS}
    for i in range(n):
        joint = joint_matrix(float(lam_h[i]), float(lam_a[i]))
        joint = dixon_coles_correct(joint, float(lam_h[i]), float(lam_a[i]), rho)
        over15, _ = prob_over_under(joint, threshold=1.5)
        over25, _ = prob_over_under(joint, threshold=2.5)
        over35, _ = prob_over_under(joint, threshold=3.5)
        btts_yes, _ = prob_btts(joint)
        p["over_under_1_5"][i] = over15
        p["over_under_2_5"][i] = over25
        p["over_under_3_5"][i] = over35
        p["btts"][i] = btts_yes
    return p


def _market_baselines(train: pd.DataFrame) -> dict[str, float]:
    outcomes = _market_outcomes(train)
    return {k: float(v.mean()) for k, v in outcomes.items()}


def fit_market_calibrators(
    *, model: PoissonXGBModel, val_calib: pd.DataFrame, rho: float,
) -> dict[str, BinaryMarketCalibrator]:
    """Fit one binary isotonic per secondary market on ``val_calib``."""
    raw_probs = _model_market_probs(model, val_calib, rho)
    outcomes = _market_outcomes(val_calib)
    return {
        market: BinaryMarketCalibrator().fit(raw_probs[market], outcomes[market])
        for market in SECONDARY_MARKETS
    }


def validate_all_markets(
    *,
    model: PoissonXGBModel,
    train: pd.DataFrame,
    val_gate: pd.DataFrame,
    rho: float,
    calibrators: dict[str, BinaryMarketCalibrator] | None = None,
    margin: float = BRIER_MARGIN,
) -> dict[str, dict]:
    """Compute Brier + log-loss per market on ``val_gate``.

    For each market: keep the calibrated probs only if their Brier improves
    over the raw probs. Returns per-market dict including ``raw_brier``,
    ``calib_brier``, ``brier`` (the kept one), ``calibrator_kept``, ``validated``.
    """
    baselines = _market_baselines(train)
    outcomes = _market_outcomes(val_gate)
    raw_probs = _model_market_probs(model, val_gate, rho)

    result: dict[str, dict] = {}
    for market in SECONDARY_MARKETS:
        y = outcomes[market]
        p_raw = raw_probs[market]
        p_base = np.full_like(y, baselines[market])

        raw_ll = _binary_log_loss(y, p_raw)
        raw_br = _binary_brier(y, p_raw)
        ll_b = _binary_log_loss(y, p_base)
        br_b = _binary_brier(y, p_base)

        calib_kept = False
        calib_ll: float | None = None
        calib_br: float | None = None
        if calibrators is not None and market in calibrators:
            p_calib = calibrators[market].predict(p_raw)
            calib_ll = _binary_log_loss(y, p_calib)
            calib_br = _binary_brier(y, p_calib)
            calib_kept = bool(calib_br < raw_br)

        chosen_ll = calib_ll if calib_kept and calib_ll is not None else raw_ll
        chosen_br = calib_br if calib_kept and calib_br is not None else raw_br

        result[market] = {
            "log_loss": float(chosen_ll),
            "brier": float(chosen_br),
            "raw_log_loss": float(raw_ll),
            "raw_brier": float(raw_br),
            "calib_log_loss": float(calib_ll) if calib_ll is not None else None,
            "calib_brier": float(calib_br) if calib_br is not None else None,
            "calibrator_kept": calib_kept,
            "baseline_log_loss": float(ll_b),
            "baseline_brier": float(br_b),
            "baseline_freq": float(baselines[market]),
            "validated": bool(chosen_br < br_b - margin),
        }
    return result
