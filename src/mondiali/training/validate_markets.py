"""Per-market validation gate (Brier + log-loss vs baseline naive).

For each secondary market (U/O 1.5/2.5/3.5, BTTS), compares the model's
predictions against a constant ``baseline = market frequency on training set``.

``validated`` flag = ``model_brier < baseline_brier - BRIER_MARGIN``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mondiali.model.dixon_coles import dixon_coles_correct, joint_matrix
from mondiali.model.markets import prob_btts, prob_over_under
from mondiali.model.poisson_xgb import PoissonXGBModel

BRIER_MARGIN = 0.005
LOG_LOSS_EPS = 1e-9

MARKETS_UO_LINES: tuple[float, ...] = (1.5, 2.5, 3.5)


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
    p = {
        "over_under_1_5": np.zeros(n),
        "over_under_2_5": np.zeros(n),
        "over_under_3_5": np.zeros(n),
        "btts": np.zeros(n),
    }
    for i in range(n):
        joint = joint_matrix(float(lam_h[i]), float(lam_a[i]))
        joint = dixon_coles_correct(joint, float(lam_h[i]), float(lam_a[i]), rho)
        for line in MARKETS_UO_LINES:
            over, _ = prob_over_under(joint, threshold=line)
            p[f"over_under_{int(line)}_{int(round((line - int(line)) * 10))}"][i] = over
        btts_yes, _ = prob_btts(joint)
        p["btts"][i] = btts_yes
    return p


def _market_baselines(train: pd.DataFrame) -> dict[str, float]:
    outcomes = _market_outcomes(train)
    return {k: float(v.mean()) for k, v in outcomes.items()}


def validate_all_markets(
    *,
    model: PoissonXGBModel,
    train: pd.DataFrame,
    val_gate: pd.DataFrame,
    rho: float,
    margin: float = BRIER_MARGIN,
) -> dict[str, dict]:
    """Compute Brier + log-loss per market on ``val_gate``.

    ``train`` is used only to compute baseline naive frequencies.

    Returns
    -------
    dict keyed by market name, each value containing:
        - ``log_loss``, ``brier`` (model)
        - ``baseline_log_loss``, ``baseline_brier`` (constant baseline)
        - ``baseline_freq`` (training market frequency)
        - ``validated`` (model_brier < baseline_brier - margin)
    """
    baselines = _market_baselines(train)
    outcomes = _market_outcomes(val_gate)
    model_probs = _model_market_probs(model, val_gate, rho)

    result: dict[str, dict] = {}
    for market in ("over_under_1_5", "over_under_2_5", "over_under_3_5", "btts"):
        y = outcomes[market]
        p = model_probs[market]
        p_base = np.full_like(y, baselines[market])
        ll = _binary_log_loss(y, p)
        br = _binary_brier(y, p)
        ll_b = _binary_log_loss(y, p_base)
        br_b = _binary_brier(y, p_base)
        result[market] = {
            "log_loss": ll,
            "brier": br,
            "baseline_log_loss": ll_b,
            "baseline_brier": br_b,
            "baseline_freq": float(baselines[market]),
            "validated": bool(br < br_b - margin),
        }
    return result
