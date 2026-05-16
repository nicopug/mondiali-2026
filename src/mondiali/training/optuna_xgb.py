"""Optuna TPE hparam search for the XGBoost Tier 2 model in v1_final.

Objective: val_calib 1X2 log-loss (calibration set, NOT val_gate — anti-leakage).
For each trial, fit XGB with early-stopping on val_es, estimate DC ρ on train,
compute 1X2 probs on val_calib.

The best params replace ``DEFAULT_PARAMS`` only if val_gate log-loss with tuned
params beats DEFAULT_PARAMS val_gate log-loss by > 0.001 (small margin since
hparam tuning gains are usually modest on this data size).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import structlog

from mondiali.config import RANDOM_STATE
from mondiali.model.dixon_coles import estimate_rho_mle, joint_matrix, dixon_coles_correct
from mondiali.model.markets import prob_1x2
from mondiali.model.poisson_xgb import DEFAULT_PARAMS, PoissonXGBModel
from mondiali.training.evaluate import log_loss_1x2

log = structlog.get_logger(__name__)


def _compute_1x2_probs(lam_h: np.ndarray, lam_a: np.ndarray, rho: float) -> np.ndarray:
    n = len(lam_h)
    out = np.empty((n, 3), dtype=float)
    for i in range(n):
        m = joint_matrix(float(lam_h[i]), float(lam_a[i]))
        m = dixon_coles_correct(m, float(lam_h[i]), float(lam_a[i]), rho=rho)
        p1, px, p2 = prob_1x2(m)
        out[i] = (p1, px, p2)
    out = np.clip(out, 0.0, 1.0)
    s = out.sum(axis=1, keepdims=True)
    return out / np.where(s > 0, s, 1.0)


@dataclass
class OptunaResult:
    best_params: dict[str, Any]
    best_val_calib_ll: float
    val_gate_ll_tuned: float
    val_gate_ll_default: float
    delta_vs_default: float
    n_trials: int
    promoted: bool  # tuned beats default by > 0.001


def _train_evaluate(
    train: pd.DataFrame, val_es: pd.DataFrame, val_calib: pd.DataFrame, val_gate: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[float, float, PoissonXGBModel, float]:
    """Return (val_calib_ll, val_gate_ll, model, rho)."""
    model = PoissonXGBModel(params=params).fit(
        train, early_stopping_val=val_es, early_stopping_rounds=50,
    )
    lam_h_tr, lam_a_tr = model.predict_lambda(train)
    rho = estimate_rho_mle(
        lam_h_tr, lam_a_tr,
        train["home_score"].to_numpy(), train["away_score"].to_numpy(),
    )
    lam_h_c, lam_a_c = model.predict_lambda(val_calib)
    probs_c = _compute_1x2_probs(lam_h_c, lam_a_c, rho=rho)
    val_c_ll = float(log_loss_1x2(val_calib, probs_c))
    lam_h_g, lam_a_g = model.predict_lambda(val_gate)
    probs_g = _compute_1x2_probs(lam_h_g, lam_a_g, rho=rho)
    val_g_ll = float(log_loss_1x2(val_gate, probs_g))
    return val_c_ll, val_g_ll, model, rho


def tune_xgb(
    matches_path: Path,
    *,
    n_trials: int = 100,
    seed: int = RANDOM_STATE,
    train_start: str = "2002-01-01",
    train_end: str = "2023-12-31",
    val_es_start: str = "2022-07-01",
    val_es_end: str = "2022-12-31",
    val_calib_start: str = "2023-01-01",
    val_calib_end: str = "2023-12-31",
    val_gate_start: str = "2024-01-01",
    val_gate_end: str = "2024-12-31",
    margin: float = 0.001,
    out_json: Path | None = None,
) -> OptunaResult:
    df = pd.read_parquet(matches_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["days_rest_home", "days_rest_away"])
    train = df[(df["date"] >= train_start) & (df["date"] <= train_end)].reset_index(drop=True)
    val_es = df[(df["date"] >= val_es_start) & (df["date"] <= val_es_end)].reset_index(drop=True)
    val_calib = df[
        (df["date"] >= val_calib_start) & (df["date"] <= val_calib_end)
    ].reset_index(drop=True)
    val_gate = df[
        (df["date"] >= val_gate_start) & (df["date"] <= val_gate_end)
    ].reset_index(drop=True)
    log.info("optuna_xgb data",
             n_train=len(train), n_val_es=len(val_es),
             n_val_calib=len(val_calib), n_val_gate=len(val_gate))

    default_c_ll, default_g_ll, _, _ = _train_evaluate(
        train, val_es, val_calib, val_gate, DEFAULT_PARAMS,
    )
    log.info("default_xgb", val_calib_ll=default_c_ll, val_gate_ll=default_g_ll)

    def objective(trial: optuna.Trial) -> float:
        params = {
            **DEFAULT_PARAMS,
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "random_state": seed,
        }
        try:
            val_c_ll, _, _, _ = _train_evaluate(train, val_es, val_calib, val_gate, params)
        except Exception as exc:
            log.warning("optuna trial failed", exc=str(exc))
            return float("inf")
        return val_c_ll

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = {**DEFAULT_PARAMS, **study.best_params, "random_state": seed}
    log.info("optuna_best", val_calib_ll=study.best_value, params=study.best_params)

    tuned_c_ll, tuned_g_ll, _, _ = _train_evaluate(
        train, val_es, val_calib, val_gate, best_params,
    )
    delta = tuned_g_ll - default_g_ll
    promoted = delta < -margin
    log.info("optuna_eval",
             tuned_val_gate_ll=tuned_g_ll, default_val_gate_ll=default_g_ll,
             delta=delta, promoted=promoted)

    result = OptunaResult(
        best_params=best_params,
        best_val_calib_ll=float(study.best_value),
        val_gate_ll_tuned=tuned_g_ll,
        val_gate_ll_default=default_g_ll,
        delta_vs_default=float(delta),
        n_trials=n_trials,
        promoted=promoted,
    )
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({
            "best_params": best_params,
            "best_val_calib_ll": float(study.best_value),
            "val_gate_ll_tuned": tuned_g_ll,
            "val_gate_ll_default": default_g_ll,
            "delta_vs_default": float(delta),
            "promoted": promoted,
            "n_trials": n_trials,
            "seed": seed,
        }, indent=2))
    return result
