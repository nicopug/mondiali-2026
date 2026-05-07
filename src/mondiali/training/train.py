"""Training pipeline Tier 1 end-to-end.

Sequenza:
1. Carica matches.parquet
2. Split train/val per date
3. Addestra PoissonXGBModel (con early stopping su val)
4. Stima ρ Dixon-Coles sul training via MLE
5. Per ogni match di val: costruisci joint → DC correct → markets 1X2
6. Calcola log-loss 1/X/2 + metriche diagnostiche
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from mondiali.features.tier3 import TIER3_COLUMNS
from mondiali.model.calibration import IsotonicCalibrator1X2
from mondiali.model.dixon_coles import dixon_coles_correct, estimate_rho_mle, joint_matrix
from mondiali.model.markets import prob_1x2
from mondiali.model.poisson_xgb import PoissonXGBModel
from mondiali.training.evaluate import brier_score_1x2, compute_outcomes, log_loss_1x2

log = structlog.get_logger(__name__)


def _compute_1x2_probs(
    lam_h: np.ndarray, lam_a: np.ndarray, rho: float
) -> np.ndarray:
    """Per ogni match, costruisce joint → DC → 1X2. Ritorna shape (n, 3)."""
    n = len(lam_h)
    out = np.empty((n, 3), dtype=float)
    for i in range(n):
        m = joint_matrix(lam_h[i], lam_a[i])
        m = dixon_coles_correct(m, lam_h[i], lam_a[i], rho=rho)
        p1, px, p2 = prob_1x2(m)
        out[i] = (p1, px, p2)
    return out


def train_tier1_pipeline(
    parquet_path: Path,
    *,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pipeline completa. Ritorna un dizionario con metriche + modello.

    ⚠ Bias noto: la val è usata sia come `early_stopping_val` sia come set di
    metrica finale. L'iterazione ottimale di XGBoost è quindi selezionata
    *sul* val set, e `val_log_loss_1x2` è ottimisticamente biased come stima
    di generalizzazione. Per un'evaluation pulita servirebbe una terza fetta
    (es. carve-out dal train per ES). Per la gate decision di Task 12 questa
    cifra è comunque comparabile con `LOGLOSS_ELO` se anche il baseline non
    usa val per tuning.

    Returns:
        dict con chiavi:
        - model: PoissonXGBModel addestrato
        - rho: float (Dixon-Coles stimato)
        - val_log_loss_1x2: float (ottimisticamente biased, vedi sopra)
        - lambda_home_mean, lambda_away_mean: float
        - n_train, n_val: int
    """
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["days_rest_home", "days_rest_away"]).copy()  # escludi prima apparizione

    train = df[(df["date"] >= train_start) & (df["date"] <= train_end)].reset_index(drop=True)
    val = df[(df["date"] >= val_start) & (df["date"] <= val_end)].reset_index(drop=True)

    log.info("tier1 pipeline start", n_train=len(train), n_val=len(val))

    model = PoissonXGBModel(params=model_params)
    model.fit(train, early_stopping_val=val, early_stopping_rounds=50)

    # Stima ρ sul training (no leakage)
    lam_h_tr, lam_a_tr = model.predict_lambda(train)
    rho = estimate_rho_mle(
        lam_h_tr,
        lam_a_tr,
        train["home_score"].to_numpy(),
        train["away_score"].to_numpy(),
    )
    log.info("rho estimated", rho=rho)

    # Inference su validation
    lam_h_va, lam_a_va = model.predict_lambda(val)
    val_probs = _compute_1x2_probs(lam_h_va, lam_a_va, rho=rho)
    val_loss = log_loss_1x2(val, val_probs)

    log.info(
        "tier1 validation",
        log_loss_1x2=val_loss,
        lam_h_mean=float(lam_h_va.mean()),
        lam_a_mean=float(lam_a_va.mean()),
    )

    return {
        "model": model,
        "rho": rho,
        "val_log_loss_1x2": val_loss,
        "lambda_home_mean": float(lam_h_va.mean()),
        "lambda_away_mean": float(lam_a_va.mean()),
        "n_train": len(train),
        "n_val": len(val),
    }


def train_tier2_pipeline(
    parquet_path: Path,
    *,
    train_start: str = "2002-01-01",
    train_end: str = "2016-12-31",
    val_es_start: str = "2017-01-01",
    val_es_end: str = "2017-12-31",
    val_calib_start: str = "2018-01-01",
    val_calib_end: str = "2018-12-31",
    val_gate_start: str = "2019-01-01",
    val_gate_end: str = "2022-06-30",
    early_stopping_rounds: int = 50,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pipeline Tier 2 con 4-way split + isotonic calibration.

    Returns:
        dict con: model, rho, calibrator, val_log_loss_raw, val_log_loss_calib,
        brier_before, brier_after, n_train, n_val_es, n_val_calib, n_val_gate.
    """
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["days_rest_home", "days_rest_away"]).copy()

    train = df[(df["date"] >= train_start) & (df["date"] <= train_end)].reset_index(drop=True)
    val_es = df[(df["date"] >= val_es_start) & (df["date"] <= val_es_end)].reset_index(drop=True)
    val_calib = df[
        (df["date"] >= val_calib_start) & (df["date"] <= val_calib_end)
    ].reset_index(drop=True)
    val_gate = df[
        (df["date"] >= val_gate_start) & (df["date"] <= val_gate_end)
    ].reset_index(drop=True)

    log.info(
        "tier2 pipeline start",
        n_train=len(train),
        n_val_es=len(val_es),
        n_val_calib=len(val_calib),
        n_val_gate=len(val_gate),
    )

    model = PoissonXGBModel(params=model_params)
    model.fit(
        train,
        early_stopping_val=val_es,
        early_stopping_rounds=early_stopping_rounds,
    )

    lam_h_tr, lam_a_tr = model.predict_lambda(train)
    rho = estimate_rho_mle(
        lam_h_tr,
        lam_a_tr,
        train["home_score"].to_numpy(),
        train["away_score"].to_numpy(),
    )
    log.info("rho estimated", rho=rho)

    lam_h_cal, lam_a_cal = model.predict_lambda(val_calib)
    raw_probs_calib = _compute_1x2_probs(lam_h_cal, lam_a_cal, rho=rho)
    outcomes_calib = compute_outcomes(val_calib)
    calibrator = IsotonicCalibrator1X2().fit(raw_probs_calib, outcomes_calib)

    lam_h_ga, lam_a_ga = model.predict_lambda(val_gate)
    raw_probs_gate = _compute_1x2_probs(lam_h_ga, lam_a_ga, rho=rho)
    cal_probs_gate = calibrator.predict(raw_probs_gate)

    val_log_loss_raw = log_loss_1x2(val_gate, raw_probs_gate)
    val_log_loss_calib = log_loss_1x2(val_gate, cal_probs_gate)
    brier_before = brier_score_1x2(val_gate, raw_probs_gate)
    brier_after = brier_score_1x2(val_gate, cal_probs_gate)

    log.info(
        "tier2 validation",
        log_loss_raw=val_log_loss_raw,
        log_loss_calib=val_log_loss_calib,
        brier_before=brier_before,
        brier_after=brier_after,
    )

    return {
        "model": model,
        "rho": rho,
        "calibrator": calibrator,
        "val_log_loss_raw": val_log_loss_raw,
        "val_log_loss_calib": val_log_loss_calib,
        "brier_before": brier_before,
        "brier_after": brier_after,
        "n_train": len(train),
        "n_val_es": len(val_es),
        "n_val_calib": len(val_calib),
        "n_val_gate": len(val_gate),
    }


def _recompute_tier2_baseline_for_gate(
    parquet_path: Path,
    val_gate_start: str,
    val_gate_end: str,
    *,
    train_start: str = "2002-01-01",
    train_end: str = "2016-12-31",
    val_es_start: str = "2017-01-01",
    val_es_end: str = "2017-12-31",
) -> float:
    """Ricomputa Tier 2 raw log-loss su un val_gate arbitrario.

    Tier 3 columns sono forzate a NaN prima del training, in modo che il
    confronto Tier 2 vs Tier 3 sul medesimo val_gate (es. 2022) sia
    apples-to-apples. XGBoost ignora le feature 100% NaN.

    Returns:
        ``val_log_loss_raw`` di Tier 2 sul val_gate richiesto.
    """
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["days_rest_home", "days_rest_away"]).copy()

    train = df[(df["date"] >= train_start) & (df["date"] <= train_end)].reset_index(drop=True)
    val_es = df[(df["date"] >= val_es_start) & (df["date"] <= val_es_end)].reset_index(drop=True)
    val_gate = df[
        (df["date"] >= val_gate_start) & (df["date"] <= val_gate_end)
    ].reset_index(drop=True)

    for col in TIER3_COLUMNS:
        train[col] = np.nan
        val_es[col] = np.nan
        val_gate[col] = np.nan

    model = PoissonXGBModel()
    model.fit(train, early_stopping_val=val_es, early_stopping_rounds=50)

    lam_h_tr, lam_a_tr = model.predict_lambda(train)
    rho = estimate_rho_mle(
        lam_h_tr, lam_a_tr,
        train["home_score"].to_numpy(), train["away_score"].to_numpy(),
    )

    lam_h_va, lam_a_va = model.predict_lambda(val_gate)
    val_probs = _compute_1x2_probs(lam_h_va, lam_a_va, rho=rho)
    return log_loss_1x2(val_gate, val_probs)


def train_tier3_pipeline(
    parquet_path: Path,
    *,
    train_start: str = "2014-01-01",
    train_end: str = "2019-12-31",
    val_es_start: str = "2020-01-01",
    val_es_end: str = "2020-12-31",
    val_calib_start: str = "2021-01-01",
    val_calib_end: str = "2021-12-31",
    val_gate_start: str = "2022-01-01",
    val_gate_end: str = "2022-12-31",
    early_stopping_rounds: int = 50,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pipeline Tier 3: training su matches 2014+ con feature TM.

    Differenze chiave vs Tier 2:
    - Filtro 2014+ obbligatorio sul training set (TM è NaN prima).
    - Returns dict include ``n_train_pre2014_dropped`` + ``tm_coverage_*``.
    """
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["days_rest_home", "days_rest_away"]).copy()

    n_pre2014 = int((df["date"] < pd.Timestamp("2014-01-01")).sum())

    train = df[(df["date"] >= train_start) & (df["date"] <= train_end)].reset_index(drop=True)
    val_es = df[(df["date"] >= val_es_start) & (df["date"] <= val_es_end)].reset_index(drop=True)
    val_calib = df[
        (df["date"] >= val_calib_start) & (df["date"] <= val_calib_end)
    ].reset_index(drop=True)
    val_gate = df[
        (df["date"] >= val_gate_start) & (df["date"] <= val_gate_end)
    ].reset_index(drop=True)

    def _tm_coverage(d: pd.DataFrame) -> float:
        if len(d) == 0:
            return 0.0
        both_present = d["home_market_value_total"].notna() & d["away_market_value_total"].notna()
        return float(both_present.mean())

    log.info(
        "tier3_pipeline_start",
        n_train=len(train), n_val_es=len(val_es),
        n_val_calib=len(val_calib), n_val_gate=len(val_gate),
        n_train_pre2014_dropped=n_pre2014,
        tm_coverage_train=_tm_coverage(train),
        tm_coverage_gate=_tm_coverage(val_gate),
    )

    model = PoissonXGBModel(params=model_params)
    model.fit(train, early_stopping_val=val_es, early_stopping_rounds=early_stopping_rounds)

    lam_h_tr, lam_a_tr = model.predict_lambda(train)
    rho = estimate_rho_mle(
        lam_h_tr, lam_a_tr,
        train["home_score"].to_numpy(), train["away_score"].to_numpy(),
    )

    lam_h_cal, lam_a_cal = model.predict_lambda(val_calib)
    raw_probs_calib = _compute_1x2_probs(lam_h_cal, lam_a_cal, rho=rho)
    outcomes_calib = compute_outcomes(val_calib)
    calibrator = IsotonicCalibrator1X2().fit(raw_probs_calib, outcomes_calib)

    lam_h_ga, lam_a_ga = model.predict_lambda(val_gate)
    raw_probs_gate = _compute_1x2_probs(lam_h_ga, lam_a_ga, rho=rho)
    cal_probs_gate = calibrator.predict(raw_probs_gate)

    val_log_loss_raw = log_loss_1x2(val_gate, raw_probs_gate)
    val_log_loss_calib = log_loss_1x2(val_gate, cal_probs_gate)
    brier_before = brier_score_1x2(val_gate, raw_probs_gate)
    brier_after = brier_score_1x2(val_gate, cal_probs_gate)

    log.info(
        "tier3_validation",
        log_loss_raw=val_log_loss_raw, log_loss_calib=val_log_loss_calib,
        brier_before=brier_before, brier_after=brier_after,
    )

    return {
        "model": model,
        "rho": rho,
        "calibrator": calibrator,
        "val_log_loss_raw": val_log_loss_raw,
        "val_log_loss_calib": val_log_loss_calib,
        "brier_before": brier_before,
        "brier_after": brier_after,
        "n_train": len(train),
        "n_val_es": len(val_es),
        "n_val_calib": len(val_calib),
        "n_val_gate": len(val_gate),
        "n_train_pre2014_dropped": n_pre2014,
        "tm_coverage_train": _tm_coverage(train),
        "tm_coverage_gate": _tm_coverage(val_gate),
    }
