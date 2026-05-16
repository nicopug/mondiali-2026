"""Tier 7 DL pipeline: team-embedding MLP Poisson.

Same splits as v1_final (Tier 2 frozen baseline):
    train      : 2002-01-01 → 2023-12-31  (drops missing days_rest rows)
    val_es     : 2022-07-01 → 2022-12-31  (early stopping)
    val_calib  : 2023-01-01 → 2023-12-31  (Dixon-Coles ρ + optional isotonic)
    val_gate   : 2024-01-01 → 2024-12-31  (final 1X2 log-loss gate)

Gate decision: DL log-loss on val_gate must beat v1_final raw log-loss
(currently 0.9044) to be promoted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from mondiali.model.dixon_coles import dixon_coles_correct, estimate_rho_mle, joint_matrix
from mondiali.model.dl_poisson import (
    DLConfig,
    build_team_index,
    predict_lambda,
    save_dl_model,
    train_dl_model,
)
from mondiali.model.markets import prob_1x2
from mondiali.training.evaluate import brier_score_1x2, log_loss_1x2

log = structlog.get_logger(__name__)

V1_FINAL_RAW_LOGLOSS = 0.9044  # benchmark from STEP 6 freeze


def _compute_1x2_probs(
    lam_h: np.ndarray, lam_a: np.ndarray, rho: float,
) -> np.ndarray:
    n = len(lam_h)
    out = np.empty((n, 3), dtype=float)
    for i in range(n):
        m = joint_matrix(float(lam_h[i]), float(lam_a[i]))
        m = dixon_coles_correct(m, float(lam_h[i]), float(lam_a[i]), rho=rho)
        p1, px, p2 = prob_1x2(m)
        out[i] = (p1, px, p2)
    # FP-noise can produce values like -2e-6; clip and renormalize.
    out = np.clip(out, 0.0, 1.0)
    s = out.sum(axis=1, keepdims=True)
    return out / np.where(s > 0, s, 1.0)


def train_tier7_pipeline(
    matches_path: Path,
    out_dir: Path,
    *,
    train_start: str = "2002-01-01",
    train_end: str = "2023-12-31",
    val_es_start: str = "2022-07-01",
    val_es_end: str = "2022-12-31",
    val_calib_start: str = "2023-01-01",
    val_calib_end: str = "2023-12-31",
    val_gate_start: str = "2024-01-01",
    val_gate_end: str = "2024-12-31",
    config: DLConfig | None = None,
) -> dict[str, Any]:
    """Train Tier 7 DL model + Dixon-Coles + return gate metrics."""
    cfg = config or DLConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(matches_path)
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
        "tier7 pipeline start",
        n_train=len(train), n_val_es=len(val_es),
        n_val_calib=len(val_calib), n_val_gate=len(val_gate),
    )

    team_idx = build_team_index(df)  # full universe (metadata, no leakage)
    log.info("team index built", n_teams=len(team_idx))

    model, stats, info = train_dl_model(train, val_es, team_idx, cfg)
    log.info("dl training done", best_val_es=info["best_val_es"],
             n_epochs=info["n_epochs_run"])

    lam_h_tr, lam_a_tr = predict_lambda(model, train, team_idx, stats)
    rho = estimate_rho_mle(
        lam_h_tr, lam_a_tr,
        train["home_score"].to_numpy(), train["away_score"].to_numpy(),
    )
    log.info("rho estimated on tier7 lambdas", rho=rho)

    lam_h_ga, lam_a_ga = predict_lambda(model, val_gate, team_idx, stats)
    raw_probs_gate = _compute_1x2_probs(lam_h_ga, lam_a_ga, rho=rho)
    gate_ll = float(log_loss_1x2(val_gate, raw_probs_gate))
    gate_brier = float(brier_score_1x2(val_gate, raw_probs_gate))

    lam_h_cal, lam_a_cal = predict_lambda(model, val_calib, team_idx, stats)
    calib_probs = _compute_1x2_probs(lam_h_cal, lam_a_cal, rho=rho)
    calib_ll = float(log_loss_1x2(val_calib, calib_probs))

    delta_vs_v1 = gate_ll - V1_FINAL_RAW_LOGLOSS

    save_dl_model(model, team_idx, stats, cfg, out_dir)
    (out_dir / "rho.txt").write_text(f"{rho:.6f}\n")

    result = {
        "model": model,
        "team_idx": team_idx,
        "stats": stats,
        "rho": rho,
        "gate_log_loss_raw": gate_ll,
        "gate_brier_raw": gate_brier,
        "val_calib_log_loss": calib_ll,
        "v1_final_raw_log_loss": V1_FINAL_RAW_LOGLOSS,
        "delta_vs_v1": delta_vs_v1,
        "promoted": gate_ll < V1_FINAL_RAW_LOGLOSS - 0.005,
        "n_train": len(train),
        "n_val_es": len(val_es),
        "n_val_calib": len(val_calib),
        "n_val_gate": len(val_gate),
        "n_teams": len(team_idx),
        "training_info": info,
        "config": cfg,
    }
    return result
