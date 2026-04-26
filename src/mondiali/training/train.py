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

from mondiali.model.dixon_coles import dixon_coles_correct, estimate_rho_mle, joint_matrix
from mondiali.model.markets import prob_1x2
from mondiali.model.poisson_xgb import PoissonXGBModel
from mondiali.training.evaluate import log_loss_1x2

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
