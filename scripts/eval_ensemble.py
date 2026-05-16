"""Evaluate ensemble of v1_final (XGB) + Tier 7 (DL) on val_gate 2024.

Strategy: average lambdas from both models, fit a fresh Dixon-Coles ρ on the
ensemble lambdas (training set), apply DC, compute 1X2 log-loss + Brier.

Reports raw v1_final, raw Tier 7, and ensemble for direct comparison.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mondiali.model.dixon_coles import estimate_rho_mle
from mondiali.model.dl_poisson import load_dl_model, predict_lambda
from mondiali.model.poisson_xgb import PoissonXGBModel
from mondiali.training.evaluate import brier_score_1x2, log_loss_1x2
from mondiali.training.train_tier7 import _compute_1x2_probs


def main() -> None:
    df = pd.read_parquet("data/processed/matches.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["days_rest_home", "days_rest_away"])
    train = df[(df["date"] >= "2002-01-01") & (df["date"] <= "2023-12-31")].reset_index(drop=True)
    val_gate = df[(df["date"] >= "2024-01-01") & (df["date"] <= "2024-12-31")].reset_index(drop=True)

    # v1_final XGB
    xgb = PoissonXGBModel().load(Path("models/v1_final/xgb_poisson.json"))
    rho_xgb = float(Path("models/v1_final/rho.txt").read_text().strip())
    lam_h_xgb_tr, lam_a_xgb_tr = xgb.predict_lambda(train)
    lam_h_xgb_g, lam_a_xgb_g = xgb.predict_lambda(val_gate)

    # Tier 7 DL
    dl, team_idx, stats, _ = load_dl_model(Path("models/tier7"))
    rho_dl = float(Path("models/tier7/rho.txt").read_text().strip())
    lam_h_dl_tr, lam_a_dl_tr = predict_lambda(dl, train, team_idx, stats)
    lam_h_dl_g, lam_a_dl_g = predict_lambda(dl, val_gate, team_idx, stats)

    print(f"XGB rho: {rho_xgb:.4f}  DL rho: {rho_dl:.4f}")
    print(f"XGB lambda means (gate): {lam_h_xgb_g.mean():.3f} / {lam_a_xgb_g.mean():.3f}")
    print(f"DL  lambda means (gate): {lam_h_dl_g.mean():.3f} / {lam_a_dl_g.mean():.3f}")

    results = {}
    for name, lam_h_tr, lam_a_tr, lam_h_g, lam_a_g, rho in [
        ("v1_final_XGB", lam_h_xgb_tr, lam_a_xgb_tr, lam_h_xgb_g, lam_a_xgb_g, rho_xgb),
        ("tier7_DL", lam_h_dl_tr, lam_a_dl_tr, lam_h_dl_g, lam_a_dl_g, rho_dl),
    ]:
        probs = _compute_1x2_probs(lam_h_g, lam_a_g, rho=rho)
        ll = log_loss_1x2(val_gate, probs)
        br = brier_score_1x2(val_gate, probs)
        results[name] = (ll, br)
        print(f"{name:24s}: log_loss={ll:.4f}  brier={br:.4f}")

    # Ensemble: average lambdas, fresh rho on ensemble train
    for weight_dl in [0.3, 0.4, 0.5, 0.6, 0.7]:
        w_xgb = 1.0 - weight_dl
        lam_h_tr = w_xgb * lam_h_xgb_tr + weight_dl * lam_h_dl_tr
        lam_a_tr = w_xgb * lam_a_xgb_tr + weight_dl * lam_a_dl_tr
        lam_h_g = w_xgb * lam_h_xgb_g + weight_dl * lam_h_dl_g
        lam_a_g = w_xgb * lam_a_xgb_g + weight_dl * lam_a_dl_g
        rho_ens = estimate_rho_mle(
            lam_h_tr, lam_a_tr,
            train["home_score"].to_numpy(), train["away_score"].to_numpy(),
        )
        probs = _compute_1x2_probs(lam_h_g, lam_a_g, rho=rho_ens)
        ll = log_loss_1x2(val_gate, probs)
        br = brier_score_1x2(val_gate, probs)
        results[f"ensemble_dl{weight_dl:.1f}"] = (ll, br, rho_ens)
        print(f"ensemble (dl={weight_dl:.1f}): log_loss={ll:.4f}  brier={br:.4f}  rho={rho_ens:.4f}")

    print("\n--- Summary ---")
    for k, v in results.items():
        print(f"{k:24s}: log_loss={v[0]:.4f}")


if __name__ == "__main__":
    main()
