"""Multi-seed L3 averaging eval.

Trains L3 bivariate with 3 seeds (42, 1, 2), averages lambdas, evaluates the
new ensemble (XGB + averaged-L3) on val_gate 2024 and OOS 2025-2026.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mondiali.model.dl_bivariate import (
    BivariateConfig, predict_lambda_rho, train_bivariate_model,
    build_team_index as build_idx_biv,
)
from mondiali.model.dl_poisson import (
    DLConfig, predict_lambda as l1_predict, train_dl_model,
)
from mondiali.model.poisson_xgb import PoissonXGBModel
from mondiali.model.dixon_coles import estimate_rho_mle
from mondiali.training.evaluate import brier_score_1x2, log_loss_1x2
from mondiali.training.optuna_xgb import _compute_1x2_probs


def main():
    df = pd.read_parquet("data/processed/matches.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["days_rest_home", "days_rest_away"])
    train = df[(df["date"] >= "2002-01-01") & (df["date"] <= "2023-12-31")].reset_index(drop=True)
    val_es = df[(df["date"] >= "2022-07-01") & (df["date"] <= "2022-12-31")].reset_index(drop=True)
    val_gate = df[(df["date"] >= "2024-01-01") & (df["date"] <= "2024-12-31")].reset_index(drop=True)
    oos = df[(df["date"] >= "2025-01-01") & (df["date"] <= "2026-03-31")].reset_index(drop=True)
    h_goals_tr = train["home_score"].to_numpy()
    a_goals_tr = train["away_score"].to_numpy()

    team_idx = build_idx_biv(df)
    xgb = PoissonXGBModel().load(Path("models/v1_final/xgb_poisson.json"))
    lam_h_xgb_tr, lam_a_xgb_tr = xgb.predict_lambda(train)
    lam_h_xgb_g, lam_a_xgb_g = xgb.predict_lambda(val_gate)
    lam_h_xgb_o, lam_a_xgb_o = xgb.predict_lambda(oos)

    print("Training L3 with 3 seeds...")
    seeds = [42, 1, 2]
    l3_lambdas_g = []  # list of (lh, la) on val_gate
    l3_lambdas_o = []
    l3_lambdas_tr = []
    for s in seeds:
        cfg = BivariateConfig(seed=s)
        model, stats, info = train_bivariate_model(train, val_es, team_idx, cfg)
        print(f"  seed={s}: best_val_es NLL={info['best_val_es']:.4f}")
        lh_g, la_g, _ = predict_lambda_rho(model, val_gate, team_idx, stats)
        lh_o, la_o, _ = predict_lambda_rho(model, oos, team_idx, stats)
        lh_tr, la_tr, _ = predict_lambda_rho(model, train, team_idx, stats)
        l3_lambdas_g.append((lh_g, la_g))
        l3_lambdas_o.append((lh_o, la_o))
        l3_lambdas_tr.append((lh_tr, la_tr))

    # Single seed (42)
    s0_lh_g, s0_la_g = l3_lambdas_g[0]
    s0_lh_o, s0_la_o = l3_lambdas_o[0]
    s0_lh_tr, s0_la_tr = l3_lambdas_tr[0]

    # Multi-seed average
    avg_lh_g = np.mean([x[0] for x in l3_lambdas_g], axis=0)
    avg_la_g = np.mean([x[1] for x in l3_lambdas_g], axis=0)
    avg_lh_o = np.mean([x[0] for x in l3_lambdas_o], axis=0)
    avg_la_o = np.mean([x[1] for x in l3_lambdas_o], axis=0)
    avg_lh_tr = np.mean([x[0] for x in l3_lambdas_tr], axis=0)
    avg_la_tr = np.mean([x[1] for x in l3_lambdas_tr], axis=0)

    def eval_ens(lh_x_tr, la_x_tr, lh_x_g, la_x_g, lh_x_o, la_x_o, w_l3, name):
        w_xgb = 1.0 - w_l3
        lh_tr = w_xgb * lam_h_xgb_tr + w_l3 * lh_x_tr
        la_tr = w_xgb * lam_a_xgb_tr + w_l3 * la_x_tr
        rho = estimate_rho_mle(lh_tr, la_tr, h_goals_tr, a_goals_tr)
        lh_g = w_xgb * lam_h_xgb_g + w_l3 * lh_x_g
        la_g = w_xgb * lam_a_xgb_g + w_l3 * la_x_g
        probs_g = _compute_1x2_probs(lh_g, la_g, rho=rho)
        g_ll = float(log_loss_1x2(val_gate, probs_g))
        g_br = float(brier_score_1x2(val_gate, probs_g))
        lh_o = w_xgb * lam_h_xgb_o + w_l3 * lh_x_o
        la_o = w_xgb * lam_a_xgb_o + w_l3 * la_x_o
        probs_o = _compute_1x2_probs(lh_o, la_o, rho=rho)
        o_ll = float(log_loss_1x2(oos, probs_o))
        o_br = float(brier_score_1x2(oos, probs_o))
        print(f"  {name:30s}  gate ll={g_ll:.4f} br={g_br:.4f}  oos ll={o_ll:.4f} br={o_br:.4f}  rho={rho:+.4f}")
        return g_ll, g_br, o_ll, o_br

    print(f"\nXGB alone: gate ll=0.9044 (baseline)\n")
    for w_l3 in [0.15, 0.20, 0.25, 0.30]:
        eval_ens(s0_lh_tr, s0_la_tr, s0_lh_g, s0_la_g, s0_lh_o, s0_la_o,
                 w_l3, f"single-seed L3 w={w_l3:.2f}")
        eval_ens(avg_lh_tr, avg_la_tr, avg_lh_g, avg_la_g, avg_lh_o, avg_la_o,
                 w_l3, f"3-seed-avg L3 w={w_l3:.2f}")
        print()


if __name__ == "__main__":
    main()
