"""Quick eval: XGB with time-decay sample weights vs without, on val_gate 2024 and OOS 2025-2026."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mondiali.model.dixon_coles import estimate_rho_mle
from mondiali.model.poisson_xgb import DEFAULT_PARAMS, PoissonXGBModel
from mondiali.training.evaluate import brier_score_1x2, log_loss_1x2
from mondiali.training.optuna_xgb import _compute_1x2_probs
from mondiali.training.time_decay import time_decay_weights


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

    def fit_and_eval(weights):
        model = PoissonXGBModel(params=DEFAULT_PARAMS).fit(
            train, early_stopping_val=val_es, sample_weight=weights,
        )
        lam_h_tr, lam_a_tr = model.predict_lambda(train)
        rho = estimate_rho_mle(lam_h_tr, lam_a_tr, h_goals_tr, a_goals_tr)
        lam_h_g, lam_a_g = model.predict_lambda(val_gate)
        probs_g = _compute_1x2_probs(lam_h_g, lam_a_g, rho=rho)
        gate_ll = float(log_loss_1x2(val_gate, probs_g))
        gate_br = float(brier_score_1x2(val_gate, probs_g))
        lam_h_o, lam_a_o = model.predict_lambda(oos)
        probs_o = _compute_1x2_probs(lam_h_o, lam_a_o, rho=rho)
        oos_ll = float(log_loss_1x2(oos, probs_o))
        oos_br = float(brier_score_1x2(oos, probs_o))
        return gate_ll, gate_br, oos_ll, oos_br, rho

    print(f"{'config':25s}  {'gate_ll':>10s}  {'gate_br':>10s}  {'oos_ll':>10s}  {'oos_br':>10s}  {'rho':>8s}")

    # No decay (baseline)
    g_ll, g_br, o_ll, o_br, rho = fit_and_eval(None)
    print(f"{'no_decay (baseline)':25s}  {g_ll:>10.4f}  {g_br:>10.4f}  {o_ll:>10.4f}  {o_br:>10.4f}  {rho:>8.4f}")

    # Various half-lives
    target = pd.Timestamp("2024-06-30")  # mid val_gate
    for hl in [365 * 5, 365 * 3, 365 * 2, 365]:
        w = time_decay_weights(train, target, half_life_days=hl, symmetric_expansion=True)
        g_ll, g_br, o_ll, o_br, rho = fit_and_eval(w)
        print(f"{'hl=' + str(hl//365) + 'y':25s}  {g_ll:>10.4f}  {g_br:>10.4f}  {o_ll:>10.4f}  {o_br:>10.4f}  {rho:>8.4f}")


if __name__ == "__main__":
    main()
