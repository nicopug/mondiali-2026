"""Quick 10-min diagnostic: perché Tier 1 non batte Elo logistic?

Estrae:
- best_iteration di XGBoost (early stopping su val)
- gain importance per feature
- SHAP mean(|contribution|) per feature
- Prova retrain con early_stopping_rounds=20 (più aggressivo)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mondiali.config import CONFIG
from mondiali.model.dixon_coles import dixon_coles_correct, estimate_rho_mle, joint_matrix
from mondiali.model.markets import prob_1x2
from mondiali.model.poisson_xgb import SYMMETRIC_FEATURES, PoissonXGBModel, build_symmetric_rows
from mondiali.training.evaluate import log_loss_1x2


def compute_1x2(lam_h, lam_a, rho):
    n = len(lam_h)
    out = np.empty((n, 3), dtype=float)
    for i in range(n):
        m = joint_matrix(lam_h[i], lam_a[i])
        m = dixon_coles_correct(m, lam_h[i], lam_a[i], rho=rho)
        out[i] = prob_1x2(m)
    return out


def evaluate(model, train, val):
    lh_tr, la_tr = model.predict_lambda(train)
    rho = estimate_rho_mle(
        lh_tr, la_tr, train["home_score"].to_numpy(), train["away_score"].to_numpy()
    )
    lh_va, la_va = model.predict_lambda(val)
    probs = compute_1x2(lh_va, la_va, rho)
    return log_loss_1x2(val, probs), rho


def main():
    df = pd.read_parquet(CONFIG.data_processed / "matches.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["days_rest_home", "days_rest_away"]).copy()

    train = df[(df["date"] >= "2002-01-01") & (df["date"] <= "2018-12-31")].reset_index(drop=True)
    val = df[(df["date"] >= "2019-01-01") & (df["date"] <= "2022-06-30")].reset_index(drop=True)

    print(f"n_train={len(train)}  n_val={len(val)}")
    print()

    # === Default (early_stopping_rounds=50) ===
    print("=" * 70)
    print("Run 1: default (early_stopping_rounds=50)")
    print("=" * 70)
    model = PoissonXGBModel()
    model.fit(train, early_stopping_val=val, early_stopping_rounds=50)
    booster = model.booster_

    print(f"best_iteration         : {booster.best_iteration}")
    print(f"n_estimators (max)     : {model.params['n_estimators']}")
    loss, rho = evaluate(model, train, val)
    print(f"val_log_loss_1x2       : {loss:.10f}")
    print(f"rho                    : {rho:.6f}")
    print()

    # Feature importance (gain)
    print("Feature importance (gain) — sum di gain per ogni split:")
    gain = booster.get_booster().get_score(importance_type="gain")
    # Map f0,f1,... -> SYMMETRIC_FEATURES
    gain_named = {SYMMETRIC_FEATURES[int(k[1:])]: v for k, v in gain.items()}
    total = sum(gain_named.values())
    for name in SYMMETRIC_FEATURES:
        v = gain_named.get(name, 0.0)
        pct = 100 * v / total if total else 0
        print(f"  {name:25s} {v:>10.2f}  ({pct:>5.2f}%)")
    print()

    # SHAP via pred_contribs (XGBoost native)
    print("SHAP mean(|contribution|) su val (XGBoost native pred_contribs):")
    X_val, _ = build_symmetric_rows(val)
    import xgboost as xgb_mod
    dval = xgb_mod.DMatrix(X_val)
    shap_vals = booster.get_booster().predict(dval, pred_contribs=True)
    # shape (n, n_features+1) — last col is bias
    mean_abs = np.abs(shap_vals[:, :-1]).mean(axis=0)
    bias = shap_vals[:, -1].mean()
    print(f"  bias (avg)             : {bias:>10.4f}")
    for i, name in enumerate(SYMMETRIC_FEATURES):
        print(f"  {name:25s} {mean_abs[i]:>10.4f}")
    print()

    # === ES=20 ===
    print("=" * 70)
    print("Run 2: aggressive early stopping (early_stopping_rounds=20)")
    print("=" * 70)
    model2 = PoissonXGBModel()
    model2.fit(train, early_stopping_val=val, early_stopping_rounds=20)
    print(f"best_iteration         : {model2.booster_.best_iteration}")
    loss2, rho2 = evaluate(model2, train, val)
    print(f"val_log_loss_1x2       : {loss2:.10f}")
    print(f"rho                    : {rho2:.6f}")
    print()

    # === Solo Elo features (drop competition_importance + days_rest) ===
    print("=" * 70)
    print("Run 3: only Elo features (drop comp_importance + days_rest)")
    print("=" * 70)
    # Hack: zero out colonne 5,6,7 in entrambi train/val per imitare drop
    from mondiali.model.poisson_xgb import DEFAULT_PARAMS
    import xgboost as xgb_mod
    X_tr, y_tr = build_symmetric_rows(train)
    X_vl, y_vl = build_symmetric_rows(val)
    X_tr_min = X_tr[:, :5].copy()  # team_elo, opp_elo, elo_diff, is_home, is_neutral
    X_vl_min = X_vl[:, :5].copy()
    params = {**DEFAULT_PARAMS, "early_stopping_rounds": 50}
    booster3 = xgb_mod.XGBRegressor(**params)
    booster3.fit(X_tr_min, y_tr, eval_set=[(X_vl_min, y_vl)], verbose=False)
    print(f"best_iteration         : {booster3.best_iteration}")
    preds_tr = booster3.predict(X_tr_min)
    preds_vl = booster3.predict(X_vl_min)
    lh_tr3, la_tr3 = preds_tr[0::2], preds_tr[1::2]
    lh_vl3, la_vl3 = preds_vl[0::2], preds_vl[1::2]
    rho3 = estimate_rho_mle(
        lh_tr3, la_tr3, train["home_score"].to_numpy(), train["away_score"].to_numpy()
    )
    probs3 = compute_1x2(lh_vl3, la_vl3, rho3)
    loss3 = log_loss_1x2(val, probs3)
    print(f"val_log_loss_1x2       : {loss3:.10f}")
    print(f"rho                    : {rho3:.6f}")
    print()

    print("=" * 70)
    print("SUMMARY (val n=3209, early-stopping bias warning):")
    print("=" * 70)
    print(f"  ELO logistic (re-eval) : 0.8524559696")
    print(f"  Tier 1 (ES=50)         : {loss:.10f}  Δ={loss - 0.8524559696:+.6f}")
    print(f"  Tier 1 (ES=20)         : {loss2:.10f}  Δ={loss2 - 0.8524559696:+.6f}")
    print(f"  Tier 1 (Elo-only)      : {loss3:.10f}  Δ={loss3 - 0.8524559696:+.6f}")


if __name__ == "__main__":
    main()
