"""True out-of-sample evaluation: 2025-01-01 → 2026-03-31.

These matches were NEVER seen during training (train end 2023-12-31, val_calib
2023, val_gate 2024). This is the only truly unbiased slice for the final
freeze decision.

Evaluates v1_final (ensemble or XGB-only depending on what's in ensemble.json)
+ XGB-only baseline + each DL alone (L1, L3 if present) on this OOS slice.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mondiali.model.dixon_coles import dixon_coles_correct, estimate_rho_mle, joint_matrix
from mondiali.model.markets import prob_1x2
from mondiali.model.poisson_xgb import PoissonXGBModel
from mondiali.training.evaluate import brier_score_1x2, log_loss_1x2


def _compute_1x2(lam_h, lam_a, rho):
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


def main(model_dir: Path = Path("models/v1_final")) -> dict:
    df = pd.read_parquet("data/processed/matches.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["days_rest_home", "days_rest_away"])

    train = df[(df["date"] >= "2002-01-01") & (df["date"] <= "2023-12-31")].reset_index(drop=True)
    oos = df[(df["date"] >= "2025-01-01") & (df["date"] <= "2026-03-31")].reset_index(drop=True)
    print(f"OOS slice: {len(oos)} matches (2025-01-01 to 2026-03-31)")
    print(f"Tournaments in OOS: {sorted(oos['tournament'].unique())[:5]}... ({oos['tournament'].nunique()} total)")

    h_goals_tr = train["home_score"].to_numpy()
    a_goals_tr = train["away_score"].to_numpy()

    # XGB v1_final
    xgb = PoissonXGBModel().load(model_dir / "xgb_poisson.json")
    rho_xgb = float((model_dir / "rho.txt").read_text().strip())
    lam_h_x, lam_a_x = xgb.predict_lambda(oos)
    probs = _compute_1x2(lam_h_x, lam_a_x, rho=rho_xgb)
    xgb_ll = float(log_loss_1x2(oos, probs))
    xgb_br = float(brier_score_1x2(oos, probs))
    print(f"\nXGB-only:        ll={xgb_ll:.4f}  brier={xgb_br:.4f}")

    results = {"xgb_only": {"log_loss": xgb_ll, "brier": xgb_br}}

    # Ensemble (if present)
    ens_path = model_dir / "ensemble.json"
    if ens_path.exists():
        ens = json.loads(ens_path.read_text())
        w_xgb = float(ens["weight_xgb"])
        w_l1 = float(ens.get("weight_l1", 0.0))
        w_l3 = float(ens.get("weight_l3", 0.0))
        lam_h = w_xgb * lam_h_x
        lam_a = w_xgb * lam_a_x

        lam_h_tr = w_xgb * xgb.predict_lambda(train)[0]
        lam_a_tr = w_xgb * xgb.predict_lambda(train)[1]

        if w_l1 > 1e-3 and (model_dir / "dl").exists():
            from mondiali.model.dl_poisson import load_dl_model, predict_lambda as l1p
            m1, idx1, st1, _ = load_dl_model(model_dir / "dl")
            lh1, la1 = l1p(m1, oos, idx1, st1)
            lam_h += w_l1 * lh1
            lam_a += w_l1 * la1
            lh1_tr, la1_tr = l1p(m1, train, idx1, st1)
            lam_h_tr += w_l1 * lh1_tr
            lam_a_tr += w_l1 * la1_tr
            print(f"  +L1 ({w_l1:.2f})")
        if w_l3 > 1e-3 and (model_dir / "l3").exists():
            from mondiali.model.dl_bivariate import load_bivariate, predict_lambda_rho
            m3, idx3, st3, _ = load_bivariate(model_dir / "l3")
            lh3, la3, _ = predict_lambda_rho(m3, oos, idx3, st3)
            lam_h += w_l3 * lh3
            lam_a += w_l3 * la3
            lh3_tr, la3_tr, _ = predict_lambda_rho(m3, train, idx3, st3)
            lam_h_tr += w_l3 * lh3_tr
            lam_a_tr += w_l3 * la3_tr
            print(f"  +L3 ({w_l3:.2f})")

        rho_ens = estimate_rho_mle(lam_h_tr, lam_a_tr, h_goals_tr, a_goals_tr)
        probs = _compute_1x2(lam_h, lam_a, rho=rho_ens)
        ens_ll = float(log_loss_1x2(oos, probs))
        ens_br = float(brier_score_1x2(oos, probs))
        delta = ens_ll - xgb_ll
        print(f"Ensemble:        ll={ens_ll:.4f}  brier={ens_br:.4f}  delta={delta:+.4f}")
        results["ensemble"] = {
            "log_loss": ens_ll, "brier": ens_br,
            "delta_vs_xgb": delta, "weights": ens,
        }

    # Per tournament breakdown
    print("\nBy tournament (top 5):")
    by_t = oos.groupby("tournament").size().sort_values(ascending=False).head(5)
    for t, n in by_t.items():
        slice_idx = oos["tournament"] == t
        ll_t = float(log_loss_1x2(oos[slice_idx], probs[slice_idx.to_numpy()]))
        print(f"  {t} ({n} matches): ll={ll_t:.4f}")

    out_path = Path("reports/oos_2025_2026_metrics.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
    return results


if __name__ == "__main__":
    main()
