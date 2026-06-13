"""STEP 0 diagnostic: does squad-talent gap explain v1.4's residual errors?

Uses XGB-only v1_final lambdas (no torch needed). On the OOS 2025-2026 slice,
on rows WITH market value, correlates talent_gap_top11 with:
  - the signed goal-diff residual: (home_score - away_score) - (lam_h - lam_a)
If positive and significant, high-talent teams beat v1.4's expectation -> the
lever exists and is currently under-exploited.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from mondiali.features.talent import add_talent_features
from mondiali.model.poisson_xgb import PoissonXGBModel


def main() -> None:
    df = pd.read_parquet("data/processed/matches.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["days_rest_home", "days_rest_away"]).copy()
    df = add_talent_features(df)

    oos = df[(df["date"] >= "2025-01-01") & (df["date"] <= "2026-03-31")].reset_index(drop=True)

    xgb = PoissonXGBModel().load(Path("models/v1_final/xgb_poisson.json"))
    lam_h, lam_a = xgb.predict_lambda(oos)

    actual_gd = (oos["home_score"] - oos["away_score"]).to_numpy(dtype=float)
    pred_gd = lam_h - lam_a
    residual = actual_gd - pred_gd  # >0: home did better than model expected

    mask = oos["talent_gap_top11"].notna().to_numpy()
    gap = oos["talent_gap_top11"].to_numpy(dtype=float)[mask]
    res = residual[mask]

    n = int(mask.sum())
    if n < 10:
        result = {"n_with_market_value": n, "verdict": "insufficient coverage"}
    else:
        pear_r, pear_p = stats.pearsonr(gap, res)
        spear_r, spear_p = stats.spearmanr(gap, res)
        slope = float(np.polyfit(gap, res, 1)[0])
        result = {
            "n_oos": int(len(oos)),
            "n_with_market_value": n,
            "coverage": round(n / len(oos), 4),
            "pearson_r": round(float(pear_r), 4),
            "pearson_p": round(float(pear_p), 6),
            "spearman_r": round(float(spear_r), 4),
            "spearman_p": round(float(spear_p), 6),
            "ols_slope": slope,
            "verdict": (
                "lever exists (talent gap predicts residual)"
                if pear_p < 0.05 and pear_r > 0
                else "no usable signal"
            ),
        }

    Path("reports").mkdir(exist_ok=True)
    Path("reports/talent_diagnostic.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
