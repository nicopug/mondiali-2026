"""Explain a single prediction by computing SHAP feature contributions.

For the XGBoost component (Poisson regressor), SHAP provides per-feature
contribution values that sum to the log-lambda output. We use these to surface
the top-K most influential features in the predicted lambdas.

For the ensemble (XGB + DL), we explain only the XGB contribution since:
- DL Shapley is more expensive and less interpretable (per-team embeddings)
- XGB carries the largest weight (0.85 in v1.2)
- The user gets the most actionable insight from XGB features anyway
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from mondiali.model.poisson_xgb import SYMMETRIC_FEATURES, build_symmetric_rows


def explain_prediction(
    row: pd.DataFrame,
    booster: xgb.XGBRegressor,
    *,
    top_k: int = 3,
) -> dict:
    """Return SHAP-based explanation for a single-row matches DataFrame.

    Output schema:
        {
            "home_lambda_drivers": [{"feature": str, "value": float, "contribution": float}, ...],
            "away_lambda_drivers": [...],
            "base_log_lambda_home": float,
            "base_log_lambda_away": float,
        }
    """
    if len(row) != 1:
        raise ValueError(f"explain_prediction expects 1-row matches; got {len(row)}")

    X, _ = build_symmetric_rows(row, include_tier4=False)
    # X has 2 rows: 0 = home perspective, 1 = away perspective
    dmat = xgb.DMatrix(X, feature_names=SYMMETRIC_FEATURES)
    # pred_contribs returns shape (n_rows, n_features + 1) where last col is the bias
    contribs = booster.get_booster().predict(dmat, pred_contribs=True)
    home_contrib = contribs[0]
    away_contrib = contribs[1]
    home_base = float(home_contrib[-1])
    away_base = float(away_contrib[-1])
    home_feat = home_contrib[:-1]
    away_feat = away_contrib[:-1]

    def _top(feat_contrib: np.ndarray, x_values: np.ndarray) -> list[dict]:
        abs_idx = np.argsort(-np.abs(feat_contrib))[:top_k]
        return [
            {
                "feature": SYMMETRIC_FEATURES[i],
                "value": float(x_values[i]),
                "contribution": float(feat_contrib[i]),
            }
            for i in abs_idx
        ]

    return {
        "home_lambda_drivers": _top(home_feat, X[0]),
        "away_lambda_drivers": _top(away_feat, X[1]),
        "base_log_lambda_home": home_base,
        "base_log_lambda_away": away_base,
    }
