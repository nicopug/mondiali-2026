# DL Levels Comparison (L1 vs L2 vs L3) — val_gate 2024

**Date:** 2026-05-16  
**v1_final XGB raw baseline:** log_loss=0.9044, brier=0.5329  
**Promotion threshold:** Δ log-loss < -0.005 vs XGB alone

Each DL model trained on identical splits (train 2002-2023, val_es Jul-Dec 2022). Ensembles: average of lambdas, fresh DC ρ estimated on training set lambdas, then standard DC correction + 1X2.

## Ranked results (best → worst)

| Rank | Model | log_loss | Δ vs XGB | brier |
|---|---|---|---|---|
| 1 | `ensemble_XGB+L1+L3_equal` | 0.8885 | -0.0159 ✅ | 0.5237 |
| 2 | `ensemble_XGB+L3_50_50` | 0.8895 | -0.0149 ✅ | 0.5247 |
| 3 | `ensemble_XGB+L1_50_50` | 0.8896 | -0.0148 ✅ | 0.5241 |
| 4 | `ensemble_4way_equal` | 0.8906 | -0.0138 ✅ | 0.5252 |
| 5 | `ensemble_XGB+L1+L2_equal` | 0.8926 | -0.0118 ✅ | 0.5262 |
| 6 | `ensemble_XGB+L2+L3_equal` | 0.8927 | -0.0117 ✅ | 0.5267 |
| 7 | `ensemble_L1+L3_50_50` | 0.8972 | -0.0072 ✅ | 0.5279 |
| 8 | `ensemble_L1+L2+L3_equal` | 0.8986 | -0.0058 ✅ | 0.5296 |
| 9 | `ensemble_XGB+L2_50_50` | 0.9006 | -0.0037 | 0.5306 |
| 10 | `L1_MLP_alone` | 0.9028 | -0.0016 | 0.5303 |
| 11 | `v1_final_XGB` | 0.9044 | +0.0000 | 0.5329 |
| 12 | `L3_Bivariate_alone` | 0.9053 | +0.0009 | 0.5333 |
| 13 | `ensemble_L1+L2_50_50` | 0.9059 | +0.0015 | 0.5335 |
| 14 | `ensemble_L2+L3_50_50` | 0.9059 | +0.0015 | 0.5342 |
| 15 | `L2_Sequence_alone` | 0.9469 | +0.0425 | 0.5542 |

## DL training info

- L1 MLP: best val_es NLL = 0.7425, epochs=100
- L2 Sequence (GRU h=32, hist_len=8): best val_es NLL = 0.6429, epochs=100
- L3 Bivariate (per-match rho): best val_es DC-NLL = 2.6335, epochs=90

## Notes

- Pairwise/3-way/4-way ensembles use **equal** weights (50/50, 33/33/33, 25/25/25/25) for fairness in comparison. Fine-grid weight search on val_calib could squeeze a bit more but risks overfitting.
- L3 uses **per-match learned ρ** (range [-0.3, 0.3] via tanh) rather than the single globally-estimated ρ for the other models. This is a structural advantage but also a harder optimization target.
- All four base models share the same train/val_es splits → no leakage in ensembles.