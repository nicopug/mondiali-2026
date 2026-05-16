# Backtest — WC2022 + Euro2024

**Date generated:** 2026-05-16  
**Model:** `models/v1_final/` (v1.0)  
**Matches:** 115 (WC2022=64, Euro2024=51)

## ⚠ Important caveat

WC2022 falls in the **val_es** window (2022-07-01 → 2022-12-31, used for XGBoost early stopping) and Euro2024 falls in the **val_gate** window. The model has seen both distributions during training. This backtest is a demonstration of model behaviour on famous matches, **not** an unbiased out-of-sample evaluation.

Walk-forward state: for each match, Elo + form-cache are rebuilt from matches strictly before `match_date`. No future leakage in the state.

## Aggregate metrics

| Market | All (115) | WC2022 (64) | Euro2024 (51) |
|---|---|---|---|
| 1x2 | ll=0.825 br=0.480 | ll=0.679 br=0.380 | ll=1.008 br=0.606 |
| over_under_1_5 | ll=0.459 br=0.147 | ll=0.388 br=0.118 | ll=0.548 br=0.183 |
| over_under_2_5 | ll=0.528 br=0.175 | ll=0.401 br=0.117 | ll=0.688 br=0.247 |
| over_under_3_5 | ll=0.367 br=0.108 | ll=0.314 br=0.088 | ll=0.433 br=0.132 |
| btts | ll=0.574 br=0.194 | ll=0.459 br=0.141 | ll=0.718 br=0.261 |

## 5 best calls (model gave highest probability to the actual outcome)

- **2022-11-23 Spain 7–0 Costa Rica** (WC2022) — P(H/D/A)=0.99/0.01/0.00, P(actual)=0.99
- **2022-11-21 England 6–2 Iran** (WC2022) — P(H/D/A)=0.92/0.04/0.03, P(actual)=0.92
- **2022-11-29 Wales 0–3 England** (WC2022) — P(H/D/A)=0.01/0.09/0.90, P(actual)=0.90
- **2022-11-22 France 4–1 Australia** (WC2022) — P(H/D/A)=0.88/0.08/0.04, P(actual)=0.88
- **2022-12-06 Portugal 6–1 Switzerland** (WC2022) — P(H/D/A)=0.87/0.08/0.05, P(actual)=0.87

## 5 biggest misses (model assigned lowest probability to the actual outcome)

- **2024-06-26 Georgia 2–0 Portugal** (Euro2024) — P(H/D/A)=0.06/0.19/0.75, P(actual)=0.06
- **2024-06-17 Belgium 0–1 Slovakia** (Euro2024) — P(H/D/A)=0.55/0.29/0.16, P(actual)=0.16
- **2024-06-25 Netherlands 2–3 Austria** (Euro2024) — P(H/D/A)=0.59/0.24/0.17, P(actual)=0.17
- **2022-12-18 Argentina 3–3 France** (WC2022) — P(H/D/A)=0.47/0.19/0.34, P(actual)=0.19
- **2024-06-24 Croatia 1–1 Italy** (Euro2024) — P(H/D/A)=0.19/0.21/0.60, P(actual)=0.21

## Notable matches (finals + key knockouts)

- **2022-12-18 Argentina 3–3 France** (WC2022) — P(H/D/A)=0.47/0.19/0.34, P(actual)=0.19
- **2024-07-14 Spain 2–1 England** (Euro2024) — P(H/D/A)=0.43/0.26/0.31, P(actual)=0.43
- **2022-12-13 Argentina 3–0 Croatia** (WC2022) — P(H/D/A)=0.71/0.20/0.09, P(actual)=0.71
- **2022-12-14 France 2–0 Morocco** (WC2022) — P(H/D/A)=0.64/0.24/0.12, P(actual)=0.64
- **2024-07-09 Spain 2–1 France** (Euro2024) — P(H/D/A)=0.44/0.32/0.24, P(actual)=0.44
- **2024-07-10 Netherlands 1–2 England** (Euro2024) — P(H/D/A)=0.38/0.29/0.33, P(actual)=0.33