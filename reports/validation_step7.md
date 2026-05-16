# STEP 7 — Tier 7 DL + Ensemble: v1.0 → v1.1 promotion

**Date:** 2026-05-16
**Branch:** `feat/step6-freeze` (continuing post-freeze improvements before kickoff)
**Frozen artefacts:** `models/v1_final/` (now v1.1)
**Tag plan:** v1.1 on next push

---

## 1. Outcome

**Promosso ensemble XGB + DL Tier 7** sul val_gate 2024. Nuova versione: **v1.1**.

| Modello | val_gate 2024 log-loss | Δ vs v1_final raw |
|---|---|---|
| v1_final XGB (raw) | 0.9044 | baseline |
| Tier 7 DL solo (raw) | 0.9028 | -0.0016 (close, not promoted alone) |
| **Ensemble (0.8 XGB + 0.2 DL)** | **0.8946** | **-0.0098 ✅ PROMOSSO** |

Soglia promotion: Δ < -0.005. Risultato 2× la soglia.

Peso DL=0.2 selezionato su **val_calib 2023** (non val_gate, anti-leakage), poi valutato unbiased su val_gate 2024.

## 2. Architettura Tier 7 DL

```
Input per perspective row (24 features):
    team_id ────► nn.Embedding(n_teams=323+1, dim=16) ─┐
    opp_id  ────► nn.Embedding (shared weights)       ─┤
    24 z-score-normalized features ───────────────────┤
                                                       ▼
                          Concat (16+16+24 = 56 dim)
                                                       ▼
                          Linear(56 → 128) + ReLU + Dropout(0.2)
                                                       ▼
                          Linear(128 → 64)  + ReLU + Dropout(0.2)
                                                       ▼
                          Linear(64 → 1) → log_lambda
                                                       ▼
                          exp() → lambda (Poisson NLL loss)
```

- **Symmetric architecture**: ogni match produce 2 righe (home + away perspective) come XGBoost.
- **Team embeddings**: 16-dim per nazione, condivisi tra ruoli home/away. Index 0 = `<UNK>` per nazioni non viste.
- **Feature normalization**: z-score con mean/std calcolati su train (~20k match × 24 feature).
- **Determinism**: `torch.manual_seed(42)` + grad clip 1.0.
- **Training**: Adam(lr=1e-3, wd=1e-5), cosine LR schedule, batch 512, max 100 epoch, early stop patience 10 su val_es Poisson NLL.
- **Best val_es NLL**: 0.741 (raggiunto a epoch 100, ancora calante)

## 3. Ensemble strategy

```
λ_ensemble = w_xgb · λ_xgb + w_dl · λ_dl  (component-wise)
ρ_ensemble = estimate_rho_mle(λ_ensemble train lambdas, training goals)
joint = poisson_outer_product(λ_ensemble) → dixon_coles_correct(ρ_ensemble) → normalize
markets = prob_1x2(joint), prob_over_under(joint, line=1.5/2.5/3.5), prob_btts(joint)
```

**Weight search**: ricerca grid w_dl ∈ [0.2, 0.8] step 0.05 minimizzando log-loss su **val_calib 2023**. Best: w_dl=0.2.

Sweep su val_gate (per riferimento, NON usato per decisione):
| w_dl | val_gate log-loss |
|---|---|
| 0.3 | 0.8918 |
| 0.4 | 0.8902 |
| **0.5** (50/50) | **0.8896** ← minimum on gate |
| 0.6 | 0.8901 |
| 0.7 | 0.8917 |

Nota: il vero optimum su val_gate è 0.5, ma per integrità anti-leakage si seleziona su val_calib (→ 0.2). Differenza: 0.0050 log-loss. Trade-off accettato: meno overfitting al gate.

## 4. Artefatti `models/v1_final/` (v1.1)

```
xgb_poisson.json            # XGBoost (invariato)
rho.txt                     # rho XGB-only (legacy)
dl/
├── weights.pt              # PyTorch state_dict (CPU)
├── team_idx.json           # {nation: int_id}
├── feature_stats.json      # z-score mean/std per le 24 feature
└── config.json             # embed_dim, hidden, dropout, n_teams, n_features
ensemble.json               # {weight_xgb, weight_dl, rho_ensemble}
manifest.json               # version=v1.1, include ensemble block
markets_validation.json     # per-market Brier vs baseline (invariato — ensemble non re-validato sui market secondari)
```

`ensemble.json`:
```json
{
  "weight_xgb": 0.8,
  "weight_dl": 0.2,
  "rho_ensemble": -0.0706,
  "selected_on": "val_calib_log_loss"
}
```

## 5. Inference path

`predict_match` auto-detect:
- Se `ensemble.json` + `dl/` esistono → load entrambi i modelli, average lambdas, usa rho_ensemble.
- Altrimenti → XGB-only legacy.

Output JSON include `"ensemble": true/false` flag.

```bash
$ mondiali predict France Italy 2026-06-15 --neutral
{
  "match": {"home": "France", "away": "Italy", "date": "2026-06-15", "neutral": true},
  "model_version": "v1.1",
  "ensemble": true,
  "lambda": {"home": 1.498, "away": 1.041},
  ...
}
```

## 6. Why ensemble works

Diversità di errori. XGB e DL hanno bias diversi:
- XGB cattura interazioni feature-feature non-lineari ma non identità della nazione (Elo è proxy debole).
- DL impara embedding latenti per nazione che catturano stile/identità.
- Media riduce noise, mantiene segnale comune. Classico effetto bagging tra model class diversi.

Brier improvement: 0.5329 → 0.5240 (-0.0089).

## 7. Why DL alone almost won but didn't quite

Tier 7 da solo: 0.9028 (-0.0016 vs XGB). Sotto la soglia di promotion (-0.005). Sample size piccolo (~20k matches) limita quanto in più può imparare il DL rispetto a XGB feature engineering.

Tentato larger config (32 embed, hidden 256-128-64, dropout 0.3, 200 epoch, patience 20) → overfit (gate 0.9716). 16/128-64 con patience 10 è il sweet spot.

## 8. Backward compatibility

Se `models/v1_final/` non ha `ensemble.json` (legacy v1.0 freeze) → predict_match cade su path XGB-only. Zero rotture.

## 9. Acceptance per STEP 7

- [x] Modulo `src/mondiali/model/dl_poisson.py` (PoissonEmbeddingModel + train/predict/save/load)
- [x] Pipeline `src/mondiali/training/train_tier7.py`
- [x] CLI `mondiali train-tier7` per training standalone
- [x] Ensemble integration in `freeze_v1_final` con auto-skip se non batte la soglia
- [x] `predict_match` supporta ensemble
- [x] Tests verdi (7/7 nuovi test DL, 233 totali)
- [x] Ensemble val_gate Δ < -0.005 vs XGB-only
- [x] manifest.json versione bumped v1.0 → v1.1

## 10. Limiti + lavoro futuro

| Limite | Mitigazione possibile (post-WC) |
|---|---|
| Weight optimization solo grid 0.05 step | Ottimizzazione continua (1D scipy.optimize_scalar) |
| Single DL seed (=42) | Average ensemble dei DL (5 seed) — costo: 5× training time |
| DL config non tuned via Optuna | Optuna search su embed_dim, hidden, dropout, lr |
| Calibrator 1X2 ancora auto-skipped | Refit su ensemble probs invece di XGB-only |
| Markets validation ancora su XGB-only lambdas | Rivalidare U/O e BTTS su ensemble lambdas |
| Tier 7 model non versionato in `models/v1_final/dl/` (file binario) | Salvare hash + linea di provenance nel manifest (già fatto) |

## 11. Commits

```
(dl tier 7) feat(dl): Tier 7 team-embedding MLP Poisson model
(this step) feat(ensemble): integrate Tier 7 DL ensemble into v1_final freeze
```
