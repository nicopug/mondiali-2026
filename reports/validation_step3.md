# STEP 3 — Tier 2 form features + isotonic calibration

**Data**: 2026-04-28
**Commit**: 67a325e
**Python**: 3.12.9
**XGBoost**: 3.2.0
**Predecessore**: STEP 2 falliva il gate con Δ = -0.0004 vs ELO baseline.

## TL;DR

- **Gate soft (`raw < LOGLOSS_ELO`): PASS.**
- **Gate hard (`raw ≤ LOGLOSS_ELO − 0.003`): PASS.**
- Tier 2 RAW log-loss = **0.8487** vs ELO baseline = **0.8525** → **Δ = -0.0038** (meglio di 0.38 punti millesimali, *oltre* la soglia hard di 0.003).
- L'isotonic calibrator su `val_calib=2018` (n=923) **degrada** la log-loss su `val_gate` (0.8487 → 0.9264). Lo lasciamo nel codice e nei file salvati come baseline futura, ma il *gate ufficiale di STEP 3 è sulla raw*.

## Setup

### Pipeline Tier 2

- 4-way split temporale (no random):
  - **Train**: 2002-01-01 → 2016-12-31 (n=14'161)
  - **Val_ES** (early stopping carve-out): 2017-01-01 → 2017-12-31 (n=924)
  - **Val_calib** (isotonic fit): 2018-01-01 → 2018-12-31 (n=923)
  - **Val_gate** (metric finale): 2019-01-01 → 2022-06-30 (n=3'209)
- Match con `days_rest_{home,away}` NaN droppati come in STEP 2.
- Tier 2 features (10 nuove, rolling N=5, `closed='left'`): `home/away_form_5`, `gd_5`, `goals_scored_5`, `goals_conceded_5`, `avg_opp_elo_5`. Il modello vede 18 feature simmetriche (era 8 in STEP 2).
- XGBoost Poisson con ES su Val_ES (rounds=50). ρ Dixon-Coles stimato MLE su Train.
- Isotonic calibrator (3 isotonic indipendenti P1/PX/P2 + rinormalizzazione riga) fittato su raw probs di Val_calib.

### Baseline confronto (STEP 2)

ELO-only logistic, ricomputato sul val_gate *filtrato* (apples-to-apples, n=3'209): **0.8525**.

## Risultati

| Metrica | Valore |
|---|---|
| `val_log_loss_raw` | **0.8487** |
| `val_log_loss_calib` | 0.9264 |
| `brier_before` | 0.4976 |
| `brier_after` | 0.5023 |
| Dixon-Coles ρ | -0.0554 |
| n_train / n_val_es / n_val_calib / n_val_gate | 14'161 / 924 / 923 / 3'209 |

### Gate

| Gate | Soglia | Risultato | Esito |
|---|---|---|---|
| Soft | `raw < 0.8525` | 0.8487 | ✅ PASS (Δ = -0.0038) |
| Hard | `raw ≤ 0.8495` | 0.8487 | ✅ PASS (Δ = -0.0038, margine 0.0008) |

### Confronto STEP-by-STEP

| STEP | Modello | log-loss val_gate | Δ vs ELO |
|---|---|---|---|
| 0 | Prior 1/X/2 | ~1.05 | +0.20 |
| 1 | ELO logistic | 0.8525 | 0 (baseline) |
| 2 | Tier 1 XGBoost (8 feat) | 0.8528 | +0.0003 ❌ |
| 3 | Tier 2 RAW (18 feat) | **0.8487** | **-0.0038** ✅ |
| 3 | Tier 2 + isotonic | 0.9264 | +0.0739 ⚠ |

## Finding empirico inatteso: l'isotonic calibrator peggiora

L'ipotesi di STEP 3 (spec §3) era: «un calibrator isotonic post-hoc su val_calib=2018 può ridurre log-loss su val_gate». **L'esperimento la falsifica**: la calibration aumenta la log-loss di 0.078 nat e il Brier di 0.005.

Cause probabili (non verificate, da raccogliere come hypothesis-list per STEP 4):

1. **Sample size insufficiente.** 923 match per fittare 3 × isotonic indipendenti significa ~300 match per classe in media. Le isotonic con `out_of_bounds='clip'` overfittano frequenze locali del 2018.
2. **Distribution shift 2018 → 2019-22.** Val_gate include Euro2020 (rinviato), qualifications WC2022, COVID-era friendlies. Le base-rate 1/X/2 e la dispersione dei lambda XGB cambiano materialmente.
3. **Per-class renormalization** in `IsotonicCalibrator1X2.predict` può amplificare rumore: se le 3 isotonic mappano a 0.4/0.05/0.4, la rinormalizzazione spinge la P_draw a ~0.06, ma il segnale isotonic di P_draw è dominato dalla varianza.

L'XGBoost RAW post-Tier 2 produce probabilità *già* meglio calibrate di Tier 1 (Brier 0.4976 vs Tier 1 ~0.50+). Il bottleneck di STEP 2 era *features*, non *calibration*.

## Decisione

- **Gate STEP 3: PASS sulla raw.** Si chiude STEP 3.
- Calibrator code mantenuto in `src/mondiali/model/calibration.py` con suite test verde (4 test fit/predict + 2 test JSON-roundtrip). Disponibile per STEP 4+ con sample maggiore.
- Artefatti salvati in `models/tier2/`:
  - `xgb_poisson.json` — booster XGBoost JSON-native
  - `calibrator.json` — isotonic calibrator (informativo, *non* usato a inference time per default)
- Slow-test gate (`tests/test_train_tier2.py::test_train_tier2_full_split_produces_reasonable_loss`) gata su `val_log_loss_raw ∈ [0.83, 0.86]`. La metric `val_log_loss_calib` resta nel risultato del pipeline ma non è bloccante.

## Anti-data-leakage

Tutti i 4 test in `tests/test_leakage.py` passano (Elo strict-pre, no future matches, days_rest strict-pre, **Tier 2 form_5 strict-pre**). Una regression catturata durante Task 2: `_team_long_form` non rompeva le ties di stessa data, causando inclusione del match corrente nella rolling per coppie di partite stessa-data per stessa squadra (es. Uruguay 1916-08-15). Fix in commit `24cd117`: tiebreaker `match_idx` nell'ordinamento. Test regressione `test_tier2_same_date_ordering` aggiunto.

## Test suite

- 124 test pre-Task 7 → **127 test** post-STEP 3 (incluso lo slow gate).
- Slow test passato (`val_log_loss_raw=0.8487 ∈ [0.83, 0.86]`, ρ=-0.0554 ∈ [-0.3, 0.05]).

## Aperti per STEP 4

1. Investigare perché isotonic peggiora — provare temperature scaling (1 scalare) o Platt scaling.
2. Optuna sui 18 feature (deferred da STEP 3).
3. Cross-fit calibration: fittare calibrator su *fold rotanti* per ridurre overfit a singolo anno.
4. Decay weights nel rolling Tier 2 (oggi peso uniforme su ultimi 5).
