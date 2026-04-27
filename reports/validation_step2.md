# STEP 2 — Tier 1 validation report

**Data**: 2026-04-27
**Commit**: 0af9c09
**Python**: 3.12.9
**XGBoost**: 3.2.0

## Dataset & Split

- Input: `data/processed/matches.parquet`
- Match con `days_rest_{home,away}` NaN droppati (prima apparizione team): **300** (su tutto lo storico)
- Training (2002-01-01 → 2018-12-31): **16'008**
- Validation (2019-01-01 → 2022-06-30): **3'209**

> **Nota sul confronto baseline**: il `LOGLOSS_ELO` canonico di CP2 (Task 5) è stato calcolato **senza** il filtro `dropna(days_rest_*)` (n_val=3215). Il pipeline Tier 1 droppa 6 match in val perché `days_rest` non è disponibile alla prima apparizione di una squadra. Per un confronto apples-to-apples ho ricalcolato il baseline anche sul val filtrato (n=3209). Entrambi i numeri sono riportati sotto.

## Baseline Elo-only logistic

**Features**: `[elo_diff, is_neutral]`

| Variante | n_train | n_val | log-loss |
|---|---|---|---|
| Originale Task 5 (no filter) | 16'063 | 3'215 | **0.8527** |
| Filtrata come Tier 1 | 16'008 | 3'209 | **0.8525** |

Coefficienti appresi (variante originale, classi `[P_home, P_draw, P_away]`):
- `elo_diff`: `[+0.00404, -0.00011, -0.00393]` (positivo per P_home, negativo per P_away — atteso)
- `is_neutral`: `[-0.334, -0.072, +0.407]` (campo neutro penalizza l'home, premia l'away — atteso)

## Tier 1 — XGBoost Poisson + Dixon-Coles

**Features (8)**: `team_elo, opponent_elo, elo_diff_signed, is_home, is_neutral, competition_importance, team_days_rest, opponent_days_rest`.

**Hparams** (hand-tuned, optuna → STEP 3):
- objective=count:poisson, tree_method=hist
- max_depth=6, lr=0.05, n_estimators=2000 + early_stopping(50)
- reg_alpha=0.1, reg_lambda=1.0, min_child_weight=1
- subsample=0.9, colsample_bytree=0.9
- best_iteration osservato: **142**

**Dixon-Coles ρ stimato** (MLE su training): `-0.0585` (atteso ∈ [-0.15, -0.03] ✓)

**λ diagnostica** su validation:
- lambda_home_mean: **1.6620** | home_score_mean osservato: **1.6223**
- lambda_away_mean: **1.1391** | away_score_mean osservato: **1.0960**

Le medie predette sovrastimano il segnato di ~0.04–0.05 gol — bias trascurabile, calibrazione di scala accettabile.

**Validation log-loss**: `LOGLOSS_TIER1 = 0.8528`

> **⚠ Bias ottimistico**: la val è usata sia per early stopping sia per il log-loss riportato. La cifra è quindi una stima ottimisticamente biased della generalizzazione. Per uno STEP 3 pulito serve un carve-out dedicato per ES.

## Gate: Tier 1 vs Elo-only — **FALLITO**

| Confronto | LOGLOSS_ELO | LOGLOSS_TIER1 | Δ (ELO − TIER1) |
|---|---|---|---|
| Apples-to-oranges (val Task 5 originale) | 0.8527 | 0.8528 | **−0.0001** |
| Apples-to-apples (val filtrato n=3209) | 0.8525 | 0.8528 | **−0.0004** |

Soglia: **Δ ≥ +0.003** (spec §7.5 tier-gate).

- [ ] Δ ≥ 0.003 → gate passato.
- [x] **Δ < 0.003 → gate fallito.** In entrambe le varianti Tier 1 è marginalmente *peggiore* di una logistic con 2 sole feature.

Considerato l'optimistic bias dell'ES su val, lo stato vero è plausibilmente ancora più sfavorevole a Tier 1.

## Feature importance (gain)

| Feature | Gain | % | SHAP mean(\|·\|) |
|---|---|---|---|
| elo_diff_signed | 38.30 | 40.16% | 0.354 |
| is_home | 16.62 | 17.43% | 0.153 |
| is_neutral | 15.25 | 15.99% | 0.082 |
| opponent_elo | 7.40 | 7.75% | 0.084 |
| team_elo | 5.34 | 5.59% | 0.060 |
| opponent_days_rest | 5.09 | 5.34% | 0.031 |
| competition_importance | 3.82 | 4.01% | 0.015 |
| team_days_rest | 3.56 | 3.73% | 0.017 |

`elo_diff_signed` domina al ~40% (non al >80%): il segnale Elo è la spina dorsale ma non l'unico contributo. `competition_importance` ha SHAP 0.015 — quasi rumore.

## Sanity check — Francia vs San Marino (neutral, qualif)

- lambda_home (France): **6.06**
- lambda_away (San Marino): **0.16**
- P(France win): **0.9947** ✓ (atteso > 0.85)
- P(draw): 0.0049
- P(SMR win): 0.0004

Il modello cattura correttamente differenze Elo massive — non è un problema di capacity sui casi facili.

## Debug trail

Ho condotto un diagnostic mirato (script `scripts/diagnose_tier1.py`) per capire perché il gate è fallito:

| Run | Setup | log-loss | Δ vs Elo logistic 0.8525 |
|---|---|---|---|
| 1 | Tier 1 full, ES=50 | 0.8528 | +0.0004 |
| 2 | Tier 1 full, ES=20 | 0.8528 (best_iter identico = 142) | +0.0004 |
| 3 | Tier 1 *senza* `competition_importance` + `days_rest` | 0.8562 | +0.0037 |

**Findings:**

1. **Le 3 nuove feature hanno segnale.** Drop-le peggiora di +0.0034. `days_rest` aggiunge ~0.05 SHAP combinato; `competition_importance` è quasi rumore (SHAP 0.015) ma non dannoso.
2. **ES non è il bottleneck.** ES=20 e ES=50 atterrano sullo stesso `best_iteration=142` e log-loss identico — non è un problema di stopping aggressivo.
3. **Il leakage non è in causa.** `tests/test_leakage.py` verde (3/3); il filtro `days_rest` è già strict pre-match; sanity case è coerente.
4. **Diagnosi: model calibration, non feature.** XGBoost con `count:poisson` su 8 feature trova interazioni non-lineari nel pattern Elo dominante (40% gain) che peggiorano la calibration vs una logistic vincolata a essere lineare in `elo_diff`. La complessità in eccesso costa ~0.0004 di log-loss.

## Lezioni apprese

- **Una logistic con 2 feature è una baseline durissima** quando `elo_diff` è il segnale dominante. La trasformazione lineare → softmax è già "near-optimal" su questo asse, e qualsiasi modello più potente paga in calibration.
- **Aggiungere feature con SHAP < 0.03 non muove l'ago.** `competition_importance` (SHAP 0.015) è quasi rumore — Tier 2 deve mirare a feature con SHAP atteso > 0.05 (`is_home`, `opponent_elo` come riferimento).
- **L'early-stopping bias va eliminato in STEP 3.** Il `LOGLOSS_TIER1=0.8528` è ottimisticamente biased; il vero gap potrebbe essere maggiore.

## Decisioni open per STEP 3

- **Tier 2 features**: rolling form (5 e 10 match: W/D/L, GF/GA, GD), qualità media avversari recenti, h2h ultimi N incontri. Priorità su feature con segnale forte (drop ulteriore di `competition_importance` se SHAP rimane <0.02 in Tier 2).
- **Modello finale 1X2**: valutare se mantenere XGBoost Poisson + DC come base o introdurre uno stacker/calibrator a valle (logistic o isotonic) che potrebbe correggere lo svantaggio di calibration osservato.
- **Carve-out per ES**: split del training in `train_inner` + `val_es` (e val per metrica finale rimane intoccato), per misurare il log-loss senza optimistic bias.
- **Friendly nel training**: includono i Mondiali 2018/2022, ma sono noisy. Tagliarli dal training del modello (mantenendoli per Elo update) potrebbe ridurre rumore.
- **Optuna**: 50 trial sui walk-forward folds di Task 3, con warm-start dagli hparams attuali.

## Test suite

```
.venv/Scripts/pytest -q  →  106 passed
.venv/Scripts/pytest tests/test_leakage.py -v  →  3 passed
```

(Output completo conservato negli artefatti di CI; `pytest -q` riporta esattamente "106 passed".)

## Conclusione

**Gate fallito** con margine reale (Δ ≈ −0.0004 apples-to-apples), non rumore numerico. Nessun bug di leakage o di pipeline; il negative result riflette una proprietà del modello: XGBoost Poisson con 8 feature non aggiunge valore rispetto a una logistic 2-feature in questo regime. Si procede a STEP 3 con le lezioni sopra: feature più ricche (Tier 2 form), eliminazione del bias di ES, e valutazione di un calibrator.

`step2-complete` **non taggato** (gate non superato). Report archiviato come ground-truth dello stato.
