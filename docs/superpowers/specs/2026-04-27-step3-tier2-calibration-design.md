# STEP 3 — Tier 2 form features + isotonic calibration (design)

**Data**: 2026-04-27
**Author**: Nicolò + Claude (brainstorm)
**Stato**: design approvato, plan da generare
**Predecessore**: `reports/validation_step2.md` (gate STEP 2 fallito Δ ≈ −0.0004 vs Elo logistic)

---

## 1. Contesto e motivazione

STEP 2 ha consegnato un Tier 1 (XGBoost Poisson + Dixon-Coles, 8 feature) che **non batte** il baseline Elo logistic (2 feature) sullo stesso val set 2019–2022 filtrato per `days_rest` (n=3209): Tier 1 perde di Δ ≈ −0.0004. Diagnostic in `scripts/diagnose_tier1.py` ha mostrato:

1. Le 3 nuove feature di Tier 1 (`competition_importance`, `days_rest_{home,away}`) **hanno** segnale: drop-le peggiora il log-loss di +0.0034.
2. L'early stopping non è il bottleneck (`rounds=20` e `rounds=50` atterrano sullo stesso `best_iteration=142`).
3. Il problema è di **calibration**, non di feature. La logistic 2-feature è meglio calibrata sull'asse `elo_diff` di un XGBoost con 8 feature che trova interazioni non-lineari spurie.
4. Il numero `LOGLOSS_TIER1=0.8528` è **ottimisticamente biased** perché ES gira sullo stesso val set della metrica finale.

STEP 3 attacca questi tre punti contemporaneamente:

- **Più segnale**: Tier 2 form features (rolling 5-match) per ridurre la dipendenza dal solo `elo_diff`.
- **ES carve-out**: separare il set per early stopping da quello per la metrica → eliminazione del bias ottimistico.
- **Isotonic calibration**: post-hoc, riallinea la curva delle probabilità raw verso le frequenze osservate. Spec §6.3 la indica come obbligatoria.

Optuna (search hparams) è esplicitamente **fuori scope** di STEP 3 e slittato a STEP 4.

## 2. Decisioni di scope

| Asse | Scelta | Razionale |
|---|---|---|
| Pacchetto | Tier 2 + ES carve-out + isotonic calibration | Pacchetto completo come spec §STEP 3, in un'unica iterazione |
| Tier 2 features | Rolling N=5: `form_5, gd_5, goals_scored_5, goals_conceded_5, avg_opp_elo_5` (×2 prospettive = 10 nuove) | Aggiunge `avg_opp_elo_5` rispetto allo spec puro per catturare qualità avversari (5W contro Gibilterra ≠ 5W contro Brasile) |
| Data split | Train 2002–2016 / Val_ES 2017 / Val_calib 2018 / Val_gate 2019–2022 | Doppio carve-out separato; pulizia massima a costo di ~1k match di training |
| Friendly | Status quo (inclusi nel training, distinguono via `competition_importance`) | Diagnostic STEP 2 non ha indicato i friendly come problema; mantieni dataset stabile |
| Gate | Doppio: soft (`< LOGLOSS_ELO`) + hard (`≤ LOGLOSS_ELO − 0.003`) | Pragmatico per ship anche con margine sottile; consistent con realtà di un dataset dominato da Elo |
| Optuna | **Fuori scope** | STEP 4 dedicato |
| `competition_importance` | Mantenuto in STEP 3 | Decisione di drop rimandata a STEP 4 dopo aver visto SHAP con Tier 2 |

## 3. Architettura componenti

| File | Stato | Responsabilità |
|---|---|---|
| `src/mondiali/features/tier2.py` | NEW | Rolling builder Tier 2 (10 feature). Funzione pura `build_tier2(matches, n=5)`. |
| `src/mondiali/features/__init__.py` | MOD | Re-export `build_tier2`. |
| `src/mondiali/data/build_processed.py` | MOD | Integra `build_tier2` nel pipeline `matches.parquet`, dopo Elo + Tier 1. |
| `src/mondiali/model/poisson_xgb.py` | MOD | Estensione `SYMMETRIC_FEATURES` (8 → 18). `build_symmetric_rows` legge le nuove colonne. |
| `src/mondiali/model/calibration.py` | NEW | `IsotonicCalibrator1X2`: fit/predict/save/load JSON-native. |
| `src/mondiali/training/train.py` | MOD | Refactor `train_tier1_pipeline` → `train_tier2_pipeline` con 4-way split. |
| `src/mondiali/training/evaluate.py` | MOD | Aggiunta `brier_score_1x2`. |
| `src/mondiali/cli/main.py` | MOD | `train-tier2` command. |
| `tests/test_tier2.py` | NEW | Tier 2 builder + leakage. |
| `tests/test_calibration.py` | NEW | IsotonicCalibrator1X2 + Brier monotonia + JSON round-trip. |
| `tests/test_train_tier2.py` | NEW | Smoke test pipeline + fast unit tests. |
| `tests/test_leakage.py` | MOD | Estensione: nuove feature Tier 2 strettamente anteriori a `match_date`. |
| `tests/test_evaluate.py` | MOD | +2 test per `brier_score_1x2`. |
| `reports/validation_step3.md` | NEW | Report finale con LOGLOSS_TIER2_CALIB, Brier prima/dopo, soft/hard gate. |

### Inference flow (post-STEP 3)

Per un nuovo match con feature complete (Elo + Tier 1 + Tier 2):

```
features → PoissonXGBModel.predict_lambda → (λ_home, λ_away)
        → joint_matrix(λ_home, λ_away)
        → dixon_coles_correct(joint, λ_home, λ_away, rho)
        → prob_1x2(joint)                    # probs raw shape (3,)
        → IsotonicCalibrator1X2.predict       # probs calibrate, somma 1
```

Sia `model.json` (XGBoost) sia `calibrator.json` (3 isotonic + thresholds) devono essere caricati per fare inference. `rho` salvato in `metadata.json` accanto al modello.

`model/dixon_coles.py` e `model/markets.py` invariati. CLI esistente `train-tier1` rimane funzionante per regression.

## 4. Tier 2 feature builder

### API

```python
TIER2_COLUMNS: list[str] = [
    "home_form_5", "away_form_5",
    "home_gd_5", "away_gd_5",
    "home_goals_scored_5", "away_goals_scored_5",
    "home_goals_conceded_5", "away_goals_conceded_5",
    "home_avg_opp_elo_5", "away_avg_opp_elo_5",
]

def build_tier2(matches: pd.DataFrame, *, n: int = 5) -> pd.DataFrame:
    """Aggiunge le 10 colonne TIER2_COLUMNS al DataFrame matches.

    Per ogni team in ogni match, considera gli ULTIMI `n` match di quel team
    strettamente anteriori a match_date (qualsiasi tipo di competizione,
    qualsiasi ruolo home/away).
    """
```

### Definizione delle 5 metriche

Calcolate per team T riferito al match M, finestra ultimi N=5 match (friendly inclusi, decisione §2):

| Feature | Definizione |
|---|---|
| `*_form_5` | Σ punti negli ultimi 5 (W=3, D=1, L=0). Min 0, max 15. |
| `*_gd_5` | Σ goal difference (segnati − subiti) negli ultimi 5. Signed. |
| `*_goals_scored_5` | Media gol segnati negli ultimi 5. |
| `*_goals_conceded_5` | Media gol subiti negli ultimi 5. |
| `*_avg_opp_elo_5` | Media `opponent_elo_before` negli ultimi 5 match. |

Le 5 metriche sono ortogonali nei segnali: form (W/D/L outcome), GD (margine), GF/GA separate (stile attacco vs difesa), avg_opp_elo (qualità avversari, contestualizza form).

### Edge cases

- **0 match precedenti** → tutte e 5 le feature `NaN`. XGBoost gestisce nativamente.
- **1 ≤ k < 5 match precedenti** → calcola sui k disponibili. `form_5` è la *somma* punti, non una media: un team con 2W in 2 match precedenti ha `form_5=6`, non scalato. `goals_scored_5` è già una media → robusto a k variabile. Nessuna colonna debug aggiuntiva (YAGNI).

### Anti-leakage

- Funzione assume `matches` ordinato per data e con colonne standard.
- Per ogni team T e match M con data D: filtra match precedenti `< D` (strict, no equality). Prende gli ultimi N.
- Test in `test_leakage.py` esteso: per ogni feature Tier 2, mock un match futuro post-D e verifica che la feature non cambi.

### Implementazione

- **Single pass**: costruisci una "long-form" view `(date, team, role, points, gf, ga, opp_elo)` con 2 righe per match (home-perspective + away-perspective), ordina per `(team, date)`, applica `groupby('team').rolling(window=N, closed='left')` per ottenere le 5 metriche aggregate. `closed='left'` garantisce strict-anteriority.
- Re-merge sul DataFrame originale via `(home_team, date)` e `(away_team, date)`.
- Complessità O(M log M) per il sort + O(M) per rolling.

### Test (≥6 in `test_tier2.py`)

1. Output ha le 10 colonne attese, no `NaN` da bug (solo da k<5 reali).
2. `home_form_5` su un team con storia controllata (sintetico): valore atteso esatto.
3. `home_avg_opp_elo_5`: dato un team che ha giocato contro 5 avversari di Elo noti, media corretta.
4. Squadra al primo match: tutte e 5 le feature `NaN`.
5. Squadra al 3° match: feature calcolate sui 2 precedenti (k=2), non scalate a 5.
6. Test simmetria: scambio home/away nel match → home_*_5 e away_*_5 si scambiano coerentemente.

## 5. Data split refactor

### Quattro fette temporali

Tutte filtrate da `dropna(subset=["days_rest_home", "days_rest_away"])` (eredità STEP 2):

| Set | Range | Stima n | Uso |
|---|---|---|---|
| Train | 2002-01-01 → 2016-12-31 | ~13'500 | Fit XGBoost |
| Val_ES | 2017-01-01 → 2017-12-31 | ~900 | `eval_set` per `early_stopping_rounds` |
| Val_calib | 2018-01-01 → 2018-12-31 | ~1'100 | Fit `IsotonicCalibrator1X2` |
| Val_gate | 2019-01-01 → 2022-06-30 | 3'209 | Misura finale, gate decision |

**Invariante**: 4 set mutualmente esclusivi e ordinati temporalmente. Val_gate non visto durante training né calibration. ES non vede mai Val_calib né Val_gate.

### Signature pipeline

```python
def train_tier2_pipeline(
    parquet_path: Path,
    *,
    train_start: str = "2002-01-01",
    train_end: str = "2016-12-31",
    val_es_start: str = "2017-01-01",
    val_es_end: str = "2017-12-31",
    val_calib_start: str = "2018-01-01",
    val_calib_end: str = "2018-12-31",
    val_gate_start: str = "2019-01-01",
    val_gate_end: str = "2022-06-30",
    early_stopping_rounds: int = 50,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ...
```

### Flusso

1. Carica parquet, filtra `dropna(days_rest_*)`, taglia 4 fette per date.
2. `model = PoissonXGBModel().fit(train, early_stopping_val=val_es, early_stopping_rounds=50)` — bias di STEP 2 **eliminato** (val_es ≠ val_gate).
3. Stima `rho` MLE su train (no leakage; come Tier 1).
4. Inference su val_calib + val_gate per ottenere probabilità raw 1X2 (joint matrix → DC correct → 1X2).
5. `calibrator = IsotonicCalibrator1X2().fit(probs_val_calib, outcomes_val_calib)`.
6. `probs_val_gate_calibrated = calibrator.predict(probs_val_gate)`.
7. Calcola `LOGLOSS_TIER2_RAW`, `LOGLOSS_TIER2_CALIB`, `BRIER_BEFORE/AFTER`, sanity France vs San Marino.
8. Return: `{model, rho, calibrator, val_log_loss_raw, val_log_loss_calib, brier_before, brier_after, n_train, n_val_es, n_val_calib, n_val_gate}`.

### Comparabilità con STEP 2

- `LOGLOSS_TIER1_CLEAN` (per debug trail del report): rieseguibile via `train_tier1_pipeline` con `early_stopping_val=val_es` invece di `val_gate`. Pipeline esistente accetta già il parametro — basta una invocation diversa nello script di report (no modifiche al codice di STEP 2).
- `LOGLOSS_ELO` baseline: ricalcolato sullo **stesso val_gate** filtrato per apples-to-apples.

### Test in `test_train_tier2.py`

- **Smoke test slow** (1): pipeline completa su parquet reale, asserisce `0.84 ≤ val_log_loss_calib ≤ 0.90` (range largo, non sul gate).
- **Fast unit** (2): split correttezza (n totali == sum di 4 fette su range globale; nessuna sovrapposizione di date).

## 6. Isotonic calibrator

### Razionale

Spec §6.3: "calibrazione obbligatoria, non opzionale". XGBoost Poisson + Dixon-Coles produce probabilità 1X2 con scala reale ma curva non lineare rispetto agli outcome osservati (sintomo della diagnostica STEP 2: la logistic batte XGBoost esattamente per il vincolo lineare). La isotonic fitta una mappatura monotona piecewise per classe, rinormalizzata.

### API

```python
class IsotonicCalibrator1X2:
    """Tre isotonic regressions indipendenti (1, X, 2) + rinormalizzazione."""

    def __init__(self) -> None:
        self.iso_home_: IsotonicRegression | None = None
        self.iso_draw_: IsotonicRegression | None = None
        self.iso_away_: IsotonicRegression | None = None

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> IsotonicCalibrator1X2:
        """probs shape (n, 3), outcomes shape (n,) con valori 0/1/2."""

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """Applica i 3 isotonic e rinormalizza ogni riga a somma 1."""

    def save(self, path: Path) -> None:
        """JSON-native (no pickle): X_thresholds + y_thresholds dei 3 isotonic."""

    @classmethod
    def load(cls, path: Path) -> IsotonicCalibrator1X2:
        ...
```

### Dettagli

- **Fit**: per ogni classe `c ∈ {0, 1, 2}` fit `IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)` su `(probs[:, c], (outcomes == c).astype(float))`. Nessun smoothing aggiuntivo.
- **Predict**: applica i 3 isotonic alle rispettive colonne raw. Rinormalizza riga per riga: `out / out.sum(axis=1, keepdims=True)`. Se `sum == 0` (degenerate): fallback alla raw probabilità della riga.
- **Serializzazione JSON-native** (vincolo CLAUDE.md "no pickle"): salva `X_thresholds_, y_thresholds_, X_min_, X_max_, increasing_` dei 3 isotonic in un dict JSON. Load: ricostruisci 3 `IsotonicRegression()` settando questi attributi direttamente. Round-trip testato al bit.

### Test (≥6 in `test_calibration.py`)

1. Fit + predict shape `(n, 3)`, righe sommano a 1 ± 1e-10.
2. **Brier dopo ≤ Brier prima** sulla stessa split (toy dataset sintetico per evitare dipendenza parquet).
3. Idempotenza: predict di probs già perfettamente calibrate è ≈ identità.
4. Edge: row con somma raw vicina a 0 → fallback non solleva.
5. Round-trip JSON: save → load → predict identico al bit.
6. Outcome bilanciato sintetico (50% home win, 25% draw, 25% away win con probs uniformi 1/3) → predict converge alle base rates osservate.

### Diagnostic addizionale

Nello script report di STEP 12-equivalente:
- `brier_score_1x2(probs, outcomes) = mean(sum_c (probs[i,c] - 1[outcomes[i]==c])^2)` su val_gate, prima e dopo calibration.
- (Opzionale) reliability diagram salvato come PNG in `reports/figures/reliability_step3.png` se matplotlib è già dipendenza; altrimenti deferred.

## 7. Inference pipeline + CLI + Reports

### Estensione `PoissonXGBModel`

```python
SYMMETRIC_FEATURES: list[str] = [
    # esistenti (8)
    "team_elo", "opponent_elo", "elo_diff_signed", "is_home", "is_neutral",
    "competition_importance", "team_days_rest", "opponent_days_rest",
    # NEW Tier 2 (10)
    "team_form_5", "opponent_form_5",
    "team_gd_5", "opponent_gd_5",
    "team_goals_scored_5", "opponent_goals_scored_5",
    "team_goals_conceded_5", "opponent_goals_conceded_5",
    "team_avg_opp_elo_5", "opponent_avg_opp_elo_5",
]
```

`build_symmetric_rows` esteso con 10 colonne aggiuntive lette da `matches`, applicando la simmetria home/away come per le esistenti. `DEFAULT_PARAMS` invariati. Modello salvato in JSON contiene `feature_names` aggiornata → `load` di un Tier 1 vecchio fallisce (intenzionale).

### CLI: `mondiali train-tier2`

```bash
mondiali train-tier2 \
  --train-start 2002-01-01 --train-end 2016-12-31 \
  --val-es-start 2017-01-01 --val-es-end 2017-12-31 \
  --val-calib-start 2018-01-01 --val-calib-end 2018-12-31 \
  --val-gate-start 2019-01-01 --val-gate-end 2022-06-30 \
  --save-model models/tier2_v1.json \
  --save-calibrator models/tier2_calibrator_v1.json
```

Default puntano ai range sopra (invocazione no-arg funziona). Stampa: `n` per ogni split, `LOGLOSS_TIER2_RAW`, `LOGLOSS_TIER2_CALIB`, `BRIER_BEFORE/AFTER`, `RHO`, `LOGLOSS_ELO` rieseguito su val_gate per soft/hard gate verdict.

### Report `reports/validation_step3.md`

Sezioni:
- **Dataset & Split**: 4 fette con conteggi reali.
- **Tier 1 baseline ricalcolato senza bias** (debug trail): `LOGLOSS_TIER1_CLEAN` con ES su val_es, val_gate=stesso. Atteso: peggiore di 0.8528.
- **Tier 2 raw**: log-loss, ρ, λ_h_mean, λ_a_mean, feature importance gain — ci aspettiamo che le 10 nuove abbiano gain non trascurabile.
- **Calibration**: Brier prima/dopo, log-loss prima/dopo. Reliability diagram opzionale.
- **Gate**:
  - Soft: `LOGLOSS_TIER2_CALIB < LOGLOSS_ELO` → ✓ ship.
  - Hard: `LOGLOSS_TIER2_CALIB ≤ LOGLOSS_ELO − 0.003` → ✓ continue improving in STEP 4.
  - Doppio fail → debug trail (SHAP delle Tier 2, ablation, considerare optuna anticipato).
- **Sanity**: France vs San Marino con Tier 2 sintetiche (form_5=15 per FRA, 0 per SMR; avg_opp_elo realistic). P(France) > 0.85.
- **Lezioni** + **Decisioni open per STEP 4** (Optuna scope, drop di `competition_importance` se SHAP < 0.02, stacker logistic vs solo isotonic).

### Test count atteso

| Set | Test |
|---|---|
| STEP 2 baseline | 106 |
| `test_tier2.py` | +6 |
| `test_calibration.py` | +6 |
| `test_train_tier2.py` | +3 (1 slow + 2 fast) |
| `test_leakage.py` extension | +1 |
| `test_evaluate.py` Brier | +2 |
| **Totale STEP 3** | **~124** |

### Workflow di chiusura

- **Soft gate pass**: tag `step3-complete`, archivia modello in `models/tier2_v1/{model.json, calibrator.json, metadata.json}`. Apri sessione STEP 4 (Optuna) con baseline `LOGLOSS_TIER2_CALIB`.
- **Soft gate fail**: NO tag, report archiviato come negative result, sessione di debug dedicata (replay diagnostic come STEP 2: SHAP, ablation Tier 2, considerare drop XGBoost in favore di logistic + Tier 2 features).
- **Hard gate fail ma soft pass**: tag `step3-complete-soft`, report documenta margine sottile, STEP 4 punta esplicitamente a chiudere il gap residuo.

Tempo stimato (analogo a STEP 2 con 12 task): ~10–12h distribuite, ~10 task discreti nel writing-plans di output.

## 8. Invarianti e vincoli ereditati da CLAUDE.md

1. Mai split random — solo temporale ✓ (4 fette ordinate per data).
2. Ogni feature strettamente anteriore alla `match_date` ✓ (`closed='left'` rolling, test esteso in `test_leakage.py`).
3. Ogni training scrive un report in `reports/` ✓ (`validation_step3.md`).
4. `random_state=42` ovunque ✓ (`PoissonXGBModel` invariato eredita da `mondiali.config:RANDOM_STATE`).
5. Modelli salvati JSON nativo, mai pickle ✓ (XGBoost `.save_model`, IsotonicCalibrator JSON con thresholds).
6. Dal 11 giugno 2026 al 19 luglio 2026: zero modifiche ✓ (STEP 3 completa prima).

## 9. Questioni open (da risolvere in STEP 4)

- **Optuna scope**: 50 trial sul Tier 2 model, search space spec §6.4. Search su `train_inner` con CV su val_es-equivalent? Da decidere in STEP 4.
- **competition_importance drop**: se in STEP 3 SHAP rimane < 0.02 anche con Tier 2 in input, valutare drop in STEP 4.
- **Stacker logistic vs solo isotonic**: alternativa di calibration architectur (Platt multivariate) se isotonic non chiude il gap. Mantenuta come fallback se STEP 3 fallisce soft gate.
- **Reliability diagram**: dipendenza matplotlib da aggiungere se reso obbligatorio (ora opzionale).
