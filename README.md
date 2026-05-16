# Mondiali 2026 — Prediction System

Sistema di predizione 1/X/2 per le partite della FIFA World Cup 2026, costruito con metodologia **baseline-first**: ogni feature tier viene aggiunto solo se batte misurabilmente il baseline precedente su uno split temporale di validazione.

```
Tier 0 (Elo)  →  Tier 1 (XGBoost+DC)  →  Tier 2 (calib+form)  →  Tier 3 (TM mv)  →  Tier 4 (injuries)
   STEP 1            STEP 2                  STEP 3 ✅            STEP 4 ❌           STEP 5 ⚠️
```

**Modello in produzione attuale: Tier 2** (calibrazione isotonic su Poisson-XGBoost Dixon-Coles corretto, baseline che bisogna battere per qualsiasi promotion futura).

## Status per STEP

| Step | Tier | Risultato | Report |
|---|---|---|---|
| 1 | Elo logistic baseline | ✅ Setup completo | [validation_step1.md](reports/validation_step1.md) |
| 2 | Poisson-XGBoost + Dixon-Coles | ✅ Promosso vs Elo | [validation_step2.md](reports/validation_step2.md) |
| 3 | Tier 2: isotonic calibration + form-5 | ✅ Promosso vs Tier 1 | [validation_step3.md](reports/validation_step3.md) |
| 4 | Tier 3: Transfermarkt market value | ❌ Non promosso (log-loss regression) | [validation_step4.md](reports/validation_step4.md) |
| 5 | Tier 4: top-5 injury impact | ⚠️ Non promosso (no training data) | [validation_step5.md](reports/validation_step5.md) |
| 6 | Model freeze v1_final + multi-market inference | ✅ Frozen (tag `v1.0`) | [validation_step6.md](reports/validation_step6.md) |

## Invarianti non negoziabili

1. Mai split random — solo temporale (`date < gate_date`).
2. Ogni feature deve essere strettamente anteriore a `match_date` (anti-leakage strict `<`).
3. Ogni training scrive un report in `reports/`.
4. `random_state=42` ovunque.
5. XGBoost salvato in formato JSON nativo (mai pickle).
6. Dal 11 giugno 2026 (kickoff) al 19 luglio 2026 (finale): zero modifiche a modello/iperparametri/feature.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate  # Windows; source .venv/bin/activate su Unix
pip install -e ".[dev]"
```

## Comandi principali

```bash
# Pipeline dati
mondiali ingest                  # International results raw + matches.parquet con tutte le feature

# Training
mondiali train-elo               # Tier 0 baseline
mondiali train-tier1             # Poisson-XGBoost + Dixon-Coles
mondiali train-tier2             # + isotonic calibration
mondiali train-tier3             # + Transfermarkt market value (NB: gate failed)
mondiali train-tier4             # + injury impact (NB: needs injuries.csv populated)

# Freeze + inference (STEP 6)
mondiali freeze-v1               # Refit Tier 2, save models/v1_final/ artefacts
mondiali update-state            # Rebuild data/state/{elo_state,form_cache}.parquet
mondiali predict France Italy 2026-06-15 --neutral
mondiali predict-batch fixtures.csv --output predictions.csv

# Data fetching
mondiali tm-scrape               # Transfermarkt national-team snapshots
mondiali tm-scrape-rosters       # Transfermarkt player-level rosters per tournament
mondiali bootstrap-injuries      # Wikipedia withdrawals → injuries.csv
```

## Daily workflow durante WC2026

```bash
# 1. Aggiorna i risultati (esegui dopo amichevoli + dopo ogni giornata mondiale)
mondiali ingest

# 2. Ricostruisci lo stato runtime (Elo + form cache)
mondiali update-state

# 3. Predici i match della prossima giornata (CSV con colonne: home,away,date,neutral?,competition_importance?)
mondiali predict-batch fixtures_round_X.csv --output predictions_round_X.csv

# Output JSON per match singolo (con tutti i 5 markets):
mondiali predict Argentina Croatia 2026-07-10 --neutral
```

Freeze invariant: tra l'11 giugno 2026 (kickoff) e il 19 luglio 2026 (finale) zero modifiche a `models/v1_final/`, agli iperparametri, o alle 24 feature. Lo stato in `data/state/` può essere aggiornato per assorbire nuove partite.

## Output example

```json
{
  "match": {"home": "France", "away": "Italy", "date": "2026-06-15", "neutral": true},
  "model_version": "v1.0",
  "lambda": {"home": 1.567, "away": 0.951},
  "markets": {
    "1x2": {"home": 0.508, "draw": 0.272, "away": 0.220, "calibrated": false},
    "over_under_2_5": {"over": 0.461, "under": 0.539, "calibrated": false, "validated": false},
    "over_under_3_5": {"over": 0.246, "under": 0.754, "calibrated": false, "validated": true},
    "btts": {"yes": 0.495, "no": 0.505, "calibrated": false, "validated": false}
  }
}
```

`calibrated` indica se è stato applicato un calibratore post-hoc (auto-skipped al freeze se non migliora Brier su val_gate). `validated` indica se il market ha battuto la baseline naive (`brier_model < brier_baseline - 0.005`) sul val_gate 2024.

## Struttura repo

```
src/mondiali/
├── cli/main.py              # Typer CLI entry
├── config.py                # Paths + Elo K-factors + RANDOM_STATE
├── data/                    # Ingestion + scrapers (TM, Wikipedia, results)
├── features/                # Elo, tier2 (form), tier3 (TM mv), tier4 (injuries)
├── model/                   # PoissonXGBModel, Dixon-Coles, isotonic, Elo logistic
└── training/                # Pipeline per ogni tier + evaluate

docs/superpowers/
├── specs/                   # Design doc per STEP (anteriore al codice)
└── plans/                   # Implementation plan task-by-task (TDD)

models/v1_final/             # Frozen model (xgb_poisson.json, rho.txt, manifest.json,
                             # markets_validation.json, markets_calibrators/, calibrator.json se kept)
data/state/                  # Runtime state per inference (elo_state, form_cache)

reports/                     # Validation report per STEP + backtest report
scripts/backtest_tournaments.py  # Walk-forward backtest WC2022+Euro2024
tests/                       # pytest, anti-leakage + per-modulo
```

## Test

```bash
pytest                              # Suite completa
pytest tests/test_leakage.py -v     # Solo invarianti anti-leakage (critici)
ruff check src/ tests/              # Lint
```

## Metodologia: baseline-first

Ispirata a [Yoav Goldberg, "Tips for working with a corpus"](https://yoavartzi.com/) e all'invariante "make it work, make it right, make it fast — in that order". Ogni STEP:

1. **Spec** in `docs/superpowers/specs/` con design + invarianti + acceptance criteria.
2. **Plan** in `docs/superpowers/plans/` con task TDD numerati (red → green → commit per task).
3. **Implementation** task-by-task, ogni task in commit isolato.
4. **Validation report** in `reports/` con metriche, gate decision e decision record (anche per gate falliti).
5. **Promote o rigetta** sulla base del log-loss su validation gate temporalmente isolato.

Gate falliti (Tier 3, Tier 4) sono valore: documentano cosa NON funziona e prevengono regressioni future quando arriverà la tentazione di riprovare.
