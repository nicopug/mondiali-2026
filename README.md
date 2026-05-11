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
| 6 | Model freeze per WC2026 | ⏳ Pianificato | — |

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
mondiali download                # International results raw
mondiali build-processed         # matches.parquet con tutte le feature

# Training
mondiali train-elo               # Tier 0 baseline
mondiali train-tier1             # Poisson-XGBoost + Dixon-Coles
mondiali train-tier2             # + isotonic calibration
mondiali train-tier3             # + Transfermarkt market value (NB: gate failed)
mondiali train-tier4             # + injury impact (NB: needs injuries.csv populated)

# Data fetching
mondiali tm-scrape               # Transfermarkt national-team snapshots
mondiali tm-scrape-rosters       # Transfermarkt player-level rosters per tournament
mondiali bootstrap-injuries      # Wikipedia withdrawals → injuries.csv
```

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

reports/                     # Validation report per STEP (decision record)
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
