# Istruzioni per Claude Code in questo repo

## Contesto
Progetto di predizione calcistica per il Mondiale FIFA 2026. Principio centrale: **baseline-first** (vedi `docs/superpowers/specs/2026-04-20-mondiali-prediction-design.md`).

## Invarianti non negoziabili
1. Mai split random — solo temporale.
2. Ogni feature deve essere strettamente anteriore alla `match_date`.
3. Ogni training scrive un report in `reports/`.
4. `random_state=42` ovunque.
5. XGBoost salvato in formato JSON nativo, mai pickle.
6. Dal 11 giugno 2026 (kickoff) al 19 luglio 2026 (finale): zero modifiche a modello/iperparametri/feature.

## Stack
Python 3.11+, pandas, XGBoost, pydantic, typer, pytest, ruff, mypy.

## Workflow
- Test-first. Ogni feature entra con test che la esercita.
- Commit frequenti, messaggi descrittivi in inglese.
- Non toccare `models/v1_final/` una volta congelato (STEP 6).
