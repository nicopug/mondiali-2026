# STEP 6 — Design: Model freeze v1_final + multi-market inference

**Date:** 2026-05-11
**Branch:** `feat/step6-freeze`
**Target completion:** before 2026-06-11 (CLAUDE.md §6 invariant kicks in at kickoff).

---

## 1. Objective

Chiudere la fase di training del progetto con:

1. Un **modello v1_final** congelato in `models/v1_final/`, addestrato sul massimo dei dati disponibili (fino al 2026-05-31 escluso), con manifest di provenienza.
2. **5 markets** derivati dal modello, **tutti validati** su val_gate storico (WC2022 + Euro2024):
   - 1X2 (Home / Draw / Away)
   - U/O 2.5
   - U/O 1.5
   - U/O 3.5
   - GG/NG
3. **Runtime inference** via CLI: dato `(home, away, date, neutral?)`, il modello costruisce le feature on-the-fly attingendo allo stato persistito e restituisce tutti i markets in JSON.

Il modello base è **Tier 2** (Poisson-XGBoost simmetrico + Dixon-Coles + isotonic 1X2). Tier 3 e Tier 4 non promossi negli step precedenti restano dormant nel codice.

## 2. Architecture

```
                                                ┌─── joint matrix (Poisson + DC) ───┐
                                                │                                   │
mondiali predict France Italy 2026-06-15        │  prob_1x2()           → 1/X/2     │
        │                                       │  prob_over_under(n=2.5) → U/O 2.5  │
        ▼                                       │  prob_over_under(n=1.5) → U/O 1.5  │
build_inference_row(home, away, date, neutral)  │  prob_over_under(n=3.5) → U/O 3.5  │
        │                                       │  prob_btts()          → GG/NG     │
        │ legge state                           └───────────────────────────────────┘
        ▼                                                       │
data/state/                                                     │
├── elo_state.parquet     ← Elo current per nation             │
├── form_cache.parquet    ← ultime 5 partite per nation        │
└── tm_snapshots.parquet  ← market value (già esistente, opt.) │
        ▼                                                       │
match_row (24 features symmetric)                               │
        │                                                       │
        ▼                                                       │
xgb_poisson.json (v1_final) → (λ_home, λ_away) → applied to ───┘
        │                                                       
        ▼                                                       
isotonic calibrator (1X2 only, others raw)                      
        │                                                       
        ▼                                                       
JSON output con tutti i 5 markets calibrati / raw               
```

## 3. New modules + responsibilities

| Modulo | Responsabilità |
|---|---|
| `src/mondiali/model/markets.py` | Add `prob_over_under(joint, line)`, `prob_btts(joint)`. Esistente `prob_1x2` resta intatta. |
| `src/mondiali/inference/state.py` | `save_state(matches_df, out_dir)` + `load_state(state_dir)` — persistenza Elo + form-5 cache + tm latest snapshot. |
| `src/mondiali/inference/predict.py` | `build_inference_row(home, away, date, neutral, state)` + `predict_match(state, model, calibrator, home, away, date, neutral)` → dict di markets. |
| `src/mondiali/training/freeze.py` | `freeze_v1_final(out_dir)` — refit Tier 2 + save model+calibrator+manifest. |
| `src/mondiali/training/validate_markets.py` | `validate_all_markets(model, calibrator, val_gate)` — Brier + log-loss per ognuno dei 5 markets vs baseline naive. |
| `src/mondiali/cli/main.py` | `predict`, `update-state`, `freeze-v1` commands. |

## 4. Schema v1_final/

```
models/v1_final/
├── xgb_poisson.json            # XGBoost native JSON (mai pickle, CLAUDE.md §5)
├── calibrator.json             # IsotonicCalibrator1X2 (3 isotonics serializzate)
├── rho.txt                     # Dixon-Coles ρ (single float)
├── manifest.json               # Provenance
└── markets_validation.json     # Brier + log-loss per market vs baseline naive
```

`manifest.json` schema:
```json
{
  "version": "v1.0",
  "created_at": "2026-05-31T12:00:00Z",
  "git_sha": "abc1234",
  "model": "PoissonXGBModel-symmetric-tier2",
  "n_features": 24,
  "feature_names": [...],
  "train_split": {"start": "2002-01-01", "end": "2023-12-31"},
  "val_gate_split": {"start": "2024-01-01", "end": "2024-12-31"},
  "data_sources": {
    "matches_parquet_sha": "...",
    "snapshots_parquet_sha": "..."
  },
  "hparams": {...},
  "random_state": 42
}
```

## 5. Markets validation gate

Per ogni market secondario (U/O 1.5/2.5/3.5, GG/NG):

| Metrica | Baseline naive | Soglia accettazione |
|---|---|---|
| Log-loss binario | Frequenza globale del market sul training (es. P(over 2.5) = mean(home_score + away_score > 2.5)) | Modello < baseline - 0.01 |
| Brier | Idem | Modello < baseline - 0.005 |

Soglie **morbide**: anche se un market fallisce il gate, lo esponiamo comunque ma marcato `validated=false` nel manifest. Decisione conscia: meglio dare al utente la probabilità con warning che nasconderla.

1X2 ricarica il risultato del gate Tier 2 esistente (no re-validation, fonte di verità: `reports/validation_step3.md`).

## 6. State persistence

**Problema:** per predire una partita futura serve ricostruire lo stato (Elo current, form-5, market value latest) alla data del match. Oggi questo è implicito nel `build_processed_matches` batch.

**Soluzione minimal:**

`data/state/elo_state.parquet`:
```
nation: str
elo: float
last_match_date: date
```

`data/state/form_cache.parquet`:
```
nation: str
match_date: date           # uno dei N più recenti
home_or_away: str
score_for: int
score_against: int
opponent_elo: float
competition_importance: float
```

Una nazione ha ≥5 righe (le ultime 5 partite alla data corrente). `build_inference_row` calcola form-5/gd-5/avg-opp-elo-5 al volo da queste righe filtrate per `date < match_date`.

`mondiali update-state` rigenera questi due parquet processando l'intera storia. Veloce (~5s). Da chiamare ogni volta che `matches.parquet` viene aggiornato (es. dopo amichevoli pre-WC).

Mercato TM: usiamo direttamente `data/raw/transfermarkt/snapshots.parquet` con `merge_asof(direction='backward')` come fa `add_tier3_features` già — no nuovo state.

## 7. CLI design

```bash
# Aggiorna state derivato (Elo + form cache)
mondiali update-state

# Refit + freeze v1_final
mondiali freeze-v1 --train-end 2023-12-31 --val-gate-start 2024-01-01 --val-gate-end 2024-12-31

# Predict singola partita
mondiali predict France Italy 2026-06-15 --neutral
# → JSON output:
# {
#   "match": {"home": "France", "away": "Italy", "date": "2026-06-15", "neutral": true},
#   "model_version": "v1.0",
#   "lambda": {"home": 1.42, "away": 1.18},
#   "markets": {
#     "1x2": {"home": 0.42, "draw": 0.29, "away": 0.29, "calibrated": true},
#     "over_under_1_5": {"over": 0.71, "under": 0.29, "validated": true},
#     "over_under_2_5": {"over": 0.49, "under": 0.51, "validated": true},
#     "over_under_3_5": {"over": 0.26, "under": 0.74, "validated": true},
#     "btts": {"yes": 0.55, "no": 0.45, "validated": true}
#   }
# }

# Batch prediction da CSV
mondiali predict-batch fixtures.csv --output predictions.csv
```

## 8. Anti-leakage at inference

Il vincolo strict-pre-match resta:
- `build_inference_row` filtra `form_cache.parquet` con `match_date < target_date` (strict).
- `merge_asof(direction='backward', allow_exact_matches=False)` per TM snapshots.
- Elo state ha `last_match_date < target_date` enforced.

Se `target_date` è anteriore alla data più recente nello state, è OK (use case: backfill/replay). Se è successiva, l'inference usa lo state attuale come "stato al kickoff".

## 9. Testing strategy

Per ogni modulo nuovo, test TDD:

- **markets.py**: `prob_over_under(joint, line=2.5)` su joint matrix conosciuta → atteso valore calcolabile a mano. `prob_btts` analogo.
- **state.py**: `save_state → load_state` round-trip. Elo current dopo N partite coincide con quello di `build_processed_matches` per le ultime partite.
- **predict.py**: `predict_match` deterministico (stesso input → stesso output). Confronto: predict su una partita storica `m` con state al `m.date - 1` deve dare predizioni vicine (entro 1e-9) a quello che faceva `build_processed_matches` + `predict_lambda(m)`.
- **freeze.py**: dopo `freeze_v1_final`, ricaricare model+calibrator+manifest, verificare che predict su 100 random matches del val_gate dia lo stesso log-loss di quello salvato nel manifest.
- **validate_markets.py**: per ogni market, Brier(model) e log-loss(model) sono ≤ baseline naive (assert nei test per assicurarsi che almeno U/O 2.5 e GG/NG passino — sono i markets target).
- **Anti-leakage**: `tests/test_leakage.py::test_predict_match_strict_pre` — predict su match storico con date=match.date deve usare SOLO state < match.date, mai uguale.

## 10. Acceptance criteria

- [ ] `models/v1_final/` contiene 5 file (xgb_poisson.json, calibrator.json, rho.txt, manifest.json, markets_validation.json).
- [ ] `mondiali predict France Italy 2026-06-15 --neutral` ritorna JSON con tutti i 5 markets in <500ms.
- [ ] Tutti i markets secondari hanno `validated=true` nel manifest (Brier modello < Brier baseline naive - 0.005).
- [ ] Test suite verde, incluso 1 anti-leakage test in più.
- [ ] `reports/validation_step6.md` con metriche di tutti i markets + tabella validazione + decisioni.
- [ ] Git tag `v1.0` sul commit del freeze.
- [ ] README aggiornato con esempio `mondiali predict`.

## 11. Out of scope

- Tier 3/4 promotion (chiuse negli step precedenti).
- Modelli non-Poisson (RNN, transformer, ecc.).
- API HTTP / web UI (CLI only).
- Aggiornamento automatico dello state da fonti esterne (futuro lavoro).
- Bookmaker market alignment / odds parsing.

## 12. Risks & mitigations

| Rischio | Mitigazione |
|---|---|
| Markets secondari sub-ottimali (modello ottimizza gol, non U/O) | Lo gestiamo nel manifest: `validated=true/false` esplicito. Anche se "false", esposti con warning, mai nascosti. |
| State drift: form_cache obsoleto al kickoff | `update-state` deve essere rifatto dopo ogni amichevole. Documentato in README. Default safe: il CLI `predict` warn-a se state più vecchio di 30 giorni. |
| Refit con date più recenti dà numeri diversi da quelli di STEP 3 | Atteso: più dati ⇒ stime migliori. Manifest registra esattamente quale split. Il log-loss di val_gate è la nuova fonte di verità v1.0. |
| ρ Dixon-Coles ri-stimato cambia | Salvato esplicitamente in `rho.txt`, immutabile dopo freeze. |
| User passa nation name sconosciuto a `predict` | CLI fa lookup case-insensitive su `data/state/elo_state.parquet`, errore esplicito se non trovato, suggerisce nomi vicini (Levenshtein). |
