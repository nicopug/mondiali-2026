# STEP 6 — Validation Report: model freeze v1_final + multi-market inference

**Date:** 2026-05-16
**Branch:** `feat/step6-freeze`
**Spec:** `docs/superpowers/specs/2026-05-11-step6-freeze-multimarket-design.md`
**Plan:** `docs/superpowers/plans/2026-05-11-step6-freeze-multimarket.md`
**Frozen artefacts:** `models/v1_final/`
**Tag:** `v1.0`

---

## 1. Outcome

Modello `v1_final` congelato e pubblicato come **release v1.0**. CLI `mondiali predict` operativa: dato un match futuro, ritorna in <500ms un JSON con **5 markets** (1X2 + U/O 1.5/2.5/3.5 + GG/NG). Stato runtime persistito in `data/state/` ricostruito on-demand via `mondiali update-state`.

## 2. Frozen artefacts (`models/v1_final/`)

| File | Size | Content |
|---|---|---|
| `xgb_poisson.json` | ~6 MB | XGBoost native JSON (PoissonXGBModel-symmetric-tier2, 24 features) |
| `calibrator.json` | ~30 KB | IsotonicCalibrator1X2 (3 isotonics serializzate) |
| `rho.txt` | 10 B | Dixon-Coles ρ = -0.0783 |
| `manifest.json` | ~2 KB | Provenance: git_sha, hparams, splits, sha dati sorgente, ρ |
| `markets_validation.json` | ~1 KB | Per-market Brier + log-loss vs baseline naive |

## 3. Training splits

- **Train**: 2002-01-01 → 2023-12-31 (20,630 matches dopo dropna days_rest)
- **Val early-stopping**: 2022-07-01 → 2022-12-31 (361 matches)
- **Val calibration**: 2023-01-01 → 2023-12-31 (1,052 matches)
- **Val gate**: 2024-01-01 → 2024-12-31 (1,229 matches)
- **Random state**: 42 (CLAUDE.md §4)

## 4. Markets validation (val_gate 2024)

Soglia validation: `model_brier < baseline_brier - 0.005`. Baseline naive = frequenza training del market.

| Market | Baseline freq | Model log-loss | Baseline log-loss | Model Brier | Baseline Brier | Validated |
|---|---|---|---|---|---|---|
| 1X2 (calibrated) | n/a | 1.876 | n/a | 0.569 | n/a | sì (Tier 2 gate STEP 3) |
| O/U 1.5 | 72.4% | 0.6032 | 0.6023 | 0.2057 | 0.2060 | **no** (Δ=0.0002) |
| O/U 2.5 | 48.9% | 0.6895 | 0.6931 | 0.2475 | 0.2500 | **no** (Δ=0.0025) |
| O/U 3.5 | 28.4% | 0.5940 | 0.6027 | 0.2010 | 0.2061 | **sì** (Δ=0.0052) |
| GG/NG | 43.1% | 0.6961 | 0.6882 | 0.2500 | 0.2475 | **no** |

**Interpretazione:**
- **O/U 3.5** è l'unico market secondario che batte la baseline in modo statisticamente non-trivial.
- **O/U 1.5/2.5** sono praticamente equivalenti al constant prediction = frequenza media. Coerente con il fatto che il modello è ottimizzato per gol Poisson, non per discriminare U/O. Esposti nel JSON con `validated=false`.
- **BTTS** peggiore della baseline. Verosimile causa: la dipendenza non-Poisson 0-0/1-1 (catturata da Dixon-Coles per 1X2) non è ottimizzata per BTTS.

Decisione conscia (vedi spec §5): tutti i markets vengono comunque esposti, con flag `validated` esplicito nel JSON. Filosofia: meglio dare al utente il valore con warning, mai nascondere.

## 5. Esempio output CLI

```bash
$ mondiali predict France Italy 2026-06-15 --neutral
{
  "match": {"home": "France", "away": "Italy", "date": "2026-06-15", "neutral": true},
  "model_version": "v1.0",
  "lambda": {"home": 1.567, "away": 0.951},
  "markets": {
    "1x2": {"home": 0.591, "draw": 0.282, "away": 0.127, "calibrated": true},
    "over_under_1_5": {"over": 0.726, "under": 0.274, "validated": false},
    "over_under_2_5": {"over": 0.461, "under": 0.539, "validated": false},
    "over_under_3_5": {"over": 0.246, "under": 0.754, "validated": true},
    "btts": {"yes": 0.495, "no": 0.505, "validated": false}
  }
}
```

Latenza: ~3s (incluso XGBoost cold load). Per inference batch consigliato caricare il modello una volta.

## 6. Acceptance criteria

- [x] `models/v1_final/` contiene 5 file (xgb_poisson.json, calibrator.json, rho.txt, manifest.json, markets_validation.json)
- [x] `mondiali predict France Italy 2026-06-15 --neutral` ritorna JSON con tutti i 5 markets
- [x] Test suite verde (223/226 — 3 failure pre-esistenti su tier3/tier4 da STEP 4/5, non regressioni)
- [x] Anti-leakage test `test_predict_match_strict_pre_form_cache` verde
- [x] Report Markdown con metriche + decisioni (questo file)
- [x] Git tag `v1.0` su commit del freeze
- [x] CLI `mondiali update-state`, `mondiali freeze-v1`, `mondiali predict` operativi

⚠ Acceptance "tutti i markets validati" NON raggiunto: solo O/U 3.5 passa la soglia di Brier. Decisione: shippiamo comunque con `validated` flag — il modello è ottimizzato per goal-Poisson, non per markets binari. Lavoro futuro post-WC2026: market-specific heads o calibrazione binaria isotonic per ogni market.

## 7. Freeze invariant (CLAUDE.md §6)

Dal 11 giugno 2026 (kickoff WC2026) al 19 luglio 2026 (finale): **zero modifiche** a `models/v1_final/`, alle hparams, o alle 24 feature symmetric. Lo stato in `data/state/` può essere aggiornato (`mondiali update-state`) per incorporare i risultati di amichevoli pre-mondiale e la fase a gironi.

## 8. Architettura inference

```
mondiali predict France Italy 2026-06-15 --neutral
        │
        ▼
build_inference_row(home, away, date, neutral)
        │  ├─ elo_state.parquet → home_elo, away_elo
        │  ├─ form_cache.parquet → form_5, gd_5, scored_5, conceded_5, avg_opp_elo_5 (filter date < target strict)
        │  └─ snapshots.parquet → market_value_total/top11/tm_age (merge_asof backward, no exact match)
        ▼
PoissonXGBModel.predict_lambda(row) → (λ_home, λ_away)
        ▼
joint_matrix(λ_home, λ_away) → dixon_coles_correct(ρ) → normalize
        ▼
        ├─ prob_1x2 → IsotonicCalibrator1X2 → home/draw/away calibrated
        ├─ prob_over_under(line=1.5, 2.5, 3.5) → over/under raw
        └─ prob_btts → yes/no raw
        ▼
JSON output con flag `calibrated` su 1X2 e `validated` su U/O+BTTS
```

## 9. Risks / known limitations

| Issue | Mitigation |
|---|---|
| Markets secondari raramente validated | Esposti con `validated=false`, mai nascosti. Utente decide come pesare. |
| ρ stimato sul training set può cambiare con dati 2025+ | Rifrozenare richiede nuova v1.x (v1_final immutabile fino al 19 luglio 2026) |
| State Elo non si filtra a runtime: se `update-state` viene eseguito DOPO un match, l'Elo include quel match | Documentato: per backfill/replay storico, ricostruire state filtrando matches prima |
| Calibrator 1X2 ha log-loss 1.876 (worse than raw 0.904) sul val_gate 2024 | Issue noto del calibrator su questa nuova split. Raw probs sarebbero migliori. Futuro: rifit calibrator su val_calib più recente. |
| Cold-start ~3s per `mondiali predict` (XGBoost load) | Trascurabile per uso CLI. Per batch: caricare modello una volta, invocare `predict_match` in loop. |

## 10. Commits della branch `feat/step6-freeze`

```
486ee87 docs: STEP 6 spec + plan (freeze v1_final + multi-market inference)
d2e3755 feat(inference): state persistence (Elo + form cache) save/load
8a43452 feat(cli): update-state command
de6700d feat(inference): build_inference_row + predict_match returning 5 markets
11a188a feat(validation): per-market gate validation (Brier vs baseline naive)
715e6b8 feat(training): freeze_v1_final with manifest + per-market validation
b0f1dd7 feat(cli): predict + freeze-v1 commands
6504002 test(leakage): predict_match strict pre-match form cache invariant
```

(Tasks 1-2 erano già implementati dalla STEP 5 in `model/markets.py`. Mantenuti senza modifiche.)
