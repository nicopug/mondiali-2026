# STEP 4 — Validation Report: Tier 3 (Transfermarkt market values)

**Date:** 2026-05-09
**Status:** ❌ **GATE FAILED** — Tier 3 does NOT improve over Tier 2 baseline.
**Decision:** Tier 3 NOT promoted. Tier 2 remains the production baseline.

---

## 1. Data scrape summary

- **Source:** Transfermarkt national-team rosters via Wayback Machine (CDX + HTML).
- **Scope:** 72 WC-2026-relevant nations (Tier 3 scope from `compute_tier3_scope`).
- **Years:** 2014–2025 inclusive (12 years × scope).
- **Fallback chain per (nation, year):** ±2 months around 1 July → entire year → 2nd half year-1.
- **Output:** `data/raw/transfermarkt/snapshots.parquet`.

| Metric | Value |
|---|---|
| Total snapshot records | **626** |
| Unique nations covered | **66 / 72** (91.7%) |
| Mean snapshots / nation | 9.5 / 12 (79%) |
| Hard-floor violators (<2) | **0** |
| Nations not in `NATION_TM_IDS` (skipped) | 6 (Basque Country, Guernsey, Jersey, Jordan, Kosovo, Uzbekistan) |

**Nations missing despite being in scope:** the 6 above. They have no TM lookup ID; their Tier 3 features are NaN by design (acceptable per design: `add_tier3_features` falls back gracefully).

## 2. Bootstrap NATION_TM_IDS hotfix

The original bootstrap had ~63/78 fake/colliding TM IDs (e.g. ID 3473 reused across 5 African nations). Fixed by `mondiali tm-discover-ids`, which queries TM `schnellsuche` live and parses real IDs by exact-slug match.

- 62 IDs corrected, 15 unchanged (already correct), 1 manual override (DR Congo: slug also wrong, fixed by hand).
- See `src/mondiali/data/tm_discover.py` and tests `tests/test_tm_discover.py`.
- Module docstring `_BOOTSTRAP_VERIFIED = True`.

## 3. Scrape resilience improvements (this session)

The original `tm-scrape` had three weaknesses, all addressed:

1. **Cache fast-path** (`_try_cached_for_year` in `transfermarkt.py`) — if HTML for `(slug, year)` is on disk, parses it directly and skips CDX. Cuts re-run cost for already-fetched snapshots from ~5s/year to ~0.1s/year.
2. **`build_from_cache`** + CLI `mondiali tm-build-from-cache` — reconstruct `snapshots.parquet` purely from cache files (zero network). Pensato per recovery dopo interruzione, since scrape_all writes parquet only at end.
3. **`--resume` flag** on `tm-scrape` (default ON) — if `snapshots.parquet` exists, skips integralmente le nazioni già coperte (incluso re-CDX su gap years).

**Real-world payoff this session:** after a DNS blip aborted ~25 nations mid-run, recovery via `tm-build-from-cache` + `tm-scrape --resume` recuperò il lavoro senza ri-querare CDX per le 41 nazioni già fatte.

## 4. Feature integration (matches.parquet)

`add_tier3_features` (in `src/mondiali/features/tier3.py`) attaches per-side TM stats with anti-leakage guardrails:

- `pd.merge_asof(direction="backward", allow_exact_matches=False)` ensures `snapshot_date < match_date` strictly.
- Min-snapshots-per-nation hard floor: ≥2.
- Age clip: `tm_age_days ≤ 540`.
- Pre-2014: forced NaN (TM coverage starts 2014).

Output coverage on full `matches.parquet` (49 215 rows):

| Side | Coverage |
|---|---|
| `tier3_total_value_home` | 7.2% |
| `tier3_total_value_away` | 6.5% |

Low overall because matches.parquet contains all international matches since 1872, of which only 2014+ involving WC2026-qualified nations are eligible.

## 5. Training & gate (Tier 3 vs Tier 2 apples-to-apples)

Same val_gate window (2022-01-01 → 2022-12-31, 963 matches), same Dixon-Coles + Poisson XGB stack:

| Model | Train range | Train n | val_gate log-loss |
|---|---|---:|---:|
| **Tier 2 baseline** (Tier 3 cols → NaN) | 2002-2016 | 28 322 | **0.9097** |
| Tier 3 RAW | 2014-2019 | 5 783 | 0.9210 |
| Tier 3 CALIB (isotonic on 2021) | 2014-2019 | 5 783 | 1.0707 |

**Δ vs baseline:**
- RAW: **+0.0113** (worse).
- CALIB: **+0.1610** (much worse).

Brier (calib): 0.5430 → 0.5457 (also worse after calibration).

### Why Tier 3 underperforms

1. **Train set 5× smaller:** 5 783 (2014-2019) vs 28 322 (2002-2016). Less data dominates the marginal value of TM features.
2. **TM coverage low** even on the eligible window: 12.2% on train, 22.8% on gate. XGBoost sees mostly-NaN columns; signal is sparse.
3. **Calibration set (2021) ≠ gate distribution (2022).** 1115 calib points are too few to fit isotonic without overfit; the calibrator pushes already-decent raw probs to extremes.

Numbers are consistent with the **baseline-first** invariant: don't promote a richer model just because it has more features. Tier 3 has not earned promotion on this validation.

## 6. Anti-leakage tests

Full `tests/test_leakage.py` suite passes (5/5), including the Tier 3 strict-pre-match test (`test_tier3_market_value_strict_pre_match`) which asserts `snapshot_date < match_date` on every non-NaN row.

## 7. Decision & next steps

- **Task 15 marked complete.** Tier 3 model & calibrator are saved (`models/tier3/xgb_poisson.json`, `models/tier3/calibrator.json`) per spec, but **not** promoted to `models/v1_final/`.
- **Production baseline remains Tier 2** as committed in STEP 3.
- **Future investigation (out of scope for STEP 4):** raise TM coverage by adding more years pre-2014 via Wayback (some nations have 2010-2013 snapshots), or restructure the calibration set to be larger / closer to gate distribution. Not pursued now — diminishing returns vs simpler levers.

## 8. Artifacts

| Artifact | Path |
|---|---|
| Snapshot dataset | `data/raw/transfermarkt/snapshots.parquet` (626 rows) |
| HTML cache | `data/raw/transfermarkt/cache/` (~1000 files) |
| Processed matches | `data/processed/matches.parquet` (49215 rows, +6 TM cols) |
| Tier 3 model | `models/tier3/xgb_poisson.json` |
| Tier 3 calibrator | `models/tier3/calibrator.json` |
| Tier3 scope | `data/processed/tier3_scope.json` (72 nations) |
| Validation report | `reports/validation_step4.md` (this file) |
