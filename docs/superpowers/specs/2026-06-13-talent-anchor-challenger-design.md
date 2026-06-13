# STEP 9 — Talent-anchor challenger (Phase 1: market-value feature)

**Date:** 2026-06-13
**Status:** DESIGN (approved, pre-implementation)
**Branch:** `feat/talent-anchor-challenger`
**Frozen baseline (untouched):** `models/v1_final/` (v1.4)

---

## 1. Motivation

`v1_final` (Elo + Poisson-XGBoost + form) systematically diverges from the
betting market on teams whose **squad talent outstrips their recent results**:
it over-rates in-form mid-tier sides (Argentina, Turkey) and under-rates
big-name sides with mediocre recent form (France, England, Brazil). Concrete
symptom: a neutral Brazil vs Turkey is predicted as a coin flip (35/30/35).

Root cause: the model has **no squad-quality signal**. Elo reacts only to
results and over-weights recency. The market prices in talent (squad value),
which the model cannot see.

A previous attempt (STEP 4, Tier 3 = Transfermarkt market value as XGBoost
features) **failed the gate** (val_gate log-loss +0.0113 raw, +0.16 calibrated).
The post-mortem (`reports/validation_step4.md`) attributes the failure to:

1. **Training set shrank 5×** — adding the feature forced the train window to
   2014-2019 (5 783 matches) instead of 2002-2016 (28 322). The data loss
   dominated the marginal feature value.
2. **Low coverage** — 12% on train, 23% on gate; XGBoost saw mostly-NaN columns.
3. **Calibration overfit** on a small set.

This STEP re-attempts the squad-quality signal while **directly fixing cause #1**
and measuring the lever's value cheaply before any data investment.

## 2. Non-negotiable constraints

- **Freeze invariant (CLAUDE.md #6):** zero changes to `models/v1_final/`,
  its hyperparameters, or its 24 features between 2026-06-11 and 2026-07-19.
  This STEP builds a **challenger** in `models/challenger_talent/` on a separate
  branch. It is NOT promoted to production during the freeze.
- All other invariants hold: temporal splits only, strict pre-match features
  (`<`), report in `reports/`, `random_state=42`, XGBoost saved as native JSON.

## 3. Goal & success criteria

Build a challenger (v2 candidate) incorporating squad market value as a talent
signal, validated on the **same temporal gate** as the freeze.

Three sequential gates:

1. **STEP 0 — diagnostic (cheap, run first):** on the OOS 2025-2026 set
   (1 161 matches, never seen by any model), quantify whether the per-match
   `talent_gap` correlates with v1.4's **residual error** (e.g. signed
   probability error on the realised outcome, or λ error). If there is no
   correlation, the lever does not exist → stop here, documented. Cost ≈ zero.
2. **Promotion gate:** challenger beats v1.4 raw 1X2 log-loss on `val_gate`
   (2024) by **≥ 0.003**, with Brier no worse. Same splits as `freeze-v1`.
3. **Anti-leakage:** every talent feature is strictly pre-match.

Outcome:
- Gate passes → v2 candidate retained on branch (NOT promoted to `v1_final`
  during freeze); promotion decision deferred to post-tournament.
- Gate fails → documented like STEP 4 / STEP 5 (failed gates are valuable).

## 4. Architecture & components

### 4.1 `src/mondiali/features/talent.py` (new) — the reusable primitive

Derives talent features from the market-value columns already present in
`matches.parquet` (`home_market_value_top11`, `away_market_value_top11`, and the
`*_total` variants, produced by `add_tier3_features`).

`add_talent_features(df) -> df` adds:
- `talent_gap_top11 = home_market_value_top11 - away_market_value_top11`
- `talent_log_ratio = log1p(home_market_value_top11) - log1p(away_market_value_top11)`
- NaN-preserving: rows without market value keep `NaN` (XGBoost handles natively).

This is the **single primitive reused by both phases**: Phase 1 (feature, this
STEP) and Phase 2 (Elo anchor, future STEP). It does no scraping and no merge —
it only transforms columns that already exist, so it inherits tier3's
anti-leakage guarantee.

### 4.2 Training pipeline (new) — `train_talent_challenger`

Mirrors `train_tier2_pipeline` / `freeze_v1_final` (XGBoost Poisson +
Dixon-Coles + optional isotonic) with two differences:
- **Full train window (2002+)** — the key fix vs STEP 4. Market value is `NaN`
  for pre-2014 / uncovered rows; the model still trains on all matches.
- Talent features from `add_talent_features` appended to the Tier 2 feature set.

Output artefacts in **`models/challenger_talent/`** (never `v1_final/`).

### 4.3 Inference

Reuse the existing `predict_match` / `BatchPredictor` path pointed at
`models/challenger_talent/`. No new inference code; the challenger is a drop-in
model directory. This lets us run it in **shadow** alongside v1.4 on the same
fixtures for live comparison during the tournament.

## 5. Data flow & anti-leakage

Market value enters `matches.parquet` via the existing `add_tier3_features`:
`merge_asof(direction="backward", allow_exact_matches=False)` enforces
`snapshot_date < match_date`, with min-2-snapshots floor and age clip ≤ 540 days
(`reports/validation_step4.md` §4). `add_talent_features` only differences those
columns, so the strict-pre-match guarantee carries over. A dedicated leakage
test asserts the talent columns are NaN or strictly pre-match on every row.

## 6. Validation & decision

Same splits as `freeze-v1`: train ≤ 2023-12-31, val_calib 2023, val_gate 2024.
Compare `challenger.raw_logloss` vs `v1.4.raw_logloss` on val_gate (apples to
apples, same DC + Poisson stack). Decision rule: retain as v2 candidate if
**Δ ≤ −0.003**; reject if **Δ ≥ +0.003**; manual review on Brier otherwise.
Also report OOS 2025-2026 metrics. All written to `reports/validation_step9.md`,
including the STEP 0 diagnostic result and a decision record (even on failure).

## 7. Testing (TDD)

- `tests/test_talent_features.py`: derived columns correct; NaN preserved where
  market value missing; strict-pre-match leakage assertion.
- Smoke test of `train_talent_challenger`: full-window training produces a saved
  model and a finite val_gate log-loss.
- Reuse `training.evaluate.{log_loss_1x2, brier_score_1x2}`.

## 8. Scope boundaries (YAGNI)

In scope (Phase 1):
- Talent primitive + NaN-aware feature, full-window challenger, STEP 0
  diagnostic, gate evaluation, report.

Out of scope (deferred):
- **Phase 2 — Elo prior anchor (Approach B):** regularise team strength toward a
  market-value talent rating (`elo_adj = shrink(elo, talent_rating, w)`). Its own
  STEP and gate. The Phase 1 primitive is designed to feed it without rework.
- **Phase 1.5 — coverage boost:** only if STEP 0 shows the lever exists but
  coverage is the bottleneck (cheap carry-forward / wider age clip before any new
  scraping).
- FIFA-ranking signal, bookmaker-odds blending — explicitly rejected during
  brainstorming.

## 9. Acceptance criteria

- [ ] `add_talent_features` implemented + tested (correctness, NaN, leakage).
- [ ] STEP 0 diagnostic run on OOS 2025-2026, result recorded.
- [ ] `train_talent_challenger` trains on full window, saves to
      `models/challenger_talent/`.
- [ ] val_gate comparison vs v1.4 with explicit Δ and decision.
- [ ] `reports/validation_step9.md` written with decision record.
- [ ] `models/v1_final/` byte-unchanged (freeze respected).
- [ ] All work on `feat/talent-anchor-challenger`.
