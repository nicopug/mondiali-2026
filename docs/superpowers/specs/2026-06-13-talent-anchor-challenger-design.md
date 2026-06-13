# STEP 9 — Talent-anchor challenger (Phase 1: diagnostic + differential feature)

**Date:** 2026-06-13
**Status:** DESIGN (approved, pre-implementation) — revised after code finding (see §1.1)
**Branch:** `feat/talent-anchor-challenger`
**Frozen baseline (untouched):** `models/v1_final/` (v1.4)

---

## 1. Motivation

`v1_final` (Elo + Poisson-XGBoost + form) systematically diverges from the
betting market on teams whose **squad talent outstrips their recent results**:
it over-rates in-form mid-tier sides (Argentina, Turkey) and under-rates
big-name sides with mediocre recent form (France, England, Brazil). Concrete
symptom: a neutral Brazil vs Turkey is predicted as a coin flip (35/30/35).

### 1.1 Correction to the original premise (important)

The first draft assumed the model had **no** squad-quality signal and that the
fix was a full-window retrain. Inspecting the code disproved this:

- `v1_final`'s 24 features (`models/v1_final/manifest.json`) **already include
  market value**: `team/opponent_market_value_total`,
  `team/opponent_market_value_top11`, `team/opponent_tm_age_days`.
- It is **already trained on the full window 2002-2023** with NaN where market
  value is missing.

So the squad-quality signal is present but **ineffective**. The likely causes:

1. **Low coverage** (~7% of rows have market value) → the feature rarely fires;
   XGBoost mostly ignores it.
2. **Absolute encoding** — the model sees raw `team_value` and `opponent_value`,
   not a direct **differential**. Combined with sparse coverage and a strong Elo
   signal, the talent information is drowned out.

The STEP 4 ("Tier 3") failure was a *dedicated model on a restricted 2014-2019
window*; the README framing ("Tier 3 not promoted") is misleading because the
market-value columns nonetheless live in the production Tier 2 feature set.

This STEP therefore does NOT re-add market value. It (a) **diagnoses why the
existing signal underperforms**, then (b) tries the cheapest effective
refinement — a **talent differential feature** (and, if warranted, a coverage
boost). The fundamentally different mechanism — an **Elo talent anchor** — is
promoted to the primary lever for Phase 2.

## 2. Non-negotiable constraints

- **Freeze invariant (CLAUDE.md #6):** zero changes to `models/v1_final/`, its
  hyperparameters, or its 24 features between 2026-06-11 and 2026-07-19. This
  STEP builds a **challenger** in `models/challenger_talent/` on a separate
  branch, NOT promoted to production during the freeze.
- All other invariants hold: temporal splits only, strict pre-match features
  (`<`), report in `reports/`, `random_state=42`, XGBoost saved as native JSON.

## 3. Goal & success criteria

Make the squad-quality signal actually correct the recency bias, validated on
the **same temporal gate** as the freeze.

Three sequential gates:

1. **STEP 0 — diagnostic (cheap, run first).** On the OOS 2025-2026 set
   (1 161 matches, never seen in training) quantify, for the subset where market
   value is present:
   - correlation between `talent_gap_top11` and v1.4's **signed residual**
     (realised goal diff − predicted λ diff, and 1X2 probability error);
   - whether high-talent / low-Elo teams systematically beat v1.4's expectation.
   This tells us if the lever exists and whether the bottleneck is coverage,
   encoding, or Elo dominance. If no signal → stop, documented. Cost ≈ zero.
2. **Promotion gate.** The challenger beats `v1.4` raw 1X2 log-loss on
   `val_gate` (2024) by **≥ 0.003**, Brier no worse. Same splits as `freeze-v1`.
3. **Anti-leakage.** Every talent feature is strictly pre-match.

Outcome: gate passes → v2 candidate kept on branch (promotion deferred past the
freeze); gate fails → documented like STEP 4 / STEP 5 (failed gates are value).

## 4. Architecture & components

### 4.1 `src/mondiali/features/talent.py` (new) — reusable primitive

`add_talent_features(df) -> df` adds **per-match** differential columns derived
from the market-value columns already in `matches.parquet`:

- `talent_gap_top11 = home_market_value_top11 - away_market_value_top11`
- `talent_log_ratio = log1p(home_market_value_top11) - log1p(away_market_value_top11)`

NaN-preserving (rows without market value stay NaN; XGBoost handles natively).
It only transforms existing columns — no scraping, no merge — so it inherits
tier3's anti-leakage guarantee. **This primitive is reused by Phase 2** (the Elo
anchor consumes the same per-match talent magnitude).

### 4.2 Feature wiring — `include_talent` flag

Extend `build_symmetric_rows` with an `include_talent: bool = False` flag (same
pattern as the existing `include_tier4`). When on, append 2 symmetric features:
- team-perspective: `+talent_gap_top11`, `+talent_log_ratio`
- away-perspective: the sign-flipped values.

`SYMMETRIC_FEATURES_TALENT_EXTRA = ["talent_gap", "talent_log_ratio"]`. The
challenger model is `PoissonXGBModel(include_talent=True)`; v1_final's builder
path (`include_talent=False`) is byte-unchanged.

### 4.3 Training pipeline — `train_talent_challenger`

Mirrors `train_tier2_pipeline` (XGBoost Poisson + Dixon-Coles + isotonic) on the
same splits, but builds rows with `include_talent=True`. Output artefacts in
**`models/challenger_talent/`** (never `v1_final/`).

### 4.4 Inference

Reuse `predict_match` / `BatchPredictor` pointed at `models/challenger_talent/`
(drop-in model dir), enabling **shadow** comparison vs v1.4 on the same fixtures.
The symmetric builder must honor `include_talent` at inference too.

## 5. Data flow & anti-leakage

Market value enters `matches.parquet` via the existing `add_tier3_features`
(`merge_asof` backward, `snapshot_date < match_date`, age clip ≤ 540 days).
`add_talent_features` only differences those columns, so the strict-pre-match
guarantee carries over. A dedicated leakage test asserts the talent columns are
NaN or strictly pre-match on every row.

## 6. Validation & decision

Same splits as `freeze-v1`: train ≤ 2023-12-31, val_calib 2023, val_gate 2024.
Compare `challenger.raw_logloss` vs `v1.4.raw_logloss` on val_gate (same DC +
Poisson stack, identical splits — the only difference is the 2 talent features).
Decision: retain as v2 candidate if **Δ ≤ −0.003**; reject if **Δ ≥ +0.003**;
manual Brier review otherwise. Also report OOS 2025-2026. Written to
`reports/validation_step9.md` with the STEP 0 result and a decision record.

## 7. Testing (TDD)

- `tests/test_talent_features.py`: differential columns correct; NaN preserved
  where market value missing; strict-pre-match leakage assertion.
- `tests/test_poisson_xgb.py` extension: `include_talent=True` produces the
  expected extra columns with correct sign-flip on the away row.
- Smoke test of `train_talent_challenger`: trains, saves model, finite val_gate
  log-loss. Reuse `training.evaluate.{log_loss_1x2, brier_score_1x2}`.

## 8. Scope boundaries (YAGNI)

In scope (Phase 1):
- STEP 0 diagnostic on OOS.
- Talent differential primitive + `include_talent` feature wiring + challenger
  training + gate evaluation + report.

Out of scope (deferred, in priority order):
- **Phase 2 — Elo talent anchor (the primary lever).** Regularise team strength
  toward a market-value talent rating (`elo_adj = shrink(elo, talent_rating, w)`),
  attacking the recency over-reaction at its root. Its own STEP + gate. Phase 1's
  primitive is built to feed it without rework. Expected to matter more than the
  feature refinement, per §1.1.
- **Coverage boost** — only if STEP 0 shows the lever exists but coverage is the
  bottleneck (cheap carry-forward / wider age clip before any new scraping).
- FIFA-ranking signal, bookmaker-odds blending — rejected during brainstorming.

## 9. Acceptance criteria

- [ ] STEP 0 diagnostic run on OOS 2025-2026; correlation/result recorded in report.
- [ ] `add_talent_features` implemented + tested (correctness, NaN, leakage).
- [ ] `build_symmetric_rows` gains `include_talent` flag; sign-flip tested;
      `include_talent=False` path unchanged.
- [ ] `train_talent_challenger` trains on the freeze splits, saves to
      `models/challenger_talent/`.
- [ ] val_gate comparison vs v1.4 with explicit Δ and decision rule applied.
- [ ] `reports/validation_step9.md` written with decision record.
- [ ] `models/v1_final/` byte-unchanged (freeze respected).
- [ ] All work on `feat/talent-anchor-challenger`.
