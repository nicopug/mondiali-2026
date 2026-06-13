# Talent-anchor challenger (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether a squad-talent *differential* feature, plus a diagnostic of why the existing market-value signal underperforms, lets a challenger model beat the frozen v1.4 on the temporal gate — without touching `models/v1_final/`.

**Architecture:** Reuse the existing Tier 2 stack (symmetric Poisson-XGBoost + Dixon-Coles + isotonic). Add a per-match talent differential primitive (`add_talent_features`) and an `include_talent` flag on the symmetric row builder. Train a challenger with the flag on, on the same splits as the freeze, and compare raw val_gate log-loss vs v1.4. A STEP 0 diagnostic on OOS gates the effort.

**Tech Stack:** Python 3.11, pandas, numpy, XGBoost (native JSON), pytest, ruff. All runs via `.venv\Scripts\python.exe`.

**Branch:** `feat/talent-anchor-challenger` (already created). Never write to `models/v1_final/`.

---

## File Structure

- Create: `src/mondiali/features/talent.py` — `add_talent_features(df)` primitive (per-match differential columns). Reused by Phase 2.
- Modify: `src/mondiali/model/poisson_xgb.py` — add `SYMMETRIC_FEATURES_TALENT_EXTRA` + `include_talent` flag on `build_symmetric_rows` and `PoissonXGBModel`.
- Modify: `src/mondiali/training/train.py` — add `train_talent_challenger(...)` (mirrors `train_tier2_pipeline` with `include_talent=True`).
- Create: `scripts/diagnose_talent_lever.py` — STEP 0 diagnostic on OOS.
- Create: `scripts/train_talent_challenger.py` — train challenger, compare vs v1.4, write report.
- Create: `tests/test_talent_features.py` — primitive correctness + NaN + leakage.
- Modify: `tests/test_poisson_xgb.py` — `include_talent` sign-flip test.
- Output: `models/challenger_talent/`, `reports/validation_step9.md`, `reports/talent_diagnostic.json`.

---

### Task 1: Talent differential primitive

**Files:**
- Create: `src/mondiali/features/talent.py`
- Test: `tests/test_talent_features.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_talent_features.py
from __future__ import annotations

import numpy as np
import pandas as pd

from mondiali.features.talent import TALENT_COLUMNS, add_talent_features


def test_talent_gap_and_log_ratio_computed():
    df = pd.DataFrame({
        "home_market_value_top11": [100.0, 50.0],
        "away_market_value_top11": [40.0, 50.0],
    })
    out = add_talent_features(df)
    assert TALENT_COLUMNS == ["talent_gap_top11", "talent_log_ratio"]
    assert np.isclose(out["talent_gap_top11"].iloc[0], 60.0)
    assert np.isclose(out["talent_gap_top11"].iloc[1], 0.0)
    # log1p(100) - log1p(40) > 0 ; log1p(50)-log1p(50) == 0
    assert out["talent_log_ratio"].iloc[0] > 0
    assert np.isclose(out["talent_log_ratio"].iloc[1], 0.0)


def test_talent_features_preserve_nan_when_value_missing():
    df = pd.DataFrame({
        "home_market_value_top11": [np.nan, 50.0],
        "away_market_value_top11": [40.0, np.nan],
    })
    out = add_talent_features(df)
    assert out["talent_gap_top11"].isna().iloc[0]
    assert out["talent_gap_top11"].isna().iloc[1]
    assert out["talent_log_ratio"].isna().iloc[0]


def test_add_talent_features_does_not_mutate_input():
    df = pd.DataFrame({
        "home_market_value_top11": [100.0],
        "away_market_value_top11": [40.0],
    })
    _ = add_talent_features(df)
    assert "talent_gap_top11" not in df.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_talent_features.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mondiali.features.talent'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mondiali/features/talent.py
"""Talent differential primitive (Phase 1 feature + Phase 2 Elo anchor).

Derives per-match squad-talent differentials from the market-value columns
already present in matches.parquet (produced by add_tier3_features). NaN where
market value is missing; XGBoost handles NaN natively. No scraping, no merge —
inherits tier3's strict-pre-match anti-leakage guarantee.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TALENT_COLUMNS: list[str] = ["talent_gap_top11", "talent_log_ratio"]


def add_talent_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `matches` with talent differential columns added.

    - talent_gap_top11 = home_market_value_top11 - away_market_value_top11
    - talent_log_ratio = log1p(home_top11) - log1p(away_top11)

    NaN-preserving: if either side's top11 value is NaN, both outputs are NaN.
    """
    out = matches.copy()
    home = out["home_market_value_top11"].astype(float)
    away = out["away_market_value_top11"].astype(float)
    out["talent_gap_top11"] = home - away
    out["talent_log_ratio"] = np.log1p(home) - np.log1p(away)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_talent_features.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/mondiali/features/talent.py tests/test_talent_features.py
git commit -m "feat(step9): talent differential primitive (add_talent_features)"
```

---

### Task 2: `include_talent` flag on the symmetric row builder

**Files:**
- Modify: `src/mondiali/model/poisson_xgb.py` (add constant + flag; `build_symmetric_rows` and `PoissonXGBModel`)
- Test: `tests/test_poisson_xgb.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_poisson_xgb.py
import numpy as np
import pandas as pd

from mondiali.model.poisson_xgb import (
    SYMMETRIC_FEATURES,
    SYMMETRIC_FEATURES_TALENT_EXTRA,
    build_symmetric_rows,
)


def _one_match_df():
    return pd.DataFrame({
        "home_elo_before": [1600.0], "away_elo_before": [1500.0],
        "neutral": [True], "competition_importance": [75.0],
        "days_rest_home": [5.0], "days_rest_away": [6.0],
        "home_score": [2], "away_score": [1],
        "home_form_5": [0.6], "away_form_5": [0.4],
        "home_gd_5": [1.0], "away_gd_5": [-1.0],
        "home_goals_scored_5": [1.5], "away_goals_scored_5": [1.0],
        "home_goals_conceded_5": [0.5], "away_goals_conceded_5": [1.2],
        "home_avg_opp_elo_5": [1500.0], "away_avg_opp_elo_5": [1490.0],
        "home_market_value_total": [800.0], "away_market_value_total": [300.0],
        "home_market_value_top11": [500.0], "away_market_value_top11": [200.0],
        "home_tm_age_days": [100.0], "away_tm_age_days": [120.0],
        "talent_gap_top11": [300.0], "talent_log_ratio": [0.9],
    })


def test_include_talent_adds_two_features_with_sign_flip():
    df = _one_match_df()
    X, _ = build_symmetric_rows(df, include_talent=True)
    assert SYMMETRIC_FEATURES_TALENT_EXTRA == ["talent_gap", "talent_log_ratio"]
    assert X.shape == (2, len(SYMMETRIC_FEATURES) + 2)
    # talent block starts right after the 24 base features (tier4 off)
    gap_col = len(SYMMETRIC_FEATURES)
    ratio_col = gap_col + 1
    # home-perspective row (index 0): +gap ; away-perspective (index 1): -gap
    assert np.isclose(X[0, gap_col], 300.0)
    assert np.isclose(X[1, gap_col], -300.0)
    assert np.isclose(X[0, ratio_col], 0.9)
    assert np.isclose(X[1, ratio_col], -0.9)


def test_include_talent_false_unchanged_width():
    df = _one_match_df()
    X, _ = build_symmetric_rows(df, include_talent=False)
    assert X.shape == (2, len(SYMMETRIC_FEATURES))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_poisson_xgb.py -k include_talent -q`
Expected: FAIL with `ImportError: cannot import name 'SYMMETRIC_FEATURES_TALENT_EXTRA'`

- [ ] **Step 3: Write minimal implementation**

In `src/mondiali/model/poisson_xgb.py`, after `SYMMETRIC_FEATURES_TIER4_EXTRA`, add:

```python
SYMMETRIC_FEATURES_TALENT_EXTRA: list[str] = ["talent_gap", "talent_log_ratio"]
```

Change the `build_symmetric_rows` signature and feature-count line:

```python
def build_symmetric_rows(  # noqa: PLR0915
    matches: pd.DataFrame, *, include_tier4: bool = False,
    include_talent: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
```

```python
    n = len(matches)
    n_features = (
        len(SYMMETRIC_FEATURES)
        + (len(SYMMETRIC_FEATURES_TIER4_EXTRA) if include_tier4 else 0)
        + (len(SYMMETRIC_FEATURES_TALENT_EXTRA) if include_talent else 0)
    )
```

At the END of `build_symmetric_rows`, just before `return X, y`, add the talent
block. The offset is after the 24 base features plus the optional 4 tier4
columns:

```python
    if include_talent:
        base = len(SYMMETRIC_FEATURES) + (
            len(SYMMETRIC_FEATURES_TIER4_EXTRA) if include_tier4 else 0
        )
        gap = matches["talent_gap_top11"].to_numpy(dtype=float)
        log_ratio = matches["talent_log_ratio"].to_numpy(dtype=float)
        X[0::2, base] = gap          # home perspective: + gap
        X[1::2, base] = -gap         # away perspective: sign-flipped
        X[0::2, base + 1] = log_ratio
        X[1::2, base + 1] = -log_ratio

    return X, y
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_poisson_xgb.py -k include_talent -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/mondiali/model/poisson_xgb.py tests/test_poisson_xgb.py
git commit -m "feat(step9): include_talent flag adds sign-flipped talent differential features"
```

---

### Task 3: Thread `include_talent` through `PoissonXGBModel`

**Files:**
- Modify: `src/mondiali/model/poisson_xgb.py` (`PoissonXGBModel.__init__`, `fit`, `predict_lambda`)
- Test: `tests/test_poisson_xgb.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_poisson_xgb.py
def test_model_include_talent_trains_and_predicts():
    df = _one_match_df()
    train = pd.concat([df] * 40, ignore_index=True)
    model = PoissonXGBModel(
        params={"n_estimators": 10}, include_talent=True,
    )
    model.fit(train)
    lam_h, lam_a = model.predict_lambda(df)
    assert lam_h.shape == (1,) and lam_a.shape == (1,)
    assert np.isfinite(lam_h[0]) and np.isfinite(lam_a[0])
```

(Add `from mondiali.model.poisson_xgb import PoissonXGBModel` to the imports if
not already present.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_poisson_xgb.py -k include_talent_trains -q`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'include_talent'`

- [ ] **Step 3: Write minimal implementation**

In `PoissonXGBModel.__init__`:

```python
    def __init__(
        self,
        params: dict[str, Any] | None = None,
        *,
        include_tier4: bool = False,
        include_talent: bool = False,
    ) -> None:
        self.params: dict[str, Any] = {**DEFAULT_PARAMS, **(params or {})}
        self.include_tier4 = include_tier4
        self.include_talent = include_talent
        self.booster_: xgb.XGBRegressor | None = None
```

In `fit`, both `build_symmetric_rows` calls become:

```python
        X, y = build_symmetric_rows(  # noqa: N806
            matches, include_tier4=self.include_tier4,
            include_talent=self.include_talent,
        )
```

```python
            X_val, y_val = build_symmetric_rows(  # noqa: N806
                early_stopping_val, include_tier4=self.include_tier4,
                include_talent=self.include_talent,
            )
```

In `predict_lambda`:

```python
        X, _ = build_symmetric_rows(  # noqa: N806
            matches, include_tier4=self.include_tier4,
            include_talent=self.include_talent,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_poisson_xgb.py -q`
Expected: PASS (all, including the new tests)

- [ ] **Step 5: Commit**

```bash
git add src/mondiali/model/poisson_xgb.py tests/test_poisson_xgb.py
git commit -m "feat(step9): thread include_talent through PoissonXGBModel"
```

---

### Task 4: Leakage test for talent features

**Files:**
- Test: `tests/test_leakage.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_leakage.py
import numpy as np
import pandas as pd

from mondiali.features.talent import add_talent_features


def test_talent_features_are_nan_or_derived_from_pre_match_values():
    """Talent columns derive ONLY from tier3 market-value cols (already strict
    pre-match). Assert: where talent_gap is non-NaN, both source values exist."""
    matches = pd.read_parquet("data/processed/matches.parquet")
    out = add_talent_features(matches)
    non_nan = out["talent_gap_top11"].notna()
    assert (out.loc[non_nan, "home_market_value_top11"].notna()).all()
    assert (out.loc[non_nan, "away_market_value_top11"].notna()).all()
    # exact identity with the differential (no extra transformation/leakage)
    recomputed = (
        out.loc[non_nan, "home_market_value_top11"]
        - out.loc[non_nan, "away_market_value_top11"]
    )
    assert np.allclose(out.loc[non_nan, "talent_gap_top11"], recomputed)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_leakage.py -k talent -q`
Expected: PASS already if `matches.parquet` exists and Task 1 is done. If it
fails on a missing column, run `.venv\Scripts\python.exe -c "from mondiali.config import CONFIG; from mondiali.data.ingestion import build_processed_matches; build_processed_matches(CONFIG.data_raw/'results.csv', CONFIG.data_processed/'matches.parquet')"` first.
Expected after: PASS (1 passed)

- [ ] **Step 3: Commit**

```bash
git add tests/test_leakage.py
git commit -m "test(step9): talent features strict-pre-match leakage guard"
```

---

### Task 5: STEP 0 diagnostic — does talent gap explain v1.4 residuals?

**Files:**
- Create: `scripts/diagnose_talent_lever.py`
- Output: `reports/talent_diagnostic.json`

- [ ] **Step 1: Write the diagnostic script**

```python
# scripts/diagnose_talent_lever.py
"""STEP 0 diagnostic: does squad-talent gap explain v1.4's residual errors?

Uses XGB-only v1_final lambdas (no torch needed). On the OOS 2025-2026 slice,
on rows WITH market value, correlates talent_gap_top11 with:
  - the signed goal-diff residual: (home_score - away_score) - (lam_h - lam_a)
If positive and significant, high-talent teams beat v1.4's expectation -> the
lever exists and is currently under-exploited.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from mondiali.features.talent import add_talent_features
from mondiali.model.poisson_xgb import PoissonXGBModel


def main() -> None:
    df = pd.read_parquet("data/processed/matches.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["days_rest_home", "days_rest_away"]).copy()
    df = add_talent_features(df)

    oos = df[(df["date"] >= "2025-01-01") & (df["date"] <= "2026-03-31")].reset_index(drop=True)

    xgb = PoissonXGBModel().load(Path("models/v1_final/xgb_poisson.json"))
    lam_h, lam_a = xgb.predict_lambda(oos)

    actual_gd = (oos["home_score"] - oos["away_score"]).to_numpy(dtype=float)
    pred_gd = lam_h - lam_a
    residual = actual_gd - pred_gd  # >0: home did better than model expected

    mask = oos["talent_gap_top11"].notna().to_numpy()
    gap = oos["talent_gap_top11"].to_numpy(dtype=float)[mask]
    res = residual[mask]

    n = int(mask.sum())
    if n < 10:
        result = {"n_with_market_value": n, "verdict": "insufficient coverage"}
    else:
        pear_r, pear_p = stats.pearsonr(gap, res)
        spear_r, spear_p = stats.spearmanr(gap, res)
        slope = float(np.polyfit(gap, res, 1)[0])
        result = {
            "n_oos": int(len(oos)),
            "n_with_market_value": n,
            "coverage": round(n / len(oos), 4),
            "pearson_r": round(float(pear_r), 4),
            "pearson_p": round(float(pear_p), 6),
            "spearman_r": round(float(spear_r), 4),
            "spearman_p": round(float(spear_p), 6),
            "ols_slope": slope,
            "verdict": (
                "lever exists (talent gap predicts residual)"
                if pear_p < 0.05 and pear_r > 0
                else "no usable signal"
            ),
        }

    Path("reports").mkdir(exist_ok=True)
    Path("reports/talent_diagnostic.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the diagnostic**

Run: `.venv\Scripts\python.exe scripts/diagnose_talent_lever.py`
Expected: prints JSON with `pearson_r`, `verdict`, and writes `reports/talent_diagnostic.json`. Record the verdict — it informs whether Task 6's challenger is worth promoting and whether a coverage boost (Phase 1.5) is needed.

- [ ] **Step 3: Commit**

```bash
git add scripts/diagnose_talent_lever.py reports/talent_diagnostic.json
git commit -m "feat(step9): STEP 0 diagnostic — talent gap vs v1.4 residuals on OOS"
```

---

### Task 6: Challenger training pipeline

**Files:**
- Modify: `src/mondiali/training/train.py` (add `train_talent_challenger`)
- Test: `tests/test_train_tier2.py` (append a smoke test) OR new `tests/test_train_talent.py`

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_train_talent.py
from __future__ import annotations

import numpy as np

from mondiali.config import CONFIG
from mondiali.training.train import train_talent_challenger


def test_train_talent_challenger_smoke():
    parquet = CONFIG.data_processed / "matches.parquet"
    result = train_talent_challenger(
        parquet_path=parquet,
        model_params={"n_estimators": 50},  # fast
    )
    assert result["n_train"] > 1000
    assert np.isfinite(result["val_log_loss_raw"])
    assert result["model"].include_talent is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_train_talent.py -q`
Expected: FAIL with `ImportError: cannot import name 'train_talent_challenger'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/mondiali/training/train.py` (after `train_tier2_pipeline`). It is
`train_tier2_pipeline` with freeze splits as defaults and `include_talent=True`:

```python
def train_talent_challenger(
    parquet_path: Path,
    *,
    train_start: str = "2002-01-01",
    train_end: str = "2023-12-31",
    val_es_start: str = "2022-07-01",
    val_es_end: str = "2022-12-31",
    val_calib_start: str = "2023-01-01",
    val_calib_end: str = "2023-12-31",
    val_gate_start: str = "2024-01-01",
    val_gate_end: str = "2024-12-31",
    early_stopping_rounds: int = 50,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Challenger: Tier 2 stack + talent differential features (include_talent).

    Same architecture and splits as the freeze; the only difference vs v1_final
    is the 2 extra talent features. Returns the same dict shape as
    train_tier2_pipeline.
    """
    from mondiali.features.talent import add_talent_features

    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["days_rest_home", "days_rest_away"]).copy()
    df = add_talent_features(df)

    train = df[(df["date"] >= train_start) & (df["date"] <= train_end)].reset_index(drop=True)
    val_es = df[(df["date"] >= val_es_start) & (df["date"] <= val_es_end)].reset_index(drop=True)
    val_calib = df[
        (df["date"] >= val_calib_start) & (df["date"] <= val_calib_end)
    ].reset_index(drop=True)
    val_gate = df[
        (df["date"] >= val_gate_start) & (df["date"] <= val_gate_end)
    ].reset_index(drop=True)

    model = PoissonXGBModel(params=model_params, include_talent=True)
    model.fit(train, early_stopping_val=val_es, early_stopping_rounds=early_stopping_rounds)

    lam_h_tr, lam_a_tr = model.predict_lambda(train)
    rho = estimate_rho_mle(
        lam_h_tr, lam_a_tr,
        train["home_score"].to_numpy(), train["away_score"].to_numpy(),
    )

    lam_h_cal, lam_a_cal = model.predict_lambda(val_calib)
    raw_probs_calib = _compute_1x2_probs(lam_h_cal, lam_a_cal, rho=rho)
    outcomes_calib = compute_outcomes(val_calib)
    calibrator = IsotonicCalibrator1X2().fit(raw_probs_calib, outcomes_calib)

    lam_h_ga, lam_a_ga = model.predict_lambda(val_gate)
    raw_probs_gate = _compute_1x2_probs(lam_h_ga, lam_a_ga, rho=rho)
    cal_probs_gate = calibrator.predict(raw_probs_gate)

    return {
        "model": model,
        "rho": rho,
        "calibrator": calibrator,
        "val_log_loss_raw": log_loss_1x2(val_gate, raw_probs_gate),
        "val_log_loss_calib": log_loss_1x2(val_gate, cal_probs_gate),
        "brier_before": brier_score_1x2(val_gate, raw_probs_gate),
        "brier_after": brier_score_1x2(val_gate, cal_probs_gate),
        "n_train": len(train),
        "n_val_es": len(val_es),
        "n_val_calib": len(val_calib),
        "n_val_gate": len(val_gate),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_train_talent.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/mondiali/training/train.py tests/test_train_talent.py
git commit -m "feat(step9): train_talent_challenger pipeline (Tier 2 stack + include_talent)"
```

---

### Task 7: Gate comparison + STEP 9 report

**Files:**
- Create: `scripts/train_talent_challenger.py`
- Output: `models/challenger_talent/`, `reports/validation_step9.md`

- [ ] **Step 1: Write the runner script**

```python
# scripts/train_talent_challenger.py
"""Train the talent challenger, compare raw val_gate log-loss vs v1.4, report.

Apples-to-apples: same splits, same DC+Poisson stack; only the 2 talent
features differ. v1.4 baseline is computed fresh on the same val_gate 2024
(XGB-only raw), so the comparison isolates the talent features.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from mondiali.config import CONFIG
from mondiali.model.poisson_xgb import PoissonXGBModel
from mondiali.training.evaluate import brier_score_1x2, log_loss_1x2
from mondiali.training.train import _compute_1x2_probs, train_talent_challenger

GATE_MARGIN = 0.003
VAL_GATE = ("2024-01-01", "2024-12-31")


def _v1_raw_logloss_on_gate(parquet: Path) -> tuple[float, float]:
    df = pd.read_parquet(parquet)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["days_rest_home", "days_rest_away"]).copy()
    gate = df[(df["date"] >= VAL_GATE[0]) & (df["date"] <= VAL_GATE[1])].reset_index(drop=True)
    xgb = PoissonXGBModel().load(CONFIG.models_dir / "v1_final" / "xgb_poisson.json")
    rho = float((CONFIG.models_dir / "v1_final" / "rho.txt").read_text().strip())
    lam_h, lam_a = xgb.predict_lambda(gate)
    probs = _compute_1x2_probs(lam_h, lam_a, rho=rho)
    return float(log_loss_1x2(gate, probs)), float(brier_score_1x2(gate, probs))


def main() -> None:
    parquet = CONFIG.data_processed / "matches.parquet"
    res = train_talent_challenger(parquet_path=parquet)

    ch_ll = float(res["val_log_loss_raw"])
    ch_br = float(res["brier_before"])
    v1_ll, v1_br = _v1_raw_logloss_on_gate(parquet)
    delta = ch_ll - v1_ll

    out_dir = CONFIG.models_dir / "challenger_talent"
    res["model"].save(out_dir / "xgb_poisson.json")
    (out_dir / "rho.txt").write_text(f"{res['rho']:.6f}\n")

    if delta <= -GATE_MARGIN:
        verdict = "PROMOTE to v2 candidate (kept on branch; not into v1_final during freeze)"
    elif delta >= GATE_MARGIN:
        verdict = "REJECT (no improvement)"
    else:
        verdict = f"NO DECISION (|delta| < {GATE_MARGIN}) — review Brier"

    lines = [
        "# STEP 9 — Talent challenger validation",
        "",
        f"**Date:** {date.today().isoformat()}  ",
        "**Challenger:** Tier 2 stack + 2 talent differential features  ",
        f"**Splits:** train<=2023, val_calib 2023, val_gate {VAL_GATE[0]}..{VAL_GATE[1]}  ",
        "",
        "## val_gate 2024 (raw 1X2)",
        "",
        "| Model | log-loss | Brier |",
        "|---|---|---|",
        f"| v1.4 (XGB-only) | {v1_ll:.4f} | {v1_br:.4f} |",
        f"| Challenger (+talent) | {ch_ll:.4f} | {ch_br:.4f} |",
        f"| **Delta** | **{delta:+.4f}** | {ch_br - v1_br:+.4f} |",
        "",
        f"Gate margin: {GATE_MARGIN}. **Verdict: {verdict}**",
        "",
        f"Artefacts: `models/challenger_talent/` (NOT promoted to v1_final).",
        "",
    ]
    Path("reports/validation_step9.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-8:]))
    print(f"\nDelta vs v1.4 raw log-loss: {delta:+.4f}  ->  {verdict}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the challenger**

Run: `.venv\Scripts\python.exe scripts/train_talent_challenger.py`
Expected: prints the delta vs v1.4 and verdict; writes `reports/validation_step9.md` and `models/challenger_talent/{xgb_poisson.json,rho.txt}`.

- [ ] **Step 3: Verify the freeze was respected**

Run: `git status --porcelain models/v1_final/`
Expected: EMPTY output (no changes to `models/v1_final/`).

- [ ] **Step 4: Commit**

```bash
git add scripts/train_talent_challenger.py reports/validation_step9.md models/challenger_talent/
git commit -m "feat(step9): talent challenger gate comparison vs v1.4 + report"
```

---

### Task 8: Final verification

- [ ] **Step 1: Run the full test suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: the 5 known-stale pre-existing failures only (nation_resolver alias, tier3 gate, freeze calibrator, 2× tier4) — NO new failures. The talent tests pass.

- [ ] **Step 2: Lint**

Run: `.venv\Scripts\python.exe -m ruff check src/ scripts/ tests/`
Expected: All checks passed (fix any new findings).

- [ ] **Step 3: Confirm v1_final untouched**

Run: `git status --porcelain models/v1_final/`
Expected: EMPTY.

---

## Self-Review

**Spec coverage:**
- §3 STEP 0 diagnostic → Task 5. ✓
- §4.1 `add_talent_features` → Task 1. ✓
- §4.2 `include_talent` flag + sign-flip → Tasks 2-3. ✓
- §4.3 `train_talent_challenger` + `models/challenger_talent/` → Tasks 6-7. ✓
- §5 anti-leakage test → Task 4. ✓
- §6 gate comparison vs v1.4 + report → Task 7. ✓
- §7 testing → Tasks 1-4, 6. ✓
- §9 freeze respected (v1_final unchanged) → Tasks 7-8. ✓

**Placeholder scan:** none — every code step is complete.

**Type consistency:** `talent_gap_top11` / `talent_log_ratio` (matches columns) vs
`SYMMETRIC_FEATURES_TALENT_EXTRA = ["talent_gap", "talent_log_ratio"]` (feature
names) used consistently across Tasks 1, 2, 6. `include_talent` flag signature
identical in builder and model (Tasks 2-3). `train_talent_challenger` return
dict matches `train_tier2_pipeline` shape (Task 6).

**Note for executor:** STEP 0 (Task 5) is the cheap gate. If its verdict is "no
usable signal", still run Task 6-7 for completeness but expect REJECT — that
result is itself the evidence to move to Phase 2 (Elo anchor), per spec §8.
