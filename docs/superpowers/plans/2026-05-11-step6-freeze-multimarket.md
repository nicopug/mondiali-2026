# STEP 6 — Implementation Plan: Freeze v1_final + multi-market inference

**Spec:** `docs/superpowers/specs/2026-05-11-step6-freeze-multimarket-design.md`
**Branch:** `feat/step6-freeze`
**Target:** complete before 2026-06-11 (freeze invariant kicks in).

12 tasks. TDD per ognuno: red test → green implementation → commit. Conventional commits in English.

---

## Task 1: U/O markets in `model/markets.py`

**Files:**
- Modify: `src/mondiali/model/markets.py`
- Modify: `tests/test_markets.py`

- [ ] **Step 1: Red test**

Append a `tests/test_markets.py`:
```python
import numpy as np
from mondiali.model.markets import prob_over_under

def test_prob_over_under_25_on_simple_joint() -> None:
    # 4x4 joint matrix con scores [0..3] x [0..3], massa uniforme su quadrante
    joint = np.zeros((4, 4))
    joint[0, 0] = 0.1  # 0-0 → 0 gol, under
    joint[1, 1] = 0.2  # 1-1 → 2 gol, under
    joint[2, 1] = 0.3  # 2-1 → 3 gol, over
    joint[3, 0] = 0.4  # 3-0 → 3 gol, over
    over, under = prob_over_under(joint, line=2.5)
    assert under == pytest.approx(0.1 + 0.2)
    assert over == pytest.approx(0.3 + 0.4)


def test_prob_over_under_15_normalizes() -> None:
    joint = np.full((6, 6), 1.0 / 36.0)
    over, under = prob_over_under(joint, line=1.5)
    assert over + under == pytest.approx(1.0)
```

- [ ] **Step 2: Implementation**

```python
def prob_over_under(joint: np.ndarray, line: float) -> tuple[float, float]:
    """Sum joint matrix cells per total goals vs `line`.

    over = P(home + away > line), under = P(home + away <= line).
    Returns (over, under) as floats summing to 1.0 (within fp error).
    """
    n_home, n_away = joint.shape
    over = 0.0
    under = 0.0
    for h in range(n_home):
        for a in range(n_away):
            total = h + a
            if total > line:
                over += joint[h, a]
            else:
                under += joint[h, a]
    return float(over), float(under)
```

- [ ] **Step 3: Commit**
```
feat(markets): prob_over_under for arbitrary goal lines
```

---

## Task 2: BTTS market in `model/markets.py`

**Files:**
- Modify: `src/mondiali/model/markets.py`
- Modify: `tests/test_markets.py`

- [ ] **Step 1: Red test**

```python
def test_prob_btts_basic() -> None:
    joint = np.zeros((4, 4))
    joint[0, 0] = 0.2  # NG
    joint[1, 0] = 0.1  # NG (home only)
    joint[0, 2] = 0.1  # NG (away only)
    joint[1, 1] = 0.3  # GG
    joint[2, 1] = 0.2  # GG
    joint[1, 2] = 0.1  # GG
    yes, no = prob_btts(joint)
    assert no == pytest.approx(0.2 + 0.1 + 0.1)
    assert yes == pytest.approx(0.3 + 0.2 + 0.1)
```

- [ ] **Step 2: Implementation**

```python
def prob_btts(joint: np.ndarray) -> tuple[float, float]:
    """Probability both teams score (yes) vs at least one scoreless (no)."""
    yes = float(joint[1:, 1:].sum())
    no = float(joint[0, :].sum() + joint[1:, 0].sum())
    return yes, no
```

- [ ] **Step 3: Commit**
```
feat(markets): prob_btts (both teams to score)
```

---

## Task 3: State persistence schema + save/load

**Files:**
- Create: `src/mondiali/inference/state.py`
- Create: `src/mondiali/inference/__init__.py`
- Create: `tests/test_inference_state.py`

- [ ] **Step 1: Red test** — round-trip + correctness

```python
import pandas as pd
from pathlib import Path
from mondiali.inference.state import save_state, load_state, ELO_STATE_COLS, FORM_CACHE_COLS

def test_save_load_roundtrip(tmp_path: Path) -> None:
    matches = pd.DataFrame([
        {"date": pd.Timestamp("2024-01-01"), "home_team": "France", "away_team": "Italy",
         "home_score": 2, "away_score": 1, "home_elo_after": 1820, "away_elo_after": 1810,
         "neutral": False, "competition_importance": 50.0},
        {"date": pd.Timestamp("2024-03-15"), "home_team": "Italy", "away_team": "France",
         "home_score": 0, "away_score": 0, "home_elo_after": 1815, "away_elo_after": 1815,
         "neutral": False, "competition_importance": 50.0},
    ])
    save_state(matches, tmp_path)
    elo, form = load_state(tmp_path)
    assert set(elo.columns) == set(ELO_STATE_COLS)
    assert set(form.columns) >= set(FORM_CACHE_COLS)
    # France's Elo = 1815 (last value after their second match)
    assert float(elo[elo["nation"] == "France"]["elo"].iloc[0]) == pytest.approx(1815)
    # Each nation has at most 5 form rows (we keep last 5)
    assert form.groupby("nation").size().max() <= 5
```

- [ ] **Step 2: Implementation**

```python
"""State persistence for runtime inference.

Two parquet files in data/state/:
- elo_state.parquet: current Elo per nation (after the latest match)
- form_cache.parquet: last 5 matches per nation for rolling features
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

ELO_STATE_COLS = ["nation", "elo", "last_match_date"]
FORM_CACHE_COLS = [
    "nation", "match_date", "is_home", "is_neutral",
    "score_for", "score_against", "opponent_elo", "competition_importance",
]
FORM_WINDOW = 5


def save_state(matches: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _build_elo_state(matches).to_parquet(out_dir / "elo_state.parquet", index=False)
    _build_form_cache(matches).to_parquet(out_dir / "form_cache.parquet", index=False)


def load_state(state_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    elo = pd.read_parquet(state_dir / "elo_state.parquet")
    form = pd.read_parquet(state_dir / "form_cache.parquet")
    return elo, form


def _build_elo_state(matches: pd.DataFrame) -> pd.DataFrame:
    # Estrai (nation, elo, date) da ogni perspective, prendi l'ultima per nation
    home = matches[["home_team", "home_elo_after", "date"]].rename(
        columns={"home_team": "nation", "home_elo_after": "elo", "date": "last_match_date"}
    )
    away = matches[["away_team", "away_elo_after", "date"]].rename(
        columns={"away_team": "nation", "away_elo_after": "elo", "date": "last_match_date"}
    )
    stacked = pd.concat([home, away], ignore_index=True).sort_values("last_match_date")
    return stacked.groupby("nation", as_index=False).last()


def _build_form_cache(matches: pd.DataFrame) -> pd.DataFrame:
    home_rows = pd.DataFrame({
        "nation": matches["home_team"],
        "match_date": matches["date"],
        "is_home": ~matches["neutral"].astype(bool),
        "is_neutral": matches["neutral"].astype(bool),
        "score_for": matches["home_score"],
        "score_against": matches["away_score"],
        "opponent_elo": matches["away_elo_before"] if "away_elo_before" in matches else matches["away_elo_after"],
        "competition_importance": matches["competition_importance"],
    })
    away_rows = pd.DataFrame({
        "nation": matches["away_team"],
        "match_date": matches["date"],
        "is_home": False,
        "is_neutral": matches["neutral"].astype(bool),
        "score_for": matches["away_score"],
        "score_against": matches["home_score"],
        "opponent_elo": matches["home_elo_before"] if "home_elo_before" in matches else matches["home_elo_after"],
        "competition_importance": matches["competition_importance"],
    })
    stacked = pd.concat([home_rows, away_rows], ignore_index=True)
    # Top-5 più recenti per nation
    stacked = stacked.sort_values(["nation", "match_date"], ascending=[True, False])
    return stacked.groupby("nation").head(FORM_WINDOW).reset_index(drop=True)
```

- [ ] **Step 3: Commit**
```
feat(inference): state persistence (Elo + form cache) save/load
```

---

## Task 4: CLI `update-state` command

**Files:**
- Modify: `src/mondiali/cli/main.py`
- Create: `tests/test_cli_update_state.py` (smoke only)

- [ ] **Step 1: Implementation**

```python
@app.command(name="update-state")
def update_state() -> None:
    """Rebuild data/state/{elo_state,form_cache}.parquet from current matches.parquet."""
    matches_path = CONFIG.data_processed / "matches.parquet"
    if not matches_path.exists():
        typer.echo("matches.parquet missing", err=True)
        raise typer.Exit(1)
    matches = pd.read_parquet(matches_path)
    state_dir = CONFIG.project_root / "data" / "state"
    save_state(matches, state_dir)
    typer.echo(f"State updated in {state_dir}")
```

- [ ] **Step 2: Smoke test**
```bash
mondiali update-state
```
Verifica: i 2 parquet creati. Eseguito anche come setup per test successivi.

- [ ] **Step 3: Commit**
```
feat(cli): update-state command
```

---

## Task 5: Runtime feature builder

**Files:**
- Create: `src/mondiali/inference/predict.py`
- Create: `tests/test_predict.py`

- [ ] **Step 1: Red test** — confronto con batch pipeline

```python
def test_build_inference_row_matches_batch(tmp_path: Path) -> None:
    """Predict row for a historical match (using state at match.date - epsilon)
    must match what build_processed_matches produced for that match."""
    matches = pd.read_parquet("data/processed/matches.parquet")
    last = matches.iloc[-1]
    # Build state from history BEFORE that match
    history = matches[matches["date"] < last["date"]]
    state_dir = tmp_path / "state"
    save_state(history, state_dir)
    elo, form = load_state(state_dir)

    snapshots = pd.read_parquet("data/raw/transfermarkt/snapshots.parquet")
    row = build_inference_row(
        home=last["home_team"], away=last["away_team"],
        date=last["date"], neutral=bool(last["neutral"]),
        elo_state=elo, form_cache=form, tm_snapshots=snapshots,
        competition_importance=float(last["competition_importance"]),
    )
    # 24 feature columns presenti
    assert {"home_elo_before", "away_elo_before", "home_form_5", "away_form_5"}.issubset(row.columns)
    # I valori coincidono con quelli del matches.parquet (tolleranza 1e-6)
    for col in ["home_elo_before", "away_elo_before", "home_form_5", "away_form_5"]:
        assert abs(float(row[col].iloc[0]) - float(last[col])) < 1e-6
```

- [ ] **Step 2: Implementation**

```python
def build_inference_row(
    home: str, away: str, date: pd.Timestamp, neutral: bool,
    *, elo_state: pd.DataFrame, form_cache: pd.DataFrame,
    tm_snapshots: pd.DataFrame | None, competition_importance: float = 30.0,
) -> pd.DataFrame:
    """Build a single-row DataFrame with the 24 SYMMETRIC_FEATURES inputs."""
    # 1. Elo current
    # 2. Days rest from form_cache
    # 3. Form-5 / gd-5 / scored-5 / conceded-5 / avg_opp_elo_5 from form_cache filtered date<target
    # 4. TM market_value_total/top11/tm_age via merge_asof on snapshots
    ...
```

- [ ] **Step 3: Run tests + commit**
```
feat(inference): runtime feature builder build_inference_row
```

---

## Task 6: Predict function + dict output

**Files:**
- Modify: `src/mondiali/inference/predict.py`
- Modify: `tests/test_predict.py`

- [ ] **Step 1: Red test**

```python
def test_predict_match_returns_all_markets(tmp_path: Path) -> None:
    # ... setup state + model + calibrator
    out = predict_match(
        home="France", away="Italy", date=pd.Timestamp("2024-06-15"), neutral=True,
        state_dir=state_dir, model_dir=Path("models/tier2_v1"),
    )
    assert "lambda" in out
    assert set(out["markets"].keys()) == {
        "1x2", "over_under_1_5", "over_under_2_5", "over_under_3_5", "btts"
    }
    assert out["markets"]["1x2"]["home"] + out["markets"]["1x2"]["draw"] + out["markets"]["1x2"]["away"] == pytest.approx(1.0, abs=1e-3)
```

- [ ] **Step 2: Implementation**

Wraps `build_inference_row` + model.predict_lambda + DC correction + isotonic for 1X2 + raw for U/O and BTTS.

- [ ] **Step 3: Commit**
```
feat(inference): predict_match returns all 5 markets
```

---

## Task 7: Markets validation pipeline

**Files:**
- Create: `src/mondiali/training/validate_markets.py`
- Create: `tests/test_validate_markets.py`

- [ ] **Step 1: Red test** — soglie sanity

```python
def test_validate_markets_returns_per_market_metrics() -> None:
    # mock: 100 matches, ground truth random, model = baseline naive
    # → modello deve essere uguale o leggermente peggio del baseline naive
    metrics = validate_all_markets(model, calibrator, val_gate, rho)
    for market in ["over_under_1_5", "over_under_2_5", "over_under_3_5", "btts"]:
        assert "log_loss" in metrics[market]
        assert "brier" in metrics[market]
        assert "baseline_log_loss" in metrics[market]
        assert "baseline_brier" in metrics[market]
        assert "validated" in metrics[market]
```

- [ ] **Step 2: Implementation**

```python
def validate_all_markets(model, calibrator, val_gate: pd.DataFrame, rho: float) -> dict:
    """Returns dict[market_name] -> dict with log_loss, brier, baseline metrics, validated flag."""
    # 1. Predict joint per ogni partita di val_gate
    # 2. Per ogni market, extract probabilities + ground truth binary outcome
    # 3. Compute log-loss e Brier su modello + baseline (frequenza training)
    # 4. validated = (model_brier < baseline_brier - 0.005)
```

- [ ] **Step 3: Commit**
```
feat(validation): per-market gate validation (Brier vs baseline naive)
```

---

## Task 8: Freeze pipeline

**Files:**
- Create: `src/mondiali/training/freeze.py`
- Create: `tests/test_freeze.py`

- [ ] **Step 1: Red test**

```python
def test_freeze_v1_final_writes_all_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "v1_final"
    result = freeze_v1_final(
        matches_path=Path("data/processed/matches.parquet"),
        out_dir=out,
        train_end="2023-12-31",
        val_gate_start="2024-01-01",
        val_gate_end="2024-12-31",
    )
    for f in ["xgb_poisson.json", "calibrator.json", "rho.txt", "manifest.json", "markets_validation.json"]:
        assert (out / f).exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["version"] == "v1.0"
    assert manifest["n_features"] == 24
    assert "git_sha" in manifest
```

- [ ] **Step 2: Implementation**

Refit Tier 2 con date aggiornate, save tutti gli artifacts + manifest con git sha.

- [ ] **Step 3: Commit**
```
feat(training): freeze_v1_final with manifest + per-market validation
```

---

## Task 9: CLI `predict` command

**Files:**
- Modify: `src/mondiali/cli/main.py`
- Create: `tests/test_cli_predict.py`

- [ ] **Step 1: Implementation**

```python
@app.command()
def predict(
    home: str,
    away: str,
    date: str,
    neutral: bool = typer.Option(False, "--neutral"),
    competition_importance: float = typer.Option(30.0, "--competition-importance"),
    model_dir: Path = typer.Option(None, "--model-dir"),
) -> None:
    """Predict 1X2 + U/O 1.5/2.5/3.5 + BTTS for a single match."""
    if model_dir is None:
        model_dir = CONFIG.models_dir / "v1_final"
    state_dir = CONFIG.project_root / "data" / "state"
    out = predict_match(...)
    typer.echo(json.dumps(out, indent=2))
```

- [ ] **Step 2: Smoke**
```bash
mondiali predict France Italy 2026-06-15 --neutral
```

- [ ] **Step 3: Commit**
```
feat(cli): predict command
```

---

## Task 10: CLI `freeze-v1` command

**Files:**
- Modify: `src/mondiali/cli/main.py`

- [ ] **Step 1: Implementation**

Thin wrapper su `freeze_v1_final`.

- [ ] **Step 2: Commit**
```
feat(cli): freeze-v1 command
```

---

## Task 11: Anti-leakage test for predict

**Files:**
- Modify: `tests/test_leakage.py`

- [ ] **Step 1: Test**

```python
def test_predict_match_strict_pre() -> None:
    """predict_match per un match storico con date=match.date deve usare SOLO state < match.date."""
    matches = pd.read_parquet("data/processed/matches.parquet")
    target = matches.iloc[len(matches) // 2]
    # State con dati ≤ target.date (incluso il target stesso! sintomo di leak)
    save_state(matches[matches["date"] <= target["date"]], state_dir)
    # State pulito
    save_state(matches[matches["date"] < target["date"]], state_dir_clean)
    out_leaked = predict_match(target["home_team"], target["away_team"], target["date"], ...)
    out_clean = predict_match(target["home_team"], target["away_team"], target["date"], ...)
    # Se la pipeline è strict, le due predizioni devono coincidere
    # (il state "leaked" include il target match ma predict deve filtrarlo via date<target.date)
    assert out_leaked["lambda"]["home"] == pytest.approx(out_clean["lambda"]["home"])
```

- [ ] **Step 2: Commit**
```
test(leakage): predict_match strict pre-match invariant
```

---

## Task 12: E2E execution + validation report + freeze tag

- [ ] **Step 1: Run freeze**
```bash
mondiali update-state
mondiali freeze-v1
```

- [ ] **Step 2: Smoke test predict**
```bash
mondiali predict Argentina France 2026-06-15 --neutral
```

- [ ] **Step 3: Write `reports/validation_step6.md`**

Tabella di tutti i markets con metriche, baseline, verdetto. Esempio output `predict`. README update.

- [ ] **Step 4: Git tag**
```bash
git tag -a v1.0 -m "Model freeze for WC2026"
git push origin v1.0
```

- [ ] **Step 5: Commit**
```
docs(step6): validation report + v1_final freeze record
```
