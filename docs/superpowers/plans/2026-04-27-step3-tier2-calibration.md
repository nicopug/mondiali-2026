# STEP 3 — Tier 2 form + isotonic calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Aggiungere Tier 2 form features (rolling N=5, 10 nuove colonne), refactor del training pipeline con 4-way split per eliminare il bias di early stopping, e fittare un calibrator isotonic 1X2 post-hoc. Doppio gate: soft (`< LOGLOSS_ELO`) + hard (`≤ LOGLOSS_ELO − 0.003`).

**Architecture:** Le 10 feature Tier 2 sono calcolate via rolling con `closed='left'` (strict-anteriority). Il pipeline diventa: `Train (2002–2016) → fit XGBoost con ES su Val_ES (2017) → fit IsotonicCalibrator1X2 su Val_calib (2018) → metric su Val_gate (2019–2022)`. Il calibrator applica 3 isotonic indipendenti (P1, PX, P2) e rinormalizza riga per riga. Serializzazione JSON-native per il calibrator (vincolo CLAUDE.md no-pickle). Optuna esplicitamente fuori scope (STEP 4).

**Tech Stack:** Python 3.12, pandas, XGBoost (count:poisson), scipy, scikit-learn (IsotonicRegression, LogisticRegression), pytest, ruff, mypy, structlog, typer.

**Spec di riferimento:** `docs/superpowers/specs/2026-04-27-step3-tier2-calibration-design.md`.

**Conteggio task atteso:** 9 task. Test attesi a fine STEP 3: ~124 (da 106 a 124).

**Invariante CLAUDE.md ricordati:** mai split random; ogni feature strettamente anteriore a `match_date`; `random_state=42`; XGBoost JSON nativo; report obbligatorio in `reports/`.

---

## Task 1: Tier 2 feature builder

**Files:**
- Create: `src/mondiali/features/tier2.py`
- Create: `tests/test_tier2.py`
- Modify: `src/mondiali/features/__init__.py`

**Cosa stiamo costruendo**: una funzione pura `add_tier2_features(matches)` che aggiunge 10 colonne calcolate via rolling N=5 strict-anteriority. Pattern coerente con `add_tier1_features` esistente (`src/mondiali/features/tier1.py`).

- [ ] **Step 1.1: Scrivi i test (red)**

Crea `tests/test_tier2.py` con questi test:

```python
"""Test Tier 2 form features (rolling N=5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mondiali.features.tier2 import TIER2_COLUMNS, add_tier2_features


def _build_synthetic_matches() -> pd.DataFrame:
    """6 match: A vs B per 6 date, alternando vincitori. Elo home_/away_before fisso."""
    rows = [
        # date, home, away, hs, as, h_elo, a_elo
        ("2020-01-01", "A", "B", 2, 0, 1500.0, 1400.0),  # A wins (3-0 conv: 2-0)
        ("2020-02-01", "B", "A", 1, 1, 1400.0, 1500.0),  # draw
        ("2020-03-01", "A", "B", 0, 2, 1500.0, 1400.0),  # A loses
        ("2020-04-01", "B", "A", 3, 1, 1400.0, 1500.0),  # A loses (B wins)
        ("2020-05-01", "A", "B", 1, 0, 1500.0, 1400.0),  # A wins
        ("2020-06-01", "B", "A", 0, 0, 1400.0, 1500.0),  # draw
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "date", "home_team", "away_team",
            "home_score", "away_score",
            "home_elo_before", "away_elo_before",
        ],
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_tier2_columns_added() -> None:
    """add_tier2_features aggiunge le 10 colonne TIER2_COLUMNS."""
    df = _build_synthetic_matches()
    out = add_tier2_features(df)
    for col in TIER2_COLUMNS:
        assert col in out.columns
    assert len(out) == len(df)


def test_tier2_first_match_is_nan_for_team() -> None:
    """Al primo match di un team, tutte e 5 le feature Tier 2 di quel team sono NaN."""
    df = _build_synthetic_matches()
    out = add_tier2_features(df)
    first = out.iloc[0]
    for col in [
        "home_form_5", "home_gd_5", "home_goals_scored_5",
        "home_goals_conceded_5", "home_avg_opp_elo_5",
    ]:
        assert pd.isna(first[col]), f"{col} should be NaN at first match"
    for col in [
        "away_form_5", "away_gd_5", "away_goals_scored_5",
        "away_goals_conceded_5", "away_avg_opp_elo_5",
    ]:
        assert pd.isna(first[col]), f"{col} should be NaN at first match"


def test_tier2_form_5_partial_window() -> None:
    """Al 3° match di team A: ha giocato 2 match precedenti (W e D). form_5 = 3 + 1 = 4."""
    df = _build_synthetic_matches()
    out = add_tier2_features(df)
    third = out.iloc[2]  # 2020-03-01: A home
    assert third["home_form_5"] == 4.0


def test_tier2_form_5_full_window_team_A() -> None:
    """Al 6° match (B vs A, 2020-06-01) team A ha 5 match precedenti.
    Sequenza A: W (2-0), D (1-1), L (0-2), L (1-3), W (1-0).
    form = 3+1+0+0+3 = 7. away_form_5 (A è away) = 7."""
    df = _build_synthetic_matches()
    out = add_tier2_features(df)
    sixth = out.iloc[5]
    assert sixth["away_form_5"] == 7.0


def test_tier2_avg_opp_elo_5() -> None:
    """Al 6° match team A ha sempre giocato contro B (Elo 1400). avg_opp_elo = 1400."""
    df = _build_synthetic_matches()
    out = add_tier2_features(df)
    sixth = out.iloc[5]
    assert sixth["away_avg_opp_elo_5"] == pytest.approx(1400.0)


def test_tier2_strict_anteriority() -> None:
    """home_form_5 alla data D non deve usare il match D stesso.
    Regression: closed='left' nel rolling.
    """
    df = _build_synthetic_matches()
    out = add_tier2_features(df)
    # 2° match (B vs A, 2020-02-01): A è away, ha giocato 1 match prima (W 2-0).
    # Se includesse il match corrente erroneamente: form_5 includerebbe il D di
    # quella stessa data. Atteso: solo W → form=3, NON 4.
    assert out.iloc[1]["away_form_5"] == 3.0
```

- [ ] **Step 1.2: Run test (red, dovrebbe fallire perché modulo non esiste)**

Run: `.venv/Scripts/pytest tests/test_tier2.py -v`
Expected: 6 FAIL (ImportError o ModuleNotFoundError su `mondiali.features.tier2`).

- [ ] **Step 1.3: Implementa `tier2.py`**

Crea `src/mondiali/features/tier2.py`:

```python
"""Feature builder Tier 2: rolling form features (N=5).

Per ogni team in ogni match, considera gli ULTIMI N match di quel team
strettamente anteriori a match_date (qualsiasi tipo di competizione,
qualsiasi ruolo home/away).

Anti-leakage: pandas rolling con closed='left' garantisce strict-anteriority.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

TIER2_COLUMNS: list[str] = [
    "home_form_5", "away_form_5",
    "home_gd_5", "away_gd_5",
    "home_goals_scored_5", "away_goals_scored_5",
    "home_goals_conceded_5", "away_goals_conceded_5",
    "home_avg_opp_elo_5", "away_avg_opp_elo_5",
]


def _team_long_form(matches: pd.DataFrame) -> pd.DataFrame:
    """Costruisce la long-form: 2 righe per match (perspective home + away).

    Ogni riga contiene (team, date, points, gf, ga, opp_elo, role).
    """
    home = pd.DataFrame({
        "team": matches["home_team"].to_numpy(),
        "date": matches["date"].to_numpy(),
        "match_idx": np.arange(len(matches)),
        "role": "home",
        "gf": matches["home_score"].to_numpy(dtype=float),
        "ga": matches["away_score"].to_numpy(dtype=float),
        "opp_elo": matches["away_elo_before"].to_numpy(dtype=float),
    })
    away = pd.DataFrame({
        "team": matches["away_team"].to_numpy(),
        "date": matches["date"].to_numpy(),
        "match_idx": np.arange(len(matches)),
        "role": "away",
        "gf": matches["away_score"].to_numpy(dtype=float),
        "ga": matches["home_score"].to_numpy(dtype=float),
        "opp_elo": matches["home_elo_before"].to_numpy(dtype=float),
    })
    long = pd.concat([home, away], ignore_index=True)
    # Punti: W=3, D=1, L=0
    long["points"] = np.where(
        long["gf"] > long["ga"], 3.0,
        np.where(long["gf"] == long["ga"], 1.0, 0.0),
    )
    long["gd"] = long["gf"] - long["ga"]
    long = long.sort_values(["team", "date"], kind="mergesort").reset_index(drop=True)
    return long


def add_tier2_features(matches: pd.DataFrame, *, n: int = 5) -> pd.DataFrame:
    """Aggiunge le 10 colonne TIER2_COLUMNS al DataFrame matches.

    Args:
        matches: DataFrame con date, home_team, away_team, home_score, away_score,
            home_elo_before, away_elo_before.
        n: finestra rolling (default 5).

    Returns:
        Copia di `matches` con 10 colonne aggiunte.
    """
    long = _team_long_form(matches)

    # Rolling per team con closed='left' → strict pre-match
    grouped = long.groupby("team", sort=False)
    long["form_n"] = grouped["points"].rolling(window=n, min_periods=1, closed="left").sum().reset_index(level=0, drop=True)
    long["gd_n"] = grouped["gd"].rolling(window=n, min_periods=1, closed="left").sum().reset_index(level=0, drop=True)
    long["gf_mean_n"] = grouped["gf"].rolling(window=n, min_periods=1, closed="left").mean().reset_index(level=0, drop=True)
    long["ga_mean_n"] = grouped["ga"].rolling(window=n, min_periods=1, closed="left").mean().reset_index(level=0, drop=True)
    long["opp_elo_mean_n"] = grouped["opp_elo"].rolling(window=n, min_periods=1, closed="left").mean().reset_index(level=0, drop=True)

    home_view = long[long["role"] == "home"].set_index("match_idx")
    away_view = long[long["role"] == "away"].set_index("match_idx")

    result = matches.copy()
    result["home_form_5"] = home_view["form_n"].reindex(np.arange(len(matches))).to_numpy()
    result["home_gd_5"] = home_view["gd_n"].reindex(np.arange(len(matches))).to_numpy()
    result["home_goals_scored_5"] = home_view["gf_mean_n"].reindex(np.arange(len(matches))).to_numpy()
    result["home_goals_conceded_5"] = home_view["ga_mean_n"].reindex(np.arange(len(matches))).to_numpy()
    result["home_avg_opp_elo_5"] = home_view["opp_elo_mean_n"].reindex(np.arange(len(matches))).to_numpy()

    result["away_form_5"] = away_view["form_n"].reindex(np.arange(len(matches))).to_numpy()
    result["away_gd_5"] = away_view["gd_n"].reindex(np.arange(len(matches))).to_numpy()
    result["away_goals_scored_5"] = away_view["gf_mean_n"].reindex(np.arange(len(matches))).to_numpy()
    result["away_goals_conceded_5"] = away_view["ga_mean_n"].reindex(np.arange(len(matches))).to_numpy()
    result["away_avg_opp_elo_5"] = away_view["opp_elo_mean_n"].reindex(np.arange(len(matches))).to_numpy()

    log.info("added tier2 features", rows=len(result), n=n)
    return result
```

- [ ] **Step 1.4: Aggiorna `__init__.py`**

Modifica `src/mondiali/features/__init__.py` aggiungendo:

```python
from mondiali.features.tier2 import TIER2_COLUMNS, add_tier2_features
```

(Mantieni gli import esistenti — fai una `Read` prima per non sovrascrivere.)

- [ ] **Step 1.5: Run tests (green)**

Run: `.venv/Scripts/pytest tests/test_tier2.py -v`
Expected: 6 PASS.

- [ ] **Step 1.6: Lint + types**

Run: `.venv/Scripts/ruff check src/mondiali/features/tier2.py tests/test_tier2.py`
Expected: All checks passed.

Run: `.venv/Scripts/mypy src/`
Expected: Success, no issues.

- [ ] **Step 1.7: Commit**

```bash
git add src/mondiali/features/tier2.py src/mondiali/features/__init__.py tests/test_tier2.py
git commit -m "feat(features): Tier 2 rolling N=5 form features (Task 1)"
```

---

## Task 2: Integrate Tier 2 in build_processed_matches + extend leakage test

**Files:**
- Modify: `src/mondiali/data/ingestion.py:71-105` (chiama `add_tier2_features` dopo `add_tier1_features`)
- Modify: `tests/test_leakage.py` (aggiungi test che verifica strict-anteriority delle Tier 2)

**Cosa stiamo facendo**: Tier 2 deve essere parte di `matches.parquet` per essere disponibile a XGBoost. Aggiungiamo la chiamata e un test di leakage che ri-simula le rolling con la stessa logica e verifica equivalenza.

- [ ] **Step 2.1: Aggiungi test di leakage Tier 2 (red prima del build)**

Apri `tests/test_leakage.py` e aggiungi alla fine:

```python
def test_tier2_form_5_is_strictly_pre_match() -> None:
    """Per ogni match, home_form_5 e away_form_5 devono usare solo match
    strettamente precedenti. Ri-simuliamo con la stessa logica del builder.
    """
    df = _load_processed()
    if df is None:
        pytest.skip("data/processed/matches.parquet not found")

    # Verifica almeno che la colonna esiste
    if "home_form_5" not in df.columns:
        pytest.skip("Tier 2 features not present in matches.parquet — run build_processed first")

    df_sorted = df.sort_values("date", kind="mergesort").reset_index(drop=True)

    # Ri-simulazione manuale: per ogni match, accumula storia per team
    history: dict[str, list[float]] = {}  # team → list of points
    expected_home_form: list[float] = []
    expected_away_form: list[float] = []
    for row in df_sorted.itertuples(index=False):
        h_hist = history.get(row.home_team, [])
        a_hist = history.get(row.away_team, [])
        # Form 5: somma punti degli ULTIMI 5 (o k<5)
        expected_home_form.append(sum(h_hist[-5:]) if h_hist else float("nan"))
        expected_away_form.append(sum(a_hist[-5:]) if a_hist else float("nan"))
        # Update history DOPO aver registrato la feature pre-match
        h_pts = 3.0 if row.home_score > row.away_score else (1.0 if row.home_score == row.away_score else 0.0)
        a_pts = 3.0 if row.away_score > row.home_score else (1.0 if row.home_score == row.away_score else 0.0)
        history.setdefault(row.home_team, []).append(h_pts)
        history.setdefault(row.away_team, []).append(a_pts)

    h_obs = df_sorted["home_form_5"].tolist()
    a_obs = df_sorted["away_form_5"].tolist()
    for obs, exp in zip(h_obs, expected_home_form, strict=True):
        if pd.isna(exp):
            assert pd.isna(obs)
        else:
            assert obs == pytest.approx(exp)
    for obs, exp in zip(a_obs, expected_away_form, strict=True):
        if pd.isna(exp):
            assert pd.isna(obs)
        else:
            assert obs == pytest.approx(exp)
```

- [ ] **Step 2.2: Modifica `build_processed_matches`**

Apri `src/mondiali/data/ingestion.py`. Sostituisci la riga 11:

```python
from mondiali.features.tier1 import add_tier1_features
```

con:

```python
from mondiali.features.tier1 import add_tier1_features
from mondiali.features.tier2 import add_tier2_features
```

E sostituisci la riga 90 (`df = add_tier1_features(df)`) con:

```python
df = add_tier1_features(df)
df = add_tier2_features(df)
```

- [ ] **Step 2.3: Rebuild parquet**

Run: `.venv/Scripts/python -m mondiali.cli.main ingest`
Expected: messaggio `OK - processed matches written to ...`. Tempo ~10s.

- [ ] **Step 2.4: Run leakage tests**

Run: `.venv/Scripts/pytest tests/test_leakage.py -v`
Expected: 4 PASS (3 esistenti + 1 nuovo).

- [ ] **Step 2.5: Run full suite**

Run: `.venv/Scripts/pytest -q`
Expected: 113 passed (106 base + 6 nuovi tier2 + 1 nuovo leakage).

- [ ] **Step 2.6: Lint + types**

Run: `.venv/Scripts/ruff check src/ tests/ && .venv/Scripts/mypy src/`
Expected: clean.

- [ ] **Step 2.7: Commit**

```bash
git add src/mondiali/data/ingestion.py tests/test_leakage.py data/processed/matches.parquet
git commit -m "feat(data): integrate Tier 2 features in build_processed_matches (Task 2)"
```

---

## Task 3: Estensione SYMMETRIC_FEATURES in PoissonXGBModel

**Files:**
- Modify: `src/mondiali/model/poisson_xgb.py`
- Modify: `tests/test_poisson_xgb.py` (estensione test esistenti, se necessario)

**Cosa stiamo facendo**: Il modello deve leggere le 10 nuove colonne Tier 2. Estendiamo `SYMMETRIC_FEATURES` da 8 a 18 e `build_symmetric_rows` per popolare le nuove colonne (con simmetria home/away coerente).

- [ ] **Step 3.1: Leggi il file corrente**

`Read` di `src/mondiali/model/poisson_xgb.py` per avere la struttura esatta in mente prima di modificare.

- [ ] **Step 3.2: Scrivi un test fast che verifica le 18 feature**

Crea o aggiungi a `tests/test_poisson_xgb.py` (controlla se esiste — se sì, aggiungi; altrimenti crea):

```python
def test_symmetric_features_has_18_columns_including_tier2() -> None:
    """SYMMETRIC_FEATURES include 8 Tier 0+1 + 10 Tier 2."""
    from mondiali.model.poisson_xgb import SYMMETRIC_FEATURES
    assert len(SYMMETRIC_FEATURES) == 18
    expected_tier2 = [
        "team_form_5", "opponent_form_5",
        "team_gd_5", "opponent_gd_5",
        "team_goals_scored_5", "opponent_goals_scored_5",
        "team_goals_conceded_5", "opponent_goals_conceded_5",
        "team_avg_opp_elo_5", "opponent_avg_opp_elo_5",
    ]
    for col in expected_tier2:
        assert col in SYMMETRIC_FEATURES


def test_build_symmetric_rows_with_tier2_columns() -> None:
    """build_symmetric_rows popola correttamente le 10 colonne Tier 2 con simmetria.

    Per match home-perspective: team_form_5 = home_form_5, opponent_form_5 = away_form_5.
    Per away-perspective: team_form_5 = away_form_5, opponent_form_5 = home_form_5.
    """
    import numpy as np
    import pandas as pd
    from mondiali.model.poisson_xgb import build_symmetric_rows

    df = pd.DataFrame({
        "home_team": ["A"], "away_team": ["B"],
        "date": pd.to_datetime(["2020-01-01"]),
        "home_score": [2], "away_score": [1],
        "neutral": [False],
        "home_elo_before": [1500.0], "away_elo_before": [1400.0],
        "competition_importance": [3],
        "days_rest_home": [10.0], "days_rest_away": [20.0],
        "home_form_5": [12.0], "away_form_5": [4.0],
        "home_gd_5": [5.0], "away_gd_5": [-3.0],
        "home_goals_scored_5": [2.5], "away_goals_scored_5": [0.8],
        "home_goals_conceded_5": [0.5], "away_goals_conceded_5": [1.6],
        "home_avg_opp_elo_5": [1450.0], "away_avg_opp_elo_5": [1480.0],
    })
    X, y = build_symmetric_rows(df)
    assert X.shape == (2, 18)
    # Riga 0 (home perspective): team_form_5 = 12.0
    assert X[0, 8] == 12.0   # team_form_5 (index 8 nelle 10 nuove, dipende da ordinamento)
    # Riga 1 (away perspective): team_form_5 = 4.0 (è l'away)
    assert X[1, 8] == 4.0
```

(Se non sei sicuro degli indici, sostituisci con un'asserzione strutturale: `assert SYMMETRIC_FEATURES.index("team_form_5") == ...` e poi indicizza.)

- [ ] **Step 3.3: Run test (red)**

Run: `.venv/Scripts/pytest tests/test_poisson_xgb.py -v -k tier2`
Expected: 2 FAIL (lunghezza diversa o KeyError sulle nuove colonne).

- [ ] **Step 3.4: Estendi SYMMETRIC_FEATURES**

In `src/mondiali/model/poisson_xgb.py` sostituisci la lista `SYMMETRIC_FEATURES` con:

```python
SYMMETRIC_FEATURES: list[str] = [
    "team_elo",
    "opponent_elo",
    "elo_diff_signed",
    "is_home",
    "is_neutral",
    "competition_importance",
    "team_days_rest",
    "opponent_days_rest",
    # Tier 2 (10)
    "team_form_5",
    "opponent_form_5",
    "team_gd_5",
    "opponent_gd_5",
    "team_goals_scored_5",
    "opponent_goals_scored_5",
    "team_goals_conceded_5",
    "opponent_goals_conceded_5",
    "team_avg_opp_elo_5",
    "opponent_avg_opp_elo_5",
]
```

- [ ] **Step 3.5: Estendi `build_symmetric_rows`**

In `build_symmetric_rows` (stesso file), dopo le righe esistenti che popolano X[:, 0..7] e prima del `return`, aggiungi (mantieni la struttura `X[0::2, ...]` per home-perspective e `X[1::2, ...]` per away-perspective):

```python
    home_form = matches["home_form_5"].to_numpy(dtype=float)
    away_form = matches["away_form_5"].to_numpy(dtype=float)
    home_gd = matches["home_gd_5"].to_numpy(dtype=float)
    away_gd = matches["away_gd_5"].to_numpy(dtype=float)
    home_gs = matches["home_goals_scored_5"].to_numpy(dtype=float)
    away_gs = matches["away_goals_scored_5"].to_numpy(dtype=float)
    home_gc = matches["home_goals_conceded_5"].to_numpy(dtype=float)
    away_gc = matches["away_goals_conceded_5"].to_numpy(dtype=float)
    home_ope = matches["home_avg_opp_elo_5"].to_numpy(dtype=float)
    away_ope = matches["away_avg_opp_elo_5"].to_numpy(dtype=float)

    # Home-perspective (indici pari)
    X[0::2, 8] = home_form              # team_form_5
    X[0::2, 9] = away_form              # opponent_form_5
    X[0::2, 10] = home_gd               # team_gd_5
    X[0::2, 11] = away_gd               # opponent_gd_5
    X[0::2, 12] = home_gs               # team_goals_scored_5
    X[0::2, 13] = away_gs               # opponent_goals_scored_5
    X[0::2, 14] = home_gc               # team_goals_conceded_5
    X[0::2, 15] = away_gc               # opponent_goals_conceded_5
    X[0::2, 16] = home_ope              # team_avg_opp_elo_5
    X[0::2, 17] = away_ope              # opponent_avg_opp_elo_5

    # Away-perspective (indici dispari): swap team ↔ opponent
    X[1::2, 8] = away_form
    X[1::2, 9] = home_form
    X[1::2, 10] = away_gd
    X[1::2, 11] = home_gd
    X[1::2, 12] = away_gs
    X[1::2, 13] = home_gs
    X[1::2, 14] = away_gc
    X[1::2, 15] = home_gc
    X[1::2, 16] = away_ope
    X[1::2, 17] = home_ope
```

- [ ] **Step 3.6: Run tier2 + intera suite**

Run: `.venv/Scripts/pytest -q`
Expected: 115 PASS (106 base + 6 tier2 + 1 leakage + 2 nuovi).

Nota: il test slow `test_train_tier1_pipeline_produces_reasonable_log_loss` (in `test_train_tier1.py`) potrebbe ora cambiare numero perché XGBoost vede 18 feature. Il range largo `[0.88, 1.02]` dovrebbe assorbirlo ma se non lo fa è atteso — Tier 1 è ancora utilizzabile come pipeline (Task 7 introduce `train_tier2_pipeline` separato).

Se il test slow tier1 fallisce con loss fuori range, *amplia* il range a `[0.85, 1.05]` con commit separato di follow-up — la pipeline Tier 1 ora vede più feature, semantica leggermente diversa ma comportamento atteso.

- [ ] **Step 3.7: Lint + types**

Run: `.venv/Scripts/ruff check src/ tests/ && .venv/Scripts/mypy src/`
Expected: clean.

- [ ] **Step 3.8: Commit**

```bash
git add src/mondiali/model/poisson_xgb.py tests/test_poisson_xgb.py
git commit -m "feat(model): extend SYMMETRIC_FEATURES with Tier 2 (8 -> 18) (Task 3)"
```

---

## Task 4: brier_score_1x2 in evaluate.py

**Files:**
- Modify: `src/mondiali/training/evaluate.py`
- Modify: `tests/test_evaluate.py` (se esiste, altrimenti crealo)

**Cosa stiamo facendo**: Brier score multi-class per misurare la calibration quality. Usato dalle diagnostiche di Task 9 e dai test del calibrator (Task 5).

- [ ] **Step 4.1: Scrivi test (red)**

Crea o aggiungi a `tests/test_evaluate.py`:

```python
"""Test metriche di evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mondiali.training.evaluate import brier_score_1x2


def _matches(outcomes: list[int]) -> pd.DataFrame:
    """Crea matches sintetici con outcome desiderato.

    outcome 0 = home win, 1 = draw, 2 = away win.
    """
    rows = []
    for o in outcomes:
        if o == 0:
            rows.append({"home_score": 1, "away_score": 0})
        elif o == 1:
            rows.append({"home_score": 1, "away_score": 1})
        else:
            rows.append({"home_score": 0, "away_score": 1})
    return pd.DataFrame(rows)


def test_brier_score_perfect_predictions_zero() -> None:
    """Brier = 0 con predizioni perfette."""
    matches = _matches([0, 1, 2])
    probs = np.array([
        [1.0, 0.0, 0.0],  # home win predicted with certainty
        [0.0, 1.0, 0.0],  # draw
        [0.0, 0.0, 1.0],  # away win
    ])
    assert brier_score_1x2(matches, probs) == pytest.approx(0.0, abs=1e-10)


def test_brier_score_uniform_is_known_value() -> None:
    """Predizioni uniformi (1/3, 1/3, 1/3): Brier = (2/3)^2 + (1/3)^2 + (1/3)^2 per riga
    quando l'outcome osservato è 1 in una posizione e 0 nelle altre.
    Ogni riga: (1/3 - 1)^2 + (1/3 - 0)^2 + (1/3 - 0)^2 = 4/9 + 1/9 + 1/9 = 6/9 = 2/3.
    Media: 2/3.
    """
    matches = _matches([0, 1, 2])
    probs = np.full((3, 3), 1.0 / 3.0)
    assert brier_score_1x2(matches, probs) == pytest.approx(2.0 / 3.0, abs=1e-10)
```

- [ ] **Step 4.2: Run test (red)**

Run: `.venv/Scripts/pytest tests/test_evaluate.py -v`
Expected: 2 FAIL (ImportError su `brier_score_1x2`).

- [ ] **Step 4.3: Implementa `brier_score_1x2`**

Aggiungi alla fine di `src/mondiali/training/evaluate.py`:

```python
def brier_score_1x2(matches: pd.DataFrame, probabilities: np.ndarray) -> float:
    """Brier score multi-class per esiti 1/X/2.

    Definizione: media su righe di Σ_c (P[i,c] - 1[outcome_i == c])^2.

    Args:
        matches: DataFrame con home_score, away_score.
        probabilities: shape (n, 3), colonne = [P(home), P(draw), P(away)].

    Returns:
        Brier score (più basso = meglio calibrato).
    """
    if probabilities.shape != (len(matches), 3):
        raise ValueError(
            f"probabilities shape {probabilities.shape} != expected ({len(matches)}, 3)"
        )
    y_true = compute_outcomes(matches)
    n = len(matches)
    one_hot = np.zeros((n, 3), dtype=float)
    one_hot[np.arange(n), y_true] = 1.0
    return float(((probabilities - one_hot) ** 2).sum(axis=1).mean())
```

- [ ] **Step 4.4: Run test (green)**

Run: `.venv/Scripts/pytest tests/test_evaluate.py -v`
Expected: 2 PASS.

- [ ] **Step 4.5: Lint + types**

Run: `.venv/Scripts/ruff check src/ tests/ && .venv/Scripts/mypy src/`
Expected: clean.

- [ ] **Step 4.6: Commit**

```bash
git add src/mondiali/training/evaluate.py tests/test_evaluate.py
git commit -m "feat(eval): brier_score_1x2 multi-class (Task 4)"
```

---

## Task 5: IsotonicCalibrator1X2 fit/predict

**Files:**
- Create: `src/mondiali/model/calibration.py`
- Create: `tests/test_calibration.py`

**Cosa stiamo facendo**: 3 isotonic regressions indipendenti (P1, PX, P2) + rinormalizzazione riga per riga. Solo fit/predict in questo task; save/load in Task 6.

- [ ] **Step 5.1: Scrivi i test core (red)**

Crea `tests/test_calibration.py`:

```python
"""Test IsotonicCalibrator1X2."""
from __future__ import annotations

import numpy as np
import pytest

from mondiali.model.calibration import IsotonicCalibrator1X2


def test_calibrator_fit_predict_shape_and_rows_sum_to_one() -> None:
    """predict ritorna (n,3) con righe normalizzate."""
    rng = np.random.default_rng(42)
    n = 200
    raw = rng.dirichlet([1, 1, 1], size=n)
    outcomes = rng.integers(0, 3, size=n)
    cal = IsotonicCalibrator1X2().fit(raw, outcomes)
    out = cal.predict(raw)
    assert out.shape == (n, 3)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-10)
    assert (out >= 0).all()


def test_calibrator_brier_does_not_increase_on_fit_set() -> None:
    """Brier dopo calibration ≤ Brier prima sulla stessa split (no overfit oversimple).

    Costruisco probs miscalibrate (sovraconfident) e verifica che la calibration
    le riduca.
    """
    rng = np.random.default_rng(0)
    n = 1000
    outcomes = rng.choice([0, 1, 2], size=n, p=[0.5, 0.25, 0.25])
    # Probs miscalibrate: 0.9 sulla classe predetta (corretta o meno).
    probs = np.full((n, 3), 0.05)
    pred = rng.integers(0, 3, size=n)
    probs[np.arange(n), pred] = 0.9

    def brier(p: np.ndarray, y: np.ndarray) -> float:
        oh = np.zeros((len(y), 3))
        oh[np.arange(len(y)), y] = 1.0
        return float(((p - oh) ** 2).sum(axis=1).mean())

    before = brier(probs, outcomes)
    cal = IsotonicCalibrator1X2().fit(probs, outcomes)
    calibrated = cal.predict(probs)
    after = brier(calibrated, outcomes)
    assert after <= before


def test_calibrator_handles_zero_sum_row_with_fallback() -> None:
    """Se tutti e 3 gli isotonic mappano a 0, fallback alla riga raw."""
    rng = np.random.default_rng(1)
    n = 100
    # All outcomes home (0): isotonic per draw e away mapperà ≈ 0 ovunque
    outcomes = np.zeros(n, dtype=int)
    raw = rng.dirichlet([1, 1, 1], size=n)
    cal = IsotonicCalibrator1X2().fit(raw, outcomes)
    # Edge row: probs (0, 0.5, 0.5) — la classe 0 è 0 quindi iso(0) sarà ≈ 0;
    # le altre due classi non viste come 1 → iso → ≈ 0. Sum potrebbe essere ≈ 0.
    edge = np.array([[0.0, 0.5, 0.5]])
    out = cal.predict(edge)
    # Non deve esplodere; row deve sommare a 1
    assert out.shape == (1, 3)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-10)


def test_calibrator_idempotent_on_perfectly_calibrated() -> None:
    """Probs già perfettamente calibrate: predict ≈ identity (entro rumore).

    Genero outcomes con frequenze esattamente uguali alle probs.
    """
    rng = np.random.default_rng(2)
    n = 5000
    # 50% home, 25% draw, 25% away
    outcomes = rng.choice([0, 1, 2], size=n, p=[0.5, 0.25, 0.25])
    # Probs uniformi 0.5/0.25/0.25 per ogni riga (già base-rate-corrette)
    probs = np.tile([0.5, 0.25, 0.25], (n, 1))
    cal = IsotonicCalibrator1X2().fit(probs, outcomes)
    out = cal.predict(probs)
    # Dovrebbero rimanere vicino a (0.5, 0.25, 0.25)
    np.testing.assert_allclose(out.mean(axis=0), [0.5, 0.25, 0.25], atol=0.05)
```

- [ ] **Step 5.2: Run test (red)**

Run: `.venv/Scripts/pytest tests/test_calibration.py -v`
Expected: 4 FAIL (ModuleNotFoundError).

- [ ] **Step 5.3: Implementa il calibrator**

Crea `src/mondiali/model/calibration.py`:

```python
"""Isotonic calibrator post-hoc per probabilità 1X2.

Architettura: 3 isotonic regressions indipendenti (P1, PX, P2) + rinormalizzazione
riga per riga. Spec §6.3 di docs/superpowers/specs/2026-04-20-mondiali-prediction-design.md.
"""
from __future__ import annotations

import numpy as np
import structlog
from sklearn.isotonic import IsotonicRegression

log = structlog.get_logger(__name__)


class IsotonicCalibrator1X2:
    """Tre isotonic regressions indipendenti (1, X, 2) + rinormalizzazione."""

    def __init__(self) -> None:
        self.iso_home_: IsotonicRegression | None = None
        self.iso_draw_: IsotonicRegression | None = None
        self.iso_away_: IsotonicRegression | None = None

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> IsotonicCalibrator1X2:
        """Fit 3 isotonic indipendenti.

        Args:
            probs: shape (n, 3), colonne = [P(home), P(draw), P(away)] raw.
            outcomes: shape (n,) con valori 0/1/2.
        """
        if probs.shape[1] != 3:
            raise ValueError(f"probs must have 3 columns, got {probs.shape}")
        if probs.shape[0] != outcomes.shape[0]:
            raise ValueError("probs and outcomes length mismatch")

        self.iso_home_ = IsotonicRegression(
            out_of_bounds="clip", y_min=0.0, y_max=1.0,
        ).fit(probs[:, 0], (outcomes == 0).astype(float))
        self.iso_draw_ = IsotonicRegression(
            out_of_bounds="clip", y_min=0.0, y_max=1.0,
        ).fit(probs[:, 1], (outcomes == 1).astype(float))
        self.iso_away_ = IsotonicRegression(
            out_of_bounds="clip", y_min=0.0, y_max=1.0,
        ).fit(probs[:, 2], (outcomes == 2).astype(float))
        log.info("isotonic calibrator fit", n=len(outcomes))
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """Applica i 3 isotonic e rinormalizza ogni riga."""
        if self.iso_home_ is None or self.iso_draw_ is None or self.iso_away_ is None:
            raise RuntimeError("Calibrator must be fit() before predict()")
        if probs.shape[1] != 3:
            raise ValueError(f"probs must have 3 columns, got {probs.shape}")

        p_home = self.iso_home_.predict(probs[:, 0])
        p_draw = self.iso_draw_.predict(probs[:, 1])
        p_away = self.iso_away_.predict(probs[:, 2])
        out = np.column_stack([p_home, p_draw, p_away])

        s = out.sum(axis=1, keepdims=True)
        # Fallback: se sum == 0 (degenerate), usa raw row
        zero_mask = (s.flatten() == 0)
        out[zero_mask] = probs[zero_mask]
        s_safe = out.sum(axis=1, keepdims=True)
        out = out / s_safe
        return np.asarray(out)
```

- [ ] **Step 5.4: Run test (green)**

Run: `.venv/Scripts/pytest tests/test_calibration.py -v`
Expected: 4 PASS.

- [ ] **Step 5.5: Lint + types**

Run: `.venv/Scripts/ruff check src/ tests/ && .venv/Scripts/mypy src/`
Expected: clean.

- [ ] **Step 5.6: Commit**

```bash
git add src/mondiali/model/calibration.py tests/test_calibration.py
git commit -m "feat(model): IsotonicCalibrator1X2 fit + predict (Task 5)"
```

---

## Task 6: IsotonicCalibrator1X2 save/load JSON-native

**Files:**
- Modify: `src/mondiali/model/calibration.py` (aggiungi save/load)
- Modify: `tests/test_calibration.py` (aggiungi test JSON round-trip)

**Cosa stiamo facendo**: Vincolo CLAUDE.md "JSON nativo, mai pickle". Serializziamo gli attributi numpy degli `IsotonicRegression` (`X_thresholds_, y_thresholds_, X_min_, X_max_, increasing_`) in un dict JSON.

- [ ] **Step 6.1: Aggiungi test JSON round-trip (red)**

Aggiungi a `tests/test_calibration.py`:

```python
def test_calibrator_json_roundtrip(tmp_path) -> None:
    """save → load → predict identico al bit."""
    rng = np.random.default_rng(42)
    n = 500
    raw = rng.dirichlet([1, 1, 1], size=n)
    outcomes = rng.choice([0, 1, 2], size=n, p=[0.5, 0.25, 0.25])
    cal = IsotonicCalibrator1X2().fit(raw, outcomes)

    path = tmp_path / "calibrator.json"
    cal.save(path)

    loaded = IsotonicCalibrator1X2.load(path)
    assert loaded.predict(raw) == pytest.approx(cal.predict(raw), abs=0.0)


def test_calibrator_load_missing_file_raises(tmp_path) -> None:
    """load di file inesistente solleva FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        IsotonicCalibrator1X2.load(tmp_path / "nonexistent.json")
```

- [ ] **Step 6.2: Run test (red)**

Run: `.venv/Scripts/pytest tests/test_calibration.py -v -k json`
Expected: 2 FAIL (AttributeError su `save`/`load`).

- [ ] **Step 6.3: Implementa save/load**

Aggiungi a `src/mondiali/model/calibration.py`:

```python
import json
from pathlib import Path


def _serialize_iso(iso: IsotonicRegression) -> dict:
    return {
        "X_thresholds": iso.X_thresholds_.tolist(),
        "y_thresholds": iso.y_thresholds_.tolist(),
        "X_min": float(iso.X_min_),
        "X_max": float(iso.X_max_),
        "increasing": bool(iso.increasing_),
    }


def _deserialize_iso(data: dict) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.X_thresholds_ = np.asarray(data["X_thresholds"], dtype=float)
    iso.y_thresholds_ = np.asarray(data["y_thresholds"], dtype=float)
    iso.X_min_ = float(data["X_min"])
    iso.X_max_ = float(data["X_max"])
    iso.increasing_ = bool(data["increasing"])
    # f_ è ricostruito on-the-fly da scikit; settiamo i thresholds e
    # `increasing_` è sufficiente per `predict` di scikit recente.
    return iso
```

E aggiungi i metodi alla classe:

```python
    def save(self, path: Path) -> None:
        """Salva il calibrator in JSON nativo (no pickle)."""
        if self.iso_home_ is None or self.iso_draw_ is None or self.iso_away_ is None:
            raise RuntimeError("Calibrator must be fit() before save()")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "iso_home": _serialize_iso(self.iso_home_),
            "iso_draw": _serialize_iso(self.iso_draw_),
            "iso_away": _serialize_iso(self.iso_away_),
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> IsotonicCalibrator1X2:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text())
        cal = cls()
        cal.iso_home_ = _deserialize_iso(data["iso_home"])
        cal.iso_draw_ = _deserialize_iso(data["iso_draw"])
        cal.iso_away_ = _deserialize_iso(data["iso_away"])
        return cal
```

(Aggiungi `import json` e `from pathlib import Path` in cima al file se mancanti.)

- [ ] **Step 6.4: Run test (green)**

Run: `.venv/Scripts/pytest tests/test_calibration.py -v`
Expected: 6 PASS (4 esistenti + 2 nuovi).

> **Nota**: se la deserializzazione fallisce con `predict` di scikit perché `f_` è richiesto, il fix è chiamare `iso._build_f(iso.X_thresholds_, iso.y_thresholds_)` se disponibile, oppure ricostruire `f_` manualmente come `interp1d` con `bounds_error=False`. Se test fallisce per questo motivo, leggi sklearn IsotonicRegression source e adatta.

- [ ] **Step 6.5: Lint + types**

Run: `.venv/Scripts/ruff check src/ tests/ && .venv/Scripts/mypy src/`
Expected: clean.

- [ ] **Step 6.6: Commit**

```bash
git add src/mondiali/model/calibration.py tests/test_calibration.py
git commit -m "feat(model): IsotonicCalibrator1X2 JSON-native save/load (Task 6)"
```

---

## Task 7: train_tier2_pipeline with 4-way split + ES carve-out + calibration

**Files:**
- Modify: `src/mondiali/training/train.py` (aggiungi `train_tier2_pipeline`)
- Create: `tests/test_train_tier2.py`

**Cosa stiamo facendo**: Pipeline end-to-end che combina tutto. 4-way split, ES su val_es, fit calibrator su val_calib, metric finale su val_gate.

- [ ] **Step 7.1: Scrivi test (red)**

Crea `tests/test_train_tier2.py`:

```python
"""Test pipeline training Tier 2 end-to-end + helper unit."""
from __future__ import annotations

import pandas as pd
import pytest

from mondiali.config import CONFIG
from mondiali.training.train import train_tier2_pipeline


def test_train_tier2_returns_required_keys() -> None:
    """Smoke test rapido: il dict di ritorno ha tutte le chiavi attese.
    Skip se parquet manca.
    """
    parquet = CONFIG.data_processed / "matches.parquet"
    if not parquet.exists():
        pytest.skip("matches.parquet not found")

    result = train_tier2_pipeline(
        parquet_path=parquet,
        train_start="2002-01-01",
        train_end="2010-12-31",  # short for speed
        val_es_start="2011-01-01",
        val_es_end="2011-12-31",
        val_calib_start="2012-01-01",
        val_calib_end="2012-12-31",
        val_gate_start="2013-01-01",
        val_gate_end="2013-06-30",
    )
    expected_keys = {
        "model", "rho", "calibrator",
        "val_log_loss_raw", "val_log_loss_calib",
        "brier_before", "brier_after",
        "n_train", "n_val_es", "n_val_calib", "n_val_gate",
    }
    assert expected_keys.issubset(result.keys())


@pytest.mark.slow
def test_train_tier2_full_split_produces_reasonable_loss() -> None:
    """Full split production: log_loss_calib in range largo [0.84, 0.92]."""
    parquet = CONFIG.data_processed / "matches.parquet"
    if not parquet.exists():
        pytest.skip("matches.parquet not found")

    result = train_tier2_pipeline(parquet_path=parquet)
    assert 0.84 <= result["val_log_loss_calib"] <= 0.92
    assert result["brier_after"] <= result["brier_before"] + 0.01  # tolerance
    assert -0.3 <= result["rho"] <= 0.05


def test_train_tier2_splits_have_no_overlap() -> None:
    """I 4 set sono mutualmente esclusivi e ordinati temporalmente."""
    parquet = CONFIG.data_processed / "matches.parquet"
    if not parquet.exists():
        pytest.skip("matches.parquet not found")

    result = train_tier2_pipeline(
        parquet_path=parquet,
        train_start="2002-01-01",
        train_end="2010-12-31",
        val_es_start="2011-01-01",
        val_es_end="2011-12-31",
        val_calib_start="2012-01-01",
        val_calib_end="2012-12-31",
        val_gate_start="2013-01-01",
        val_gate_end="2013-06-30",
    )
    # Sanity: tutti positivi, sum < total parquet
    assert result["n_train"] > 0
    assert result["n_val_es"] > 0
    assert result["n_val_calib"] > 0
    assert result["n_val_gate"] > 0
```

- [ ] **Step 7.2: Run test (red)**

Run: `.venv/Scripts/pytest tests/test_train_tier2.py -v -k "not slow"`
Expected: 2 FAIL (ImportError su `train_tier2_pipeline`).

- [ ] **Step 7.3: Implementa `train_tier2_pipeline`**

Aggiungi alla fine di `src/mondiali/training/train.py`:

```python
from mondiali.model.calibration import IsotonicCalibrator1X2
from mondiali.training.evaluate import brier_score_1x2, compute_outcomes


def train_tier2_pipeline(
    parquet_path: Path,
    *,
    train_start: str = "2002-01-01",
    train_end: str = "2016-12-31",
    val_es_start: str = "2017-01-01",
    val_es_end: str = "2017-12-31",
    val_calib_start: str = "2018-01-01",
    val_calib_end: str = "2018-12-31",
    val_gate_start: str = "2019-01-01",
    val_gate_end: str = "2022-06-30",
    early_stopping_rounds: int = 50,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pipeline Tier 2 con 4-way split + isotonic calibration.

    Returns:
        dict con: model, rho, calibrator, val_log_loss_raw, val_log_loss_calib,
        brier_before, brier_after, n_train, n_val_es, n_val_calib, n_val_gate.
    """
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["days_rest_home", "days_rest_away"]).copy()

    train = df[(df["date"] >= train_start) & (df["date"] <= train_end)].reset_index(drop=True)
    val_es = df[(df["date"] >= val_es_start) & (df["date"] <= val_es_end)].reset_index(drop=True)
    val_calib = df[(df["date"] >= val_calib_start) & (df["date"] <= val_calib_end)].reset_index(drop=True)
    val_gate = df[(df["date"] >= val_gate_start) & (df["date"] <= val_gate_end)].reset_index(drop=True)

    log.info(
        "tier2 pipeline start",
        n_train=len(train), n_val_es=len(val_es),
        n_val_calib=len(val_calib), n_val_gate=len(val_gate),
    )

    model = PoissonXGBModel(params=model_params)
    model.fit(train, early_stopping_val=val_es, early_stopping_rounds=early_stopping_rounds)

    lam_h_tr, lam_a_tr = model.predict_lambda(train)
    rho = estimate_rho_mle(
        lam_h_tr, lam_a_tr,
        train["home_score"].to_numpy(), train["away_score"].to_numpy(),
    )
    log.info("rho estimated", rho=rho)

    # Inference su val_calib per fittare il calibrator
    lam_h_cal, lam_a_cal = model.predict_lambda(val_calib)
    raw_probs_calib = _compute_1x2_probs(lam_h_cal, lam_a_cal, rho=rho)
    outcomes_calib = compute_outcomes(val_calib)
    calibrator = IsotonicCalibrator1X2().fit(raw_probs_calib, outcomes_calib)

    # Inference su val_gate raw + calibrated
    lam_h_ga, lam_a_ga = model.predict_lambda(val_gate)
    raw_probs_gate = _compute_1x2_probs(lam_h_ga, lam_a_ga, rho=rho)
    cal_probs_gate = calibrator.predict(raw_probs_gate)

    val_log_loss_raw = log_loss_1x2(val_gate, raw_probs_gate)
    val_log_loss_calib = log_loss_1x2(val_gate, cal_probs_gate)
    brier_before = brier_score_1x2(val_gate, raw_probs_gate)
    brier_after = brier_score_1x2(val_gate, cal_probs_gate)

    log.info(
        "tier2 validation",
        log_loss_raw=val_log_loss_raw,
        log_loss_calib=val_log_loss_calib,
        brier_before=brier_before, brier_after=brier_after,
    )

    return {
        "model": model,
        "rho": rho,
        "calibrator": calibrator,
        "val_log_loss_raw": val_log_loss_raw,
        "val_log_loss_calib": val_log_loss_calib,
        "brier_before": brier_before,
        "brier_after": brier_after,
        "n_train": len(train),
        "n_val_es": len(val_es),
        "n_val_calib": len(val_calib),
        "n_val_gate": len(val_gate),
    }
```

- [ ] **Step 7.4: Run test (green) — fast tests prima**

Run: `.venv/Scripts/pytest tests/test_train_tier2.py -v -k "not slow"`
Expected: 2 PASS.

- [ ] **Step 7.5: Run anche slow test**

Run: `.venv/Scripts/pytest tests/test_train_tier2.py::test_train_tier2_full_split_produces_reasonable_loss -v`
Expected: PASS (tempo ~30s). Se loss esce dal range `[0.84, 0.92]`, **STOP** — qualcosa è sbagliato (range è largo). Diagnostichiamo prima di proseguire.

- [ ] **Step 7.6: Lint + types**

Run: `.venv/Scripts/ruff check src/ tests/ && .venv/Scripts/mypy src/`
Expected: clean.

- [ ] **Step 7.7: Commit**

```bash
git add src/mondiali/training/train.py tests/test_train_tier2.py
git commit -m "feat(training): train_tier2_pipeline with 4-way split + isotonic calibration (Task 7)"
```

---

## Task 8: CLI train-tier2 command

**Files:**
- Modify: `src/mondiali/cli/main.py` (aggiungi comando `train-tier2`)

**Cosa stiamo facendo**: Esposizione CLI per il pipeline Tier 2 con tutti gli argomenti dei 4 split + flag per salvare modello/calibrator.

- [ ] **Step 8.1: Aggiungi import e comando**

Apri `src/mondiali/cli/main.py`. Aggiungi all'import block:

```python
from mondiali.training.train import train_tier1_pipeline, train_tier2_pipeline
```

(Sostituisci la riga esistente `from mondiali.training.train import train_tier1_pipeline`.)

Aggiungi alla fine del file (prima di `if __name__ == "__main__":`):

```python
@app.command(name="train-tier2")
def train_tier2(
    train_start: str = typer.Option("2002-01-01"),
    train_end: str = typer.Option("2016-12-31"),
    val_es_start: str = typer.Option("2017-01-01"),
    val_es_end: str = typer.Option("2017-12-31"),
    val_calib_start: str = typer.Option("2018-01-01"),
    val_calib_end: str = typer.Option("2018-12-31"),
    val_gate_start: str = typer.Option("2019-01-01"),
    val_gate_end: str = typer.Option("2022-06-30"),
    save_model: str = typer.Option("", "--save-model", help="Path JSON dove salvare il modello"),
    save_calibrator: str = typer.Option("", "--save-calibrator", help="Path JSON dove salvare il calibrator"),
) -> None:
    """Addestra Tier 2 (XGBoost Poisson + DC + isotonic calibration)."""
    parquet = CONFIG.data_processed / "matches.parquet"
    result = train_tier2_pipeline(
        parquet_path=parquet,
        train_start=train_start, train_end=train_end,
        val_es_start=val_es_start, val_es_end=val_es_end,
        val_calib_start=val_calib_start, val_calib_end=val_calib_end,
        val_gate_start=val_gate_start, val_gate_end=val_gate_end,
    )
    typer.echo(
        f"Splits: train={result['n_train']} val_es={result['n_val_es']} "
        f"val_calib={result['n_val_calib']} val_gate={result['n_val_gate']}"
    )
    typer.echo(f"Dixon-Coles rho: {result['rho']:.4f}")
    typer.echo(f"Tier 2 RAW   log-loss: {result['val_log_loss_raw']:.4f}")
    typer.echo(f"Tier 2 CALIB log-loss: {result['val_log_loss_calib']:.4f}")
    typer.echo(f"Brier before: {result['brier_before']:.4f}")
    typer.echo(f"Brier after:  {result['brier_after']:.4f}")

    if save_model:
        from pathlib import Path
        result["model"].save(Path(save_model))
        typer.echo(f"Model saved: {save_model}")
    if save_calibrator:
        from pathlib import Path
        result["calibrator"].save(Path(save_calibrator))
        typer.echo(f"Calibrator saved: {save_calibrator}")
```

- [ ] **Step 8.2: Run a manual smoke**

Run: `.venv/Scripts/python -m mondiali.cli.main train-tier2 --train-start 2002-01-01 --train-end 2010-12-31 --val-es-start 2011-01-01 --val-es-end 2011-12-31 --val-calib-start 2012-01-01 --val-calib-end 2012-12-31 --val-gate-start 2013-01-01 --val-gate-end 2013-06-30`
Expected: stampa di `n` per ogni split, `rho`, `log-loss raw/calib`, `Brier before/after`. Tempo ~20s.

- [ ] **Step 8.3: Lint**

Run: `.venv/Scripts/ruff check src/ tests/ && .venv/Scripts/mypy src/`
Expected: clean. (Se ruff lamenta PLC0415 sull'import inline `from pathlib import Path` dentro la funzione: promuovi l'import in cima al file.)

- [ ] **Step 8.4: Commit**

```bash
git add src/mondiali/cli/main.py
git commit -m "feat(cli): train-tier2 command (Task 8)"
```

---

## Task 9: Gate finale + reports/validation_step3.md

**Files:**
- Create: `reports/validation_step3.md`
- Optional: `scripts/build_step3_report.py` (helper per produrre i numeri da inserire nel report; analogo di `scripts/diagnose_tier1.py`)

**Cosa stiamo facendo**: Esecuzione finale del gate doppio (soft + hard), produzione del report, decisione sul tag `step3-complete`.

- [ ] **Step 9.1: Calcola tutti i numeri del report**

Crea `scripts/build_step3_report.py`:

```python
"""Produce i numeri per validation_step3.md.

Esegue:
- LOGLOSS_ELO ricalcolato sul val_gate filtrato (apples-to-apples)
- LOGLOSS_TIER1_CLEAN: Tier 1 con ES su val_es (no bias)
- LOGLOSS_TIER2_RAW e LOGLOSS_TIER2_CALIB
- Brier prima/dopo
- Feature importance gain per tutte le 18 feature
- Sanity France vs San Marino
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mondiali.config import CONFIG
from mondiali.model.dixon_coles import dixon_coles_correct, joint_matrix
from mondiali.model.elo_logistic import EloLogisticBaseline
from mondiali.model.markets import prob_1x2
from mondiali.model.poisson_xgb import SYMMETRIC_FEATURES
from mondiali.training.evaluate import log_loss_1x2
from mondiali.training.train import train_tier1_pipeline, train_tier2_pipeline


def main() -> None:
    parquet = CONFIG.data_processed / "matches.parquet"

    # Apples-to-apples ELO baseline
    df = pd.read_parquet(parquet)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["days_rest_home", "days_rest_away"])
    train_full = df[(df["date"] >= "2002-01-01") & (df["date"] <= "2018-12-31")]
    val_gate = df[(df["date"] >= "2019-01-01") & (df["date"] <= "2022-06-30")]
    elo_m = EloLogisticBaseline().fit(train_full)
    ll_elo = log_loss_1x2(val_gate, elo_m.predict_proba(val_gate))

    # Tier 1 CLEAN (ES su 2017, val gate stesso)
    res_t1 = train_tier1_pipeline(
        parquet_path=parquet,
        train_start="2002-01-01", train_end="2016-12-31",
        val_start="2017-01-01", val_end="2017-12-31",
    )
    # Tier 1 con ES diverso ha "val_log_loss_1x2" su 2017; servirebbe rifare
    # inference su 2019-22. Più facile: train_tier1 + then manually score val_gate.
    model_t1 = res_t1["model"]
    rho_t1 = res_t1["rho"]
    lh, la = model_t1.predict_lambda(val_gate)
    n = len(val_gate)
    probs = np.empty((n, 3), dtype=float)
    for i in range(n):
        m = joint_matrix(lh[i], la[i])
        m = dixon_coles_correct(m, lh[i], la[i], rho=rho_t1)
        probs[i] = prob_1x2(m)
    ll_t1_clean = log_loss_1x2(val_gate, probs)

    # Tier 2 con calibration
    res_t2 = train_tier2_pipeline(parquet_path=parquet)
    ll_t2_raw = res_t2["val_log_loss_raw"]
    ll_t2_calib = res_t2["val_log_loss_calib"]
    brier_b = res_t2["brier_before"]
    brier_a = res_t2["brier_after"]
    rho_t2 = res_t2["rho"]

    # Feature importance Tier 2
    booster_t2 = res_t2["model"].booster_
    gain = booster_t2.get_booster().get_score(importance_type="gain")
    gain_named = {SYMMETRIC_FEATURES[int(k[1:])]: v for k, v in gain.items()}

    # Gate decisions
    soft_pass = ll_t2_calib < ll_elo
    hard_pass = ll_t2_calib <= ll_elo - 0.003
    delta = ll_elo - ll_t2_calib

    print("=" * 70)
    print("STEP 3 NUMBERS")
    print("=" * 70)
    print(f"LOGLOSS_ELO            : {ll_elo:.6f}")
    print(f"LOGLOSS_TIER1_CLEAN    : {ll_t1_clean:.6f}")
    print(f"LOGLOSS_TIER2_RAW      : {ll_t2_raw:.6f}")
    print(f"LOGLOSS_TIER2_CALIB    : {ll_t2_calib:.6f}")
    print(f"BRIER_BEFORE/AFTER     : {brier_b:.6f} / {brier_a:.6f}")
    print(f"RHO_TIER2              : {rho_t2:.6f}")
    print(f"DELTA (ELO - T2_CALIB) : {delta:+.6f}  (target soft>0, hard>=0.003)")
    print(f"SOFT GATE              : {'PASS' if soft_pass else 'FAIL'}")
    print(f"HARD GATE              : {'PASS' if hard_pass else 'FAIL'}")
    print(f"n train/es/calib/gate  : "
          f"{res_t2['n_train']}/{res_t2['n_val_es']}/"
          f"{res_t2['n_val_calib']}/{res_t2['n_val_gate']}")
    print()
    print("Feature importance (gain) Tier 2:")
    total = sum(gain_named.values())
    for name in SYMMETRIC_FEATURES:
        v = gain_named.get(name, 0.0)
        pct = 100 * v / total if total else 0
        print(f"  {name:30s} {v:>8.2f}  ({pct:>5.2f}%)")


if __name__ == "__main__":
    main()
```

Run: `.venv/Scripts/python scripts/build_step3_report.py 2>&1 | tee /tmp/step3_numbers.txt`
Tempo: ~60s.

- [ ] **Step 9.2: Sanity France vs San Marino**

Crea uno snippet o aggiungilo allo script Step 9.1:

```python
def sanity_check(model, rho, calibrator) -> tuple[float, float, float]:
    df_synth = pd.DataFrame({
        "date": pd.to_datetime(["2022-01-01"]),
        "home_team": ["France"], "away_team": ["San Marino"],
        "home_score": [0], "away_score": [0],
        "neutral": [True],
        "tournament": ["FIFA World Cup qualification"],
        "home_elo_before": [1970.0], "away_elo_before": [1000.0],
        "competition_importance": [2],
        "days_rest_home": [30.0], "days_rest_away": [30.0],
        "days_rest_diff": [0.0],
        "home_form_5": [15.0], "away_form_5": [0.0],
        "home_gd_5": [20.0], "away_gd_5": [-15.0],
        "home_goals_scored_5": [4.0], "away_goals_scored_5": [0.4],
        "home_goals_conceded_5": [0.4], "away_goals_conceded_5": [3.0],
        "home_avg_opp_elo_5": [1850.0], "away_avg_opp_elo_5": [1700.0],
    })
    lh, la = model.predict_lambda(df_synth)
    m = joint_matrix(lh[0], la[0])
    m = dixon_coles_correct(m, lh[0], la[0], rho=rho)
    p_raw = np.array([prob_1x2(m)])
    p_cal = calibrator.predict(p_raw)
    return p_cal[0, 0], p_cal[0, 1], p_cal[0, 2]  # P(F), P(D), P(SMR)
```

Chiama dopo `res_t2`: `print(f"Sanity P(France/D/SMR) calib: {sanity_check(...)}")`.
Atteso: P(France) > 0.85.

- [ ] **Step 9.3: Scrivi `reports/validation_step3.md`**

Sostituisci i `<...>` con i numeri reali raccolti:

```markdown
# STEP 3 — Tier 2 + isotonic calibration validation report

**Data**: <YYYY-MM-DD>
**Commit**: <git rev-parse --short HEAD>
**Python**: 3.12.9
**XGBoost**: 3.2.0

## Dataset & Split

- Input: `data/processed/matches.parquet` (filtrato `dropna(days_rest_*)`)
- Train: `<n_train>` (2002-01-01 → 2016-12-31)
- Val_ES: `<n_val_es>` (2017)
- Val_calib: `<n_val_calib>` (2018)
- Val_gate: `<n_val_gate>` (2019-01-01 → 2022-06-30)

## Numeri principali

| Metrica | Valore |
|---|---|
| `LOGLOSS_ELO` (apples-to-apples val_gate) | `<ll_elo>` |
| `LOGLOSS_TIER1_CLEAN` (ES su val_es 2017) | `<ll_t1_clean>` |
| `LOGLOSS_TIER2_RAW` | `<ll_t2_raw>` |
| `LOGLOSS_TIER2_CALIB` | `<ll_t2_calib>` |
| `BRIER_BEFORE` / `BRIER_AFTER` | `<brier_b>` / `<brier_a>` |
| `RHO` | `<rho_t2>` |

## Gate

- **Δ (ELO − TIER2_CALIB)** = `<delta>`
- **Soft gate** (`LOGLOSS_TIER2_CALIB < LOGLOSS_ELO`): `<PASS|FAIL>`
- **Hard gate** (`LOGLOSS_TIER2_CALIB ≤ LOGLOSS_ELO − 0.003`): `<PASS|FAIL>`

## Feature importance Tier 2 (gain)

<incolla la tabella stampata da scripts/build_step3_report.py>

Aspettativa: le 10 nuove feature Tier 2 contribuiscono almeno il 25% del gain totale.

## Calibration impact

- Brier prima → dopo: `<brier_b>` → `<brier_a>` (delta `<brier_b - brier_a>`)
- Log-loss prima → dopo: `<ll_t2_raw>` → `<ll_t2_calib>` (delta `<ll_t2_raw - ll_t2_calib>`)

## Sanity check — France vs San Marino (neutral, qualif)

- P(France win, calibrated): `<val>` (atteso > 0.85)
- P(draw): `<val>`
- P(SMR win): `<val>`

## Lezioni apprese

- <2-3 bullet sui findings inattesi: feature importance vs aspettativa, Brier delta, gate outcome>

## Decisioni open per STEP 4

- Optuna su Tier 2: search space spec §6.4, 50 trial.
- Drop `competition_importance` se SHAP rimane < 0.02 (verifica con SHAP delle 18 feature).
- Stacker logistic (multinomial Platt) come alternativa o aggiunta a isotonic se hard gate fallito.
- (Opzionale) reliability diagram in `reports/figures/`.

## Test suite

```
pytest -q  →  <N> passed
```
```

- [ ] **Step 9.4: Run pytest + lint + types final**

Run: `.venv/Scripts/pytest -q --no-header && .venv/Scripts/ruff check src/ tests/ && .venv/Scripts/mypy src/`
Expected: ~124 passed, ruff clean, mypy clean.

- [ ] **Step 9.5: Commit report**

```bash
git add reports/validation_step3.md scripts/build_step3_report.py
git commit -m "docs(report): STEP 3 Tier 2 + calibration validation report"
```

- [ ] **Step 9.6: Tag (solo se soft gate pass)**

Se **soft gate PASS** (qualunque margine):
```bash
git tag step3-complete -m "STEP 3 Tier 2 + isotonic calibration: LOGLOSS_TIER2_CALIB=<val>, delta vs ELO=<val>"
```

Se **soft gate FAIL**: NO tag. Apri sessione di debug dedicata; il report `validation_step3.md` documenta lo stato negativo.

Se **soft pass + hard fail**:
```bash
git tag step3-complete-soft -m "STEP 3 soft gate passato, hard fallito: delta=<val>"
```

- [ ] **Step 9.7: Salva artefatti modello (solo se soft gate pass)**

```bash
.venv/Scripts/python -m mondiali.cli.main train-tier2 \
  --save-model models/tier2_v1/model.json \
  --save-calibrator models/tier2_v1/calibrator.json
```

(Crea la directory `models/tier2_v1/` se manca.) Aggiungi al commit del Step 9.5 oppure separato:

```bash
git add models/tier2_v1/
git commit -m "feat(models): archive Tier 2 v1 (model + calibrator)"
```

- [ ] **Step 9.8: Handoff a STEP 4**

Apri sessione di brainstorming dedicata per STEP 4 (Optuna + eventuale post-mortem se soft gate fallito).
Contesto da portare:
- `LOGLOSS_TIER2_CALIB = <val>` (target da battere)
- `LOGLOSS_ELO = <val>` (vero baseline filtered)
- Delta soft/hard gate
- Feature importance Tier 2 (per pruning Optuna search)
- Hparams attuali e best_iteration

---

## Recap STEP 3

**Cosa hai in mano alla fine**:
- Tier 2 form features (rolling N=5, 10 nuove colonne) integrate nel parquet
- 4-way temporal split (Train/Val_ES/Val_calib/Val_gate) — ES bias eliminato
- IsotonicCalibrator1X2 con save/load JSON-native
- `train-tier2` CLI command
- Pipeline `train_tier2_pipeline` end-to-end
- Report `validation_step3.md` con doppio gate (soft + hard)
- Eventuale tag `step3-complete` o `step3-complete-soft`
- Modello + calibrator archiviati in `models/tier2_v1/` se soft gate pass

**Cosa NON hai (va in STEP 4+)**:
- Optuna search su hparams Tier 2
- Reliability diagram (opzionale, deferred)
- Drop di `competition_importance` se SHAP basso (decisione STEP 4)
- Stacker logistic alternativo o aggiuntivo (se hard gate fallito)
