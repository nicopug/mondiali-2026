# STEP 2 — Tier 1 XGBoost Poisson Implementation Plan

> **Per worker agentici:** REQUIRED SUB-SKILL: usa superpowers:subagent-driven-development o superpowers:executing-plans. Steps usano checkbox `- [ ]`.

**Goal:** Costruire un modello XGBoost Poisson Tier 1 che batta il baseline Elo-only logistic in log-loss su validation di ≥0.003, con Dixon-Coles correction e derivazione dei mercati 1X2/O/U 2.5/BTTS.

**Architecture:** Symmetric single-model (ogni match → 2 righe, prospettiva home + away). Feature set Tier 1: Elo pre-match, is_neutral, competition_importance ordinale 1-4, days_rest per team. Baseline di confronto: sklearn LogisticRegression su `[elo_diff, is_neutral]`. Dixon-Coles stimato con MLE sulla storia training. Markets derivati dal joint goal matrix 11×11 (i,j ∈ [0,10]).

**Tech Stack:** Python 3.11+, xgboost 2.x, scikit-learn, scipy.stats.poisson, pandas, pydantic, typer, pytest, ruff, mypy.

**Checkpoint intermedi** (NO separate tag git, solo "fermati e verifica"):
- **CP1** (dopo Task 2): Tier 1 features estratte, anti-leakage `days_rest` verde
- **CP2** (dopo Task 5): Elo-only logistic addestrato, log-loss validation ~0.98 (±0.02)
- **CP3** (dopo Task 7): XGBoost λ medi ~1.3, log-loss raw documentato
- **CP4** (dopo Task 10): Dixon-Coles ρ ∈ [-0.3, 0.0], markets 1X2 sum=1
- **Gate finale** (Task 12): Tier 1 log-loss < Elo-only - 0.003 su validation

**Scelte di design bloccate da brainstorming:**
1. Optuna → STEP 3 (hparams hand-tuned difensivi qui)
2. Isotonic calibration → STEP 3
3. Symmetric single-model dal giorno 1
4. Feature Tier 1 scritte in `matches.parquet` (estendiamo `build_processed_matches`)
5. Walk-forward CV (3 fold expanding) definito ma usato per diagnostica, non per ottimizzazione

---

## Task 1: Tier 1 feature builder — competition_importance + days_rest

**Files:**
- Create: `src/mondiali/features/tier1.py`
- Create: `tests/test_features_tier1.py`

- [ ] **Step 1.1: Test failing per `competition_importance_from_tournament`**

Crea `tests/test_features_tier1.py`:

```python
"""Test feature builder Tier 1."""
from __future__ import annotations

import pandas as pd
import pytest

from mondiali.features.tier1 import (
    add_days_rest,
    add_tier1_features,
    competition_importance_from_tournament,
)


@pytest.mark.parametrize(
    ("tournament", "expected"),
    [
        ("FIFA World Cup", 4),
        ("FIFA World Cup qualification", 2),
        ("UEFA Euro", 3),
        ("UEFA Euro qualification", 2),
        ("Copa América", 3),
        ("Friendly", 1),
        ("UEFA Nations League", 1),
        ("Random thing", 1),
    ],
)
def test_competition_importance_ordinal(tournament: str, expected: int) -> None:
    """Mappa il torneo a importanza ordinale 1-4."""
    assert competition_importance_from_tournament(tournament) == expected
```

- [ ] **Step 1.2: Verifica fallimento**

```bash
.venv/Scripts/pytest tests/test_features_tier1.py -v -k "competition_importance"
```

Expected: 8 fail con `ImportError`.

- [ ] **Step 1.3: Implementa `competition_importance_from_tournament`**

Crea `src/mondiali/features/tier1.py`:

```python
"""Feature builder Tier 1: competition_importance, days_rest."""
from __future__ import annotations

import pandas as pd
import structlog

from mondiali.features.elo import classify_tournament

log = structlog.get_logger(__name__)

_CATEGORY_TO_IMPORTANCE = {
    "world_cup": 4,
    "continental": 3,
    "qualification": 2,
    "friendly": 1,
    "default": 1,
}


def competition_importance_from_tournament(tournament: str) -> int:
    """Ordinal 1-4: 1=friendly/minor, 2=qualif, 3=continental, 4=World Cup."""
    return _CATEGORY_TO_IMPORTANCE[classify_tournament(tournament)]
```

- [ ] **Step 1.4: Verifica test competition_importance verdi**

```bash
.venv/Scripts/pytest tests/test_features_tier1.py -v -k "competition_importance"
```

Expected: 8 passed.

- [ ] **Step 1.5: Test failing per `add_days_rest`**

Aggiungi in `tests/test_features_tier1.py`:

```python
def test_add_days_rest_first_match_per_team_is_nan() -> None:
    """Il primo match di una squadra non ha storia → days_rest_home/away = NaN."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-10"]),
            "home_team": ["A", "A"],
            "away_team": ["B", "C"],
        }
    )
    result = add_days_rest(df)
    assert pd.isna(result.iloc[0]["days_rest_home"])
    assert pd.isna(result.iloc[0]["days_rest_away"])
    assert result.iloc[1]["days_rest_home"] == 9.0
    assert pd.isna(result.iloc[1]["days_rest_away"])


def test_add_days_rest_tracks_each_team_separately() -> None:
    """A gioca 2020-01-01 e 2020-01-20 (sempre home): days_rest_home seconda riga = 19."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-20"]),
            "home_team": ["A", "A"],
            "away_team": ["B", "C"],
        }
    )
    result = add_days_rest(df)
    assert result.iloc[1]["days_rest_home"] == 19.0


def test_add_days_rest_counts_as_team_regardless_of_home_away() -> None:
    """A gioca 2020-01-01 in casa, 2020-01-10 in trasferta: days_rest_away seconda = 9."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-10"]),
            "home_team": ["A", "B"],
            "away_team": ["C", "A"],
        }
    )
    result = add_days_rest(df)
    assert result.iloc[1]["days_rest_away"] == 9.0


def test_add_days_rest_diff_column_present() -> None:
    """Colonna days_rest_diff = home - away (entrambi i lati NaN ok)."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-05", "2020-01-20"]),
            "home_team": ["A", "B", "A"],
            "away_team": ["B", "A", "B"],
        }
    )
    result = add_days_rest(df)
    assert "days_rest_diff" in result.columns
    assert result.iloc[2]["days_rest_diff"] == pytest.approx(15.0 - 15.0)
```

- [ ] **Step 1.6: Verifica fallimento**

```bash
.venv/Scripts/pytest tests/test_features_tier1.py -v -k "days_rest"
```

Expected: 4 fail con ImportError.

- [ ] **Step 1.7: Implementa `add_days_rest`**

Aggiungi in `src/mondiali/features/tier1.py`:

```python
def add_days_rest(matches: pd.DataFrame) -> pd.DataFrame:
    """Calcola days_rest_home, days_rest_away, days_rest_diff per ogni match.

    Itera cronologicamente: per ogni team tiene traccia della data dell'ultimo
    match (home o away, non importa). NaN se prima volta che vediamo il team.

    Assume `matches` ordinato per data crescente (stesso invariante di
    EloSystem.build_history).

    Raises:
        ValueError: se `matches` non è ordinato per data crescente.
    """
    dates = matches["date"]
    if not dates.is_monotonic_increasing:
        raise ValueError(
            "matches must be sorted by date ascending before calling add_days_rest"
        )

    last_seen: dict[str, pd.Timestamp] = {}
    rest_home: list[float] = []
    rest_away: list[float] = []

    for row in matches.itertuples(index=False):
        date = row.date
        prev_h = last_seen.get(row.home_team)
        prev_a = last_seen.get(row.away_team)
        rest_home.append(float("nan") if prev_h is None else (date - prev_h).days)
        rest_away.append(float("nan") if prev_a is None else (date - prev_a).days)
        last_seen[row.home_team] = date
        last_seen[row.away_team] = date

    result = matches.copy()
    result["days_rest_home"] = rest_home
    result["days_rest_away"] = rest_away
    result["days_rest_diff"] = result["days_rest_home"] - result["days_rest_away"]
    return result
```

- [ ] **Step 1.8: Verifica test days_rest verdi**

```bash
.venv/Scripts/pytest tests/test_features_tier1.py -v -k "days_rest"
```

Expected: 4 passed.

- [ ] **Step 1.9: Test + implementazione `add_tier1_features`**

In `tests/test_features_tier1.py`:

```python
def test_add_tier1_features_adds_all_columns() -> None:
    """add_tier1_features aggiunge competition_importance + days_rest_*."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-10"]),
            "home_team": ["A", "A"],
            "away_team": ["B", "C"],
            "tournament": ["FIFA World Cup", "Friendly"],
        }
    )
    result = add_tier1_features(df)
    assert "competition_importance" in result.columns
    assert "days_rest_home" in result.columns
    assert "days_rest_away" in result.columns
    assert "days_rest_diff" in result.columns
    assert result.iloc[0]["competition_importance"] == 4
    assert result.iloc[1]["competition_importance"] == 1
```

Aggiungi in `src/mondiali/features/tier1.py`:

```python
def add_tier1_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge competition_importance + days_rest_{home, away, diff}."""
    result = add_days_rest(matches)
    result["competition_importance"] = result["tournament"].map(
        competition_importance_from_tournament
    )
    log.info("added tier1 features", rows=len(result))
    return result
```

- [ ] **Step 1.10: Lint + type check + commit**

```bash
.venv/Scripts/pytest tests/test_features_tier1.py -v
.venv/Scripts/ruff check src/mondiali/features/tier1.py tests/test_features_tier1.py
.venv/Scripts/mypy src/mondiali/features/tier1.py
```

Expected: 13 passed, 0 errori.

```bash
git add src/mondiali/features/tier1.py tests/test_features_tier1.py
git commit -m "feat(features): tier1 builder with competition_importance and days_rest"
```

---

## Task 2: Integrazione Tier 1 in `build_processed_matches` + estensione anti-leakage

**Files:**
- Modify: `src/mondiali/data/ingestion.py`
- Modify: `tests/test_ingestion.py`
- Modify: `tests/test_leakage.py`

- [ ] **Step 2.1: Test failing — build_processed_matches produce colonne Tier 1**

Aggiungi in `tests/test_ingestion.py`:

```python
def test_build_processed_matches_includes_tier1_features(tmp_path: Path) -> None:
    """matches.parquet deve includere competition_importance + days_rest_*."""
    raw_csv = tmp_path / "results.csv"
    raw_csv.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2018-07-15,France,Croatia,4,2,FIFA World Cup,Moscow,Russia,TRUE\n"
        "2018-09-06,France,Germany,0,0,UEFA Nations League,Munich,Germany,FALSE\n"
    )
    out_path = tmp_path / "matches.parquet"
    build_processed_matches(raw_csv, out_path)

    df = pd.read_parquet(out_path)
    for col in ("competition_importance", "days_rest_home", "days_rest_away", "days_rest_diff"):
        assert col in df.columns

    # France match 1 = WC → 4, match 2 = Nations League → 1
    assert df.iloc[0]["competition_importance"] == 4
    assert df.iloc[1]["competition_importance"] == 1
    # days_rest_home per Francia nella seconda riga = 53 giorni
    assert df.iloc[1]["days_rest_home"] == 53.0
```

- [ ] **Step 2.2: Verifica fallimento**

```bash
.venv/Scripts/pytest tests/test_ingestion.py -v -k "tier1"
```

Expected: FAIL — colonne non presenti.

- [ ] **Step 2.3: Estendi `build_processed_matches`**

In `src/mondiali/data/ingestion.py`, aggiungi import e chiamata:

```python
from mondiali.features.tier1 import add_tier1_features
```

Dentro `build_processed_matches`, subito dopo `df = elo.build_history(df)`:

```python
    df = add_tier1_features(df)
```

- [ ] **Step 2.4: Verifica test passa**

```bash
.venv/Scripts/pytest tests/test_ingestion.py -v
```

Expected: 8 passed (7 precedenti + 1 nuovo).

- [ ] **Step 2.5: Estensione anti-leakage — days_rest strettamente pre-match**

Aggiungi in `tests/test_leakage.py`:

```python
def test_days_rest_is_strictly_pre_match() -> None:
    """Per ogni match, days_rest_home/away riflette la storia PRIMA di quella data.
    Ri-simuliamo e confrontiamo.
    """
    df = _load_processed()
    if df is None:
        pytest.skip("data/processed/matches.parquet not found")

    df_sorted = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    last_seen: dict[str, pd.Timestamp] = {}
    expected_home: list[float] = []
    expected_away: list[float] = []
    for row in df_sorted.itertuples(index=False):
        prev_h = last_seen.get(row.home_team)
        prev_a = last_seen.get(row.away_team)
        expected_home.append(float("nan") if prev_h is None else (row.date - prev_h).days)
        expected_away.append(float("nan") if prev_a is None else (row.date - prev_a).days)
        last_seen[row.home_team] = row.date
        last_seen[row.away_team] = row.date

    # Confronto con NaN-aware
    h_obs = df_sorted["days_rest_home"].tolist()
    a_obs = df_sorted["days_rest_away"].tolist()
    for obs, exp in zip(h_obs, expected_home, strict=True):
        if pd.isna(exp):
            assert pd.isna(obs)
        else:
            assert obs == pytest.approx(exp)
    for obs, exp in zip(a_obs, expected_away, strict=True):
        if pd.isna(exp):
            assert pd.isna(obs)
        else:
            assert obs == pytest.approx(exp)
```

- [ ] **Step 2.6: Rigenera matches.parquet con le nuove colonne**

```bash
.venv/Scripts/mondiali ingest --force
```

Expected: log "wrote processed matches rows=49215". Nota: `--force` ri-scarica il CSV per essere sicuri della freshness; se preferisci evitare il download, rimuovi solo `data/processed/matches.parquet` e chiama `mondiali ingest` senza `--force`.

- [ ] **Step 2.7: Verifica tutti i test verdi**

```bash
.venv/Scripts/pytest -v
```

Expected: **>= 48 passed** (44 di STEP 1 + 13 Task 1 + 1 Task 2 + 1 leakage = 59 total, scope margine d'errore su parametrizzati già contati).

- [ ] **Step 2.8: Commit**

```bash
git add src/mondiali/data/ingestion.py tests/test_ingestion.py tests/test_leakage.py
git commit -m "feat(data): integrate tier1 features into processed matches pipeline"
```

**🏁 CHECKPOINT CP1 — Tier 1 features disponibili in `matches.parquet`. Anti-leakage su days_rest verde.**

Verifica manuale:

```bash
.venv/Scripts/python -c "
import pandas as pd
df = pd.read_parquet('data/processed/matches.parquet')
print(df[['date','home_team','away_team','competition_importance','days_rest_home','days_rest_away']].tail(10))
print('NaN in days_rest_home:', df['days_rest_home'].isna().sum(), '(atteso ~230, uno per ogni team mai visto)')
"
```

Se NaN troppo elevati (>500) o troppo pochi (<100) investiga prima di proseguire.

---

## Task 3: Walk-forward CV splits

**Files:**
- Create: `src/mondiali/training/splits.py`
- Create: `tests/test_splits.py`

- [ ] **Step 3.1: Test failing**

Crea `tests/test_splits.py`:

```python
"""Test per walk-forward CV splits."""
from __future__ import annotations

import pandas as pd
import pytest

from mondiali.training.splits import walk_forward_splits


def _fake_df(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(dates), "x": range(len(dates))})


def test_walk_forward_splits_produces_expanding_train() -> None:
    """Fold i usa train=[2002-01-01, year_i-12-31], val=[year_i+1, year_i+1-12-31]."""
    dates = [f"{y}-06-15" for y in range(2002, 2019)]
    df = _fake_df(dates)
    folds = list(walk_forward_splits(df, n_folds=3, val_years=1))

    assert len(folds) == 3
    # Fold 1: train 2002-2015, val 2016
    train1, val1 = folds[0]
    assert train1["date"].max().year == 2015
    assert val1["date"].min().year == 2016
    assert val1["date"].max().year == 2016
    # Fold 2: train 2002-2016, val 2017
    train2, val2 = folds[1]
    assert train2["date"].max().year == 2016
    assert val2["date"].min().year == 2017
    # Fold 3
    train3, val3 = folds[2]
    assert train3["date"].max().year == 2017
    assert val3["date"].min().year == 2018


def test_walk_forward_splits_no_overlap_train_val() -> None:
    """Nessuna data di validation cade dentro training."""
    dates = [f"{y}-06-15" for y in range(2002, 2019)]
    df = _fake_df(dates)
    for train, val in walk_forward_splits(df, n_folds=3, val_years=1):
        assert train["date"].max() < val["date"].min()


def test_walk_forward_splits_expands_train() -> None:
    """Il training set cresce fold dopo fold."""
    dates = [f"{y}-06-15" for y in range(2002, 2019)]
    df = _fake_df(dates)
    folds = list(walk_forward_splits(df, n_folds=3, val_years=1))
    sizes = [len(train) for train, _ in folds]
    assert sizes[0] < sizes[1] < sizes[2]
```

- [ ] **Step 3.2: Verifica fallimento**

```bash
.venv/Scripts/pytest tests/test_splits.py -v
```

Expected: 3 fail ImportError.

- [ ] **Step 3.3: Implementa `walk_forward_splits`**

Crea `src/mondiali/training/splits.py`:

```python
"""Walk-forward CV splits (expanding window, mai random).

Conforme a spec §7.2: 3 fold su 2002-2018 con val di 1 anno ciascuno.
"""
from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def walk_forward_splits(
    matches: pd.DataFrame,
    *,
    n_folds: int = 3,
    val_years: int = 1,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Itera `n_folds` fold expanding-window.

    Richiede `matches` con colonna `date` datetime. L'ultimo val window termina
    al `max(date)`; il primo training set inizia al `min(date)` ed espande.

    Args:
        matches: DataFrame con colonna `date`.
        n_folds: numero di fold.
        val_years: ampiezza del validation window per fold, in anni.

    Yields:
        (train_df, val_df) — fold-size crescenti, nessun overlap.
    """
    if matches.empty:
        return
    max_year = matches["date"].dt.year.max()
    for i in range(n_folds):
        val_year_end = max_year - (n_folds - 1 - i) * val_years
        val_year_start = val_year_end - val_years + 1
        train_end = pd.Timestamp(year=val_year_start - 1, month=12, day=31)
        val_start = pd.Timestamp(year=val_year_start, month=1, day=1)
        val_end = pd.Timestamp(year=val_year_end, month=12, day=31)
        train = matches[matches["date"] <= train_end]
        val = matches[(matches["date"] >= val_start) & (matches["date"] <= val_end)]
        yield train, val
```

- [ ] **Step 3.4: Verifica test verdi + lint**

```bash
.venv/Scripts/pytest tests/test_splits.py -v
.venv/Scripts/ruff check src/mondiali/training/splits.py tests/test_splits.py
.venv/Scripts/mypy src/mondiali/training/splits.py
```

Expected: 3 passed, 0 errori.

- [ ] **Step 3.5: Commit**

```bash
git add src/mondiali/training/splits.py tests/test_splits.py
git commit -m "feat(training): walk-forward CV splits"
```

---

## Task 4: Elo-only logistic baseline

**Files:**
- Create: `src/mondiali/model/__init__.py`
- Create: `src/mondiali/model/elo_logistic.py`
- Create: `tests/test_elo_logistic.py`

- [ ] **Step 4.1: Test failing**

Crea `tests/test_elo_logistic.py`:

```python
"""Test del baseline Elo-only logistic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mondiali.model.elo_logistic import EloLogisticBaseline


def _make_df(n: int = 200) -> pd.DataFrame:
    """Genera df sintetico con elo_diff e outcome Bernoulli based on elo."""
    rng = np.random.default_rng(42)
    elo_diff = rng.normal(0, 150, n)
    # P(home_win) cresce con elo_diff
    logits = elo_diff / 200
    p_home = 1 / (1 + np.exp(-logits))
    r = rng.uniform(0, 1, n)
    home_score = np.where(r < p_home * 0.6, 2, np.where(r < p_home * 0.6 + 0.25, 1, 0))
    away_score = np.where(r < p_home * 0.6, 0, np.where(r < p_home * 0.6 + 0.25, 1, 2))
    return pd.DataFrame(
        {
            "home_elo_before": 1500 + elo_diff / 2,
            "away_elo_before": 1500 - elo_diff / 2,
            "neutral": [False] * n,
            "home_score": home_score,
            "away_score": away_score,
        }
    )


def test_fit_learns_positive_elo_diff_coefficient() -> None:
    """Coefficiente su elo_diff positivo → Elo alto vince di più."""
    df = _make_df(500)
    model = EloLogisticBaseline()
    model.fit(df)
    assert model.model_ is not None
    # La colonna "elo_diff" (indice 0) deve avere coefficiente positivo per la classe home_win
    coef_elo_diff = model.model_.coef_[0, 0]  # classe 0 = home_win
    assert coef_elo_diff > 0


def test_predict_proba_shape_and_sum_to_one() -> None:
    """Shape (n, 3), ogni riga somma a 1."""
    df = _make_df(300)
    model = EloLogisticBaseline().fit(df)
    probs = model.predict_proba(df.head(50))
    assert probs.shape == (50, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)


def test_predict_proba_strong_home_favorite_gt_away() -> None:
    """Home con Elo molto più alto in casa → P(home_win) > P(away_win)."""
    df_train = _make_df(500)
    model = EloLogisticBaseline().fit(df_train)
    df_test = pd.DataFrame(
        {
            "home_elo_before": [2000.0],
            "away_elo_before": [1500.0],
            "neutral": [False],
            "home_score": [0],
            "away_score": [0],
        }
    )
    probs = model.predict_proba(df_test)
    assert probs[0, 0] > probs[0, 2]


def test_predict_before_fit_raises() -> None:
    model = EloLogisticBaseline()
    df = _make_df(5)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_proba(df)
```

- [ ] **Step 4.2: Verifica fallimento**

```bash
.venv/Scripts/pytest tests/test_elo_logistic.py -v
```

Expected: 4 fail ImportError.

- [ ] **Step 4.3: Implementa `EloLogisticBaseline`**

Crea `src/mondiali/model/__init__.py`:

```python
"""Modelli Tier 1+."""
```

Crea `src/mondiali/model/elo_logistic.py`:

```python
"""Baseline Elo-only logistic.

Features: [elo_diff, is_neutral_int].
Target: outcome 1/X/2 (0=home, 1=draw, 2=away).

Serve come comparatore obbligatorio per Tier 1: se XGBoost Poisson Tier 1 non
batte questo baseline in log-loss su validation di almeno 0.003, STOP — debug
features/leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mondiali.training.evaluate import compute_outcomes


class EloLogisticBaseline:
    """LogisticRegression multi-classe su [elo_diff, is_neutral_int]."""

    def __init__(self, *, C: float = 1.0, random_state: int = 42) -> None:
        self.C = C
        self.random_state = random_state
        self.model_: LogisticRegression | None = None

    def _design_matrix(self, matches: pd.DataFrame) -> np.ndarray:
        elo_diff = matches["home_elo_before"].to_numpy() - matches["away_elo_before"].to_numpy()
        is_neutral = matches["neutral"].astype(int).to_numpy()
        return np.column_stack([elo_diff, is_neutral])

    def fit(self, matches: pd.DataFrame) -> EloLogisticBaseline:
        """Fit su matches con home_elo_before, away_elo_before, neutral, home_score, away_score."""
        X = self._design_matrix(matches)
        y = compute_outcomes(matches)
        self.model_ = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state,
        ).fit(X, y)
        return self

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("EloLogisticBaseline must be fit() before predict_proba")
        X = self._design_matrix(matches)
        return np.asarray(self.model_.predict_proba(X))
```

- [ ] **Step 4.4: Verifica verde + lint**

```bash
.venv/Scripts/pytest tests/test_elo_logistic.py -v
.venv/Scripts/ruff check src/mondiali/model/ tests/test_elo_logistic.py
.venv/Scripts/mypy src/mondiali/model/elo_logistic.py
```

Expected: 4 passed, 0 errori.

Nota: sklearn in versioni recenti deprecata il parametro `multi_class="multinomial"`. Se vedi `FutureWarning`, rimuovilo — il default di `LogisticRegression` in sklearn ≥1.5 è già multinomial quando lbfgs.

- [ ] **Step 4.5: Commit**

```bash
git add src/mondiali/model/__init__.py src/mondiali/model/elo_logistic.py tests/test_elo_logistic.py
git commit -m "feat(model): Elo-only logistic baseline"
```

---

## Task 5: CLI command `mondiali train elo-logistic` + CP2 log-loss documentato

**Files:**
- Modify: `src/mondiali/cli/main.py`

- [ ] **Step 5.1: Aggiungi comando `train-elo`**

In `src/mondiali/cli/main.py`, aggiungi import e nuovo comando:

```python
from mondiali.model.elo_logistic import EloLogisticBaseline
```

Sotto il comando `baseline`:

```python
@app.command(name="train-elo")
def train_elo(
    train_start: str = typer.Option("2002-01-01"),
    train_end: str = typer.Option("2018-12-31"),
    val_start: str = typer.Option("2019-01-01"),
    val_end: str = typer.Option("2022-06-30"),
) -> None:
    """Fit Elo-only logistic baseline, report log-loss su validation."""
    processed = CONFIG.data_processed / "matches.parquet"
    if not processed.exists():
        typer.echo("matches.parquet non trovato - esegui `mondiali ingest` prima", err=True)
        raise typer.Exit(1)

    df = pd.read_parquet(processed)
    df["date"] = pd.to_datetime(df["date"])
    train = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
    val = df[(df["date"] >= val_start) & (df["date"] <= val_end)]

    typer.echo(f"Train: {len(train)} | Val: {len(val)}")

    model = EloLogisticBaseline().fit(train)
    val_probs = model.predict_proba(val)
    val_loss = log_loss_1x2(val, val_probs)
    typer.echo(f"Elo-only logistic validation log-loss: {val_loss:.4f}")
    assert model.model_ is not None
    coef = model.model_.coef_[0]
    typer.echo(f"Coefficienti: elo_diff={coef[0]:.6f}, is_neutral={coef[1]:.5f}")
```

- [ ] **Step 5.2: Esegui il comando e annota il log-loss**

```bash
.venv/Scripts/mondiali train-elo
```

**Risultato canonico (eseguito 2026-04-25):**
- Train: 16063 | Val: 3215
- `Elo-only logistic validation log-loss: 0.8527`
- Coefficienti: `elo_diff=0.00404, is_neutral=-0.33446`

**`LOGLOSS_ELO = 0.8527`** — questo è il valore di riferimento da qui in avanti. Andrà nel report STEP 2 e nel gate finale di Task 12.

Sanity check (eseguiti):
- Class distribution val 2019-2022: home_win=48.5%, draw=22.5%, away_win=29.0%
- Class-freq baseline log-loss = 1.0457 (predire sempre la marginale)
- elo_diff std=224, max=±1026 — coda lunga di mismatch internazionali
- 34.5% del val ha |elo_diff|>200 — mismatch grossi facili da predire
- Coef × 1σ: 0.00404 × 224 ≈ 0.91 logit → p_home≈0.71 per favorito di 1σ. Coerente.

Il range originale del plan era [0.93, 1.02] ma è stato calibrato pessimisticamente assumendo solo match equilibrati. Il valore reale 0.8527 è plausibile sul dataset international_results (heavy tail di mismatch). Non è leakage.

**Implicazione critica per il gate Tier 1 (Task 12):** target Tier 1 < `LOGLOSS_ELO − 0.003` = **< 0.8497**. Margine stretto. Se il Tier 1 atterra in [0.8497, 0.8527], STOP — debug features.

- [ ] **Step 5.3: Lint + commit**

```bash
.venv/Scripts/ruff check src/mondiali/cli/main.py
.venv/Scripts/mypy src/mondiali/cli/main.py
git add src/mondiali/cli/main.py
git commit -m "feat(cli): train-elo command reporting Elo-only logistic log-loss"
```

**🏁 CHECKPOINT CP2 — Elo-only log-loss `LOGLOSS_ELO` documentato. Target da battere in Task 12.**

---

## Task 6: Symmetric row builder per XGBoost Poisson

**Files:**
- Create: `src/mondiali/model/poisson_xgb.py`
- Create: `tests/test_poisson_xgb.py`

- [ ] **Step 6.1: Test failing per `build_symmetric_rows`**

Crea `tests/test_poisson_xgb.py`:

```python
"""Test symmetric row builder + XGBoost Poisson training."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mondiali.model.poisson_xgb import SYMMETRIC_FEATURES, build_symmetric_rows


def _sample_processed() -> pd.DataFrame:
    """Mini DataFrame con le colonne attese da build_processed_matches."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-10"]),
            "home_team": ["France", "Spain"],
            "away_team": ["Brazil", "France"],
            "home_score": [2, 1],
            "away_score": [1, 1],
            "neutral": [False, True],
            "tournament": ["Friendly", "FIFA World Cup"],
            "home_elo_before": [1900.0, 1850.0],
            "away_elo_before": [1950.0, 1920.0],
            "competition_importance": [1, 4],
            "days_rest_home": [5.0, 30.0],
            "days_rest_away": [7.0, 9.0],
            "days_rest_diff": [-2.0, 21.0],
        }
    )


def test_build_symmetric_rows_doubles_dataframe() -> None:
    """Ogni match → 2 righe (home perspective, away perspective)."""
    df = _sample_processed()
    X, y = build_symmetric_rows(df)
    assert X.shape[0] == 2 * len(df)
    assert y.shape[0] == 2 * len(df)


def test_build_symmetric_rows_targets_are_goals_from_team_perspective() -> None:
    """Riga home-perspective → target = home_score; away-perspective → away_score."""
    df = _sample_processed()
    _, y = build_symmetric_rows(df)
    # Ordine atteso: riga0 home, riga1 away, riga2 home, riga3 away
    assert y.tolist() == [2, 1, 1, 1]


def test_build_symmetric_rows_is_home_flag_alternates() -> None:
    """Colonna is_home = 1 per le righe home-perspective, 0 per away-perspective."""
    df = _sample_processed()
    X, _ = build_symmetric_rows(df)
    is_home_col = SYMMETRIC_FEATURES.index("is_home")
    # Match 0 (non-neutral): home=1, away=0. Match 1 (neutral): home=0, away=0
    # perché il vantaggio casa reale non si applica in venue neutrale.
    assert X[:, is_home_col].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_build_symmetric_rows_flips_elo_per_perspective() -> None:
    """Nella riga home-perspective team_elo = home_elo_before; in away-perspective opposto."""
    df = _sample_processed()
    X, _ = build_symmetric_rows(df)
    team_elo_col = SYMMETRIC_FEATURES.index("team_elo")
    opp_elo_col = SYMMETRIC_FEATURES.index("opponent_elo")
    # Match 0: France vs Brazil, France elo 1900, Brazil elo 1950
    assert X[0, team_elo_col] == 1900.0  # home-perspective: team = France
    assert X[0, opp_elo_col] == 1950.0
    assert X[1, team_elo_col] == 1950.0  # away-perspective: team = Brazil
    assert X[1, opp_elo_col] == 1900.0


def test_build_symmetric_rows_respects_neutral_flag() -> None:
    """is_home in venue neutral è 0 per entrambe le prospettive (no vantaggio casa reale)."""
    df = _sample_processed()
    X, _ = build_symmetric_rows(df)
    is_home_col = SYMMETRIC_FEATURES.index("is_home")
    # Match 1: FIFA WC, neutral=True → entrambe le righe devono avere is_home=0
    assert X[2, is_home_col] == 0.0
    assert X[3, is_home_col] == 0.0
```

- [ ] **Step 6.2: Verifica fallimento**

```bash
.venv/Scripts/pytest tests/test_poisson_xgb.py -v -k "build_symmetric"
```

Expected: 5 fail ImportError.

- [ ] **Step 6.3: Implementa `build_symmetric_rows`**

Crea `src/mondiali/model/poisson_xgb.py`:

```python
"""XGBoost Poisson symmetric single-model per predizione gol.

Ogni match produce 2 righe: una home-perspective (team=home, opp=away,
is_home=1 se non neutral altrimenti 0) e una away-perspective (simmetrica).
Target: gol segnati dal team in quel match.

Conforme a spec §6.1 (symmetric single-model).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

SYMMETRIC_FEATURES: list[str] = [
    "team_elo",
    "opponent_elo",
    "elo_diff_signed",
    "is_home",
    "is_neutral",
    "competition_importance",
    "team_days_rest",
    "opponent_days_rest",
]


def build_symmetric_rows(matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Ritorna (X, y) dove per ogni match crea 2 righe consecutive.

    Ordine righe: [match0_home, match0_away, match1_home, match1_away, ...].

    X shape: (2 * len(matches), len(SYMMETRIC_FEATURES))
    y shape: (2 * len(matches),)
    """
    n = len(matches)
    X = np.empty((2 * n, len(SYMMETRIC_FEATURES)), dtype=float)
    y = np.empty(2 * n, dtype=float)

    home_elo = matches["home_elo_before"].to_numpy(dtype=float)
    away_elo = matches["away_elo_before"].to_numpy(dtype=float)
    neutral = matches["neutral"].astype(bool).to_numpy()
    comp_imp = matches["competition_importance"].to_numpy(dtype=float)
    rest_h = matches["days_rest_home"].to_numpy(dtype=float)
    rest_a = matches["days_rest_away"].to_numpy(dtype=float)
    h_goals = matches["home_score"].to_numpy(dtype=float)
    a_goals = matches["away_score"].to_numpy(dtype=float)

    # Home-perspective rows (indici pari 0, 2, 4, ...)
    X[0::2, 0] = home_elo                               # team_elo
    X[0::2, 1] = away_elo                               # opponent_elo
    X[0::2, 2] = home_elo - away_elo                    # elo_diff_signed
    X[0::2, 3] = (~neutral).astype(float)               # is_home (0 se neutral)
    X[0::2, 4] = neutral.astype(float)                  # is_neutral
    X[0::2, 5] = comp_imp                               # competition_importance
    X[0::2, 6] = rest_h                                 # team_days_rest
    X[0::2, 7] = rest_a                                 # opponent_days_rest
    y[0::2] = h_goals

    # Away-perspective rows (indici dispari 1, 3, 5, ...)
    X[1::2, 0] = away_elo
    X[1::2, 1] = home_elo
    X[1::2, 2] = away_elo - home_elo
    X[1::2, 3] = 0.0  # away in venue non-neutral: is_home=0 già corretto
    X[1::2, 4] = neutral.astype(float)
    X[1::2, 5] = comp_imp
    X[1::2, 6] = rest_a
    X[1::2, 7] = rest_h
    y[1::2] = a_goals

    return X, y
```

- [ ] **Step 6.4: Verifica verde**

```bash
.venv/Scripts/pytest tests/test_poisson_xgb.py -v -k "build_symmetric"
```

Expected: 5 passed.

- [ ] **Step 6.5: Lint + commit**

```bash
.venv/Scripts/ruff check src/mondiali/model/poisson_xgb.py tests/test_poisson_xgb.py
.venv/Scripts/mypy src/mondiali/model/poisson_xgb.py
git add src/mondiali/model/poisson_xgb.py tests/test_poisson_xgb.py
git commit -m "feat(model): symmetric row builder for Poisson XGBoost"
```

---

## Task 7: XGBoost Poisson wrapper (fit + predict_lambda) + CP3

**Files:**
- Modify: `src/mondiali/model/poisson_xgb.py`
- Modify: `tests/test_poisson_xgb.py`

- [ ] **Step 7.1: Test failing per `PoissonXGBModel`**

Aggiungi in `tests/test_poisson_xgb.py`:

```python
from mondiali.model.poisson_xgb import PoissonXGBModel


def test_poisson_xgb_fit_returns_self() -> None:
    df = _sample_processed()
    # Duplica con rng per avere abbastanza dati
    df_big = pd.concat([df] * 100, ignore_index=True)
    model = PoissonXGBModel()
    result = model.fit(df_big)
    assert result is model


def test_poisson_xgb_predict_lambda_positive_and_shape() -> None:
    """predict_lambda ritorna (lambda_home, lambda_away) > 0 per ogni match."""
    df = _sample_processed()
    df_big = pd.concat([df] * 100, ignore_index=True)
    model = PoissonXGBModel().fit(df_big)

    lam_h, lam_a = model.predict_lambda(df)
    assert lam_h.shape == (len(df),)
    assert lam_a.shape == (len(df),)
    assert (lam_h > 0).all()
    assert (lam_a > 0).all()


def test_poisson_xgb_predict_before_fit_raises() -> None:
    df = _sample_processed()
    model = PoissonXGBModel()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_lambda(df)


def test_poisson_xgb_json_serialization_roundtrip(tmp_path) -> None:
    """Serializzazione JSON nativa: dopo save/load predict_lambda è identico."""
    df = _sample_processed()
    df_big = pd.concat([df] * 50, ignore_index=True)
    model = PoissonXGBModel().fit(df_big)
    lam_h_before, lam_a_before = model.predict_lambda(df)

    json_path = tmp_path / "model.json"
    model.save(json_path)

    loaded = PoissonXGBModel()
    loaded.load(json_path)
    lam_h_after, lam_a_after = loaded.predict_lambda(df)
    np.testing.assert_allclose(lam_h_before, lam_h_after, rtol=1e-6)
    np.testing.assert_allclose(lam_a_before, lam_a_after, rtol=1e-6)
```

- [ ] **Step 7.2: Verifica fallimento**

```bash
.venv/Scripts/pytest tests/test_poisson_xgb.py -v -k "poisson_xgb"
```

Expected: 4 fail ImportError.

- [ ] **Step 7.3: Implementa `PoissonXGBModel`**

Aggiungi in `src/mondiali/model/poisson_xgb.py`:

```python
from pathlib import Path

import xgboost as xgb

from mondiali.config import RANDOM_STATE

DEFAULT_PARAMS: dict[str, float | int | str] = {
    "objective": "count:poisson",
    "tree_method": "hist",
    "max_depth": 6,
    "learning_rate": 0.05,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "n_estimators": 2000,
    "random_state": RANDOM_STATE,
    "verbosity": 0,
}


class PoissonXGBModel:
    """Wrapper XGBoost symmetric single-model per predizione λ gol.

    `fit(matches)` costruisce le righe simmetriche e addestra XGBRegressor con
    objective `count:poisson`. `predict_lambda(matches)` ritorna
    (lambda_home, lambda_away) per ogni match.
    """

    def __init__(self, params: dict | None = None) -> None:
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.booster_: xgb.XGBRegressor | None = None

    def fit(
        self,
        matches: pd.DataFrame,
        *,
        early_stopping_val: pd.DataFrame | None = None,
        early_stopping_rounds: int = 50,
    ) -> PoissonXGBModel:
        """Addestra il modello. Se `early_stopping_val` è fornito, early stop."""
        X, y = build_symmetric_rows(matches)
        fit_kwargs: dict = {}
        if early_stopping_val is not None:
            X_val, y_val = build_symmetric_rows(early_stopping_val)
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False
            params = {**self.params, "early_stopping_rounds": early_stopping_rounds}
        else:
            params = self.params
        self.booster_ = xgb.XGBRegressor(**params)
        self.booster_.fit(X, y, **fit_kwargs)
        log.info("poisson_xgb fit done", n_rows=len(X))
        return self

    def predict_lambda(self, matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Ritorna (lambda_home, lambda_away) per ogni match (shape (n,), (n,))."""
        if self.booster_ is None:
            raise RuntimeError("PoissonXGBModel must be fit() before predict_lambda")
        X, _ = build_symmetric_rows(matches)
        preds = self.booster_.predict(X)
        lam_h = preds[0::2]
        lam_a = preds[1::2]
        return lam_h, lam_a

    def save(self, path: Path) -> None:
        """Salva il booster in formato JSON nativo (non pickle)."""
        if self.booster_ is None:
            raise RuntimeError("fit() prima di save()")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster_.save_model(str(path))

    def load(self, path: Path) -> PoissonXGBModel:
        """Carica un booster salvato."""
        self.booster_ = xgb.XGBRegressor(**self.params)
        self.booster_.load_model(str(path))
        return self
```

- [ ] **Step 7.4: Verifica verde**

```bash
.venv/Scripts/pytest tests/test_poisson_xgb.py -v
```

Expected: 9 passed (5 symmetric rows + 4 model).

- [ ] **Step 7.5: Lint + type check**

```bash
.venv/Scripts/ruff check src/mondiali/model/poisson_xgb.py tests/test_poisson_xgb.py
.venv/Scripts/mypy src/mondiali/model/poisson_xgb.py
```

Expected: 0 errori.

- [ ] **Step 7.6: Smoke run su dati reali — verifica λ plausibili**

```bash
.venv/Scripts/python -c "
import pandas as pd
from mondiali.config import CONFIG
from mondiali.model.poisson_xgb import PoissonXGBModel

df = pd.read_parquet(CONFIG.data_processed / 'matches.parquet')
df['date'] = pd.to_datetime(df['date'])
train = df[(df['date'] >= '2002-01-01') & (df['date'] <= '2016-12-31')]
val   = df[(df['date'] >= '2017-01-01') & (df['date'] <= '2018-12-31')]
m = PoissonXGBModel().fit(train, early_stopping_val=val, early_stopping_rounds=50)
lam_h, lam_a = m.predict_lambda(val)
print(f'lambda_home mean={lam_h.mean():.3f} std={lam_h.std():.3f}')
print(f'lambda_away mean={lam_a.mean():.3f} std={lam_a.std():.3f}')
print(f'observed home_goals mean={val[\"home_score\"].mean():.3f}')
print(f'observed away_goals mean={val[\"away_score\"].mean():.3f}')
"
```

Expected (ordine di grandezza):
- `lambda_home mean` ~1.4 ± 0.3
- `lambda_away mean` ~1.1 ± 0.3
- Le medie λ devono essere vicine alle medie osservate sul validation (max 0.2 di discrepanza). Se no → bug in symmetric_rows o Poisson objective.

- [ ] **Step 7.7: Commit**

```bash
git add src/mondiali/model/poisson_xgb.py tests/test_poisson_xgb.py
git commit -m "feat(model): XGBoost Poisson wrapper with symmetric fit and JSON serialization"
```

**🏁 CHECKPOINT CP3 — λ plausibili. Ora abbiamo (λ_home, λ_away) per ogni match di validation. Prossima: joint matrix + Dixon-Coles + markets.**

---

## Task 8: Joint goal probability matrix

**Files:**
- Create: `src/mondiali/model/dixon_coles.py`
- Create: `tests/test_dixon_coles.py`

- [ ] **Step 8.1: Test failing per `joint_matrix`**

Crea `tests/test_dixon_coles.py`:

```python
"""Test Dixon-Coles correction + joint goal matrix."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import poisson

from mondiali.model.dixon_coles import (
    MAX_GOALS,
    dixon_coles_correct,
    estimate_rho_mle,
    joint_matrix,
)


def test_joint_matrix_shape_and_sums_close_to_one() -> None:
    """Shape (MAX_GOALS+1, MAX_GOALS+1), sum ≈ 1 (≥ 0.99 con lam ≤ 3)."""
    m = joint_matrix(lam_home=1.5, lam_away=1.2)
    assert m.shape == (MAX_GOALS + 1, MAX_GOALS + 1)
    assert 0.99 <= m.sum() <= 1.0


def test_joint_matrix_is_outer_product_of_truncated_pmfs() -> None:
    """P(i,j) = pmf_h[i] * pmf_a[j] (pre-correzione DC).

    Test la struttura outer-product senza passare dai marginali: il
    troncamento a MAX_GOALS=10 lascia ~8e-6 di coda Poisson per λ=2,
    incompatibile con un confronto sui marginali a rtol=1e-10. La forma
    outer è invece esatta per costruzione, qui pinzata a precisione macchina.
    """
    m = joint_matrix(lam_home=1.0, lam_away=2.0)
    pmf_h = poisson.pmf(np.arange(MAX_GOALS + 1), mu=1.0)
    pmf_a = poisson.pmf(np.arange(MAX_GOALS + 1), mu=2.0)
    expected = np.outer(pmf_h, pmf_a)
    np.testing.assert_allclose(m, expected, rtol=1e-12)


def test_task9_stubs_raise_not_implemented() -> None:
    """dixon_coles_correct + estimate_rho_mle sono stub Task 9: pinia il
    contratto, evita silent no-op se Task 9 viene mergiato senza implementarli.
    """
    with pytest.raises(NotImplementedError, match="Task 9"):
        dixon_coles_correct(np.zeros((11, 11)), 1.0, 1.0, -0.1)
    with pytest.raises(NotImplementedError, match="Task 9"):
        estimate_rho_mle(object())
```

- [ ] **Step 8.2: Verifica fallimento**

```bash
.venv/Scripts/pytest tests/test_dixon_coles.py -v
```

Expected: collection error (ImportError su `MAX_GOALS`/`joint_matrix`/stub).

- [ ] **Step 8.3: Implementa `joint_matrix` + stub Task 9**

Crea `src/mondiali/model/dixon_coles.py`:

```python
"""Dixon-Coles correction + joint goal matrix.

Pipeline di inference (spec §6.2):
1. Dato (λ_home, λ_away), costruisci `P(i,j) = P(i|λ_h) * P(j|λ_a)` per
   i,j ∈ [0, MAX_GOALS].
2. Applica correzione Dixon-Coles (bassi punteggi).
3. Rinormalizza a somma 1.

ρ stimato via MLE sul training set (funzione separata).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

MAX_GOALS: int = 10


def joint_matrix(lam_home: float, lam_away: float) -> np.ndarray:
    """Matrice P(i,j) = Poisson(i|lam_home) * Poisson(j|lam_away)."""
    pmf_h = poisson.pmf(np.arange(MAX_GOALS + 1), mu=lam_home)
    pmf_a = poisson.pmf(np.arange(MAX_GOALS + 1), mu=lam_away)
    return np.outer(pmf_h, pmf_a)


# Task 9 stubs - full implementation in next task
def dixon_coles_correct(
    matrix: np.ndarray, lam_home: float, lam_away: float, rho: float
) -> np.ndarray:
    raise NotImplementedError("dixon_coles_correct: implemented in Task 9")


def estimate_rho_mle(matches: object) -> float:
    raise NotImplementedError("estimate_rho_mle: implemented in Task 9")
```

NOTA: `scipy.optimize.minimize_scalar` arriva in Task 9, non importarlo qui (ruff F401). Gli stub sono necessari perché il test top-level importa `dixon_coles_correct` ed `estimate_rho_mle`: senza stub la collection di pytest fallisce. Lo stub `estimate_rho_mle(matches: object)` evita il `# type: ignore[no-untyped-def]`.

- [ ] **Step 8.4: Verifica verde**

```bash
.venv/Scripts/pytest tests/test_dixon_coles.py -v -W error
```

Expected: 3 passed (i 2 joint + lo stub-contract test).

---

## Task 9: Dixon-Coles correction + ρ MLE

**Files:**
- Modify: `src/mondiali/model/dixon_coles.py`
- Modify: `tests/test_dixon_coles.py`

- [ ] **Step 9.1: Test failing per `dixon_coles_correct`**

Aggiungi in `tests/test_dixon_coles.py`:

```python
def test_dixon_coles_correct_sum_to_one_after_normalize() -> None:
    """Dopo correzione + rinormalizzazione la matrice somma a 1 esattamente."""
    m_corrected = dixon_coles_correct(
        joint_matrix(1.5, 1.2),
        lam_home=1.5,
        lam_away=1.2,
        rho=-0.1,
    )
    assert m_corrected.sum() == pytest.approx(1.0, abs=1e-10)


def test_dixon_coles_correct_zero_rho_is_identity() -> None:
    """Con ρ=0 la correzione è l'identità (a meno di rinormalizzazione)."""
    m_before = joint_matrix(1.5, 1.2)
    m_before_norm = m_before / m_before.sum()
    m_after = dixon_coles_correct(m_before, 1.5, 1.2, rho=0.0)
    np.testing.assert_allclose(m_after, m_before_norm, rtol=1e-10)


def test_dixon_coles_correct_affects_only_low_score_cells() -> None:
    """La correzione tocca solo (0,0), (0,1), (1,0), (1,1). Cella (5,5) invariata
    a meno di rinormalizzazione uniforme."""
    m_before = joint_matrix(1.5, 1.2)
    m_before_norm = m_before / m_before.sum()
    m_after = dixon_coles_correct(m_before, 1.5, 1.2, rho=-0.1)
    # La cella (5, 5) cambia solo per rinormalizzazione, non per correzione
    # Ratio (5,5) before_norm vs after deve essere = (1 / sum_after_pre_norm)
    ratio_55 = m_after[5, 5] / m_before_norm[5, 5]
    # Tutti i ratio sulle celle non-DC devono essere identici (solo rinormalizzazione)
    high_cells = m_after[2:, 2:] / m_before_norm[2:, 2:]
    np.testing.assert_allclose(high_cells, ratio_55, rtol=1e-10)
```

- [ ] **Step 9.2: Verifica fallimento**

```bash
.venv/Scripts/pytest tests/test_dixon_coles.py -v -k "dixon_coles_correct"
```

Expected: 3 fail ImportError.

- [ ] **Step 9.3: Implementa `dixon_coles_correct`**

Aggiungi in `src/mondiali/model/dixon_coles.py`:

```python
def dixon_coles_correct(
    matrix: np.ndarray,
    lam_home: float,
    lam_away: float,
    rho: float,
) -> np.ndarray:
    """Applica correzione Dixon-Coles (spec §6.2) e rinormalizza a somma 1.

    Correzione solo sulle 4 celle basso-punteggio:
        P(0,0) *= 1 - lam_home * lam_away * rho
        P(0,1) *= 1 + lam_home * rho
        P(1,0) *= 1 + lam_away * rho
        P(1,1) *= 1 - rho

    ρ tipico empirico ≈ -0.1 (correla leggermente 0-0 e 1-1 con excess rispetto
    a indipendenza Poisson).
    """
    m = matrix.copy()
    m[0, 0] *= 1.0 - lam_home * lam_away * rho
    m[0, 1] *= 1.0 + lam_home * rho
    m[1, 0] *= 1.0 + lam_away * rho
    m[1, 1] *= 1.0 - rho
    s = m.sum()
    if s <= 0:
        raise ValueError(f"Dixon-Coles matrix sum <= 0 (rho={rho}): non rinormalizzabile")
    return m / s
```

- [ ] **Step 9.4: Verifica verde**

```bash
.venv/Scripts/pytest tests/test_dixon_coles.py -v -k "dixon_coles_correct"
```

Expected: 3 passed.

- [ ] **Step 9.5: Test failing per `estimate_rho_mle`**

Aggiungi in `tests/test_dixon_coles.py`:

```python
def test_estimate_rho_mle_returns_value_in_range() -> None:
    """ρ stimato su dati sintetici con mild low-score clustering ∈ [-0.3, 0.0]."""
    rng = np.random.default_rng(42)
    n = 1000
    # Genera match con λ ~1.3 ciascuno
    lam_h = np.full(n, 1.3)
    lam_a = np.full(n, 1.1)
    # Gol simulati — aggiungo bias leggero verso 0-0 per avere ρ non nullo
    home_goals = rng.poisson(lam_h)
    away_goals = rng.poisson(lam_a)
    # Inietto 5% di match 0-0 extra
    mask = rng.uniform(size=n) < 0.05
    home_goals[mask] = 0
    away_goals[mask] = 0

    rho = estimate_rho_mle(lam_h, lam_a, home_goals, away_goals)
    assert -0.3 <= rho <= 0.05


def test_estimate_rho_mle_on_independent_poisson_close_to_zero() -> None:
    """Su dati puramente indipendenti Poisson, ρ stimato dovrebbe essere ~0."""
    rng = np.random.default_rng(7)
    n = 2000
    lam_h = np.full(n, 1.3)
    lam_a = np.full(n, 1.1)
    home_goals = rng.poisson(lam_h)
    away_goals = rng.poisson(lam_a)

    rho = estimate_rho_mle(lam_h, lam_a, home_goals, away_goals)
    assert abs(rho) < 0.05
```

- [ ] **Step 9.6: Verifica fallimento**

```bash
.venv/Scripts/pytest tests/test_dixon_coles.py -v -k "estimate_rho"
```

Expected: 2 fail ImportError.

- [ ] **Step 9.7: Implementa `estimate_rho_mle`**

Aggiungi in `src/mondiali/model/dixon_coles.py`:

```python
def _tau(home_goals: int, away_goals: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Fattore di correzione DC per la cella (home_goals, away_goals)."""
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lam_h * lam_a * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lam_h * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + lam_a * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def estimate_rho_mle(
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    *,
    bounds: tuple[float, float] = (-0.3, 0.1),
) -> float:
    """Stima ρ via MLE massimizzando la log-likelihood congiunta.

    Per ogni match: logL_i = log(τ(h_i, a_i, λ_h_i, λ_a_i, ρ))
                         + log(Poisson(h_i | λ_h_i))
                         + log(Poisson(a_i | λ_a_i))

    I termini Poisson non dipendono da ρ: ottimizziamo solo Σ log(τ).
    """
    lh = np.asarray(lam_home, dtype=float)
    la = np.asarray(lam_away, dtype=float)
    hg = np.asarray(home_goals, dtype=int)
    ag = np.asarray(away_goals, dtype=int)

    # Indici delle celle DC-rilevanti (altrove τ=1, log(τ)=0)
    mask00 = (hg == 0) & (ag == 0)
    mask01 = (hg == 0) & (ag == 1)
    mask10 = (hg == 1) & (ag == 0)
    mask11 = (hg == 1) & (ag == 1)

    def neg_log_likelihood(rho: float) -> float:
        total = 0.0
        if mask00.any():
            vals = 1.0 - lh[mask00] * la[mask00] * rho
            if (vals <= 0).any():
                return np.inf
            total += np.log(vals).sum()
        if mask01.any():
            vals = 1.0 + lh[mask01] * rho
            if (vals <= 0).any():
                return np.inf
            total += np.log(vals).sum()
        if mask10.any():
            vals = 1.0 + la[mask10] * rho
            if (vals <= 0).any():
                return np.inf
            total += np.log(vals).sum()
        if mask11.any():
            vals = 1.0 - rho
            if vals <= 0:
                return np.inf
            total += np.log(vals) * mask11.sum()
        return -total

    result = minimize_scalar(neg_log_likelihood, bounds=bounds, method="bounded")
    return float(result.x)
```

- [ ] **Step 9.8: Verifica tutti i test DC verdi**

```bash
.venv/Scripts/pytest tests/test_dixon_coles.py -v
```

Expected: 7 passed.

- [ ] **Step 9.9: Lint + commit**

```bash
.venv/Scripts/ruff check src/mondiali/model/dixon_coles.py tests/test_dixon_coles.py
.venv/Scripts/mypy src/mondiali/model/dixon_coles.py
git add src/mondiali/model/dixon_coles.py tests/test_dixon_coles.py
git commit -m "feat(model): Dixon-Coles correction with rho MLE estimation"
```

---

## Task 10: Markets derivation (1X2, O/U 2.5, BTTS) + CP4

**Files:**
- Create: `src/mondiali/model/markets.py`
- Create: `tests/test_markets.py`

- [ ] **Step 10.1: Test failing**

Crea `tests/test_markets.py`:

```python
"""Test derivazione mercati dal joint goal matrix."""
from __future__ import annotations

import numpy as np
import pytest

from mondiali.model.dixon_coles import MAX_GOALS, joint_matrix
from mondiali.model.markets import (
    prob_1x2,
    prob_btts,
    prob_over_under,
)


def _normalized_joint(lam_h: float, lam_a: float) -> np.ndarray:
    m = joint_matrix(lam_h, lam_a)
    return m / m.sum()


def test_prob_1x2_sums_to_one() -> None:
    """P(1) + P(X) + P(2) = 1."""
    m = _normalized_joint(1.5, 1.2)
    p1, px, p2 = prob_1x2(m)
    assert p1 + px + p2 == pytest.approx(1.0, abs=1e-10)


def test_prob_1x2_home_favorite_has_highest_p1() -> None:
    """λ_home >> λ_away → P(1) > P(2)."""
    m = _normalized_joint(2.5, 0.8)
    p1, _, p2 = prob_1x2(m)
    assert p1 > p2


def test_prob_over_under_complementary() -> None:
    """P(Over 2.5) + P(Under 2.5) = 1."""
    m = _normalized_joint(1.5, 1.2)
    p_over, p_under = prob_over_under(m, threshold=2.5)
    assert p_over + p_under == pytest.approx(1.0, abs=1e-10)


def test_prob_over_under_threshold_monotonic() -> None:
    """P(Over 2.5) > P(Over 3.5)."""
    m = _normalized_joint(1.8, 1.5)
    p_over_25, _ = prob_over_under(m, threshold=2.5)
    p_over_35, _ = prob_over_under(m, threshold=3.5)
    assert p_over_25 > p_over_35


def test_prob_btts_complementary_to_not_btts() -> None:
    """P(BTTS=Yes) + P(BTTS=No) = 1."""
    m = _normalized_joint(1.5, 1.2)
    p_yes, p_no = prob_btts(m)
    assert p_yes + p_no == pytest.approx(1.0, abs=1e-10)


def test_prob_btts_high_lambdas_increases_p_yes() -> None:
    """λ alti per entrambe le squadre → P(BTTS=Yes) alto."""
    m_low = _normalized_joint(0.5, 0.5)
    m_high = _normalized_joint(2.0, 2.0)
    p_yes_low, _ = prob_btts(m_low)
    p_yes_high, _ = prob_btts(m_high)
    assert p_yes_high > p_yes_low
```

- [ ] **Step 10.2: Verifica fallimento**

```bash
.venv/Scripts/pytest tests/test_markets.py -v
```

Expected: 6 fail ImportError.

- [ ] **Step 10.3: Implementa `src/mondiali/model/markets.py`**

```python
"""Derivazione mercati 1X2, O/U 2.5, BTTS dal joint goal matrix.

Input: matrice (MAX_GOALS+1) x (MAX_GOALS+1) normalizzata (somma=1).
- P(1) = Σ_{i>j} P(i,j)
- P(X) = Σ_{i=j} P(i,j)
- P(2) = Σ_{i<j} P(i,j)
- P(Over k) = Σ_{i+j>k} P(i,j)
- P(BTTS=Y) = Σ_{i>0 ∧ j>0} P(i,j)
"""
from __future__ import annotations

import numpy as np


def prob_1x2(joint: np.ndarray) -> tuple[float, float, float]:
    """Ritorna (P(home_win), P(draw), P(away_win))."""
    n = joint.shape[0]
    idx = np.arange(n)
    i_grid, j_grid = np.meshgrid(idx, idx, indexing="ij")
    p_home = float(joint[i_grid > j_grid].sum())
    p_draw = float(joint[i_grid == j_grid].sum())
    p_away = float(joint[i_grid < j_grid].sum())
    return p_home, p_draw, p_away


def prob_over_under(joint: np.ndarray, *, threshold: float = 2.5) -> tuple[float, float]:
    """Ritorna (P(over), P(under)) per `total goals` rispetto a `threshold`."""
    n = joint.shape[0]
    idx = np.arange(n)
    i_grid, j_grid = np.meshgrid(idx, idx, indexing="ij")
    total = i_grid + j_grid
    p_over = float(joint[total > threshold].sum())
    p_under = float(joint[total < threshold].sum())
    return p_over, p_under


def prob_btts(joint: np.ndarray) -> tuple[float, float]:
    """Ritorna (P(BTTS=Yes), P(BTTS=No)).

    BTTS=Yes sse entrambe le squadre segnano almeno 1 gol.
    """
    n = joint.shape[0]
    idx = np.arange(n)
    i_grid, j_grid = np.meshgrid(idx, idx, indexing="ij")
    both_score = (i_grid > 0) & (j_grid > 0)
    p_yes = float(joint[both_score].sum())
    p_no = 1.0 - p_yes
    return p_yes, p_no
```

- [ ] **Step 10.4: Verifica verde**

```bash
.venv/Scripts/pytest tests/test_markets.py -v
```

Expected: 6 passed.

- [ ] **Step 10.5: Lint + commit**

```bash
.venv/Scripts/ruff check src/mondiali/model/markets.py tests/test_markets.py
.venv/Scripts/mypy src/mondiali/model/markets.py
git add src/mondiali/model/markets.py tests/test_markets.py
git commit -m "feat(model): markets derivation (1X2, O/U, BTTS) from joint matrix"
```

**🏁 CHECKPOINT CP4 — Pipeline inference completa: λ → joint → DC → markets. Invarianti ok. Prossimo: integrarla end-to-end per avere log-loss Tier 1.**

---

## Task 11: Training pipeline Tier 1 end-to-end + CLI

**Files:**
- Create: `src/mondiali/training/train.py`
- Create: `tests/test_train_tier1.py`
- Modify: `src/mondiali/cli/main.py`

- [ ] **Step 11.1: Test failing — pipeline end-to-end produce log-loss sensato**

Crea `tests/test_train_tier1.py`:

```python
"""Test pipeline training Tier 1 end-to-end (smoke test)."""
from __future__ import annotations

import pandas as pd
import pytest

from mondiali.config import CONFIG
from mondiali.training.train import train_tier1_pipeline


@pytest.mark.slow
def test_train_tier1_pipeline_produces_reasonable_log_loss() -> None:
    """Smoke test con dati reali: il pipeline completa e produce log-loss ∈ [0.88, 1.02].

    Salta se matches.parquet non esiste.
    """
    parquet = CONFIG.data_processed / "matches.parquet"
    if not parquet.exists():
        pytest.skip("matches.parquet not found")

    result = train_tier1_pipeline(
        parquet_path=parquet,
        train_start="2002-01-01",
        train_end="2016-12-31",
        val_start="2017-01-01",
        val_end="2018-12-31",
    )
    assert 0.88 <= result["val_log_loss_1x2"] <= 1.02
    assert -0.3 <= result["rho"] <= 0.05
    assert 0.8 <= result["lambda_home_mean"] <= 2.0
```

Aggiungi in `pyproject.toml` sezione `[tool.pytest.ini_options]` se non c'è già:

```toml
markers = ["slow: test lenti (training XGBoost su dati reali)"]
```

- [ ] **Step 11.2: Verifica fallimento**

```bash
.venv/Scripts/pytest tests/test_train_tier1.py -v -m slow
```

Expected: fail ImportError.

- [ ] **Step 11.3: Implementa `train_tier1_pipeline`**

Crea `src/mondiali/training/train.py`:

```python
"""Training pipeline Tier 1 end-to-end.

Sequenza:
1. Carica matches.parquet
2. Split train/val per date
3. Addestra PoissonXGBModel (con early stopping su val)
4. Stima ρ Dixon-Coles sul training via MLE
5. Per ogni match di val: costruisci joint → DC correct → markets 1X2
6. Calcola log-loss 1/X/2 + metriche diagnostiche
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from mondiali.model.dixon_coles import dixon_coles_correct, estimate_rho_mle, joint_matrix
from mondiali.model.markets import prob_1x2
from mondiali.model.poisson_xgb import PoissonXGBModel
from mondiali.training.evaluate import log_loss_1x2

log = structlog.get_logger(__name__)


def _compute_1x2_probs(
    lam_h: np.ndarray, lam_a: np.ndarray, rho: float
) -> np.ndarray:
    """Per ogni match, costruisce joint → DC → 1X2. Ritorna shape (n, 3)."""
    n = len(lam_h)
    out = np.empty((n, 3), dtype=float)
    for i in range(n):
        m = joint_matrix(lam_h[i], lam_a[i])
        m = dixon_coles_correct(m, lam_h[i], lam_a[i], rho=rho)
        p1, px, p2 = prob_1x2(m)
        out[i] = (p1, px, p2)
    return out


def train_tier1_pipeline(
    parquet_path: Path,
    *,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pipeline completa. Ritorna un dizionario con metriche + modello.

    Returns:
        dict con chiavi:
        - model: PoissonXGBModel addestrato
        - rho: float (Dixon-Coles stimato)
        - val_log_loss_1x2: float
        - lambda_home_mean, lambda_away_mean: float
        - n_train, n_val: int
    """
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["days_rest_home", "days_rest_away"]).copy()  # escludi prima apparizione

    train = df[(df["date"] >= train_start) & (df["date"] <= train_end)].reset_index(drop=True)
    val = df[(df["date"] >= val_start) & (df["date"] <= val_end)].reset_index(drop=True)

    log.info("tier1 pipeline start", n_train=len(train), n_val=len(val))

    model = PoissonXGBModel(params=model_params)
    model.fit(train, early_stopping_val=val, early_stopping_rounds=50)

    # Stima ρ sul training (no leakage)
    lam_h_tr, lam_a_tr = model.predict_lambda(train)
    rho = estimate_rho_mle(
        lam_h_tr,
        lam_a_tr,
        train["home_score"].to_numpy(),
        train["away_score"].to_numpy(),
    )
    log.info("rho estimated", rho=rho)

    # Inference su validation
    lam_h_va, lam_a_va = model.predict_lambda(val)
    val_probs = _compute_1x2_probs(lam_h_va, lam_a_va, rho=rho)
    val_loss = log_loss_1x2(val, val_probs)

    log.info(
        "tier1 validation",
        log_loss_1x2=val_loss,
        lam_h_mean=float(lam_h_va.mean()),
        lam_a_mean=float(lam_a_va.mean()),
    )

    return {
        "model": model,
        "rho": rho,
        "val_log_loss_1x2": val_loss,
        "lambda_home_mean": float(lam_h_va.mean()),
        "lambda_away_mean": float(lam_a_va.mean()),
        "n_train": len(train),
        "n_val": len(val),
    }
```

- [ ] **Step 11.4: Verifica test smoke verde (lento)**

```bash
.venv/Scripts/pytest tests/test_train_tier1.py -v -m slow
```

Expected: 1 passed (impiega ~2-5 min a training). Se il log-loss è > 1.02, fermati — c'è un bug strutturale nella pipeline.

- [ ] **Step 11.5: Aggiungi CLI `mondiali train-tier1`**

In `src/mondiali/cli/main.py`, aggiungi:

```python
from mondiali.training.train import train_tier1_pipeline
```

Sotto `train-elo`:

```python
@app.command(name="train-tier1")
def train_tier1(
    train_start: str = typer.Option("2002-01-01"),
    train_end: str = typer.Option("2018-12-31"),
    val_start: str = typer.Option("2019-01-01"),
    val_end: str = typer.Option("2022-06-30"),
    save_model: bool = typer.Option(False, "--save", help="Salva il modello in models/tier1/"),
) -> None:
    """Addestra Tier 1 (XGBoost Poisson + Dixon-Coles), report log-loss."""
    parquet = CONFIG.data_processed / "matches.parquet"
    result = train_tier1_pipeline(
        parquet_path=parquet,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
    )
    typer.echo(f"Train: {result['n_train']} | Val: {result['n_val']}")
    typer.echo(f"Dixon-Coles rho: {result['rho']:.4f}")
    typer.echo(f"lambda_home_mean: {result['lambda_home_mean']:.3f} | lambda_away_mean: {result['lambda_away_mean']:.3f}")
    typer.echo(f"Tier 1 validation log-loss (1X2 calibrated by DC only): {result['val_log_loss_1x2']:.4f}")

    if save_model:
        from mondiali.config import CONFIG as C
        out = C.models_dir / "tier1" / "xgb_poisson.json"
        result["model"].save(out)
        typer.echo(f"Model saved: {out}")
```

- [ ] **Step 11.6: Esegui training Tier 1 su full-range e annota**

```bash
.venv/Scripts/mondiali train-tier1
```

Expected (ordine di grandezza):
- `Train: ~16063 | Val: ~3215` (qualche match in meno per days_rest NaN droppati)
- `Dixon-Coles rho: ~-0.08 ± 0.05`
- `lambda_home_mean: ~1.4`, `lambda_away_mean: ~1.1`
- `Tier 1 validation log-loss: ~0.94-0.97` ← **LOGLOSS_TIER1**

**Annota `LOGLOSS_TIER1` per il report. Va confrontato con `LOGLOSS_ELO` del Task 5.**

- [ ] **Step 11.7: Lint + commit**

```bash
.venv/Scripts/ruff check src/mondiali/training/train.py src/mondiali/cli/main.py tests/test_train_tier1.py
.venv/Scripts/mypy src/mondiali/training/train.py src/mondiali/cli/main.py
git add src/mondiali/training/train.py src/mondiali/cli/main.py tests/test_train_tier1.py pyproject.toml
git commit -m "feat(training): Tier 1 XGBoost Poisson + Dixon-Coles pipeline + CLI"
```

---

## Task 12: Gate finale + report validation_step2.md

**Files:**
- Create: `reports/validation_step2.md`

- [ ] **Step 12.1: Calcola delta ELO vs Tier 1 e SHAP feature importance**

```bash
.venv/Scripts/python -c "
import numpy as np
import pandas as pd
from mondiali.config import CONFIG
from mondiali.model.elo_logistic import EloLogisticBaseline
from mondiali.training.evaluate import log_loss_1x2
from mondiali.training.train import train_tier1_pipeline

parquet = CONFIG.data_processed / 'matches.parquet'
df = pd.read_parquet(parquet)
df['date'] = pd.to_datetime(df['date'])
df = df.dropna(subset=['days_rest_home', 'days_rest_away'])

train = df[(df['date'] >= '2002-01-01') & (df['date'] <= '2018-12-31')]
val = df[(df['date'] >= '2019-01-01') & (df['date'] <= '2022-06-30')]

# Elo-only
elo_m = EloLogisticBaseline().fit(train)
elo_probs = elo_m.predict_proba(val)
ll_elo = log_loss_1x2(val, elo_probs)

# Tier 1
res = train_tier1_pipeline(parquet, train_start='2002-01-01', train_end='2018-12-31',
                            val_start='2019-01-01', val_end='2022-06-30')
ll_t1 = res['val_log_loss_1x2']

# Feature importance (gain)
booster = res['model'].booster_
from mondiali.model.poisson_xgb import SYMMETRIC_FEATURES
imp = dict(zip([f'f{i}' for i in range(len(SYMMETRIC_FEATURES))], booster.feature_importances_))
imp_named = sorted(zip(SYMMETRIC_FEATURES, booster.feature_importances_), key=lambda x: -x[1])
print(f'LOGLOSS_ELO  = {ll_elo:.4f}')
print(f'LOGLOSS_TIER1= {ll_t1:.4f}')
print(f'DELTA        = {ll_elo - ll_t1:.4f}  (target: >= 0.003)')
print(f'RHO          = {res[\"rho\"]:.4f}')
print('Feature importance (gain):')
for name, v in imp_named:
    print(f'  {name:25s} {v:.4f}')
"
```

**Riferimento canonico (Task 5, CP2):** `LOGLOSS_ELO = 0.8527` → soglia gate Tier 1 = **`< 0.8497`**. Se la rerun di Step 12.1 sopra produce un `ll_elo` ≠ 0.8527, qualcosa è cambiato nel codice del baseline (verifica prima di confrontare con Tier 1).

Copia l'output. Se `DELTA < 0.003` → **fermati**: Tier 1 non ha battuto Elo-only. Debug:
1. Controlla che `test_leakage.py` sia verde (rerun)
2. Verifica che `days_rest_home/away` non contengano partite future (c'è un test apposito)
3. Ispeziona SHAP: se `elo_diff_signed` domina al 95% → il segnale è solo Elo, XGBoost sta solo ri-rappresentando la logistica. Prova ad aggiungere Tier 2 in STEP 3 prima di concludere.
4. Prova early stopping più aggressivo: `early_stopping_rounds=20`.

Se `DELTA >= 0.003` → Gate passato, prosegui.

- [ ] **Step 12.2: Sanity check — France vs San Marino in campo neutro**

```bash
.venv/Scripts/python -c "
import pandas as pd
from mondiali.config import CONFIG
from mondiali.model.dixon_coles import dixon_coles_correct, joint_matrix
from mondiali.model.markets import prob_1x2
from mondiali.training.train import train_tier1_pipeline

parquet = CONFIG.data_processed / 'matches.parquet'
res = train_tier1_pipeline(parquet, train_start='2002-01-01', train_end='2018-12-31',
                            val_start='2019-01-01', val_end='2022-06-30')
model, rho = res['model'], res['rho']

# Costruisci una riga sintetica: France (Elo ~1970) vs San Marino (Elo ~1000) in campo neutro
df_synth = pd.DataFrame({
    'date': pd.to_datetime(['2022-01-01']),
    'home_team': ['France'],
    'away_team': ['San Marino'],
    'home_score': [0], 'away_score': [0],
    'neutral': [True],
    'tournament': ['FIFA World Cup qualification'],
    'home_elo_before': [1970.0],
    'away_elo_before': [1000.0],
    'competition_importance': [2],
    'days_rest_home': [30.0], 'days_rest_away': [30.0],
    'days_rest_diff': [0.0],
})
lam_h, lam_a = model.predict_lambda(df_synth)
m = joint_matrix(lam_h[0], lam_a[0])
m = dixon_coles_correct(m, lam_h[0], lam_a[0], rho=rho)
p1, px, p2 = prob_1x2(m)
print(f'France vs San Marino (neutral, qualif):')
print(f'  lambda_h={lam_h[0]:.2f}  lambda_a={lam_a[0]:.2f}')
print(f'  P(France) = {p1:.3f}')
print(f'  P(draw)   = {px:.3f}')
print(f'  P(SMR)    = {p2:.3f}')
print('Sanity: P(France) > 0.85 atteso')
assert p1 > 0.85, f'FAIL sanity: P(France)={p1:.3f}'
print('OK')
"
```

Expected: `P(France) > 0.85`. Se no → modello non cattura la differenza Elo massiva; probabile bug o modello troppo poco capiente.

- [ ] **Step 12.3: Scrivi `reports/validation_step2.md`**

Crea `reports/validation_step2.md` sostituendo i valori `<...>` con quelli reali raccolti:

```markdown
# STEP 2 — Tier 1 validation report

**Data**: <YYYY-MM-DD>
**Commit**: <git rev-parse --short HEAD>
**Python**: <python --version>
**XGBoost**: <xgboost.__version__>

## Dataset & Split

- Input: `data/processed/matches.parquet` (~49'215 match)
- Match con days_rest NaN droppati (prima apparizione team): ~<N>
- Training (2002-2018): <N_train>
- Validation (2019 → giugno 2022): <N_val>

## Baseline Elo-only logistic

**Features**: `[elo_diff, is_neutral]`
**Validation log-loss**: `LOGLOSS_ELO = <val>`

Coefficienti appresi:
- `elo_diff`: <coef_elo> (positivo atteso — Elo alto vince di più)
- `is_neutral`: <coef_neu>

## Tier 1 — XGBoost Poisson + Dixon-Coles

**Features (8)**: team_elo, opponent_elo, elo_diff_signed, is_home, is_neutral, competition_importance, team_days_rest, opponent_days_rest.

**Hparams** (hand-tuned, optuna → STEP 3):
- objective=count:poisson, tree_method=hist
- max_depth=6, lr=0.05, n_estimators=2000 + early_stopping(50)
- reg_alpha=0.1, reg_lambda=1.0, min_child_weight=1
- subsample=0.9, colsample_bytree=0.9

**Dixon-Coles ρ stimato** (MLE su training): `<rho>` (atteso ∈ [-0.15, -0.03])

**λ diagnostica** su validation:
- lambda_home_mean: <val>
- lambda_away_mean: <val>
- home_score_mean (osservato): <val>
- away_score_mean (osservato): <val>

**Validation log-loss**: `LOGLOSS_TIER1 = <val>`

## Gate: Tier 1 vs Elo-only

**Delta**: `LOGLOSS_ELO - LOGLOSS_TIER1 = <val>`

Soglia: **≥ 0.003** (spec §7.5 tier-gate).

- [ ] Delta ≥ 0.003 → **gate passato**, si prosegue a STEP 3 (Tier 2 + calibration).
- [ ] Delta < 0.003 → **gate fallito**, debug (vedi checklist Task 12.1).

## Feature importance (gain)

| Feature | Importance |
|---|---|
| <feature_1> | <val> |
| <feature_2> | <val> |
| ... | ... |

Se `elo_diff_signed` domina al >80% → segnale quasi solo da Elo. Tier 2 (form) è il candidato per aggiungere segnale reale in STEP 3.

## Sanity check — Francia vs San Marino (neutral, qualif)

- lambda_home (France): <val>
- lambda_away (San Marino): <val>
- P(France win): <val> (atteso > 0.85)
- P(draw): <val>
- P(SMR win): <val>

## Test suite

```
<output pytest --tb=short>
```

## Lezioni apprese

<2-3 bullet su cosa è risultato sorprendente>

## Decisioni open per STEP 3

- Tier 2 features: quali rolling window (5, 10, entrambi)?
- Escludere Friendly dal training del modello (mantenendoli solo per aggiornare Elo)?
- Calibrazione isotonic: fittarla su quale subset del validation? (split val in val-fit + val-eval, o ricorrere a walk-forward CV?)
- Optuna: 50 o 100 trial? Su quale fold?
```

- [ ] **Step 12.4: Compila il report con i numeri reali**

Sostituisci tutti i `<val>` con l'output dei comandi sopra. Metti le checkbox a `[x]` se gate passato.

Se gate **fallito**: scrivi nel report la sezione "Debug trail" con:
- Delta osservato vs atteso
- Quali feature importance sono sospette
- Piano di debug (rerun anti-leakage, ispezione outliers in days_rest, ecc.)

- [ ] **Step 12.5: Commit report**

```bash
git add reports/validation_step2.md
git commit -m "docs(report): STEP 2 Tier 1 validation report"
```

- [ ] **Step 12.6: Gate finale — tutti i test verdi + lint**

```bash
.venv/Scripts/pytest -v
.venv/Scripts/ruff check src/ tests/
.venv/Scripts/mypy src/
```

Expected: tutti passed, 0 errori. Contatore test atteso:
- STEP 1 totale: 44
- Task 1: +13
- Task 2: +1 (ingestion) +1 (leakage)
- Task 3: +3
- Task 4: +4
- Task 6+7: +9
- Task 8+9: +7
- Task 10: +6
- Task 11: +1 (slow)
- **Totale STEP 2**: ~89 test

- [ ] **Step 12.7: Tag git step2-complete (solo se gate passato)**

```bash
git tag step2-complete -m "STEP 2 Tier 1 completato: LOGLOSS_TIER1=<val>, delta vs Elo-only=<val>"
git log --oneline -20
```

Se gate fallito → **non taggare**, apri sessione di debug dedicata; il report `validation_step2.md` documenta comunque lo stato.

- [ ] **Step 12.8: Handoff a STEP 3**

Apri nuova sessione per il plan di STEP 3 (Tier 2 Form + Calibration + prima passata Optuna).

Contesto da portare:
- `LOGLOSS_TIER1 = <val>` (target da battere con Tier 1+2 calibrated)
- `LOGLOSS_ELO = <val>`
- Delta Tier 1 vs Elo-only = `<val>`
- Feature attuali in matches.parquet: tutte quelle di STEP 1 + competition_importance, days_rest_{home,away,diff}
- Hparams Tier 1 attuali (per warm-start di optuna in STEP 3)
- Dixon-Coles ρ stimato = `<val>`

---

## Recap STEP 2

**Cosa hai in mano alla fine**:
- Baseline Elo-only logistic con log-loss documentato (floor superiore al Tier 0)
- Modello Tier 1: XGBoost Poisson symmetric + Dixon-Coles correction → probabilità 1X2 (grezze, no isotonic ancora)
- Pipeline markets completa: 1X2, Over/Under 2.5, BTTS dal joint goal matrix
- Walk-forward CV splits pronti per uso in STEP 3
- CLI aggiornata: `mondiali train-elo`, `mondiali train-tier1`
- Anti-leakage framework esteso: `days_rest` strict pre-match
- Report `validation_step2.md` archiviato con log-loss, ρ, feature importance, sanity

**Cosa NON hai (va in STEP 3+)**:
- Tier 2 form features (rolling 5-match W/D/L, GD, goals scored/conceded)
- Isotonic calibration post-hoc
- Optuna hyperparam search
- Tier 3 Transfermarkt
- Tier 4 infortuni

**Tempo stimato STEP 2**: 15-20h (training XGBoost locale è il collo di bottiglia principale).
