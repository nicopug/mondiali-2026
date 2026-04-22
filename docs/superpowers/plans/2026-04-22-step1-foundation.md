# STEP 1 — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mettere a terra lo scaffolding del progetto `mondiali`, scaricare i risultati internazionali 1872-oggi, costruire il sistema Elo custom con K variabile, validare la correttezza via test sanity (Francia fine 2018), implementare il framework anti-data-leakage, e produrre il baseline Tier 0 (prior costante) con log-loss documentato come floor di riferimento.

**Architecture:** Package Python installabile in editable mode (`pip install -e .`). Codice in `src/mondiali/`, test in `tests/`. Moduli disaccoppiati (data, features, training, cli). Zero DB: tutti i dati in file Parquet/CSV. Classe `EloSystem` con K-factor variabile per competizione; history completa salvata in `team_elo_history.parquet`. `PriorBaseline` per Tier 0 (predice sempre le frequenze storiche 1/X/2 del training set).

**Tech Stack:** Python 3.11+, `hatchling` build backend, `pandas`/`pyarrow`, `pydantic` v2, `typer`, `structlog`, `pytest`, `pytest-xdist`, `ruff`, `mypy`, `requests`.

**Spec di riferimento:** `docs/superpowers/specs/2026-04-20-mondiali-prediction-design.md` (STEP 1, sezioni 2-5, 7, 8).

**Gate STEP 1** (dallo spec): test tutti verdi, log-loss prior documentato in `reports/validation_step1.md`, Elo Francia fine 2018 in range di sanity.

## Scope note — cosa NON c'è in questo plan

Lo spec STEP 1 cita "snapshot FIFA ranking" tra le task di ingest. È **escluso** da questo plan perché FIFA ranking non appare nelle feature di Tier 0 né Tier 1 (sezione 5 dello spec). Se servirà come feature comparativa all'Elo custom, verrà aggiunta nel plan di STEP 2 o successivi.

Esplicitamente fuori scope qui: XGBoost, Dixon-Coles, market derivation, walk-forward CV, optuna, form features, Transfermarkt, injuries. Tutti in plan successivi.

---

## File Structure

**Creati in STEP 1** (ordine di introduzione):

```
progetto_mondiali/
├── pyproject.toml                              # Task 1
├── README.md                                   # Task 1
├── CLAUDE.md                                   # Task 1 (istruzioni per Claude Code nel repo)
├── .env.example                                # Task 1
├── src/mondiali/
│   ├── __init__.py                             # Task 1
│   ├── config.py                               # Task 2 (paths, K_FACTORS, HOME_ADVANTAGE)
│   ├── data/
│   │   ├── __init__.py                         # Task 3
│   │   └── ingestion.py                        # Task 3-4 (download + parsing international_results)
│   ├── features/
│   │   ├── __init__.py                         # Task 5
│   │   └── elo.py                              # Task 5-9 (EloSystem)
│   ├── training/
│   │   ├── __init__.py                         # Task 12
│   │   ├── baseline_prior.py                   # Task 12 (PriorBaseline Tier 0)
│   │   └── evaluate.py                         # Task 13 (log_loss 1X2)
│   └── cli/
│       ├── __init__.py                         # Task 14
│       └── main.py                             # Task 14 (typer entry)
├── tests/
│   ├── __init__.py                             # Task 1
│   ├── test_config.py                          # Task 2
│   ├── test_ingestion.py                       # Task 3-4
│   ├── test_elo.py                             # Task 5-10
│   ├── test_leakage.py                         # Task 11 (framework anti-leakage)
│   ├── test_baseline_prior.py                  # Task 12
│   └── test_evaluate.py                        # Task 13
├── data/
│   ├── raw/.gitkeep                            # Task 1 (directory gitignored ma placeholder tracked)
│   ├── processed/.gitkeep                      # Task 1
│   └── manual/.gitkeep                         # Task 1
├── models/.gitkeep                             # Task 1
└── reports/
    ├── .gitkeep                                # Task 1
    └── validation_step1.md                     # Task 15 (output finale)
```

**Nota**: `.gitignore` è già committato nella root commit, non viene modificato qui.

---

## Task 1: Project scaffolding + package installabile

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `CLAUDE.md`
- Create: `.env.example`
- Create: `src/mondiali/__init__.py`
- Create: `tests/__init__.py`
- Create: `data/raw/.gitkeep`, `data/processed/.gitkeep`, `data/manual/.gitkeep`, `models/.gitkeep`, `reports/.gitkeep`

- [ ] **Step 1.1: Creare `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "mondiali"
version = "0.1.0"
description = "World Cup 2026 prediction system"
readme = "README.md"
requires-python = ">=3.11"
authors = [{ name = "Nicolò" }]
dependencies = [
    "xgboost>=2.0",
    "pandas>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "scipy>=1.11",
    "optuna>=3.4",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "typer>=0.9",
    "requests>=2.31",
    "beautifulsoup4>=4.12",
    "pyarrow>=14.0",
    "structlog>=23.2",
    "shap>=0.44",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-xdist>=3.5",
    "pytest-cov>=4.1",
    "ruff>=0.1",
    "mypy>=1.7",
    "types-requests",
    "ipython",
    "jupyter",
]

[project.scripts]
mondiali = "mondiali.cli.main:app"

[tool.hatch.build.targets.wheel]
packages = ["src/mondiali"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "SIM", "RET", "PL"]
ignore = ["PLR0913", "PLR2004"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = ["xgboost.*", "shap.*", "optuna.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
addopts = "-ra -q --strict-markers"
```

- [ ] **Step 1.2: Creare `README.md`**

```markdown
# mondiali

World Cup 2026 prediction system.

Stato: STEP 1 — Foundation.
Documentazione: `docs/superpowers/specs/2026-04-20-mondiali-prediction-design.md`.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate  # Windows; source .venv/bin/activate su Unix
pip install -e ".[dev]"
```

## Uso

```bash
mondiali --help
```

## Test

```bash
pytest
```
```

- [ ] **Step 1.3: Creare `CLAUDE.md`**

```markdown
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
```

- [ ] **Step 1.4: Creare `.env.example`**

```
# Placeholder per future API keys (NewsAPI in STEP 9, ecc.). Copia in .env e compila.
# NEWSAPI_KEY=
# ANTHROPIC_API_KEY=
```

- [ ] **Step 1.5: Creare file `__init__.py` vuoti e `.gitkeep`**

Crea questi file (tutti con contenuto vuoto o `# mondiali package` nel caso dei `__init__.py`):

- `src/mondiali/__init__.py` con contenuto: `"""World Cup 2026 prediction system."""`
- `tests/__init__.py` vuoto
- `data/raw/.gitkeep` vuoto
- `data/processed/.gitkeep` vuoto
- `data/manual/.gitkeep` vuoto
- `models/.gitkeep` vuoto
- `reports/.gitkeep` vuoto

- [ ] **Step 1.6: Creare virtualenv e installare**

```bash
python -m venv .venv
.venv/Scripts/pip install --upgrade pip
.venv/Scripts/pip install -e ".[dev]"
```

Expected: installazione completa senza errori, binario `mondiali` presente in `.venv/Scripts/`.

- [ ] **Step 1.7: Verificare import del package**

```bash
.venv/Scripts/python -c "import mondiali; print(mondiali.__doc__)"
```

Expected output: `World Cup 2026 prediction system.`

- [ ] **Step 1.8: Verificare pytest e ruff funzionano**

```bash
.venv/Scripts/pytest --version
.venv/Scripts/ruff --version
.venv/Scripts/mypy --version
```

Expected: tutti rispondono con la loro versione, nessun errore.

- [ ] **Step 1.9: Aggiungere `.venv` al gitignore (se non c'è già)**

Verifica che `.gitignore` contiene `.venv/`. Il root commit include già questa regola, ma doublecheck.

```bash
grep "\.venv" .gitignore
```

Expected: match trovato. Se no, aggiungerlo.

- [ ] **Step 1.10: Commit**

```bash
git add pyproject.toml README.md CLAUDE.md .env.example src/ tests/ data/ models/ reports/
git commit -m "chore: project scaffolding + installable package"
```

---

## Task 2: Config module con pydantic

**Files:**
- Create: `src/mondiali/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 2.1: Scrivere il test failing**

Crea `tests/test_config.py`:

```python
"""Test del modulo config: paths, K-factors, home advantage."""
from pathlib import Path

from mondiali.config import CONFIG, K_FACTORS, HOME_ADVANTAGE


def test_paths_are_absolute_and_project_scoped() -> None:
    """I path della config devono essere risolti e puntare dentro la project root."""
    assert CONFIG.data_raw.is_absolute()
    assert CONFIG.data_processed.is_absolute()
    assert CONFIG.models_dir.is_absolute()
    assert CONFIG.reports_dir.is_absolute()
    project_root = Path(__file__).parent.parent.resolve()
    for p in [CONFIG.data_raw, CONFIG.data_processed, CONFIG.models_dir, CONFIG.reports_dir]:
        assert str(p).startswith(str(project_root)), f"{p} is outside project root"


def test_k_factors_cover_all_tournament_categories() -> None:
    """K-factors devono coprire World Cup, continental, qualification, friendly, default."""
    assert K_FACTORS["world_cup"] == 60
    assert K_FACTORS["continental"] == 50
    assert K_FACTORS["qualification"] == 40
    assert K_FACTORS["friendly"] == 20
    assert K_FACTORS["default"] == 30


def test_home_advantage_standard_value() -> None:
    """Home advantage deve essere il valore standard eloratings.net."""
    assert HOME_ADVANTAGE == 65
```

- [ ] **Step 2.2: Verificare che il test fallisce**

```bash
.venv/Scripts/pytest tests/test_config.py -v
```

Expected: FAIL con `ModuleNotFoundError: No module named 'mondiali.config'`.

- [ ] **Step 2.3: Implementare `config.py`**

Crea `src/mondiali/config.py`:

```python
"""Configurazione globale del progetto mondiali.

Paths, costanti Elo, e parametri condivisi. Tutti i path sono risolti rispetto
alla project root (una directory sopra `src/`).
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Paths(BaseModel):
    """Filesystem paths del progetto."""

    project_root: Path = Field(default_factory=lambda: _PROJECT_ROOT)
    data_raw: Path = Field(default_factory=lambda: _PROJECT_ROOT / "data" / "raw")
    data_processed: Path = Field(default_factory=lambda: _PROJECT_ROOT / "data" / "processed")
    data_manual: Path = Field(default_factory=lambda: _PROJECT_ROOT / "data" / "manual")
    models_dir: Path = Field(default_factory=lambda: _PROJECT_ROOT / "models")
    reports_dir: Path = Field(default_factory=lambda: _PROJECT_ROOT / "reports")


CONFIG = Paths()


# K-factor per aggiornamento Elo, variabile per importanza competizione.
# Valori allineati allo standard eloratings.net.
K_FACTORS: dict[str, int] = {
    "world_cup": 60,
    "continental": 50,
    "qualification": 40,
    "friendly": 20,
    "default": 30,
}

# Home advantage in punti Elo (sommato al rating casa nel calcolo expected).
# Azzerato quando is_neutral_venue=True.
HOME_ADVANTAGE: int = 65

# Random state globale per riproducibilità.
RANDOM_STATE: int = 42
```

- [ ] **Step 2.4: Verificare che il test passa**

```bash
.venv/Scripts/pytest tests/test_config.py -v
```

Expected: 3 passed.

- [ ] **Step 2.5: Lint + type check**

```bash
.venv/Scripts/ruff check src/mondiali/config.py tests/test_config.py
.venv/Scripts/mypy src/mondiali/config.py
```

Expected: nessun errore in ruff; mypy 0 errori.

- [ ] **Step 2.6: Commit**

```bash
git add src/mondiali/config.py tests/test_config.py
git commit -m "feat(config): paths, K-factors, home advantage"
```

---

## Task 3: Data ingestion — download results.csv

**Files:**
- Create: `src/mondiali/data/__init__.py`
- Create: `src/mondiali/data/ingestion.py`
- Create: `tests/test_ingestion.py`

- [ ] **Step 3.1: Scrivere il test failing per `download_international_results`**

Crea `tests/test_ingestion.py`:

```python
"""Test per data ingestion: download e parsing di international_results."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mondiali.data.ingestion import (
    INTERNATIONAL_RESULTS_URL,
    download_international_results,
    load_international_results,
)


def test_download_writes_csv_to_destination(tmp_path: Path) -> None:
    """Il download scrive il CSV alla destinazione specificata."""
    dest = tmp_path / "results.csv"
    fake_csv = b"date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n1872-11-30,Scotland,England,0,0,Friendly,Glasgow,Scotland,FALSE\n"

    with patch("mondiali.data.ingestion.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.content = fake_csv
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result_path = download_international_results(dest)

    assert result_path == dest
    assert dest.exists()
    assert dest.read_bytes() == fake_csv
    mock_get.assert_called_once_with(INTERNATIONAL_RESULTS_URL, timeout=60)


def test_download_skips_if_file_exists_and_force_false(tmp_path: Path) -> None:
    """Se il file esiste e force=False, non ri-scarica."""
    dest = tmp_path / "results.csv"
    dest.write_bytes(b"existing content")

    with patch("mondiali.data.ingestion.requests.get") as mock_get:
        download_international_results(dest, force=False)

    mock_get.assert_not_called()
    assert dest.read_bytes() == b"existing content"


def test_download_raises_on_http_error(tmp_path: Path) -> None:
    """HTTP error propagato correttamente."""
    dest = tmp_path / "results.csv"

    with patch("mondiali.data.ingestion.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = RuntimeError("HTTP 500")
        mock_get.return_value = mock_response

        with pytest.raises(RuntimeError, match="HTTP 500"):
            download_international_results(dest)
```

- [ ] **Step 3.2: Verificare che il test fallisce**

```bash
.venv/Scripts/pytest tests/test_ingestion.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'mondiali.data.ingestion'`.

- [ ] **Step 3.3: Implementare `data/__init__.py` e `data/ingestion.py` (download only)**

Crea `src/mondiali/data/__init__.py`:

```python
"""Data ingestion e storage layer."""
```

Crea `src/mondiali/data/ingestion.py`:

```python
"""Download e parsing del dataset `martj42/international_results`."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
import structlog

log = structlog.get_logger(__name__)

INTERNATIONAL_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)


def download_international_results(dest: Path, *, force: bool = False) -> Path:
    """Scarica `results.csv` in `dest`. Se esiste e `force=False`, salta il download.

    Args:
        dest: percorso del file CSV di destinazione.
        force: se True, ri-scarica anche se già presente.

    Returns:
        il path `dest`.

    Raises:
        qualsiasi eccezione propagata da `requests` (HTTPError, ConnectionError, ecc.).
    """
    if dest.exists() and not force:
        log.info("results.csv already present, skipping download", path=str(dest))
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading international_results", url=INTERNATIONAL_RESULTS_URL)
    response = requests.get(INTERNATIONAL_RESULTS_URL, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    log.info("downloaded", path=str(dest), size_bytes=len(response.content))
    return dest


def load_international_results(csv_path: Path) -> pd.DataFrame:
    """Placeholder — implementato in Task 4."""
    raise NotImplementedError("load_international_results implementato in Task 4")
```

- [ ] **Step 3.4: Verificare che i test download passano**

```bash
.venv/Scripts/pytest tests/test_ingestion.py -v -k "download"
```

Expected: 3 passed (i 3 test che iniziano con `test_download_`).

- [ ] **Step 3.5: Lint + type check**

```bash
.venv/Scripts/ruff check src/mondiali/data/
.venv/Scripts/mypy src/mondiali/data/ingestion.py
```

Expected: 0 errori.

- [ ] **Step 3.6: Commit**

```bash
git add src/mondiali/data/__init__.py src/mondiali/data/ingestion.py tests/test_ingestion.py
git commit -m "feat(data): download international_results CSV with caching"
```

---

## Task 4: Data ingestion — parsing e schema normalization

**Files:**
- Modify: `src/mondiali/data/ingestion.py`
- Modify: `tests/test_ingestion.py`

- [ ] **Step 4.1: Scrivere il test failing per `load_international_results`**

Aggiungi in fondo a `tests/test_ingestion.py`:

```python
def test_load_parses_dates_and_normalizes_columns(tmp_path: Path) -> None:
    """Parsing del CSV produce DataFrame con date pandas e tipi coerenti."""
    csv = tmp_path / "results.csv"
    csv.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2018-07-15,France,Croatia,4,2,FIFA World Cup,Moscow,Russia,TRUE\n"
        "2022-12-18,Argentina,France,3,3,FIFA World Cup,Lusail,Qatar,TRUE\n"
    )

    df = load_international_results(csv)

    assert list(df.columns) == [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    ]
    assert df["date"].dtype == "datetime64[ns]"
    assert df["home_score"].dtype == "int64"
    assert df["away_score"].dtype == "int64"
    assert df["neutral"].dtype == "bool"
    assert len(df) == 2
    assert df.iloc[0]["home_team"] == "France"
    assert df.iloc[0]["neutral"] is True or df.iloc[0]["neutral"] == True  # noqa: E712


def test_load_drops_rows_with_missing_scores(tmp_path: Path) -> None:
    """Match senza punteggio (future o cancellati) sono droppati."""
    csv = tmp_path / "results.csv"
    csv.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2018-07-15,France,Croatia,4,2,FIFA World Cup,Moscow,Russia,TRUE\n"
        "2026-06-11,Mexico,USA,,,FIFA World Cup,,USA,FALSE\n"
    )

    df = load_international_results(csv)

    assert len(df) == 1
    assert df.iloc[0]["home_team"] == "France"


def test_load_sorts_by_date_ascending(tmp_path: Path) -> None:
    """Rows ordinate per data crescente."""
    csv = tmp_path / "results.csv"
    csv.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2022-12-18,Argentina,France,3,3,FIFA World Cup,Lusail,Qatar,TRUE\n"
        "2018-07-15,France,Croatia,4,2,FIFA World Cup,Moscow,Russia,TRUE\n"
    )

    df = load_international_results(csv)

    assert df.iloc[0]["date"] < df.iloc[1]["date"]
```

- [ ] **Step 4.2: Verificare che i test load falliscono**

```bash
.venv/Scripts/pytest tests/test_ingestion.py -v -k "load"
```

Expected: 3 failures con `NotImplementedError`.

- [ ] **Step 4.3: Implementare `load_international_results`**

Sostituisci il body di `load_international_results` in `src/mondiali/data/ingestion.py`:

```python
def load_international_results(csv_path: Path) -> pd.DataFrame:
    """Carica `results.csv` con schema normalizzato.

    - Parse delle date in `datetime64[ns]`.
    - Cast `neutral` da stringa 'TRUE'/'FALSE' a bool.
    - Droppa righe con `home_score` o `away_score` mancanti (match futuri/cancellati).
    - Ordina per data crescente.

    Args:
        csv_path: path del CSV scaricato.

    Returns:
        DataFrame pronto per feature engineering.
    """
    df = pd.read_csv(csv_path, dtype={"neutral": "string"})
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype("int64")
    df["away_score"] = df["away_score"].astype("int64")
    df["neutral"] = df["neutral"].str.upper().map({"TRUE": True, "FALSE": False}).astype("bool")
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    log.info("loaded international_results", rows=len(df))
    return df
```

- [ ] **Step 4.4: Verificare che tutti i test ingestion passano**

```bash
.venv/Scripts/pytest tests/test_ingestion.py -v
```

Expected: 6 passed.

- [ ] **Step 4.5: Lint + type check**

```bash
.venv/Scripts/ruff check src/mondiali/data/ tests/test_ingestion.py
.venv/Scripts/mypy src/mondiali/data/ingestion.py
```

Expected: 0 errori.

- [ ] **Step 4.6: Commit**

```bash
git add src/mondiali/data/ingestion.py tests/test_ingestion.py
git commit -m "feat(data): parse international_results with schema normalization"
```

---

## Task 5: Elo module — `EloSystem` init a 1500

**Files:**
- Create: `src/mondiali/features/__init__.py`
- Create: `src/mondiali/features/elo.py`
- Create: `tests/test_elo.py`

- [ ] **Step 5.1: Scrivere il test failing per init**

Crea `tests/test_elo.py`:

```python
"""Test del sistema Elo custom."""
from __future__ import annotations

import pandas as pd
import pytest

from mondiali.features.elo import DEFAULT_ELO, EloSystem, classify_tournament


def test_elo_system_initializes_teams_at_default() -> None:
    """Team mai visto restituisce DEFAULT_ELO (1500)."""
    elo = EloSystem()
    assert elo.get("France") == DEFAULT_ELO
    assert elo.get("San Marino") == DEFAULT_ELO
    assert DEFAULT_ELO == 1500
```

- [ ] **Step 5.2: Verificare che il test fallisce**

```bash
.venv/Scripts/pytest tests/test_elo.py -v
```

Expected: FAIL con `ModuleNotFoundError: No module named 'mondiali.features.elo'`.

- [ ] **Step 5.3: Implementare `features/__init__.py` e skeleton `features/elo.py`**

Crea `src/mondiali/features/__init__.py`:

```python
"""Feature engineering modules."""
```

Crea `src/mondiali/features/elo.py`:

```python
"""Sistema Elo custom per squadre nazionali.

K-factor variabile per importanza competizione (vedi `config.K_FACTORS`),
home advantage standard a 65 punti (zero per venue neutral).

Conforme allo spec sezione 4.4.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import structlog

from mondiali.config import HOME_ADVANTAGE, K_FACTORS

log = structlog.get_logger(__name__)

DEFAULT_ELO: int = 1500


def classify_tournament(tournament: str) -> str:
    """Mappa il nome del torneo alle categorie di K-factor.

    Returns one of: 'world_cup', 'continental', 'qualification', 'friendly', 'default'.
    """
    raise NotImplementedError  # Task 7


@dataclass
class EloSystem:
    """Elo storico in-memory. `get(team)` restituisce il rating corrente."""

    ratings: dict[str, float] = field(default_factory=dict)

    def get(self, team: str) -> float:
        """Rating corrente di `team`; DEFAULT_ELO se mai visto."""
        return self.ratings.get(team, float(DEFAULT_ELO))
```

- [ ] **Step 5.4: Verificare che il test init passa**

```bash
.venv/Scripts/pytest tests/test_elo.py::test_elo_system_initializes_teams_at_default -v
```

Expected: 1 passed.

- [ ] **Step 5.5: Commit**

```bash
git add src/mondiali/features/__init__.py src/mondiali/features/elo.py tests/test_elo.py
git commit -m "feat(elo): EloSystem skeleton with default rating 1500"
```

---

## Task 6: Elo — update dopo un singolo match

**Files:**
- Modify: `src/mondiali/features/elo.py`
- Modify: `tests/test_elo.py`

- [ ] **Step 6.1: Scrivere i test failing per update**

Aggiungi in `tests/test_elo.py`:

```python
def test_elo_update_home_win_zero_sum() -> None:
    """Dopo una vittoria casa, la somma dei rating si conserva (zero-sum)."""
    elo = EloSystem()
    elo.update(home="France", away="Brazil", home_goals=2, away_goals=0, k_factor=30, neutral=False)
    total = elo.get("France") + elo.get("Brazil")
    assert total == pytest.approx(2 * DEFAULT_ELO, abs=0.01)


def test_elo_update_home_win_increases_home_rating() -> None:
    """Vittoria casa → rating casa aumenta, ospite diminuisce."""
    elo = EloSystem()
    elo.update(home="France", away="Brazil", home_goals=2, away_goals=0, k_factor=30, neutral=False)
    assert elo.get("France") > DEFAULT_ELO
    assert elo.get("Brazil") < DEFAULT_ELO


def test_elo_update_draw_between_equal_teams_neutral_no_change() -> None:
    """Pareggio tra squadre di pari Elo in venue neutral → nessun cambiamento."""
    elo = EloSystem()
    elo.update(home="A", away="B", home_goals=1, away_goals=1, k_factor=30, neutral=True)
    assert elo.get("A") == pytest.approx(DEFAULT_ELO, abs=0.01)
    assert elo.get("B") == pytest.approx(DEFAULT_ELO, abs=0.01)


def test_elo_update_away_win_increases_away_rating() -> None:
    """Vittoria trasferta → rating ospite aumenta."""
    elo = EloSystem()
    elo.update(home="A", away="B", home_goals=0, away_goals=3, k_factor=30, neutral=False)
    assert elo.get("B") > DEFAULT_ELO
    assert elo.get("A") < DEFAULT_ELO


def test_elo_update_magnitude_proportional_to_k() -> None:
    """K=60 produce delta doppio rispetto a K=30 a parità di condizioni."""
    elo_a = EloSystem()
    elo_b = EloSystem()
    elo_a.update(home="X", away="Y", home_goals=1, away_goals=0, k_factor=30, neutral=True)
    elo_b.update(home="X", away="Y", home_goals=1, away_goals=0, k_factor=60, neutral=True)
    delta_a = elo_a.get("X") - DEFAULT_ELO
    delta_b = elo_b.get("X") - DEFAULT_ELO
    assert delta_b == pytest.approx(2 * delta_a, abs=0.01)
```

- [ ] **Step 6.2: Verificare i test falliscono**

```bash
.venv/Scripts/pytest tests/test_elo.py -v -k "update"
```

Expected: 5 failures con `AttributeError: 'EloSystem' object has no attribute 'update'`.

- [ ] **Step 6.3: Implementare `update` method**

Aggiungi in `src/mondiali/features/elo.py` dentro la classe `EloSystem`:

```python
    def update(
        self,
        *,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
        k_factor: float,
        neutral: bool,
    ) -> tuple[float, float]:
        """Applica l'update Elo per un singolo match. Zero-sum.

        Formula:
            expected_home = 1 / (1 + 10^((elo_away - elo_home_adj) / 400))
            dove elo_home_adj = elo_home + (HOME_ADVANTAGE if not neutral else 0)
            score_home = 1 if home_goals > away_goals, 0.5 if tie, 0 otherwise
            delta = k_factor * (score_home - expected_home)
            elo_home_new = elo_home + delta
            elo_away_new = elo_away - delta

        Args:
            home, away: nomi squadre.
            home_goals, away_goals: gol segnati.
            k_factor: K per questa partita.
            neutral: True se venue neutrale (disattiva home advantage).

        Returns:
            (elo_home_pre, elo_away_pre) — i rating PRIMA dell'update (utile per
            snapshot per il match stesso, dove serve il pre-match).
        """
        elo_h = self.get(home)
        elo_a = self.get(away)

        adv = 0.0 if neutral else float(HOME_ADVANTAGE)
        expected_home = 1.0 / (1.0 + 10.0 ** ((elo_a - (elo_h + adv)) / 400.0))

        if home_goals > away_goals:
            score_home = 1.0
        elif home_goals < away_goals:
            score_home = 0.0
        else:
            score_home = 0.5

        delta = k_factor * (score_home - expected_home)
        self.ratings[home] = elo_h + delta
        self.ratings[away] = elo_a - delta
        return elo_h, elo_a
```

- [ ] **Step 6.4: Verificare i test update passano**

```bash
.venv/Scripts/pytest tests/test_elo.py -v -k "update"
```

Expected: 5 passed.

- [ ] **Step 6.5: Lint + type check**

```bash
.venv/Scripts/ruff check src/mondiali/features/elo.py tests/test_elo.py
.venv/Scripts/mypy src/mondiali/features/elo.py
```

Expected: 0 errori.

- [ ] **Step 6.6: Commit**

```bash
git add src/mondiali/features/elo.py tests/test_elo.py
git commit -m "feat(elo): zero-sum update with home advantage"
```

---

## Task 7: Elo — K-factor classification per tournament

**Files:**
- Modify: `src/mondiali/features/elo.py`
- Modify: `tests/test_elo.py`

- [ ] **Step 7.1: Scrivere i test failing per `classify_tournament`**

Aggiungi in `tests/test_elo.py`:

```python
@pytest.mark.parametrize(
    ("tournament", "expected"),
    [
        ("FIFA World Cup", "world_cup"),
        ("FIFA World Cup qualification", "qualification"),
        ("UEFA Euro", "continental"),
        ("UEFA Euro qualification", "qualification"),
        ("Copa América", "continental"),
        ("African Cup of Nations", "continental"),
        ("AFC Asian Cup", "continental"),
        ("Gold Cup", "continental"),
        ("Friendly", "friendly"),
        ("UEFA Nations League", "default"),
        ("Something random", "default"),
    ],
)
def test_classify_tournament_maps_correctly(tournament: str, expected: str) -> None:
    """`classify_tournament` associa ogni nome alla categoria K corretta."""
    assert classify_tournament(tournament) == expected
```

- [ ] **Step 7.2: Verificare i test falliscono**

```bash
.venv/Scripts/pytest tests/test_elo.py -v -k "classify"
```

Expected: 11 failures con `NotImplementedError`.

- [ ] **Step 7.3: Implementare `classify_tournament`**

Sostituisci il body di `classify_tournament` in `src/mondiali/features/elo.py`:

```python
def classify_tournament(tournament: str) -> str:
    """Mappa il nome del torneo alle categorie di K-factor.

    Regole (ordine di precedenza):
    1. se contiene 'qualification' → 'qualification' (batte tutto)
    2. 'FIFA World Cup' (senza qualification) → 'world_cup'
    3. Euro, Copa, AFC Asian Cup, African Cup, Gold Cup → 'continental'
    4. 'Friendly' → 'friendly'
    5. altrimenti (Nations League, tornei minori) → 'default'
    """
    t = tournament.lower()
    if "qualification" in t:
        return "qualification"
    if "fifa world cup" in t:
        return "world_cup"
    continental_keywords = (
        "uefa euro",
        "copa américa",
        "copa america",
        "african cup of nations",
        "africa cup of nations",
        "afc asian cup",
        "gold cup",
    )
    if any(kw in t for kw in continental_keywords):
        return "continental"
    if t == "friendly":
        return "friendly"
    return "default"
```

- [ ] **Step 7.4: Verificare i test classify passano**

```bash
.venv/Scripts/pytest tests/test_elo.py -v -k "classify"
```

Expected: 11 passed.

- [ ] **Step 7.5: Aggiungere test di integrazione con K_FACTORS**

Aggiungi in `tests/test_elo.py`:

```python
def test_classify_tournament_keys_match_k_factors_dict() -> None:
    """Ogni categoria ritornata da classify_tournament deve esistere in K_FACTORS."""
    from mondiali.config import K_FACTORS

    categories = {"world_cup", "continental", "qualification", "friendly", "default"}
    assert categories == set(K_FACTORS.keys())
```

- [ ] **Step 7.6: Verificare passa**

```bash
.venv/Scripts/pytest tests/test_elo.py -v -k "classify_tournament_keys"
```

Expected: 1 passed.

- [ ] **Step 7.7: Commit**

```bash
git add src/mondiali/features/elo.py tests/test_elo.py
git commit -m "feat(elo): classify tournament to K-factor category"
```

---

## Task 8: Elo — build history da DataFrame di match

**Files:**
- Modify: `src/mondiali/features/elo.py`
- Modify: `tests/test_elo.py`

- [ ] **Step 8.1: Scrivere i test failing per `build_history`**

Aggiungi in `tests/test_elo.py`:

```python
def test_build_history_returns_pre_match_ratings_per_row() -> None:
    """build_history aggiunge colonne home_elo_before e away_elo_before per ogni match."""
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "home_team": ["A", "A"],
            "away_team": ["B", "C"],
            "home_score": [2, 1],
            "away_score": [0, 1],
            "tournament": ["Friendly", "Friendly"],
            "neutral": [True, True],
        }
    )
    elo = EloSystem()
    result = elo.build_history(matches)

    assert "home_elo_before" in result.columns
    assert "away_elo_before" in result.columns

    # Match 1: entrambi al default
    assert result.iloc[0]["home_elo_before"] == pytest.approx(DEFAULT_ELO, abs=0.01)
    assert result.iloc[0]["away_elo_before"] == pytest.approx(DEFAULT_ELO, abs=0.01)

    # Match 2: A ha già aggiornato il rating (vinse il primo)
    assert result.iloc[1]["home_elo_before"] > DEFAULT_ELO
    assert result.iloc[1]["away_elo_before"] == pytest.approx(DEFAULT_ELO, abs=0.01)


def test_build_history_preserves_row_order() -> None:
    """L'output ha stessa lunghezza e stesso ordine dell'input."""
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "home_team": ["A", "B", "C"],
            "away_team": ["B", "C", "A"],
            "home_score": [1, 2, 0],
            "away_score": [1, 0, 3],
            "tournament": ["Friendly", "Friendly", "Friendly"],
            "neutral": [True, True, True],
        }
    )
    result = EloSystem().build_history(matches)

    assert len(result) == len(matches)
    pd.testing.assert_series_equal(
        result["date"].reset_index(drop=True), matches["date"].reset_index(drop=True)
    )


def test_build_history_uses_correct_k_factor_per_tournament() -> None:
    """Match in WC usa K=60, Friendly K=20, quindi gli update sono diversi."""
    matches_wc = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "home_team": ["A"],
            "away_team": ["B"],
            "home_score": [1],
            "away_score": [0],
            "tournament": ["FIFA World Cup"],
            "neutral": [True],
        }
    )
    matches_friendly = matches_wc.copy()
    matches_friendly["tournament"] = ["Friendly"]

    elo_wc = EloSystem()
    elo_wc.build_history(matches_wc)

    elo_fr = EloSystem()
    elo_fr.build_history(matches_friendly)

    delta_wc = elo_wc.get("A") - DEFAULT_ELO
    delta_fr = elo_fr.get("A") - DEFAULT_ELO
    assert delta_wc > delta_fr
    assert delta_wc == pytest.approx(3 * delta_fr, abs=0.1)  # K=60 vs K=20


def test_build_history_raises_if_not_sorted_by_date() -> None:
    """Input non ordinato per data → ValueError (protegge da data leakage sottile)."""
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-01"]),
            "home_team": ["A", "A"],
            "away_team": ["B", "C"],
            "home_score": [1, 1],
            "away_score": [0, 0],
            "tournament": ["Friendly", "Friendly"],
            "neutral": [True, True],
        }
    )
    with pytest.raises(ValueError, match="must be sorted by date ascending"):
        EloSystem().build_history(matches)
```

- [ ] **Step 8.2: Verificare i test falliscono**

```bash
.venv/Scripts/pytest tests/test_elo.py -v -k "build_history"
```

Expected: 4 failures con `AttributeError: 'EloSystem' object has no attribute 'build_history'`.

- [ ] **Step 8.3: Implementare `build_history`**

Aggiungi il metodo `build_history` in `EloSystem` in `src/mondiali/features/elo.py`:

```python
    def build_history(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Itera sui match (ordinati per data) e ritorna df con Elo pre-match per riga.

        Colonne richieste in input: date, home_team, away_team, home_score, away_score,
        tournament, neutral.

        Output: stesse colonne + `home_elo_before`, `away_elo_before`, `k_factor_used`.

        Muta lo stato interno (`self.ratings`) con i rating finali dopo tutti i match.

        Raises:
            ValueError: se `matches` non è ordinato per data crescente.
        """
        dates = matches["date"]
        if not dates.is_monotonic_increasing:
            raise ValueError(
                "matches must be sorted by date ascending before calling build_history"
            )

        home_elo_before: list[float] = []
        away_elo_before: list[float] = []
        k_factors_used: list[int] = []

        for row in matches.itertuples(index=False):
            category = classify_tournament(row.tournament)
            k = K_FACTORS[category]
            pre_home, pre_away = self.update(
                home=row.home_team,
                away=row.away_team,
                home_goals=int(row.home_score),
                away_goals=int(row.away_score),
                k_factor=float(k),
                neutral=bool(row.neutral),
            )
            home_elo_before.append(pre_home)
            away_elo_before.append(pre_away)
            k_factors_used.append(k)

        result = matches.copy()
        result["home_elo_before"] = home_elo_before
        result["away_elo_before"] = away_elo_before
        result["k_factor_used"] = k_factors_used
        return result
```

- [ ] **Step 8.4: Verificare i test build_history passano**

```bash
.venv/Scripts/pytest tests/test_elo.py -v -k "build_history"
```

Expected: 4 passed.

- [ ] **Step 8.5: Lint + type check**

```bash
.venv/Scripts/ruff check src/mondiali/features/elo.py tests/test_elo.py
.venv/Scripts/mypy src/mondiali/features/elo.py
```

Expected: 0 errori.

- [ ] **Step 8.6: Commit**

```bash
git add src/mondiali/features/elo.py tests/test_elo.py
git commit -m "feat(elo): build_history iterates matches and produces pre-match ratings"
```

---

## Task 9: Elo — sanity test su dati reali (Francia fine 2018)

**Files:**
- Modify: `tests/test_elo.py`

- [ ] **Step 9.1: Scrivere il test di integrazione**

Aggiungi in `tests/test_elo.py`:

```python
def test_elo_france_end_2018_in_plausible_range() -> None:
    """Sanity check: costruito l'Elo su tutti i match dal 2002 a fine 2018,
    la Francia (che ha appena vinto il WC2018) deve avere un Elo in [1950, 2200].

    Requisiti: `data/raw/results.csv` presente (scaricato con `mondiali ingest`
    in questa STEP 1, o manualmente). Se non c'è, skip con motivo.
    """
    from pathlib import Path
    from mondiali.config import CONFIG
    from mondiali.data.ingestion import load_international_results

    csv_path = CONFIG.data_raw / "results.csv"
    if not csv_path.exists():
        pytest.skip(f"{csv_path} not found — run `mondiali ingest` first")

    df = load_international_results(csv_path)
    df = df[(df["date"] >= "2002-01-01") & (df["date"] <= "2018-12-31")].copy()
    df = df.sort_values("date").reset_index(drop=True)

    elo = EloSystem()
    elo.build_history(df)

    france_elo = elo.get("France")
    assert 1950.0 <= france_elo <= 2200.0, (
        f"Francia Elo = {france_elo:.1f}, fuori range [1950, 2200] — "
        "formula Elo o K-factor potrebbero essere buggate"
    )
```

- [ ] **Step 9.2: Scaricare il CSV reale**

Esegui una-tantum:

```bash
.venv/Scripts/python -c "from pathlib import Path; from mondiali.data.ingestion import download_international_results; from mondiali.config import CONFIG; download_international_results(CONFIG.data_raw / 'results.csv')"
```

Expected: file `data/raw/results.csv` creato (~4-5 MB).

- [ ] **Step 9.3: Eseguire il test sanity**

```bash
.venv/Scripts/pytest tests/test_elo.py::test_elo_france_end_2018_in_plausible_range -v
```

Expected: 1 passed. Se fallisce, stampa l'Elo osservato — se è fuori range [1950, 2200], probabile bug nella formula Elo o nella classificazione K-factor. Debugga prima di proseguire.

- [ ] **Step 9.4: Bonus — sanity check addizionale Germany e Brazil**

Aggiungi in `tests/test_elo.py`:

```python
def test_elo_top_teams_end_2018_all_high() -> None:
    """Germany, Brazil, Spain devono tutti stare sopra 1900 a fine 2018."""
    from mondiali.config import CONFIG
    from mondiali.data.ingestion import load_international_results

    csv_path = CONFIG.data_raw / "results.csv"
    if not csv_path.exists():
        pytest.skip(f"{csv_path} not found")

    df = load_international_results(csv_path)
    df = df[(df["date"] >= "2002-01-01") & (df["date"] <= "2018-12-31")].copy()

    elo = EloSystem()
    elo.build_history(df)

    for team in ["Germany", "Brazil", "Spain"]:
        assert elo.get(team) > 1900.0, f"{team} Elo = {elo.get(team):.1f}, below 1900"
```

```bash
.venv/Scripts/pytest tests/test_elo.py::test_elo_top_teams_end_2018_all_high -v
```

Expected: 1 passed.

- [ ] **Step 9.5: Commit**

```bash
git add tests/test_elo.py
git commit -m "test(elo): sanity checks against real 2002-2018 results"
```

---

## Task 10: Matches processed — pipeline end-to-end data → `matches.parquet`

**Files:**
- Modify: `src/mondiali/data/ingestion.py`
- Modify: `tests/test_ingestion.py`

- [ ] **Step 10.1: Scrivere il test failing per `build_processed_matches`**

Aggiungi in `tests/test_ingestion.py`:

```python
def test_build_processed_matches_produces_expected_schema(tmp_path: Path) -> None:
    """Pipeline ingest → processed produce parquet con schema atteso."""
    from mondiali.data.ingestion import build_processed_matches

    raw_csv = tmp_path / "results.csv"
    raw_csv.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2018-07-15,France,Croatia,4,2,FIFA World Cup,Moscow,Russia,TRUE\n"
        "2018-09-06,France,Germany,0,0,UEFA Nations League,Munich,Germany,FALSE\n"
    )
    out_path = tmp_path / "matches.parquet"

    result_path = build_processed_matches(raw_csv, out_path)

    assert result_path == out_path
    df = pd.read_parquet(out_path)
    expected_cols = {
        "match_id",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
        "home_elo_before",
        "away_elo_before",
        "k_factor_used",
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 2
    assert df["match_id"].is_unique
```

- [ ] **Step 10.2: Verificare il test fallisce**

```bash
.venv/Scripts/pytest tests/test_ingestion.py -v -k "build_processed"
```

Expected: FAIL — `build_processed_matches` non esiste.

- [ ] **Step 10.3: Implementare `build_processed_matches`**

Aggiungi in fondo a `src/mondiali/data/ingestion.py`:

```python
def build_processed_matches(raw_csv: Path, out_path: Path) -> Path:
    """Pipeline: raw CSV → matches.parquet con Elo pre-match per riga.

    - Carica il raw
    - Ordina per data (già fatto da `load_international_results`)
    - Costruisce `EloSystem.build_history`
    - Aggiunge `match_id` stabile (hash di date+home+away)
    - Scrive `matches.parquet`

    Args:
        raw_csv: path del CSV scaricato.
        out_path: dove scrivere il parquet.

    Returns:
        out_path.
    """
    from mondiali.features.elo import EloSystem

    df = load_international_results(raw_csv)
    elo = EloSystem()
    df = elo.build_history(df)

    df["match_id"] = (
        df["date"].dt.strftime("%Y%m%d")
        + "_"
        + df["home_team"].str.replace(" ", "_")
        + "_vs_"
        + df["away_team"].str.replace(" ", "_")
    )
    if not df["match_id"].is_unique:
        df["match_id"] = df["match_id"] + "_" + df.groupby("match_id").cumcount().astype(str)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("wrote processed matches", path=str(out_path), rows=len(df))
    return out_path
```

- [ ] **Step 10.4: Verificare il test passa**

```bash
.venv/Scripts/pytest tests/test_ingestion.py -v -k "build_processed"
```

Expected: 1 passed.

- [ ] **Step 10.5: Commit**

```bash
git add src/mondiali/data/ingestion.py tests/test_ingestion.py
git commit -m "feat(data): build_processed_matches pipeline with Elo pre-match"
```

---

## Task 11: Anti-leakage test framework

**Files:**
- Create: `tests/test_leakage.py`

- [ ] **Step 11.1: Creare `tests/test_leakage.py` con check foundation**

Crea `tests/test_leakage.py`:

```python
"""Framework anti-data-leakage.

Ogni feature deve essere calcolata usando esclusivamente informazioni strettamente
anteriori a `match_date`. Questo file contiene:
1. Una sentinella che verifica l'invariante sull'Elo history (home_elo_before di
   un match alla data D deve essere l'Elo di prima di D, mai di D-stesso o dopo).
2. Hook futuri per Tier 2+ (form, market value, ecc.) — implementati negli STEP
   successivi.

Regola: se `log_loss < 0.92` in validation, questo test framework deve essere
eseguito prima di qualsiasi claim di miglioramento — log-loss troppo basso è
sintomo #1 di leakage.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mondiali.config import CONFIG


def _load_processed() -> pd.DataFrame | None:
    """Carica matches.parquet se esiste, altrimenti None."""
    path = CONFIG.data_processed / "matches.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def test_elo_before_is_strictly_pre_match() -> None:
    """Per ogni match, home_elo_before deve essere il rating PRIMA dell'update di
    quel match. Test: ri-simuliamo l'Elo history e confrontiamo.
    """
    df = _load_processed()
    if df is None:
        pytest.skip("data/processed/matches.parquet not found — run build_processed_matches first")

    from mondiali.features.elo import EloSystem

    elo = EloSystem()
    df_sorted = df.sort_values("date").reset_index(drop=True)
    expected_home = []
    expected_away = []
    for row in df_sorted.itertuples(index=False):
        expected_home.append(elo.get(row.home_team))
        expected_away.append(elo.get(row.away_team))
        elo.update(
            home=row.home_team,
            away=row.away_team,
            home_goals=int(row.home_score),
            away_goals=int(row.away_score),
            k_factor=float(row.k_factor_used),
            neutral=bool(row.neutral),
        )

    assert df_sorted["home_elo_before"].tolist() == pytest.approx(expected_home, abs=1e-6)
    assert df_sorted["away_elo_before"].tolist() == pytest.approx(expected_away, abs=1e-6)


def test_no_future_matches_in_processed() -> None:
    """matches.parquet non deve contenere partite future (date > oggi)."""
    df = _load_processed()
    if df is None:
        pytest.skip("data/processed/matches.parquet not found")

    today = pd.Timestamp.now().normalize()
    future_rows = df[df["date"] > today]
    assert len(future_rows) == 0, (
        f"Found {len(future_rows)} future matches in processed set — "
        f"likely ingestion bug or unresolved fixtures slipped through"
    )
```

- [ ] **Step 11.2: Generare `matches.parquet` (serve per far passare i test)**

```bash
.venv/Scripts/python -c "from mondiali.config import CONFIG; from mondiali.data.ingestion import build_processed_matches; build_processed_matches(CONFIG.data_raw / 'results.csv', CONFIG.data_processed / 'matches.parquet')"
```

Expected: file `data/processed/matches.parquet` creato.

- [ ] **Step 11.3: Verificare i test leakage passano**

```bash
.venv/Scripts/pytest tests/test_leakage.py -v
```

Expected: 2 passed.

- [ ] **Step 11.4: Lint**

```bash
.venv/Scripts/ruff check tests/test_leakage.py
```

Expected: 0 errori.

- [ ] **Step 11.5: Commit**

```bash
git add tests/test_leakage.py
git commit -m "test(leakage): framework + Elo strict-pre-match invariant"
```

---

## Task 12: Baseline prior — Tier 0 (PriorBaseline)

**Files:**
- Create: `src/mondiali/training/__init__.py`
- Create: `src/mondiali/training/baseline_prior.py`
- Create: `tests/test_baseline_prior.py`

- [ ] **Step 12.1: Scrivere il test failing**

Crea `tests/test_baseline_prior.py`:

```python
"""Test per baseline Tier 0 (prior costante 1/X/2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mondiali.training.baseline_prior import PriorBaseline


def _make_df(n_home_win: int, n_draw: int, n_away_win: int) -> pd.DataFrame:
    """Helper: df con esiti 1/X/2 in proporzioni richieste."""
    rows = (
        [{"home_score": 2, "away_score": 0}] * n_home_win
        + [{"home_score": 1, "away_score": 1}] * n_draw
        + [{"home_score": 0, "away_score": 2}] * n_away_win
    )
    return pd.DataFrame(rows)


def test_prior_fit_computes_frequencies() -> None:
    """fit() calcola le frequenze delle 3 classi dal training set."""
    df = _make_df(n_home_win=45, n_draw=25, n_away_win=30)
    model = PriorBaseline()
    model.fit(df)

    assert model.prior_ == pytest.approx([0.45, 0.25, 0.30], abs=0.001)


def test_prior_predict_proba_returns_constant_rows() -> None:
    """predict_proba restituisce lo stesso vettore di prior per ogni riga di input."""
    df_train = _make_df(50, 20, 30)
    model = PriorBaseline()
    model.fit(df_train)

    df_test = _make_df(1, 1, 1)  # 3 righe
    probs = model.predict_proba(df_test)

    assert probs.shape == (3, 3)
    for row in probs:
        assert row == pytest.approx([0.50, 0.20, 0.30], abs=0.001)


def test_prior_proba_rows_sum_to_one() -> None:
    """Ogni riga di probabilità somma a 1."""
    df = _make_df(10, 20, 30)
    model = PriorBaseline()
    model.fit(df)
    probs = model.predict_proba(df.head(5))

    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)


def test_prior_raises_if_predict_before_fit() -> None:
    """predict_proba prima di fit() solleva."""
    model = PriorBaseline()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_proba(pd.DataFrame({"home_score": [1], "away_score": [0]}))
```

- [ ] **Step 12.2: Verificare i test falliscono**

```bash
.venv/Scripts/pytest tests/test_baseline_prior.py -v
```

Expected: 4 failures con `ModuleNotFoundError`.

- [ ] **Step 12.3: Implementare `training/__init__.py` e `training/baseline_prior.py`**

Crea `src/mondiali/training/__init__.py`:

```python
"""Training loop, baselines, evaluation."""
```

Crea `src/mondiali/training/baseline_prior.py`:

```python
"""Baseline Tier 0 — prior costante 1/X/2.

Predice sempre le frequenze storiche del training set. Serve come floor di
riferimento: qualsiasi modello successivo deve batterlo in log-loss o è
indistinguibile dal rumore.

Classi (ordine fisso): 0 = home win, 1 = draw, 2 = away win.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class PriorBaseline:
    """Predice sempre le frequenze storiche 1/X/2 del training set."""

    def __init__(self) -> None:
        self.prior_: list[float] | None = None

    def fit(self, matches: pd.DataFrame) -> "PriorBaseline":
        """Calcola le frequenze 1/X/2 dai match di training.

        Args:
            matches: DataFrame con colonne `home_score`, `away_score`.

        Returns:
            self (per chaining).
        """
        outcomes = _compute_outcomes(matches)
        counts = np.bincount(outcomes, minlength=3).astype(float)
        self.prior_ = (counts / counts.sum()).tolist()
        return self

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        """Ritorna shape (n, 3) con riga costante = prior."""
        if self.prior_ is None:
            raise RuntimeError("PriorBaseline must be fit() before predict_proba")
        n = len(matches)
        return np.tile(np.array(self.prior_), (n, 1))


def _compute_outcomes(matches: pd.DataFrame) -> np.ndarray:
    """0 = home win, 1 = draw, 2 = away win."""
    home = matches["home_score"].to_numpy()
    away = matches["away_score"].to_numpy()
    out = np.where(home > away, 0, np.where(home == away, 1, 2))
    return out.astype(np.int64)
```

- [ ] **Step 12.4: Verificare i test passano**

```bash
.venv/Scripts/pytest tests/test_baseline_prior.py -v
```

Expected: 4 passed.

- [ ] **Step 12.5: Lint + type check**

```bash
.venv/Scripts/ruff check src/mondiali/training/ tests/test_baseline_prior.py
.venv/Scripts/mypy src/mondiali/training/baseline_prior.py
```

Expected: 0 errori.

- [ ] **Step 12.6: Commit**

```bash
git add src/mondiali/training/__init__.py src/mondiali/training/baseline_prior.py tests/test_baseline_prior.py
git commit -m "feat(training): PriorBaseline Tier 0 (constant 1X2 frequencies)"
```

---

## Task 13: Evaluation — log-loss 1X2

**Files:**
- Create: `src/mondiali/training/evaluate.py`
- Create: `tests/test_evaluate.py`

- [ ] **Step 13.1: Scrivere il test failing**

Crea `tests/test_evaluate.py`:

```python
"""Test della funzione di evaluation (log-loss 1/X/2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mondiali.training.evaluate import compute_outcomes, log_loss_1x2


def test_compute_outcomes_encodes_1x2_correctly() -> None:
    """home_win=0, draw=1, away_win=2."""
    df = pd.DataFrame(
        {"home_score": [2, 1, 0, 3], "away_score": [0, 1, 2, 3]}
    )
    assert compute_outcomes(df).tolist() == [0, 1, 2, 1]


def test_log_loss_perfect_prediction_is_zero() -> None:
    """Predizione perfetta (probabilità 1.0 alla classe vera) → log-loss ~ 0."""
    df = pd.DataFrame({"home_score": [2, 1, 0], "away_score": [0, 1, 2]})
    # Epsilon per evitare log(0)
    probs = np.array(
        [
            [1 - 2e-15, 1e-15, 1e-15],  # home win
            [1e-15, 1 - 2e-15, 1e-15],  # draw
            [1e-15, 1e-15, 1 - 2e-15],  # away win
        ]
    )
    loss = log_loss_1x2(df, probs)
    assert loss < 1e-10


def test_log_loss_uniform_prediction_is_log3() -> None:
    """Predizione uniforme 1/3 per tutte le classi → log-loss = ln(3) ≈ 1.0986."""
    df = pd.DataFrame({"home_score": [2, 1, 0], "away_score": [0, 1, 2]})
    probs = np.full((3, 3), 1 / 3)
    loss = log_loss_1x2(df, probs)
    assert loss == pytest.approx(np.log(3), abs=0.001)


def test_log_loss_raises_on_shape_mismatch() -> None:
    """Probabilità con shape sbagliata → ValueError."""
    df = pd.DataFrame({"home_score": [1, 2], "away_score": [0, 0]})
    probs = np.array([[0.5, 0.3, 0.2]])  # 1 riga invece di 2
    with pytest.raises(ValueError, match="shape"):
        log_loss_1x2(df, probs)
```

- [ ] **Step 13.2: Verificare i test falliscono**

```bash
.venv/Scripts/pytest tests/test_evaluate.py -v
```

Expected: 4 failures con ModuleNotFoundError.

- [ ] **Step 13.3: Implementare `training/evaluate.py`**

Crea `src/mondiali/training/evaluate.py`:

```python
"""Evaluation metrics: log-loss 1/X/2, Brier score.

Classi (ordine fisso): 0 = home win, 1 = draw, 2 = away win.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def compute_outcomes(matches: pd.DataFrame) -> np.ndarray:
    """0 = home win, 1 = draw, 2 = away win."""
    home = matches["home_score"].to_numpy()
    away = matches["away_score"].to_numpy()
    out = np.where(home > away, 0, np.where(home == away, 1, 2))
    return out.astype(np.int64)


def log_loss_1x2(matches: pd.DataFrame, probabilities: np.ndarray) -> float:
    """Log-loss multi-classe su esiti 1/X/2.

    Args:
        matches: DataFrame con home_score, away_score (verità).
        probabilities: shape (n, 3), colonne = [P(home), P(draw), P(away)].

    Returns:
        log-loss (media).

    Raises:
        ValueError: shape mismatch o probabilità invalide.
    """
    if probabilities.shape != (len(matches), 3):
        raise ValueError(
            f"probabilities shape {probabilities.shape} != expected ({len(matches)}, 3)"
        )
    y_true = compute_outcomes(matches)
    return float(log_loss(y_true, probabilities, labels=[0, 1, 2]))
```

- [ ] **Step 13.4: Verificare i test passano**

```bash
.venv/Scripts/pytest tests/test_evaluate.py -v
```

Expected: 4 passed.

- [ ] **Step 13.5: Lint + type check**

```bash
.venv/Scripts/ruff check src/mondiali/training/evaluate.py tests/test_evaluate.py
.venv/Scripts/mypy src/mondiali/training/evaluate.py
```

Expected: 0 errori.

- [ ] **Step 13.6: Commit**

```bash
git add src/mondiali/training/evaluate.py tests/test_evaluate.py
git commit -m "feat(training): log_loss_1x2 evaluation metric"
```

---

## Task 14: CLI — `mondiali ingest` + `mondiali baseline`

**Files:**
- Create: `src/mondiali/cli/__init__.py`
- Create: `src/mondiali/cli/main.py`

- [ ] **Step 14.1: Creare `cli/__init__.py` e `cli/main.py`**

Crea `src/mondiali/cli/__init__.py`:

```python
"""Command-line interface."""
```

Crea `src/mondiali/cli/main.py`:

```python
"""Entry point Typer CLI per il package `mondiali`.

Comandi disponibili in STEP 1:
    mondiali ingest        Download + parsing + Elo history → matches.parquet
    mondiali baseline      Fit PriorBaseline su training set, report log-loss
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import typer

from mondiali.config import CONFIG
from mondiali.data.ingestion import build_processed_matches, download_international_results
from mondiali.training.baseline_prior import PriorBaseline
from mondiali.training.evaluate import log_loss_1x2

app = typer.Typer(add_completion=False, no_args_is_help=True)
log = structlog.get_logger(__name__)


@app.command()
def ingest(force: bool = typer.Option(False, "--force", help="Re-download anche se presente")) -> None:
    """Scarica international_results e produce `matches.parquet`."""
    raw_csv = CONFIG.data_raw / "results.csv"
    download_international_results(raw_csv, force=force)
    processed_path = CONFIG.data_processed / "matches.parquet"
    build_processed_matches(raw_csv, processed_path)
    typer.echo(f"OK — processed matches written to {processed_path}")


@app.command()
def baseline(
    train_start: str = typer.Option("2002-01-01", help="Inizio training set"),
    train_end: str = typer.Option("2018-12-31", help="Fine training (esclusivo)"),
    val_start: str = typer.Option("2019-01-01"),
    val_end: str = typer.Option("2022-06-30", help="Fine validation (esclusivo del WC2022)"),
) -> None:
    """Fit PriorBaseline su training, valuta su validation. Tier 0 floor."""
    processed = CONFIG.data_processed / "matches.parquet"
    if not processed.exists():
        typer.echo("matches.parquet non trovato — esegui `mondiali ingest` prima", err=True)
        raise typer.Exit(1)

    df = pd.read_parquet(processed)
    df["date"] = pd.to_datetime(df["date"])
    train = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
    val = df[(df["date"] >= val_start) & (df["date"] <= val_end)]

    typer.echo(f"Train: {len(train)} matches ({train_start} → {train_end})")
    typer.echo(f"Val:   {len(val)} matches ({val_start} → {val_end})")

    model = PriorBaseline()
    model.fit(train)
    assert model.prior_ is not None
    typer.echo(f"Prior 1/X/2 (dal training): {np.round(model.prior_, 4).tolist()}")

    val_probs = model.predict_proba(val)
    val_loss = log_loss_1x2(val, val_probs)
    typer.echo(f"Validation log-loss (Tier 0 prior baseline): {val_loss:.4f}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 14.2: Eseguire `mondiali --help`**

```bash
.venv/Scripts/mondiali --help
```

Expected: output Typer con i due comandi `ingest` e `baseline`.

- [ ] **Step 14.3: Eseguire `mondiali ingest` (il CSV è già scaricato ma verifichiamo idempotenza)**

```bash
.venv/Scripts/mondiali ingest
```

Expected: log strutturato di "already present, skipping download" + creazione matches.parquet.

- [ ] **Step 14.4: Eseguire `mondiali baseline` e catturare il log-loss**

```bash
.venv/Scripts/mondiali baseline
```

Expected: output con:
- `Train: ~XXXX matches (2002-01-01 → 2018-12-31)`
- `Val:   ~XXXX matches (2019-01-01 → 2022-06-30)`
- `Prior 1/X/2 (dal training): [~0.45, ~0.25, ~0.30]` (numeri indicativi)
- `Validation log-loss (Tier 0 prior baseline): ~1.05`

**Annotare il numero esatto** — serve per il report in Task 15.

- [ ] **Step 14.5: Lint + type check**

```bash
.venv/Scripts/ruff check src/mondiali/cli/
.venv/Scripts/mypy src/mondiali/cli/main.py
```

Expected: 0 errori.

- [ ] **Step 14.6: Commit**

```bash
git add src/mondiali/cli/
git commit -m "feat(cli): ingest and baseline commands"
```

---

## Task 15: Report `validation_step1.md`

**Files:**
- Create: `reports/validation_step1.md`

- [ ] **Step 15.1: Eseguire il run completo e raccogliere i numeri**

```bash
.venv/Scripts/mondiali ingest
.venv/Scripts/mondiali baseline
.venv/Scripts/pytest --tb=short
```

Annotare:
- Numero totale di match nel dataset raw
- Numero match training (2002-2018)
- Numero match validation (2019-giugno 2022)
- Prior 1/X/2 esatto
- Log-loss Tier 0 su validation
- Elo finale (al 2018-12-31) di Francia, Germania, Brasile, Spagna

- [ ] **Step 15.2: Scrivere `reports/validation_step1.md`**

Crea `reports/validation_step1.md` sostituendo i placeholder `<...>` con i numeri reali raccolti allo step precedente:

```markdown
# STEP 1 — Foundation validation report

**Data**: <YYYY-MM-DD di esecuzione>
**Commit**: <git rev-parse --short HEAD>
**Python**: <python --version>

## Dataset

- Fonte: `martj42/international_results` (results.csv).
- Totale match nel raw: <N_raw>
- Match dopo filtro date (2002-2018 training, 2019-Q2/2022 validation): <N_train> / <N_val>

## Tier 0 — Prior baseline

**Prior 1/X/2 (dal training 2002-2018)**: [<p_home>, <p_draw>, <p_away>]

**Validation log-loss (2019-2022 pre-WC)**: **<log_loss_tier0>**

Interpretazione: è il floor. Qualsiasi modello successivo che non lo batta in log-loss
è indistinguibile dal "non sapere niente" — prior costante.

## Elo sanity check (fine 2018)

| Team | Elo |
|---|---|
| France | <elo_fra> |
| Germany | <elo_ger> |
| Brazil | <elo_bra> |
| Spain | <elo_esp> |
| San Marino | <elo_sma> |

Range atteso per top-tier post-WC2018: ~1950-2150. Francia appena vinto WC → top.

## Test suite

```
<output di `pytest --tb=short` riassunto: X passed in Ys>
```

## Gate STEP 1 — soddisfatto?

- [ ] Tutti i test verdi (incluso `test_leakage.py` e `test_elo.py` sanity)
- [ ] `mondiali ingest` e `mondiali baseline` funzionano end-to-end
- [ ] Log-loss Tier 0 documentato e in range atteso (~1.05 ± 0.05)
- [ ] Elo Francia fine 2018 in [1950, 2200]

Se tutti ✅ → procedi a scrivere il plan di STEP 2 (Tier 1 XGBoost Poisson).

## Lezioni apprese

<scrivere 2-3 bullet di cosa è risultato più semplice/difficile del previsto>

## Decisioni open per STEP 2

<eventuali domande aperte: formula Elo con GD multiplier? Escludere amichevoli dal training? ecc.>
```

- [ ] **Step 15.3: Compilare i placeholder con i numeri reali**

Sostituisci i `<...>` nel report con i valori effettivi ottenuti. Se log-loss Tier 0 è fuori dal range [0.95, 1.15], **stop**: probabile bug nel conteggio delle classi o nel split — investiga prima di proseguire.

- [ ] **Step 15.4: Mettere le checkbox a ✅ se soddisfatte**

Modifica le `- [ ]` in `- [x]` nel report per i gate passati.

- [ ] **Step 15.5: Commit**

```bash
git add reports/validation_step1.md
git commit -m "docs(report): STEP 1 validation report with Tier 0 log-loss"
```

---

## Task 16: Gate finale STEP 1

**Files:**
- (nessuno — solo verifica)

- [ ] **Step 16.1: Eseguire la suite completa una volta finale**

```bash
.venv/Scripts/pytest -v
```

Expected: **tutti** i test verdi. Contare:
- `test_config.py`: 3
- `test_ingestion.py`: 7
- `test_elo.py`: ~14 (variabile in base ai parametrizzati)
- `test_leakage.py`: 2
- `test_baseline_prior.py`: 4
- `test_evaluate.py`: 4

Totale atteso: **~30-35 test passed**.

- [ ] **Step 16.2: Verificare lint + type check sull'intero src/**

```bash
.venv/Scripts/ruff check src/ tests/
.venv/Scripts/mypy src/
```

Expected: 0 errori.

- [ ] **Step 16.3: Verificare che il report è presente e compilato**

```bash
cat reports/validation_step1.md | head -40
```

Expected: output con numeri reali, non placeholder `<...>`.

- [ ] **Step 16.4: Tag git `step1-complete`**

```bash
git tag step1-complete -m "STEP 1 Foundation completato, Tier 0 baseline documentato"
git log --oneline -20
```

Expected: tag visibile, ~16 commit nella history di STEP 1.

- [ ] **Step 16.5: Handoff a STEP 2**

Aprire sessione per il plan di STEP 2 (Tier 1 XGBoost Poisson + Dixon-Coles + Markets).

Contesto da portare:
- Log-loss Tier 0 = <valore reale da battere>
- Le feature disponibili in `matches.parquet`: date, teams, scores, tournament, neutral, home_elo_before, away_elo_before, k_factor_used
- Baseline Elo-only logistic ancora da costruire: è il primo target comparativo di Tier 1

---

## Recap STEP 1

**Cosa hai in mano alla fine:**
- Package `mondiali` installabile con CLI `mondiali ingest` e `mondiali baseline`
- `data/processed/matches.parquet` con ~40k+ match internazionali + Elo pre-match per ogni riga
- Sistema Elo custom con K-factor variabile, testato su dati reali (Francia post-WC2018 in range atteso)
- Framework anti-data-leakage attivo in CI test
- Tier 0 log-loss documentato come floor
- Report `validation_step1.md` archiviato

**Cosa NON hai (va in STEP 2):**
- Modello XGBoost Poisson
- Correzione Dixon-Coles
- Derivazione mercati (1X2, O/U, BTTS)
- Walk-forward CV
- Optuna hyperparam search

**Tempo stimato totale STEP 1**: 10-15h, spalmati su 1-2 weekend o equivalente.
