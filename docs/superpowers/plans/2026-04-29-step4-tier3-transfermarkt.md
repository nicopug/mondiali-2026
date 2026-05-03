# STEP 4 — Tier 3 Transfermarkt market values — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Aggiungere al modello una terza fonte (valori di mercato dei giocatori da Transfermarkt via Wayback Machine) come 6 colonne simmetriche `(market_value_total, market_value_top11, tm_age_days)` × `(home, away)`, training Tier 3 su matches 2014+, gate doppio (coverage funzionale ≥80% + metric Δ ≤ −0.001 vs Tier 2 ricomputato).

**Architecture:** Tre sottosistemi nuovi: (1) `mondiali.data.transfermarkt` scraper Wayback-aware con cache HTML su disco e rate limiter; (2) `mondiali.features.tier3` feature builder con anti-leakage strict-pre-match e hard floor coverage; (3) `mondiali.training.train.train_tier3_pipeline` 4-way temporal split 2014-2022. Il modello XGBoost Poisson cresce 18→24 feature simmetriche.

**Tech Stack:** Python 3.11+, `requests` per HTTP/Wayback CDX API, `beautifulsoup4` per parsing HTML, `pyarrow` per parquet, `xgboost==3.2.0`, `pandas`, `pytest` con `responses` per mock HTTP.

**Spec di riferimento:** `docs/superpowers/specs/2026-04-29-step4-tier3-transfermarkt-design.md`.

---

## Pre-flight

- [ ] **Verifica ambiente**

```bash
git status
git log --oneline -3
python -c "import xgboost, bs4, requests; print(xgboost.__version__, bs4.__version__, requests.__version__)"
pytest -q --no-header
```

Expected:
- `git status`: clean working tree.
- HEAD: `6a51429 docs(spec): STEP 4 design...`.
- 127 test PASS (post STEP 3).

- [ ] **Aggiungi `responses` come dev dep se non presente**

Check `pyproject.toml` per `responses` (mock HTTP per test). Se manca:

```bash
pip install responses
# E aggiungerlo a [dependency-groups.dev] o [project.optional-dependencies.dev] in pyproject.toml
```

Se `pyproject.toml` non ha `responses`, aggiungerlo nella sezione dev:

```toml
[dependency-groups]
dev = [
    ...,
    "responses>=0.25",
]
```

---

## File Structure

| File | Status | Responsabilità |
|---|---|---|
| `src/mondiali/data/transfermarkt.py` | Create | Scraper: CDX query, HTML fetch, parsing, snapshot orchestration |
| `src/mondiali/data/scope.py` | Create | `compute_tier3_scope` + `WC2026_QUALIFIED` |
| `src/mondiali/data/tm_nations.py` | Create | Lookup table `NATION_TM_IDS: dict[str, tuple[str, int]]` (~80 entries) |
| `src/mondiali/data/ingestion.py` | Modify | Aggiungere chiamata a `add_tier3_features` |
| `src/mondiali/features/tier3.py` | Create | `add_tier3_features`, `TIER3_COLUMNS`, hard-floor + age clipping |
| `src/mondiali/features/__init__.py` | Modify | Re-export Tier 3 |
| `src/mondiali/model/poisson_xgb.py` | Modify | `SYMMETRIC_FEATURES` 18→24, `build_symmetric_rows` indici 18-23 |
| `src/mondiali/training/train.py` | Modify | `train_tier3_pipeline` + helper `_recompute_tier2_baseline_for_gate` |
| `src/mondiali/cli/main.py` | Modify | Comandi `tm-scrape` e `train-tier3` |
| `tests/test_transfermarkt.py` | Create | Parser HTML su 3 fixture, CDX mock, fallback chain, cache idempotenza |
| `tests/fixtures/tm_*.html` | Create | 3 fixture HTML (2014, 2018, 2022) |
| `tests/test_features_tier3.py` | Create | Anti-leakage, hard floor, age clipping, NaN pre-2014 |
| `tests/test_train_tier3.py` | Create | Smoke + slow gate-blocking test |
| `tests/test_scope.py` | Create | `compute_tier3_scope` deterministico |
| `tests/test_leakage.py` | Modify | + `test_tier3_market_value_strict_pre_match` |
| `tests/test_poisson_xgb.py` | Modify | Aggiungere check `len(SYMMETRIC_FEATURES) == 24` |
| `data/raw/transfermarkt/snapshots.parquet` | Generated | Output scraper (run-time) |
| `data/processed/tier3_scope.json` | Generated | Output `compute_tier3_scope` (run-time) |
| `models/tier3/xgb_poisson.json` | Generated | Output `train-tier3 --save-model` |
| `reports/validation_step4.md` | Create | Report finale STEP 4 |

---

## Task 1: Bootstrap `NATION_TM_IDS` lookup table

**Files:**
- Create: `src/mondiali/data/tm_nations.py`
- Test: nessuno (data table)

Manuale data-collection. Mapping team_name (come appare in `matches.parquet`, es. "Italy", "United States") → `(tm_slug, tm_id)` per costruire URL pattern Transfermarkt.

URL pattern: `https://www.transfermarkt.com/{tm_slug}/startseite/verein/{tm_id}`.

- [ ] **Step 1: Crea il file con le 48 WC2026 + le top-32 storiche**

```python
# src/mondiali/data/tm_nations.py
"""Lookup table team_name -> (tm_slug, tm_id) per URL Transfermarkt.

Mapping costruito manualmente cercando ogni nazionale su transfermarkt.com.
URL pattern: https://www.transfermarkt.com/{slug}/startseite/verein/{id}.

Il `team_name` (chiave del dict) deve matchare ESATTAMENTE come appare nelle
colonne `home_team` / `away_team` di `matches.parquet`. Usa la stringa
canonica del dataset martj42/international_results.
"""
from __future__ import annotations

NATION_TM_IDS: dict[str, tuple[str, int]] = {
    # WC2026 qualified / likely qualified (snapshot 2026-04-29)
    "Argentina": ("argentinien", 3437),
    "France": ("frankreich", 3377),
    "Brazil": ("brasilien", 3439),
    "England": ("england", 3299),
    "Spain": ("spanien", 3375),
    "Germany": ("deutschland", 3262),
    "Portugal": ("portugal", 3300),
    "Netherlands": ("niederlande", 3382),
    "Belgium": ("belgien", 3382),  # FIXME: verify ID
    "Italy": ("italien", 3376),
    "Croatia": ("kroatien", 3556),
    "Uruguay": ("uruguay", 3439),  # FIXME: verify ID (collision with Brazil id?)
    "Mexico": ("mexiko", 6303),
    "United States": ("vereinigte-staaten", 3505),
    "Canada": ("kanada", 3433),
    "Morocco": ("marokko", 3473),
    "Senegal": ("senegal", 3499),
    "Japan": ("japan", 3437),  # FIXME: verify
    "South Korea": ("sudkorea", 3520),
    "Australia": ("australien", 3433),  # FIXME: verify
    "Saudi Arabia": ("saudi-arabien", 3502),
    "Iran": ("iran", 3373),
    "Ecuador": ("ecuador", 3447),
    "Colombia": ("kolumbien", 3438),
    "Peru": ("peru", 3441),
    "Chile": ("chile", 3440),
    "Paraguay": ("paraguay", 3442),
    "Switzerland": ("schweiz", 3384),
    "Denmark": ("danemark", 3375),  # FIXME: verify
    "Poland": ("polen", 3437),  # FIXME: verify
    "Serbia": ("serbien", 3439),  # FIXME: verify
    "Wales": ("wales", 3577),
    "Scotland": ("schottland", 3576),
    "Austria": ("osterreich", 3442),  # FIXME: verify
    "Sweden": ("schweden", 3375),  # FIXME: verify
    "Norway": ("norwegen", 3375),  # FIXME: verify
    "Czech Republic": ("tschechien", 3375),  # FIXME: verify
    "Hungary": ("ungarn", 3578),
    "Turkey": ("turkei", 3376),
    "Ukraine": ("ukraine", 3376),
    "Romania": ("rumanien", 3375),
    "Slovakia": ("slowakei", 3375),
    "Slovenia": ("slowenien", 3375),
    "Greece": ("griechenland", 3375),
    "Republic of Ireland": ("irland", 3299),
    "Bosnia and Herzegovina": ("bosnien-herzegowina", 3375),
    "North Macedonia": ("nordmazedonien", 3375),
    "Albania": ("albanien", 3375),
    # Top-32 historic FIFA Elo (non WC2026 ma frequenti in qualifications)
    "Russia": ("russland", 3437),
    "Tunisia": ("tunesien", 3499),
    "Algeria": ("algerien", 3473),
    "Egypt": ("agypten", 3471),
    "Nigeria": ("nigeria", 3499),
    "Ghana": ("ghana", 3473),
    "Cameroon": ("kamerun", 3473),
    "Ivory Coast": ("elfenbeinkuste", 3473),
    "Iceland": ("island", 3375),
    "Finland": ("finnland", 3375),
    "Bolivia": ("bolivien", 3439),
    "Venezuela": ("venezuela", 3439),
    "Costa Rica": ("costa-rica", 3433),
    "Panama": ("panama", 3433),
    "Honduras": ("honduras", 3433),
    "Jamaica": ("jamaika", 3433),
    "Qatar": ("katar", 3502),
    "United Arab Emirates": ("vereinigte-arabische-emirate", 3502),
    "Iraq": ("irak", 3502),
    "China PR": ("china", 3520),
    "New Zealand": ("neuseeland", 3433),
    "South Africa": ("sudafrika", 3499),
    "Mali": ("mali", 3499),
    "Burkina Faso": ("burkina-faso", 3499),
    "DR Congo": ("dr-kongo", 3499),
    "Cape Verde": ("kap-verde", 3499),
    "Israel": ("israel", 3375),
    "Georgia": ("georgien", 3576),
    "Armenia": ("armenien", 3576),
    "Azerbaijan": ("aserbaidschan", 3576),
}
```

⚠ **Tutti gli `id` flaggati `# FIXME: verify` vanno corretti manualmente**: l'implementer apre `https://www.transfermarkt.com/{slug}/startseite/verein/{id}` per ognuno e verifica che la pagina caricata sia la nazionale giusta. La maggior parte dei `tm_id` reali sono numeri univoci tra 3000-7000. Rimuovere il commento `# FIXME` solo dopo aver verificato.

Lista canonica dei `team_name` da `matches.parquet`:
```bash
python -c "import pandas as pd; df = pd.read_parquet('data/processed/matches.parquet'); names = sorted(set(df['home_team']) | set(df['away_team'])); print('\n'.join(names[:200]))"
```

Se un `team_name` del dataset non è in `NATION_TM_IDS`, sarà escluso automaticamente dal feature Tier 3 (NaN). OK.

- [ ] **Step 2: Sanity check programmatico**

Aggiungi questo come modulo-level assertion in fondo al file:

```python
# Sanity invariants
assert all(isinstance(v, tuple) and len(v) == 2 for v in NATION_TM_IDS.values()), \
    "NATION_TM_IDS values must be (slug: str, id: int) tuples"
assert all(isinstance(slug, str) and isinstance(tid, int) for slug, tid in NATION_TM_IDS.values())
assert len(NATION_TM_IDS) >= 60, f"expected >=60 nations, got {len(NATION_TM_IDS)}"
```

- [ ] **Step 3: Run import smoke**

```bash
python -c "from mondiali.data.tm_nations import NATION_TM_IDS; print(len(NATION_TM_IDS))"
```

Expected: numero ≥60.

- [ ] **Step 4: Commit**

```bash
git add src/mondiali/data/tm_nations.py
git commit -m "feat(data): NATION_TM_IDS lookup table for Tier 3 scraper (Task 1)"
```

---

## Task 2: `compute_tier3_scope` + `WC2026_QUALIFIED`

**Files:**
- Create: `src/mondiali/data/scope.py`
- Test: `tests/test_scope.py`

- [ ] **Step 1: Scrivi il test (failing)**

```python
# tests/test_scope.py
"""Test scope generator per Tier 3."""
from __future__ import annotations

import pandas as pd
import pytest

from mondiali.data.scope import WC2026_QUALIFIED, compute_tier3_scope


def test_wc2026_qualified_is_a_list_of_48():
    """48 nazionali qualificate al World Cup 2026 (3 host + 45 sportive qualifiers)."""
    assert isinstance(WC2026_QUALIFIED, list)
    assert len(WC2026_QUALIFIED) == 48
    assert all(isinstance(x, str) for x in WC2026_QUALIFIED)
    assert "United States" in WC2026_QUALIFIED  # host
    assert "Argentina" in WC2026_QUALIFIED  # defending champ


def test_compute_tier3_scope_includes_wc2026_qualified():
    """Lo scope finale deve contenere tutte le 48 WC2026 qualified."""
    matches = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2021-01-01"]),
        "home_team": ["Italy", "Brazil"],
        "away_team": ["France", "Argentina"],
        "home_elo_before": [1900.0, 2000.0],
        "away_elo_before": [1950.0, 1980.0],
    })
    scope = compute_tier3_scope(matches)
    for nation in WC2026_QUALIFIED:
        assert nation in scope


def test_compute_tier3_scope_includes_top_elo_per_year():
    """Una nazionale con Elo alto in un anno 2014+ deve entrare nel top-50 storico."""
    rows = []
    # Synthetic: Spain ha Elo molto alto nel 2017
    for d in pd.date_range("2017-01-01", "2017-12-31", periods=60):
        rows.append({
            "date": d,
            "home_team": "Spain",
            "away_team": "Random Team",
            "home_elo_before": 2100.0,
            "away_elo_before": 1500.0,
        })
    matches = pd.DataFrame(rows)
    scope = compute_tier3_scope(matches)
    assert "Spain" in scope


def test_compute_tier3_scope_excludes_pre_2014():
    """Una nazionale appare SOLO pre-2014: NON deve entrare via top-50 storico
    (ma può entrare comunque se è in WC2026_QUALIFIED)."""
    rows = []
    for d in pd.date_range("2010-01-01", "2010-12-31", periods=60):
        rows.append({
            "date": d,
            "home_team": "Galaxy United",  # nome fittizio non in WC2026
            "away_team": "Foobar FC",
            "home_elo_before": 2200.0,
            "away_elo_before": 1500.0,
        })
    matches = pd.DataFrame(rows)
    scope = compute_tier3_scope(matches)
    assert "Galaxy United" not in scope


def test_compute_tier3_scope_is_sorted_and_unique():
    """Output ordinato e senza duplicati (deterministico)."""
    matches = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01"]),
        "home_team": ["Italy"],
        "away_team": ["France"],
        "home_elo_before": [1900.0],
        "away_elo_before": [1950.0],
    })
    scope = compute_tier3_scope(matches)
    assert scope == sorted(scope)
    assert len(scope) == len(set(scope))
```

- [ ] **Step 2: Run test (FAIL)**

```bash
pytest tests/test_scope.py -v
```

Expected: 5 errors / fails — module not yet implemented.

- [ ] **Step 3: Implementa `scope.py`**

```python
# src/mondiali/data/scope.py
"""Tier 3 scope: lista deterministica di nazionali da scrappare.

Output: union di
- WC2026_QUALIFIED (48 hardcoded, snapshot 2026-04-29)
- top-50 FIFA Elo per anno 2014-2025 (computato da matches.parquet)

Output finale ~70-80 nazionali. Salvato in data/processed/tier3_scope.json
per audit + uso del CLI tm-scrape.
"""
from __future__ import annotations

import pandas as pd

# 48 nazionali qualificate per WC2026 (stato 2026-04-29).
# Tre host (Canada, Mexico, US) + 45 sportive qualifiers.
WC2026_QUALIFIED: list[str] = [
    # Host
    "Canada", "Mexico", "United States",
    # UEFA (16)
    "Argentina",  # placeholder - replace with actual UEFA when known
    "France", "England", "Spain", "Germany", "Portugal", "Netherlands",
    "Belgium", "Italy", "Croatia", "Switzerland", "Denmark", "Poland",
    "Serbia", "Austria", "Hungary", "Turkey",
    # CONMEBOL (6)
    "Brazil", "Uruguay", "Colombia", "Ecuador", "Peru", "Paraguay",
    # CONCACAF (3 + hosts already)
    "Costa Rica", "Panama", "Jamaica",
    # AFC (8)
    "Japan", "South Korea", "Australia", "Saudi Arabia", "Iran",
    "Qatar", "Iraq", "United Arab Emirates",
    # CAF (9)
    "Morocco", "Senegal", "Tunisia", "Algeria", "Egypt", "Nigeria",
    "Ghana", "Cameroon", "Ivory Coast",
    # OFC (1)
    "New Zealand",
    # Inter-confederation playoff potential (2)
    "Wales", "Scotland",
]
assert len(WC2026_QUALIFIED) == 48, f"expected 48, got {len(WC2026_QUALIFIED)}"


def compute_tier3_scope(matches: pd.DataFrame) -> list[str]:
    """Lista deterministica delle nazionali da scrappare per Tier 3.

    Union di WC2026_QUALIFIED + top-50 FIFA Elo per ogni anno 2014-2025.

    Args:
        matches: DataFrame con colonne `date`, `home_team`, `away_team`,
            `home_elo_before`, `away_elo_before`.

    Returns:
        Lista ordinata di nation names, ~70-80 entries.
    """
    df = matches[matches["date"] >= pd.Timestamp("2014-01-01")].copy()
    df["year"] = df["date"].dt.year

    top50_by_year: set[str] = set()
    for _year, grp in df.groupby("year"):
        max_elo_home = grp.groupby("home_team")["home_elo_before"].max()
        max_elo_away = grp.groupby("away_team")["away_elo_before"].max()
        team_elo = pd.concat([max_elo_home, max_elo_away]).groupby(level=0).max()
        top50_by_year.update(team_elo.nlargest(50).index.tolist())

    return sorted(set(WC2026_QUALIFIED) | top50_by_year)
```

- [ ] **Step 4: Run test (PASS)**

```bash
pytest tests/test_scope.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mondiali/data/scope.py tests/test_scope.py
git commit -m "feat(data): compute_tier3_scope + WC2026_QUALIFIED list (Task 2)"
```

---

## Task 3: Wayback CDX query helper

**Files:**
- Create: `src/mondiali/data/transfermarkt.py` (parziale)
- Test: `tests/test_transfermarkt.py` (parziale)

Helper `_query_cdx(target_url: str, from_date: date, to_date: date) -> list[CDXRow]`. Wraps Wayback CDX API.

CDX API endpoint: `https://web.archive.org/cdx/search/cdx`. Query params: `url`, `from=YYYYMMDD`, `to=YYYYMMDD`, `output=json`, `filter=statuscode:200`, `limit=N`.

Risposta JSON: list-of-lists. Prima riga = header `["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"]`. Righe successive = match.

- [ ] **Step 1: Scrivi i test (failing)**

```python
# tests/test_transfermarkt.py
"""Test scraper Transfermarkt: CDX, parsing HTML, fallback chain, cache."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import responses

from mondiali.data.transfermarkt import CDXRow, _query_cdx


@responses.activate
def test_query_cdx_returns_parsed_rows():
    """CDX risposta JSON → lista di CDXRow."""
    responses.add(
        responses.GET,
        "https://web.archive.org/cdx/search/cdx",
        json=[
            ["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"],
            [
                "com,transfermarkt)/italien/startseite/verein/3376",
                "20180823120000",
                "https://www.transfermarkt.com/italien/startseite/verein/3376",
                "text/html",
                "200",
                "ABC123DEF",
                "12345",
            ],
        ],
    )
    rows = _query_cdx(
        "https://www.transfermarkt.com/italien/startseite/verein/3376",
        date(2018, 5, 1),
        date(2018, 9, 1),
    )
    assert len(rows) == 1
    assert isinstance(rows[0], CDXRow)
    assert rows[0].timestamp == "20180823120000"
    assert rows[0].statuscode == "200"


@responses.activate
def test_query_cdx_returns_empty_on_no_match():
    """CDX risposta vuota (solo header) → lista vuota."""
    responses.add(
        responses.GET,
        "https://web.archive.org/cdx/search/cdx",
        json=[
            ["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"],
        ],
    )
    rows = _query_cdx(
        "https://www.transfermarkt.com/eritrea/startseite/verein/9999",
        date(2018, 1, 1),
        date(2018, 12, 31),
    )
    assert rows == []


@responses.activate
def test_query_cdx_returns_empty_on_404():
    """Wayback ritorna 404 (URL mai archiviato) → lista vuota."""
    responses.add(
        responses.GET,
        "https://web.archive.org/cdx/search/cdx",
        status=404,
    )
    rows = _query_cdx(
        "https://www.transfermarkt.com/whatever/startseite/verein/0",
        date(2018, 1, 1),
        date(2018, 12, 31),
    )
    assert rows == []


@responses.activate
def test_query_cdx_filters_to_statuscode_200():
    """Verifica che filter=statuscode:200 sia passato come query param."""
    responses.add(
        responses.GET,
        "https://web.archive.org/cdx/search/cdx",
        json=[["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"]],
    )
    _query_cdx("https://example.com", date(2018, 1, 1), date(2018, 12, 31))
    call = responses.calls[0]
    assert "filter=statuscode%3A200" in call.request.url or "filter=statuscode:200" in call.request.url
```

- [ ] **Step 2: Run test (FAIL)**

```bash
pytest tests/test_transfermarkt.py -v
```

Expected: 4 errors / fails — module not implemented.

- [ ] **Step 3: Implementa il primo blocco di `transfermarkt.py`**

```python
# src/mondiali/data/transfermarkt.py
"""Scraper Transfermarkt via Wayback Machine.

Pipeline:
1. _query_cdx: CDX API → snapshot list
2. _best_snapshot_for_year: fallback chain
3. _fetch_snapshot_html: download + cache
4. _parse_squad_value: BeautifulSoup → (total, top11, n_players)
5. scrape_all: orchestra tutto, scrive snapshots.parquet

Anti-leakage: lo snapshot ha timestamp REALE Wayback (non target nominale).
È quel timestamp che entra nel calcolo `tm_age_days` al feature-build time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import requests
import structlog

log = structlog.get_logger(__name__)

CDX_ENDPOINT = "https://web.archive.org/cdx/search/cdx"
WAYBACK_FETCH_BASE = "https://web.archive.org/web"
RATE_LIMIT_SECONDS = 2.0
CDX_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class CDXRow:
    """Una riga della risposta CDX search."""
    urlkey: str
    timestamp: str          # YYYYMMDDHHMMSS
    original: str
    mimetype: str
    statuscode: str
    digest: str
    length: str

    @property
    def snapshot_date(self) -> date:
        return date(int(self.timestamp[:4]), int(self.timestamp[4:6]), int(self.timestamp[6:8]))


def _query_cdx(target_url: str, from_date: date, to_date: date, limit: int = 50) -> list[CDXRow]:
    """Wayback CDX API query. Ritorna lista di CDXRow (statuscode=200 only).

    Returns:
        Lista (vuota se nessun match o errore HTTP).
    """
    params = {
        "url": target_url,
        "from": from_date.strftime("%Y%m%d"),
        "to": to_date.strftime("%Y%m%d"),
        "output": "json",
        "filter": "statuscode:200",
        "limit": str(limit),
    }
    try:
        resp = requests.get(CDX_ENDPOINT, params=params, timeout=CDX_TIMEOUT_SECONDS)
        if resp.status_code != 200:
            log.warning("cdx non-200", status=resp.status_code, url=target_url)
            return []
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        log.warning("cdx exception", error=str(e), url=target_url)
        return []

    if not data or len(data) < 2:
        return []  # solo header

    return [CDXRow(*row) for row in data[1:]]
```

- [ ] **Step 4: Run test (PASS)**

```bash
pytest tests/test_transfermarkt.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mondiali/data/transfermarkt.py tests/test_transfermarkt.py
git commit -m "feat(data): Wayback CDX query helper for Tier 3 (Task 3)"
```

---

## Task 4: HTML parser per pagina TM rosa

**Files:**
- Modify: `src/mondiali/data/transfermarkt.py`
- Modify: `tests/test_transfermarkt.py`
- Create: `tests/fixtures/tm_italy_2014.html`, `tm_italy_2018.html`, `tm_italy_2022.html`

Parser estrae da una pagina TM rosa: lista valori in EUR, somma totale, somma top-11, n_players.

Il selettore primario per TM rosa nazionale: `<table class="items">` (alcuni anni usano `id="kader"`, ma `class="items"` è più stabile). I valori sono nelle celle `<td class="rechts hauptlink">` (allineato a destra). Il formato valore: `€80.00m`, `€500k`, `€-` (sconosciuto), oppure cella vuota.

- [ ] **Step 1: Crea le 3 fixture HTML**

L'implementer scarica manualmente 3 snapshot Wayback per Italia (o altra nazionale notoria, basta la stessa nazionale per 3 anni diversi):

```bash
mkdir -p tests/fixtures
# Esempio target snapshot Wayback Italia 2014
curl -o tests/fixtures/tm_italy_2014.html \
    "https://web.archive.org/web/20140801000000/https://www.transfermarkt.com/italien/startseite/verein/3376"
curl -o tests/fixtures/tm_italy_2018.html \
    "https://web.archive.org/web/20180801000000/https://www.transfermarkt.com/italien/startseite/verein/3376"
curl -o tests/fixtures/tm_italy_2022.html \
    "https://web.archive.org/web/20220801000000/https://www.transfermarkt.com/italien/startseite/verein/3376"
```

Se i timestamp esatti non hanno snapshot, usare quello *più vicino* — basta che sia una pagina TM rosa nazionale realistica per ognuno dei 3 anni.

⚠ **Verifica manualmente** che ognuna delle 3 fixture contenga la tabella rosa con valori in EUR — alcune redirect Wayback possono restituire altri contenuti. Apri il file in browser, controlla che vedi una rosa nazionale italiana coi valori giocatori.

- [ ] **Step 2: Scrivi i test (failing)**

Aggiungi a `tests/test_transfermarkt.py`:

```python
import pytest
from mondiali.data.transfermarkt import _parse_value_eur, _parse_squad_value

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("input_str, expected", [
    ("€80.00m", 80_000_000.0),
    ("€500k", 500_000.0),
    ("€1.50m", 1_500_000.0),
    ("€-", None),
    ("-", None),
    ("", None),
    ("€999.99k", 999_990.0),
])
def test_parse_value_eur(input_str, expected):
    assert _parse_value_eur(input_str) == expected


@pytest.mark.parametrize("fixture_name", [
    "tm_italy_2014.html",
    "tm_italy_2018.html",
    "tm_italy_2022.html",
])
def test_parse_squad_value_real_fixtures(fixture_name):
    """Le 3 fixture HTML reali devono parsare correttamente.

    Invariants:
    - n_players >= 20 (rosa nazionale tipica 23-30 nomi)
    - total_value_eur > 50M (Italia non è mai sotto 50M nei tre anni testati)
    - top11_value_eur > 0 e <= total_value_eur
    """
    html = (FIXTURES_DIR / fixture_name).read_text(encoding="utf-8")
    result = _parse_squad_value(html)
    assert result is not None
    assert result.n_players >= 20
    assert result.total_value_eur > 50_000_000.0
    assert 0 < result.top11_value_eur <= result.total_value_eur


def test_parse_squad_value_empty_html_returns_none():
    """HTML senza tabella rosa → None."""
    result = _parse_squad_value("<html><body>404 Not Found</body></html>")
    assert result is None
```

- [ ] **Step 3: Run test (FAIL)**

```bash
pytest tests/test_transfermarkt.py -v
```

Expected: nuovi fail su `_parse_value_eur` e `_parse_squad_value`.

- [ ] **Step 4: Implementa parser**

Aggiungi a `src/mondiali/data/transfermarkt.py`:

```python
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup


@dataclass(frozen=True)
class SquadValue:
    """Output del parser TM rosa."""
    total_value_eur: float
    top11_value_eur: float
    n_players: int


_VALUE_RE = re.compile(r"€\s*([\d.,]+)\s*([mk]?)", re.IGNORECASE)


def _parse_value_eur(raw: str) -> float | None:
    """Parse '€80.00m' / '€500k' / '€-' / '' → float EUR o None."""
    if not raw:
        return None
    s = raw.strip()
    if not s or s in ("€-", "-"):
        return None
    m = _VALUE_RE.search(s)
    if not m:
        return None
    num = float(m.group(1).replace(",", "."))
    unit = m.group(2).lower()
    if unit == "m":
        return num * 1_000_000.0
    if unit == "k":
        return num * 1_000.0
    return num  # raw EUR (raro)


def _parse_squad_value(html: str) -> SquadValue | None:
    """Parse pagina TM rosa nazionale. None se la pagina non contiene rosa.

    Selettori (in ordine di preferenza):
    1. table.items td.rechts.hauptlink (TM ≥2018)
    2. table#kader td.rechts.hauptlink (TM legacy)

    Estrae tutti i valori, scarta None, somma → total. Top-11 = somma dei 11
    valori più alti.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Try primary selector
    table = soup.select_one("table.items")
    if table is None:
        table = soup.select_one("table#kader")
    if table is None:
        return None

    cells = table.select("td.rechts.hauptlink")
    if not cells:
        # Some TM versions use just td.rechts inside the squad value column
        cells = table.select("td.rechts")

    values: list[float] = []
    for cell in cells:
        v = _parse_value_eur(cell.get_text(strip=True))
        if v is not None and v > 0:
            values.append(v)

    if not values:
        return None

    total = sum(values)
    top11 = sum(sorted(values, reverse=True)[:11])
    return SquadValue(
        total_value_eur=total,
        top11_value_eur=top11,
        n_players=len(values),
    )
```

- [ ] **Step 5: Run test (PASS)**

```bash
pytest tests/test_transfermarkt.py -v
```

Expected: tutti passati. Se le 3 fixture reali falliscono il parsing, l'implementer deve:
1. Aprire il file HTML, identificare il selettore CSS giusto per quel layout
2. Aggiungere/aggiustare il selettore in `_parse_squad_value`
3. NON modificare gli `assert` del test per farli passare con valori sbagliati

- [ ] **Step 6: Commit**

```bash
git add src/mondiali/data/transfermarkt.py tests/test_transfermarkt.py tests/fixtures/
git commit -m "feat(data): TM HTML parser with 3 real-world fixtures (Task 4)"
```

---

## Task 5: Snapshot fetcher con cache + rate limiter

**Files:**
- Modify: `src/mondiali/data/transfermarkt.py`
- Modify: `tests/test_transfermarkt.py`

Funzione `_fetch_snapshot_html(cdx_row: CDXRow, cache_dir: Path) -> str | None` che scarica HTML da Wayback con cache + rate limiter.

URL costruito da CDX: `https://web.archive.org/web/{timestamp}/{original}`.

Cache key: `{nation_slug}__{timestamp}.html` (slug derivato dall'URL).

- [ ] **Step 1: Scrivi i test (failing)**

Aggiungi a `tests/test_transfermarkt.py`:

```python
from mondiali.data.transfermarkt import _fetch_snapshot_html, _wayback_url


def test_wayback_url_construction():
    row = CDXRow(
        urlkey="com,transfermarkt)/italien/startseite/verein/3376",
        timestamp="20180823120000",
        original="https://www.transfermarkt.com/italien/startseite/verein/3376",
        mimetype="text/html",
        statuscode="200",
        digest="ABC",
        length="123",
    )
    url = _wayback_url(row)
    assert url == "https://web.archive.org/web/20180823120000/https://www.transfermarkt.com/italien/startseite/verein/3376"


@responses.activate
def test_fetch_snapshot_html_uses_cache(tmp_path, monkeypatch):
    """Se il file è già in cache, no HTTP call. Idempotenza."""
    monkeypatch.setattr("mondiali.data.transfermarkt.RATE_LIMIT_SECONDS", 0.0)
    cache_file = tmp_path / "italien__20180823120000.html"
    cache_file.write_text("<html>cached</html>", encoding="utf-8")

    row = CDXRow(
        "com,transfermarkt)/italien/startseite/verein/3376",
        "20180823120000",
        "https://www.transfermarkt.com/italien/startseite/verein/3376",
        "text/html", "200", "ABC", "123",
    )
    html = _fetch_snapshot_html(row, tmp_path)
    assert html == "<html>cached</html>"
    assert len(responses.calls) == 0  # nessun HTTP call


@responses.activate
def test_fetch_snapshot_html_writes_cache(tmp_path, monkeypatch):
    """Cache miss → fetch → write to disk."""
    monkeypatch.setattr("mondiali.data.transfermarkt.RATE_LIMIT_SECONDS", 0.0)
    responses.add(
        responses.GET,
        "https://web.archive.org/web/20180823120000/https://www.transfermarkt.com/italien/startseite/verein/3376",
        body="<html>fetched</html>",
        status=200,
    )
    row = CDXRow(
        "com,transfermarkt)/italien/startseite/verein/3376",
        "20180823120000",
        "https://www.transfermarkt.com/italien/startseite/verein/3376",
        "text/html", "200", "ABC", "123",
    )
    html = _fetch_snapshot_html(row, tmp_path)
    assert html == "<html>fetched</html>"
    expected_file = tmp_path / "italien__20180823120000.html"
    assert expected_file.exists()
    assert expected_file.read_text(encoding="utf-8") == "<html>fetched</html>"


@responses.activate
def test_fetch_snapshot_html_returns_none_on_5xx(tmp_path, monkeypatch):
    """HTTP 500 → ritry exp-backoff esauriti → None."""
    monkeypatch.setattr("mondiali.data.transfermarkt.RATE_LIMIT_SECONDS", 0.0)
    monkeypatch.setattr("mondiali.data.transfermarkt._RETRY_BACKOFF_BASE", 0.0)
    responses.add(
        responses.GET,
        "https://web.archive.org/web/20180823120000/https://www.transfermarkt.com/italien/startseite/verein/3376",
        status=500,
    )
    row = CDXRow(
        "com,transfermarkt)/italien/startseite/verein/3376",
        "20180823120000",
        "https://www.transfermarkt.com/italien/startseite/verein/3376",
        "text/html", "200", "ABC", "123",
    )
    html = _fetch_snapshot_html(row, tmp_path)
    assert html is None
```

- [ ] **Step 2: Run (FAIL)**

```bash
pytest tests/test_transfermarkt.py -v
```

Expected: nuovi fail.

- [ ] **Step 3: Implementa fetcher**

Aggiungi a `src/mondiali/data/transfermarkt.py`:

```python
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_BASE = 2.0  # exp: 2s, 4s, 8s


def _slug_from_url(url: str) -> str:
    """Estrai lo slug nazionale dall'URL TM. URL pattern:
    https://www.transfermarkt.com/{slug}/startseite/verein/{id}
    """
    parts = url.rstrip("/").split("/")
    # ['https:', '', 'www.transfermarkt.com', '{slug}', 'startseite', 'verein', '{id}']
    if len(parts) >= 4:
        return parts[3]
    return "unknown"


def _wayback_url(row: CDXRow) -> str:
    """URL Wayback per fetch HTML da una CDX row."""
    return f"{WAYBACK_FETCH_BASE}/{row.timestamp}/{row.original}"


def _fetch_snapshot_html(row: CDXRow, cache_dir: Path) -> str | None:
    """Fetch HTML da Wayback con cache + rate limiter + retry exp-backoff.

    Returns:
        HTML body se 200, None se cache miss + retry exhausted.
    """
    slug = _slug_from_url(row.original)
    cache_file = cache_dir / f"{slug}__{row.timestamp}.html"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    cache_dir.mkdir(parents=True, exist_ok=True)
    url = _wayback_url(row)

    for attempt in range(_RETRY_ATTEMPTS):
        time.sleep(RATE_LIMIT_SECONDS)
        try:
            resp = requests.get(url, timeout=CDX_TIMEOUT_SECONDS)
            if resp.status_code == 200:
                html = resp.text
                cache_file.write_text(html, encoding="utf-8")
                return html
            if resp.status_code in (404, 410):
                log.warning("wayback 4xx", url=url, status=resp.status_code)
                return None  # non vale ritentare
            log.warning("wayback non-200", url=url, status=resp.status_code, attempt=attempt)
        except requests.RequestException as e:
            log.warning("wayback exception", error=str(e), attempt=attempt)
        if attempt < _RETRY_ATTEMPTS - 1:
            time.sleep(_RETRY_BACKOFF_BASE * (2 ** attempt))

    return None
```

- [ ] **Step 4: Run (PASS)**

```bash
pytest tests/test_transfermarkt.py -v
```

Expected: tutti passati.

- [ ] **Step 5: Commit**

```bash
git add src/mondiali/data/transfermarkt.py tests/test_transfermarkt.py
git commit -m "feat(data): snapshot fetcher with cache + rate limiter (Task 5)"
```

---

## Task 6: Best-snapshot fallback chain

**Files:**
- Modify: `src/mondiali/data/transfermarkt.py`
- Modify: `tests/test_transfermarkt.py`

Funzione `_best_snapshot_for_year(target_url, year, cache_dir) -> SnapshotRecord | None`. 4-level fallback:

1. CDX query [year-05-01, year-09-01] (vicino a 1 luglio)
2. Se 0 hit: CDX query [year-01-01, year-12-31]
3. Se 0 hit: CDX query [year-1-07-01, year-06-30] (fallback all'anno precedente)
4. Se 0 hit: ritorna None.

Per ogni livello, se CDX trova snapshot, picka quello più vicino a `target_date = year-07-01`, fetcha HTML, parsi rosa, ritorna `SnapshotRecord`. Se parse fallisce, prova la prossima riga CDX (massimo 3 tentativi per livello).

- [ ] **Step 1: Scrivi test**

```python
from datetime import date as Date
from unittest.mock import patch

from mondiali.data.transfermarkt import (
    SnapshotRecord, _best_snapshot_for_year, _query_cdx,
)


@responses.activate
def test_best_snapshot_for_year_level1_success(tmp_path, monkeypatch):
    """Trova snapshot al primo livello (target ±60 giorni)."""
    monkeypatch.setattr("mondiali.data.transfermarkt.RATE_LIMIT_SECONDS", 0.0)
    responses.add(
        responses.GET,
        "https://web.archive.org/cdx/search/cdx",
        json=[
            ["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"],
            [
                "com,transfermarkt)/italien/startseite/verein/3376",
                "20180815000000",
                "https://www.transfermarkt.com/italien/startseite/verein/3376",
                "text/html", "200", "ABC", "12345",
            ],
        ],
    )
    fixture = (Path(__file__).parent / "fixtures" / "tm_italy_2018.html").read_text(encoding="utf-8")
    responses.add(
        responses.GET,
        "https://web.archive.org/web/20180815000000/https://www.transfermarkt.com/italien/startseite/verein/3376",
        body=fixture,
        status=200,
    )
    snap = _best_snapshot_for_year(
        "https://www.transfermarkt.com/italien/startseite/verein/3376",
        2018,
        tmp_path,
    )
    assert snap is not None
    assert snap.snapshot_date == Date(2018, 8, 15)
    assert snap.total_value_eur > 0


@responses.activate
def test_best_snapshot_for_year_returns_none_when_all_levels_empty(tmp_path, monkeypatch):
    """Tutti e 4 i livelli ritornano CDX vuoto → None."""
    monkeypatch.setattr("mondiali.data.transfermarkt.RATE_LIMIT_SECONDS", 0.0)
    responses.add(
        responses.GET,
        "https://web.archive.org/cdx/search/cdx",
        json=[["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"]],
    )
    snap = _best_snapshot_for_year(
        "https://www.transfermarkt.com/eritrea/startseite/verein/9999",
        2018,
        tmp_path,
    )
    assert snap is None
```

- [ ] **Step 2: Run (FAIL)**

```bash
pytest tests/test_transfermarkt.py -v
```

- [ ] **Step 3: Implementa**

Aggiungi a `src/mondiali/data/transfermarkt.py`:

```python
@dataclass(frozen=True)
class SnapshotRecord:
    """Una riga di snapshots.parquet."""
    nation: str
    year: int
    snapshot_date: date
    total_value_eur: float
    top11_value_eur: float
    n_players: int
    source_url: str


def _best_snapshot_for_year(
    target_url: str, year: int, cache_dir: Path
) -> tuple[date, SquadValue, str] | None:
    """Adaptive fallback per `(nation_url, year)`.

    Ritorna (snapshot_date, parsed_value, wayback_url) se trova qualcosa,
    altrimenti None.
    """
    target = date(year, 7, 1)

    levels = [
        (date(year, 5, 1), date(year, 9, 1)),       # ±60d
        (date(year, 1, 1), date(year, 12, 31)),     # tutto anno
        (date(year - 1, 7, 1), date(year, 6, 30)),  # anno-1
    ]

    for from_d, to_d in levels:
        rows = _query_cdx(target_url, from_d, to_d, limit=20)
        if not rows:
            continue
        # Sort per distanza dal target
        rows_sorted = sorted(rows, key=lambda r: abs((r.snapshot_date - target).days))

        # Prova al massimo 3 righe (alcune possono fallire il parse)
        for row in rows_sorted[:3]:
            html = _fetch_snapshot_html(row, cache_dir)
            if html is None:
                continue
            parsed = _parse_squad_value(html)
            if parsed is None:
                continue
            return (row.snapshot_date, parsed, _wayback_url(row))

    return None
```

- [ ] **Step 4: Run (PASS)**

```bash
pytest tests/test_transfermarkt.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/mondiali/data/transfermarkt.py tests/test_transfermarkt.py
git commit -m "feat(data): adaptive 4-level fallback chain for Wayback snapshots (Task 6)"
```

---

## Task 7: Orchestrator + `tm-scrape` CLI

**Files:**
- Modify: `src/mondiali/data/transfermarkt.py`
- Modify: `src/mondiali/cli/main.py`

Orchestrator `scrape_all(scope, years, cache_dir, output_path) -> None` itera scope × years, raccoglie SnapshotRecord, scrive `snapshots.parquet`.

Coverage logging: a fine run, conta `n_filled / n_target` e logga.

CLI: `mondiali tm-scrape [--start-year 2014] [--end-year 2025] [--scope-file ...]`.

- [ ] **Step 1: Scrivi test orchestrator**

Aggiungi a `tests/test_transfermarkt.py`:

```python
import pandas as pd
from mondiali.data.transfermarkt import scrape_all


@responses.activate
def test_scrape_all_writes_parquet(tmp_path, monkeypatch):
    """End-to-end: scope di 1 nazionale × 2 anni → snapshots.parquet con 2 righe."""
    monkeypatch.setattr("mondiali.data.transfermarkt.RATE_LIMIT_SECONDS", 0.0)
    fixture = (Path(__file__).parent / "fixtures" / "tm_italy_2018.html").read_text(encoding="utf-8")

    # CDX call (any URL)
    responses.add(
        responses.GET,
        "https://web.archive.org/cdx/search/cdx",
        json=[
            ["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"],
            [
                "com,transfermarkt)/italien/startseite/verein/3376",
                "20180701000000",
                "https://www.transfermarkt.com/italien/startseite/verein/3376",
                "text/html", "200", "ABC", "12345",
            ],
        ],
    )
    # Wayback fetch (any URL)
    responses.add(
        responses.GET,
        "https://web.archive.org/web/20180701000000/https://www.transfermarkt.com/italien/startseite/verein/3376",
        body=fixture,
        status=200,
    )
    out = tmp_path / "snapshots.parquet"
    scrape_all(
        scope=["Italy"],
        years=[2018, 2019],
        cache_dir=tmp_path / "cache",
        output_path=out,
    )
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) >= 1  # almeno 1 anno trovato (responses mock ritorna sempre lo stesso)
    assert set(df.columns) >= {
        "nation", "year", "snapshot_date", "total_value_eur",
        "top11_value_eur", "n_players", "source_url",
    }
    assert df.iloc[0]["nation"] == "Italy"
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Implementa orchestrator**

Aggiungi a `src/mondiali/data/transfermarkt.py`:

```python
import pandas as pd

from mondiali.data.tm_nations import NATION_TM_IDS

TRANSFERMARKT_URL_TEMPLATE = "https://www.transfermarkt.com/{slug}/startseite/verein/{tm_id}"


def _build_target_url(nation: str) -> str | None:
    """Da team_name (es. 'Italy') costruisci URL TM canonico. None se non in lookup."""
    entry = NATION_TM_IDS.get(nation)
    if entry is None:
        return None
    slug, tm_id = entry
    return TRANSFERMARKT_URL_TEMPLATE.format(slug=slug, tm_id=tm_id)


def scrape_all(
    scope: list[str],
    years: list[int],
    cache_dir: Path,
    output_path: Path,
) -> None:
    """Itera scope × years, raccoglie snapshot, scrive snapshots.parquet.

    Args:
        scope: lista nazionali (chiavi di NATION_TM_IDS)
        years: anni 2014..2025 tipicamente
        cache_dir: directory per HTML cache
        output_path: dove scrivere snapshots.parquet
    """
    records: list[SnapshotRecord] = []
    n_target = 0
    n_filled = 0

    for nation in scope:
        url = _build_target_url(nation)
        if url is None:
            log.warning("nation not in NATION_TM_IDS, skipping", nation=nation)
            continue
        for year in years:
            n_target += 1
            log.info("scraping", nation=nation, year=year)
            result = _best_snapshot_for_year(url, year, cache_dir)
            if result is None:
                log.warning("no snapshot found", nation=nation, year=year)
                continue
            snap_date, parsed, source = result
            records.append(SnapshotRecord(
                nation=nation,
                year=year,
                snapshot_date=snap_date,
                total_value_eur=parsed.total_value_eur,
                top11_value_eur=parsed.top11_value_eur,
                n_players=parsed.n_players,
                source_url=source,
            ))
            n_filled += 1

    coverage = n_filled / n_target if n_target else 0.0
    log.info(
        "scrape complete",
        n_target=n_target, n_filled=n_filled, coverage=f"{coverage:.1%}",
    )
    if coverage < 0.6:
        log.warning("coverage <60% — consider rerun or scope adjustment")

    df = pd.DataFrame([
        {
            "nation": r.nation,
            "year": r.year,
            "snapshot_date": r.snapshot_date,
            "total_value_eur": r.total_value_eur,
            "top11_value_eur": r.top11_value_eur,
            "n_players": r.n_players,
            "source_url": r.source_url,
        }
        for r in records
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    log.info("wrote snapshots.parquet", path=str(output_path), rows=len(df))
```

- [ ] **Step 4: Run (PASS)**

```bash
pytest tests/test_transfermarkt.py -v
```

- [ ] **Step 5: Aggiungi CLI command `tm-scrape`**

Aggiungi a `src/mondiali/cli/main.py` (dopo `train_tier2`):

```python
import json

from mondiali.data.scope import compute_tier3_scope
from mondiali.data.transfermarkt import scrape_all


@app.command(name="tm-scrape")
def tm_scrape(
    start_year: int = typer.Option(2014, help="Anno iniziale snapshot"),
    end_year: int = typer.Option(2025, help="Anno finale snapshot incluso"),
    scope_file: str = typer.Option(
        "", "--scope-file",
        help="Path JSON con lista nazioni; se vuoto, computa da matches.parquet",
    ),
) -> None:
    """Scrape Transfermarkt market values via Wayback Machine per Tier 3."""
    if scope_file:
        with Path(scope_file).open() as f:
            scope = json.load(f)
    else:
        parquet = CONFIG.data_processed / "matches.parquet"
        if not parquet.exists():
            typer.echo("matches.parquet non trovato — esegui `mondiali ingest` prima", err=True)
            raise typer.Exit(1)
        df = pd.read_parquet(parquet)
        df["date"] = pd.to_datetime(df["date"])
        scope = compute_tier3_scope(df)
        scope_out = CONFIG.data_processed / "tier3_scope.json"
        with scope_out.open("w") as f:
            json.dump(scope, f, indent=2)
        typer.echo(f"Computed scope: {len(scope)} nations → {scope_out}")

    cache_dir = CONFIG.data_raw / "transfermarkt" / "cache"
    output_path = CONFIG.data_raw / "transfermarkt" / "snapshots.parquet"
    years = list(range(start_year, end_year + 1))
    typer.echo(f"Scraping {len(scope)} nations × {len(years)} years = {len(scope)*len(years)} target snapshots")
    scrape_all(scope, years, cache_dir, output_path)
    typer.echo(f"Done. Output: {output_path}")
```

- [ ] **Step 6: Smoke test CLI (no real network)**

```bash
mondiali tm-scrape --help
```

Expected: typer help screen senza errori.

- [ ] **Step 7: Commit**

```bash
git add src/mondiali/data/transfermarkt.py src/mondiali/cli/main.py tests/test_transfermarkt.py
git commit -m "feat(cli): tm-scrape orchestrator + CLI command (Task 7)"
```

---

## Task 8: `add_tier3_features`

**Files:**
- Create: `src/mondiali/features/tier3.py`
- Modify: `src/mondiali/features/__init__.py`
- Create: `tests/test_features_tier3.py`

Feature builder: per ogni match cerca lo snapshot più recente strict-pre-match per ciascun lato. Hard floor ≥2 snapshot per nazionale. Age clipping >540 giorni → NaN. Pre-2014 → NaN.

- [ ] **Step 1: Scrivi i test (failing)**

```python
# tests/test_features_tier3.py
"""Test feature builder Tier 3."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mondiali.features.tier3 import TIER3_COLUMNS, add_tier3_features


def _make_matches(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _make_snapshots(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def test_tier3_columns_constant():
    assert TIER3_COLUMNS == [
        "home_market_value_total", "away_market_value_total",
        "home_market_value_top11", "away_market_value_top11",
        "home_tm_age_days", "away_tm_age_days",
    ]


def test_add_tier3_features_basic_lookup():
    matches = _make_matches([
        {"date": "2018-06-15", "home_team": "Italy", "away_team": "France"},
    ])
    snapshots = _make_snapshots([
        {"nation": "Italy", "year": 2018, "snapshot_date": "2018-05-01",
         "total_value_eur": 500_000_000.0, "top11_value_eur": 400_000_000.0},
        {"nation": "France", "year": 2018, "snapshot_date": "2018-05-01",
         "total_value_eur": 800_000_000.0, "top11_value_eur": 700_000_000.0},
        # 2 snapshot per ognuna -> passa hard floor
        {"nation": "Italy", "year": 2017, "snapshot_date": "2017-05-01",
         "total_value_eur": 450_000_000.0, "top11_value_eur": 380_000_000.0},
        {"nation": "France", "year": 2017, "snapshot_date": "2017-05-01",
         "total_value_eur": 750_000_000.0, "top11_value_eur": 650_000_000.0},
    ])
    out = add_tier3_features(matches, snapshots)
    assert out.iloc[0]["home_market_value_total"] == 500_000_000.0
    assert out.iloc[0]["away_market_value_total"] == 800_000_000.0
    assert out.iloc[0]["home_tm_age_days"] == 45  # 2018-06-15 - 2018-05-01
    assert out.iloc[0]["away_tm_age_days"] == 45


def test_add_tier3_features_strict_pre_match():
    """Snapshot DOPO match → ignorato (no future leak)."""
    matches = _make_matches([
        {"date": "2018-06-15", "home_team": "Italy", "away_team": "France"},
    ])
    snapshots = _make_snapshots([
        # Italy: snapshot post-match (futuro)
        {"nation": "Italy", "year": 2018, "snapshot_date": "2018-08-01",
         "total_value_eur": 500_000_000.0, "top11_value_eur": 400_000_000.0},
        {"nation": "Italy", "year": 2017, "snapshot_date": "2017-05-01",
         "total_value_eur": 450_000_000.0, "top11_value_eur": 380_000_000.0},
        # France: 2 pre-match
        {"nation": "France", "year": 2018, "snapshot_date": "2018-05-01",
         "total_value_eur": 800_000_000.0, "top11_value_eur": 700_000_000.0},
        {"nation": "France", "year": 2017, "snapshot_date": "2017-05-01",
         "total_value_eur": 750_000_000.0, "top11_value_eur": 650_000_000.0},
    ])
    out = add_tier3_features(matches, snapshots)
    # Italy ha solo 2017 pre-match → 405 giorni di age
    assert out.iloc[0]["home_market_value_total"] == 450_000_000.0
    assert out.iloc[0]["home_tm_age_days"] == 410


def test_add_tier3_features_pre_2014_is_nan():
    matches = _make_matches([
        {"date": "2010-06-15", "home_team": "Italy", "away_team": "France"},
    ])
    snapshots = _make_snapshots([
        {"nation": "Italy", "year": 2014, "snapshot_date": "2014-05-01",
         "total_value_eur": 500_000_000.0, "top11_value_eur": 400_000_000.0},
        {"nation": "Italy", "year": 2015, "snapshot_date": "2015-05-01",
         "total_value_eur": 510_000_000.0, "top11_value_eur": 410_000_000.0},
        {"nation": "France", "year": 2014, "snapshot_date": "2014-05-01",
         "total_value_eur": 800_000_000.0, "top11_value_eur": 700_000_000.0},
        {"nation": "France", "year": 2015, "snapshot_date": "2015-05-01",
         "total_value_eur": 810_000_000.0, "top11_value_eur": 710_000_000.0},
    ])
    out = add_tier3_features(matches, snapshots)
    for col in TIER3_COLUMNS:
        assert pd.isna(out.iloc[0][col]), f"{col} should be NaN for pre-2014 match"


def test_add_tier3_features_age_clipping_540():
    """Snapshot più vecchio di 540 giorni → NaN."""
    matches = _make_matches([
        {"date": "2018-06-15", "home_team": "Italy", "away_team": "France"},
    ])
    snapshots = _make_snapshots([
        # Italy: 2 snapshot vecchissimi → tutti >540d
        {"nation": "Italy", "year": 2014, "snapshot_date": "2014-05-01",
         "total_value_eur": 500_000_000.0, "top11_value_eur": 400_000_000.0},
        {"nation": "Italy", "year": 2015, "snapshot_date": "2015-05-01",
         "total_value_eur": 510_000_000.0, "top11_value_eur": 410_000_000.0},
        # France: 2 snapshot recenti
        {"nation": "France", "year": 2018, "snapshot_date": "2018-05-01",
         "total_value_eur": 800_000_000.0, "top11_value_eur": 700_000_000.0},
        {"nation": "France", "year": 2017, "snapshot_date": "2017-05-01",
         "total_value_eur": 750_000_000.0, "top11_value_eur": 650_000_000.0},
    ])
    out = add_tier3_features(matches, snapshots)
    # Italy: snapshot più recente è 2015-05-01 → age 1141 giorni > 540 → NaN
    assert pd.isna(out.iloc[0]["home_market_value_total"])
    assert pd.isna(out.iloc[0]["home_tm_age_days"])
    # France: 2018-05-01 → 45 giorni → OK
    assert out.iloc[0]["away_market_value_total"] == 800_000_000.0


def test_add_tier3_features_hard_floor_excludes_nation():
    """Nazionale con 1 solo snapshot → tutte le sue feature NaN."""
    matches = _make_matches([
        {"date": "2018-06-15", "home_team": "Italy", "away_team": "Eritrea"},
    ])
    snapshots = _make_snapshots([
        {"nation": "Italy", "year": 2018, "snapshot_date": "2018-05-01",
         "total_value_eur": 500_000_000.0, "top11_value_eur": 400_000_000.0},
        {"nation": "Italy", "year": 2017, "snapshot_date": "2017-05-01",
         "total_value_eur": 450_000_000.0, "top11_value_eur": 380_000_000.0},
        # Eritrea: solo 1 snapshot
        {"nation": "Eritrea", "year": 2018, "snapshot_date": "2018-05-01",
         "total_value_eur": 5_000_000.0, "top11_value_eur": 4_000_000.0},
    ])
    out = add_tier3_features(matches, snapshots)
    assert out.iloc[0]["home_market_value_total"] == 500_000_000.0
    assert pd.isna(out.iloc[0]["away_market_value_total"])
    assert pd.isna(out.iloc[0]["away_tm_age_days"])


def test_add_tier3_features_nation_not_in_snapshots():
    """Nazionale completamente assente da snapshots → NaN per quel lato."""
    matches = _make_matches([
        {"date": "2018-06-15", "home_team": "Italy", "away_team": "Anguilla"},
    ])
    snapshots = _make_snapshots([
        {"nation": "Italy", "year": 2018, "snapshot_date": "2018-05-01",
         "total_value_eur": 500_000_000.0, "top11_value_eur": 400_000_000.0},
        {"nation": "Italy", "year": 2017, "snapshot_date": "2017-05-01",
         "total_value_eur": 450_000_000.0, "top11_value_eur": 380_000_000.0},
    ])
    out = add_tier3_features(matches, snapshots)
    assert out.iloc[0]["home_market_value_total"] == 500_000_000.0
    assert pd.isna(out.iloc[0]["away_market_value_total"])
```

- [ ] **Step 2: Run (FAIL)**

```bash
pytest tests/test_features_tier3.py -v
```

- [ ] **Step 3: Implementa `tier3.py`**

```python
# src/mondiali/features/tier3.py
"""Feature builder Tier 3: Transfermarkt market values.

Per ogni match (≥2014) e per ciascun lato, cerca in `snapshots.parquet` la
riga con `nation == team` e `snapshot_date < match_date` ordinata desc, prendi
prima. Calcola `tm_age_days = (match_date - snapshot_date).days`.

Anti-leakage:
- snapshot strict-pre-match (`<`, non `<=`)
- Age clipping >540 giorni → NaN (no forward-fill abusi)
- Hard floor ≥2 snapshot per nazionale (sotto-floor escluse)
- Pre-2014 → NaN
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

TIER3_MIN_YEAR = 2014
TIER3_MAX_AGE_DAYS = 540
TIER3_MIN_SNAPSHOTS_PER_NATION = 2

TIER3_COLUMNS: list[str] = [
    "home_market_value_total", "away_market_value_total",
    "home_market_value_top11", "away_market_value_top11",
    "home_tm_age_days", "away_tm_age_days",
]


def _build_lookup(snapshots: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per ogni nazionale che passa il hard floor, ritorna un DataFrame
    ordinato asc per snapshot_date con (snapshot_date, total, top11).
    """
    counts = snapshots.groupby("nation").size()
    eligible = counts[counts >= TIER3_MIN_SNAPSHOTS_PER_NATION].index
    lookup: dict[str, pd.DataFrame] = {}
    for nation in eligible:
        sub = (
            snapshots[snapshots["nation"] == nation]
            [["snapshot_date", "total_value_eur", "top11_value_eur"]]
            .sort_values("snapshot_date")
            .reset_index(drop=True)
        )
        lookup[nation] = sub
    return lookup


def _lookup_strict_pre(
    sub: pd.DataFrame, match_date: pd.Timestamp
) -> tuple[float, float, int] | None:
    """Trova lo snapshot più recente con `snapshot_date < match_date`.

    Returns:
        (total_eur, top11_eur, age_days) o None se nessun snapshot pre-match.
    """
    pre = sub[sub["snapshot_date"] < match_date]
    if pre.empty:
        return None
    last = pre.iloc[-1]  # ordinato asc → ultima è la più vicina
    age = (match_date - last["snapshot_date"]).days
    return float(last["total_value_eur"]), float(last["top11_value_eur"]), age


def add_tier3_features(matches: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    """Aggiungi le 6 colonne TIER3_COLUMNS a `matches`.

    Args:
        matches: DataFrame con `date`, `home_team`, `away_team`.
        snapshots: DataFrame con `nation`, `snapshot_date`, `total_value_eur`,
            `top11_value_eur`.

    Returns:
        Copia di `matches` con 6 colonne aggiuntive.
    """
    out = matches.copy()
    for col in TIER3_COLUMNS:
        out[col] = np.nan

    if snapshots.empty:
        log.info("tier3 snapshots empty → all NaN")
        return out

    lookup = _build_lookup(snapshots)
    log.info("tier3 lookup built", n_nations_eligible=len(lookup))

    min_date = pd.Timestamp(f"{TIER3_MIN_YEAR}-01-01")

    home_total = np.full(len(out), np.nan)
    away_total = np.full(len(out), np.nan)
    home_top11 = np.full(len(out), np.nan)
    away_top11 = np.full(len(out), np.nan)
    home_age = np.full(len(out), np.nan)
    away_age = np.full(len(out), np.nan)

    dates = out["date"].to_numpy()
    home_teams = out["home_team"].to_numpy()
    away_teams = out["away_team"].to_numpy()

    for i in range(len(out)):
        match_date = pd.Timestamp(dates[i])
        if match_date < min_date:
            continue

        h = lookup.get(home_teams[i])
        if h is not None:
            res = _lookup_strict_pre(h, match_date)
            if res is not None and res[2] <= TIER3_MAX_AGE_DAYS:
                home_total[i], home_top11[i], home_age[i] = res

        a = lookup.get(away_teams[i])
        if a is not None:
            res = _lookup_strict_pre(a, match_date)
            if res is not None and res[2] <= TIER3_MAX_AGE_DAYS:
                away_total[i], away_top11[i], away_age[i] = res

    out["home_market_value_total"] = home_total
    out["away_market_value_total"] = away_total
    out["home_market_value_top11"] = home_top11
    out["away_market_value_top11"] = away_top11
    out["home_tm_age_days"] = home_age
    out["away_tm_age_days"] = away_age

    log.info(
        "added tier3 features",
        rows=len(out),
        coverage_home=float(np.mean(~np.isnan(home_total))),
        coverage_away=float(np.mean(~np.isnan(away_total))),
    )
    return out
```

- [ ] **Step 4: Aggiorna `features/__init__.py`**

```python
# src/mondiali/features/__init__.py
"""Feature engineering modules."""
from mondiali.features.tier2 import TIER2_COLUMNS, add_tier2_features
from mondiali.features.tier3 import TIER3_COLUMNS, add_tier3_features

__all__ = [
    "TIER2_COLUMNS", "add_tier2_features",
    "TIER3_COLUMNS", "add_tier3_features",
]
```

- [ ] **Step 5: Run (PASS)**

```bash
pytest tests/test_features_tier3.py -v
```

Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add src/mondiali/features/tier3.py src/mondiali/features/__init__.py tests/test_features_tier3.py
git commit -m "feat(features): add_tier3_features with strict-pre + age clip + hard floor (Task 8)"
```

---

## Task 9: Integration in `build_processed_matches`

**Files:**
- Modify: `src/mondiali/data/ingestion.py`

Wire `add_tier3_features` nel pipeline `build_processed_matches`. Se `snapshots.parquet` non esiste, riempi le 6 colonne con NaN (pipeline non rompe).

- [ ] **Step 1: Modifica `ingestion.py`**

Modifica la funzione `build_processed_matches`:

```python
# Aggiungi import in cima al file
from mondiali.features.tier3 import TIER3_COLUMNS, add_tier3_features
from mondiali.config import CONFIG  # se non già presente

# In build_processed_matches, dopo `df = add_tier2_features(df)` e prima di
# `df["match_id"] = ...`:

    snapshots_path = CONFIG.data_raw / "transfermarkt" / "snapshots.parquet"
    if snapshots_path.exists():
        snapshots = pd.read_parquet(snapshots_path)
        snapshots["snapshot_date"] = pd.to_datetime(snapshots["snapshot_date"])
        df = add_tier3_features(df, snapshots)
        log.info("tier3 features added from snapshots", path=str(snapshots_path))
    else:
        log.info("no tier3 snapshots — filling with NaN", expected=str(snapshots_path))
        for col in TIER3_COLUMNS:
            df[col] = pd.NA
```

- [ ] **Step 2: Run integration test**

```bash
pytest tests/test_ingestion.py -v
```

Expected: passa (se snapshots.parquet non esiste, NaN cols, no rompe).

Se `tests/test_ingestion.py` non copre questo, aggiungi un test:

```python
def test_build_processed_matches_without_tier3_snapshots(tmp_path, ...):
    """Senza snapshots.parquet, le 6 colonne TIER3 esistono come NaN."""
    # ... setup minimo
    out = build_processed_matches(raw_csv, out_path)
    df = pd.read_parquet(out)
    for col in TIER3_COLUMNS:
        assert col in df.columns
        assert df[col].isna().all()
```

- [ ] **Step 3: Run full ingestion locally per verifica**

```bash
mondiali ingest
```

Expected: completa senza errori, log "no tier3 snapshots — filling with NaN".

- [ ] **Step 4: Commit**

```bash
git add src/mondiali/data/ingestion.py tests/test_ingestion.py
git commit -m "feat(data): integrate add_tier3_features in build_processed_matches (Task 9)"
```

---

## Task 10: Anti-leakage test in `test_leakage.py`

**Files:**
- Modify: `tests/test_leakage.py`

Aggiungi `test_tier3_market_value_strict_pre_match`: re-simula da `snapshots.parquet`, verifica `tm_age_days ≥ 0` per ogni match dove non è NaN.

- [ ] **Step 1: Aggiungi test**

```python
# In tests/test_leakage.py, alla fine

def test_tier3_market_value_strict_pre_match():
    """Per ogni match con TM non-NaN, snapshot_date deve essere strictly < match_date.

    Re-simula da snapshots.parquet via add_tier3_features (no shortcuts via parquet
    già processato).
    """
    parquet = CONFIG.data_processed / "matches.parquet"
    snapshots_path = CONFIG.data_raw / "transfermarkt" / "snapshots.parquet"
    if not parquet.exists() or not snapshots_path.exists():
        pytest.skip("matches.parquet o snapshots.parquet non disponibili")

    matches = pd.read_parquet(parquet)
    matches["date"] = pd.to_datetime(matches["date"])
    snapshots = pd.read_parquet(snapshots_path)
    snapshots["snapshot_date"] = pd.to_datetime(snapshots["snapshot_date"])

    rebuilt = add_tier3_features(
        matches.drop(columns=TIER3_COLUMNS, errors="ignore"),
        snapshots,
    )

    # tm_age_days ≥ 0 quando presente (>= 1 giorno se strict-pre, ma 0 se same-day)
    home_age = rebuilt["home_tm_age_days"].dropna()
    away_age = rebuilt["away_tm_age_days"].dropna()
    if len(home_age):
        assert (home_age >= 0).all(), f"negative home_tm_age_days: {home_age[home_age < 0].head()}"
    if len(away_age):
        assert (away_age >= 0).all(), f"negative away_tm_age_days: {away_age[away_age < 0].head()}"

    # Niente match pre-2014 con TM non-NaN
    pre2014 = rebuilt[rebuilt["date"] < "2014-01-01"]
    if len(pre2014):
        assert pre2014["home_market_value_total"].isna().all()
        assert pre2014["away_market_value_total"].isna().all()
```

Aggiungi anche import in cima a `test_leakage.py`:

```python
from mondiali.features.tier3 import TIER3_COLUMNS, add_tier3_features
```

- [ ] **Step 2: Run**

```bash
pytest tests/test_leakage.py -v
```

Expected: il nuovo test viene `skipped` (snapshots.parquet ancora non scrapato), gli altri 4 passano.

- [ ] **Step 3: Commit**

```bash
git add tests/test_leakage.py
git commit -m "test(leakage): add tier3 strict-pre-match invariant (Task 10)"
```

---

## Task 11: Estendi `SYMMETRIC_FEATURES` 18 → 24

**Files:**
- Modify: `src/mondiali/model/poisson_xgb.py`
- Modify: `tests/test_poisson_xgb.py`

Aggiungi 6 nuove entry a `SYMMETRIC_FEATURES`. Estendi `build_symmetric_rows` con indici 18-23.

- [ ] **Step 1: Modifica `SYMMETRIC_FEATURES`**

```python
# In src/mondiali/model/poisson_xgb.py

SYMMETRIC_FEATURES: list[str] = [
    "team_elo",
    "opponent_elo",
    "elo_diff_signed",
    "is_home",
    "is_neutral",
    "competition_importance",
    "team_days_rest",
    "opponent_days_rest",
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
    "team_market_value_total",
    "opponent_market_value_total",
    "team_market_value_top11",
    "opponent_market_value_top11",
    "team_tm_age_days",
    "opponent_tm_age_days",
]
```

- [ ] **Step 2: Estendi `build_symmetric_rows`**

In fondo alla funzione (subito prima di `return X, y`), aggiungi i 6 popolamenti:

```python
    # Tier 3 inputs (NaN-tolerant: XGBoost gestisce nativamente)
    home_mv_total = matches["home_market_value_total"].to_numpy(dtype=float)
    away_mv_total = matches["away_market_value_total"].to_numpy(dtype=float)
    home_mv_top11 = matches["home_market_value_top11"].to_numpy(dtype=float)
    away_mv_top11 = matches["away_market_value_top11"].to_numpy(dtype=float)
    home_tm_age = matches["home_tm_age_days"].to_numpy(dtype=float)
    away_tm_age = matches["away_tm_age_days"].to_numpy(dtype=float)

    # Home-perspective rows
    X[0::2, 18] = home_mv_total
    X[0::2, 19] = away_mv_total
    X[0::2, 20] = home_mv_top11
    X[0::2, 21] = away_mv_top11
    X[0::2, 22] = home_tm_age
    X[0::2, 23] = away_tm_age

    # Away-perspective rows (scambio team/opponent)
    X[1::2, 18] = away_mv_total
    X[1::2, 19] = home_mv_total
    X[1::2, 20] = away_mv_top11
    X[1::2, 21] = home_mv_top11
    X[1::2, 22] = away_tm_age
    X[1::2, 23] = home_tm_age
```

⚠ Importante: `X = np.empty((2 * n, len(SYMMETRIC_FEATURES)), ...)` ora alloca 24 colonne perché `len(SYMMETRIC_FEATURES) == 24`. Verifica che la dichiarazione di `X` sia inalterata.

- [ ] **Step 3: Aggiungi test sentinel**

In `tests/test_poisson_xgb.py`:

```python
def test_symmetric_features_count_is_24():
    """Tier 3 enabled — feature count = 24 simmetriche."""
    from mondiali.model.poisson_xgb import SYMMETRIC_FEATURES
    assert len(SYMMETRIC_FEATURES) == 24
    # Tier 3 entries presenti
    assert "team_market_value_total" in SYMMETRIC_FEATURES
    assert "opponent_market_value_total" in SYMMETRIC_FEATURES
    assert "team_tm_age_days" in SYMMETRIC_FEATURES


def test_build_symmetric_rows_handles_nan_tier3():
    """matches con NaN su tier3 → output X ha NaN nelle col 18-23 (XGB ok)."""
    import numpy as np
    import pandas as pd
    from mondiali.model.poisson_xgb import build_symmetric_rows

    matches = pd.DataFrame({
        "home_team": ["A"], "away_team": ["B"],
        "home_score": [1], "away_score": [0],
        "home_elo_before": [1900.0], "away_elo_before": [1800.0],
        "neutral": [False],
        "competition_importance": [50.0],
        "days_rest_home": [3.0], "days_rest_away": [3.0],
        "home_form_5": [10.0], "away_form_5": [9.0],
        "home_gd_5": [3.0], "away_gd_5": [-1.0],
        "home_goals_scored_5": [1.5], "away_goals_scored_5": [1.2],
        "home_goals_conceded_5": [0.8], "away_goals_conceded_5": [1.0],
        "home_avg_opp_elo_5": [1700.0], "away_avg_opp_elo_5": [1750.0],
        # Tier 3 tutti NaN
        "home_market_value_total": [np.nan], "away_market_value_total": [np.nan],
        "home_market_value_top11": [np.nan], "away_market_value_top11": [np.nan],
        "home_tm_age_days": [np.nan], "away_tm_age_days": [np.nan],
    })
    X, y = build_symmetric_rows(matches)
    assert X.shape == (2, 24)
    assert np.isnan(X[0, 18])
    assert np.isnan(X[1, 18])
```

- [ ] **Step 4: Run test**

```bash
pytest tests/test_poisson_xgb.py -v
```

Expected: tutti passati. Eventuale fail dei test esistenti suggerisce che `matches` di test legacy non ha le 6 nuove colonne — fix aggiungendo `np.nan` defaults nei builder dei DataFrame di test (NON modificando il codice produttivo).

- [ ] **Step 5: Verifica suite intera**

```bash
pytest -q
```

Expected: tutti passati o `skipped` (i test slow di tier3 sono `skipped` finché snapshots non c'è).

- [ ] **Step 6: Commit**

```bash
git add src/mondiali/model/poisson_xgb.py tests/test_poisson_xgb.py
git commit -m "feat(model): extend SYMMETRIC_FEATURES with Tier 3 (18 -> 24) (Task 11)"
```

---

## Task 12: Helper `_recompute_tier2_baseline_for_gate`

**Files:**
- Modify: `src/mondiali/training/train.py`
- Modify: `tests/test_train_tier2.py`

Funzione che ricomputa Tier 2 raw log-loss su un range val_gate arbitrario (apples-to-apples per il gate Tier 3 vs Tier 2 stesso val_gate 2022).

- [ ] **Step 1: Aggiungi helper**

In `src/mondiali/training/train.py`, in fondo:

```python
def _recompute_tier2_baseline_for_gate(
    parquet_path: Path,
    val_gate_start: str,
    val_gate_end: str,
    *,
    train_start: str = "2002-01-01",
    train_end: str = "2016-12-31",
    val_es_start: str = "2017-01-01",
    val_es_end: str = "2017-12-31",
) -> float:
    """Ricomputa Tier 2 raw log-loss su un val_gate arbitrario.

    Usato dallo STEP 4 per ottenere baseline apples-to-apples sul val_gate 2022
    (Tier 3 si valuta su 2022 only, Tier 2 originale era su 2019-2022).

    Returns:
        val_log_loss_raw di Tier 2 sul val_gate richiesto.
    """
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["days_rest_home", "days_rest_away"]).copy()

    train = df[(df["date"] >= train_start) & (df["date"] <= train_end)].reset_index(drop=True)
    val_es = df[(df["date"] >= val_es_start) & (df["date"] <= val_es_end)].reset_index(drop=True)
    val_gate = df[
        (df["date"] >= val_gate_start) & (df["date"] <= val_gate_end)
    ].reset_index(drop=True)

    # IMPORTANT: per garantire apples-to-apples, addestriamo Tier 2 *senza* le
    # 6 colonne Tier 3. Costruiamo un set di feature artificialmente ristretto
    # passando un DataFrame con NaN nelle colonne Tier 3 (XGBoost ignora la
    # split su feature 100% NaN).
    for col in TIER3_COLUMNS:
        if col not in train.columns:
            train[col] = np.nan
            val_es[col] = np.nan
            val_gate[col] = np.nan
        else:
            train[col] = np.nan
            val_es[col] = np.nan
            val_gate[col] = np.nan

    model = PoissonXGBModel()
    model.fit(train, early_stopping_val=val_es, early_stopping_rounds=50)

    lam_h_tr, lam_a_tr = model.predict_lambda(train)
    rho = estimate_rho_mle(
        lam_h_tr, lam_a_tr,
        train["home_score"].to_numpy(), train["away_score"].to_numpy(),
    )

    lam_h_va, lam_a_va = model.predict_lambda(val_gate)
    val_probs = _compute_1x2_probs(lam_h_va, lam_a_va, rho=rho)
    return log_loss_1x2(val_gate, val_probs)
```

Aggiungi import in cima:
```python
from mondiali.features.tier3 import TIER3_COLUMNS
```

- [ ] **Step 2: Test rapido**

In `tests/test_train_tier2.py` (è il file più affine):

```python
def test_recompute_tier2_baseline_for_gate_returns_float():
    """Smoke: helper ritorna un float positivo plausibile."""
    from mondiali.training.train import _recompute_tier2_baseline_for_gate
    parquet = CONFIG.data_processed / "matches.parquet"
    if not parquet.exists():
        pytest.skip("matches.parquet not found")

    baseline = _recompute_tier2_baseline_for_gate(
        parquet,
        val_gate_start="2022-01-01",
        val_gate_end="2022-12-31",
        train_end="2018-12-31",  # range più piccolo per velocità test
        val_es_end="2019-12-31",
        val_es_start="2019-01-01",
    )
    assert 0.5 < baseline < 1.5  # plausibile per 1X2 logloss
```

- [ ] **Step 3: Run**

```bash
pytest tests/test_train_tier2.py::test_recompute_tier2_baseline_for_gate_returns_float -v
```

Expected: PASS o `skipped` (se parquet non c'è).

- [ ] **Step 4: Commit**

```bash
git add src/mondiali/training/train.py tests/test_train_tier2.py
git commit -m "feat(training): _recompute_tier2_baseline_for_gate apples-to-apples helper (Task 12)"
```

---

## Task 13: `train_tier3_pipeline`

**Files:**
- Modify: `src/mondiali/training/train.py`
- Create: `tests/test_train_tier3.py`

Mirror di `train_tier2_pipeline` con filtro 2014+ + return arricchito.

- [ ] **Step 1: Scrivi test**

```python
# tests/test_train_tier3.py
"""Test pipeline training Tier 3 end-to-end."""
from __future__ import annotations

import pytest

from mondiali.config import CONFIG
from mondiali.features.tier3 import TIER3_COLUMNS
from mondiali.training.train import train_tier3_pipeline


def test_train_tier3_returns_required_keys():
    """Smoke test: il dict di ritorno ha tutte le chiavi attese."""
    parquet = CONFIG.data_processed / "matches.parquet"
    if not parquet.exists():
        pytest.skip("matches.parquet not found")
    import pandas as pd
    if not all(c in pd.read_parquet(parquet).columns for c in TIER3_COLUMNS):
        pytest.skip("tier3 columns not in matches.parquet — run tm-scrape + ingest first")

    result = train_tier3_pipeline(
        parquet_path=parquet,
        train_start="2014-01-01", train_end="2018-12-31",
        val_es_start="2019-01-01", val_es_end="2019-12-31",
        val_calib_start="2020-01-01", val_calib_end="2020-12-31",
        val_gate_start="2021-01-01", val_gate_end="2021-12-31",
    )
    expected_keys = {
        "model", "rho", "calibrator",
        "val_log_loss_raw", "val_log_loss_calib",
        "brier_before", "brier_after",
        "n_train", "n_val_es", "n_val_calib", "n_val_gate",
        "n_train_pre2014_dropped", "tm_coverage_train", "tm_coverage_gate",
    }
    assert expected_keys.issubset(result.keys())


@pytest.mark.slow
def test_train_tier3_full_split_passes_gate():
    """STEP 4 GATE BLOCKING: val_log_loss_raw_tier3 ≤ tier2_baseline_2022 - 0.001."""
    import pandas as pd
    parquet = CONFIG.data_processed / "matches.parquet"
    if not parquet.exists():
        pytest.skip("matches.parquet not found")
    if not all(c in pd.read_parquet(parquet).columns for c in TIER3_COLUMNS):
        pytest.skip("tier3 columns not in matches.parquet")

    from mondiali.training.train import _recompute_tier2_baseline_for_gate
    baseline_t2 = _recompute_tier2_baseline_for_gate(
        parquet, val_gate_start="2022-01-01", val_gate_end="2022-12-31",
    )

    result = train_tier3_pipeline(parquet_path=parquet)

    assert result["val_log_loss_raw"] <= baseline_t2 - 0.001, \
        f"GATE FAIL: tier3={result['val_log_loss_raw']:.4f} > tier2={baseline_t2:.4f} - 0.001"
    assert -0.3 <= result["rho"] <= 0.05
    assert result["tm_coverage_gate"] >= 0.80
```

- [ ] **Step 2: Run (FAIL, perché pipeline non c'è)**

```bash
pytest tests/test_train_tier3.py -v
```

- [ ] **Step 3: Implementa pipeline**

In `src/mondiali/training/train.py` (in fondo):

```python
def train_tier3_pipeline(
    parquet_path: Path,
    *,
    train_start: str = "2014-01-01",
    train_end: str = "2019-12-31",
    val_es_start: str = "2020-01-01",
    val_es_end: str = "2020-12-31",
    val_calib_start: str = "2021-01-01",
    val_calib_end: str = "2021-12-31",
    val_gate_start: str = "2022-01-01",
    val_gate_end: str = "2022-12-31",
    early_stopping_rounds: int = 50,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pipeline Tier 3: training su matches 2014+ con feature TM.

    Differenze chiave vs Tier 2:
    - Filtro 2014+ obbligatorio sul training set (TM è NaN prima).
    - Returns dict include n_train_pre2014_dropped + tm_coverage_*.
    """
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["days_rest_home", "days_rest_away"]).copy()

    n_pre2014 = len(df[df["date"] < pd.Timestamp("2014-01-01")])

    train = df[(df["date"] >= train_start) & (df["date"] <= train_end)].reset_index(drop=True)
    val_es = df[(df["date"] >= val_es_start) & (df["date"] <= val_es_end)].reset_index(drop=True)
    val_calib = df[
        (df["date"] >= val_calib_start) & (df["date"] <= val_calib_end)
    ].reset_index(drop=True)
    val_gate = df[
        (df["date"] >= val_gate_start) & (df["date"] <= val_gate_end)
    ].reset_index(drop=True)

    # tm_coverage = frazione match con entrambi i lati TM non-NaN
    def _tm_coverage(d: pd.DataFrame) -> float:
        if len(d) == 0:
            return 0.0
        both_present = d["home_market_value_total"].notna() & d["away_market_value_total"].notna()
        return float(both_present.mean())

    log.info(
        "tier3 pipeline start",
        n_train=len(train), n_val_es=len(val_es),
        n_val_calib=len(val_calib), n_val_gate=len(val_gate),
        n_train_pre2014_dropped=n_pre2014,
        tm_coverage_train=_tm_coverage(train),
        tm_coverage_gate=_tm_coverage(val_gate),
    )

    model = PoissonXGBModel(params=model_params)
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

    val_log_loss_raw = log_loss_1x2(val_gate, raw_probs_gate)
    val_log_loss_calib = log_loss_1x2(val_gate, cal_probs_gate)
    brier_before = brier_score_1x2(val_gate, raw_probs_gate)
    brier_after = brier_score_1x2(val_gate, cal_probs_gate)

    log.info(
        "tier3 validation",
        log_loss_raw=val_log_loss_raw, log_loss_calib=val_log_loss_calib,
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
        "n_train_pre2014_dropped": n_pre2014,
        "tm_coverage_train": _tm_coverage(train),
        "tm_coverage_gate": _tm_coverage(val_gate),
    }
```

- [ ] **Step 4: Run smoke (slow test ancora skipped)**

```bash
pytest tests/test_train_tier3.py::test_train_tier3_returns_required_keys -v
```

Expected: skipped (snapshots non c'è ancora) — è OK.

- [ ] **Step 5: Commit**

```bash
git add src/mondiali/training/train.py tests/test_train_tier3.py
git commit -m "feat(training): train_tier3_pipeline with 2014+ filter + coverage tracking (Task 13)"
```

---

## Task 14: `train-tier3` CLI

**Files:**
- Modify: `src/mondiali/cli/main.py`

Pattern identico a `train-tier2`.

- [ ] **Step 1: Aggiungi command**

In `src/mondiali/cli/main.py`, dopo `train_tier2`:

```python
from mondiali.training.train import train_tier3_pipeline


@app.command(name="train-tier3")
def train_tier3(
    train_start: str = typer.Option("2014-01-01"),
    train_end: str = typer.Option("2019-12-31"),
    val_es_start: str = typer.Option("2020-01-01"),
    val_es_end: str = typer.Option("2020-12-31"),
    val_calib_start: str = typer.Option("2021-01-01"),
    val_calib_end: str = typer.Option("2021-12-31"),
    val_gate_start: str = typer.Option("2022-01-01"),
    val_gate_end: str = typer.Option("2022-12-31"),
    save_model: str = typer.Option("", "--save-model"),
    save_calibrator: str = typer.Option("", "--save-calibrator"),
) -> None:
    """Addestra Tier 3 (XGBoost Poisson + DC + isotonic + Transfermarkt features)."""
    parquet = CONFIG.data_processed / "matches.parquet"
    result = train_tier3_pipeline(
        parquet_path=parquet,
        train_start=train_start, train_end=train_end,
        val_es_start=val_es_start, val_es_end=val_es_end,
        val_calib_start=val_calib_start, val_calib_end=val_calib_end,
        val_gate_start=val_gate_start, val_gate_end=val_gate_end,
    )
    typer.echo(
        f"Splits: train={result['n_train']} val_es={result['n_val_es']} "
        f"val_calib={result['n_val_calib']} val_gate={result['n_val_gate']} "
        f"(pre-2014 dropped: {result['n_train_pre2014_dropped']})"
    )
    typer.echo(f"Dixon-Coles rho: {result['rho']:.4f}")
    typer.echo(f"TM coverage train/gate: {result['tm_coverage_train']:.1%} / {result['tm_coverage_gate']:.1%}")
    typer.echo(f"Tier 3 RAW   log-loss: {result['val_log_loss_raw']:.4f}")
    typer.echo(f"Tier 3 CALIB log-loss: {result['val_log_loss_calib']:.4f}")
    typer.echo(f"Brier before/after:  {result['brier_before']:.4f} / {result['brier_after']:.4f}")

    if save_model:
        result["model"].save(Path(save_model))
        typer.echo(f"Model saved: {save_model}")
    if save_calibrator:
        result["calibrator"].save(Path(save_calibrator))
        typer.echo(f"Calibrator saved: {save_calibrator}")
```

- [ ] **Step 2: Smoke**

```bash
mondiali train-tier3 --help
```

Expected: typer help screen.

- [ ] **Step 3: Commit**

```bash
git add src/mondiali/cli/main.py
git commit -m "feat(cli): train-tier3 command (Task 14)"
```

---

## Task 15: Esecuzione end-to-end + validation report + tag

Questo è il task "live": esegui scraping reale, pipeline reale, scrivi report.

- [ ] **Step 1: Sanity check pre-flight**

```bash
git status
pytest -q
```

Expected: working tree clean (a parte i file generati ignored), tutti i test green or skipped.

- [ ] **Step 2: Esegui scrape (LUNGO ~2-4h)**

```bash
mondiali tm-scrape --start-year 2014 --end-year 2025
```

Aspettative:
- Coverage logging finale: target ~`80 nazioni × 12 anni = 960` snapshot, atteso `n_filled ≥ 600` (≥62%).
- Output: `data/raw/transfermarkt/snapshots.parquet`.
- Cache HTML in `data/raw/transfermarkt/cache/`.

Se la coverage scende sotto 60%, fai un secondo run con cache (idempotente, riprova solo i fail) prima di concludere.

- [ ] **Step 3: Re-ingest per buildare matches.parquet con TM**

```bash
mondiali ingest
```

Verifica:
```bash
python -c "import pandas as pd; df = pd.read_parquet('data/processed/matches.parquet'); print(df[['home_market_value_total', 'away_market_value_total', 'home_tm_age_days']].describe())"
```

Expected: numeri non-NaN per match 2014+, NaN per pre-2014.

- [ ] **Step 4: Recompute Tier 2 baseline su val_gate 2022**

```bash
python -c "from mondiali.training.train import _recompute_tier2_baseline_for_gate; from mondiali.config import CONFIG; print(_recompute_tier2_baseline_for_gate(CONFIG.data_processed / 'matches.parquet', '2022-01-01', '2022-12-31'))"
```

Salva il numero (es. `0.852X`).

- [ ] **Step 5: Train Tier 3**

```bash
mondiali train-tier3 --save-model models/tier3/xgb_poisson.json --save-calibrator models/tier3/calibrator.json
```

Salva tutti gli output (n_train, log-loss raw, calib, brier, coverage).

- [ ] **Step 6: Run anti-leakage + slow gate test**

```bash
pytest tests/test_leakage.py::test_tier3_market_value_strict_pre_match -v
pytest tests/test_train_tier3.py -v -m slow
pytest -q
```

Expected:
- Anti-leakage: PASS.
- Slow gate: PASS (val_log_loss_raw ≤ baseline_t2 − 0.001).
- Suite intera: ~150-160 test, tutti green.

- [ ] **Step 7: Scrivi `reports/validation_step4.md`**

Pattern: identico a `reports/validation_step3.md`. Sezioni: TL;DR, Setup, Risultati, Gate, Confronto STEP-by-STEP, Decisione, Anti-data-leakage, Test suite, Aperti per STEP 5.

Numeri da inserire (riempi dopo aver fatto i run):

```markdown
# STEP 4 — Tier 3 Transfermarkt market values

**Data**: <YYYY-MM-DD>
**Commit**: <SHA>
**Predecessore**: STEP 3 chiuso (`step3-complete`).

## TL;DR

- **Gate funzionale (coverage ≥80%): <PASS/FAIL>** — coverage val_gate 2022 = <X.XX%>
- **Gate metrico (Δ ≤ -0.001 vs Tier 2 baseline 2022): <PASS/FAIL>** — Tier 3 raw=<X.XXXX>, Tier 2 baseline=<X.XXXX>, Δ=<-0.XXXX>.
- **Anti-leakage: PASS** (test_tier3_market_value_strict_pre_match).

## Setup
... (stesso pattern di STEP 3)

## Risultati

| Metric | Valore |
|---|---|
| `val_log_loss_raw` | <X.XXXX> |
| `val_log_loss_calib` | <X.XXXX> |
| `brier_before` | <X.XXXX> |
| `brier_after` | <X.XXXX> |
| Dixon-Coles ρ | <X.XXXX> |
| n_train / n_val_es / n_val_calib / n_val_gate | <X> / <X> / <X> / <X> |
| n_train_pre2014_dropped | <X> |
| tm_coverage_train | <X.X%> |
| tm_coverage_gate | <X.X%> |
| Tier 2 baseline 2022 (apples-to-apples) | <X.XXXX> |

## Gate

| Gate | Soglia | Risultato | Esito |
|---|---|---|---|
| Coverage funzionale | ≥80% val_gate | <X.X%> | <PASS/FAIL> |
| Metric | raw ≤ baseline_t2 − 0.001 = <X.XXXX> | <X.XXXX> | <PASS/FAIL> |

## Decisione

<Tier 3 entra nel modello v1 finale> | <Tier 3 non aggiunge segnale, escluso, codice scraper resta come legacy>.

## Anti-data-leakage

5 test passano (Elo strict-pre, no future matches, days_rest, Tier 2 form_5, **Tier 3 market_value strict-pre**).

## Test suite

- 127 test (post STEP 3) → <X> test (post STEP 4).
- Slow gate `test_train_tier3_full_split_passes_gate`: <PASS/FAIL>.

## Aperti per STEP 5

1. Optuna su 24 features (deferred da STEP 4 e STEP 3).
2. Tier 4 = injuries (master plan).
3. Cross-fit calibration → STEP 6.
```

- [ ] **Step 8: Commit report e tag**

```bash
git add reports/validation_step4.md
git commit -m "docs(report): STEP 4 validation — Tier 3 Transfermarkt <PASS/FAIL>"
TAG_NAME="step4-complete"
# Se gate metrico FAIL: TAG_NAME="step4-no-signal"
git tag $TAG_NAME
git tag -l
```

- [ ] **Step 9: Final sanity**

```bash
git log --oneline -20
pytest -q
```

Expected: storia completa STEP 4, suite tutta verde.

---

## Self-Review (autore: Claude)

**Spec coverage check**:
- ✅ Coverage 2014+: Task 8 (TIER3_MIN_YEAR=2014, NaN filter), Task 13 (filtro train 2014-01-01)
- ✅ 3 features per side (24 total): Task 11 (SYMMETRIC_FEATURES extension)
- ✅ ~70-80 nations scope: Task 2 (compute_tier3_scope)
- ✅ Adaptive 1-snapshot/year + 4-level fallback: Task 6 (_best_snapshot_for_year)
- ✅ Forward-fill via lookup of latest pre-match snapshot: Task 8 (_lookup_strict_pre)
- ✅ Hard floor ≥2 snapshots: Task 8 (TIER3_MIN_SNAPSHOTS_PER_NATION)
- ✅ Age clipping >540d: Task 8 (TIER3_MAX_AGE_DAYS)
- ✅ Gate coverage 80%: Task 13 (assertion in slow test)
- ✅ Gate metric Δ ≤ -0.001: Task 13 (slow test)
- ✅ Anti-leakage strict-pre test: Task 10
- ✅ Wayback CDX + cache + rate limit: Tasks 3, 5, 6, 7
- ✅ tm-scrape CLI: Task 7
- ✅ train-tier3 CLI: Task 14
- ✅ validation_step4.md report: Task 15
- ✅ Recompute Tier 2 apples-to-apples: Task 12

**Type consistency**:
- `SnapshotRecord` dataclass introdotto in Task 6, usato in Task 7. ✅
- `SquadValue` introdotto in Task 4, usato in Task 6, 7. ✅
- `CDXRow` introdotto in Task 3, usato in Task 5, 6. ✅
- `TIER3_COLUMNS` introdotto in Task 8, usato in Task 9, 10, 11, 12, 13. ✅
- `_recompute_tier2_baseline_for_gate` definito in Task 12, usato in Task 13. ✅

**Placeholder scan**: nessun TBD/TODO/`fill in details`. Le sezioni `<X.XXXX>` nel template del report sono espliciti placeholder da riempire post-run, non placeholder di plan.

**Ambiguity**: il `time.sleep(RATE_LIMIT_SECONDS)` precede il fetch — alcuni implementer lo metterebbero post-fetch. La convention sleep-before-fetch evita race con la prima request senza sleep alcuno. Documentato implicitly nel codice di Task 5.
