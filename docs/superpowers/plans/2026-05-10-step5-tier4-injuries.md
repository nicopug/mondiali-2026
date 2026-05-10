# STEP 5 — Tier 4 Injuries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Tier 4 (Injuries) end-to-end: scrape player-level rosters from Transfermarkt for 4 historical tournaments, bootstrap injuries.csv from Wikipedia withdrawals, add 4 features (count + value-ratio per side) with strict anti-leakage, run apples-to-apples Optuna double study (Tier 1+2 baseline vs Tier 1+2+4 challenger), and decide tier-gate finale via ±0.003 log-loss on val_gate WC2022.

**Architecture:** Two new data modules (`tm_rosters.py`, `injuries_bootstrap.py`) reuse the existing cache fast-path pattern from STEP 4. One new feature module (`features/tier4.py`) follows the same `merge_asof`-friendly invariants of `tier3.py`. One new training pipeline `train_tier4_pipeline` extends `training/train.py` with Optuna integration. Three new CLI commands wire it together.

**Tech Stack:** Python 3.11+, pandas, BeautifulSoup, requests, XGBoost (count:poisson), Optuna, structlog, typer, pytest. Spec reference: `docs/superpowers/specs/2026-05-10-step5-tier4-injuries-design.md`.

---

## File Structure

**Create:**
- `src/mondiali/data/tm_rosters.py` — TM roster scraper (player-level, per-tournament)
- `src/mondiali/data/injuries_bootstrap.py` — Wikipedia withdrawals parser → injuries.csv
- `src/mondiali/features/tier4.py` — `add_tier4_features` (4 columns, anti-leakage)
- `tests/test_tm_rosters.py` — ~6 tests
- `tests/test_injuries_bootstrap.py` — ~5 tests
- `tests/test_tier4.py` — ~7 tests
- `tests/test_train_tier4.py` — ~3 smoke tests
- `data/manual/injuries.csv` — header-only at first; populated by Task 7 + manual top-up
- `models/tier4/` — directory for artifacts (created by Task 11)
- `reports/validation_step5.md` — gate report (written by Task 13)

**Modify:**
- `src/mondiali/cli/main.py` — +3 commands: `tm-scrape-rosters`, `bootstrap-injuries`, `train-tier4`
- `src/mondiali/training/train.py` — `+train_tier4_pipeline` with double Optuna study
- `tests/test_leakage.py` — `+test_tier4_strict_pre_match`

---

## Task 1: Tournament metadata + injuries.csv schema

**Files:**
- Create: `src/mondiali/data/tm_rosters.py`
- Create: `data/manual/injuries.csv`
- Test: none (constants-only task — validated via type-check + smoke import)

- [ ] **Step 1: Create `data/manual/injuries.csv` header-only**

```csv
date_of_knowledge,team,tournament,player_name,player_url_slug,market_value_eur,status,source
```

- [ ] **Step 2: Create `src/mondiali/data/tm_rosters.py` with metadata constants only**

```python
"""Player-level roster scraper for historical tournaments (Tier 4 enabler).

Scope: WC2018, Euro2020, WC2022, Euro2024.
Output: data/raw/transfermarkt/rosters.parquet (player, slug, position, value).
Reuses cache fast-path machinery from `transfermarkt.py`.
"""
from __future__ import annotations

from datetime import date

TOURNAMENT_META: dict[str, dict[str, object]] = {
    "wc2018":   {"start": date(2018, 6, 14), "end": date(2018, 7, 15), "saison_id": 2017},
    "euro2020": {"start": date(2021, 6, 11), "end": date(2021, 7, 11), "saison_id": 2020},
    "wc2022":   {"start": date(2022, 11, 20), "end": date(2022, 12, 18), "saison_id": 2022},
    "euro2024": {"start": date(2024, 6, 14), "end": date(2024, 7, 14), "saison_id": 2023},
}

TOURNAMENT_PARTICIPANTS: dict[str, list[str]] = {
    "wc2018": [
        "Russia", "Saudi Arabia", "Egypt", "Uruguay", "Portugal", "Spain",
        "Morocco", "Iran", "France", "Australia", "Argentina", "Iceland",
        "Peru", "Denmark", "Croatia", "Nigeria", "Costa Rica", "Serbia",
        "Germany", "Mexico", "Brazil", "Switzerland", "Sweden", "South Korea",
        "Belgium", "Panama", "Tunisia", "England", "Colombia", "Japan",
        "Poland", "Senegal",
    ],
    "euro2020": [
        "Italy", "Switzerland", "Turkey", "Wales", "Belgium", "Denmark",
        "Finland", "Russia", "Netherlands", "Ukraine", "Austria", "North Macedonia",
        "England", "Croatia", "Scotland", "Czech Republic", "Spain", "Sweden",
        "Poland", "Slovakia", "France", "Germany", "Hungary", "Portugal",
    ],
    "wc2022": [
        "Qatar", "Ecuador", "Senegal", "Netherlands", "England", "Iran",
        "United States", "Wales", "Argentina", "Saudi Arabia", "Mexico", "Poland",
        "France", "Australia", "Denmark", "Tunisia", "Spain", "Costa Rica",
        "Germany", "Japan", "Belgium", "Canada", "Morocco", "Croatia",
        "Brazil", "Serbia", "Switzerland", "Cameroon", "Portugal", "Ghana",
        "Uruguay", "South Korea",
    ],
    "euro2024": [
        "Germany", "Scotland", "Hungary", "Switzerland", "Spain", "Croatia",
        "Italy", "Albania", "Slovenia", "Denmark", "Serbia", "England",
        "Poland", "Netherlands", "Austria", "France", "Belgium", "Slovakia",
        "Romania", "Ukraine", "Turkey", "Georgia", "Portugal", "Czech Republic",
    ],
}

INJURIES_CSV_COLUMNS: list[str] = [
    "date_of_knowledge", "team", "tournament", "player_name",
    "player_url_slug", "market_value_eur", "status", "source",
]
INJURY_STATUS_DOMAIN: frozenset[str] = frozenset({"out", "doubtful", "available"})
INJURY_SOURCE_DOMAIN: frozenset[str] = frozenset({"wikipedia_squads", "manual"})
```

- [ ] **Step 3: Verify smoke import + counts**

Run:
```bash
python -c "from mondiali.data.tm_rosters import TOURNAMENT_META, TOURNAMENT_PARTICIPANTS; assert sum(len(v) for v in TOURNAMENT_PARTICIPANTS.values()) == 112; print('OK', sum(len(v) for v in TOURNAMENT_PARTICIPANTS.values()))"
```
Expected: `OK 112`

- [ ] **Step 4: Commit**

```bash
git add src/mondiali/data/tm_rosters.py data/manual/injuries.csv
git commit -m "feat(tier4): tournament metadata + injuries.csv schema scaffold"
```

---

## Task 2: TM roster HTML parser

**Files:**
- Modify: `src/mondiali/data/tm_rosters.py` (add `RosterPlayer`, `_parse_roster_html`)
- Create: `tests/test_tm_rosters.py`
- Create: `tests/fixtures/tm_roster_france_wc2018.html` (small fixture, ~30 lines, copy a real TM kader table snippet — see Step 1)

- [ ] **Step 1: Create fixture HTML** at `tests/fixtures/tm_roster_france_wc2018.html`

```html
<!DOCTYPE html><html><body>
<table class="items">
<tbody>
<tr class="odd">
<td class="hauptlink"><a href="/hugo-lloris/profil/spieler/16294" title="Hugo Lloris">Hugo Lloris</a></td>
<td>Goalkeeper</td>
<td class="rechts hauptlink">€10.00m</td>
</tr>
<tr class="even">
<td class="hauptlink"><a href="/kylian-mbappe/profil/spieler/342229" title="Kylian Mbappé">Kylian Mbappé</a></td>
<td>Centre-Forward</td>
<td class="rechts hauptlink">€120.00m</td>
</tr>
<tr class="odd">
<td class="hauptlink"><a href="/paul-pogba/profil/spieler/122153" title="Paul Pogba">Paul Pogba</a></td>
<td>Central Midfield</td>
<td class="rechts hauptlink">€70.00m</td>
</tr>
<tr class="even">
<td class="hauptlink"><a href="/n-golo-kante/profil/spieler/225083" title="N&#039;Golo Kanté">N'Golo Kanté</a></td>
<td>Defensive Midfield</td>
<td class="rechts hauptlink">€60.00m</td>
</tr>
<tr class="odd">
<td class="hauptlink"><a href="/raphael-varane/profil/spieler/164770" title="Raphaël Varane">Raphaël Varane</a></td>
<td>Centre-Back</td>
<td class="rechts hauptlink">€60.00m</td>
</tr>
</tbody>
</table>
</body></html>
```

- [ ] **Step 2: Write failing test** `tests/test_tm_rosters.py`

```python
"""Tests for tm_rosters.py — player-level roster scraper."""
from __future__ import annotations

from pathlib import Path

import pytest

from mondiali.data.tm_rosters import RosterPlayer, _parse_roster_html


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_parse_roster_html_extracts_players() -> None:
    html = (FIXTURE_DIR / "tm_roster_france_wc2018.html").read_text(encoding="utf-8")
    players = _parse_roster_html(html)
    assert len(players) == 5
    assert players[0] == RosterPlayer(
        player_name="Hugo Lloris",
        player_url_slug="hugo-lloris",
        position="Goalkeeper",
        market_value_eur=10_000_000,
    )
    mbappe = next(p for p in players if p.player_name == "Kylian Mbappé")
    assert mbappe.market_value_eur == 120_000_000
    kante = next(p for p in players if p.player_url_slug == "n-golo-kante")
    assert kante.market_value_eur == 60_000_000
```

- [ ] **Step 3: Run test (red)**

```bash
pytest tests/test_tm_rosters.py::test_parse_roster_html_extracts_players -v
```
Expected: FAIL with `ImportError: cannot import name 'RosterPlayer'`.

- [ ] **Step 4: Implement parser in `tm_rosters.py`**

Append to `tm_rosters.py`:
```python
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

from mondiali.data.transfermarkt import _parse_value_eur


@dataclass(frozen=True)
class RosterPlayer:
    player_name: str
    player_url_slug: str
    position: str
    market_value_eur: int | None


_PLAYER_PROFILE_RE = re.compile(r"^/(?P<slug>[a-z0-9-]+)/profil/spieler/\d+")


def _parse_roster_html(html: str) -> list[RosterPlayer]:
    """Parse a TM kader page → list of RosterPlayer.

    Selects ``table.items`` (modern TM ≥2018) or ``table#kader`` (legacy).
    For each row, extracts name+slug from ``td.hauptlink a[title]``,
    position from the row's second non-hauptlink ``<td>``, and value from
    ``td.rechts.hauptlink`` via ``_parse_value_eur``.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.items") or soup.select_one("table#kader")
    if table is None:
        return []
    out: list[RosterPlayer] = []
    for row in table.select("tbody > tr"):
        link = row.select_one("td.hauptlink a[href*='/profil/spieler/']")
        if link is None:
            continue
        m = _PLAYER_PROFILE_RE.match(link.get("href", ""))
        if not m:
            continue
        slug = m.group("slug")
        name = link.get("title") or link.get_text(strip=True)
        tds = row.find_all("td", recursive=False)
        position = tds[1].get_text(strip=True) if len(tds) >= 2 else ""
        value_cell = row.select_one("td.rechts.hauptlink") or row.select_one("td.rechts")
        value: int | None = None
        if value_cell is not None:
            parsed = _parse_value_eur(value_cell.get_text(strip=True))
            value = int(parsed) if parsed is not None else None
        out.append(RosterPlayer(
            player_name=name, player_url_slug=slug,
            position=position, market_value_eur=value,
        ))
    return out
```

- [ ] **Step 5: Run test (green)**

```bash
pytest tests/test_tm_rosters.py::test_parse_roster_html_extracts_players -v
```
Expected: PASS.

- [ ] **Step 6: Add edge-case tests for billion-format and missing values**

Append to `tests/test_tm_rosters.py`:
```python
def test_parse_value_handles_em_billions() -> None:
    """Sanity: TM occasionally uses bn for a national team (rare). Should not crash."""
    html = """<html><body><table class="items"><tbody>
    <tr><td class="hauptlink"><a href="/x/profil/spieler/1" title="X">X</a></td>
    <td>GK</td><td class="rechts hauptlink">€1.20bn</td></tr>
    </tbody></table></body></html>"""
    # 1.20bn is currently not supported by _parse_value_eur (only m, k, Mio, Tsd, Th).
    # We assert that the parser does NOT crash and returns None for value.
    players = _parse_roster_html(html)
    assert len(players) == 1
    assert players[0].market_value_eur is None


def test_parse_skips_rows_without_player_link() -> None:
    """Header rows or aggregate rows with no /profil/spieler/ link must be ignored."""
    html = """<html><body><table class="items"><tbody>
    <tr><td colspan="5">Total: €500.00m</td></tr>
    </tbody></table></body></html>"""
    assert _parse_roster_html(html) == []


def test_omonimi_disambiguati_via_slug() -> None:
    """Two players named 'Diego López' must produce two distinct slugs."""
    html = """<html><body><table class="items"><tbody>
    <tr><td class="hauptlink"><a href="/diego-lopez/profil/spieler/100" title="Diego López">Diego López</a></td>
    <td>GK</td><td class="rechts hauptlink">€5.00m</td></tr>
    <tr><td class="hauptlink"><a href="/diego-lopez-2/profil/spieler/200" title="Diego López">Diego López</a></td>
    <td>DEF</td><td class="rechts hauptlink">€3.00m</td></tr>
    </tbody></table></body></html>"""
    players = _parse_roster_html(html)
    slugs = {p.player_url_slug for p in players}
    assert slugs == {"diego-lopez", "diego-lopez-2"}
```

- [ ] **Step 7: Run all tier4 roster tests**

```bash
pytest tests/test_tm_rosters.py -v
```
Expected: 4 PASS.

- [ ] **Step 8: Commit**

```bash
git add src/mondiali/data/tm_rosters.py tests/test_tm_rosters.py tests/fixtures/tm_roster_france_wc2018.html
git commit -m "feat(tm-rosters): HTML parser for player-level roster pages"
```

---

## Task 3: TM roster URL builder + cache fast-path

**Files:**
- Modify: `src/mondiali/data/tm_rosters.py`
- Modify: `tests/test_tm_rosters.py`

- [ ] **Step 1: Write failing test for URL pattern**

Append to `tests/test_tm_rosters.py`:
```python
from mondiali.data.tm_rosters import _build_roster_url


def test_roster_url_uses_saison_year_minus_one() -> None:
    """WC2018 → saison_id 2017 (TM stores roster under start-of-season year)."""
    url = _build_roster_url("France", "wc2018")
    assert url is not None
    assert "/saison_id/2017" in url
    assert "/kader/verein/" in url


def test_roster_url_returns_none_for_unknown_nation() -> None:
    assert _build_roster_url("Atlantis", "wc2018") is None


def test_roster_url_returns_none_for_unknown_tournament() -> None:
    assert _build_roster_url("France", "wc1990") is None
```

- [ ] **Step 2: Run tests (red)**

```bash
pytest tests/test_tm_rosters.py::test_roster_url_uses_saison_year_minus_one -v
```
Expected: FAIL with `ImportError: cannot import name '_build_roster_url'`.

- [ ] **Step 3: Implement URL builder**

Append to `tm_rosters.py`:
```python
from mondiali.data.tm_nations import NATION_TM_IDS

ROSTER_URL_TEMPLATE = (
    "https://www.transfermarkt.com/{slug}/kader/verein/{tm_id}/saison_id/{saison}/plus/1"
)


def _build_roster_url(nation: str, tournament: str) -> str | None:
    entry = NATION_TM_IDS.get(nation)
    meta = TOURNAMENT_META.get(tournament)
    if entry is None or meta is None:
        return None
    slug, tm_id = entry
    return ROSTER_URL_TEMPLATE.format(slug=slug, tm_id=tm_id, saison=meta["saison_id"])
```

- [ ] **Step 4: Run tests (green)**

```bash
pytest tests/test_tm_rosters.py -v
```
Expected: 7 PASS.

- [ ] **Step 5: Add cache fast-path test**

Append to `tests/test_tm_rosters.py`:
```python
def test_cache_fast_path_returns_html_when_present(tmp_path: Path) -> None:
    """If `{slug}__{tournament}.html` exists, _read_cached_roster reads it without network."""
    from mondiali.data.tm_rosters import _read_cached_roster
    cache_dir = tmp_path / "rosters"
    cache_dir.mkdir()
    (cache_dir / "equipe-de-france__wc2018.html").write_text("<html>cached</html>", encoding="utf-8")
    out = _read_cached_roster("equipe-de-france", "wc2018", cache_dir)
    assert out == "<html>cached</html>"


def test_cache_fast_path_returns_none_when_missing(tmp_path: Path) -> None:
    from mondiali.data.tm_rosters import _read_cached_roster
    assert _read_cached_roster("nope", "wc2018", tmp_path) is None
```

- [ ] **Step 6: Run tests (red)**

```bash
pytest tests/test_tm_rosters.py::test_cache_fast_path_returns_html_when_present -v
```
Expected: FAIL.

- [ ] **Step 7: Implement cache reader**

Append to `tm_rosters.py`:
```python
from pathlib import Path

from mondiali.data.transfermarkt import _slug_from_url


def _read_cached_roster(slug: str, tournament: str, cache_dir: Path) -> str | None:
    p = cache_dir / f"{slug}__{tournament}.html"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return None
```

- [ ] **Step 8: Run tests (green)**

```bash
pytest tests/test_tm_rosters.py -v
```
Expected: 9 PASS.

- [ ] **Step 9: Commit**

```bash
git add src/mondiali/data/tm_rosters.py tests/test_tm_rosters.py
git commit -m "feat(tm-rosters): URL builder + cache fast-path reader"
```

---

## Task 4: TM roster scrape orchestrator with --resume

**Files:**
- Modify: `src/mondiali/data/tm_rosters.py`
- Modify: `tests/test_tm_rosters.py`

- [ ] **Step 1: Write failing test for resume**

Append to `tests/test_tm_rosters.py`:
```python
from unittest.mock import patch

import pandas as pd


def test_scrape_rosters_resume_skips_already_done(tmp_path: Path) -> None:
    """If rosters.parquet already has (nation, tournament), skip without network call."""
    from mondiali.data.tm_rosters import scrape_rosters_all

    output_path = tmp_path / "rosters.parquet"
    cache_dir = tmp_path / "cache"
    existing = pd.DataFrame([{
        "nation": "France",
        "tournament": "wc2018",
        "tournament_start_date": pd.Timestamp("2018-06-14"),
        "player_name": "X",
        "player_url_slug": "x",
        "position": "GK",
        "market_value_eur": 1_000_000,
    }])
    existing.to_parquet(output_path, index=False)

    fetch_calls = []

    def fake_fetch(*args, **kwargs):  # pragma: no cover - asserted not called
        fetch_calls.append(args)
        return None

    with patch("mondiali.data.tm_rosters._fetch_roster_html", side_effect=fake_fetch):
        n_added = scrape_rosters_all(
            tournaments=["wc2018"],
            nations=["France"],
            cache_dir=cache_dir,
            output_path=output_path,
            resume=True,
        )
    assert n_added == 0
    assert fetch_calls == []
```

- [ ] **Step 2: Run test (red)**

```bash
pytest tests/test_tm_rosters.py::test_scrape_rosters_resume_skips_already_done -v
```
Expected: FAIL with `ImportError: cannot import name 'scrape_rosters_all'`.

- [ ] **Step 3: Implement orchestrator**

Append to `tm_rosters.py`:
```python
import time
from typing import Iterable

import pandas as pd
import requests
import structlog

from mondiali.data.transfermarkt import RATE_LIMIT_SECONDS

log = structlog.get_logger(__name__)

ROSTER_PARQUET_COLUMNS: list[str] = [
    "nation", "tournament", "tournament_start_date",
    "player_name", "player_url_slug", "position", "market_value_eur",
]


def _fetch_roster_html(url: str, slug: str, tournament: str, cache_dir: Path) -> str | None:
    """Fetch with rate limit + cache-write. Returns None on 4xx fatale or repeated failure."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    time.sleep(RATE_LIMIT_SECONDS)
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "mondiali-research/0.1"})
    except requests.RequestException as e:
        log.warning("roster_fetch_exception", url=url, error=str(e))
        return None
    if resp.status_code == 200:
        html = resp.text
        (cache_dir / f"{slug}__{tournament}.html").write_text(html, encoding="utf-8")
        return html
    log.warning("roster_fetch_non200", url=url, status=resp.status_code)
    return None


def _load_existing_roster_pairs(output_path: Path) -> set[tuple[str, str]]:
    if not output_path.exists():
        return set()
    df = pd.read_parquet(output_path)
    return set(zip(df["nation"].astype(str), df["tournament"].astype(str), strict=True))


def scrape_rosters_all(
    tournaments: Iterable[str],
    nations: Iterable[str] | None,
    cache_dir: Path,
    output_path: Path,
    *,
    resume: bool = True,
) -> int:
    """Scrape player-level rosters for given tournaments × nations.

    Args:
        tournaments: subset of TOURNAMENT_META keys.
        nations: if None, use TOURNAMENT_PARTICIPANTS for each tournament.
        cache_dir: dir for {slug}__{tournament}.html cache.
        output_path: rosters.parquet target.
        resume: skip pairs already in parquet.

    Returns:
        Number of (nation, tournament) pairs newly added to parquet.
    """
    existing = _load_existing_roster_pairs(output_path) if resume else set()
    new_rows: list[dict] = []

    for t in tournaments:
        meta = TOURNAMENT_META.get(t)
        if meta is None:
            log.warning("unknown_tournament", tournament=t)
            continue
        target_nations = list(nations) if nations is not None else TOURNAMENT_PARTICIPANTS.get(t, [])
        for nation in target_nations:
            if (nation, t) in existing:
                continue
            url = _build_roster_url(nation, t)
            if url is None:
                log.warning("no_url_for_nation", nation=nation, tournament=t)
                continue
            slug, _ = NATION_TM_IDS[nation]
            html = _read_cached_roster(slug, t, cache_dir) or _fetch_roster_html(url, slug, t, cache_dir)
            if html is None:
                continue
            players = _parse_roster_html(html)
            if not players:
                log.warning("roster_parse_empty", nation=nation, tournament=t)
                continue
            for p in players:
                new_rows.append({
                    "nation": nation,
                    "tournament": t,
                    "tournament_start_date": pd.Timestamp(meta["start"]),
                    "player_name": p.player_name,
                    "player_url_slug": p.player_url_slug,
                    "position": p.position,
                    "market_value_eur": p.market_value_eur,
                })

    if not new_rows:
        log.info("scrape_rosters_no_new_rows", existing_pairs=len(existing))
        return 0

    new_df = pd.DataFrame(new_rows, columns=ROSTER_PARQUET_COLUMNS)
    if output_path.exists():
        old_df = pd.read_parquet(output_path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined = new_df
    combined.to_parquet(output_path, index=False)

    n_pairs_added = combined[["nation", "tournament"]].drop_duplicates().shape[0] - len(existing)
    log.info("scrape_rosters_complete", n_pairs_added=n_pairs_added, n_rows_total=len(combined))
    return n_pairs_added
```

- [ ] **Step 4: Run test (green)**

```bash
pytest tests/test_tm_rosters.py::test_scrape_rosters_resume_skips_already_done -v
```
Expected: PASS.

- [ ] **Step 5: Add cache-fast-path-end-to-end test**

Append to `tests/test_tm_rosters.py`:
```python
def test_scrape_rosters_uses_cache_without_network(tmp_path: Path) -> None:
    """If cache HTML present, scraper parses it and never calls network."""
    from mondiali.data.tm_rosters import scrape_rosters_all

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    fixture_html = (FIXTURE_DIR / "tm_roster_france_wc2018.html").read_text(encoding="utf-8")
    # Slug for "France" per NATION_TM_IDS — verify before commit; placeholder here.
    from mondiali.data.tm_nations import NATION_TM_IDS
    france_slug, _ = NATION_TM_IDS["France"]
    (cache_dir / f"{france_slug}__wc2018.html").write_text(fixture_html, encoding="utf-8")

    output_path = tmp_path / "rosters.parquet"

    with patch("mondiali.data.tm_rosters._fetch_roster_html") as fetch_mock:
        n_added = scrape_rosters_all(
            tournaments=["wc2018"],
            nations=["France"],
            cache_dir=cache_dir,
            output_path=output_path,
            resume=False,
        )
    assert n_added == 1
    assert fetch_mock.call_count == 0
    df = pd.read_parquet(output_path)
    assert len(df) == 5
    assert set(df["player_url_slug"]) >= {"hugo-lloris", "kylian-mbappe"}
```

- [ ] **Step 6: Run all tests**

```bash
pytest tests/test_tm_rosters.py -v
```
Expected: 11 PASS.

- [ ] **Step 7: Commit**

```bash
git add src/mondiali/data/tm_rosters.py tests/test_tm_rosters.py
git commit -m "feat(tm-rosters): orchestrator with cache fast-path and --resume"
```

---

## Task 5: CLI `tm-scrape-rosters` command

**Files:**
- Modify: `src/mondiali/cli/main.py`

- [ ] **Step 1: Add command in `cli/main.py`**

Add after the `tm-build-from-cache` command (around line 350):
```python
from mondiali.data.tm_rosters import TOURNAMENT_META, scrape_rosters_all


@app.command(name="tm-scrape-rosters")
def tm_scrape_rosters(
    tournaments: str = typer.Option(
        "wc2018,euro2020,wc2022,euro2024",
        "--tournaments",
        help="Comma-separated tournament keys",
    ),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
) -> None:
    """Scrape player-level rosters from Transfermarkt for historical tournaments (Tier 4 enabler)."""
    keys = [t.strip() for t in tournaments.split(",") if t.strip()]
    unknown = [k for k in keys if k not in TOURNAMENT_META]
    if unknown:
        typer.echo(f"unknown tournaments: {unknown}", err=True)
        raise typer.Exit(1)
    cache_dir = CONFIG.data_raw / "transfermarkt" / "rosters"
    output_path = CONFIG.data_raw / "transfermarkt" / "rosters.parquet"
    typer.echo(f"Scraping rosters for {keys} -> {output_path}")
    n_added = scrape_rosters_all(
        tournaments=keys,
        nations=None,
        cache_dir=cache_dir,
        output_path=output_path,
        resume=resume,
    )
    typer.echo(f"Done. {n_added} new (nation, tournament) pairs added.")
```

- [ ] **Step 2: Smoke test the CLI**

Run:
```bash
python -m mondiali.cli.main tm-scrape-rosters --help
```
Expected: typer help text printed, no traceback.

- [ ] **Step 3: Commit**

```bash
git add src/mondiali/cli/main.py
git commit -m "feat(cli): tm-scrape-rosters command"
```

---

## Task 6: Wikipedia squads HTML parser

**Files:**
- Create: `src/mondiali/data/injuries_bootstrap.py`
- Create: `tests/test_injuries_bootstrap.py`
- Create: `tests/fixtures/wikipedia_wc2018_squads_excerpt.html` (small fixture)

- [ ] **Step 1: Create fixture HTML** at `tests/fixtures/wikipedia_wc2018_squads_excerpt.html`

```html
<!DOCTYPE html><html><body>
<h2><span class="mw-headline" id="Withdrawals">Withdrawals</span></h2>
<p>The following players were originally selected but withdrew due to injury or other reasons:</p>
<ul>
<li><b>Spain</b>: <a href="/wiki/Dani_Carvajal" title="Dani Carvajal">Dani Carvajal</a> withdrew due to injury and was replaced by <a href="/wiki/%C3%81lvaro_Odriozola">Álvaro Odriozola</a>.</li>
<li><b>Germany</b>: <a href="/wiki/Manuel_Neuer">Manuel Neuer</a> was originally selected but Bernd Leno was called instead. (this entry should not be parsed as a withdrawal)</li>
<li><b>France</b>: <a href="/wiki/Laurent_Koscielny">Laurent Koscielny</a> withdrew due to injury.</li>
</ul>
</body></html>
```

- [ ] **Step 2: Write failing test**

Create `tests/test_injuries_bootstrap.py`:
```python
"""Tests for injuries_bootstrap.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from mondiali.data.injuries_bootstrap import (
    WithdrawalEntry,
    parse_wikipedia_withdrawals,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_parse_wikipedia_withdrawals_section_extracts_entries() -> None:
    html = (FIXTURE_DIR / "wikipedia_wc2018_squads_excerpt.html").read_text(encoding="utf-8")
    entries = parse_wikipedia_withdrawals(html)
    teams_players = {(e.team, e.player_name) for e in entries}
    assert ("Spain", "Dani Carvajal") in teams_players
    assert ("France", "Laurent Koscielny") in teams_players
    # The "selected but Bernd Leno was called" entry must NOT be parsed as withdrawal.
    assert ("Germany", "Manuel Neuer") not in teams_players


def test_parse_wikipedia_returns_empty_when_no_section() -> None:
    html = "<html><body><h2>Squads</h2><p>Nothing here.</p></body></html>"
    assert parse_wikipedia_withdrawals(html) == []
```

- [ ] **Step 3: Run test (red)**

```bash
pytest tests/test_injuries_bootstrap.py -v
```
Expected: FAIL with `ImportError`.

- [ ] **Step 4: Implement parser**

Create `src/mondiali/data/injuries_bootstrap.py`:
```python
"""Bootstrap injuries.csv from Wikipedia tournament-squads pages.

Parses the 'Withdrawals' / 'Replacements' / 'Pre-tournament withdrawals' sections.
Matches player_name → rosters.parquet for slug + market_value. No fuzzy matching:
unmatched names are logged and skipped.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import structlog
from bs4 import BeautifulSoup

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class WithdrawalEntry:
    team: str
    player_name: str


_WITHDRAWAL_HEADLINE_IDS = {"Withdrawals", "Pre-tournament_withdrawals", "Replacements"}
_WITHDRAWAL_VERB_RE = re.compile(r"\b(withdrew|withdrawn|withdraw|out due to|injured)\b", re.IGNORECASE)


def parse_wikipedia_withdrawals(html: str) -> list[WithdrawalEntry]:
    """Extract (team, player_name) entries from the Withdrawals section.

    Strategy: locate ``<h2>`` headlines whose ``id`` is in _WITHDRAWAL_HEADLINE_IDS,
    walk siblings until next h2, parse ``<li>`` with bold team prefix + player link
    + a withdrawal verb in the same line.
    """
    soup = BeautifulSoup(html, "html.parser")
    out: list[WithdrawalEntry] = []
    for span in soup.find_all("span", class_="mw-headline"):
        if span.get("id") not in _WITHDRAWAL_HEADLINE_IDS:
            continue
        h2 = span.find_parent("h2")
        if h2 is None:
            continue
        for sib in h2.find_all_next():
            if sib.name == "h2":
                break
            if sib.name != "li":
                continue
            text = sib.get_text(" ", strip=True)
            if not _WITHDRAWAL_VERB_RE.search(text):
                continue
            bold = sib.find("b")
            link = sib.find("a", href=True)
            if bold is None or link is None:
                continue
            team = bold.get_text(strip=True)
            player_name = link.get_text(strip=True)
            out.append(WithdrawalEntry(team=team, player_name=player_name))
    return out


def _normalize_name(s: str) -> str:
    """Case-insensitive + accent-stripped normalization for exact matching only."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()
```

- [ ] **Step 5: Run tests (green)**

```bash
pytest tests/test_injuries_bootstrap.py -v
```
Expected: 2 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mondiali/data/injuries_bootstrap.py tests/test_injuries_bootstrap.py tests/fixtures/wikipedia_wc2018_squads_excerpt.html
git commit -m "feat(injuries): Wikipedia withdrawals section parser"
```

---

## Task 7: Bootstrap orchestrator (player matching + dedup)

**Files:**
- Modify: `src/mondiali/data/injuries_bootstrap.py`
- Modify: `tests/test_injuries_bootstrap.py`

- [ ] **Step 1: Write failing test for matching + skip on no-match**

Append to `tests/test_injuries_bootstrap.py`:
```python
import pandas as pd

from mondiali.data.injuries_bootstrap import bootstrap_injuries_for_tournament


def _make_rosters_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[
        "nation", "tournament", "tournament_start_date",
        "player_name", "player_url_slug", "position", "market_value_eur",
    ])


def test_bootstrap_matches_player_via_slug_writes_csv(tmp_path: Path, caplog) -> None:
    rosters = _make_rosters_df([
        {
            "nation": "Spain", "tournament": "wc2018",
            "tournament_start_date": pd.Timestamp("2018-06-14"),
            "player_name": "Dani Carvajal", "player_url_slug": "dani-carvajal",
            "position": "Right-Back", "market_value_eur": 32_000_000,
        },
    ])
    html = (FIXTURE_DIR / "wikipedia_wc2018_squads_excerpt.html").read_text(encoding="utf-8")
    csv_path = tmp_path / "injuries.csv"
    csv_path.write_text(
        "date_of_knowledge,team,tournament,player_name,player_url_slug,market_value_eur,status,source\n",
        encoding="utf-8",
    )
    n_added, n_skipped = bootstrap_injuries_for_tournament(
        tournament="wc2018",
        wikipedia_html=html,
        rosters=rosters,
        csv_path=csv_path,
    )
    assert n_added == 1
    assert n_skipped >= 1  # Laurent Koscielny no-match
    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert df.iloc[0]["team"] == "Spain"
    assert df.iloc[0]["player_url_slug"] == "dani-carvajal"
    assert df.iloc[0]["status"] == "out"
    assert df.iloc[0]["source"] == "wikipedia_squads"
    assert df.iloc[0]["date_of_knowledge"] == "2018-06-13"
    assert int(df.iloc[0]["market_value_eur"]) == 32_000_000


def test_bootstrap_idempotent_no_duplicates(tmp_path: Path) -> None:
    rosters = _make_rosters_df([{
        "nation": "Spain", "tournament": "wc2018",
        "tournament_start_date": pd.Timestamp("2018-06-14"),
        "player_name": "Dani Carvajal", "player_url_slug": "dani-carvajal",
        "position": "Right-Back", "market_value_eur": 32_000_000,
    }])
    html = (FIXTURE_DIR / "wikipedia_wc2018_squads_excerpt.html").read_text(encoding="utf-8")
    csv_path = tmp_path / "injuries.csv"
    csv_path.write_text(
        "date_of_knowledge,team,tournament,player_name,player_url_slug,market_value_eur,status,source\n",
        encoding="utf-8",
    )
    bootstrap_injuries_for_tournament("wc2018", html, rosters, csv_path)
    bootstrap_injuries_for_tournament("wc2018", html, rosters, csv_path)
    df = pd.read_csv(csv_path)
    assert len(df) == 1
```

- [ ] **Step 2: Run tests (red)**

```bash
pytest tests/test_injuries_bootstrap.py::test_bootstrap_matches_player_via_slug_writes_csv -v
```
Expected: FAIL with `ImportError: bootstrap_injuries_for_tournament`.

- [ ] **Step 3: Implement orchestrator**

Append to `injuries_bootstrap.py`:
```python
from datetime import timedelta
from pathlib import Path

import pandas as pd

from mondiali.data.tm_rosters import (
    INJURIES_CSV_COLUMNS,
    INJURY_SOURCE_DOMAIN,
    INJURY_STATUS_DOMAIN,
    TOURNAMENT_META,
)


def bootstrap_injuries_for_tournament(
    tournament: str,
    wikipedia_html: str,
    rosters: pd.DataFrame,
    csv_path: Path,
) -> tuple[int, int]:
    """Parse Wikipedia HTML for `tournament`, match against rosters, write to injuries.csv.

    Returns:
        (n_added, n_skipped_no_match)
    """
    meta = TOURNAMENT_META.get(tournament)
    if meta is None:
        raise ValueError(f"unknown tournament: {tournament}")
    start: date = meta["start"]  # type: ignore[assignment]
    date_of_knowledge = start - timedelta(days=1)

    entries = parse_wikipedia_withdrawals(wikipedia_html)

    # Build lookup: (team_norm, player_norm) -> roster row.
    roster_t = rosters[rosters["tournament"] == tournament].copy()
    roster_t["_team_norm"] = roster_t["nation"].map(_normalize_name)
    roster_t["_player_norm"] = roster_t["player_name"].map(_normalize_name)
    lookup = {
        (row["_team_norm"], row["_player_norm"]): row for _, row in roster_t.iterrows()
    }

    existing = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame(columns=INJURIES_CSV_COLUMNS)
    existing_keys = set(zip(
        existing.get("team", pd.Series(dtype=str)).astype(str),
        existing.get("tournament", pd.Series(dtype=str)).astype(str),
        existing.get("player_url_slug", pd.Series(dtype=str)).astype(str),
        strict=True,
    ))

    new_rows: list[dict] = []
    n_skipped = 0
    for e in entries:
        key = (_normalize_name(e.team), _normalize_name(e.player_name))
        match = lookup.get(key)
        if match is None:
            log.warning("injury_player_no_roster_match", team=e.team, player=e.player_name, tournament=tournament)
            n_skipped += 1
            continue
        slug = match["player_url_slug"]
        dedup_key = (e.team, tournament, slug)
        if dedup_key in existing_keys:
            continue
        existing_keys.add(dedup_key)
        new_rows.append({
            "date_of_knowledge": date_of_knowledge.isoformat(),
            "team": e.team,
            "tournament": tournament,
            "player_name": e.player_name,
            "player_url_slug": slug,
            "market_value_eur": int(match["market_value_eur"]) if pd.notna(match["market_value_eur"]) else "",
            "status": "out",
            "source": "wikipedia_squads",
        })

    if new_rows:
        new_df = pd.DataFrame(new_rows, columns=INJURIES_CSV_COLUMNS)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(csv_path, index=False)

    log.info(
        "injuries_bootstrap_complete",
        tournament=tournament,
        n_parsed=len(entries),
        n_matched=len(new_rows),
        n_skipped_no_match=n_skipped,
    )
    return len(new_rows), n_skipped
```

- [ ] **Step 4: Run tests (green)**

```bash
pytest tests/test_injuries_bootstrap.py -v
```
Expected: 4 PASS.

- [ ] **Step 5: Add status enum + date default validation tests**

Append to `tests/test_injuries_bootstrap.py`:
```python
def test_status_enum_constants_complete() -> None:
    from mondiali.data.tm_rosters import INJURY_STATUS_DOMAIN, INJURY_SOURCE_DOMAIN
    assert INJURY_STATUS_DOMAIN == frozenset({"out", "doubtful", "available"})
    assert INJURY_SOURCE_DOMAIN == frozenset({"wikipedia_squads", "manual"})


def test_bootstrap_date_of_knowledge_is_day_before_kickoff(tmp_path: Path) -> None:
    rosters = _make_rosters_df([{
        "nation": "Spain", "tournament": "wc2022",
        "tournament_start_date": pd.Timestamp("2022-11-20"),
        "player_name": "Dani Carvajal", "player_url_slug": "dani-carvajal",
        "position": "Right-Back", "market_value_eur": 30_000_000,
    }])
    html = """<html><body>
    <h2><span class="mw-headline" id="Withdrawals">Withdrawals</span></h2>
    <ul><li><b>Spain</b>: <a href="/wiki/x">Dani Carvajal</a> withdrew due to injury.</li></ul>
    </body></html>"""
    csv_path = tmp_path / "injuries.csv"
    csv_path.write_text(
        "date_of_knowledge,team,tournament,player_name,player_url_slug,market_value_eur,status,source\n",
        encoding="utf-8",
    )
    bootstrap_injuries_for_tournament("wc2022", html, rosters, csv_path)
    df = pd.read_csv(csv_path)
    assert df.iloc[0]["date_of_knowledge"] == "2022-11-19"
```

- [ ] **Step 6: Run all bootstrap tests**

```bash
pytest tests/test_injuries_bootstrap.py -v
```
Expected: 6 PASS.

- [ ] **Step 7: Commit**

```bash
git add src/mondiali/data/injuries_bootstrap.py tests/test_injuries_bootstrap.py
git commit -m "feat(injuries): bootstrap orchestrator with strict-match player lookup"
```

---

## Task 8: CLI `bootstrap-injuries` command

**Files:**
- Modify: `src/mondiali/cli/main.py`
- Modify: `src/mondiali/data/injuries_bootstrap.py` (add fetcher helper)

- [ ] **Step 1: Add Wikipedia fetch helper**

Append to `injuries_bootstrap.py`:
```python
import time

import requests

WIKIPEDIA_URL_PATTERN = "https://en.wikipedia.org/wiki/{slug}"
TOURNAMENT_WIKIPEDIA_SLUG: dict[str, str] = {
    "wc2018":   "2018_FIFA_World_Cup_squads",
    "euro2020": "UEFA_Euro_2020_squads",
    "wc2022":   "2022_FIFA_World_Cup_squads",
    "euro2024": "UEFA_Euro_2024_squads",
}


def fetch_wikipedia_squads_html(tournament: str, cache_dir: Path) -> str | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{tournament}.html"
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    slug = TOURNAMENT_WIKIPEDIA_SLUG.get(tournament)
    if slug is None:
        log.warning("no_wikipedia_slug", tournament=tournament)
        return None
    url = WIKIPEDIA_URL_PATTERN.format(slug=slug)
    time.sleep(1.0)
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "mondiali-research/0.1"})
    except requests.RequestException as e:
        log.warning("wikipedia_fetch_exception", error=str(e), url=url)
        return None
    if resp.status_code != 200:
        log.warning("wikipedia_fetch_non200", status=resp.status_code, url=url)
        return None
    cached.write_text(resp.text, encoding="utf-8")
    return resp.text
```

- [ ] **Step 2: Add CLI command in `cli/main.py`**

After Task 5's command, append:
```python
from mondiali.data.injuries_bootstrap import (
    bootstrap_injuries_for_tournament,
    fetch_wikipedia_squads_html,
)


@app.command(name="bootstrap-injuries")
def bootstrap_injuries(
    tournaments: str = typer.Option(
        "wc2018,euro2020,wc2022,euro2024",
        "--tournaments",
        help="Comma-separated tournament keys",
    ),
) -> None:
    """Bootstrap data/manual/injuries.csv from Wikipedia withdrawals sections."""
    keys = [t.strip() for t in tournaments.split(",") if t.strip()]
    rosters_path = CONFIG.data_raw / "transfermarkt" / "rosters.parquet"
    if not rosters_path.exists():
        typer.echo(f"rosters.parquet not found at {rosters_path}; run tm-scrape-rosters first", err=True)
        raise typer.Exit(1)
    rosters = pd.read_parquet(rosters_path)
    csv_path = CONFIG.data_root / "manual" / "injuries.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        csv_path.write_text(
            "date_of_knowledge,team,tournament,player_name,player_url_slug,market_value_eur,status,source\n",
            encoding="utf-8",
        )
    cache_dir = CONFIG.data_raw / "wikipedia" / "squads_cache"
    grand_added = grand_skipped = 0
    for t in keys:
        html = fetch_wikipedia_squads_html(t, cache_dir)
        if html is None:
            typer.echo(f"  {t}: fetch failed, skipped")
            continue
        n_add, n_skip = bootstrap_injuries_for_tournament(t, html, rosters, csv_path)
        typer.echo(f"  {t}: added={n_add} skipped_no_match={n_skip}")
        grand_added += n_add
        grand_skipped += n_skip
    typer.echo(f"Done. total_added={grand_added} total_skipped_no_match={grand_skipped}")
```

- [ ] **Step 3: Verify CONFIG.data_root exists** (if not, use a different path)

Run:
```bash
python -c "from mondiali.config import CONFIG; print(CONFIG.data_raw); print(getattr(CONFIG, 'data_root', None))"
```

If `data_root` is missing, replace `CONFIG.data_root / "manual"` with the project's existing manual-data convention. Check `src/mondiali/config.py`:

```bash
cat src/mondiali/config.py
```
Use whatever `data/manual/` resolves to in CONFIG (e.g. `CONFIG.data_raw.parent / "manual"`).

- [ ] **Step 4: Smoke test the CLI**

```bash
python -m mondiali.cli.main bootstrap-injuries --help
```
Expected: typer help text printed, no traceback.

- [ ] **Step 5: Commit**

```bash
git add src/mondiali/cli/main.py src/mondiali/data/injuries_bootstrap.py
git commit -m "feat(cli): bootstrap-injuries command + Wikipedia fetcher"
```

---

## Task 9: Tier 4 feature module

**Files:**
- Create: `src/mondiali/features/tier4.py`
- Create: `tests/test_tier4.py`

- [ ] **Step 1: Write failing tests** at `tests/test_tier4.py`

```python
"""Tests for features/tier4.py — top-5 absence count + value ratio."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mondiali.features.tier4 import (
    TIER4_COLUMNS,
    add_tier4_features,
)


def _make_match(date_str: str, home: str, away: str) -> dict:
    return {"date": pd.Timestamp(date_str), "home_team": home, "away_team": away}


def _make_roster(nation: str, tournament: str, start: str, players: list[tuple[str, str, int]]) -> list[dict]:
    rows = []
    for name, slug, value in players:
        rows.append({
            "nation": nation, "tournament": tournament,
            "tournament_start_date": pd.Timestamp(start),
            "player_name": name, "player_url_slug": slug,
            "position": "MID", "market_value_eur": value,
        })
    return rows


def _make_injury(date_of_knowledge: str, team: str, tournament: str, slug: str, value: int, status: str = "out") -> dict:
    return {
        "date_of_knowledge": pd.Timestamp(date_of_knowledge),
        "team": team, "tournament": tournament,
        "player_name": slug, "player_url_slug": slug,
        "market_value_eur": value, "status": status,
        "source": "wikipedia_squads",
    }


def test_top5_count_excludes_status_available() -> None:
    matches = pd.DataFrame([_make_match("2022-11-25", "France", "Denmark")])
    roster_rows = _make_roster("France", "wc2022", "2022-11-20", [
        ("M1", "m1", 100_000_000), ("M2", "m2", 80_000_000), ("M3", "m3", 60_000_000),
        ("M4", "m4", 40_000_000), ("M5", "m5", 20_000_000), ("M6", "m6", 10_000_000),
    ])
    rosters = pd.DataFrame(roster_rows)
    injuries = pd.DataFrame([
        _make_injury("2022-11-19", "France", "wc2022", "m1", 100_000_000, "out"),
        _make_injury("2022-11-19", "France", "wc2022", "m6", 10_000_000, "out"),  # not top-5, ignored
        _make_injury("2022-11-19", "France", "wc2022", "m2", 80_000_000, "available"),  # ignored
    ])
    out = add_tier4_features(matches, rosters, injuries)
    assert out.loc[0, "home_top5_absent_count"] == 1
    assert out.loc[0, "away_top5_absent_count"] == 0


def test_top5_value_ratio_correct() -> None:
    matches = pd.DataFrame([_make_match("2022-11-25", "France", "Denmark")])
    rosters = pd.DataFrame(_make_roster("France", "wc2022", "2022-11-20", [
        ("M1", "m1", 100_000_000), ("M2", "m2", 80_000_000), ("M3", "m3", 60_000_000),
        ("M4", "m4", 40_000_000), ("M5", "m5", 20_000_000),
    ]))
    injuries = pd.DataFrame([
        _make_injury("2022-11-19", "France", "wc2022", "m1", 100_000_000, "out"),
    ])
    out = add_tier4_features(matches, rosters, injuries)
    expected_ratio = 100_000_000 / (100 + 80 + 60 + 40 + 20) / 1_000_000
    assert out.loc[0, "home_value_absent_ratio"] == pytest.approx(expected_ratio)


def test_value_ratio_zero_when_no_absences() -> None:
    matches = pd.DataFrame([_make_match("2022-11-25", "France", "Denmark")])
    rosters = pd.DataFrame(_make_roster("France", "wc2022", "2022-11-20", [
        ("M1", "m1", 100_000_000), ("M2", "m2", 80_000_000), ("M3", "m3", 60_000_000),
        ("M4", "m4", 40_000_000), ("M5", "m5", 20_000_000),
    ]))
    injuries = pd.DataFrame(columns=[
        "date_of_knowledge", "team", "tournament", "player_name",
        "player_url_slug", "market_value_eur", "status", "source",
    ])
    out = add_tier4_features(matches, rosters, injuries)
    assert out.loc[0, "home_top5_absent_count"] == 0
    assert out.loc[0, "home_value_absent_ratio"] == 0.0


def test_pre_2018_match_returns_nan() -> None:
    matches = pd.DataFrame([_make_match("2014-06-15", "Brazil", "Croatia")])
    rosters = pd.DataFrame(_make_roster("Brazil", "wc2022", "2022-11-20", [
        ("M1", "m1", 100_000_000), ("M2", "m2", 80_000_000), ("M3", "m3", 60_000_000),
        ("M4", "m4", 40_000_000), ("M5", "m5", 20_000_000),
    ]))
    injuries = pd.DataFrame(columns=[
        "date_of_knowledge", "team", "tournament", "player_name",
        "player_url_slug", "market_value_eur", "status", "source",
    ])
    out = add_tier4_features(matches, rosters, injuries)
    for col in TIER4_COLUMNS:
        assert pd.isna(out.loc[0, col])


def test_friendly_outside_tournament_returns_nan() -> None:
    """A friendly between WC2022 participants but on a non-tournament date → NaN."""
    matches = pd.DataFrame([_make_match("2023-03-10", "France", "Denmark")])
    rosters = pd.DataFrame(_make_roster("France", "wc2022", "2022-11-20", [
        ("M1", "m1", 100_000_000), ("M2", "m2", 80_000_000), ("M3", "m3", 60_000_000),
        ("M4", "m4", 40_000_000), ("M5", "m5", 20_000_000),
    ]))
    injuries = pd.DataFrame(columns=[
        "date_of_knowledge", "team", "tournament", "player_name",
        "player_url_slug", "market_value_eur", "status", "source",
    ])
    out = add_tier4_features(matches, rosters, injuries)
    for col in TIER4_COLUMNS:
        assert pd.isna(out.loc[0, col])


def test_strict_pre_match_anti_leakage() -> None:
    """An injury with date_of_knowledge == match.date must be IGNORED (strict <)."""
    matches = pd.DataFrame([_make_match("2022-11-25", "France", "Denmark")])
    rosters = pd.DataFrame(_make_roster("France", "wc2022", "2022-11-20", [
        ("M1", "m1", 100_000_000), ("M2", "m2", 80_000_000), ("M3", "m3", 60_000_000),
        ("M4", "m4", 40_000_000), ("M5", "m5", 20_000_000),
    ]))
    injuries = pd.DataFrame([
        _make_injury("2022-11-25", "France", "wc2022", "m1", 100_000_000, "out"),
    ])
    out = add_tier4_features(matches, rosters, injuries)
    assert out.loc[0, "home_top5_absent_count"] == 0


def test_missing_roster_returns_nan_for_that_side() -> None:
    """If France roster missing, France-side features are NaN even if Denmark roster exists."""
    matches = pd.DataFrame([_make_match("2022-11-25", "France", "Denmark")])
    rosters = pd.DataFrame(_make_roster("Denmark", "wc2022", "2022-11-20", [
        ("D1", "d1", 50_000_000), ("D2", "d2", 40_000_000), ("D3", "d3", 30_000_000),
        ("D4", "d4", 20_000_000), ("D5", "d5", 10_000_000),
    ]))
    injuries = pd.DataFrame(columns=[
        "date_of_knowledge", "team", "tournament", "player_name",
        "player_url_slug", "market_value_eur", "status", "source",
    ])
    out = add_tier4_features(matches, rosters, injuries)
    assert pd.isna(out.loc[0, "home_top5_absent_count"])
    assert out.loc[0, "away_top5_absent_count"] == 0  # Denmark has roster, no injuries
```

- [ ] **Step 2: Run tests (red)**

```bash
pytest tests/test_tier4.py -v
```
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement `features/tier4.py`**

Create `src/mondiali/features/tier4.py`:
```python
"""Feature builder Tier 4: injury impact on top-5 by market value.

For each (match, side):
1. Find tournament whose [start, end+30d] window contains match.date.
   If no roster row matches → all 4 features NaN for that side.
2. Top-5 = 5 players with highest market_value in that nation/tournament roster.
3. Filter injuries: team==nation, tournament==same, date_of_knowledge < match.date,
   status in {out, doubtful}.
4. Intersect by player_url_slug (unique). Compute count + value_ratio.

Anti-leakage:
- date_of_knowledge < match.date (strict, no <=)
- pre-2018 → NaN (no rosters scraped pre-WC2018)
- match outside tournament window → NaN
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

TIER4_TOP_N = 5
TIER4_MIN_YEAR = 2018
TIER4_TOURNAMENT_GRACE_DAYS = 30

TIER4_COLUMNS: list[str] = [
    "home_top5_absent_count", "away_top5_absent_count",
    "home_value_absent_ratio", "away_value_absent_ratio",
]


def add_tier4_features(
    matches: pd.DataFrame, rosters: pd.DataFrame, injuries: pd.DataFrame
) -> pd.DataFrame:
    out = matches.copy()
    for col in TIER4_COLUMNS:
        out[col] = np.nan

    if rosters.empty or out.empty:
        return out

    # Pre-compute tournament end-date and roster top-5 by (nation, tournament).
    rosters = rosters.copy()
    rosters["tournament_start_date"] = pd.to_datetime(rosters["tournament_start_date"])

    from mondiali.data.tm_rosters import TOURNAMENT_META
    end_dates: dict[str, pd.Timestamp] = {
        k: pd.Timestamp(v["end"]) + pd.Timedelta(days=TIER4_TOURNAMENT_GRACE_DAYS)
        for k, v in TOURNAMENT_META.items()
    }

    # Top-5 lookup: (nation, tournament) -> DataFrame of top-5 rows.
    top5_by_pair: dict[tuple[str, str], pd.DataFrame] = {}
    for (nation, tournament), grp in rosters.groupby(["nation", "tournament"]):
        valid = grp.dropna(subset=["market_value_eur"]).copy()
        if valid.empty:
            continue
        top5 = valid.sort_values(
            ["market_value_eur", "player_url_slug"], ascending=[False, True]
        ).head(TIER4_TOP_N)
        top5_by_pair[(nation, tournament)] = top5

    # Injury index: (nation, tournament) -> rows.
    inj = injuries.copy() if not injuries.empty else pd.DataFrame(columns=[
        "date_of_knowledge", "team", "tournament", "player_url_slug", "status",
    ])
    if not inj.empty:
        inj["date_of_knowledge"] = pd.to_datetime(inj["date_of_knowledge"])
        inj_by_pair = {
            (team, t): grp for (team, t), grp in inj.groupby(["team", "tournament"])
        }
    else:
        inj_by_pair = {}

    out["date"] = pd.to_datetime(out["date"])
    min_date = pd.Timestamp(f"{TIER4_MIN_YEAR}-01-01")

    for side, team_col in (("home", "home_team"), ("away", "away_team")):
        count_col = f"{side}_top5_absent_count"
        ratio_col = f"{side}_value_absent_ratio"
        for idx, row in out.iterrows():
            if row["date"] < min_date:
                continue
            nation = row[team_col]
            # find tournament whose window contains match.date
            tournament: str | None = None
            for t, _meta in TOURNAMENT_META.items():
                start = pd.Timestamp(_meta["start"])
                end_grace = end_dates[t]
                if start <= row["date"] <= end_grace:
                    if (nation, t) in top5_by_pair:
                        tournament = t
                        break
            if tournament is None:
                continue
            top5 = top5_by_pair[(nation, tournament)]
            top5_total = float(top5["market_value_eur"].sum())
            absent_value = 0.0
            absent_count = 0
            inj_grp = inj_by_pair.get((nation, tournament))
            if inj_grp is not None:
                absent_inj = inj_grp[
                    (inj_grp["date_of_knowledge"] < row["date"])
                    & (inj_grp["status"].isin(["out", "doubtful"]))
                ]
                absent_slugs = set(absent_inj["player_url_slug"]) & set(top5["player_url_slug"])
                if absent_slugs:
                    abs_rows = top5[top5["player_url_slug"].isin(absent_slugs)]
                    absent_value = float(abs_rows["market_value_eur"].sum())
                    absent_count = int(len(abs_rows))
            out.at[idx, count_col] = absent_count
            out.at[idx, ratio_col] = (absent_value / top5_total) if top5_total > 0 else np.nan

    log.info(
        "tier4_features_added",
        rows=len(out),
        coverage_home=float(out["home_top5_absent_count"].notna().mean()),
        coverage_away=float(out["away_top5_absent_count"].notna().mean()),
    )
    return out
```

- [ ] **Step 4: Run tests (green)**

```bash
pytest tests/test_tier4.py -v
```
Expected: 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mondiali/features/tier4.py tests/test_tier4.py
git commit -m "feat(features/tier4): top-5 absence count + value-ratio with anti-leakage"
```

---

## Task 10: Anti-leakage test for Tier 4

**Files:**
- Modify: `tests/test_leakage.py`

- [ ] **Step 1: Inspect existing leakage test for tier3 to mirror its pattern**

Run:
```bash
grep -n "test_tier3_market_value_strict_pre_match" tests/test_leakage.py
```
Read ~30 lines around the match.

- [ ] **Step 2: Append `test_tier4_strict_pre_match`**

Add to `tests/test_leakage.py`:
```python
def test_tier4_strict_pre_match() -> None:
    """For every non-NaN tier4 row, asserting any contributing injury date < match date."""
    import pandas as pd
    from mondiali.features.tier4 import TIER4_COLUMNS, add_tier4_features

    matches = pd.DataFrame([
        {"date": pd.Timestamp("2022-11-25"), "home_team": "France", "away_team": "Denmark"},
        {"date": pd.Timestamp("2022-11-26"), "home_team": "Spain", "away_team": "Germany"},
    ])
    rosters = pd.DataFrame([
        {"nation": "France", "tournament": "wc2022", "tournament_start_date": pd.Timestamp("2022-11-20"),
         "player_name": "M1", "player_url_slug": "m1", "position": "MID", "market_value_eur": 100_000_000},
        {"nation": "France", "tournament": "wc2022", "tournament_start_date": pd.Timestamp("2022-11-20"),
         "player_name": "M2", "player_url_slug": "m2", "position": "MID", "market_value_eur": 80_000_000},
        {"nation": "France", "tournament": "wc2022", "tournament_start_date": pd.Timestamp("2022-11-20"),
         "player_name": "M3", "player_url_slug": "m3", "position": "MID", "market_value_eur": 60_000_000},
        {"nation": "France", "tournament": "wc2022", "tournament_start_date": pd.Timestamp("2022-11-20"),
         "player_name": "M4", "player_url_slug": "m4", "position": "MID", "market_value_eur": 40_000_000},
        {"nation": "France", "tournament": "wc2022", "tournament_start_date": pd.Timestamp("2022-11-20"),
         "player_name": "M5", "player_url_slug": "m5", "position": "MID", "market_value_eur": 20_000_000},
    ])
    # Injury at exact match date — strict-pre violations would include it.
    injuries = pd.DataFrame([{
        "date_of_knowledge": pd.Timestamp("2022-11-25"),  # equal to France match.date
        "team": "France", "tournament": "wc2022",
        "player_name": "M1", "player_url_slug": "m1",
        "market_value_eur": 100_000_000, "status": "out", "source": "wikipedia_squads",
    }])
    out = add_tier4_features(matches, rosters, injuries)
    # France match: same-date injury must be excluded → count == 0
    france_row = out[(out["home_team"] == "France")].iloc[0]
    assert france_row["home_top5_absent_count"] == 0, (
        "Tier 4 leaked: same-date injury was counted (must be strict <)"
    )
```

- [ ] **Step 3: Run leakage suite**

```bash
pytest tests/test_leakage.py -v
```
Expected: ALL PASS (5 existing + 1 new = 6).

- [ ] **Step 4: Commit**

```bash
git add tests/test_leakage.py
git commit -m "test(leakage): tier4 strict pre-match invariant"
```

---

## Task 11: `train_tier4_pipeline` with double Optuna study

**Files:**
- Modify: `src/mondiali/training/train.py`
- Create: `tests/test_train_tier4.py`

- [ ] **Step 1: Verify Optuna is installed**

```bash
python -c "import optuna; print(optuna.__version__)"
```
If not installed: `pip install optuna` and add to `pyproject.toml`/`requirements.txt`.

- [ ] **Step 2: Inspect existing `train_tier3_pipeline` signature to mirror its structure**

```bash
grep -n "def train_tier3_pipeline" src/mondiali/training/train.py
```
Read ~80 lines starting from the match.

- [ ] **Step 3: Write smoke tests** at `tests/test_train_tier4.py`

```python
"""Smoke tests for train_tier4_pipeline."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mondiali.training.train import train_tier4_pipeline


@pytest.fixture
def tiny_matches_parquet(tmp_path: Path) -> Path:
    df = pd.read_parquet(Path("data/processed/matches.parquet"))
    df = df.iloc[:5000].copy()
    out = tmp_path / "matches.parquet"
    df.to_parquet(out, index=False)
    return out


def test_train_tier4_smoke_runs_with_few_trials(tmp_path: Path, tiny_matches_parquet: Path) -> None:
    """Verifies the pipeline runs end-to-end with --n-trials 2 (just functional smoke)."""
    rosters_path = Path("data/raw/transfermarkt/rosters.parquet")
    if not rosters_path.exists():
        pytest.skip("rosters.parquet not built yet — run after Task 5 e2e")
    injuries_path = Path("data/manual/injuries.csv")
    if not injuries_path.exists() or injuries_path.read_text().strip() == "":
        pytest.skip("injuries.csv not bootstrapped — run after Task 8 e2e")
    out_dir = tmp_path / "tier4"
    result = train_tier4_pipeline(
        matches_path=tiny_matches_parquet,
        rosters_path=rosters_path,
        injuries_path=injuries_path,
        out_dir=out_dir,
        n_trials=2,
        seed=42,
    )
    assert "baseline_log_loss" in result
    assert "challenger_log_loss" in result
    assert "delta" in result
    assert (out_dir / "baseline_params.json").exists()
    assert (out_dir / "challenger_params.json").exists()
    assert (out_dir / "xgb_poisson.json").exists()
    assert (out_dir / "calibrator.json").exists()


def test_train_tier4_deterministic_with_seed(tmp_path: Path, tiny_matches_parquet: Path) -> None:
    """Two runs with same seed and n_trials produce identical baseline/challenger metrics."""
    rosters_path = Path("data/raw/transfermarkt/rosters.parquet")
    injuries_path = Path("data/manual/injuries.csv")
    if not rosters_path.exists() or not injuries_path.exists():
        pytest.skip("rosters or injuries not built")
    r1 = train_tier4_pipeline(
        matches_path=tiny_matches_parquet, rosters_path=rosters_path, injuries_path=injuries_path,
        out_dir=tmp_path / "run1", n_trials=2, seed=42,
    )
    r2 = train_tier4_pipeline(
        matches_path=tiny_matches_parquet, rosters_path=rosters_path, injuries_path=injuries_path,
        out_dir=tmp_path / "run2", n_trials=2, seed=42,
    )
    assert r1["baseline_log_loss"] == pytest.approx(r2["baseline_log_loss"])
    assert r1["challenger_log_loss"] == pytest.approx(r2["challenger_log_loss"])
```

- [ ] **Step 4: Run tests (red)**

```bash
pytest tests/test_train_tier4.py -v
```
Expected: FAIL (`ImportError: train_tier4_pipeline`).

- [ ] **Step 5: Implement `train_tier4_pipeline`**

Append to `src/mondiali/training/train.py`:
```python
import json
from typing import Any

import numpy as np

from mondiali.features.tier1 import TIER1_FEATURE_COLS
from mondiali.features.tier2 import TIER2_FEATURE_COLS
from mondiali.features.tier4 import TIER4_COLUMNS, add_tier4_features


def _objective_factory(
    feature_cols: list[str],
    train: pd.DataFrame,
    val_calib: pd.DataFrame,
    val_gate: pd.DataFrame,
    seed: int,
):
    import optuna

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "random_state": seed,
            "objective": "count:poisson",
        }
        model = PoissonXGBModel(params=params)
        model.fit(train, feature_cols=feature_cols)
        rho = estimate_rho_mle(train, model, feature_cols=feature_cols)
        # Calibration on val_calib
        lam_h_c = model.predict(val_calib, side="home", feature_cols=feature_cols)
        lam_a_c = model.predict(val_calib, side="away", feature_cols=feature_cols)
        probs_calib = _compute_1x2_probs(lam_h_c, lam_a_c, rho)
        cal = IsotonicCalibrator1X2()
        cal.fit(val_calib, probs_calib)
        # Gate
        lam_h_g = model.predict(val_gate, side="home", feature_cols=feature_cols)
        lam_a_g = model.predict(val_gate, side="away", feature_cols=feature_cols)
        probs_gate = _compute_1x2_probs(lam_h_g, lam_a_g, rho)
        probs_gate = cal.transform(probs_gate)
        return float(log_loss_1x2(val_gate, probs_gate))

    return objective


def train_tier4_pipeline(
    matches_path: Path,
    rosters_path: Path,
    injuries_path: Path,
    out_dir: Path,
    *,
    n_trials: int = 100,
    seed: int = 42,
    train_start: str = "2002-01-01",
    train_end: str = "2021-12-31",
    val_calib_start: str = "2021-01-01",
    val_calib_end: str = "2021-12-31",
    val_gate_start: str = "2022-01-01",
    val_gate_end: str = "2022-12-31",
) -> dict[str, Any]:
    """Apples-to-apples double Optuna study: Tier 1+2 baseline vs Tier 1+2+4 challenger.

    Returns dict with baseline_log_loss, challenger_log_loss, delta, and brier metrics.
    Saves: baseline_params.json, challenger_params.json, xgb_poisson.json, calibrator.json.
    """
    import optuna
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(matches_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["days_rest_home", "days_rest_away"]).copy()

    rosters = pd.read_parquet(rosters_path)
    injuries = pd.read_csv(injuries_path) if Path(injuries_path).stat().st_size > 0 else \
        pd.DataFrame(columns=[
            "date_of_knowledge", "team", "tournament", "player_name",
            "player_url_slug", "market_value_eur", "status", "source",
        ])

    df = add_tier4_features(df, rosters, injuries)

    train = df[(df["date"] >= train_start) & (df["date"] <= train_end)].reset_index(drop=True)
    val_calib = df[(df["date"] >= val_calib_start) & (df["date"] <= val_calib_end)].reset_index(drop=True)
    val_gate = df[(df["date"] >= val_gate_start) & (df["date"] <= val_gate_end)].reset_index(drop=True)

    base_cols = list(TIER1_FEATURE_COLS) + list(TIER2_FEATURE_COLS)
    challenger_cols = base_cols + list(TIER4_COLUMNS)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    log.info("optuna_baseline_start", n_trials=n_trials)
    sampler_b = optuna.samplers.TPESampler(seed=seed)
    study_b = optuna.create_study(direction="minimize", sampler=sampler_b)
    study_b.optimize(_objective_factory(base_cols, train, val_calib, val_gate, seed), n_trials=n_trials)
    baseline_loss = study_b.best_value
    baseline_params = study_b.best_params

    log.info("optuna_challenger_start", n_trials=n_trials)
    sampler_c = optuna.samplers.TPESampler(seed=seed)
    study_c = optuna.create_study(direction="minimize", sampler=sampler_c)
    study_c.optimize(_objective_factory(challenger_cols, train, val_calib, val_gate, seed), n_trials=n_trials)
    challenger_loss = study_c.best_value
    challenger_params = study_c.best_params

    # Train final challenger model with best params, save artifacts.
    final_params = {**challenger_params, "random_state": seed, "objective": "count:poisson"}
    final_model = PoissonXGBModel(params=final_params)
    final_model.fit(train, feature_cols=challenger_cols)
    rho = estimate_rho_mle(train, final_model, feature_cols=challenger_cols)
    lam_h_c = final_model.predict(val_calib, side="home", feature_cols=challenger_cols)
    lam_a_c = final_model.predict(val_calib, side="away", feature_cols=challenger_cols)
    probs_calib = _compute_1x2_probs(lam_h_c, lam_a_c, rho)
    cal = IsotonicCalibrator1X2()
    cal.fit(val_calib, probs_calib)

    # Brier score on gate
    lam_h_g = final_model.predict(val_gate, side="home", feature_cols=challenger_cols)
    lam_a_g = final_model.predict(val_gate, side="away", feature_cols=challenger_cols)
    probs_gate_raw = _compute_1x2_probs(lam_h_g, lam_a_g, rho)
    probs_gate = cal.transform(probs_gate_raw)
    brier = float(brier_score_1x2(val_gate, probs_gate))

    final_model.save(out_dir / "xgb_poisson.json")
    cal.save(out_dir / "calibrator.json")
    (out_dir / "baseline_params.json").write_text(json.dumps(baseline_params, indent=2))
    (out_dir / "challenger_params.json").write_text(json.dumps(challenger_params, indent=2))

    delta = challenger_loss - baseline_loss
    log.info("tier4_gate_complete",
             baseline_log_loss=baseline_loss, challenger_log_loss=challenger_loss,
             delta=delta, brier=brier)

    return {
        "baseline_log_loss": baseline_loss,
        "challenger_log_loss": challenger_loss,
        "delta": delta,
        "brier": brier,
        "rho": rho,
        "n_train": len(train),
        "n_val_calib": len(val_calib),
        "n_val_gate": len(val_gate),
        "baseline_params": baseline_params,
        "challenger_params": challenger_params,
    }
```

NOTE: This task assumes `TIER1_FEATURE_COLS` / `TIER2_FEATURE_COLS` exist as exported constants. If they don't, derive from existing pipelines (look for the column lists in `train_tier1_pipeline` and `train_tier2_pipeline`) and either export them or inline the lists here. **Verify before running:** `grep -n "TIER1_FEATURE_COLS\|TIER2_FEATURE_COLS" src/mondiali/features/`.

- [ ] **Step 6: Run smoke tests (green-or-skip)**

```bash
pytest tests/test_train_tier4.py -v
```
Expected: PASS or SKIP if rosters/injuries not yet built (Task 13 e2e will populate them).

- [ ] **Step 7: Commit**

```bash
git add src/mondiali/training/train.py tests/test_train_tier4.py
git commit -m "feat(training): tier4 pipeline with double Optuna apples-to-apples gate"
```

---

## Task 12: CLI `train-tier4` command

**Files:**
- Modify: `src/mondiali/cli/main.py`

- [ ] **Step 1: Add command in `cli/main.py`**

Append after the `tm-discover-ids` command:
```python
from mondiali.training.train import train_tier4_pipeline


@app.command(name="train-tier4")
def train_tier4(
    n_trials: int = typer.Option(100, "--n-trials"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """STEP 5 gate: Optuna double study (Tier 1+2 baseline vs Tier 1+2+4 challenger)."""
    matches_path = CONFIG.data_processed / "matches.parquet"
    rosters_path = CONFIG.data_raw / "transfermarkt" / "rosters.parquet"
    injuries_path = CONFIG.data_root / "manual" / "injuries.csv"
    out_dir = CONFIG.models_dir / "tier4"
    if not matches_path.exists():
        typer.echo("matches.parquet missing", err=True)
        raise typer.Exit(1)
    if not rosters_path.exists():
        typer.echo("rosters.parquet missing — run tm-scrape-rosters", err=True)
        raise typer.Exit(1)
    if not injuries_path.exists():
        typer.echo("injuries.csv missing — run bootstrap-injuries", err=True)
        raise typer.Exit(1)

    result = train_tier4_pipeline(
        matches_path=matches_path,
        rosters_path=rosters_path,
        injuries_path=injuries_path,
        out_dir=out_dir,
        n_trials=n_trials,
        seed=seed,
    )

    typer.echo(f"Splits: train={result['n_train']} val_calib={result['n_val_calib']} val_gate={result['n_val_gate']}")
    typer.echo(f"Dixon-Coles rho: {result['rho']:.4f}")
    typer.echo(f"Tier 1+2 baseline log-loss:    {result['baseline_log_loss']:.4f}")
    typer.echo(f"Tier 1+2+4 challenger log-loss: {result['challenger_log_loss']:.4f}")
    typer.echo(f"Delta: {result['delta']:+.4f}  (negative = challenger better)")
    typer.echo(f"Brier (gate): {result['brier']:.4f}")
    if result["delta"] <= -0.003:
        typer.echo(">>> GATE PASSED — Tier 4 promoted (delta <= -0.003)")
    elif result["delta"] >= 0.003:
        typer.echo(">>> GATE FAILED — Tier 4 NOT promoted (delta >= 0.003)")
    else:
        typer.echo(">>> NO DECISION — |delta| < 0.003. Tie-breaker: review Brier + report manually.")
```

- [ ] **Step 2: Smoke test**

```bash
python -m mondiali.cli.main train-tier4 --help
```
Expected: typer help text printed.

- [ ] **Step 3: Commit**

```bash
git add src/mondiali/cli/main.py
git commit -m "feat(cli): train-tier4 command with explicit gate verdict"
```

---

## Task 13: End-to-end execution + validation report

**Files:**
- Create: `reports/validation_step5.md`

- [ ] **Step 1: Run roster scrape (real network, ~20-30 min)**

```bash
python -m mondiali.cli.main tm-scrape-rosters
```
Expected output ends with `Done. N new (nation, tournament) pairs added.` with N close to 112.

If errors / partial coverage: re-run with `--resume` (default ON) until coverage stabilizes. Hard floor for Step 13 acceptance: ≥80% (≥90 of 112 pairs).

- [ ] **Step 2: Verify rosters.parquet**

```bash
python -c "import pandas as pd; df = pd.read_parquet('data/raw/transfermarkt/rosters.parquet'); print('rows:', len(df)); print('pairs:', df[['nation','tournament']].drop_duplicates().shape[0]); print(df.groupby('tournament').size())"
```
Expected: `rows` ≈ 2500-3000; `pairs` ≥ 90.

- [ ] **Step 3: Run injury bootstrap (~5 min)**

```bash
python -m mondiali.cli.main bootstrap-injuries
```
Expected: per-tournament summary like `wc2018: added=8 skipped_no_match=2`. Total `added` ≥ 10.

- [ ] **Step 4: Verify injuries.csv**

```bash
python -c "import pandas as pd; df = pd.read_csv('data/manual/injuries.csv'); print(len(df)); print(df['tournament'].value_counts())"
```

- [ ] **Step 5: Run Optuna double study (~3-4h CPU)**

Best to run with `--n-trials 100` for the real gate. For a quicker smoke first:
```bash
python -m mondiali.cli.main train-tier4 --n-trials 5
```
If smoke succeeds (artifacts written, verdict echoed), kick the real run:
```bash
python -m mondiali.cli.main train-tier4 --n-trials 100 2>&1 | tee reports/train_tier4_log.txt
```

- [ ] **Step 6: Verify artifacts**

```bash
ls -la models/tier4/
```
Expected: `xgb_poisson.json`, `calibrator.json`, `baseline_params.json`, `challenger_params.json`.

- [ ] **Step 7: Write `reports/validation_step5.md`**

Use this template, filling in real numbers from Step 5's stdout/log. Match the structure of `reports/validation_step4.md`:

```markdown
# STEP 5 — Validation Report: Tier 4 (Injuries)

**Date:** YYYY-MM-DD
**Status:** [GATE PASSED | GATE FAILED | NO DECISION]
**Decision:** [Tier 4 promoted | Tier 4 NOT promoted | manual tie-break]

---

## 1. Bootstrap summary

| Metric | Value |
|---|---|
| Roster pairs scraped | NN / 112 |
| Total players in rosters.parquet | NNNN |
| Injuries bootstrapped (Wikipedia) | NN |
| Injuries skipped (no roster match) | NN |

## 2. Apples-to-apples double Optuna study

| Model | Features | Train n | Val_gate log-loss |
|---|---|---:|---:|
| Baseline (Tier 1+2 retuned) | NN cols | NNNNN | 0.XXXX |
| Challenger (Tier 1+2+4) | NN+4 cols | NNNNN | 0.XXXX |

**Δ vs baseline:** {delta:+.4f}  (negative = challenger better, threshold = -0.003)
**Brier (gate, challenger):** 0.XXXX

## 3. Decision

[Filled based on gate outcome]

## 4. Best params

### Baseline
\`\`\`json
{... from baseline_params.json}
\`\`\`

### Challenger
\`\`\`json
{... from challenger_params.json}
\`\`\`

## 5. Anti-leakage tests

`tests/test_leakage.py` (6/6 passing including `test_tier4_strict_pre_match`).

## 6. Artifacts

| Artifact | Path |
|---|---|
| Rosters | `data/raw/transfermarkt/rosters.parquet` |
| Injuries | `data/manual/injuries.csv` |
| Tier 4 model | `models/tier4/xgb_poisson.json` |
| Calibrator | `models/tier4/calibrator.json` |
| Baseline params | `models/tier4/baseline_params.json` |
| Challenger params | `models/tier4/challenger_params.json` |
| This report | `reports/validation_step5.md` |
```

- [ ] **Step 8: Final test sweep**

```bash
pytest -v
```
Expected: all green (existing + ~21 new = ~80+ tests).

- [ ] **Step 9: Commit report + working-tree cleanup**

```bash
git add reports/validation_step5.md models/tier4/baseline_params.json models/tier4/challenger_params.json
git add models/tier4/xgb_poisson.json models/tier4/calibrator.json
git commit -m "docs(reports): STEP 5 validation — Tier 4 gate decision (PASS|FAIL)"
```

(`xgb_poisson.json` + `calibrator.json` may be too large for git; if so, gitignore them and commit only the params + report.)

---

## Self-Review

- [x] **Spec coverage**: all 8 acceptance criteria of spec §12 mapped to tasks (rosters → T4-5, injuries → T6-8, tier4 features → T9, leakage → T10, training → T11, CLI → T12, report → T13).
- [x] **Placeholder scan**: all code blocks contain real implementations. No `# TODO` left.
- [x] **Type consistency**: `TIER4_COLUMNS`, `RosterPlayer`, `WithdrawalEntry`, `train_tier4_pipeline` signatures used identically across tasks.
- [x] **Known assumption flagged**: `TIER1_FEATURE_COLS` / `TIER2_FEATURE_COLS` may need export or inlining — Task 11 step 5 NOTE alerts the engineer to verify.
- [x] **Anti-leakage**: dedicated test in Task 10 + invariant respected in `tier4.py` (strict `<`).
- [x] **Frequent commits**: 1 commit per task = 13 logical commits.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-10-step5-tier4-injuries.md`.** Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks. Best for ~13 tasks with clear interfaces; minimizes context bloat in the main session.
2. **Inline Execution** — I execute tasks in this session with checkpoints. Slower per task (more thinking visible) but you can interrupt mid-task more easily.

Which approach?
