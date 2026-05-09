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

import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import structlog
from bs4 import BeautifulSoup

from mondiali.data.tm_nations import NATION_TM_IDS

log = structlog.get_logger(__name__)

CDX_ENDPOINT = "https://web.archive.org/cdx/search/cdx"
WAYBACK_FETCH_BASE = "https://web.archive.org/web"
RATE_LIMIT_SECONDS = 2.0
CDX_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class SquadValue:
    """Output del parser TM rosa."""

    total_value_eur: float
    top11_value_eur: float
    n_players: int


# US/EN format: "€80.00m", "€500k", "€1.50m"
_VALUE_RE_US = re.compile(r"€\s*([\d.,]+)\s*([mk]?)", re.IGNORECASE)
# DE format: "20,50 Mill. €", "5,00 Tsd. €", "1,50 Mio. €", "75 Th. €"
_VALUE_RE_DE = re.compile(r"([\d.,]+)\s+(Mio|Mill|Tsd|Th)\.?\s*€", re.IGNORECASE)


_DE_MULTIPLIERS = {"mio": 1_000_000.0, "mill": 1_000_000.0, "tsd": 1_000.0, "th": 1_000.0}
_US_MULTIPLIERS = {"m": 1_000_000.0, "k": 1_000.0}


def _parse_value_eur(raw: str) -> float | None:
    """Parse TM market-value string in EUR (US o DE locale).

    Esempi:
    - US: "€80.00m" → 80_000_000.0
    - DE: "20,50 Mill. €" → 20_500_000.0
    - DE: "75 Th. €" → 75_000.0
    - Sentinel ("€-", "-", ""): None
    """
    s = raw.strip() if raw else ""
    if not s or s in ("€-", "-"):
        return None

    # DE format first (more specific: requires unit word after number)
    m = _VALUE_RE_DE.search(s)
    if m:
        try:
            num = float(m.group(1).replace(".", "").replace(",", "."))
        except ValueError:
            log.warning("tm_parse_value_malformed", raw=s)
            return None
        return num * _DE_MULTIPLIERS[m.group(2).lower()]

    # US format
    m = _VALUE_RE_US.search(s)
    if m:
        try:
            num = float(m.group(1).replace(",", "."))
        except ValueError:
            log.warning("tm_parse_value_malformed", raw=s)
            return None
        return num * _US_MULTIPLIERS.get(m.group(2).lower(), 1.0)
    return None


def _parse_squad_value(html: str) -> SquadValue | None:
    """Parse pagina TM rosa nazionale. None se la pagina non contiene rosa.

    Cerca `table.items` (TM ≥2018) o `table#kader` (legacy), estrae celle
    `td.rechts.hauptlink` (TM moderno) o `td.rechts` come fallback, parsa ogni valore via `_parse_value_eur`,
    scarta None/<=0. Top-11 = somma dei 11 valori più alti.
    """
    soup = BeautifulSoup(html, "html.parser")

    table = soup.select_one("table.items")
    if table is None:
        table = soup.select_one("table#kader")
    if table is None:
        return None

    cells = table.select("td.rechts.hauptlink")
    if not cells:
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


@dataclass(frozen=True)
class CDXRow:
    """Una riga della risposta CDX search."""

    urlkey: str
    timestamp: str  # YYYYMMDDHHMMSS
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


_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_BASE = 2.0  # exp backoff between retries: 2s, 4s
_PARSE_ATTEMPTS_PER_LEVEL = 5


def _slug_from_url(url: str) -> str:
    """Estrai lo slug nazionale dall'URL TM.

    Pattern: ``https://www.transfermarkt.com/{slug}/startseite/verein/{id}``.
    Robusto a trailing slash, query params, fragment, e variazioni di subdomain.
    """
    path_parts = [p for p in urlparse(url).path.split("/") if p]
    return path_parts[0] if path_parts else "unknown"


def _wayback_url(row: CDXRow) -> str:
    """URL Wayback per fetch HTML da una CDX row."""
    return f"{WAYBACK_FETCH_BASE}/{row.timestamp}/{row.original}"


@dataclass(frozen=True)
class SnapshotRecord:
    """Una riga di snapshots.parquet (popolata in Task 7)."""

    nation: str
    year: int
    snapshot_date: date
    total_value_eur: float
    top11_value_eur: float
    n_players: int
    source_url: str


def _fetch_snapshot_html(row: CDXRow, cache_dir: Path) -> str | None:
    """Fetch HTML da Wayback con cache + rate limiter + retry exp-backoff.

    Cache key: ``{slug}__{timestamp}.html`` in ``cache_dir``.

    Returns:
        HTML body se 200, None se cache miss + retry exhausted oppure 4xx fatale.
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
                return None
            log.warning("wayback non-200", url=url, status=resp.status_code, attempt=attempt)
        except requests.RequestException as e:
            log.warning("wayback exception", error=str(e), attempt=attempt)
        if attempt < _RETRY_ATTEMPTS - 1:
            time.sleep(_RETRY_BACKOFF_BASE * (2 ** attempt))

    return None


TRANSFERMARKT_URL_TEMPLATE = "https://www.transfermarkt.com/{slug}/startseite/verein/{tm_id}"


def _build_target_url(nation: str) -> str | None:
    """Da team_name (es. 'Italy') costruisci URL TM canonico.

    Returns None se la nazionale non è in `NATION_TM_IDS`.
    """
    entry = NATION_TM_IDS.get(nation)
    if entry is None:
        return None
    slug, tm_id = entry
    return TRANSFERMARKT_URL_TEMPLATE.format(slug=slug, tm_id=tm_id)


_CACHE_FILE_RE = re.compile(r"^(?P<slug>[a-z0-9-]+)__(?P<ts>\d{14})\.html$")


def _scan_cache_for_slug(slug: str, cache_dir: Path) -> list[tuple[str, Path]]:
    """Lista (timestamp, path) di tutti i cache file per slug. Slug match esatto."""
    if not cache_dir.exists():
        return []
    out: list[tuple[str, Path]] = []
    for p in cache_dir.glob(f"{slug}__*.html"):
        m = _CACHE_FILE_RE.match(p.name)
        if m and m.group("slug") == slug:
            out.append((m.group("ts"), p))
    return out


def _ts_to_date(ts: str) -> date | None:
    try:
        return date(int(ts[:4]), int(ts[4:6]), int(ts[6:8]))
    except ValueError:
        return None


def _try_cached_for_year(
    target_url: str, year: int, cache_dir: Path
) -> tuple[date, SquadValue, str] | None:
    """Soddisfa (slug, year) dalla cache locale senza CDX query.

    Replica la fallback chain di `_best_snapshot_for_year` ma usando solo file
    già scaricati. Stesso ordine: ±2 mesi da 1 luglio → tutto l'anno → 2° semestre
    anno-1. Stessa logica "closest-to-target". Se nessun file della cache
    soddisfa, ritorna None (caller fa fallback CDX normale).
    """
    slug = _slug_from_url(target_url)
    entries = _scan_cache_for_slug(slug, cache_dir)
    if not entries:
        return None

    target = date(year, 7, 1)
    levels = [
        (date(year, 5, 1), date(year, 9, 1)),
        (date(year, 1, 1), date(year, 12, 31)),
        (date(year - 1, 7, 1), date(year - 1, 12, 31)),
    ]

    for from_d, to_d in levels:
        candidates: list[tuple[date, str, Path]] = []
        for ts, path in entries:
            d = _ts_to_date(ts)
            if d is None:
                continue
            if from_d <= d <= to_d:
                candidates.append((d, ts, path))
        if not candidates:
            continue
        candidates.sort(key=lambda c: abs((c[0] - target).days))
        for snap_date, ts, path in candidates[:_PARSE_ATTEMPTS_PER_LEVEL]:
            html = path.read_text(encoding="utf-8")
            parsed = _parse_squad_value(html)
            if parsed is None:
                continue
            source = f"{WAYBACK_FETCH_BASE}/{ts}/{target_url}"
            return (snap_date, parsed, source)
    return None


def _best_snapshot_for_year(
    target_url: str, year: int, cache_dir: Path
) -> tuple[date, SquadValue, str] | None:
    """Adaptive fallback per ``(nation_url, year)``.

    Fast path: se la cache locale contiene già snapshot validi per questo
    (slug, year), li usa direttamente saltando CDX (zero network).

    Tre livelli (sia su cache che su CDX):
    1. [year-05-01, year-09-01] (vicino a 1 luglio)
    2. [year-01-01, year-12-31] (tutto l'anno)
    3. [year-1-07-01, year-1-12-31] (anno precedente, secondo semestre)

    Per ogni livello sceglie lo snapshot più vicino al target (1 luglio),
    fetch HTML, parsa rosa. Se parse fallisce, prova la prossima riga
    (massimo `_PARSE_ATTEMPTS_PER_LEVEL` tentativi per livello). Se tutti i livelli falliscono → None.
    """
    cached = _try_cached_for_year(target_url, year, cache_dir)
    if cached is not None:
        return cached

    target = date(year, 7, 1)

    levels = [
        (date(year, 5, 1), date(year, 9, 1)),
        (date(year, 1, 1), date(year, 12, 31)),
        (date(year - 1, 7, 1), date(year - 1, 12, 31)),
    ]

    for from_d, to_d in levels:
        rows = _query_cdx(target_url, from_d, to_d, limit=20)
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: abs((r.snapshot_date - target).days))

        for row in rows_sorted[:_PARSE_ATTEMPTS_PER_LEVEL]:
            html = _fetch_snapshot_html(row, cache_dir)
            if html is None:
                continue
            parsed = _parse_squad_value(html)
            if parsed is None:
                continue
            return (row.snapshot_date, parsed, _wayback_url(row))

    return None


def _records_to_parquet(records: list[SnapshotRecord], output_path: Path) -> int:
    """Serializza records a parquet (helper condiviso scrape_all + build_from_cache)."""
    columns = [
        "nation", "year", "snapshot_date", "total_value_eur",
        "top11_value_eur", "n_players", "source_url",
    ]
    df = pd.DataFrame(
        [
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
        ],
        columns=columns,
    )
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return len(df)


def build_from_cache(
    scope: list[str],
    years: list[int],
    cache_dir: Path,
    output_path: Path,
) -> tuple[int, int]:
    """Costruisce snapshots.parquet usando SOLO HTML già in cache (zero network).

    Per ogni (nation, year) chiama `_try_cached_for_year`. Salta silenziosamente
    le coppie senza match in cache. Pensato per recuperare lavoro fatto da
    `scrape_all` interrotto prima della scrittura del parquet.

    Returns:
        (n_target, n_filled): numero di coppie target e numero di record scritti.
    """
    records: list[SnapshotRecord] = []
    n_target = 0
    n_filled = 0

    for nation in scope:
        url = _build_target_url(nation)
        if url is None:
            log.warning("nation_not_in_lookup", nation=nation)
            continue
        for year in years:
            n_target += 1
            result = _try_cached_for_year(url, year, cache_dir)
            if result is None:
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
        "build_from_cache_complete",
        n_target=n_target, n_filled=n_filled, coverage=f"{coverage:.1%}",
    )
    _records_to_parquet(records, output_path)
    log.info("wrote_snapshots_parquet", path=str(output_path), rows=n_filled)
    return n_target, n_filled


def _load_existing_records(output_path: Path) -> tuple[list[SnapshotRecord], set[str]]:
    """Carica records già presenti in snapshots.parquet (per --resume).

    Returns:
        (records, nations_done): records esistenti + set di nazioni già coperte.
    """
    if not output_path.exists():
        return [], set()
    df = pd.read_parquet(output_path)
    records: list[SnapshotRecord] = []
    for _, row in df.iterrows():
        snap_date = row["snapshot_date"]
        if hasattr(snap_date, "date"):
            snap_date = snap_date.date()
        records.append(SnapshotRecord(
            nation=str(row["nation"]),
            year=int(row["year"]),
            snapshot_date=snap_date,
            total_value_eur=float(row["total_value_eur"]),
            top11_value_eur=float(row["top11_value_eur"]),
            n_players=int(row["n_players"]),
            source_url=str(row["source_url"]),
        ))
    nations_done = set(df["nation"].unique())
    return records, nations_done


def scrape_all(
    scope: list[str],
    years: list[int],
    cache_dir: Path,
    output_path: Path,
    *,
    resume: bool = False,
) -> None:
    """Itera scope × years, raccoglie snapshot, scrive snapshots.parquet.

    Args:
        scope: lista nazionali (chiavi di NATION_TM_IDS)
        years: anni 2014..2025 tipicamente
        cache_dir: directory per HTML cache (creata se mancante)
        output_path: path dove scrivere snapshots.parquet (parent creato se mancante)
        resume: se True e ``output_path`` esiste, salta integralmente le
            nazioni già presenti (gap inclusi → niente CDX retry).
    """
    if resume:
        existing_records, nations_done = _load_existing_records(output_path)
        records: list[SnapshotRecord] = list(existing_records)
        log.info("resume_mode_active", existing_records=len(existing_records),
                 nations_skipped=len(nations_done))
    else:
        existing_records = []
        records = []
        nations_done = set()
    n_target = 0
    n_filled_new = 0

    for nation in scope:
        if nation in nations_done:
            continue
        url = _build_target_url(nation)
        if url is None:
            log.warning("nation_not_in_lookup", nation=nation)
            continue
        for year in years:
            n_target += 1
            log.info("scraping", nation=nation, year=year)
            result = _best_snapshot_for_year(url, year, cache_dir)
            if result is None:
                log.warning("no_snapshot_found", nation=nation, year=year)
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
            n_filled_new += 1

    coverage = n_filled_new / n_target if n_target else 0.0
    log.info(
        "scrape_complete",
        n_target=n_target, n_filled_new=n_filled_new, coverage=f"{coverage:.1%}",
        existing=len(existing_records),
    )
    if n_target > 0 and coverage < 0.6:
        log.warning("coverage_below_60pct_new", coverage=coverage)

    rows_written = _records_to_parquet(records, output_path)
    log.info("wrote_snapshots_parquet", path=str(output_path), rows=rows_written)
