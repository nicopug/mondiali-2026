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
from dataclasses import dataclass
from datetime import date

import requests
import structlog
from bs4 import BeautifulSoup

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
_VALUE_RE_DE = re.compile(r"([\d.,]+)\s*(Mio|Mill|Tsd|Th)\.?\s*€", re.IGNORECASE)


_DE_MULTIPLIERS = {"mio": 1_000_000.0, "mill": 1_000_000.0, "tsd": 1_000.0, "th": 1_000.0}
_US_MULTIPLIERS = {"m": 1_000_000.0, "k": 1_000.0, "": 1.0}


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
        num = float(m.group(1).replace(".", "").replace(",", "."))
        return num * _DE_MULTIPLIERS[m.group(2).lower()]

    # US format
    m = _VALUE_RE_US.search(s)
    if m:
        num = float(m.group(1).replace(",", "."))
        return num * _US_MULTIPLIERS.get(m.group(2).lower(), 1.0)
    return None


def _parse_squad_value(html: str) -> SquadValue | None:
    """Parse pagina TM rosa nazionale. None se la pagina non contiene rosa.

    Cerca `table.items` (TM ≥2018) o `table#kader` (legacy), estrae celle
    `td.rechts.hauptlink`, parsa ogni valore via `_parse_value_eur`,
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
