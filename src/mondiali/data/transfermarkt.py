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

from dataclasses import dataclass
from datetime import date

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
