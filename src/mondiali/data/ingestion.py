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
