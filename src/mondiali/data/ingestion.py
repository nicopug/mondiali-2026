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
    df["date"] = pd.to_datetime(df["date"], errors="raise").astype("datetime64[ns]")
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype("int64")
    df["away_score"] = df["away_score"].astype("int64")
    df["neutral"] = df["neutral"].str.upper().map({"TRUE": True, "FALSE": False}).astype("bool")
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    log.info("loaded international_results", rows=len(df))
    return df
