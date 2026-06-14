"""Download e parsing del dataset `martj42/international_results`."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
import structlog

from mondiali.config import CONFIG
from mondiali.features.elo import EloSystem
from mondiali.features.tier1 import add_tier1_features
from mondiali.features.tier2 import add_tier2_features
from mondiali.features.tier3 import TIER3_COLUMNS, add_tier3_features

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


def append_manual_results(df: pd.DataFrame, manual_csv: Path) -> pd.DataFrame:
    """Inietta nel grezzo i risultati inseriti a mano prima del feature engineering.

    Serve per le partite gia' giocate ma non ancora pubblicate dal dataset
    community (martj42), che tipicamente ritarda 1-2 giorni sui risultati di
    giornata. Iniettandole qui entrano in ``matches.parquet`` -> stato Elo ->
    predizioni forward, non solo nello scoring.

    Dedup robusto su ``(date, coppia-squadre NON orientata)``: la fonte martj42 ha
    **precedenza**, quindi appena pubblica una partita (anche con casa/trasferta
    invertite) la riga manuale viene scartata -> supplemento auto-pulente. Alle
    righe senza ``tournament`` viene assegnato ``"FIFA World Cup"`` cosi' l'Elo
    applica il k-factor corretto.
    """
    if not manual_csv.exists():
        return df
    manual = pd.read_csv(manual_csv, dtype={"neutral": "string"})
    if manual.empty:
        return df

    manual["date"] = pd.to_datetime(manual["date"], errors="raise").astype("datetime64[ns]")
    manual = manual.dropna(subset=["home_score", "away_score"]).copy()
    manual["home_score"] = manual["home_score"].astype("int64")
    manual["away_score"] = manual["away_score"].astype("int64")
    manual["neutral"] = (
        manual["neutral"].str.upper().map({"TRUE": True, "FALSE": False}).astype("bool")
    )
    if "tournament" not in manual.columns:
        manual["tournament"] = "FIFA World Cup"
    for col in df.columns:
        if col not in manual.columns:
            manual[col] = pd.NA
    manual = manual[df.columns]

    # Chiave (data, coppia-squadre NON orientata). Scartiamo SOLO le righe manuali
    # gia' presenti in martj42 (orientamento incluso) e gli eventuali duplicati
    # interni del file manuale: il dataframe martj42 non viene mai toccato.
    def _keys(frame: pd.DataFrame) -> list[str]:
        return [
            f"{pd.Timestamp(d).strftime('%Y%m%d')}|" + "|".join(sorted((h, a)))
            for d, h, a in zip(
                frame["date"], frame["home_team"], frame["away_team"], strict=True
            )
        ]

    existing = set(_keys(df))
    manual = manual.assign(_k=_keys(manual))
    manual = manual[~manual["_k"].isin(existing)]
    manual = manual.drop_duplicates(subset="_k", keep="first").drop(columns="_k")
    if manual.empty:
        return df

    combined = (
        pd.concat([df, manual], ignore_index=True)
        .sort_values("date", kind="mergesort")
        .reset_index(drop=True)
    )
    log.info("manual_results_injected", added=len(manual), path=str(manual_csv))
    return combined


def build_processed_matches(
    raw_csv: Path, out_path: Path, manual_csv: Path | None = None
) -> Path:
    """Pipeline: raw CSV → matches.parquet con Elo pre-match per riga.

    - Carica il raw
    - Ordina per data (già fatto da `load_international_results`)
    - Costruisce `EloSystem.build_history`
    - Aggiunge Tier 1 features (competition_importance, days_rest_*)
    - Aggiunge Tier 2 features (form rolling, gd, goals)
    - Aggiunge Tier 3 features (market values, age) se snapshots.parquet presente, altrimenti pd.NA
    - Aggiunge `match_id` stabile (derivato da date+home+away)
    - Scrive `matches.parquet`

    Args:
        raw_csv: path del CSV scaricato.
        out_path: dove scrivere il parquet.
        manual_csv: se fornito ed esistente, inietta i risultati inseriti a mano
            (partite non ancora pubblicate da martj42) via ``append_manual_results``.

    Returns:
        out_path.
    """
    df = load_international_results(raw_csv)
    if manual_csv is not None:
        df = append_manual_results(df, manual_csv)
    elo = EloSystem()
    df = elo.build_history(df)
    df = add_tier1_features(df)
    df = add_tier2_features(df)

    snapshots_path = CONFIG.data_raw / "transfermarkt" / "snapshots.parquet"
    if snapshots_path.exists():
        snapshots = pd.read_parquet(snapshots_path)
        snapshots["snapshot_date"] = pd.to_datetime(snapshots["snapshot_date"])
        df = add_tier3_features(df, snapshots)
        log.info("tier3_features_loaded", path=str(snapshots_path))
    else:
        log.info("tier3_snapshots_missing_filling_nan", expected=str(snapshots_path))
        for col in TIER3_COLUMNS:
            df[col] = pd.NA

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
