"""Feature builder Tier 3: Transfermarkt market values.

Per ogni match (>=2014) e per ciascun lato, cerca in `snapshots.parquet` la
riga con `nation == team` e `snapshot_date < match_date` ordinata desc, prendi
prima. Calcola `tm_age_days = (match_date - snapshot_date).days`.

Anti-leakage:
- snapshot strict-pre-match (`<`, non `<=`)
- Age clipping >540 giorni → NaN (no forward-fill abusi)
- Hard floor >=2 snapshot per nazionale (sotto-floor escluse)
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
    last = pre.iloc[-1]
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
        log.info("tier3_snapshots_empty")
        return out

    lookup = _build_lookup(snapshots)
    log.info("tier3_lookup_built", n_nations_eligible=len(lookup))

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
        "tier3_features_added",
        rows=len(out),
        coverage_home=float(np.mean(~np.isnan(home_total))),
        coverage_away=float(np.mean(~np.isnan(away_total))),
    )
    return out
