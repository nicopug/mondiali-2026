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


def add_tier3_features(matches: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    """Aggiungi le 6 colonne TIER3_COLUMNS a `matches`.

    Args:
        matches: DataFrame con `date`, `home_team`, `away_team`.
        snapshots: DataFrame con `nation`, `snapshot_date`, `total_value_eur`,
            `top11_value_eur`.

    Returns:
        Copia di `matches` con 6 colonne aggiuntive.

    Note:
        Implementato via ``pd.merge_asof(direction="backward",
        allow_exact_matches=False)`` per preservare l'invariante strict-pre
        (snapshot_date < match_date) in modo vettorizzato.
    """
    out = matches.copy()
    for col in TIER3_COLUMNS:
        out[col] = np.nan

    if snapshots.empty or out.empty:
        log.info("tier3_input_empty", n_snapshots=len(snapshots), n_matches=len(out))
        return out

    counts = snapshots.groupby("nation").size()
    eligible_nations = counts[counts >= TIER3_MIN_SNAPSHOTS_PER_NATION].index
    snaps = (
        snapshots[snapshots["nation"].isin(eligible_nations)]
        [["nation", "snapshot_date", "total_value_eur", "top11_value_eur"]]
        .sort_values("snapshot_date")
        .reset_index(drop=True)
    )
    # Normalize snapshot_date dtype to match matches['date']: parquet roundtrip può
    # produrre [ms]/[us] mentre matches è tipicamente [ns]; merge_asof esige stessa unit.
    target_dtype = out["date"].dtype
    snaps["snapshot_date"] = pd.to_datetime(snaps["snapshot_date"]).astype(target_dtype)
    log.info("tier3_lookup_built", n_nations_eligible=len(eligible_nations))

    if snaps.empty:
        return out

    min_date = pd.Timestamp(f"{TIER3_MIN_YEAR}-01-01")
    out_sorted_idx = out.sort_values("date").index

    for side, team_col in (("home", "home_team"), ("away", "away_team")):
        left = out.loc[out_sorted_idx, ["date", team_col]].rename(columns={team_col: "nation"})
        merged = pd.merge_asof(
            left,
            snaps,
            left_on="date",
            right_on="snapshot_date",
            by="nation",
            direction="backward",
            allow_exact_matches=False,
        )
        age = (merged["date"] - merged["snapshot_date"]).dt.days
        mask = (merged["date"] >= min_date) & (age <= TIER3_MAX_AGE_DAYS) & age.notna()

        total = merged["total_value_eur"].where(mask)
        top11 = merged["top11_value_eur"].where(mask)
        age_clipped = age.where(mask)

        out.loc[out_sorted_idx, f"{side}_market_value_total"] = total.to_numpy()
        out.loc[out_sorted_idx, f"{side}_market_value_top11"] = top11.to_numpy()
        out.loc[out_sorted_idx, f"{side}_tm_age_days"] = age_clipped.to_numpy()

    coverage_home = (
        float(out["home_market_value_total"].notna().mean()) if len(out) else 0.0
    )
    coverage_away = (
        float(out["away_market_value_total"].notna().mean()) if len(out) else 0.0
    )
    log.info(
        "tier3_features_added",
        rows=len(out),
        coverage_home=coverage_home,
        coverage_away=coverage_away,
    )
    return out
