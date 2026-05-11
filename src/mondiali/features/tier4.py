"""Feature builder Tier 4: injury impact on top-5 by market value.

For each (match, side):
1. Find tournament whose ``[start, end+grace]`` window contains ``match.date``.
   If no roster row matches → all 4 features NaN for that side.
2. Top-5 = 5 players with highest market_value in that nation/tournament roster.
3. Filter injuries: ``team == nation``, ``tournament == same``,
   ``date_of_knowledge < match.date``, ``status in {out, doubtful}``.
4. Intersect by ``player_url_slug``. Compute count + value_ratio.

Anti-leakage:
- ``date_of_knowledge < match.date`` (strict, no <=).
- Pre-2018 → NaN (no rosters scraped pre-WC2018).
- Match outside any tournament window → NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from mondiali.data.tm_rosters import TOURNAMENT_META

log = structlog.get_logger(__name__)

TIER4_TOP_N = 5
TIER4_MIN_YEAR = 2018
TIER4_TOURNAMENT_GRACE_DAYS = 30

TIER4_COLUMNS: list[str] = [
    "home_top5_absent_count",
    "away_top5_absent_count",
    "home_value_absent_ratio",
    "away_value_absent_ratio",
]


def _build_top5_lookup(rosters: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    top5_by_pair: dict[tuple[str, str], pd.DataFrame] = {}
    for (nation, tournament), grp in rosters.groupby(["nation", "tournament"]):
        valid = grp.dropna(subset=["market_value_eur"])
        if valid.empty:
            continue
        top5_by_pair[(nation, tournament)] = valid.sort_values(
            ["market_value_eur", "player_url_slug"], ascending=[False, True]
        ).head(TIER4_TOP_N)
    return top5_by_pair


def _build_injury_lookup(
    injuries: pd.DataFrame,
) -> dict[tuple[str, str], pd.DataFrame]:
    if injuries.empty:
        return {}
    inj = injuries.copy()
    inj["date_of_knowledge"] = pd.to_datetime(inj["date_of_knowledge"])
    return {(team, t): grp for (team, t), grp in inj.groupby(["team", "tournament"])}


def _find_tournament(
    nation: str,
    match_date: pd.Timestamp,
    top5_by_pair: dict[tuple[str, str], pd.DataFrame],
    end_dates: dict[str, pd.Timestamp],
) -> str | None:
    for t, meta in TOURNAMENT_META.items():
        start = pd.Timestamp(meta["start"])
        if start <= match_date <= end_dates[t] and (nation, t) in top5_by_pair:
            return t
    return None


def _compute_absence(
    top5: pd.DataFrame,
    inj_grp: pd.DataFrame | None,
    match_date: pd.Timestamp,
) -> tuple[int, float]:
    if inj_grp is None:
        return 0, 0.0
    absent_inj = inj_grp[
        (inj_grp["date_of_knowledge"] < match_date)
        & (inj_grp["status"].isin(["out", "doubtful"]))
    ]
    absent_slugs = set(absent_inj["player_url_slug"]) & set(top5["player_url_slug"])
    if not absent_slugs:
        return 0, 0.0
    abs_rows = top5[top5["player_url_slug"].isin(absent_slugs)]
    return int(len(abs_rows)), float(abs_rows["market_value_eur"].sum())


def add_tier4_features(
    matches: pd.DataFrame, rosters: pd.DataFrame, injuries: pd.DataFrame
) -> pd.DataFrame:
    out = matches.copy()
    for col in TIER4_COLUMNS:
        out[col] = np.nan

    if rosters.empty or out.empty:
        return out

    rosters = rosters.copy()
    rosters["tournament_start_date"] = pd.to_datetime(rosters["tournament_start_date"])

    end_dates: dict[str, pd.Timestamp] = {
        k: pd.Timestamp(v["end"]) + pd.Timedelta(days=TIER4_TOURNAMENT_GRACE_DAYS)
        for k, v in TOURNAMENT_META.items()
    }
    top5_by_pair = _build_top5_lookup(rosters)
    inj_by_pair = _build_injury_lookup(injuries)

    out["date"] = pd.to_datetime(out["date"])
    min_date = pd.Timestamp(f"{TIER4_MIN_YEAR}-01-01")

    for side, team_col in (("home", "home_team"), ("away", "away_team")):
        count_col = f"{side}_top5_absent_count"
        ratio_col = f"{side}_value_absent_ratio"
        for idx, row in out.iterrows():
            if row["date"] < min_date:
                continue
            nation = row[team_col]
            tournament = _find_tournament(nation, row["date"], top5_by_pair, end_dates)
            if tournament is None:
                continue
            top5 = top5_by_pair[(nation, tournament)]
            top5_total = float(top5["market_value_eur"].sum())
            absent_count, absent_value = _compute_absence(
                top5, inj_by_pair.get((nation, tournament)), row["date"]
            )
            out.at[idx, count_col] = absent_count
            out.at[idx, ratio_col] = (
                (absent_value / top5_total) if top5_total > 0 else np.nan
            )

    log.info(
        "tier4_features_added",
        rows=len(out),
        coverage_home=float(out["home_top5_absent_count"].notna().mean()),
        coverage_away=float(out["away_top5_absent_count"].notna().mean()),
    )
    return out
