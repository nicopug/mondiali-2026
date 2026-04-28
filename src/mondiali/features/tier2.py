"""Feature builder Tier 2: rolling form features (N=5).

Per ogni team in ogni match, considera gli ULTIMI N match di quel team
strettamente anteriori a match_date (qualsiasi tipo di competizione,
qualsiasi ruolo home/away).

Anti-leakage: pandas rolling con closed='left' garantisce strict-anteriority.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

TIER2_COLUMNS: list[str] = [
    "home_form_5", "away_form_5",
    "home_gd_5", "away_gd_5",
    "home_goals_scored_5", "away_goals_scored_5",
    "home_goals_conceded_5", "away_goals_conceded_5",
    "home_avg_opp_elo_5", "away_avg_opp_elo_5",
]


def _team_long_form(matches: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame({
        "team": matches["home_team"].to_numpy(),
        "date": matches["date"].to_numpy(),
        "match_idx": np.arange(len(matches)),
        "role": "home",
        "gf": matches["home_score"].to_numpy(dtype=float),
        "ga": matches["away_score"].to_numpy(dtype=float),
        "opp_elo": matches["away_elo_before"].to_numpy(dtype=float),
    })
    away = pd.DataFrame({
        "team": matches["away_team"].to_numpy(),
        "date": matches["date"].to_numpy(),
        "match_idx": np.arange(len(matches)),
        "role": "away",
        "gf": matches["away_score"].to_numpy(dtype=float),
        "ga": matches["home_score"].to_numpy(dtype=float),
        "opp_elo": matches["home_elo_before"].to_numpy(dtype=float),
    })
    long = pd.concat([home, away], ignore_index=True)
    long["points"] = np.where(
        long["gf"] > long["ga"], 3.0,
        np.where(long["gf"] == long["ga"], 1.0, 0.0),
    )
    long["gd"] = long["gf"] - long["ga"]
    return long.sort_values(["team", "date"], kind="mergesort").reset_index(drop=True)


def _rolling(grouped: pd.api.typing.DataFrameGroupBy, col: str, n: int, agg: str) -> pd.Series:
    rolled = grouped[col].rolling(window=n, min_periods=1, closed="left")
    series = rolled.sum() if agg == "sum" else rolled.mean()
    return series.reset_index(level=0, drop=True)


def add_tier2_features(matches: pd.DataFrame, *, n: int = 5) -> pd.DataFrame:
    long = _team_long_form(matches)
    grouped = long.groupby("team", sort=False)
    long["form_n"] = _rolling(grouped, "points", n, "sum")
    long["gd_n"] = _rolling(grouped, "gd", n, "sum")
    long["gf_mean_n"] = _rolling(grouped, "gf", n, "mean")
    long["ga_mean_n"] = _rolling(grouped, "ga", n, "mean")
    long["opp_elo_mean_n"] = _rolling(grouped, "opp_elo", n, "mean")

    home_view = long[long["role"] == "home"].set_index("match_idx")
    away_view = long[long["role"] == "away"].set_index("match_idx")
    idx = np.arange(len(matches))

    result = matches.copy()
    result["home_form_5"] = home_view["form_n"].reindex(idx).to_numpy()
    result["home_gd_5"] = home_view["gd_n"].reindex(idx).to_numpy()
    result["home_goals_scored_5"] = home_view["gf_mean_n"].reindex(idx).to_numpy()
    result["home_goals_conceded_5"] = home_view["ga_mean_n"].reindex(idx).to_numpy()
    result["home_avg_opp_elo_5"] = home_view["opp_elo_mean_n"].reindex(idx).to_numpy()

    result["away_form_5"] = away_view["form_n"].reindex(idx).to_numpy()
    result["away_gd_5"] = away_view["gd_n"].reindex(idx).to_numpy()
    result["away_goals_scored_5"] = away_view["gf_mean_n"].reindex(idx).to_numpy()
    result["away_goals_conceded_5"] = away_view["ga_mean_n"].reindex(idx).to_numpy()
    result["away_avg_opp_elo_5"] = away_view["opp_elo_mean_n"].reindex(idx).to_numpy()

    log.info("added tier2 features", rows=len(result), n=n)
    return result
