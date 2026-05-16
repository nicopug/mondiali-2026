"""State persistence for runtime inference.

Two parquet files in ``data/state/``:

- ``elo_state.parquet``: current Elo per nation (after the latest match they played).
- ``form_cache.parquet``: last ``FORM_WINDOW`` matches per nation, with the raw
  fields needed to recompute rolling form-5 / gd-5 / scored-5 / conceded-5 /
  avg_opp_elo_5 at inference time, strictly filtered by ``match_date < target_date``.

Schema invariants:
- ``elo_state`` has exactly one row per nation.
- ``form_cache`` has at most ``FORM_WINDOW`` rows per nation.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from mondiali.features.elo import EloSystem

ELO_STATE_COLS: list[str] = ["nation", "elo", "last_match_date"]
FORM_CACHE_COLS: list[str] = [
    "nation",
    "match_date",
    "is_home",
    "is_neutral",
    "score_for",
    "score_against",
    "opponent_elo",
    "competition_importance",
]
FORM_WINDOW = 5


def save_state(matches: pd.DataFrame, out_dir: Path) -> None:
    """Rebuild Elo + form caches from full history and write parquet files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    elo = _build_elo_state(matches)
    form = _build_form_cache(matches)
    elo.to_parquet(out_dir / "elo_state.parquet", index=False)
    form.to_parquet(out_dir / "form_cache.parquet", index=False)


def load_state(state_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    elo = pd.read_parquet(state_dir / "elo_state.parquet")
    form = pd.read_parquet(state_dir / "form_cache.parquet")
    return elo, form


def _build_elo_state(matches: pd.DataFrame) -> pd.DataFrame:
    sorted_m = matches.sort_values("date").reset_index(drop=True)
    system = EloSystem()
    system.build_history(sorted_m)
    last_dates: dict[str, pd.Timestamp] = {}
    for row in sorted_m.itertuples(index=False):
        d = pd.Timestamp(row.date)
        last_dates[row.home_team] = d
        last_dates[row.away_team] = d
    rows = [
        {"nation": nation, "elo": float(rating), "last_match_date": last_dates[nation]}
        for nation, rating in system.ratings.items()
        if nation in last_dates
    ]
    return pd.DataFrame(rows, columns=ELO_STATE_COLS)


def _build_form_cache(matches: pd.DataFrame) -> pd.DataFrame:
    sorted_m = matches.sort_values("date").reset_index(drop=True)
    home = pd.DataFrame({
        "nation": sorted_m["home_team"],
        "match_date": pd.to_datetime(sorted_m["date"]),
        "is_home": (~sorted_m["neutral"].astype(bool)),
        "is_neutral": sorted_m["neutral"].astype(bool),
        "score_for": sorted_m["home_score"].astype(int),
        "score_against": sorted_m["away_score"].astype(int),
        "opponent_elo": sorted_m["away_elo_before"].astype(float),
        "competition_importance": sorted_m["competition_importance"].astype(float),
    })
    away = pd.DataFrame({
        "nation": sorted_m["away_team"],
        "match_date": pd.to_datetime(sorted_m["date"]),
        "is_home": False,
        "is_neutral": sorted_m["neutral"].astype(bool),
        "score_for": sorted_m["away_score"].astype(int),
        "score_against": sorted_m["home_score"].astype(int),
        "opponent_elo": sorted_m["home_elo_before"].astype(float),
        "competition_importance": sorted_m["competition_importance"].astype(float),
    })
    stacked = pd.concat([home, away], ignore_index=True)
    stacked = stacked.sort_values(["nation", "match_date"], ascending=[True, False])
    return stacked.groupby("nation", as_index=False).head(FORM_WINDOW).reset_index(drop=True)
