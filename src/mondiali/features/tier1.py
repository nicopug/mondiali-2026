"""Feature builder Tier 1: competition_importance, days_rest."""
from __future__ import annotations

import pandas as pd
import structlog

from mondiali.features.elo import classify_tournament

log = structlog.get_logger(__name__)

_CATEGORY_TO_IMPORTANCE = {
    "world_cup": 4,
    "continental": 3,
    "qualification": 2,
    "friendly": 1,
    "default": 1,
}


def competition_importance_from_tournament(tournament: str) -> int:
    """Ordinal 1-4: 1=friendly/minor, 2=qualif, 3=continental, 4=World Cup."""
    return _CATEGORY_TO_IMPORTANCE[classify_tournament(tournament)]


def add_days_rest(matches: pd.DataFrame) -> pd.DataFrame:
    """Calcola days_rest_home, days_rest_away, days_rest_diff per ogni match.

    Itera cronologicamente: per ogni team tiene traccia della data dell'ultimo
    match (home o away, non importa). NaN se prima volta che vediamo il team.

    Assume `matches` ordinato per data crescente (stesso invariante di
    EloSystem.build_history).

    Raises:
        ValueError: se `matches` non è ordinato per data crescente.
    """
    dates = matches["date"]
    if not dates.is_monotonic_increasing:
        raise ValueError(
            "matches must be sorted by date ascending before calling add_days_rest"
        )

    last_seen: dict[str, pd.Timestamp] = {}
    rest_home: list[float] = []
    rest_away: list[float] = []

    for row in matches.itertuples(index=False):
        date = row.date
        prev_h = last_seen.get(row.home_team)
        prev_a = last_seen.get(row.away_team)
        rest_home.append(float("nan") if prev_h is None else (date - prev_h).days)
        rest_away.append(float("nan") if prev_a is None else (date - prev_a).days)
        last_seen[row.home_team] = date
        last_seen[row.away_team] = date

    result = matches.copy()
    result["days_rest_home"] = rest_home
    result["days_rest_away"] = rest_away
    result["days_rest_diff"] = result["days_rest_home"] - result["days_rest_away"]
    return result


def add_tier1_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge competition_importance + days_rest_{home, away, diff}."""
    result = add_days_rest(matches)
    result["competition_importance"] = result["tournament"].map(
        competition_importance_from_tournament
    )
    log.info("added tier1 features", rows=len(result))
    return result
