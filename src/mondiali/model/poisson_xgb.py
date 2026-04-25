"""XGBoost Poisson symmetric single-model per predizione gol.

Ogni match produce 2 righe: una home-perspective (team=home, opp=away,
is_home=1 se non neutral altrimenti 0) e una away-perspective (simmetrica).
Target: gol segnati dal team in quel match.

Conforme a spec §6.1 (symmetric single-model).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SYMMETRIC_FEATURES: list[str] = [
    "team_elo",
    "opponent_elo",
    "elo_diff_signed",
    "is_home",
    "is_neutral",
    "competition_importance",
    "team_days_rest",
    "opponent_days_rest",
]


def build_symmetric_rows(matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Ritorna (X, y) dove per ogni match crea 2 righe consecutive.

    Ordine righe: [match0_home, match0_away, match1_home, match1_away, ...].

    X shape: (2 * len(matches), len(SYMMETRIC_FEATURES))
    y shape: (2 * len(matches),)
    """
    n = len(matches)
    X = np.empty((2 * n, len(SYMMETRIC_FEATURES)), dtype=float)  # noqa: N806
    y = np.empty(2 * n, dtype=float)

    home_elo = matches["home_elo_before"].to_numpy(dtype=float)
    away_elo = matches["away_elo_before"].to_numpy(dtype=float)
    neutral = matches["neutral"].astype(bool).to_numpy()
    comp_imp = matches["competition_importance"].to_numpy(dtype=float)
    rest_h = matches["days_rest_home"].to_numpy(dtype=float)
    rest_a = matches["days_rest_away"].to_numpy(dtype=float)
    h_goals = matches["home_score"].to_numpy(dtype=float)
    a_goals = matches["away_score"].to_numpy(dtype=float)

    # Home-perspective rows (indici pari 0, 2, 4, ...)
    X[0::2, 0] = home_elo                               # team_elo
    X[0::2, 1] = away_elo                               # opponent_elo
    X[0::2, 2] = home_elo - away_elo                    # elo_diff_signed
    X[0::2, 3] = (~neutral).astype(float)               # is_home (0 se neutral)
    X[0::2, 4] = neutral.astype(float)                  # is_neutral
    X[0::2, 5] = comp_imp                               # competition_importance
    X[0::2, 6] = rest_h                                 # team_days_rest
    X[0::2, 7] = rest_a                                 # opponent_days_rest
    y[0::2] = h_goals

    # Away-perspective rows (indici dispari 1, 3, 5, ...)
    X[1::2, 0] = away_elo
    X[1::2, 1] = home_elo
    X[1::2, 2] = away_elo - home_elo
    X[1::2, 3] = 0.0  # away in venue non-neutral: is_home=0 già corretto
    X[1::2, 4] = neutral.astype(float)
    X[1::2, 5] = comp_imp
    X[1::2, 6] = rest_a
    X[1::2, 7] = rest_h
    y[1::2] = a_goals

    return X, y
