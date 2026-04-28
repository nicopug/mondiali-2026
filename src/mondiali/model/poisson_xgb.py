"""XGBoost Poisson symmetric single-model per predizione gol.

Ogni match produce 2 righe: una home-perspective (team=home, opp=away,
is_home=1 se non neutral altrimenti 0) e una away-perspective (simmetrica).
Target: gol segnati dal team in quel match.

Conforme a spec §6.1 (symmetric single-model).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
import xgboost as xgb

from mondiali.config import RANDOM_STATE

log = structlog.get_logger(__name__)

SYMMETRIC_FEATURES: list[str] = [
    "team_elo",
    "opponent_elo",
    "elo_diff_signed",
    "is_home",
    "is_neutral",
    "competition_importance",
    "team_days_rest",
    "opponent_days_rest",
    "team_form_5",
    "opponent_form_5",
    "team_gd_5",
    "opponent_gd_5",
    "team_goals_scored_5",
    "opponent_goals_scored_5",
    "team_goals_conceded_5",
    "opponent_goals_conceded_5",
    "team_avg_opp_elo_5",
    "opponent_avg_opp_elo_5",
]

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "count:poisson",
    "tree_method": "hist",
    "max_depth": 6,
    "learning_rate": 0.05,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "n_estimators": 2000,
    "random_state": RANDOM_STATE,
    "verbosity": 0,
}


def build_symmetric_rows(matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:  # noqa: PLR0915
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

    home_form = matches["home_form_5"].to_numpy(dtype=float)
    away_form = matches["away_form_5"].to_numpy(dtype=float)
    home_gd = matches["home_gd_5"].to_numpy(dtype=float)
    away_gd = matches["away_gd_5"].to_numpy(dtype=float)
    home_gs = matches["home_goals_scored_5"].to_numpy(dtype=float)
    away_gs = matches["away_goals_scored_5"].to_numpy(dtype=float)
    home_gc = matches["home_goals_conceded_5"].to_numpy(dtype=float)
    away_gc = matches["away_goals_conceded_5"].to_numpy(dtype=float)
    home_ope = matches["home_avg_opp_elo_5"].to_numpy(dtype=float)
    away_ope = matches["away_avg_opp_elo_5"].to_numpy(dtype=float)

    # Home-perspective rows (indici pari 0, 2, 4, ...)
    X[0::2, 0] = home_elo                               # team_elo
    X[0::2, 1] = away_elo                               # opponent_elo
    X[0::2, 2] = home_elo - away_elo                    # elo_diff_signed
    X[0::2, 3] = (~neutral).astype(float)               # is_home (0 se neutral)
    X[0::2, 4] = neutral.astype(float)                  # is_neutral
    X[0::2, 5] = comp_imp                               # competition_importance
    X[0::2, 6] = rest_h                                 # team_days_rest
    X[0::2, 7] = rest_a                                 # opponent_days_rest
    X[0::2, 8] = home_form                              # team_form_5
    X[0::2, 9] = away_form                              # opponent_form_5
    X[0::2, 10] = home_gd                               # team_gd_5
    X[0::2, 11] = away_gd                               # opponent_gd_5
    X[0::2, 12] = home_gs                               # team_goals_scored_5
    X[0::2, 13] = away_gs                               # opponent_goals_scored_5
    X[0::2, 14] = home_gc                               # team_goals_conceded_5
    X[0::2, 15] = away_gc                               # opponent_goals_conceded_5
    X[0::2, 16] = home_ope                              # team_avg_opp_elo_5
    X[0::2, 17] = away_ope                              # opponent_avg_opp_elo_5
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
    X[1::2, 8] = away_form
    X[1::2, 9] = home_form
    X[1::2, 10] = away_gd
    X[1::2, 11] = home_gd
    X[1::2, 12] = away_gs
    X[1::2, 13] = home_gs
    X[1::2, 14] = away_gc
    X[1::2, 15] = home_gc
    X[1::2, 16] = away_ope
    X[1::2, 17] = home_ope
    y[1::2] = a_goals

    return X, y


class PoissonXGBModel:
    """Wrapper XGBoost symmetric single-model per predizione lambda gol.

    `fit(matches)` costruisce le righe simmetriche e addestra XGBRegressor con
    objective `count:poisson`. `predict_lambda(matches)` ritorna
    (lambda_home, lambda_away) per ogni match.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = {**DEFAULT_PARAMS, **(params or {})}
        self.booster_: xgb.XGBRegressor | None = None

    def fit(
        self,
        matches: pd.DataFrame,
        *,
        early_stopping_val: pd.DataFrame | None = None,
        early_stopping_rounds: int = 50,
    ) -> PoissonXGBModel:
        """Addestra il modello. Se `early_stopping_val` è fornito, early stop."""
        X, y = build_symmetric_rows(matches)  # noqa: N806
        fit_kwargs: dict[str, Any] = {}
        if early_stopping_val is not None:
            X_val, y_val = build_symmetric_rows(early_stopping_val)  # noqa: N806
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False
            params = {**self.params, "early_stopping_rounds": early_stopping_rounds}
        else:
            params = self.params
        self.booster_ = xgb.XGBRegressor(**params)
        self.booster_.fit(X, y, **fit_kwargs)
        log.info("poisson_xgb fit done", n_rows=len(X))
        return self

    def predict_lambda(self, matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Ritorna (lambda_home, lambda_away) per ogni match (shape (n,), (n,))."""
        if self.booster_ is None:
            raise RuntimeError("PoissonXGBModel must be fit() before predict_lambda")
        X, _ = build_symmetric_rows(matches)  # noqa: N806
        preds = self.booster_.predict(X)
        lam_h = preds[0::2]
        lam_a = preds[1::2]
        return lam_h, lam_a

    def save(self, path: Path) -> None:
        """Salva il booster in formato JSON nativo (non pickle)."""
        if self.booster_ is None:
            raise RuntimeError("fit() prima di save()")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster_.save_model(str(path))

    def load(self, path: Path) -> PoissonXGBModel:
        """Carica un booster salvato.

        Nota: il regressor è costruito senza `self.params` perché `load_model`
        ripristina gli iperparametri di training; usare `self.params` farebbe
        sì che `.get_params()` mentisse quando l'istanza è stata creata con
        params diversi da quelli usati al training.
        """
        if not path.exists():
            raise FileNotFoundError(path)
        self.booster_ = xgb.XGBRegressor()
        self.booster_.load_model(str(path))
        return self
