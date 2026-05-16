"""Runtime inference: single-match prediction.

Produces a 1-row matches DataFrame with the 24 SYMMETRIC_FEATURES inputs,
strictly enforcing ``match_date < target_date`` anti-leakage.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mondiali.features.tier3 import add_tier3_features
from mondiali.inference.state import FORM_WINDOW
from mondiali.model.calibration import BinaryMarketCalibrator, IsotonicCalibrator1X2
from mondiali.model.dixon_coles import dixon_coles_correct, joint_matrix
from mondiali.model.markets import prob_1x2, prob_btts, prob_over_under
from mondiali.model.poisson_xgb import PoissonXGBModel


def _form_aggregate(form: pd.DataFrame, nation: str, target_date: pd.Timestamp) -> dict:
    """Compute (form_5, gd_5, goals_scored_5, goals_conceded_5, avg_opp_elo_5)
    from the last ``FORM_WINDOW`` matches strictly before ``target_date``."""
    rows = form[
        (form["nation"] == nation) & (form["match_date"] < target_date)
    ].sort_values("match_date", ascending=False).head(FORM_WINDOW)
    if rows.empty:
        return {
            "form_5": np.nan, "gd_5": np.nan,
            "goals_scored_5": np.nan, "goals_conceded_5": np.nan,
            "avg_opp_elo_5": np.nan,
            "last_match_date": pd.NaT,
        }
    gf = rows["score_for"].astype(float)
    ga = rows["score_against"].astype(float)
    points = np.where(gf > ga, 3.0, np.where(gf == ga, 1.0, 0.0))
    return {
        "form_5": float(points.sum()),
        "gd_5": float((gf - ga).sum()),
        "goals_scored_5": float(gf.mean()),
        "goals_conceded_5": float(ga.mean()),
        "avg_opp_elo_5": float(rows["opponent_elo"].astype(float).mean()),
        "last_match_date": pd.Timestamp(rows["match_date"].max()),
    }


def _elo_lookup(elo_state: pd.DataFrame, nation: str) -> float:
    matches = elo_state[elo_state["nation"] == nation]
    if matches.empty:
        return 1500.0
    return float(matches["elo"].iloc[0])


def build_inference_row(
    *,
    home: str,
    away: str,
    date: pd.Timestamp,
    neutral: bool,
    elo_state: pd.DataFrame,
    form_cache: pd.DataFrame,
    tm_snapshots: pd.DataFrame | None,
    competition_importance: float = 30.0,
) -> pd.DataFrame:
    """Build a 1-row matches DataFrame with the 24 SYMMETRIC_FEATURES inputs.

    Anti-leakage: form aggregates filter ``match_date < date`` strictly; TM
    snapshots use ``merge_asof(direction='backward', allow_exact_matches=False)``.
    """
    date = pd.Timestamp(date)
    home_elo = _elo_lookup(elo_state, home)
    away_elo = _elo_lookup(elo_state, away)
    home_form = _form_aggregate(form_cache, home, date)
    away_form = _form_aggregate(form_cache, away, date)

    days_rest_home = (
        (date - home_form["last_match_date"]).days
        if pd.notna(home_form["last_match_date"]) else np.nan
    )
    days_rest_away = (
        (date - away_form["last_match_date"]).days
        if pd.notna(away_form["last_match_date"]) else np.nan
    )

    row = pd.DataFrame([{
        "date": date,
        "home_team": home,
        "away_team": away,
        "home_score": 0,
        "away_score": 0,
        "neutral": neutral,
        "home_elo_before": home_elo,
        "away_elo_before": away_elo,
        "competition_importance": float(competition_importance),
        "days_rest_home": float(days_rest_home) if pd.notna(days_rest_home) else np.nan,
        "days_rest_away": float(days_rest_away) if pd.notna(days_rest_away) else np.nan,
        "home_form_5": home_form["form_5"],
        "away_form_5": away_form["form_5"],
        "home_gd_5": home_form["gd_5"],
        "away_gd_5": away_form["gd_5"],
        "home_goals_scored_5": home_form["goals_scored_5"],
        "away_goals_scored_5": away_form["goals_scored_5"],
        "home_goals_conceded_5": home_form["goals_conceded_5"],
        "away_goals_conceded_5": away_form["goals_conceded_5"],
        "home_avg_opp_elo_5": home_form["avg_opp_elo_5"],
        "away_avg_opp_elo_5": away_form["avg_opp_elo_5"],
    }])

    if tm_snapshots is not None and not tm_snapshots.empty:
        snaps = tm_snapshots.rename(columns={"total_value_eur": "total_value_eur"})
        row = add_tier3_features(row, snaps)
    else:
        for col in (
            "home_market_value_total", "away_market_value_total",
            "home_market_value_top11", "away_market_value_top11",
            "home_tm_age_days", "away_tm_age_days",
        ):
            row[col] = np.nan
    return row


def predict_match(
    *,
    home: str,
    away: str,
    date: pd.Timestamp,
    neutral: bool,
    state_dir: Path,
    model_dir: Path,
    tm_snapshots_path: Path | None = None,
    competition_importance: float = 30.0,
    explain: bool = False,
) -> dict:
    """Predict 5 markets (1X2 + 3×U/O + BTTS) for a single match.

    Reads state from ``state_dir`` and frozen model artefacts from ``model_dir``:
    ``xgb_poisson.json``, ``calibrator.json``, ``rho.txt``, ``manifest.json``,
    ``markets_validation.json``.
    """
    from mondiali.inference.state import load_state

    elo_state, form_cache = load_state(state_dir)
    tm_snapshots = (
        pd.read_parquet(tm_snapshots_path)
        if tm_snapshots_path is not None and tm_snapshots_path.exists()
        else None
    )
    row = build_inference_row(
        home=home, away=away, date=pd.Timestamp(date), neutral=neutral,
        elo_state=elo_state, form_cache=form_cache, tm_snapshots=tm_snapshots,
        competition_importance=competition_importance,
    )

    xgb_model = PoissonXGBModel().load(model_dir / "xgb_poisson.json")
    lam_h_xgb, lam_a_xgb = xgb_model.predict_lambda(row)
    lam_h_xgb, lam_a_xgb = float(lam_h_xgb[0]), float(lam_a_xgb[0])

    ensemble_path = model_dir / "ensemble.json"
    if ensemble_path.exists():
        ens_cfg = json.loads(ensemble_path.read_text())
        w_xgb = float(ens_cfg["weight_xgb"])
        # Backward-compat: old freezes had "weight_dl" instead of weight_l1/weight_l3
        w_l1 = float(ens_cfg.get("weight_l1", ens_cfg.get("weight_dl", 0.0)))
        w_l3 = float(ens_cfg.get("weight_l3", 0.0))
        rho = float(ens_cfg["rho_ensemble"])
        lam_h = w_xgb * lam_h_xgb
        lam_a = w_xgb * lam_a_xgb
        if w_l1 > 1e-3 and (model_dir / "dl").exists():
            from mondiali.model.dl_poisson import (
                load_dl_model, predict_lambda as l1_predict,
            )
            m1, idx1, st1, _ = load_dl_model(model_dir / "dl")
            lh1, la1 = l1_predict(m1, row, idx1, st1)
            lam_h += w_l1 * float(lh1[0])
            lam_a += w_l1 * float(la1[0])
        if w_l3 > 1e-3 and (model_dir / "l3").exists():
            from mondiali.model.dl_bivariate import (
                load_bivariate, predict_lambda_rho,
            )
            m3, idx3, st3, _ = load_bivariate(model_dir / "l3")
            lh3, la3, _ = predict_lambda_rho(m3, row, idx3, st3)
            lam_h += w_l3 * float(lh3[0])
            lam_a += w_l3 * float(la3[0])
        ensemble_used = True
    else:
        rho = float((model_dir / "rho.txt").read_text().strip())
        lam_h, lam_a = lam_h_xgb, lam_a_xgb
        ensemble_used = False

    joint = joint_matrix(lam_h, lam_a)
    joint = dixon_coles_correct(joint, lam_h, lam_a, rho)

    p_home_raw, p_draw_raw, p_away_raw = prob_1x2(joint)
    calibrator_path = model_dir / "calibrator.json"
    if calibrator_path.exists():
        calibrator = IsotonicCalibrator1X2.load(calibrator_path)
        probs_calib = calibrator.predict(
            np.array([[p_home_raw, p_draw_raw, p_away_raw]])
        )[0]
        p_home, p_draw, p_away = float(probs_calib[0]), float(probs_calib[1]), float(probs_calib[2])
        calibrated = True
    else:
        p_home, p_draw, p_away = p_home_raw, p_draw_raw, p_away_raw
        calibrated = False

    over15, _ = prob_over_under(joint, threshold=1.5)
    over25, _ = prob_over_under(joint, threshold=2.5)
    over35, _ = prob_over_under(joint, threshold=3.5)
    btts_yes, _ = prob_btts(joint)

    markets_calib_dir = model_dir / "markets_calibrators"
    market_p = {
        "over_under_1_5": over15,
        "over_under_2_5": over25,
        "over_under_3_5": over35,
        "btts": btts_yes,
    }
    market_calibrated: dict[str, bool] = {}
    for market, raw_p in list(market_p.items()):
        calib_path = markets_calib_dir / f"{market}.json"
        if calib_path.exists():
            cal = BinaryMarketCalibrator.load(calib_path)
            market_p[market] = float(cal.predict(np.array([raw_p]))[0])
            market_calibrated[market] = True
        else:
            market_calibrated[market] = False
    over15, over25, over35, btts_yes = (
        market_p["over_under_1_5"], market_p["over_under_2_5"],
        market_p["over_under_3_5"], market_p["btts"],
    )
    under15, under25, under35 = 1 - over15, 1 - over25, 1 - over35
    btts_no = 1 - btts_yes

    validation_path = model_dir / "markets_validation.json"
    validated = {}
    if validation_path.exists():
        v = json.loads(validation_path.read_text())
        for key in ("over_under_1_5", "over_under_2_5", "over_under_3_5", "btts"):
            validated[key] = bool(v.get(key, {}).get("validated", False))

    manifest_path = model_dir / "manifest.json"
    model_version = (
        json.loads(manifest_path.read_text()).get("version", "unknown")
        if manifest_path.exists() else "unknown"
    )

    out_dict: dict = {
        "match": {"home": home, "away": away, "date": str(pd.Timestamp(date).date()),
                  "neutral": neutral},
        "model_version": model_version,
        "ensemble": ensemble_used,
        "lambda": {"home": float(lam_h), "away": float(lam_a)},
        "markets": {
            "1x2": {"home": p_home, "draw": p_draw, "away": p_away,
                    "calibrated": calibrated},
            "over_under_1_5": {"over": float(over15), "under": float(under15),
                               "calibrated": market_calibrated["over_under_1_5"],
                               "validated": validated.get("over_under_1_5", False)},
            "over_under_2_5": {"over": float(over25), "under": float(under25),
                               "calibrated": market_calibrated["over_under_2_5"],
                               "validated": validated.get("over_under_2_5", False)},
            "over_under_3_5": {"over": float(over35), "under": float(under35),
                               "calibrated": market_calibrated["over_under_3_5"],
                               "validated": validated.get("over_under_3_5", False)},
            "btts": {"yes": float(btts_yes), "no": float(btts_no),
                     "calibrated": market_calibrated["btts"],
                     "validated": validated.get("btts", False)},
        },
    }
    if explain and xgb_model.booster_ is not None:
        from mondiali.inference.explain import explain_prediction
        out_dict["explanation"] = explain_prediction(row, xgb_model.booster_, top_k=3)
    return out_dict


class BatchPredictor:
    """Cached model + state loader for high-throughput batch prediction.

    Loads XGB + ensemble DL artefacts ONCE and reuses for many matches.
    Use for Monte Carlo simulations and large fixture batches where the
    per-call model-load cost of ``predict_match`` is prohibitive.
    """

    def __init__(
        self, model_dir: Path, state_dir: Path,
        tm_snapshots_path: Path | None = None,
    ) -> None:
        from mondiali.inference.state import load_state
        self.model_dir = Path(model_dir)
        self.state_dir = Path(state_dir)
        self.elo_state, self.form_cache = load_state(state_dir)
        self.tm_snapshots = (
            pd.read_parquet(tm_snapshots_path)
            if tm_snapshots_path is not None and tm_snapshots_path.exists()
            else None
        )
        self.xgb_model = PoissonXGBModel().load(self.model_dir / "xgb_poisson.json")
        self.rho_xgb = float((self.model_dir / "rho.txt").read_text().strip())
        ens_path = self.model_dir / "ensemble.json"
        if ens_path.exists():
            ens = json.loads(ens_path.read_text())
            self.w_xgb = float(ens["weight_xgb"])
            self.w_l1 = float(ens.get("weight_l1", ens.get("weight_dl", 0.0)))
            self.w_l3 = float(ens.get("weight_l3", 0.0))
            self.rho_active = float(ens["rho_ensemble"])
            if self.w_l1 > 1e-3 and (self.model_dir / "dl").exists():
                from mondiali.model.dl_poisson import load_dl_model
                self.l1_model, self.l1_idx, self.l1_stats, _ = load_dl_model(
                    self.model_dir / "dl"
                )
            else:
                self.l1_model = None
            if self.w_l3 > 1e-3 and (self.model_dir / "l3").exists():
                from mondiali.model.dl_bivariate import load_bivariate
                self.l3_model, self.l3_idx, self.l3_stats, _ = load_bivariate(
                    self.model_dir / "l3"
                )
            else:
                self.l3_model = None
        else:
            self.w_xgb, self.w_l1, self.w_l3 = 1.0, 0.0, 0.0
            self.rho_active = self.rho_xgb
            self.l1_model = None
            self.l3_model = None

    def predict_lambdas(
        self, matches_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Vectorized lambda prediction for many matches.

        ``matches_df`` rows must contain columns produced by build_inference_row.
        Returns (lam_home, lam_away, rho_active).
        """
        lam_h_xgb, lam_a_xgb = self.xgb_model.predict_lambda(matches_df)
        lam_h = self.w_xgb * lam_h_xgb
        lam_a = self.w_xgb * lam_a_xgb
        if self.l1_model is not None:
            from mondiali.model.dl_poisson import predict_lambda as l1_predict
            lh1, la1 = l1_predict(self.l1_model, matches_df, self.l1_idx, self.l1_stats)
            lam_h = lam_h + self.w_l1 * lh1
            lam_a = lam_a + self.w_l1 * la1
        if self.l3_model is not None:
            from mondiali.model.dl_bivariate import predict_lambda_rho
            lh3, la3, _ = predict_lambda_rho(
                self.l3_model, matches_df, self.l3_idx, self.l3_stats,
            )
            lam_h = lam_h + self.w_l3 * lh3
            lam_a = lam_a + self.w_l3 * la3
        return lam_h, lam_a, self.rho_active

    def build_row(
        self, home: str, away: str, date: pd.Timestamp, *,
        neutral: bool = True, competition_importance: float = 75.0,
    ) -> pd.DataFrame:
        """Build the runtime inference row for a single match."""
        return build_inference_row(
            home=home, away=away, date=pd.Timestamp(date), neutral=neutral,
            elo_state=self.elo_state, form_cache=self.form_cache,
            tm_snapshots=self.tm_snapshots,
            competition_importance=competition_importance,
        )

    def predict_pair_cache(
        self, teams: list[str], date: pd.Timestamp, *,
        neutral: bool = True, competition_importance: float = 75.0,
    ) -> dict[tuple[str, str], tuple[float, float]]:
        """Predict (lam_a, lam_b) for ALL ordered pairs (a, b) of distinct teams.

        Use as a fast lookup during knockout MC where each pairing may be
        revisited many times across simulations.
        """
        pairs = [(a, b) for a in teams for b in teams if a != b]
        rows = []
        for a, b in pairs:
            row = self.build_row(
                a, b, date, neutral=neutral,
                competition_importance=competition_importance,
            )
            rows.append(row.iloc[0])
        matches_df = pd.DataFrame(rows).reset_index(drop=True)
        lam_h, lam_a, _ = self.predict_lambdas(matches_df)
        return {
            (a, b): (float(lam_h[i]), float(lam_a[i]))
            for i, (a, b) in enumerate(pairs)
        }
