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
from mondiali.model.calibration import IsotonicCalibrator1X2
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

    model = PoissonXGBModel().load(model_dir / "xgb_poisson.json")
    lam_h, lam_a = model.predict_lambda(row)
    lam_h, lam_a = float(lam_h[0]), float(lam_a[0])

    rho = float((model_dir / "rho.txt").read_text().strip())
    joint = joint_matrix(lam_h, lam_a)
    joint = dixon_coles_correct(joint, lam_h, lam_a, rho)

    p_home_raw, p_draw_raw, p_away_raw = prob_1x2(joint)
    calibrator_path = model_dir / "calibrator.json"
    if calibrator_path.exists():
        calibrator = IsotonicCalibrator1X2().load(calibrator_path)
        probs_calib = calibrator.transform(
            np.array([[p_home_raw, p_draw_raw, p_away_raw]])
        )[0]
        p_home, p_draw, p_away = float(probs_calib[0]), float(probs_calib[1]), float(probs_calib[2])
        calibrated = True
    else:
        p_home, p_draw, p_away = p_home_raw, p_draw_raw, p_away_raw
        calibrated = False

    over15, under15 = prob_over_under(joint, threshold=1.5)
    over25, under25 = prob_over_under(joint, threshold=2.5)
    over35, under35 = prob_over_under(joint, threshold=3.5)
    btts_yes, btts_no = prob_btts(joint)

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

    return {
        "match": {"home": home, "away": away, "date": str(pd.Timestamp(date).date()),
                  "neutral": neutral},
        "model_version": model_version,
        "lambda": {"home": lam_h, "away": lam_a},
        "markets": {
            "1x2": {"home": p_home, "draw": p_draw, "away": p_away,
                    "calibrated": calibrated},
            "over_under_1_5": {"over": float(over15), "under": float(under15),
                               "validated": validated.get("over_under_1_5", False)},
            "over_under_2_5": {"over": float(over25), "under": float(under25),
                               "validated": validated.get("over_under_2_5", False)},
            "over_under_3_5": {"over": float(over35), "under": float(under35),
                               "validated": validated.get("over_under_3_5", False)},
            "btts": {"yes": float(btts_yes), "no": float(btts_no),
                     "validated": validated.get("btts", False)},
        },
    }
