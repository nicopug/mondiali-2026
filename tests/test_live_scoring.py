"""Test per lo scoring leak-free delle predizioni ex-ante WC2026."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from mondiali.evaluation.live_scoring import (
    merge_actual_results,
    score_completed_matches,
)


def _pred(team_a, team_b, p_a, p_d, p_b, p_o25, p_btts, group="A", neutral=False):
    return {
        "group": group, "team_a": team_a, "team_b": team_b, "neutral": neutral,
        "lam_a": 1.5, "lam_b": 1.0,
        "p_a_wins": p_a, "p_draw": p_d, "p_b_wins": p_b,
        "p_over_2_5": p_o25, "p_btts": p_btts,
    }


def test_scores_home_win_same_orientation():
    pred = pd.DataFrame([_pred("X", "Y", 0.6, 0.25, 0.15, 0.7, 0.6)])
    actual = pd.DataFrame([{
        "date": "2026-06-11", "home_team": "X", "away_team": "Y",
        "home_score": 2, "away_score": 0,
    }])
    scored, summary = score_completed_matches(pred, actual)

    assert len(scored) == 1
    row = scored.iloc[0]
    # esito = home win -> P = p_a_wins = 0.6
    assert math.isclose(row["p_actual_1x2"], 0.6, rel_tol=1e-9)
    assert math.isclose(row["log_loss_1x2"], -math.log(0.6), rel_tol=1e-9)
    # total = 2 -> under 2.5 (y=0): penalizza p_over alto
    assert math.isclose(row["log_loss_ou25"], -math.log(1 - 0.7), rel_tol=1e-9)
    # 2-0 -> btts no (y=0)
    assert math.isclose(row["log_loss_btts"], -math.log(1 - 0.6), rel_tol=1e-9)
    assert summary["n_matches"] == 1


def test_handles_reversed_orientation():
    # Predizione per (P,Q); la partita reale e' Q in casa contro P fuori.
    pred = pd.DataFrame([_pred("P", "Q", 0.5, 0.3, 0.2, 0.6, 0.5)])
    actual = pd.DataFrame([{
        "date": "2026-06-12", "home_team": "Q", "away_team": "P",
        "home_score": 1, "away_score": 3,
    }])
    scored, summary = score_completed_matches(pred, actual)

    row = scored.iloc[0]
    # away (P) vince -> P(away) deve mappare su p_a_wins di P = 0.5
    assert math.isclose(row["p_actual_1x2"], 0.5, rel_tol=1e-9)
    assert math.isclose(row["log_loss_1x2"], -math.log(0.5), rel_tol=1e-9)


def test_matches_despite_mojibake_team_name():
    # Le predizioni congelate hanno un nome a doppio-encoding ("CuraÃ§ao"),
    # mentre i risultati reali (martj42) hanno l'UTF-8 corretto ("Curaçao").
    # Il matching deve agganciare comunque la partita.
    pred = pd.DataFrame([_pred("Germany", "CuraÃ§ao", 0.85, 0.10, 0.05, 0.8, 0.5)])
    actual = pd.DataFrame([{
        "date": "2026-06-18", "home_team": "Germany", "away_team": "Curaçao",
        "home_score": 7, "away_score": 1,
    }])
    scored, summary = score_completed_matches(pred, actual)
    assert summary["n_matches"] == 1
    row = scored.iloc[0]
    assert row["actual_1x2"] == "H"
    assert math.isclose(row["p_actual_1x2"], 0.85, rel_tol=1e-9)


def test_matches_mojibake_reversed_orientation():
    # Stesso mismatch ma con la squadra accentata in casa nel risultato reale.
    pred = pd.DataFrame([_pred("CuraÃ§ao", "Germany", 0.05, 0.10, 0.85, 0.8, 0.5)])
    actual = pd.DataFrame([{
        "date": "2026-06-18", "home_team": "Curaçao", "away_team": "Germany",
        "home_score": 0, "away_score": 3,
    }])
    scored, summary = score_completed_matches(pred, actual)
    assert summary["n_matches"] == 1
    row = scored.iloc[0]
    # Curaçao (casa) perde -> away win -> P = p_b_wins = 0.85
    assert row["actual_1x2"] == "A"
    assert math.isclose(row["p_actual_1x2"], 0.85, rel_tol=1e-9)


def test_skips_matches_without_prediction():
    pred = pd.DataFrame([_pred("X", "Y", 0.6, 0.25, 0.15, 0.7, 0.6)])
    actual = pd.DataFrame([{
        "date": "2026-06-11", "home_team": "Z", "away_team": "W",
        "home_score": 1, "away_score": 0,
    }])
    scored, summary = score_completed_matches(pred, actual)
    assert len(scored) == 0
    assert summary["n_matches"] == 0


def test_merge_actual_results_appends_supplement():
    primary = pd.DataFrame([{
        "date": "2026-06-11", "home_team": "X", "away_team": "Y",
        "home_score": 2, "away_score": 0, "neutral": True,
    }])
    supplement = pd.DataFrame([{
        "date": "2026-06-13", "home_team": "P", "away_team": "Q",
        "home_score": 1, "away_score": 1, "neutral": True,
    }])
    merged = merge_actual_results(primary, supplement)
    assert len(merged) == 2
    pairs = set(zip(merged["home_team"], merged["away_team"], strict=True))
    assert pairs == {("X", "Y"), ("P", "Q")}


def test_merge_actual_results_primary_wins_on_conflict():
    # Stessa coppia (P,Q): la sorgente primaria (martj42) vince sul manuale.
    primary = pd.DataFrame([{
        "date": "2026-06-13", "home_team": "P", "away_team": "Q",
        "home_score": 1, "away_score": 1, "neutral": True,
    }])
    supplement = pd.DataFrame([{
        "date": "2026-06-13", "home_team": "P", "away_team": "Q",
        "home_score": 9, "away_score": 9, "neutral": True,  # valore errato
    }])
    merged = merge_actual_results(primary, supplement)
    assert len(merged) == 1
    assert int(merged.iloc[0]["home_score"]) == 1
    assert int(merged.iloc[0]["away_score"]) == 1


def test_merge_actual_results_none_supplement_is_noop():
    primary = pd.DataFrame([{
        "date": "2026-06-11", "home_team": "X", "away_team": "Y",
        "home_score": 2, "away_score": 0, "neutral": True,
    }])
    assert len(merge_actual_results(primary, None)) == 1
    assert len(merge_actual_results(primary, pd.DataFrame())) == 1


def test_summary_aggregates_and_compares_baseline():
    preds = pd.DataFrame([
        _pred("X", "Y", 0.6, 0.25, 0.15, 0.7, 0.6),
        _pred("A", "B", 0.34, 0.33, 0.33, 0.5, 0.5, group="B"),
    ])
    actuals = pd.DataFrame([
        {"date": "2026-06-11", "home_team": "X", "away_team": "Y",
         "home_score": 2, "away_score": 0},
        {"date": "2026-06-11", "home_team": "A", "away_team": "B",
         "home_score": 1, "away_score": 1},
    ])
    scored, summary = score_completed_matches(preds, actuals)
    assert summary["n_matches"] == 2
    # media coerente con le righe
    assert math.isclose(
        summary["log_loss_1x2"], scored["log_loss_1x2"].mean(), rel_tol=1e-9
    )
    # baseline uniforme 1X2 = ln(3)
    assert math.isclose(summary["baseline_log_loss_1x2"], math.log(3), rel_tol=1e-9)
    assert "edge_vs_uniform_1x2" in summary
    assert np.isfinite(summary["brier_1x2"])
