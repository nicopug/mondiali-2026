"""Test per il baseline Elo-only (helper Elo pre-torneo)."""
from __future__ import annotations

import pandas as pd

from mondiali.evaluation.elo_baseline import apply_frozen_elo, pretournament_elo_map


def _m(date, h, a, he, ae):
    return {
        "date": pd.Timestamp(date), "home_team": h, "away_team": a,
        "home_elo_before": he, "away_elo_before": ae,
    }


def test_pretournament_elo_takes_first_match_per_team():
    matches = pd.DataFrame([
        # pre-cutoff: ignorate
        _m("2026-06-01", "A", "B", 1000, 2000),
        # giornata 1 (cutoff): Elo pre-torneo da qui
        _m("2026-06-11", "A", "B", 1500, 1600),
        # giornata 2: Elo gia' aggiornato, NON deve sovrascrivere
        _m("2026-06-15", "A", "C", 1530, 1700),
    ])
    elo = pretournament_elo_map(matches, pd.Timestamp("2026-06-11"))
    assert elo["A"] == 1500.0  # dalla prima partita post-cutoff, non dalla 2a
    assert elo["B"] == 1600.0
    assert elo["C"] == 1700.0  # prima apparizione di C e' in g2


def test_apply_frozen_elo_overwrites_with_map():
    test = pd.DataFrame([_m("2026-06-15", "A", "C", 1530, 1700)])
    frozen = apply_frozen_elo(test, {"A": 1500.0, "C": 1690.0})
    assert frozen.iloc[0]["home_elo_before"] == 1500.0
    assert frozen.iloc[0]["away_elo_before"] == 1690.0


def test_apply_frozen_elo_fallback_keeps_original():
    test = pd.DataFrame([_m("2026-06-15", "A", "Z", 1530, 1700)])
    frozen = apply_frozen_elo(test, {"A": 1500.0})  # Z assente
    assert frozen.iloc[0]["home_elo_before"] == 1500.0
    assert frozen.iloc[0]["away_elo_before"] == 1700.0  # fallback all'originale
