"""Baseline Elo-only sulle 72 partite dei gironi WC2026 (scrupolo benchmark).

Confronta il modello v1_final col baseline Elo-logistico (feature: elo_diff +
neutral) per rispondere alla domanda onesta: *gli XGBoost tier 2/3 aggiungono
valore oltre al puro Elo?* Battere il random uniforme (ln3) e' un'asticella
bassa; battere l'Elo nudo, no.

Due varianti riportate:
  - **ex-ante (frozen)**: per ogni squadra si usa l'Elo *prima* della sua prima
    partita del torneo, costante sulle 3 giornate. Stesso set informativo delle
    predizioni congelate del modello -> confronto apples-to-apples.
  - **live**: si usa l'``home/away_elo_before`` reale di ogni partita (l'Elo
    assorbe le giornate precedenti). Riferimento leggermente piu' forte.

Tutto leak-free: il baseline e' fittato solo su partite *anteriori* al kickoff
(2026-06-11) e l'Elo usato e' sempre quello *precedente* alla partita.
"""
from __future__ import annotations

import pandas as pd


def pretournament_elo_map(matches: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, float]:
    """Mappa squadra -> Elo prima della sua prima partita con ``date >= cutoff``.

    Args:
        matches: deve contenere ``date, home_team, away_team, home_elo_before,
            away_elo_before``.
        cutoff: data di kickoff del torneo (es. ``2026-06-11``).

    Returns:
        Dizionario ``{squadra: elo_pre_torneo}`` (Elo congelato pre-torneo).
    """
    df = matches[matches["date"] >= cutoff].sort_values("date", kind="mergesort")
    elo: dict[str, float] = {}
    for _, r in df.iterrows():
        for team, value in (
            (r["home_team"], r["home_elo_before"]),
            (r["away_team"], r["away_elo_before"]),
        ):
            if team not in elo:
                elo[team] = float(value)
    return elo


def apply_frozen_elo(test_matches: pd.DataFrame, elo_map: dict[str, float]) -> pd.DataFrame:
    """Sovrascrive ``home/away_elo_before`` con l'Elo congelato pre-torneo.

    Le squadre assenti dalla mappa mantengono l'Elo originale (fallback).
    """
    out = test_matches.copy()
    out["home_elo_before"] = [
        elo_map.get(t, e)
        for t, e in zip(out["home_team"], out["home_elo_before"], strict=True)
    ]
    out["away_elo_before"] = [
        elo_map.get(t, e)
        for t, e in zip(out["away_team"], out["away_elo_before"], strict=True)
    ]
    return out
