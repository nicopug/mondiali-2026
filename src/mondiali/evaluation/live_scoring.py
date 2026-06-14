"""Scoring leak-free delle predizioni ex-ante contro i risultati reali.

Principio anti-leakage: si valutano *esclusivamente* le probabilita' generate
PRIMA del torneo (congelate in ``reports/wc2026_groups_predictions.csv``), mai
ri-predette dopo aver assorbito i risultati nello stato Elo. Le predizioni dei
gironi sono per fixture orientati (team_a vs team_b); i mercati binari
(over/under, BTTS) sono invarianti rispetto all'orientamento.

Convenzione classi 1X2 (coerente con ``training.evaluate``):
    0 = home win, 1 = draw, 2 = away win.
"""
from __future__ import annotations

import math

import pandas as pd

from mondiali.training.evaluate import brier_score_1x2, log_loss_1x2

_EPS = 1e-9
_LN3 = math.log(3.0)
_LN2 = math.log(2.0)


def _binary_log_loss(p_event: float, y: int) -> float:
    """-log della probabilita' assegnata all'esito osservato (y in {0,1})."""
    p = p_event if y == 1 else (1.0 - p_event)
    return -math.log(max(p, _EPS))


def score_completed_matches(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Confronta predizioni ex-ante con risultati reali.

    Args:
        predictions: righe con colonne ``team_a, team_b, p_a_wins, p_draw,
            p_b_wins, p_over_2_5, p_btts`` (formato wc2026_groups_predictions.csv).
        actuals: righe con ``home_team, away_team, home_score, away_score``
            (+ opzionale ``date``).

    Returns:
        ``(scored_df, summary)`` dove ``scored_df`` ha una riga per partita
        valutata e ``summary`` aggrega le metriche con confronto vs baseline
        uniforme.
    """
    # Lookup orientato: (team_a, team_b) -> riga predizione.
    pred_index: dict[tuple[str, str], pd.Series] = {
        (r["team_a"], r["team_b"]): r for _, r in predictions.iterrows()
    }

    rows: list[dict] = []
    for _, m in actuals.iterrows():
        home, away = m["home_team"], m["away_team"]
        hs, as_ = int(m["home_score"]), int(m["away_score"])

        if (home, away) in pred_index:
            pr = pred_index[(home, away)]
            p_home, p_draw, p_away = pr["p_a_wins"], pr["p_draw"], pr["p_b_wins"]
        elif (away, home) in pred_index:
            pr = pred_index[(away, home)]
            # fixture invertito: P(home reale) = prob di vittoria di team_b
            p_home, p_draw, p_away = pr["p_b_wins"], pr["p_draw"], pr["p_a_wins"]
        else:
            continue  # nessuna predizione ex-ante per questa partita

        # esito 1X2 dal punto di vista della squadra di casa reale
        if hs > as_:
            outcome, p_actual = "H", p_home
        elif hs < as_:
            outcome, p_actual = "A", p_away
        else:
            outcome, p_actual = "D", p_draw

        total = hs + as_
        y_o25 = int(total > 2.5)
        y_btts = int(hs > 0 and as_ > 0)

        rows.append({
            "date": m.get("date", ""),
            "home_team": home, "away_team": away,
            "home_score": hs, "away_score": as_,
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "p_over_2_5": pr["p_over_2_5"], "p_btts": pr["p_btts"],
            "actual_1x2": outcome,
            "actual_o25": y_o25, "actual_btts": y_btts,
            "p_actual_1x2": p_actual,
            "log_loss_1x2": -math.log(max(p_actual, _EPS)),
            "log_loss_ou25": _binary_log_loss(pr["p_over_2_5"], y_o25),
            "log_loss_btts": _binary_log_loss(pr["p_btts"], y_btts),
        })

    scored = pd.DataFrame(rows)
    summary = _summarize(scored)
    return scored, summary


def _summarize(scored: pd.DataFrame) -> dict:
    n = len(scored)
    if n == 0:
        return {"n_matches": 0}

    # matrice probabilita' 1X2 e DataFrame "matches" per riusare evaluate.*
    probs = scored[["p_home", "p_draw", "p_away"]].to_numpy()
    matches = pd.DataFrame({
        "home_score": scored["home_score"].to_numpy(),
        "away_score": scored["away_score"].to_numpy(),
    })

    ll_1x2 = log_loss_1x2(matches, probs)
    br_1x2 = brier_score_1x2(matches, probs)
    ll_ou25 = float(scored["log_loss_ou25"].mean())
    ll_btts = float(scored["log_loss_btts"].mean())

    # Brier mercati binari
    br_ou25 = float(((scored["p_over_2_5"] - scored["actual_o25"]) ** 2).mean())
    br_btts = float(((scored["p_btts"] - scored["actual_btts"]) ** 2).mean())

    return {
        "n_matches": n,
        "log_loss_1x2": ll_1x2,
        "brier_1x2": br_1x2,
        "baseline_log_loss_1x2": _LN3,
        "edge_vs_uniform_1x2": _LN3 - ll_1x2,  # positivo = meglio del random
        "log_loss_ou25": ll_ou25,
        "brier_ou25": br_ou25,
        "baseline_log_loss_binary": _LN2,
        "edge_vs_uniform_ou25": _LN2 - ll_ou25,
        "log_loss_btts": ll_btts,
        "brier_btts": br_btts,
        "edge_vs_uniform_btts": _LN2 - ll_btts,
        "hit_rate_1x2": float((scored["p_actual_1x2"] > 1 / 3).mean()),
    }


def load_actual_wc2026_results(results_csv, since: str = "2026-06-01") -> pd.DataFrame:
    """Estrae le partite WC2026 gia' giocate da un results.csv (martj42)."""
    df = pd.read_csv(results_csv)
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    mask = (df["date"] >= since) & (
        df["tournament"].str.contains("World Cup", case=False, na=False)
    )
    out = df.loc[mask, [
        "date", "home_team", "away_team", "home_score", "away_score", "neutral",
    ]].reset_index(drop=True)
    out["home_score"] = out["home_score"].astype(int)
    out["away_score"] = out["away_score"].astype(int)
    return out


def merge_actual_results(
    primary: pd.DataFrame,
    supplement: pd.DataFrame | None,
) -> pd.DataFrame:
    """Unisce una sorgente supplementare ai risultati primari.

    Serve per le partite gia' giocate ma non ancora pubblicate dal dataset
    community (martj42), inserite a mano in ``data/wc2026/manual_results.csv``.
    La sorgente ``primary`` (martj42) ha **precedenza**: appena pubblica una
    partita, l'eventuale riga manuale per la stessa coppia (home, away) viene
    scartata automaticamente, cosi' il supplemento e' auto-pulente.
    """
    if supplement is None or supplement.empty:
        return primary.reset_index(drop=True)
    combined = pd.concat([primary, supplement], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["home_team", "away_team"], keep="first"
    )
    return combined.reset_index(drop=True)
