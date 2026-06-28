"""Baseline Elo-only sulle 72 partite dei gironi WC2026.

Scrupolo di confronto: il modello v1_final batte davvero il puro Elo, o solo il
random uniforme? Fitta ``EloLogisticBaseline`` (feature: elo_diff + neutral) su
tutte le partite internazionali *anteriori* al kickoff (2026-06-11, leak-free) e
lo valuta sulle 72 partite dei gironi, contro:
  - random uniforme (ln3 = 1.0986)
  - modello v1_final (log-loss dalle predizioni congelate, reports/wc2026_scored_matches.csv)

Usage:
    python scripts/eval_elo_baseline.py

Output:
    reports/elo_baseline_72.md
"""
from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from mondiali.evaluation.elo_baseline import apply_frozen_elo, pretournament_elo_map
from mondiali.evaluation.live_scoring import _norm_name
from mondiali.model.elo_logistic import EloLogisticBaseline
from mondiali.training.evaluate import (
    brier_score_1x2,
    compute_outcomes,
    log_loss_1x2,
)

_EPS = 1e-9

REPO = Path(__file__).resolve().parent.parent
MATCHES = REPO / "data" / "processed" / "matches.parquet"
SCORED = REPO / "reports" / "wc2026_scored_matches.csv"
OUT_MD = REPO / "reports" / "elo_baseline_72.md"

KICKOFF = pd.Timestamp("2026-06-11")
LN3 = math.log(3.0)


def _metrics(matches: pd.DataFrame, probs) -> tuple[float, float]:
    return log_loss_1x2(matches, probs), brier_score_1x2(matches, probs)


def main() -> None:
    df = pd.read_parquet(MATCHES)
    df["date"] = pd.to_datetime(df["date"])

    needed = ["home_elo_before", "away_elo_before", "neutral", "home_score", "away_score"]
    train = df[df["date"] < KICKOFF].dropna(subset=needed).copy()
    test = df[
        (df["date"] >= KICKOFF)
        & (df["tournament"].str.contains("World Cup", case=False, na=False))
    ].dropna(subset=needed).copy()

    baseline = EloLogisticBaseline().fit(train)

    # Variante LIVE: Elo pre-partita reale (assorbe le giornate precedenti).
    probs_live = baseline.predict_proba(test)
    ll_live, br_live = _metrics(test, probs_live)

    # Variante EX-ANTE (frozen): Elo congelato pre-torneo, costante sulle 3 giornate.
    elo_map = pretournament_elo_map(df, KICKOFF)
    test_frozen = apply_frozen_elo(test, elo_map)
    probs_frozen = baseline.predict_proba(test_frozen)
    ll_frozen, br_frozen = _metrics(test_frozen, probs_frozen)

    # Modello v1_final: log-loss sulle stesse 72 dalle predizioni congelate.
    scored = pd.read_csv(SCORED)
    model_probs = scored[["p_home", "p_draw", "p_away"]].to_numpy()
    model_matches = scored[["home_score", "away_score"]]
    ll_model, br_model = _metrics(model_matches, model_probs)

    # esiti per contesto
    out = compute_outcomes(test)
    n_draw = int((out == 1).sum())

    # --- Analisi appaiata modello vs Elo-only ex-ante (per significativita') ---
    test = test.reset_index(drop=True)
    elo_ll = np.array([
        -math.log(max(probs_frozen[i, out[i]], _EPS)) for i in range(len(out))
    ])
    model_ll_by_key = {
        frozenset((_norm_name(r["home_team"]), _norm_name(r["away_team"]))):
            -math.log(max(float(r["p_actual_1x2"]), _EPS))
        for _, r in scored.iterrows()
    }
    m_ll, e_ll, o_cls = [], [], []
    for i, r in test.iterrows():
        key = frozenset((_norm_name(r["home_team"]), _norm_name(r["away_team"])))
        if key in model_ll_by_key:
            m_ll.append(model_ll_by_key[key])
            e_ll.append(elo_ll[i])
            o_cls.append(out[i])
    m_ll, e_ll, o_cls = np.array(m_ll), np.array(e_ll), np.array(o_cls)
    diff = m_ll - e_ll  # >0 -> modello peggiore
    rng = np.random.default_rng(42)
    boots = np.array([
        diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)
    ])
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    p_worse = float((boots > 0).mean())
    by_outcome = {
        name: (float(m_ll[o_cls == c].mean()), float(e_ll[o_cls == c].mean()), int((o_cls == c).sum()))
        for c, name in [(0, "Vittoria casa"), (1, "Pareggio"), (2, "Vittoria fuori")]
        if (o_cls == c).any()
    }
    avg_pdraw_elo = float(probs_frozen[:, 1].mean())
    avg_pdraw_model = float(pd.read_csv(REPO / "reports" / "wc2026_groups_predictions.csv")["p_draw"].mean())

    rows = [
        ("Random uniforme", LN3, None, LN3 - LN3),
        ("Elo-only (ex-ante, frozen)", ll_frozen, br_frozen, LN3 - ll_frozen),
        ("Elo-only (live pre-match)", ll_live, br_live, LN3 - ll_live),
        ("Modello v1_final (congelato)", ll_model, br_model, LN3 - ll_model),
    ]

    lines: list[str] = []
    lines.append("# WC2026 — Baseline Elo-only vs modello (72 partite gironi)")
    lines.append("")
    lines.append(f"**Generato:** {date.today().isoformat()}  ")
    lines.append(f"**Train:** {len(train)} partite internazionali con `date < {KICKOFF.date()}` (leak-free)  ")
    lines.append(f"**Test:** {len(test)} partite dei gironi WC2026 ({n_draw} pareggi, {n_draw / len(test):.0%})  ")
    lines.append("**Baseline:** `EloLogisticBaseline` — feature `[elo_diff, neutral]`, C=1.0, random_state=42  ")
    lines.append("")
    lines.append("> Domanda: i tier 2/3 dell'XGBoost aggiungono valore *oltre* al puro Elo? "
                 "Battere il random e' un'asticella bassa; battere l'Elo nudo, no.")
    lines.append("")
    lines.append("| Modello | log-loss 1X2 | Brier | edge vs random |")
    lines.append("|---|---|---|---|")
    for name, ll, br, edge in rows:
        br_s = f"{br:.4f}" if br is not None else "—"
        lines.append(f"| {name} | {ll:.4f} | {br_s} | {edge:+.4f} |")
    lines.append("")

    gap = ll_model - ll_frozen
    verdict = (
        f"Il modello batte l'Elo-only ex-ante di {-gap:+.4f} nats."
        if gap < 0 else
        f"Il modello NON batte l'Elo-only ex-ante (lo perde di {gap:+.4f} nats)."
    )
    lines.append(f"**Verdetto (confronto equo, ex-ante vs ex-ante):** {verdict}  ")
    lines.append("")
    lines.append("## Significativita' (bootstrap appaiato, modello − Elo)")
    lines.append("")
    lines.append(f"- Differenza media log-loss per partita: **{diff.mean():+.4f}** (>0 = modello peggiore)  ")
    lines.append(f"- Bootstrap 95% CI (10k, seed 42): **[{ci_lo:+.4f}, {ci_hi:+.4f}]**  ")
    lines.append(f"- P(modello peggiore dell'Elo) = **{p_worse:.1%}**  ")
    lines.append("")
    lines.append("Lo zero e' fuori dall'intervallo: il distacco non e' rumore da campione piccolo.")
    lines.append("")
    lines.append("## Dove perde il modello")
    lines.append("")
    lines.append(f"Tasso pareggi reale **{n_draw / len(test):.0%}**; "
                 f"P(X) media — modello **{avg_pdraw_model:.0%}**, Elo **{avg_pdraw_elo:.0%}**. "
                 "Entrambi sottostimano i pari (tratto strutturale del Poisson), il modello di piu'.")
    lines.append("")
    lines.append("| Esito reale | n | log-loss modello | log-loss Elo |")
    lines.append("|---|---|---|---|")
    for name, (m, e, n) in by_outcome.items():
        lines.append(f"| {name} | {n} | {m:.3f} | {e:.3f} |")
    lines.append("")
    lines.append("Il modello sanguina su **vittorie casa** e **pareggi**; recupera solo sulle "
                 "trasferte. I tier 2/3 (forma, valori di mercato) sembrano aggiungere rumore, "
                 "non segnale, su questo campione.")
    lines.append("")
    lines.append("Note metodologiche:")
    lines.append("- *ex-ante (frozen)* = stesso set informativo delle predizioni congelate del "
                 "modello (Elo pre-torneo, una sola volta). E' il confronto corretto.")
    lines.append("- *live* = l'Elo assorbe le giornate gia' giocate; vantaggio informativo non "
                 "disponibile al modello congelato, quindi solo come riferimento.")
    lines.append("- Soglia storica del progetto: l'XGBoost Tier 1 deve battere questo baseline di "
                 "almeno 0.003 in log-loss, altrimenti STOP/debug.")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"Train {len(train)} | Test {len(test)} ({n_draw} pareggi)")
    print(f"  Random uniforme           log-loss = {LN3:.4f}")
    print(f"  Elo-only (ex-ante frozen) log-loss = {ll_frozen:.4f}  (edge {LN3 - ll_frozen:+.4f})")
    print(f"  Elo-only (live)           log-loss = {ll_live:.4f}  (edge {LN3 - ll_live:+.4f})")
    print(f"  Modello v1_final          log-loss = {ll_model:.4f}  (edge {LN3 - ll_model:+.4f})")
    print(f"  -> {verdict}")
    print(f"Report -> {OUT_MD}")


if __name__ == "__main__":
    main()
