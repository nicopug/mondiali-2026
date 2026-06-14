"""Valuta le predizioni ex-ante WC2026 contro i risultati reali (leak-free).

Confronta le probabilita' CONGELATE prima del torneo
(``reports/wc2026_groups_predictions.csv``, generate il 2026-05-16) con i
risultati realmente avvenuti (``data/raw/results.csv``, dataset martj42).

NON ri-predice nulla: niente torch, niente rischio leakage. Misura solo.

Usage:
    python scripts/score_wc2026_predictions.py

Output:
    reports/wc2026_scored_matches.csv   (una riga per partita valutata)
    reports/wc2026_live_scoring.md      (report aggregato + per-match)
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from mondiali.evaluation.live_scoring import (
    load_actual_wc2026_results,
    merge_actual_results,
    score_completed_matches,
)

REPO = Path(__file__).resolve().parent.parent
PRED_CSV = REPO / "reports" / "wc2026_groups_predictions.csv"
RESULTS_CSV = REPO / "data" / "raw" / "results.csv"
MANUAL_CSV = REPO / "data" / "wc2026" / "manual_results.csv"
OUT_CSV = REPO / "reports" / "wc2026_scored_matches.csv"
OUT_MD = REPO / "reports" / "wc2026_live_scoring.md"


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def main() -> None:
    preds = pd.read_csv(PRED_CSV)
    actuals = load_actual_wc2026_results(RESULTS_CSV)
    # Supplemento manuale per partite gia' giocate ma non ancora pubblicate da
    # martj42 (auto-pulente: martj42 ha precedenza quando le pubblica).
    n_manual = 0
    if MANUAL_CSV.exists():
        manual = pd.read_csv(MANUAL_CSV)
        before = len(actuals)
        actuals = merge_actual_results(actuals, manual)
        n_manual = len(actuals) - before
    scored, s = score_completed_matches(preds, actuals)

    if s["n_matches"] == 0:
        print("Nessuna partita WC2026 valutabile (results.csv non ancora aggiornato).")
        return

    scored.to_csv(OUT_CSV, index=False, float_format="%.4f")

    lines: list[str] = []
    lines.append("# WC2026 — Live scoring (ex-ante vs reale)")
    lines.append("")
    lines.append(f"**Generato:** {date.today().isoformat()}  ")
    lines.append("**Predizioni:** `reports/wc2026_groups_predictions.csv` (congelate 2026-05-16)  ")
    lines.append("**Risultati:** `data/raw/results.csv` (martj42)  ")
    if n_manual:
        lines.append(
            f"**Supplemento manuale:** `data/wc2026/manual_results.csv` "
            f"({n_manual} partite non ancora su martj42)  "
        )
    lines.append(f"**Partite valutate:** {s['n_matches']}")
    lines.append("")
    lines.append("> Leak-free: si valutano solo le probabilita' ex-ante, mai ri-predette.")
    lines.append("")
    lines.append("## Metriche aggregate")
    lines.append("")
    lines.append("| Mercato | log-loss modello | log-loss baseline uniforme | edge | Brier |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| 1X2 | {_fmt(s['log_loss_1x2'])} | {_fmt(s['baseline_log_loss_1x2'])} "
        f"| {s['edge_vs_uniform_1x2']:+.4f} | {_fmt(s['brier_1x2'])} |"
    )
    lines.append(
        f"| Over/Under 2.5 | {_fmt(s['log_loss_ou25'])} | {_fmt(s['baseline_log_loss_binary'])} "
        f"| {s['edge_vs_uniform_ou25']:+.4f} | {_fmt(s['brier_ou25'])} |"
    )
    lines.append(
        f"| BTTS | {_fmt(s['log_loss_btts'])} | {_fmt(s['baseline_log_loss_binary'])} "
        f"| {s['edge_vs_uniform_btts']:+.4f} | {_fmt(s['brier_btts'])} |"
    )
    lines.append("")
    lines.append(
        "`edge` positivo = il modello batte la predizione casuale; "
        "negativo = peggio del random."
    )
    lines.append("")
    lines.append("## Dettaglio partite")
    lines.append("")
    lines.append("| Data | Partita | Risultato | P(H/X/A) | Esito | P(esito) | LL 1X2 |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in scored.iterrows():
        lines.append(
            f"| {r['date']} | {r['home_team']}–{r['away_team']} "
            f"| {r['home_score']}-{r['away_score']} "
            f"| {r['p_home']:.2f}/{r['p_draw']:.2f}/{r['p_away']:.2f} "
            f"| {r['actual_1x2']} | {r['p_actual_1x2']:.2f} | {r['log_loss_1x2']:.3f} |"
        )
    lines.append("")
    lines.append(
        f"**Hit-rate 1X2** (esito reale sopra 1/3 di probabilita'): "
        f"{s['hit_rate_1x2']:.0%}"
    )
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"Valutate {s['n_matches']} partite (di cui {n_manual} da supplemento manuale).")
    print(f"  1X2 log-loss = {s['log_loss_1x2']:.4f} (baseline {s['baseline_log_loss_1x2']:.4f}, "
          f"edge {s['edge_vs_uniform_1x2']:+.4f})")
    print(f"  O/U 2.5 log-loss = {s['log_loss_ou25']:.4f}")
    print(f"  BTTS log-loss = {s['log_loss_btts']:.4f}")
    print(f"Report -> {OUT_MD}")
    print(f"CSV    -> {OUT_CSV}")


if __name__ == "__main__":
    main()
