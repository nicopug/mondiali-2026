"""Predizione del tabellone knockout WC2026 (R32 -> Finale -> vincitore).

A differenza di ``predict_wc2026_full.py`` (che ri-semina per Elo ad ogni turno,
gonfiando le big), questo usa il **tabellone ufficiale FIFA**: accoppiamenti
posizionali corretti, conservati turno dopo turno via ``simulate_knockout_bracket``.

Workflow a gironi finiti:
    1. python scripts/predict_wc2026_bracket.py            # crea il template se assente
    2. compila data/wc2026/bracket_r32.json coi nomi reali delle 32 qualificate
    3. python scripts/predict_wc2026_bracket.py            # simula e scrive il report

Solo inferenza sul modello congelato: non viola il freeze.

Output:
    reports/wc2026_bracket_simulation.md
    reports/wc2026_bracket_per_team.csv
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from mondiali.config import CONFIG
from mondiali.inference.bracket import (
    OFFICIAL_R32,
    ROUND_LABELS,
    BracketError,
    load_bracket,
)
from mondiali.inference.monte_carlo import simulate_knockout_bracket
from mondiali.inference.predict import BatchPredictor

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PATH = REPO / "data" / "wc2026" / "bracket_r32.json"
KNOCKOUT_DATE = "2026-07-01"
COMP_IMPORTANCE = 75.0


def write_template(path: Path) -> None:
    """Scrive il template del tabellone ufficiale con slot da compilare."""
    pairs = [
        {
            "match": s["match"],
            "slot_a": s["slot_a"], "slot_b": s["slot_b"],
            "team_a": "", "team_b": "",
        }
        for s in OFFICIAL_R32
    ]
    template = {
        "_README": (
            "Tabellone ufficiale WC2026 in ordine a foglie. Compila team_a/team_b "
            "coi nomi REALI delle qualificate (come appaiono in data/state/elo_state.parquet). "
            "Slot: W-X=vincitrice girone X, RU-X=seconda, 3-XYZ=terza da uno dei gironi elencati. "
            "L'assegnazione delle 8 terze segue l'Annex C FIFA in base a quali terze passano."
        ),
        "knockout_date": KNOCKOUT_DATE,
        "bracket_r32": pairs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_predict_fn(teams: list[str], knockout_date: str):
    model_dir = CONFIG.models_dir / "v1_final"
    state_dir = CONFIG.project_root / "data" / "state"
    snaps_path = CONFIG.data_raw / "transfermarkt" / "snapshots.parquet"
    bp = BatchPredictor(model_dir, state_dir, snaps_path)
    cache = bp.predict_pair_cache(
        teams, pd.Timestamp(knockout_date),
        neutral=True, competition_importance=COMP_IMPORTANCE,
    )
    rho = bp.rho_active

    def predict_fn(home: str, away: str):
        lam_h, lam_a = cache[(home, away)]
        return lam_h, lam_a, rho

    return predict_fn


def main(bracket_path: Path = DEFAULT_PATH) -> None:
    if not bracket_path.exists():
        write_template(bracket_path)
        print(f"Template scritto in {bracket_path}")
        print("Compila i 32 slot coi nomi reali delle qualificate, poi rilancia.")
        return

    data = json.loads(bracket_path.read_text(encoding="utf-8"))
    try:
        bracket = load_bracket(data)
    except BracketError as exc:
        print(f"ERRORE bracket: {exc}", file=sys.stderr)
        sys.exit(1)

    knockout_date = data.get("knockout_date", KNOCKOUT_DATE)
    teams = [t for pair in bracket for t in (pair["team_a"], pair["team_b"])]
    print(f"Tabellone valido: 32 squadre, simulazione al {knockout_date}.")

    predict_fn = _build_predict_fn(teams, knockout_date)
    result = simulate_knockout_bracket(bracket, predict_fn, n_sims=10000, seed=42)
    per_team = result["per_team"]
    n_rounds = result["n_rounds"]  # 5 per 32 squadre

    # Rinomina p_round_1..5 -> R16/QF/SF/Final/Winner
    rename = {f"p_round_{r}": ROUND_LABELS[r - 1] for r in range(1, n_rounds + 1)}
    table = per_team.rename(columns=rename)[["team", *ROUND_LABELS]]

    out_csv = REPO / "reports" / "wc2026_bracket_per_team.csv"
    table.to_csv(out_csv, index=False, float_format="%.4f")

    lines = [
        "# WC2026 — Tabellone knockout (bracket ufficiale)",
        "",
        f"**Generato:** {date.today().isoformat()}  ",
        "**Modello:** v1_final (congelato)  ",
        f"**Simulazioni:** {result['n_sims']}  ",
        f"**Data inferenza:** {knockout_date}  ",
        "",
        "> Accoppiamenti posizionali ufficiali FIFA (non ri-seminati per Elo).",
        "",
        "## R32 — accoppiamenti simulati (ordine a foglie)",
        "",
    ]
    for i, (pair, slot) in enumerate(zip(bracket, OFFICIAL_R32, strict=True), 1):
        lines.append(
            f"{i:2d}. (M{slot['match']}) {pair['team_a']} [{slot['slot_a']}] "
            f"vs {pair['team_b']} [{slot['slot_b']}]"
        )
    lines += [
        "",
        "## Probabilita' per turno",
        "",
        "| Squadra | " + " | ".join(f"P({lbl})" for lbl in ROUND_LABELS) + " |",
        "|---|" + "|".join(["---"] * len(ROUND_LABELS)) + "|",
    ]
    for _, row in table.iterrows():
        cells = [f"{row[lbl] * 100:.1f}%" for lbl in ROUND_LABELS]
        lines.append(f"| {row['team']} | " + " | ".join(cells) + " |")
    lines.append("")

    out_md = REPO / "reports" / "wc2026_bracket_simulation.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print("\n=== Top 8 P(vincitore) ===")
    top = table.sort_values("Winner", ascending=False).head(8)
    for _, r in top.iterrows():
        print(f"  {r['team']:24s} Winner={r['Winner'] * 100:5.1f}%  "
              f"Final={r['Final'] * 100:5.1f}%  SF={r['SF'] * 100:5.1f}%")
    print(f"\nReport -> {out_md}")
    print(f"CSV    -> {out_csv}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    main(path)
