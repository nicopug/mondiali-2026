"""WC2026 knockout bracket Monte Carlo simulation.

Given a list of 32 teams in bracket order (positions 1-32 for round-of-32),
simulates the entire knockout (R32 → R16 → QF → SF → Final) 10,000 times and
reports per-team probabilities of reaching each round.

WC2026 format: 32-team knockout after group stage (top 2 of 12 groups + 8 best 3rd).
Bracket pairings are fixed positionally: pos[0] vs pos[1] in R32, winner vs winner
of pos[2]/pos[3] in R16, etc.

Input: a JSON file or dict with key "bracket" = list of 32 team names in order.
For demo: uses the top finishers from groups_template.json simulation (taking
the highest-qualified team per group + 8 best 3rd placed approximated by
remaining best teams).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mondiali.config import CONFIG
from mondiali.inference.monte_carlo import simulate_knockout_bracket
from mondiali.inference.predict import BatchPredictor


def _make_cached_predictor(teams: list[str]):
    """Build BatchPredictor + pre-compute all pairwise lambdas once."""
    model_dir = CONFIG.models_dir / "v1_final"
    state_dir = CONFIG.project_root / "data" / "state"
    snaps_path = CONFIG.data_raw / "transfermarkt" / "snapshots.parquet"
    bp = BatchPredictor(model_dir, state_dir, snaps_path)
    knockout_date = pd.Timestamp("2026-07-01")
    print(f"Pre-computing lambdas for {len(teams)*(len(teams)-1)} ordered pairs...")
    cache = bp.predict_pair_cache(
        teams, knockout_date, neutral=True, competition_importance=75.0,
    )
    rho = bp.rho_active

    def predict_fn(home: str, away: str) -> tuple[float, float, float]:
        lam_h, lam_a = cache[(home, away)]
        return lam_h, lam_a, rho
    return predict_fn


def _qualifiers_from_groups_sim(groups_csv: Path) -> list[str]:
    """Derive 32 R32 qualifiers from group simulation: top 2 per group + best 8 thirds."""
    groups_df = pd.read_csv(groups_csv)
    # Re-aggregate per-team P(qualified), P(third) using avg_points proxy
    # NOTE: this is a heuristic since we don't have per-team P(third) directly;
    # use the same Monte Carlo we just ran but extend it. Pragmatically here we
    # take top 2 highest avg_points per group as the qualifiers and then top 8
    # by avg_points among the 3rd-placed of each group.
    from mondiali.inference.monte_carlo import simulate_group
    import json
    cfg = json.loads(Path("data/wc2026/groups_template.json").read_text())
    manifest = json.loads(Path("models/v1_final/manifest.json").read_text())
    rho = float(manifest.get("rho_active", manifest.get("rho_xgb", -0.05)))

    top_two: list[str] = []
    thirds_with_pts: list[tuple[str, float]] = []
    for group_letter, _teams in cfg["groups"].items():
        group_matches = groups_df[groups_df["group"] == group_letter]
        sim_input = [
            {"team_a": r["team_a"], "team_b": r["team_b"],
             "lam_a": r["lam_a"], "lam_b": r["lam_b"], "rho": rho}
            for _, r in group_matches.iterrows()
        ]
        sim = simulate_group(sim_input, n_sims=10000, seed=42)
        sim = sim.sort_values("avg_points", ascending=False).reset_index(drop=True)
        top_two.append(sim.iloc[0]["team"])
        top_two.append(sim.iloc[1]["team"])
        thirds_with_pts.append((sim.iloc[2]["team"], float(sim.iloc[2]["avg_points"])))

    # Top 8 best thirds
    thirds_with_pts.sort(key=lambda x: -x[1])
    best_thirds = [t for t, _ in thirds_with_pts[:8]]
    return top_two + best_thirds


def main(bracket_path: Path | None = None) -> None:
    if bracket_path is None:
        groups_csv = Path("reports/wc2026_groups_predictions.csv")
        if not groups_csv.exists():
            print("ERROR: run scripts/predict_wc2026_groups.py first")
            return
        bracket_teams = _qualifiers_from_groups_sim(groups_csv)
        print(f"Selected {len(bracket_teams)} qualifiers (24 top-2 per group + 8 best thirds)")
        # Seed-style pairings: 1 vs 32, 2 vs 31, etc.
        # Sort by Elo for a meaningful seed order
        elo = pd.read_parquet(
            CONFIG.project_root / "data" / "state" / "elo_state.parquet"
        )
        elo_dict = dict(zip(elo["nation"], elo["elo"]))
        bracket_teams = sorted(
            bracket_teams, key=lambda t: -elo_dict.get(t, 1500.0)
        )
        n = len(bracket_teams)
        bracket = []
        for i in range(n // 2):
            bracket.append({
                "team_a": bracket_teams[i], "team_b": bracket_teams[n - 1 - i],
            })
    else:
        cfg = json.loads(bracket_path.read_text())
        # bracket_r32 e' una lista di dict con team_a/team_b (gia' risolti).
        bracket = [
            {"team_a": p["team_a"], "team_b": p["team_b"]}
            for p in cfg["bracket_r32"]
        ]

    print(f"Bracket: {len(bracket)} matches in R{len(bracket)*2}")
    print("R32 pairings:")
    for i, m in enumerate(bracket, 1):
        print(f"  {i:2d}. {m['team_a']:25s} vs {m['team_b']}")

    all_teams_in_bracket = []
    for pair in bracket:
        all_teams_in_bracket.extend([pair["team_a"], pair["team_b"]])
    predict_fn = _make_cached_predictor(all_teams_in_bracket)
    print("Running 10000 simulations of full knockout...")
    result = simulate_knockout_bracket(bracket, predict_fn, n_sims=10000, seed=42)
    per_team = result["per_team"]
    n_rounds = result["n_rounds"]

    # Map round indices to names
    round_names = {
        1: "R16", 2: "QF", 3: "SF", 4: "Final", 5: "Winner",
    }
    print(f"\n=== Top 16 by P(win tournament) ===")
    print(f"{'team':25s}  {'R16':>7s}  {'QF':>7s}  {'SF':>7s}  {'Final':>7s}  {'Winner':>7s}")
    for _, row in per_team.head(16).iterrows():
        cols = [f"{row[f'p_round_{r}']*100:6.1f}%" for r in range(1, n_rounds + 1)]
        print(f"{row['team']:25s}  " + "  ".join(cols))

    from datetime import date
    md_path = Path("reports/wc2026_knockout_simulation.md")
    bracket_src = (
        "tabellone reale (data/wc2026/bracket_r32.json)" if bracket_path
        else "placeholder top-Elo dai gironi"
    )
    lines = [
        "# WC2026 — Knockout Bracket Monte Carlo (v1_final)",
        "",
        f"**Generated:** {date.today().isoformat()}  ",
        f"**Bracket:** 32 squadre — {bracket_src}  ",
        f"**Simulations:** 10,000  ",
        "",
        "> Forward prediction leak-free: lo stato Elo riflette tutti i 72 risultati "
        "dei gironi; il modello v1_final (congelato) e' solo *usato*, mai ri-allenato.",
        "",
        "## R32 pairings (positional bracket)",
        "",
    ]
    for i, m in enumerate(bracket, 1):
        lines.append(f"{i}. {m['team_a']} vs {m['team_b']}")
    lines.extend([
        "",
        "## Per-team round reach probabilities",
        "",
        f"| Team | {' | '.join(round_names.get(r, f'R{r}') for r in range(1, n_rounds + 1))} |",
        "|---|" + "|".join(["---"] * n_rounds) + "|",
    ])
    for _, row in per_team.iterrows():
        cells = [f"{row[f'p_round_{r}']*100:.1f}%" for r in range(1, n_rounds + 1)]
        lines.append(f"| {row['team']} | " + " | ".join(cells) + " |")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {md_path}")


if __name__ == "__main__":
    import sys
    arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(arg)
