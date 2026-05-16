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


def main(bracket_path: Path | None = None) -> None:
    if bracket_path is None:
        # Default: use top 2 from each group + best 3rd from groups_template
        # For 12 groups + 8 best 3rd = 32 → we approximate by top 32 nations by Elo
        cfg = json.loads(Path("data/wc2026/groups_template.json").read_text())
        all_teams: list[str] = []
        for teams in cfg["groups"].values():
            all_teams.extend(teams[:3])  # top 3 per group = 36 candidates
        # Pick top 32 by Elo
        elo = pd.read_parquet(
            CONFIG.project_root / "data" / "state" / "elo_state.parquet"
        )
        elo_dict = dict(zip(elo["nation"], elo["elo"]))
        ranked = sorted(
            (t for t in all_teams if t in elo_dict),
            key=lambda t: -elo_dict[t],
        )
        bracket_teams = ranked[:32]
        # Pair them: seed-1 vs seed-32, seed-2 vs seed-31, etc. (standard tennis-style)
        n = len(bracket_teams)
        bracket = []
        for i in range(n // 2):
            bracket.append({
                "team_a": bracket_teams[i], "team_b": bracket_teams[n - 1 - i],
            })
    else:
        cfg = json.loads(bracket_path.read_text())
        bracket = [{"team_a": p[0], "team_b": p[1]} for p in cfg["bracket_r32"]]

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

    md_path = Path("reports/wc2026_knockout_simulation.md")
    lines = [
        "# WC2026 — Knockout Bracket Monte Carlo (v1_final)",
        "",
        f"**Generated:** 2026-05-16  ",
        f"**Bracket:** 32 teams (top by Elo from group placeholder)  ",
        f"**Simulations:** 10,000  ",
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
    main()
