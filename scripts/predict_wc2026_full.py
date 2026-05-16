"""End-to-end WC2026 tournament prediction (integrated Monte Carlo).

Single Monte Carlo loop over (group stage + knockout): each sim plays out the
entire tournament from group matches through Final. Per-team probabilities are
**unconditional**: a team that's unlikely to qualify gets a correspondingly
small P(reach knockout rounds).

Outputs:
- reports/wc2026_per_team.csv (rebuilt with unconditional probs)
- reports/wc2026_tournament_simulation.md (markdown report)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mondiali.config import CONFIG
from mondiali.inference.monte_carlo import simulate_tournament
from mondiali.inference.predict import BatchPredictor


def main(n_sims: int = 10000, seed: int = 42) -> None:
    cfg = json.loads(Path("data/wc2026/groups_template.json").read_text())
    groups = cfg["groups"]
    comp_imp = float(cfg["competition_importance"])
    all_teams = [t for grp in groups.values() for t in grp]

    print(f"Tournament MC: {len(all_teams)} teams, {n_sims} simulations")

    model_dir = CONFIG.models_dir / "v1_final"
    state_dir = CONFIG.project_root / "data" / "state"
    snaps_path = CONFIG.data_raw / "transfermarkt" / "snapshots.parquet"
    bp = BatchPredictor(model_dir, state_dir, snaps_path)
    rho = bp.rho_active
    tournament_date = pd.Timestamp(cfg["start_date"])

    print(f"Pre-computing {len(all_teams)*(len(all_teams)-1)} pair lambdas...")
    pair_lambdas = bp.predict_pair_cache(
        all_teams, tournament_date, neutral=True,
        competition_importance=comp_imp,
    )

    # Elo lookup for bracket seeding
    elo_df = pd.read_parquet(state_dir / "elo_state.parquet")
    elo_dict = dict(zip(elo_df["nation"], elo_df["elo"]))

    print(f"Running integrated tournament Monte Carlo ({n_sims} sims)...")
    per_team = simulate_tournament(
        groups, pair_lambdas, rho,
        elo_dict=elo_dict, n_sims=n_sims, seed=seed,
    )

    # Sanity check
    sums = {
        "winners (sum P_first should = 12)": per_team["p_first"].sum(),
        "runners-up (sum P_second should = 12)": per_team["p_second"].sum(),
        "thirds (sum P_third should = 12)": per_team["p_third"].sum(),
        "qualified_r32 (should = 32)": per_team["p_qualified_r32"].sum(),
        "round_of_16 (should = 16)": per_team["p_round_of_16"].sum(),
        "quarterfinal (should = 8)": per_team["p_quarterfinal"].sum(),
        "semifinal (should = 4)": per_team["p_semifinal"].sum(),
        "final (should = 2)": per_team["p_final"].sum(),
        "winner (should = 1)": per_team["p_winner"].sum(),
    }
    print("Calibration sanity:")
    for k, v in sums.items():
        print(f"  {k:50s}  {v:.4f}")

    out_csv = Path("reports/wc2026_per_team.csv")
    per_team.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"\nWrote {out_csv}")

    print("\nTop 20 by P(winner):")
    print(per_team.head(20)[[
        "team", "group", "p_first", "p_qualified_r32",
        "p_round_of_16", "p_quarterfinal", "p_semifinal", "p_final", "p_winner",
    ]].to_string(index=False))

    # Markdown report
    manifest = json.loads((model_dir / "manifest.json").read_text())
    lines = [
        "# WC2026 — Tournament Monte Carlo (Integrated)",
        "",
        f"**Generated:** 2026-05-16  ",
        f"**Model:** v1_final ({manifest.get('version', 'unknown')})  ",
        f"**Simulations:** {n_sims} integrated group+knockout sims  ",
        f"**Methodology:** single MC propagates group uncertainty into knockout. "
        "Bracket seeded by Elo descending (1 vs 32) without enforcing same-group "
        "constraint. Group winners + runners-up + 8 best 3rd-placed teams advance.",
        "",
        "## Top 20 by P(winner)",
        "",
        "| Team | Group | P(1st) | P(qualif) | P(R16) | P(QF) | P(SF) | P(Final) | **P(Win)** |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in per_team.head(20).iterrows():
        lines.append(
            f"| {r['team']} | {r['group']} | "
            f"{r['p_first']*100:.1f}% | {r['p_qualified_r32']*100:.1f}% | "
            f"{r['p_round_of_16']*100:.1f}% | {r['p_quarterfinal']*100:.1f}% | "
            f"{r['p_semifinal']*100:.1f}% | {r['p_final']*100:.1f}% | "
            f"**{r['p_winner']*100:.1f}%** |"
        )
    lines.extend([
        "",
        "## All 48 teams",
        "",
        "| Team | Group | P(1st) | P(2nd) | P(qualif) | P(R16) | P(QF) | P(SF) | P(F) | P(Win) | Avg pts | Avg GD |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ])
    for _, r in per_team.iterrows():
        lines.append(
            f"| {r['team']} | {r['group']} | "
            f"{r['p_first']*100:.1f}% | {r['p_second']*100:.1f}% | "
            f"{r['p_qualified_r32']*100:.1f}% | {r['p_round_of_16']*100:.1f}% | "
            f"{r['p_quarterfinal']*100:.1f}% | {r['p_semifinal']*100:.1f}% | "
            f"{r['p_final']*100:.1f}% | {r['p_winner']*100:.1f}% | "
            f"{r['avg_points']:.2f} | {r['avg_gd']:+.2f} |"
        )
    out_md = Path("reports/wc2026_tournament_simulation.md")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
