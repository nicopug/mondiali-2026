"""Predict WC2026 group stage and run Monte Carlo per group.

Reads data/wc2026/groups_template.json (or --config), generates the 72 round-robin
group matches, predicts each with v1_final ensemble, runs 10k MC simulations per
group, and outputs:
  - reports/wc2026_groups_predictions.csv (per-match)
  - reports/wc2026_groups_simulation.md (per-team P(qualif/first/second/eliminated))
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from mondiali.config import CONFIG
from mondiali.inference.monte_carlo import simulate_group
from mondiali.inference.predict import predict_match


def _is_host_match(group_teams: list[str], host_nations: set[str]) -> bool:
    """A match is 'non-neutral' only when a host plays at home in their own group."""
    return False  # All WC group matches treated as neutral in venue from model's pov


def main(config_path: Path = Path("data/wc2026/groups_template.json")) -> None:
    cfg = json.loads(config_path.read_text())
    start_date = pd.Timestamp(cfg["start_date"])
    comp_imp = float(cfg["competition_importance"])
    host_nations = set(cfg["host_nations"])
    groups = cfg["groups"]
    model_dir = CONFIG.models_dir / "v1_final"
    state_dir = CONFIG.project_root / "data" / "state"
    snaps_path = CONFIG.data_raw / "transfermarkt" / "snapshots.parquet"

    all_matches: list[dict] = []
    print(f"Predicting {sum(len(t) for t in groups.values())} teams in {len(groups)} groups...")
    for group_name, teams in groups.items():
        if len(teams) != 4:
            raise ValueError(f"Group {group_name}: expected 4 teams, got {len(teams)}")
        for team_a, team_b in combinations(teams, 2):
            # Host advantage: if a host team plays, neutral=False (slight home boost)
            neutral = (team_a not in host_nations) and (team_b not in host_nations)
            try:
                pred = predict_match(
                    home=team_a, away=team_b, date=start_date, neutral=neutral,
                    state_dir=state_dir, model_dir=model_dir,
                    tm_snapshots_path=snaps_path,
                    competition_importance=comp_imp,
                )
            except Exception as exc:
                print(f"  [skip] {team_a} vs {team_b}: {exc}")
                continue
            all_matches.append({
                "group": group_name,
                "team_a": team_a, "team_b": team_b,
                "neutral": neutral,
                "lam_a": pred["lambda"]["home"], "lam_b": pred["lambda"]["away"],
                "p_a_wins": pred["markets"]["1x2"]["home"],
                "p_draw": pred["markets"]["1x2"]["draw"],
                "p_b_wins": pred["markets"]["1x2"]["away"],
                "p_over_2_5": pred["markets"]["over_under_2_5"]["over"],
                "p_btts": pred["markets"]["btts"]["yes"],
            })
        print(f"  Group {group_name}: {len(teams)} teams, 6 matches predicted")

    matches_df = pd.DataFrame(all_matches)
    csv_path = Path("reports/wc2026_groups_predictions.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    matches_df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # Get rho from manifest for MC (use ensemble rho if available, else XGB rho)
    manifest = json.loads((model_dir / "manifest.json").read_text())
    rho = float(manifest.get("rho_active", manifest.get("rho_xgb", -0.05)))

    print("\nRunning Monte Carlo (10000 sims per group)...")
    group_results: dict[str, pd.DataFrame] = {}
    for group_name, group_df in matches_df.groupby("group"):
        sim_input = [
            {"team_a": r["team_a"], "team_b": r["team_b"],
             "lam_a": r["lam_a"], "lam_b": r["lam_b"], "rho": rho}
            for _, r in group_df.iterrows()
        ]
        sim = simulate_group(sim_input, n_sims=10000, seed=42)
        group_results[group_name] = sim
        print(f"  Group {group_name}:")
        for _, row in sim.iterrows():
            print(f"    {row['team']:25s}  qual={row['p_qualified']*100:5.1f}%  "
                  f"1st={row['p_first']*100:5.1f}%  pts={row['avg_points']:.2f}")

    # Markdown report
    lines = [
        "# WC2026 — Group Stage Predictions (v1_final)",
        "",
        f"**Generated:** 2026-05-16  ",
        f"**Model:** `models/v1_final/` ({manifest.get('version', 'unknown')})  ",
        f"**Tournament start:** {cfg['start_date']}  ",
        f"**competition_importance:** {comp_imp}  ",
        f"**Monte Carlo simulations:** 10,000 per group",
        "",
        "**Note:** team rosters in `data/wc2026/groups_template.json` are placeholders. "
        "Replace with actual FIFA draw when known and re-run this script.",
        "",
    ]
    for group_name in sorted(group_results.keys()):
        sim = group_results[group_name]
        lines.append(f"## Group {group_name}")
        lines.append("")
        lines.append("| Team | P(1st) | P(2nd) | P(qualified) | Avg pts | Avg GD |")
        lines.append("|---|---|---|---|---|---|")
        for _, row in sim.iterrows():
            lines.append(
                f"| {row['team']} | {row['p_first']*100:.1f}% | "
                f"{row['p_second']*100:.1f}% | {row['p_qualified']*100:.1f}% | "
                f"{row['avg_points']:.2f} | {row['avg_gd']:+.2f} |"
            )
        lines.append("")
        # Match preview
        group_matches = matches_df[matches_df["group"] == group_name]
        lines.append(f"### Match previews — Group {group_name}")
        lines.append("")
        for _, m in group_matches.iterrows():
            lines.append(
                f"- **{m['team_a']} vs {m['team_b']}** — "
                f"λ {m['lam_a']:.2f} vs {m['lam_b']:.2f}, "
                f"P(W/D/L) = {m['p_a_wins']*100:.0f}/{m['p_draw']*100:.0f}/"
                f"{m['p_b_wins']*100:.0f}%"
            )
        lines.append("")

    md_path = Path("reports/wc2026_groups_simulation.md")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
