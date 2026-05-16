"""Build a single CSV with per-team WC2026 probabilities.

Combines:
- Group-stage MC: P(1st), P(2nd), P(qualified) per team
- Knockout MC: P(R16/QF/SF/Final/Winner) per team

Output: reports/wc2026_per_team.csv with 48 rows, one per qualified nation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mondiali.inference.monte_carlo import simulate_group


def main() -> None:
    cfg = json.loads(Path("data/wc2026/groups_template.json").read_text())
    groups_df = pd.read_csv("reports/wc2026_groups_predictions.csv")
    manifest = json.loads(Path("models/v1_final/manifest.json").read_text())
    rho = float(manifest.get("rho_active", manifest.get("rho_xgb", -0.05)))

    # Per-team group probabilities
    rows: list[dict] = []
    for group_letter, teams in cfg["groups"].items():
        group_matches = groups_df[groups_df["group"] == group_letter]
        sim_input = [
            {"team_a": r["team_a"], "team_b": r["team_b"],
             "lam_a": r["lam_a"], "lam_b": r["lam_b"], "rho": rho}
            for _, r in group_matches.iterrows()
        ]
        sim = simulate_group(sim_input, n_sims=10000, seed=42)
        for _, srow in sim.iterrows():
            rows.append({
                "team": srow["team"],
                "group": group_letter,
                "p_first": float(srow["p_first"]),
                "p_second": float(srow["p_second"]),
                "p_qualified_r32": float(srow["p_qualified"]),
                "avg_points_group": float(srow["avg_points"]),
                "avg_gd_group": float(srow["avg_gd"]),
            })
    per_team = pd.DataFrame(rows)

    # Join knockout probabilities (only top 32 qualifiers have them)
    ko_md = Path("reports/wc2026_knockout_simulation.md").read_text()
    # Parse markdown table: lines starting with `| TeamName |`
    ko_rows: list[dict] = []
    in_table = False
    for line in ko_md.splitlines():
        if line.startswith("| Team |"):
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("|") and "%" in line:
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) >= 6:
                team = parts[0]
                pcts = [float(p.rstrip("%")) / 100.0 for p in parts[1:6]]
                ko_rows.append({
                    "team": team,
                    "p_round_of_16": pcts[0],
                    "p_quarterfinal": pcts[1],
                    "p_semifinal": pcts[2],
                    "p_final": pcts[3],
                    "p_winner": pcts[4],
                })
        elif in_table and not line.startswith("|"):
            break
    ko_df = pd.DataFrame(ko_rows)

    merged = per_team.merge(ko_df, on="team", how="left")
    # Teams not in top 32 (didn't qualify for R32) get NaN ko probs — fill with 0
    for col in ["p_round_of_16", "p_quarterfinal", "p_semifinal", "p_final", "p_winner"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    # Sort by P(winner) desc, then P(qualified) desc
    merged = merged.sort_values(["p_winner", "p_qualified_r32"], ascending=False).reset_index(drop=True)

    out_csv = Path("reports/wc2026_per_team.csv")
    merged.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"Wrote {out_csv} ({len(merged)} teams)")
    print()
    print("Top 16 by P(winner):")
    print(merged.head(16).to_string(index=False))


if __name__ == "__main__":
    main()
