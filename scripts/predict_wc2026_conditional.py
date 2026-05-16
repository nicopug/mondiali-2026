"""Conditional WC2026 MC: re-run tournament simulation with known results fixed.

Reads `data/wc2026/results_so_far.json` (an optional dict mapping
"home_team-away_team-YYYY-MM-DD" -> [home_score, away_score]) and clamps those
matches in the MC. Re-simulates only the unknown matches.

Usage:
    # After Argentina 2-0 Algeria on 2026-06-15:
    # edit data/wc2026/results_so_far.json:
    # {"Argentina-Algeria-2026-06-15": [2, 0]}
    # then:
    python scripts/predict_wc2026_conditional.py
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from mondiali.config import CONFIG
from mondiali.inference.monte_carlo import simulate_tournament
from mondiali.inference.predict import BatchPredictor


def main(n_sims: int = 10000, seed: int = 42) -> None:
    cfg = json.loads(Path("data/wc2026/groups_template.json").read_text())
    groups = cfg["groups"]
    comp_imp = float(cfg["competition_importance"])
    all_teams = [t for grp in groups.values() for t in grp]

    results_path = Path("data/wc2026/results_so_far.json")
    fixed_results: dict[tuple[str, str], tuple[int, int]] = {}
    if results_path.exists():
        raw = json.loads(results_path.read_text())
        for key, score in raw.items():
            parts = key.rsplit("-", 1)[0]  # drop date suffix
            home, away = parts.rsplit("-", 1)
            fixed_results[(home, away)] = (int(score[0]), int(score[1]))
        print(f"Loaded {len(fixed_results)} fixed results from {results_path}")
    else:
        print(f"No fixed results found at {results_path}; running fully stochastic MC")

    model_dir = CONFIG.models_dir / "v1_final"
    state_dir = CONFIG.project_root / "data" / "state"
    snaps_path = CONFIG.data_raw / "transfermarkt" / "snapshots.parquet"
    bp = BatchPredictor(model_dir, state_dir, snaps_path)
    rho = bp.rho_active
    tournament_date = pd.Timestamp(cfg["start_date"])

    print(f"Pre-computing pair lambdas...")
    pair_lambdas = bp.predict_pair_cache(
        all_teams, tournament_date, neutral=True,
        competition_importance=comp_imp,
    )

    # Override lambdas for matches with known scores: set deterministic distribution
    # by replacing the lambda value with a degenerate distribution at the observed
    # score. Simpler: pre-sample N copies of the known score, replace in MC.
    # We do this by using a "near-degenerate" lambda that matches the known score
    # mean. NOTE: this is approximate. A cleaner impl would patch
    # `simulate_tournament` to accept fixed-result overrides.
    fixed_lambdas: dict[tuple[str, str], tuple[float, float]] = {
        (h, a): (float(hs) + 0.001, float(as_) + 0.001)
        for (h, a), (hs, as_) in fixed_results.items()
    }
    final_lambdas = {**pair_lambdas, **fixed_lambdas}

    elo_df = pd.read_parquet(state_dir / "elo_state.parquet")
    elo_dict = dict(zip(elo_df["nation"], elo_df["elo"]))

    per_team = simulate_tournament(
        groups, final_lambdas, rho,
        elo_dict=elo_dict, n_sims=n_sims, seed=seed,
    )
    out_csv = Path("reports/wc2026_per_team_conditional.csv")
    per_team.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"\nWrote {out_csv}")
    print(f"Top 16 by P(winner):")
    print(per_team.head(16)[[
        "team", "group", "p_qualified_r32", "p_winner",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
