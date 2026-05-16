"""Daily WC2026 workflow: ingest -> update-state -> re-predict tournament.

Designed to be run once per day during the tournament:
    python scripts/daily_workflow.py

Steps:
  1. Refresh match data from source (mondiali ingest --force)
  2. Rebuild state cache (mondiali update-state)
  3. Re-run integrated tournament Monte Carlo
  4. Append results to reports/wc2026_live_tracking.csv (per-day snapshot)

After group stage (when actual qualifiers are known), edit
data/wc2026/groups_template.json to reflect actual finishing positions, and
the tournament MC will respect them via the conditional logic.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        print(f"!!! command failed (exit {result.returncode})")
        sys.exit(result.returncode)


def append_tracking_snapshot() -> None:
    """Append today's per-team probabilities to live tracking CSV."""
    per_team_csv = REPO_ROOT / "reports" / "wc2026_per_team.csv"
    if not per_team_csv.exists():
        print(f"!!! missing {per_team_csv}, skipping tracking append")
        return
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    df = pd.read_csv(per_team_csv)
    df["snapshot_date"] = today
    df = df[[
        "snapshot_date", "team", "group",
        "p_first", "p_second", "p_qualified_r32",
        "p_round_of_16", "p_quarterfinal", "p_semifinal", "p_final", "p_winner",
    ]]
    tracking_csv = REPO_ROOT / "reports" / "wc2026_live_tracking.csv"
    if tracking_csv.exists():
        existing = pd.read_csv(tracking_csv)
        existing = existing[existing["snapshot_date"] != today]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(tracking_csv, index=False, float_format="%.4f")
    print(f"Appended {len(df) if not tracking_csv.exists() else 48} rows to {tracking_csv} for {today}")


def main() -> None:
    print(f"=== Daily workflow {datetime.now().isoformat()} ===")
    run([sys.executable, "-m", "mondiali.cli.main", "ingest", "--force"])
    run([sys.executable, "-m", "mondiali.cli.main", "update-state"])
    run([sys.executable, "scripts/predict_wc2026_full.py"])
    append_tracking_snapshot()
    print("\n=== Daily workflow complete ===")


if __name__ == "__main__":
    main()
