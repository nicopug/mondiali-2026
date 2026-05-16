"""Live match result tracker.

Usage:
    python scripts/log_match_result.py <home> <away> <home_score> <away_score>

For each completed WC2026 match: record the model's pre-match prediction +
the actual outcome + log-loss. Builds a running ledger of predictive
performance during the tournament.

Output: reports/wc2026_live_results.csv (append-only)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mondiali.config import CONFIG
from mondiali.inference.predict import predict_match


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("home")
    ap.add_argument("away")
    ap.add_argument("home_score", type=int)
    ap.add_argument("away_score", type=int)
    ap.add_argument("--date", default=None, help="YYYY-MM-DD; default: today")
    ap.add_argument("--neutral", action="store_true", default=True)
    ap.add_argument("--competition-importance", type=float, default=75.0)
    args = ap.parse_args()

    date = pd.Timestamp(args.date) if args.date else pd.Timestamp.utcnow().normalize()
    state_dir = CONFIG.project_root / "data" / "state"
    model_dir = CONFIG.models_dir / "v1_final"
    snaps_path = CONFIG.data_raw / "transfermarkt" / "snapshots.parquet"

    pred = predict_match(
        home=args.home, away=args.away, date=date, neutral=args.neutral,
        state_dir=state_dir, model_dir=model_dir,
        tm_snapshots_path=snaps_path,
        competition_importance=args.competition_importance,
    )

    # Determine actual outcome
    if args.home_score > args.away_score:
        actual_1x2 = "H"; p_actual = pred["markets"]["1x2"]["home"]
    elif args.home_score < args.away_score:
        actual_1x2 = "A"; p_actual = pred["markets"]["1x2"]["away"]
    else:
        actual_1x2 = "D"; p_actual = pred["markets"]["1x2"]["draw"]
    log_loss = -np.log(max(p_actual, 1e-9))

    total = args.home_score + args.away_score
    actual_o15 = int(total > 1.5)
    actual_o25 = int(total > 2.5)
    actual_o35 = int(total > 3.5)
    actual_btts = int(args.home_score > 0 and args.away_score > 0)

    row = {
        "logged_at": datetime.utcnow().isoformat(),
        "match_date": str(date.date()),
        "home": args.home, "away": args.away,
        "home_score": args.home_score, "away_score": args.away_score,
        "model_version": pred["model_version"],
        "ensemble": pred["ensemble"],
        "lambda_home": pred["lambda"]["home"],
        "lambda_away": pred["lambda"]["away"],
        "p_home": pred["markets"]["1x2"]["home"],
        "p_draw": pred["markets"]["1x2"]["draw"],
        "p_away": pred["markets"]["1x2"]["away"],
        "p_over_2_5": pred["markets"]["over_under_2_5"]["over"],
        "p_btts": pred["markets"]["btts"]["yes"],
        "actual_1x2": actual_1x2,
        "actual_o15": actual_o15, "actual_o25": actual_o25, "actual_o35": actual_o35,
        "actual_btts": actual_btts,
        "p_actual_1x2": p_actual,
        "log_loss_1x2": log_loss,
    }
    out_csv = Path("reports/wc2026_live_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        df = pd.read_csv(out_csv)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(out_csv, index=False)
    print(f"Logged {args.home} {args.home_score}-{args.away_score} {args.away}")
    print(f"  P(predicted)={p_actual:.3f}  log_loss={log_loss:.4f}")
    print(f"  Cumulative: {len(df)} matches, mean log_loss = {df['log_loss_1x2'].mean():.4f}")


if __name__ == "__main__":
    main()
