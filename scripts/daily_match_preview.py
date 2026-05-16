"""Generate markdown match preview for a specified date.

Reads `data/wc2026/groups_template.json` to find matches that would naturally
happen on a given date (heuristic: assigns dates to group matches starting
from start_date, 4 matches/day across all groups). Predicts each and outputs
a markdown report.

Usage:
    python scripts/daily_match_preview.py 2026-06-15
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from mondiali.config import CONFIG
from mondiali.inference.predict import BatchPredictor


def _generate_match_calendar(groups: dict[str, list[str]], start: pd.Timestamp) -> list[dict]:
    """Spread 72 group matches across 24 days (3 per group, 6 per group total).

    WC2026 schedule has 4 matchdays in group stage (each team plays 3). We
    use a simple round-robin scheduler: matchday k assigns one match per group.
    """
    rows = []
    for group_letter, teams in groups.items():
        pairings = list(combinations(teams, 2))  # 6 per group
        for k, (a, b) in enumerate(pairings):
            day_offset = (k * 12 + ord(group_letter) - ord("A")) // 4
            rows.append({
                "match_date": (start + pd.Timedelta(days=day_offset)).strftime("%Y-%m-%d"),
                "group": group_letter, "team_a": a, "team_b": b,
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("date", help="YYYY-MM-DD")
    ap.add_argument("--config", default="data/wc2026/groups_template.json")
    args = ap.parse_args()
    target = pd.Timestamp(args.date)

    cfg = json.loads(Path(args.config).read_text())
    start = pd.Timestamp(cfg["start_date"])
    comp_imp = float(cfg["competition_importance"])

    cal = _generate_match_calendar(cfg["groups"], start)
    today = [r for r in cal if r["match_date"] == args.date]
    if not today:
        print(f"No matches scheduled for {args.date}")
        print(f"Tournament start: {cfg['start_date']}")
        return

    model_dir = CONFIG.models_dir / "v1_final"
    state_dir = CONFIG.project_root / "data" / "state"
    snaps_path = CONFIG.data_raw / "transfermarkt" / "snapshots.parquet"
    bp = BatchPredictor(model_dir, state_dir, snaps_path)

    rows = []
    for m in today:
        row = bp.build_row(
            m["team_a"], m["team_b"], target,
            neutral=True, competition_importance=comp_imp,
        )
        lh, la, _ = bp.predict_lambdas(row)
        rows.append({**m, "lam_h": float(lh[0]), "lam_a": float(la[0])})

    # Use full predict_match for markets
    from mondiali.inference.predict import predict_match
    lines = [
        f"# WC2026 — Match preview for {args.date}",
        "",
        f"**Model:** v1_final  ",
        f"**competition_importance:** {comp_imp}  ",
        f"**Neutral venue:** true",
        "",
    ]
    for m in today:
        pred = predict_match(
            home=m["team_a"], away=m["team_b"], date=target, neutral=True,
            state_dir=state_dir, model_dir=model_dir,
            tm_snapshots_path=snaps_path, competition_importance=comp_imp,
        )
        ll = pred["markets"]["1x2"]
        lh, la = pred["lambda"]["home"], pred["lambda"]["away"]
        favorite = max(ll, key=ll.get) if isinstance(ll, dict) else None
        lines.append(f"## Group {m['group']}: {m['team_a']} vs {m['team_b']}")
        lines.append("")
        lines.append(
            f"- λ {m['team_a']}: {lh:.2f}, λ {m['team_b']}: {la:.2f}"
        )
        lines.append(
            f"- **{m['team_a']}** win: {ll['home']*100:.1f}% | "
            f"draw: {ll['draw']*100:.1f}% | **{m['team_b']}** win: {ll['away']*100:.1f}%"
        )
        lines.append(
            f"- O/U 2.5: {pred['markets']['over_under_2_5']['over']*100:.0f}% over | "
            f"BTTS: {pred['markets']['btts']['yes']*100:.0f}% yes"
        )
        lines.append("")

    out = Path(f"reports/wc2026_preview_{args.date}.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out} ({len(today)} matches)")


if __name__ == "__main__":
    main()
