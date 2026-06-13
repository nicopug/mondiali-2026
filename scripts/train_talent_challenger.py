"""Train the talent challenger, compare raw val_gate log-loss vs v1.4, report.

Apples-to-apples: same splits, same DC+Poisson stack; only the 2 talent
features differ. v1.4 baseline is computed fresh on the same val_gate 2024
(XGB-only raw), so the comparison isolates the talent features.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from mondiali.config import CONFIG
from mondiali.model.poisson_xgb import PoissonXGBModel
from mondiali.training.evaluate import brier_score_1x2, log_loss_1x2
from mondiali.training.train import _compute_1x2_probs, train_talent_challenger

GATE_MARGIN = 0.003
VAL_GATE = ("2024-01-01", "2024-12-31")


def _v1_raw_logloss_on_gate(parquet: Path) -> tuple[float, float]:
    df = pd.read_parquet(parquet)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["days_rest_home", "days_rest_away"]).copy()
    gate = df[(df["date"] >= VAL_GATE[0]) & (df["date"] <= VAL_GATE[1])].reset_index(drop=True)
    xgb = PoissonXGBModel().load(CONFIG.models_dir / "v1_final" / "xgb_poisson.json")
    rho = float((CONFIG.models_dir / "v1_final" / "rho.txt").read_text().strip())
    lam_h, lam_a = xgb.predict_lambda(gate)
    probs = _compute_1x2_probs(lam_h, lam_a, rho=rho)
    return float(log_loss_1x2(gate, probs)), float(brier_score_1x2(gate, probs))


def main() -> None:
    parquet = CONFIG.data_processed / "matches.parquet"
    res = train_talent_challenger(parquet_path=parquet)

    ch_ll = float(res["val_log_loss_raw"])
    ch_br = float(res["brier_before"])
    v1_ll, v1_br = _v1_raw_logloss_on_gate(parquet)
    delta = ch_ll - v1_ll

    out_dir = CONFIG.models_dir / "challenger_talent"
    res["model"].save(out_dir / "xgb_poisson.json")
    (out_dir / "rho.txt").write_text(f"{res['rho']:.6f}\n")

    if delta <= -GATE_MARGIN:
        verdict = "PROMOTE to v2 candidate (kept on branch; not into v1_final during freeze)"
    elif delta >= GATE_MARGIN:
        verdict = "REJECT (no improvement)"
    else:
        verdict = f"NO DECISION (|delta| < {GATE_MARGIN}) — review Brier"

    lines = [
        "# STEP 9 — Talent challenger validation",
        "",
        f"**Date:** {date.today().isoformat()}  ",
        "**Challenger:** Tier 2 stack + 2 talent differential features  ",
        f"**Splits:** train<=2023, val_calib 2023, val_gate {VAL_GATE[0]}..{VAL_GATE[1]}  ",
        "",
        "## val_gate 2024 (raw 1X2)",
        "",
        "| Model | log-loss | Brier |",
        "|---|---|---|",
        f"| v1.4 (XGB-only) | {v1_ll:.4f} | {v1_br:.4f} |",
        f"| Challenger (+talent) | {ch_ll:.4f} | {ch_br:.4f} |",
        f"| **Delta** | **{delta:+.4f}** | {ch_br - v1_br:+.4f} |",
        "",
        f"Gate margin: {GATE_MARGIN}. **Verdict: {verdict}**",
        "",
        "Artefacts: `models/challenger_talent/` (NOT promoted to v1_final).",
        "",
    ]
    Path("reports/validation_step9.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-8:]))
    print(f"\nDelta vs v1.4 raw log-loss: {delta:+.4f}  ->  {verdict}")


if __name__ == "__main__":
    main()
