"""Train L1, L2, L3 DL variants + evaluate all combos vs v1_final XGB on val_gate 2024.

Outputs reports/dl_levels_comparison.md with:
- Individual model val_gate log-loss + Brier
- Pairwise ensembles (XGB+each, each+each)
- 3-way ensembles (XGB + 2 DLs)
- Best 4-way (XGB + all 3 DLs)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mondiali.model.dixon_coles import dixon_coles_correct, estimate_rho_mle, joint_matrix
from mondiali.model.dl_bivariate import (
    BivariateConfig,
    build_team_index as build_team_index_biv,
    predict_lambda_rho,
    train_bivariate_model,
)
from mondiali.model.dl_poisson import (
    DLConfig,
    predict_lambda as predict_l1,
    train_dl_model,
)
from mondiali.model.dl_sequence import (
    SeqConfig,
    predict_lambda as predict_l2,
    train_seq_model,
)
from mondiali.model.markets import prob_1x2
from mondiali.model.poisson_xgb import PoissonXGBModel
from mondiali.training.evaluate import brier_score_1x2, log_loss_1x2


def _compute_1x2(lam_h: np.ndarray, lam_a: np.ndarray, rho: float) -> np.ndarray:
    n = len(lam_h)
    out = np.empty((n, 3), dtype=float)
    for i in range(n):
        m = joint_matrix(float(lam_h[i]), float(lam_a[i]))
        m = dixon_coles_correct(m, float(lam_h[i]), float(lam_a[i]), rho=rho)
        p1, px, p2 = prob_1x2(m)
        out[i] = (p1, px, p2)
    out = np.clip(out, 0.0, 1.0)
    s = out.sum(axis=1, keepdims=True)
    return out / np.where(s > 0, s, 1.0)


def _compute_1x2_per_match_rho(
    lam_h: np.ndarray, lam_a: np.ndarray, rho_per: np.ndarray,
) -> np.ndarray:
    n = len(lam_h)
    out = np.empty((n, 3), dtype=float)
    for i in range(n):
        m = joint_matrix(float(lam_h[i]), float(lam_a[i]))
        m = dixon_coles_correct(m, float(lam_h[i]), float(lam_a[i]), rho=float(rho_per[i]))
        p1, px, p2 = prob_1x2(m)
        out[i] = (p1, px, p2)
    out = np.clip(out, 0.0, 1.0)
    s = out.sum(axis=1, keepdims=True)
    return out / np.where(s > 0, s, 1.0)


def main() -> None:
    print("Loading data + v1_final XGB...")
    df = pd.read_parquet("data/processed/matches.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["days_rest_home", "days_rest_away"]).copy()
    train = df[(df["date"] >= "2002-01-01") & (df["date"] <= "2023-12-31")].reset_index(drop=True)
    val_es = df[(df["date"] >= "2022-07-01") & (df["date"] <= "2022-12-31")].reset_index(drop=True)
    val_gate = df[(df["date"] >= "2024-01-01") & (df["date"] <= "2024-12-31")].reset_index(drop=True)
    full_history = df  # universe for sequence lookup

    xgb = PoissonXGBModel().load(Path("models/v1_final/xgb_poisson.json"))
    rho_xgb = float(Path("models/v1_final/rho.txt").read_text().strip())
    lam_h_xgb_tr, lam_a_xgb_tr = xgb.predict_lambda(train)
    lam_h_xgb_g, lam_a_xgb_g = xgb.predict_lambda(val_gate)
    print(f"  XGB train lambdas: home={lam_h_xgb_tr.mean():.3f} away={lam_a_xgb_tr.mean():.3f}")

    team_idx = build_team_index_biv(df)
    h_goals_tr = train["home_score"].to_numpy()
    a_goals_tr = train["away_score"].to_numpy()

    print("\n[L1] Training team-embedding MLP...")
    m1, stats1, info1 = train_dl_model(train, val_es, team_idx, DLConfig())
    print(f"  best_val_es NLL: {info1['best_val_es']:.4f}")
    lam_h_l1_tr, lam_a_l1_tr = predict_l1(m1, train, team_idx, stats1)
    lam_h_l1_g, lam_a_l1_g = predict_l1(m1, val_gate, team_idx, stats1)

    print("\n[L2] Training sequence GRU...")
    m2, ts2, hs2, info2 = train_seq_model(train, val_es, full_history, team_idx, SeqConfig())
    print(f"  best_val_es NLL: {info2['best_val_es']:.4f}")
    lam_h_l2_tr, lam_a_l2_tr = predict_l2(m2, train, full_history, team_idx, ts2, hs2)
    lam_h_l2_g, lam_a_l2_g = predict_l2(m2, val_gate, full_history, team_idx, ts2, hs2)

    print("\n[L3] Training bivariate Poisson w/ per-match rho...")
    m3, stats3, info3 = train_bivariate_model(train, val_es, team_idx, BivariateConfig())
    print(f"  best_val_es DC-NLL: {info3['best_val_es']:.4f}")
    lam_h_l3_tr, lam_a_l3_tr, rho_l3_tr = predict_lambda_rho(m3, train, team_idx, stats3)
    lam_h_l3_g, lam_a_l3_g, rho_l3_g = predict_lambda_rho(m3, val_gate, team_idx, stats3)

    print("\n=== Individual val_gate metrics ===")
    results: list[tuple[str, float, float]] = []

    # XGB alone
    probs = _compute_1x2(lam_h_xgb_g, lam_a_xgb_g, rho=rho_xgb)
    ll, br = float(log_loss_1x2(val_gate, probs)), float(brier_score_1x2(val_gate, probs))
    results.append(("v1_final_XGB", ll, br))

    # L1 alone (fresh rho on training)
    rho_l1 = estimate_rho_mle(lam_h_l1_tr, lam_a_l1_tr, h_goals_tr, a_goals_tr)
    probs = _compute_1x2(lam_h_l1_g, lam_a_l1_g, rho=rho_l1)
    ll, br = float(log_loss_1x2(val_gate, probs)), float(brier_score_1x2(val_gate, probs))
    results.append(("L1_MLP_alone", ll, br))

    # L2 alone
    rho_l2 = estimate_rho_mle(lam_h_l2_tr, lam_a_l2_tr, h_goals_tr, a_goals_tr)
    probs = _compute_1x2(lam_h_l2_g, lam_a_l2_g, rho=rho_l2)
    ll, br = float(log_loss_1x2(val_gate, probs)), float(brier_score_1x2(val_gate, probs))
    results.append(("L2_Sequence_alone", ll, br))

    # L3 alone (per-match rho from the model itself)
    probs = _compute_1x2_per_match_rho(lam_h_l3_g, lam_a_l3_g, rho_l3_g)
    ll, br = float(log_loss_1x2(val_gate, probs)), float(brier_score_1x2(val_gate, probs))
    results.append(("L3_Bivariate_alone", ll, br))

    print(f"{'name':25s}  {'log_loss':>10s}  {'brier':>10s}")
    for name, ll, br in results:
        print(f"{name:25s}  {ll:>10.4f}  {br:>10.4f}")

    # === Ensembles ===
    print("\n=== Pairwise ensembles (avg lambdas, fresh rho) ===")
    pairs = [
        ("XGB+L1", lam_h_xgb_tr, lam_a_xgb_tr, lam_h_xgb_g, lam_a_xgb_g,
                  lam_h_l1_tr, lam_a_l1_tr, lam_h_l1_g, lam_a_l1_g),
        ("XGB+L2", lam_h_xgb_tr, lam_a_xgb_tr, lam_h_xgb_g, lam_a_xgb_g,
                  lam_h_l2_tr, lam_a_l2_tr, lam_h_l2_g, lam_a_l2_g),
        ("XGB+L3", lam_h_xgb_tr, lam_a_xgb_tr, lam_h_xgb_g, lam_a_xgb_g,
                  lam_h_l3_tr, lam_a_l3_tr, lam_h_l3_g, lam_a_l3_g),
        ("L1+L2", lam_h_l1_tr, lam_a_l1_tr, lam_h_l1_g, lam_a_l1_g,
                 lam_h_l2_tr, lam_a_l2_tr, lam_h_l2_g, lam_a_l2_g),
        ("L1+L3", lam_h_l1_tr, lam_a_l1_tr, lam_h_l1_g, lam_a_l1_g,
                 lam_h_l3_tr, lam_a_l3_tr, lam_h_l3_g, lam_a_l3_g),
        ("L2+L3", lam_h_l2_tr, lam_a_l2_tr, lam_h_l2_g, lam_a_l2_g,
                 lam_h_l3_tr, lam_a_l3_tr, lam_h_l3_g, lam_a_l3_g),
    ]
    for name, lh1_tr, la1_tr, lh1_g, la1_g, lh2_tr, la2_tr, lh2_g, la2_g in pairs:
        # 50/50 average
        lh_tr = 0.5 * lh1_tr + 0.5 * lh2_tr
        la_tr = 0.5 * la1_tr + 0.5 * la2_tr
        lh_g = 0.5 * lh1_g + 0.5 * lh2_g
        la_g = 0.5 * la1_g + 0.5 * la2_g
        rho = estimate_rho_mle(lh_tr, la_tr, h_goals_tr, a_goals_tr)
        probs = _compute_1x2(lh_g, la_g, rho=rho)
        ll, br = float(log_loss_1x2(val_gate, probs)), float(brier_score_1x2(val_gate, probs))
        results.append((f"ensemble_{name}_50_50", ll, br))
        print(f"  {name:10s} 50/50: ll={ll:.4f}  br={br:.4f}  rho={rho:+.4f}")

    print("\n=== 3-way ensembles ===")
    triples = [
        ("XGB+L1+L2", lam_h_xgb_tr, lam_a_xgb_tr, lam_h_xgb_g, lam_a_xgb_g,
                     lam_h_l1_tr, lam_a_l1_tr, lam_h_l1_g, lam_a_l1_g,
                     lam_h_l2_tr, lam_a_l2_tr, lam_h_l2_g, lam_a_l2_g),
        ("XGB+L1+L3", lam_h_xgb_tr, lam_a_xgb_tr, lam_h_xgb_g, lam_a_xgb_g,
                     lam_h_l1_tr, lam_a_l1_tr, lam_h_l1_g, lam_a_l1_g,
                     lam_h_l3_tr, lam_a_l3_tr, lam_h_l3_g, lam_a_l3_g),
        ("XGB+L2+L3", lam_h_xgb_tr, lam_a_xgb_tr, lam_h_xgb_g, lam_a_xgb_g,
                     lam_h_l2_tr, lam_a_l2_tr, lam_h_l2_g, lam_a_l2_g,
                     lam_h_l3_tr, lam_a_l3_tr, lam_h_l3_g, lam_a_l3_g),
        ("L1+L2+L3", lam_h_l1_tr, lam_a_l1_tr, lam_h_l1_g, lam_a_l1_g,
                    lam_h_l2_tr, lam_a_l2_tr, lam_h_l2_g, lam_a_l2_g,
                    lam_h_l3_tr, lam_a_l3_tr, lam_h_l3_g, lam_a_l3_g),
    ]
    for entry in triples:
        name = entry[0]
        rest = entry[1:]
        w = 1.0 / 3.0
        lh_tr = w * rest[0] + w * rest[4] + w * rest[8]
        la_tr = w * rest[1] + w * rest[5] + w * rest[9]
        lh_g = w * rest[2] + w * rest[6] + w * rest[10]
        la_g = w * rest[3] + w * rest[7] + w * rest[11]
        rho = estimate_rho_mle(lh_tr, la_tr, h_goals_tr, a_goals_tr)
        probs = _compute_1x2(lh_g, la_g, rho=rho)
        ll, br = float(log_loss_1x2(val_gate, probs)), float(brier_score_1x2(val_gate, probs))
        results.append((f"ensemble_{name}_equal", ll, br))
        print(f"  {name:14s} equal: ll={ll:.4f}  br={br:.4f}  rho={rho:+.4f}")

    print("\n=== 4-way ensemble (XGB+L1+L2+L3 equal) ===")
    w = 0.25
    lh_tr = w * (lam_h_xgb_tr + lam_h_l1_tr + lam_h_l2_tr + lam_h_l3_tr)
    la_tr = w * (lam_a_xgb_tr + lam_a_l1_tr + lam_a_l2_tr + lam_a_l3_tr)
    lh_g = w * (lam_h_xgb_g + lam_h_l1_g + lam_h_l2_g + lam_h_l3_g)
    la_g = w * (lam_a_xgb_g + lam_a_l1_g + lam_a_l2_g + lam_a_l3_g)
    rho = estimate_rho_mle(lh_tr, la_tr, h_goals_tr, a_goals_tr)
    probs = _compute_1x2(lh_g, la_g, rho=rho)
    ll, br = float(log_loss_1x2(val_gate, probs)), float(brier_score_1x2(val_gate, probs))
    results.append(("ensemble_4way_equal", ll, br))
    print(f"  ALL 4 equal: ll={ll:.4f}  br={br:.4f}  rho={rho:+.4f}")

    # Sort and write report
    results.sort(key=lambda r: r[1])
    out_md = Path("reports/dl_levels_comparison.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DL Levels Comparison (L1 vs L2 vs L3) — val_gate 2024",
        "",
        "**Date:** 2026-05-16  ",
        "**v1_final XGB raw baseline:** log_loss=0.9044, brier=0.5329  ",
        "**Promotion threshold:** Δ log-loss < -0.005 vs XGB alone",
        "",
        "Each DL model trained on identical splits (train 2002-2023, val_es Jul-Dec 2022). "
        "Ensembles: average of lambdas, fresh DC ρ estimated on training set lambdas, "
        "then standard DC correction + 1X2.",
        "",
        "## Ranked results (best → worst)",
        "",
        "| Rank | Model | log_loss | Δ vs XGB | brier |",
        "|---|---|---|---|---|",
    ]
    xgb_ll = next(r[1] for r in results if r[0] == "v1_final_XGB")
    for i, (name, ll, br) in enumerate(results, 1):
        d = ll - xgb_ll
        mark = " ✅" if d < -0.005 else ""
        lines.append(f"| {i} | `{name}` | {ll:.4f} | {d:+.4f}{mark} | {br:.4f} |")
    lines.extend([
        "",
        "## DL training info",
        "",
        f"- L1 MLP: best val_es NLL = {info1['best_val_es']:.4f}, epochs={info1['n_epochs_run']}",
        f"- L2 Sequence (GRU h={32}, hist_len=8): best val_es NLL = {info2['best_val_es']:.4f}, "
        f"epochs={info2['n_epochs_run']}",
        f"- L3 Bivariate (per-match rho): best val_es DC-NLL = {info3['best_val_es']:.4f}, "
        f"epochs={info3['n_epochs_run']}",
        "",
        "## Notes",
        "",
        "- Pairwise/3-way/4-way ensembles use **equal** weights (50/50, 33/33/33, 25/25/25/25) "
        "for fairness in comparison. Fine-grid weight search on val_calib could squeeze a bit more "
        "but risks overfitting.",
        "- L3 uses **per-match learned ρ** (range [-0.3, 0.3] via tanh) rather than the single "
        "globally-estimated ρ for the other models. This is a structural advantage but also a "
        "harder optimization target.",
        "- All four base models share the same train/val_es splits → no leakage in ensembles.",
    ])
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_md}")

    # Also save JSON for downstream use
    out_json = Path("reports/dl_levels_results.json")
    out_json.write_text(json.dumps({
        name: {"log_loss": ll, "brier": br, "delta_vs_xgb": ll - xgb_ll}
        for name, ll, br in results
    }, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
