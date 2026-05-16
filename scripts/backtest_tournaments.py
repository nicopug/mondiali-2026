"""Walk-forward backtest of v1_final on WC2022 + Euro2024.

For each tournament match, builds state from history strictly before that
match, predicts all 5 markets, and compares against actual outcomes.

⚠ These tournaments fall partly inside the training window (WC2022 in val_es,
Euro2024 in val_gate). This is NOT an unbiased out-of-sample evaluation — it
is a sanity-check / demonstration of model behaviour on famous matches.

Outputs:
- ``reports/backtest_predictions.csv`` — per-match raw predictions + outcomes
- ``reports/backtest_wc2022_euro2024.md`` — aggregated metrics + notable matches
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from mondiali.inference.predict import build_inference_row
from mondiali.inference.state import _build_elo_state, _build_form_cache
from mondiali.model.calibration import BinaryMarketCalibrator, IsotonicCalibrator1X2
from mondiali.model.dixon_coles import dixon_coles_correct, joint_matrix
from mondiali.model.markets import prob_1x2, prob_btts, prob_over_under
from mondiali.model.poisson_xgb import PoissonXGBModel

log = structlog.get_logger(__name__)

MODEL_DIR = Path("models/v1_final")
TOURNAMENTS = {
    "WC2022": {"start": "2022-11-20", "end": "2022-12-18", "key": "FIFA World Cup"},
    "Euro2024": {"start": "2024-06-14", "end": "2024-07-14", "key": "UEFA Euro"},
}


def _load_artefacts() -> tuple[PoissonXGBModel, IsotonicCalibrator1X2 | None,
                                dict[str, BinaryMarketCalibrator], float]:
    model = PoissonXGBModel().load(MODEL_DIR / "xgb_poisson.json")
    rho = float((MODEL_DIR / "rho.txt").read_text().strip())
    cal_1x2 = None
    cal_path = MODEL_DIR / "calibrator.json"
    if cal_path.exists():
        cal_1x2 = IsotonicCalibrator1X2.load(cal_path)
    market_cals: dict[str, BinaryMarketCalibrator] = {}
    md_dir = MODEL_DIR / "markets_calibrators"
    if md_dir.exists():
        for f in md_dir.glob("*.json"):
            market_cals[f.stem] = BinaryMarketCalibrator.load(f)
    return model, cal_1x2, market_cals, rho


def _predict_single(
    row: pd.DataFrame, model: PoissonXGBModel, cal_1x2: IsotonicCalibrator1X2 | None,
    market_cals: dict[str, BinaryMarketCalibrator], rho: float,
) -> dict:
    lam_h_arr, lam_a_arr = model.predict_lambda(row)
    lam_h, lam_a = float(lam_h_arr[0]), float(lam_a_arr[0])
    joint = joint_matrix(lam_h, lam_a)
    joint = dixon_coles_correct(joint, lam_h, lam_a, rho)
    p_h, p_d, p_a = prob_1x2(joint)
    if cal_1x2 is not None:
        c = cal_1x2.predict(np.array([[p_h, p_d, p_a]]))[0]
        p_h, p_d, p_a = float(c[0]), float(c[1]), float(c[2])
    over15, _ = prob_over_under(joint, threshold=1.5)
    over25, _ = prob_over_under(joint, threshold=2.5)
    over35, _ = prob_over_under(joint, threshold=3.5)
    btts_yes, _ = prob_btts(joint)
    raws = {
        "over_under_1_5": over15, "over_under_2_5": over25,
        "over_under_3_5": over35, "btts": btts_yes,
    }
    out = {"lam_h": lam_h, "lam_a": lam_a,
           "p_home": p_h, "p_draw": p_d, "p_away": p_a}
    for market, raw_p in raws.items():
        if market in market_cals:
            out[f"p_{market}"] = float(market_cals[market].predict(np.array([raw_p]))[0])
        else:
            out[f"p_{market}"] = float(raw_p)
    return out


def run_backtest(out_csv: Path) -> pd.DataFrame:
    matches = pd.read_parquet("data/processed/matches.parquet")
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.dropna(subset=["days_rest_home", "days_rest_away"])
    snapshots = pd.read_parquet("data/raw/transfermarkt/snapshots.parquet")

    model, cal_1x2, market_cals, rho = _load_artefacts()

    targets: list[pd.DataFrame] = []
    for name, t in TOURNAMENTS.items():
        sel = matches[
            (matches["date"] >= t["start"]) & (matches["date"] <= t["end"])
            & matches["tournament"].str.contains(t["key"], case=False, na=False)
        ].copy()
        sel["tournament_label"] = name
        targets.append(sel)
    target_df = pd.concat(targets, ignore_index=True).sort_values("date").reset_index(drop=True)
    log.info("backtest start", n_matches=len(target_df))

    rows = []
    for i, m in target_df.iterrows():
        history = matches[matches["date"] < m["date"]]
        elo = _build_elo_state(history)
        form = _build_form_cache(history)
        inf_row = build_inference_row(
            home=m["home_team"], away=m["away_team"],
            date=pd.Timestamp(m["date"]), neutral=bool(m["neutral"]),
            elo_state=elo, form_cache=form, tm_snapshots=snapshots,
            competition_importance=float(m["competition_importance"]),
        )
        pred = _predict_single(inf_row, model, cal_1x2, market_cals, rho)
        actual_total = int(m["home_score"]) + int(m["away_score"])
        actual = {
            "actual_home": int(m["home_score"]),
            "actual_away": int(m["away_score"]),
            "actual_1x2": (
                "H" if m["home_score"] > m["away_score"]
                else "D" if m["home_score"] == m["away_score"]
                else "A"
            ),
            "actual_o15": 1 if actual_total > 1.5 else 0,
            "actual_o25": 1 if actual_total > 2.5 else 0,
            "actual_o35": 1 if actual_total > 3.5 else 0,
            "actual_btts": 1 if (m["home_score"] > 0 and m["away_score"] > 0) else 0,
        }
        rows.append({
            "tournament": m["tournament_label"], "date": m["date"],
            "home": m["home_team"], "away": m["away_team"],
            "neutral": bool(m["neutral"]),
            **pred, **actual,
        })
        if i % 20 == 0:
            log.info("backtest progress", i=i, total=len(target_df))

    out_df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    log.info("backtest predictions written", path=str(out_csv), n=len(out_df))
    return out_df


def _metrics(df: pd.DataFrame) -> dict:
    eps = 1e-9
    y_h = (df["actual_1x2"] == "H").astype(int).to_numpy()
    y_d = (df["actual_1x2"] == "D").astype(int).to_numpy()
    y_a = (df["actual_1x2"] == "A").astype(int).to_numpy()
    p_h = df["p_home"].to_numpy()
    p_d = df["p_draw"].to_numpy()
    p_a = df["p_away"].to_numpy()
    p_chosen = np.where(y_h == 1, p_h, np.where(y_d == 1, p_d, p_a))
    ll_1x2 = float(-np.log(np.clip(p_chosen, eps, 1.0)).mean())
    brier_1x2 = float(((p_h - y_h) ** 2 + (p_d - y_d) ** 2 + (p_a - y_a) ** 2).mean())
    metrics = {"1x2": {"log_loss": ll_1x2, "brier": brier_1x2}}
    for market, actual_col in (
        ("over_under_1_5", "actual_o15"),
        ("over_under_2_5", "actual_o25"),
        ("over_under_3_5", "actual_o35"),
        ("btts", "actual_btts"),
    ):
        y = df[actual_col].to_numpy()
        p = df[f"p_{market}"].to_numpy()
        p_clip = np.clip(p, eps, 1.0 - eps)
        ll = float(-(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip)).mean())
        br = float(((p - y) ** 2).mean())
        metrics[market] = {"log_loss": ll, "brier": br}
    return metrics


def _notable_matches(df: pd.DataFrame, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (n best correct calls, n worst surprises) sorted by 1X2 log-loss."""
    eps = 1e-9
    y_h = (df["actual_1x2"] == "H").astype(int).to_numpy()
    y_d = (df["actual_1x2"] == "D").astype(int).to_numpy()
    y_a = (df["actual_1x2"] == "A").astype(int).to_numpy()
    p_chosen = np.where(y_h == 1, df["p_home"].to_numpy(),
                        np.where(y_d == 1, df["p_draw"].to_numpy(),
                                 df["p_away"].to_numpy()))
    ll = -np.log(np.clip(p_chosen, eps, 1.0))
    df = df.copy()
    df["match_log_loss"] = ll
    df["p_actual"] = p_chosen
    best = df.nsmallest(n, "match_log_loss")
    worst = df.nlargest(n, "match_log_loss")
    return best, worst


def _format_match_row(r: pd.Series) -> str:
    return (
        f"- **{r['date'].strftime('%Y-%m-%d')} {r['home']} {r['actual_home']}–"
        f"{r['actual_away']} {r['away']}** "
        f"({r['tournament']}) — "
        f"P(H/D/A)={r['p_home']:.2f}/{r['p_draw']:.2f}/{r['p_away']:.2f}, "
        f"P(actual)={r['p_actual']:.2f}"
    )


def write_report(df: pd.DataFrame, out_md: Path) -> None:
    overall = _metrics(df)
    wc22 = _metrics(df[df["tournament"] == "WC2022"])
    eu24 = _metrics(df[df["tournament"] == "Euro2024"])
    eps = 1e-9
    y_h = (df["actual_1x2"] == "H").astype(int).to_numpy()
    y_d = (df["actual_1x2"] == "D").astype(int).to_numpy()
    p_actual_all = np.where(y_h == 1, df["p_home"].to_numpy(),
                            np.where(y_d == 1, df["p_draw"].to_numpy(),
                                     df["p_away"].to_numpy()))
    df = df.copy()
    df["p_actual"] = p_actual_all
    df["match_log_loss"] = -np.log(np.clip(p_actual_all, eps, 1.0))
    best = df.nsmallest(5, "match_log_loss")
    worst = df.nlargest(5, "match_log_loss")

    lines = [
        "# Backtest — WC2022 + Euro2024",
        "",
        "**Date generated:** 2026-05-16  ",
        "**Model:** `models/v1_final/` (v1.0)  ",
        f"**Matches:** {len(df)} (WC2022={len(df[df['tournament']=='WC2022'])}, "
        f"Euro2024={len(df[df['tournament']=='Euro2024'])})",
        "",
        "## ⚠ Important caveat",
        "",
        "WC2022 falls in the **val_es** window (2022-07-01 → 2022-12-31, used for "
        "XGBoost early stopping) and Euro2024 falls in the **val_gate** window. The "
        "model has seen both distributions during training. This backtest is a "
        "demonstration of model behaviour on famous matches, **not** an unbiased "
        "out-of-sample evaluation.",
        "",
        "Walk-forward state: for each match, Elo + form-cache are rebuilt from "
        "matches strictly before `match_date`. No future leakage in the state.",
        "",
        "## Aggregate metrics",
        "",
        "| Market | All (115) | WC2022 (64) | Euro2024 (51) |",
        "|---|---|---|---|",
    ]
    for market in ("1x2", "over_under_1_5", "over_under_2_5", "over_under_3_5", "btts"):
        lines.append(
            f"| {market} | ll={overall[market]['log_loss']:.3f} "
            f"br={overall[market]['brier']:.3f} | "
            f"ll={wc22[market]['log_loss']:.3f} br={wc22[market]['brier']:.3f} | "
            f"ll={eu24[market]['log_loss']:.3f} br={eu24[market]['brier']:.3f} |"
        )

    lines.extend([
        "",
        "## 5 best calls (model gave highest probability to the actual outcome)",
        "",
    ])
    for _, r in best.iterrows():
        lines.append(_format_match_row(r))

    lines.extend([
        "",
        "## 5 biggest misses (model assigned lowest probability to the actual outcome)",
        "",
    ])
    for _, r in worst.iterrows():
        lines.append(_format_match_row(r))

    lines.extend([
        "",
        "## Notable matches (finals + key knockouts)",
        "",
    ])
    notable_dates = ["2022-12-18", "2024-07-14",  # finals
                     "2022-12-13", "2022-12-14",  # WC22 semifinals
                     "2024-07-09", "2024-07-10"]  # Euro24 semifinals
    for d in notable_dates:
        matches_on = df[df["date"] == pd.Timestamp(d)]
        for _, r in matches_on.iterrows():
            lines.append(_format_match_row(r))

    out_md.write_text("\n".join(lines), encoding="utf-8")
    log.info("backtest report written", path=str(out_md))

    aggregate_path = out_md.parent / "backtest_metrics.json"
    aggregate_path.write_text(json.dumps({
        "overall": overall, "wc2022": wc22, "euro2024": eu24,
    }, indent=2))


if __name__ == "__main__":
    csv_path = Path("reports/backtest_predictions.csv")
    md_path = Path("reports/backtest_wc2022_euro2024.md")
    df = run_backtest(csv_path)
    write_report(df, md_path)
    print(f"Wrote {csv_path} and {md_path}")
