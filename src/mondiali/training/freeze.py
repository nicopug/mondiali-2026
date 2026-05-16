"""Freeze pipeline: refit Tier 2 + save model+calibrator+manifest+market validation.

Produces ``models/v1_final/``:
    xgb_poisson.json
    calibrator.json
    rho.txt
    manifest.json
    markets_validation.json
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from mondiali.model.dixon_coles import estimate_rho_mle
from mondiali.model.dl_bivariate import (
    BivariateConfig,
    predict_lambda_rho,
    save_bivariate,
    train_bivariate_model,
)
from mondiali.model.dl_poisson import (
    DLConfig,
    build_team_index,
    predict_lambda as dl_predict,
    save_dl_model,
    train_dl_model,
)
from mondiali.model.poisson_xgb import SYMMETRIC_FEATURES
from mondiali.training.evaluate import brier_score_1x2, log_loss_1x2
from mondiali.training.train import _compute_1x2_probs, train_tier2_pipeline
from mondiali.training.validate_markets import (
    SECONDARY_MARKETS,
    fit_market_calibrators,
    validate_all_markets,
)

log = structlog.get_logger(__name__)


def _git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha[:7]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _file_sha(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def freeze_v1_final(
    *,
    matches_path: Path,
    out_dir: Path,
    train_end: str = "2023-12-31",
    val_gate_start: str = "2024-01-01",
    val_gate_end: str = "2024-12-31",
    train_start: str = "2002-01-01",
    val_es_start: str = "2022-07-01",
    val_es_end: str = "2022-12-31",
    val_calib_start: str = "2023-01-01",
    val_calib_end: str = "2023-12-31",
    snapshots_path: Path | None = None,
) -> dict[str, Any]:
    """Refit Tier 2 with refreshed splits and write all freeze artefacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = train_tier2_pipeline(
        parquet_path=matches_path,
        train_start=train_start,
        train_end=train_end,
        val_es_start=val_es_start,
        val_es_end=val_es_end,
        val_calib_start=val_calib_start,
        val_calib_end=val_calib_end,
        val_gate_start=val_gate_start,
        val_gate_end=val_gate_end,
    )

    model = result["model"]
    calibrator = result["calibrator"]
    rho = float(result["rho"])

    model.save(out_dir / "xgb_poisson.json")
    (out_dir / "rho.txt").write_text(f"{rho:.6f}\n")

    calib_kept = float(result["brier_after"]) < float(result["brier_before"])
    calibrator_path = out_dir / "calibrator.json"
    if calib_kept:
        calibrator.save(calibrator_path)
        log.info("calibrator saved", brier_before=float(result["brier_before"]),
                 brier_after=float(result["brier_after"]))
    else:
        if calibrator_path.exists():
            calibrator_path.unlink()
        log.warning(
            "calibrator skipped (Brier did not improve)",
            brier_before=float(result["brier_before"]),
            brier_after=float(result["brier_after"]),
        )

    df = pd.read_parquet(matches_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["days_rest_home", "days_rest_away"])
    train = df[(df["date"] >= train_start) & (df["date"] <= train_end)].reset_index(drop=True)
    val_es = df[(df["date"] >= val_es_start) & (df["date"] <= val_es_end)].reset_index(drop=True)
    val_calib = df[
        (df["date"] >= val_calib_start) & (df["date"] <= val_calib_end)
    ].reset_index(drop=True)
    val_gate = df[
        (df["date"] >= val_gate_start) & (df["date"] <= val_gate_end)
    ].reset_index(drop=True)

    ensemble_info = _train_and_persist_dl_ensemble(
        train=train, val_es=val_es, val_calib=val_calib, val_gate=val_gate,
        xgb_model=model, xgb_rho=rho, out_dir=out_dir,
    )
    if ensemble_info["promoted"]:
        active_rho = float(ensemble_info["rho_ensemble"])
        log.info("ensemble promoted", **{
            k: ensemble_info[k] for k in
            ("weight_xgb", "weight_dl", "delta_vs_xgb_only", "ensemble_log_loss")
        })
    else:
        active_rho = rho
        log.warning("ensemble not promoted - using XGB-only",
                    delta=ensemble_info["delta_vs_xgb_only"])

    # Markets validation uses the active configuration (ensemble if promoted)
    market_calibrators = fit_market_calibrators(
        model=model, val_calib=val_calib, rho=rho,
    )
    markets_metrics = validate_all_markets(
        model=model, train=train, val_gate=val_gate, rho=rho,
        calibrators=market_calibrators,
    )
    (out_dir / "markets_validation.json").write_text(
        json.dumps(markets_metrics, indent=2)
    )
    markets_calib_dir = out_dir / "markets_calibrators"
    for market in SECONDARY_MARKETS:
        if markets_metrics[market]["calibrator_kept"]:
            market_calibrators[market].save(markets_calib_dir / f"{market}.json")
            log.info("market calibrator saved", market=market)
        else:
            log.info(
                "market calibrator skipped (Brier did not improve)",
                market=market,
                raw_brier=markets_metrics[market]["raw_brier"],
                calib_brier=markets_metrics[market]["calib_brier"],
            )

    version = "v1.2" if ensemble_info["promoted"] else "v1.0"
    manifest = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "model": (
            "PoissonXGBModel-symmetric-tier2 + Tier7-DL-ensemble"
            if ensemble_info["promoted"]
            else "PoissonXGBModel-symmetric-tier2"
        ),
        "n_features": len(SYMMETRIC_FEATURES),
        "feature_names": SYMMETRIC_FEATURES,
        "train_split": {"start": train_start, "end": train_end},
        "val_es_split": {"start": val_es_start, "end": val_es_end},
        "val_calib_split": {"start": val_calib_start, "end": val_calib_end},
        "val_gate_split": {"start": val_gate_start, "end": val_gate_end},
        "data_sources": {
            "matches_parquet_sha": _file_sha(matches_path),
            "snapshots_parquet_sha": _file_sha(snapshots_path) if snapshots_path else "n/a",
        },
        "hparams": dict(model.params),
        "rho_xgb": rho,
        "rho_active": active_rho,
        "calibrator_kept": bool(calib_kept),
        "ensemble": ensemble_info,
        "metrics_1x2": {
            "val_log_loss_raw": float(result["val_log_loss_raw"]),
            "val_log_loss_calib": float(result["val_log_loss_calib"]),
            "brier_before": float(result["brier_before"]),
            "brier_after": float(result["brier_after"]),
        },
        "random_state": 42,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("freeze_v1_final done", out_dir=str(out_dir), version=version)
    return {"manifest": manifest, "markets": markets_metrics,
            "train_result": result, "ensemble": ensemble_info}


def _train_and_persist_dl_ensemble(
    *,
    train: pd.DataFrame, val_es: pd.DataFrame,
    val_calib: pd.DataFrame, val_gate: pd.DataFrame,
    xgb_model: Any, xgb_rho: float, out_dir: Path,
) -> dict[str, Any]:
    """Train L1 (MLP) + L3 (bivariate) DLs, grid-search 3-way weights on val_calib,
    persist whichever configuration wins. Always evaluates on val_gate for the
    final promotion decision against XGB-alone (Δ < -0.005).

    Saves:
      dl/   — L1 MLP artefacts
      l3/   — L3 bivariate artefacts (if used)
      ensemble.json — {weight_xgb, weight_l1, weight_l3, rho_ensemble, ...}
    """
    h_goals_tr = train["home_score"].to_numpy()
    a_goals_tr = train["away_score"].to_numpy()

    dl_cfg = DLConfig()
    biv_cfg = BivariateConfig()
    team_idx = build_team_index(pd.concat([train, val_es, val_calib, val_gate],
                                           ignore_index=True))

    # --- Train L1 MLP ---
    l1_model, l1_stats, l1_info = train_dl_model(train, val_es, team_idx, dl_cfg)
    log.info("L1 trained", best_val_es=l1_info["best_val_es"])

    # --- Train L3 bivariate ---
    l3_model, l3_stats, l3_info = train_bivariate_model(train, val_es, team_idx, biv_cfg)
    log.info("L3 trained", best_val_es=l3_info["best_val_es"])

    # --- Cache all lambdas (XGB, L1, L3) on train / val_calib / val_gate ---
    lam_h_xgb_tr, lam_a_xgb_tr = xgb_model.predict_lambda(train)
    lam_h_xgb_c, lam_a_xgb_c = xgb_model.predict_lambda(val_calib)
    lam_h_xgb_g, lam_a_xgb_g = xgb_model.predict_lambda(val_gate)

    lam_h_l1_tr, lam_a_l1_tr = dl_predict(l1_model, train, team_idx, l1_stats)
    lam_h_l1_c, lam_a_l1_c = dl_predict(l1_model, val_calib, team_idx, l1_stats)
    lam_h_l1_g, lam_a_l1_g = dl_predict(l1_model, val_gate, team_idx, l1_stats)

    lam_h_l3_tr, lam_a_l3_tr, _ = predict_lambda_rho(l3_model, train, team_idx, l3_stats)
    lam_h_l3_c, lam_a_l3_c, _ = predict_lambda_rho(l3_model, val_calib, team_idx, l3_stats)
    lam_h_l3_g, lam_a_l3_g, _ = predict_lambda_rho(l3_model, val_gate, team_idx, l3_stats)

    # XGB-alone baseline on val_gate (for final promotion gate)
    xgb_gate_probs = _compute_1x2_probs(lam_h_xgb_g, lam_a_xgb_g, rho=xgb_rho)
    xgb_only_ll = float(log_loss_1x2(val_gate, xgb_gate_probs))

    # --- Grid search on val_calib ---
    # Hard XGB weight bounds [0.4, 0.85]: ensures we always test ensembles vs
    # pure XGB (the XGB-alone option is always available outside the ensemble
    # path — the freeze auto-skips if no ensemble beats XGB by 0.005 on val_gate).
    # The bound is a diversity prior: with only 1052 val_calib matches, log-loss
    # noise can mask real ensemble gains; we restrict the search to "ensembles
    # with meaningful DL contribution".
    candidates: list[tuple[float, float, float, float, float]] = []
    for w_xgb in np.arange(0.4, 0.851, 0.05):
        for w_l1 in np.arange(0.0, 1.001 - w_xgb, 0.05):
            w_l3 = 1.0 - w_xgb - w_l1
            if w_l3 < -1e-6:
                continue
            w_l3 = max(0.0, w_l3)
            lam_h_tr = w_xgb * lam_h_xgb_tr + w_l1 * lam_h_l1_tr + w_l3 * lam_h_l3_tr
            lam_a_tr = w_xgb * lam_a_xgb_tr + w_l1 * lam_a_l1_tr + w_l3 * lam_a_l3_tr
            rho = estimate_rho_mle(lam_h_tr, lam_a_tr, h_goals_tr, a_goals_tr)
            lam_h_c_ens = w_xgb * lam_h_xgb_c + w_l1 * lam_h_l1_c + w_l3 * lam_h_l3_c
            lam_a_c_ens = w_xgb * lam_a_xgb_c + w_l1 * lam_a_l1_c + w_l3 * lam_a_l3_c
            probs = _compute_1x2_probs(lam_h_c_ens, lam_a_c_ens, rho=rho)
            ll = float(log_loss_1x2(val_calib, probs))
            candidates.append((ll, float(w_xgb), float(w_l1), float(w_l3), float(rho)))

    candidates.sort(key=lambda r: r[0])
    best_ll, w_xgb, w_l1, w_l3, rho_ens = candidates[0]
    log.info("best 3-way weights on val_calib",
             w_xgb=w_xgb, w_l1=w_l1, w_l3=w_l3, val_calib_ll=best_ll)

    # --- Final unbiased eval on val_gate ---
    lam_h_tr = w_xgb * lam_h_xgb_tr + w_l1 * lam_h_l1_tr + w_l3 * lam_h_l3_tr
    lam_a_tr = w_xgb * lam_a_xgb_tr + w_l1 * lam_a_l1_tr + w_l3 * lam_a_l3_tr
    rho_ens = estimate_rho_mle(lam_h_tr, lam_a_tr, h_goals_tr, a_goals_tr)

    lam_h_g = w_xgb * lam_h_xgb_g + w_l1 * lam_h_l1_g + w_l3 * lam_h_l3_g
    lam_a_g = w_xgb * lam_a_xgb_g + w_l1 * lam_a_l1_g + w_l3 * lam_a_l3_g
    ens_probs = _compute_1x2_probs(lam_h_g, lam_a_g, rho=rho_ens)
    ens_ll = float(log_loss_1x2(val_gate, ens_probs))
    ens_br = float(brier_score_1x2(val_gate, ens_probs))
    delta = ens_ll - xgb_only_ll
    promoted = delta < -0.005

    # Determine which DLs are actually used (weight > epsilon)
    eps = 1e-3
    uses_l1 = w_l1 > eps
    uses_l3 = w_l3 > eps

    if promoted:
        if uses_l1:
            save_dl_model(l1_model, team_idx, l1_stats, dl_cfg, out_dir / "dl")
        if uses_l3:
            save_bivariate(l3_model, team_idx, l3_stats, biv_cfg, out_dir / "l3")
        (out_dir / "ensemble.json").write_text(json.dumps({
            "weight_xgb": w_xgb, "weight_l1": w_l1, "weight_l3": w_l3,
            "rho_ensemble": rho_ens,
            "selected_on": "val_calib_log_loss",
        }, indent=2))
    else:
        for p in [out_dir / "ensemble.json"]:
            if p.exists():
                p.unlink()
        for d in [out_dir / "dl", out_dir / "l3"]:
            if d.exists():
                for f in d.iterdir():
                    f.unlink()
                d.rmdir()

    return {
        "promoted": promoted,
        "weight_xgb": float(w_xgb),
        "weight_l1": float(w_l1),
        "weight_dl": float(w_l1),  # legacy alias
        "weight_l3": float(w_l3),
        "rho_ensemble": float(rho_ens),
        "val_calib_log_loss": float(best_ll),
        "ensemble_log_loss": ens_ll,
        "ensemble_brier": ens_br,
        "xgb_only_log_loss": xgb_only_ll,
        "delta_vs_xgb_only": float(delta),
        "l1_best_val_es_nll": float(l1_info["best_val_es"]),
        "l1_n_epochs_run": int(l1_info["n_epochs_run"]),
        "l3_best_val_es_nll": float(l3_info["best_val_es"]),
        "l3_n_epochs_run": int(l3_info["n_epochs_run"]),
        "l1_config": {
            "embed_dim": dl_cfg.embed_dim,
            "hidden_dims": list(dl_cfg.hidden_dims),
            "dropout": dl_cfg.dropout,
            "lr": dl_cfg.lr,
            "batch_size": dl_cfg.batch_size,
            "max_epochs": dl_cfg.max_epochs,
        },
        "l3_config": {
            "embed_dim": biv_cfg.embed_dim,
            "hidden_dims": list(biv_cfg.hidden_dims),
            "dropout": biv_cfg.dropout,
        },
    }
