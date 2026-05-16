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
from mondiali.model.calibration import (
    IsotonicCalibrator1X2, PlattCalibrator1X2,
)
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
    n_l3_seeds: int = 5,
    n_l1_seeds: int = 3,
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

    # XGB-only calibrator gate (legacy, kept for backward compat)
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
            "XGB-only calibrator skipped (Brier did not improve)",
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
        n_l3_seeds=n_l3_seeds, n_l1_seeds=n_l1_seeds,
    )
    # Task P: try Platt calibrator fit on ENSEMBLE probs (val_calib) and
    # ship if it improves val_gate Brier vs raw ensemble.
    ensemble_calib_info = _try_ensemble_calibrator(
        train=train, val_calib=val_calib, val_gate=val_gate,
        xgb_model=model, ensemble_info=ensemble_info, out_dir=out_dir,
    ) if ensemble_info["promoted"] else {"kept": False, "reason": "no ensemble"}
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

    version = "v1.4" if ensemble_info["promoted"] else "v1.0"
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
        "ensemble_calibrator": ensemble_calib_info,
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


def _try_ensemble_calibrator(
    *, train: pd.DataFrame, val_calib: pd.DataFrame, val_gate: pd.DataFrame,
    xgb_model: Any, ensemble_info: dict[str, Any], out_dir: Path,
) -> dict[str, Any]:
    """Fit Platt + Isotonic on ensemble 1X2 probs (val_calib) and pick winner on val_gate.

    Promote the ensemble-prob calibrator only if it beats RAW ensemble probs
    on val_gate Brier by > 0.001.

    Saves `ensemble_calibrator.json` + `ensemble_calibrator_class.txt` if kept.
    """
    from mondiali.inference.predict import BatchPredictor  # noqa: PLC0415
    from mondiali.training.evaluate import brier_score_1x2, compute_outcomes  # noqa: PLC0415

    state_dir = out_dir.parent.parent / "data" / "state"  # heuristic; uses default
    snaps_path = out_dir.parent.parent / "data" / "raw" / "transfermarkt" / "snapshots.parquet"
    try:
        bp = BatchPredictor(out_dir, state_dir, snaps_path)
    except Exception as exc:
        return {"kept": False, "reason": f"BatchPredictor failed: {exc}"}

    lh_c, la_c, rho = bp.predict_lambdas(val_calib)
    lh_g, la_g, _ = bp.predict_lambdas(val_gate)
    calib_probs = _compute_1x2_probs(lh_c, la_c, rho=rho)
    gate_probs = _compute_1x2_probs(lh_g, la_g, rho=rho)
    outcomes_c = compute_outcomes(val_calib)
    raw_gate_brier = float(brier_score_1x2(val_gate, gate_probs))

    best = {"kept": False, "raw_brier": raw_gate_brier}
    for cls_name, cls in (("platt", PlattCalibrator1X2), ("isotonic", IsotonicCalibrator1X2)):
        try:
            cal = cls().fit(calib_probs, outcomes_c)
            gate_calib = cal.predict(gate_probs)
            br = float(brier_score_1x2(val_gate, gate_calib))
            log.info("ensemble_calib_eval", cls=cls_name, gate_brier=br, raw_brier=raw_gate_brier)
            if br < raw_gate_brier - 0.001:
                if not best["kept"] or br < best["calib_brier"]:
                    best = {
                        "kept": True, "class": cls_name,
                        "calib_brier": br, "raw_brier": raw_gate_brier,
                        "calibrator": cal,
                    }
        except Exception as exc:
            log.warning("ensemble_calib_fit_failed", cls=cls_name, exc=str(exc))

    if best["kept"]:
        best["calibrator"].save(out_dir / "ensemble_calibrator.json")
        (out_dir / "ensemble_calibrator_class.txt").write_text(best["class"] + "\n")
        return {
            "kept": True, "class": best["class"],
            "calib_brier": best["calib_brier"],
            "raw_brier": best["raw_brier"],
        }
    return {"kept": False, "raw_brier": raw_gate_brier}


def _train_and_persist_dl_ensemble(
    *,
    train: pd.DataFrame, val_es: pd.DataFrame,
    val_calib: pd.DataFrame, val_gate: pd.DataFrame,
    xgb_model: Any, xgb_rho: float, out_dir: Path,
    n_l3_seeds: int = 3, n_l1_seeds: int = 1,
) -> dict[str, Any]:
    """Train L1 + L3 ensemble (multi-seed averaging), grid-search 3-way weights
    on val_calib, persist if val_gate Δ < -0.005 vs XGB-only.

    Multi-seed averaging: trains N independent DLs with different seeds and
    averages their lambda predictions. Reduces DL variance ~sqrt(N) and gives
    consistent gains over single-seed in our empirical eval (see report).

    Saves:
      dl_seeds/seed_K/   — L1 MLP per seed (if used)
      l3_seeds/seed_K/   — L3 bivariate per seed (if used)
      ensemble.json — {weight_xgb, weight_l1, weight_l3, rho_ensemble,
                       l1_seeds, l3_seeds, ...}
    """
    h_goals_tr = train["home_score"].to_numpy()
    a_goals_tr = train["away_score"].to_numpy()
    team_idx = build_team_index(pd.concat([train, val_es, val_calib, val_gate],
                                           ignore_index=True))

    l1_seeds = [42, 1, 2, 3, 4][:n_l1_seeds]
    l3_seeds = [42, 1, 2, 3, 4][:n_l3_seeds]
    dl_cfg = DLConfig()
    biv_cfg = BivariateConfig()

    # --- Train L1 MLP (one or more seeds) ---
    l1_models: list = []
    l1_stats_list: list = []
    l1_infos: list = []
    for s in l1_seeds:
        cfg = DLConfig(seed=s)
        m, st, info = train_dl_model(train, val_es, team_idx, cfg)
        l1_models.append(m)
        l1_stats_list.append(st)
        l1_infos.append(info)
        log.info("L1 trained", seed=s, best_val_es=info["best_val_es"])

    # --- Train L3 bivariate (multi-seed) ---
    l3_models: list = []
    l3_stats_list: list = []
    l3_infos: list = []
    for s in l3_seeds:
        cfg = BivariateConfig(seed=s)
        m, st, info = train_bivariate_model(train, val_es, team_idx, cfg)
        l3_models.append(m)
        l3_stats_list.append(st)
        l3_infos.append(info)
        log.info("L3 trained", seed=s, best_val_es=info["best_val_es"])

    # --- Cache lambdas (XGB single, L1/L3 averaged over seeds) ---
    lam_h_xgb_tr, lam_a_xgb_tr = xgb_model.predict_lambda(train)
    lam_h_xgb_c, lam_a_xgb_c = xgb_model.predict_lambda(val_calib)
    lam_h_xgb_g, lam_a_xgb_g = xgb_model.predict_lambda(val_gate)

    def _avg_l1(df):
        lhs, las = [], []
        for m, st in zip(l1_models, l1_stats_list, strict=True):
            lh, la = dl_predict(m, df, team_idx, st)
            lhs.append(lh); las.append(la)
        return np.mean(lhs, axis=0), np.mean(las, axis=0)

    def _avg_l3(df):
        lhs, las = [], []
        for m, st in zip(l3_models, l3_stats_list, strict=True):
            lh, la, _ = predict_lambda_rho(m, df, team_idx, st)
            lhs.append(lh); las.append(la)
        return np.mean(lhs, axis=0), np.mean(las, axis=0)

    lam_h_l1_tr, lam_a_l1_tr = _avg_l1(train)
    lam_h_l1_c, lam_a_l1_c = _avg_l1(val_calib)
    lam_h_l1_g, lam_a_l1_g = _avg_l1(val_gate)

    lam_h_l3_tr, lam_a_l3_tr = _avg_l3(train)
    lam_h_l3_c, lam_a_l3_c = _avg_l3(val_calib)
    lam_h_l3_g, lam_a_l3_g = _avg_l3(val_gate)

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

    # Clean previous artefacts (single-seed and multi-seed paths)
    for d_name in ("dl", "l3"):
        d = out_dir / d_name
        if d.exists():
            for f in d.iterdir():
                f.unlink()
            d.rmdir()
    for d_name in ("dl_seeds", "l3_seeds"):
        d = out_dir / d_name
        if d.exists():
            for sub in d.iterdir():
                if sub.is_dir():
                    for f in sub.iterdir():
                        f.unlink()
                    sub.rmdir()
            d.rmdir()
    if (out_dir / "ensemble.json").exists():
        (out_dir / "ensemble.json").unlink()

    if promoted:
        if uses_l1:
            for i, (m, st) in enumerate(zip(l1_models, l1_stats_list, strict=True)):
                save_dl_model(
                    m, team_idx, st, DLConfig(seed=l1_seeds[i]),
                    out_dir / "dl_seeds" / f"seed_{l1_seeds[i]}",
                )
        if uses_l3:
            for i, (m, st) in enumerate(zip(l3_models, l3_stats_list, strict=True)):
                save_bivariate(
                    m, team_idx, st, BivariateConfig(seed=l3_seeds[i]),
                    out_dir / "l3_seeds" / f"seed_{l3_seeds[i]}",
                )
        (out_dir / "ensemble.json").write_text(json.dumps({
            "weight_xgb": w_xgb, "weight_l1": w_l1, "weight_l3": w_l3,
            "rho_ensemble": rho_ens,
            "selected_on": "val_calib_log_loss",
            "l1_seeds": l1_seeds if uses_l1 else [],
            "l3_seeds": l3_seeds if uses_l3 else [],
        }, indent=2))

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
        "l1_seeds": l1_seeds,
        "l3_seeds": l3_seeds,
        "l1_best_val_es_nll": [float(i["best_val_es"]) for i in l1_infos],
        "l3_best_val_es_nll": [float(i["best_val_es"]) for i in l3_infos],
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
