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

import pandas as pd
import structlog

from mondiali.model.poisson_xgb import SYMMETRIC_FEATURES
from mondiali.training.train import train_tier2_pipeline
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
    val_calib = df[
        (df["date"] >= val_calib_start) & (df["date"] <= val_calib_end)
    ].reset_index(drop=True)
    val_gate = df[
        (df["date"] >= val_gate_start) & (df["date"] <= val_gate_end)
    ].reset_index(drop=True)

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

    manifest = {
        "version": "v1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "model": "PoissonXGBModel-symmetric-tier2",
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
        "rho": rho,
        "calibrator_kept": bool(calib_kept),
        "metrics_1x2": {
            "val_log_loss_raw": float(result["val_log_loss_raw"]),
            "val_log_loss_calib": float(result["val_log_loss_calib"]),
            "brier_before": float(result["brier_before"]),
            "brier_after": float(result["brier_after"]),
        },
        "random_state": 42,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("freeze_v1_final done", out_dir=str(out_dir))
    return {"manifest": manifest, "markets": markets_metrics, "train_result": result}
