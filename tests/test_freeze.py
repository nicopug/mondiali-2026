"""Smoke test for freeze_v1_final — checks artefacts written + manifest shape."""
from __future__ import annotations

import json
from pathlib import Path

from mondiali.training.freeze import freeze_v1_final


def test_freeze_v1_final_writes_all_artefacts(tmp_path: Path) -> None:
    out = tmp_path / "v1_final"
    freeze_v1_final(
        matches_path=Path("data/processed/matches.parquet"),
        out_dir=out,
        train_end="2021-12-31",
        val_es_start="2022-01-01",
        val_es_end="2022-06-30",
        val_calib_start="2023-01-01",
        val_calib_end="2023-12-31",
        val_gate_start="2024-01-01",
        val_gate_end="2024-12-31",
    )
    for f in ("xgb_poisson.json", "calibrator.json", "rho.txt",
              "manifest.json", "markets_validation.json"):
        assert (out / f).exists(), f"missing {f}"

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["version"] == "v1.0"
    assert manifest["n_features"] == 24
    assert "git_sha" in manifest
    assert "rho" in manifest

    markets = json.loads((out / "markets_validation.json").read_text())
    assert set(markets.keys()) == {
        "over_under_1_5", "over_under_2_5", "over_under_3_5", "btts",
    }
    for m in markets.values():
        assert "validated" in m
        assert "brier" in m
        assert "baseline_brier" in m
