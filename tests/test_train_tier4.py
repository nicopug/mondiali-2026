"""Smoke tests for train_tier4_pipeline."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mondiali.training.train import train_tier4_pipeline


@pytest.fixture
def tiny_matches_parquet(tmp_path: Path) -> Path:
    src = Path("data/processed/matches.parquet")
    if not src.exists():
        pytest.skip("matches.parquet not built — run build_processed_matches first")
    df = pd.read_parquet(src)
    df = df.iloc[:5000].copy()
    out = tmp_path / "matches.parquet"
    df.to_parquet(out, index=False)
    return out


def test_train_tier4_smoke_runs_with_few_trials(
    tmp_path: Path, tiny_matches_parquet: Path
) -> None:
    """Pipeline runs end-to-end with --n-trials 2 (functional smoke)."""
    rosters_path = Path("data/raw/transfermarkt/rosters.parquet")
    if not rosters_path.exists():
        pytest.skip("rosters.parquet not built yet — run after Task 5 e2e")
    injuries_path = Path("data/manual/injuries.csv")
    if not injuries_path.exists() or injuries_path.read_text().strip() == "":
        pytest.skip("injuries.csv not bootstrapped — run after Task 8 e2e")
    out_dir = tmp_path / "tier4"
    result = train_tier4_pipeline(
        matches_path=tiny_matches_parquet,
        rosters_path=rosters_path,
        injuries_path=injuries_path,
        out_dir=out_dir,
        n_trials=2,
        seed=42,
    )
    assert "baseline_log_loss" in result
    assert "challenger_log_loss" in result
    assert "delta" in result
    assert (out_dir / "baseline_params.json").exists()
    assert (out_dir / "challenger_params.json").exists()
    assert (out_dir / "xgb_poisson.json").exists()
    assert (out_dir / "calibrator.json").exists()


def test_train_tier4_deterministic_with_seed(
    tmp_path: Path, tiny_matches_parquet: Path
) -> None:
    """Two runs with same seed and n_trials produce identical metrics."""
    rosters_path = Path("data/raw/transfermarkt/rosters.parquet")
    injuries_path = Path("data/manual/injuries.csv")
    if not rosters_path.exists() or not injuries_path.exists():
        pytest.skip("rosters or injuries not built")
    r1 = train_tier4_pipeline(
        matches_path=tiny_matches_parquet,
        rosters_path=rosters_path,
        injuries_path=injuries_path,
        out_dir=tmp_path / "run1",
        n_trials=2,
        seed=42,
    )
    r2 = train_tier4_pipeline(
        matches_path=tiny_matches_parquet,
        rosters_path=rosters_path,
        injuries_path=injuries_path,
        out_dir=tmp_path / "run2",
        n_trials=2,
        seed=42,
    )
    assert r1["baseline_log_loss"] == pytest.approx(r2["baseline_log_loss"])
    assert r1["challenger_log_loss"] == pytest.approx(r2["challenger_log_loss"])
