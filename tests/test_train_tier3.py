"""Test pipeline training Tier 3 end-to-end."""
from __future__ import annotations

import pandas as pd
import pytest

from mondiali.config import CONFIG
from mondiali.features.tier3 import TIER3_COLUMNS
from mondiali.training.train import train_tier3_pipeline


def test_train_tier3_returns_required_keys() -> None:
    """Smoke test: il dict di ritorno ha tutte le chiavi attese."""
    parquet = CONFIG.data_processed / "matches.parquet"
    if not parquet.exists():
        pytest.skip("matches.parquet not found")

    df = pd.read_parquet(parquet)
    if not all(c in df.columns for c in TIER3_COLUMNS):
        pytest.skip("tier3 columns not in matches.parquet — run tm-scrape + ingest first")
    if df[TIER3_COLUMNS[0]].notna().sum() == 0:
        pytest.skip("tier3 columns are all NaN — snapshots.parquet not yet scraped")

    result = train_tier3_pipeline(
        parquet_path=parquet,
        train_start="2014-01-01", train_end="2018-12-31",
        val_es_start="2019-01-01", val_es_end="2019-12-31",
        val_calib_start="2020-01-01", val_calib_end="2020-12-31",
        val_gate_start="2021-01-01", val_gate_end="2021-12-31",
    )
    expected_keys = {
        "model", "rho", "calibrator",
        "val_log_loss_raw", "val_log_loss_calib",
        "brier_before", "brier_after",
        "n_train", "n_val_es", "n_val_calib", "n_val_gate",
        "n_train_pre2014_dropped", "tm_coverage_train", "tm_coverage_gate",
    }
    assert expected_keys.issubset(result.keys())


@pytest.mark.slow
def test_train_tier3_full_split_passes_gate() -> None:
    """STEP 4 GATE BLOCKING: val_log_loss_raw_tier3 <= tier2_baseline_2022 - 0.001."""
    parquet = CONFIG.data_processed / "matches.parquet"
    if not parquet.exists():
        pytest.skip("matches.parquet not found")
    df = pd.read_parquet(parquet)
    if not all(c in df.columns for c in TIER3_COLUMNS):
        pytest.skip("tier3 columns not in matches.parquet")
    if df[TIER3_COLUMNS[0]].notna().sum() == 0:
        pytest.skip("tier3 columns are all NaN — snapshots.parquet not yet scraped")

    from mondiali.training.train import _recompute_tier2_baseline_for_gate
    baseline_t2 = _recompute_tier2_baseline_for_gate(
        parquet, val_gate_start="2022-01-01", val_gate_end="2022-12-31",
    )

    result = train_tier3_pipeline(parquet_path=parquet)

    assert result["val_log_loss_raw"] <= baseline_t2 - 0.001, (
        f"GATE FAIL: tier3={result['val_log_loss_raw']:.4f} > "
        f"tier2={baseline_t2:.4f} - 0.001"
    )
    assert -0.3 <= result["rho"] <= 0.05
    assert result["tm_coverage_gate"] >= 0.80
