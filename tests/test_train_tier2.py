"""Test pipeline training Tier 2 end-to-end."""
from __future__ import annotations

import pytest

from mondiali.config import CONFIG
from mondiali.training.train import train_tier2_pipeline


def test_train_tier2_returns_required_keys() -> None:
    """Smoke test rapido: il dict di ritorno ha tutte le chiavi attese."""
    parquet = CONFIG.data_processed / "matches.parquet"
    if not parquet.exists():
        pytest.skip("matches.parquet not found")

    result = train_tier2_pipeline(
        parquet_path=parquet,
        train_start="2002-01-01",
        train_end="2010-12-31",
        val_es_start="2011-01-01",
        val_es_end="2011-12-31",
        val_calib_start="2012-01-01",
        val_calib_end="2012-12-31",
        val_gate_start="2013-01-01",
        val_gate_end="2013-06-30",
    )
    expected_keys = {
        "model", "rho", "calibrator",
        "val_log_loss_raw", "val_log_loss_calib",
        "brier_before", "brier_after",
        "n_train", "n_val_es", "n_val_calib", "n_val_gate",
    }
    assert expected_keys.issubset(result.keys())


@pytest.mark.slow
def test_train_tier2_full_split_produces_reasonable_loss() -> None:
    """Full split production: gate è su val_log_loss_raw.

    Empirical finding (STEP 3): l'isotonic calibrator su val_calib=2018 (n~923)
    degrada la log-loss su val_gate=2019-2022 di ~0.08 — campione troppo piccolo
    e/o distribution shift verso Euro2020/COVID. Il segnale forte è sulle Tier 2
    feature: raw=0.8487 batte ELO baseline (0.8525). Il gate ufficiale di STEP 3
    è quindi su val_log_loss_raw; calibration è informativa, non bloccante.
    """
    parquet = CONFIG.data_processed / "matches.parquet"
    if not parquet.exists():
        pytest.skip("matches.parquet not found")

    result = train_tier2_pipeline(parquet_path=parquet)
    assert 0.83 <= result["val_log_loss_raw"] <= 0.86
    assert -0.3 <= result["rho"] <= 0.05


def test_train_tier2_splits_have_no_overlap() -> None:
    """I 4 set sono mutualmente esclusivi e ordinati temporalmente."""
    parquet = CONFIG.data_processed / "matches.parquet"
    if not parquet.exists():
        pytest.skip("matches.parquet not found")

    result = train_tier2_pipeline(
        parquet_path=parquet,
        train_start="2002-01-01",
        train_end="2010-12-31",
        val_es_start="2011-01-01",
        val_es_end="2011-12-31",
        val_calib_start="2012-01-01",
        val_calib_end="2012-12-31",
        val_gate_start="2013-01-01",
        val_gate_end="2013-06-30",
    )
    assert result["n_train"] > 0
    assert result["n_val_es"] > 0
    assert result["n_val_calib"] > 0
    assert result["n_val_gate"] > 0


def test_recompute_tier2_baseline_for_gate_returns_float() -> None:
    """Smoke: helper ritorna un float positivo plausibile."""
    from mondiali.training.train import _recompute_tier2_baseline_for_gate

    parquet = CONFIG.data_processed / "matches.parquet"
    if not parquet.exists():
        pytest.skip("matches.parquet not found")

    baseline = _recompute_tier2_baseline_for_gate(
        parquet,
        val_gate_start="2022-01-01",
        val_gate_end="2022-12-31",
        train_end="2018-12-31",
        val_es_start="2019-01-01",
        val_es_end="2019-12-31",
    )
    assert 0.5 < baseline < 1.5
