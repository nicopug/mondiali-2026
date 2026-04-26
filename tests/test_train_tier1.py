"""Test pipeline training Tier 1 end-to-end (smoke test)."""
from __future__ import annotations

import pytest

from mondiali.config import CONFIG
from mondiali.training.train import train_tier1_pipeline


@pytest.mark.slow
def test_train_tier1_pipeline_produces_reasonable_log_loss() -> None:
    """Smoke test con dati reali: il pipeline completa e produce log-loss ∈ [0.88, 1.02].

    Salta se matches.parquet non esiste.
    """
    parquet = CONFIG.data_processed / "matches.parquet"
    if not parquet.exists():
        pytest.skip("matches.parquet not found")

    result = train_tier1_pipeline(
        parquet_path=parquet,
        train_start="2002-01-01",
        train_end="2016-12-31",
        val_start="2017-01-01",
        val_end="2018-12-31",
    )
    assert 0.88 <= result["val_log_loss_1x2"] <= 1.02
    assert -0.3 <= result["rho"] <= 0.05
    assert 0.8 <= result["lambda_home_mean"] <= 2.0
