from __future__ import annotations

import numpy as np

from mondiali.config import CONFIG
from mondiali.training.train import train_talent_challenger


def test_train_talent_challenger_smoke():
    parquet = CONFIG.data_processed / "matches.parquet"
    result = train_talent_challenger(
        parquet_path=parquet,
        model_params={"n_estimators": 50},  # fast
    )
    assert result["n_train"] > 1000
    assert np.isfinite(result["val_log_loss_raw"])
    assert result["model"].include_talent is True
