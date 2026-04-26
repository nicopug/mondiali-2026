"""Test pipeline training Tier 1 end-to-end (smoke test) + helper unit tests."""
from __future__ import annotations

import numpy as np
import pytest

from mondiali.config import CONFIG
from mondiali.training.train import _compute_1x2_probs, train_tier1_pipeline


def test_compute_1x2_probs_shape_and_rows_sum_to_one() -> None:
    """Fast unit test: shape (n,3) e somma riga ≈ 1 senza richiedere parquet."""
    lam_h = np.array([1.5, 2.0, 0.8])
    lam_a = np.array([1.2, 1.0, 1.5])
    out = _compute_1x2_probs(lam_h, lam_a, rho=-0.1)
    assert out.shape == (3, 3)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-10)
    assert (out >= 0).all()


def test_compute_1x2_probs_home_favorite_yields_p1_greater_than_p2() -> None:
    """λ_home >> λ_away → P(1) > P(2). Sanity sull'orientamento delle colonne."""
    out = _compute_1x2_probs(np.array([2.5]), np.array([0.7]), rho=-0.1)
    p1, _, p2 = out[0]
    assert p1 > p2


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
