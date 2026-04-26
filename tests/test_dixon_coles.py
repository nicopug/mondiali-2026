"""Test Dixon-Coles correction + joint goal matrix."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import poisson

from mondiali.model.dixon_coles import (
    MAX_GOALS,
    dixon_coles_correct,
    estimate_rho_mle,
    joint_matrix,
)


def test_joint_matrix_shape_and_sums_close_to_one() -> None:
    """Shape (MAX_GOALS+1, MAX_GOALS+1), sum ≈ 1 (≥ 0.99 con lam ≤ 3)."""
    m = joint_matrix(lam_home=1.5, lam_away=1.2)
    assert m.shape == (MAX_GOALS + 1, MAX_GOALS + 1)
    assert 0.99 <= m.sum() <= 1.0


def test_joint_matrix_is_outer_product_of_truncated_pmfs() -> None:
    """P(i,j) = pmf_h[i] * pmf_a[j] (pre-correzione DC).

    Test esatto sulla struttura outer-product senza passare dai marginali —
    il troncamento a MAX_GOALS=10 lascia ~8e-6 di coda residua per λ=2,
    incompatibile con un confronto sui marginali a rtol=1e-10. La struttura
    outer-product invece è esatta per costruzione e qui pinzata a precisione
    macchina.
    """
    m = joint_matrix(lam_home=1.0, lam_away=2.0)
    pmf_h = poisson.pmf(np.arange(MAX_GOALS + 1), mu=1.0)
    pmf_a = poisson.pmf(np.arange(MAX_GOALS + 1), mu=2.0)
    expected = np.outer(pmf_h, pmf_a)
    np.testing.assert_allclose(m, expected, rtol=1e-12)


def test_task9_stubs_raise_not_implemented() -> None:
    """dixon_coles_correct ed estimate_rho_mle sono stub Task 9: pinia il
    contratto perché un futuro merge non li lasci silenziosamente no-op.
    """
    with pytest.raises(NotImplementedError, match="Task 9"):
        dixon_coles_correct(np.zeros((11, 11)), 1.0, 1.0, -0.1)
    with pytest.raises(NotImplementedError, match="Task 9"):
        estimate_rho_mle(object())
