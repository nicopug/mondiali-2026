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


def test_dixon_coles_correct_sum_to_one_after_normalize() -> None:
    """Dopo correzione + rinormalizzazione la matrice somma a 1 esattamente."""
    m_corrected = dixon_coles_correct(
        joint_matrix(1.5, 1.2),
        lam_home=1.5,
        lam_away=1.2,
        rho=-0.1,
    )
    assert m_corrected.sum() == pytest.approx(1.0, abs=1e-10)


def test_dixon_coles_correct_zero_rho_is_identity() -> None:
    """Con ρ=0 la correzione è l'identità (a meno di rinormalizzazione)."""
    m_before = joint_matrix(1.5, 1.2)
    m_before_norm = m_before / m_before.sum()
    m_after = dixon_coles_correct(m_before, 1.5, 1.2, rho=0.0)
    np.testing.assert_allclose(m_after, m_before_norm, rtol=1e-10)


def test_dixon_coles_correct_affects_only_low_score_cells() -> None:
    """La correzione tocca solo (0,0), (0,1), (1,0), (1,1). Cella (5,5) invariata
    a meno di rinormalizzazione uniforme."""
    m_before = joint_matrix(1.5, 1.2)
    m_before_norm = m_before / m_before.sum()
    m_after = dixon_coles_correct(m_before, 1.5, 1.2, rho=-0.1)
    ratio_55 = m_after[5, 5] / m_before_norm[5, 5]
    high_cells = m_after[2:, 2:] / m_before_norm[2:, 2:]
    np.testing.assert_allclose(high_cells, ratio_55, rtol=1e-10)


def test_estimate_rho_mle_returns_value_in_range() -> None:
    """ρ stimato su dati sintetici con mild low-score clustering ∈ [-0.3, 0.0]."""
    rng = np.random.default_rng(42)
    n = 1000
    lam_h = np.full(n, 1.3)
    lam_a = np.full(n, 1.1)
    home_goals = rng.poisson(lam_h)
    away_goals = rng.poisson(lam_a)
    mask = rng.uniform(size=n) < 0.05
    home_goals[mask] = 0
    away_goals[mask] = 0

    rho = estimate_rho_mle(lam_h, lam_a, home_goals, away_goals)
    assert -0.3 <= rho <= 0.05


def test_estimate_rho_mle_on_independent_poisson_close_to_zero() -> None:
    """Su dati puramente indipendenti Poisson, ρ stimato dovrebbe essere ~0."""
    rng = np.random.default_rng(7)
    n = 2000
    lam_h = np.full(n, 1.3)
    lam_a = np.full(n, 1.1)
    home_goals = rng.poisson(lam_h)
    away_goals = rng.poisson(lam_a)

    rho = estimate_rho_mle(lam_h, lam_a, home_goals, away_goals)
    assert abs(rho) < 0.05
