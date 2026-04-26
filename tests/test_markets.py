"""Test derivazione mercati dal joint goal matrix."""
from __future__ import annotations

import numpy as np
import pytest

from mondiali.model.dixon_coles import joint_matrix
from mondiali.model.markets import (
    prob_1x2,
    prob_btts,
    prob_over_under,
)


def _normalized_joint(lam_h: float, lam_a: float) -> np.ndarray:
    m = joint_matrix(lam_h, lam_a)
    return m / m.sum()


def test_prob_1x2_sums_to_one() -> None:
    """P(1) + P(X) + P(2) = 1."""
    m = _normalized_joint(1.5, 1.2)
    p1, px, p2 = prob_1x2(m)
    assert p1 + px + p2 == pytest.approx(1.0, abs=1e-10)


def test_prob_1x2_home_favorite_has_highest_p1() -> None:
    """λ_home >> λ_away → P(1) > P(2)."""
    m = _normalized_joint(2.5, 0.8)
    p1, _, p2 = prob_1x2(m)
    assert p1 > p2


def test_prob_over_under_complementary() -> None:
    """P(Over 2.5) + P(Under 2.5) = 1."""
    m = _normalized_joint(1.5, 1.2)
    p_over, p_under = prob_over_under(m, threshold=2.5)
    assert p_over + p_under == pytest.approx(1.0, abs=1e-10)


def test_prob_over_under_threshold_monotonic() -> None:
    """P(Over 2.5) > P(Over 3.5)."""
    m = _normalized_joint(1.8, 1.5)
    p_over_25, _ = prob_over_under(m, threshold=2.5)
    p_over_35, _ = prob_over_under(m, threshold=3.5)
    assert p_over_25 > p_over_35


def test_prob_btts_complementary_to_not_btts() -> None:
    """P(BTTS=Yes) + P(BTTS=No) = 1."""
    m = _normalized_joint(1.5, 1.2)
    p_yes, p_no = prob_btts(m)
    assert p_yes + p_no == pytest.approx(1.0, abs=1e-10)


def test_prob_btts_high_lambdas_increases_p_yes() -> None:
    """λ alti per entrambe le squadre → P(BTTS=Yes) alto."""
    m_low = _normalized_joint(0.5, 0.5)
    m_high = _normalized_joint(2.0, 2.0)
    p_yes_low, _ = prob_btts(m_low)
    p_yes_high, _ = prob_btts(m_high)
    assert p_yes_high > p_yes_low


def test_prob_over_under_rejects_integer_threshold() -> None:
    """Soglia intera (es. 2.0) creerebbe over+under<1 sui pareggi sulla linea.

    La convenzione di mercato è half-integer; rifiutare integer è hardening
    contro un footgun silenzioso.
    """
    m = _normalized_joint(1.5, 1.2)
    with pytest.raises(ValueError, match="half-integer"):
        prob_over_under(m, threshold=2.0)
    with pytest.raises(ValueError, match="half-integer"):
        prob_over_under(m, threshold=2.7)
