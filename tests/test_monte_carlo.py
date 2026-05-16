"""Tests for monte_carlo.py."""
from __future__ import annotations

import numpy as np
import pytest

from mondiali.inference.monte_carlo import (
    points_for_result,
    sample_match_scores,
    simulate_group,
)


def test_points_for_result() -> None:
    assert points_for_result(2, 1) == (3, 0)
    assert points_for_result(0, 0) == (1, 1)
    assert points_for_result(1, 3) == (0, 3)


def test_sample_match_scores_distribution() -> None:
    rng = np.random.default_rng(42)
    h, a = sample_match_scores(1.5, 1.5, rho=-0.05, n_sims=10000, rng=rng)
    assert h.shape == (10000,)
    assert a.shape == (10000,)
    # Empirical means should be close to lambda (Poisson mean property)
    assert abs(h.mean() - 1.5) < 0.1
    assert abs(a.mean() - 1.5) < 0.1


def test_simulate_group_sums_to_one() -> None:
    matches = [
        {"team_a": "A", "team_b": "B", "lam_a": 1.5, "lam_b": 1.0, "rho": -0.05},
        {"team_a": "A", "team_b": "C", "lam_a": 2.0, "lam_b": 0.8, "rho": -0.05},
        {"team_a": "A", "team_b": "D", "lam_a": 1.8, "lam_b": 1.2, "rho": -0.05},
        {"team_a": "B", "team_b": "C", "lam_a": 1.0, "lam_b": 1.0, "rho": -0.05},
        {"team_a": "B", "team_b": "D", "lam_a": 1.2, "lam_b": 1.2, "rho": -0.05},
        {"team_a": "C", "team_b": "D", "lam_a": 0.8, "lam_b": 1.5, "rho": -0.05},
    ]
    df = simulate_group(matches, n_sims=2000, seed=42)
    # P(qualified) + P(eliminated) = 1 per team
    np.testing.assert_allclose(df["p_qualified"] + df["p_eliminated"], 1.0, atol=1e-9)
    # P(first) sums to 1 across teams (exactly one first per sim)
    assert df["p_first"].sum() == pytest.approx(1.0)
    assert df["p_second"].sum() == pytest.approx(1.0)


def test_simulate_group_favors_stronger_team() -> None:
    """Team A has much higher lambdas → should qualify >90% of time."""
    matches = [
        {"team_a": "A", "team_b": "B", "lam_a": 2.5, "lam_b": 0.5, "rho": -0.05},
        {"team_a": "A", "team_b": "C", "lam_a": 2.5, "lam_b": 0.5, "rho": -0.05},
        {"team_a": "A", "team_b": "D", "lam_a": 2.5, "lam_b": 0.5, "rho": -0.05},
        {"team_a": "B", "team_b": "C", "lam_a": 1.0, "lam_b": 1.0, "rho": -0.05},
        {"team_a": "B", "team_b": "D", "lam_a": 1.0, "lam_b": 1.0, "rho": -0.05},
        {"team_a": "C", "team_b": "D", "lam_a": 1.0, "lam_b": 1.0, "rho": -0.05},
    ]
    df = simulate_group(matches, n_sims=2000, seed=42)
    p_a_qualified = df[df["team"] == "A"]["p_qualified"].iloc[0]
    assert p_a_qualified > 0.9
