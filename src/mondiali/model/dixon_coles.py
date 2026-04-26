"""Dixon-Coles correction + joint goal matrix.

Pipeline di inference (spec §6.2):
1. Dato (λ_home, λ_away), costruisci `P(i,j) = P(i|λ_h) * P(j|λ_a)` per
   i,j ∈ [0, MAX_GOALS].
2. Applica correzione Dixon-Coles (bassi punteggi).
3. Rinormalizza a somma 1.

ρ stimato via MLE sul training set (funzione separata).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

MAX_GOALS: int = 10


def joint_matrix(lam_home: float, lam_away: float) -> np.ndarray:
    """Matrice P(i,j) = Poisson(i|lam_home) * Poisson(j|lam_away)."""
    pmf_h = poisson.pmf(np.arange(MAX_GOALS + 1), mu=lam_home)
    pmf_a = poisson.pmf(np.arange(MAX_GOALS + 1), mu=lam_away)
    return np.outer(pmf_h, pmf_a)


# Task 9 stubs - full implementation in next task
def dixon_coles_correct(
    matrix: np.ndarray, lam_home: float, lam_away: float, rho: float
) -> np.ndarray:
    raise NotImplementedError("dixon_coles_correct: implemented in Task 9")


def estimate_rho_mle(matches: object) -> float:
    raise NotImplementedError("estimate_rho_mle: implemented in Task 9")
