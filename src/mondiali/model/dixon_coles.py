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
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

MAX_GOALS: int = 10


def joint_matrix(lam_home: float, lam_away: float) -> np.ndarray:
    """Matrice P(i,j) = Poisson(i|lam_home) * Poisson(j|lam_away)."""
    pmf_h = poisson.pmf(np.arange(MAX_GOALS + 1), mu=lam_home)
    pmf_a = poisson.pmf(np.arange(MAX_GOALS + 1), mu=lam_away)
    return np.outer(pmf_h, pmf_a)


def dixon_coles_correct(
    matrix: np.ndarray,
    lam_home: float,
    lam_away: float,
    rho: float,
) -> np.ndarray:
    """Applica correzione Dixon-Coles (spec §6.2) e rinormalizza a somma 1.

    Correzione solo sulle 4 celle basso-punteggio:
        P(0,0) *= 1 - lam_home * lam_away * rho
        P(0,1) *= 1 + lam_home * rho
        P(1,0) *= 1 + lam_away * rho
        P(1,1) *= 1 - rho

    ρ tipico empirico ≈ -0.1 (correla leggermente 0-0 e 1-1 con excess rispetto
    a indipendenza Poisson).
    """
    m = matrix.copy()
    m[0, 0] *= 1.0 - lam_home * lam_away * rho
    m[0, 1] *= 1.0 + lam_home * rho
    m[1, 0] *= 1.0 + lam_away * rho
    m[1, 1] *= 1.0 - rho
    s = m.sum()
    if s <= 0:
        raise ValueError(f"Dixon-Coles matrix sum <= 0 (rho={rho}): non rinormalizzabile")
    result: np.ndarray = m / s
    return result


def estimate_rho_mle(
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    *,
    bounds: tuple[float, float] = (-0.3, 0.1),
) -> float:
    """Stima ρ via MLE massimizzando la log-likelihood congiunta.

    Per ogni match: logL_i = log(τ(h_i, a_i, λ_h_i, λ_a_i, ρ))
                         + log(Poisson(h_i | λ_h_i))
                         + log(Poisson(a_i | λ_a_i))

    I termini Poisson non dipendono da ρ: ottimizziamo solo Σ log(τ).
    """
    lh = np.asarray(lam_home, dtype=float)
    la = np.asarray(lam_away, dtype=float)
    hg = np.asarray(home_goals, dtype=int)
    ag = np.asarray(away_goals, dtype=int)

    mask00 = (hg == 0) & (ag == 0)
    mask01 = (hg == 0) & (ag == 1)
    mask10 = (hg == 1) & (ag == 0)
    mask11 = (hg == 1) & (ag == 1)

    def neg_log_likelihood(rho: float) -> float:
        total = 0.0
        if mask00.any():
            vals = 1.0 - lh[mask00] * la[mask00] * rho
            if (vals <= 0).any():
                return np.inf
            total += np.log(vals).sum()
        if mask01.any():
            vals = 1.0 + lh[mask01] * rho
            if (vals <= 0).any():
                return np.inf
            total += np.log(vals).sum()
        if mask10.any():
            vals = 1.0 + la[mask10] * rho
            if (vals <= 0).any():
                return np.inf
            total += np.log(vals).sum()
        if mask11.any():
            vals_scalar = 1.0 - rho
            if vals_scalar <= 0:
                return np.inf
            total += np.log(vals_scalar) * mask11.sum()
        return -total

    result = minimize_scalar(neg_log_likelihood, bounds=bounds, method="bounded")
    return float(result.x)
