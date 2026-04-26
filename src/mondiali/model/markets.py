"""Derivazione mercati 1X2, O/U 2.5, BTTS dal joint goal matrix.

Input: matrice (MAX_GOALS+1) x (MAX_GOALS+1) normalizzata (somma=1).
- P(1) = Σ_{i>j} P(i,j)
- P(X) = Σ_{i=j} P(i,j)
- P(2) = Σ_{i<j} P(i,j)
- P(Over k) = Σ_{i+j>k} P(i,j)
- P(BTTS=Y) = Σ_{i>0 ∧ j>0} P(i,j)
"""
from __future__ import annotations

import numpy as np


def prob_1x2(joint: np.ndarray) -> tuple[float, float, float]:
    """Ritorna (P(home_win), P(draw), P(away_win))."""
    n = joint.shape[0]
    idx = np.arange(n)
    i_grid, j_grid = np.meshgrid(idx, idx, indexing="ij")
    p_home = float(joint[i_grid > j_grid].sum())
    p_draw = float(joint[i_grid == j_grid].sum())
    p_away = float(joint[i_grid < j_grid].sum())
    return p_home, p_draw, p_away


def prob_over_under(joint: np.ndarray, *, threshold: float = 2.5) -> tuple[float, float]:
    """Ritorna (P(over), P(under)) per `total goals` rispetto a `threshold`.

    `threshold` deve essere half-integer (2.5, 3.5, ...): convenzione standard
    dei mercati di calcio. Soglie intere (es. 2.0) creerebbero un caso "void"
    e farebbero P(over)+P(under)<1 sui pareggi su quella linea.
    """
    doubled = threshold * 2.0
    if not float(doubled).is_integer() or int(doubled) % 2 == 0:
        raise ValueError(
            f"threshold must be half-integer (e.g. 2.5, 3.5); got {threshold}"
        )
    n = joint.shape[0]
    idx = np.arange(n)
    i_grid, j_grid = np.meshgrid(idx, idx, indexing="ij")
    total = i_grid + j_grid
    p_over = float(joint[total > threshold].sum())
    p_under = float(joint[total < threshold].sum())
    return p_over, p_under


def prob_btts(joint: np.ndarray) -> tuple[float, float]:
    """Ritorna (P(BTTS=Yes), P(BTTS=No)).

    BTTS=Yes sse entrambe le squadre segnano almeno 1 gol.
    """
    n = joint.shape[0]
    idx = np.arange(n)
    i_grid, j_grid = np.meshgrid(idx, idx, indexing="ij")
    both_score = (i_grid > 0) & (j_grid > 0)
    p_yes = float(joint[both_score].sum())
    p_no = 1.0 - p_yes
    return p_yes, p_no
