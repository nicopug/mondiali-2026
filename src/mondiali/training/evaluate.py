"""Evaluation metrics: log-loss 1/X/2.

Classi (ordine fisso): 0 = home win, 1 = draw, 2 = away win.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def compute_outcomes(matches: pd.DataFrame) -> np.ndarray:
    """0 = home win, 1 = draw, 2 = away win."""
    home = matches["home_score"].to_numpy()
    away = matches["away_score"].to_numpy()
    out = np.where(home > away, 0, np.where(home == away, 1, 2))
    return out.astype(np.int64)


def log_loss_1x2(matches: pd.DataFrame, probabilities: np.ndarray) -> float:
    """Log-loss multi-classe su esiti 1/X/2.

    Args:
        matches: DataFrame con home_score, away_score (verità).
        probabilities: shape (n, 3), colonne = [P(home), P(draw), P(away)].

    Returns:
        log-loss (media).

    Raises:
        ValueError: shape mismatch o probabilità invalide.
    """
    if probabilities.shape != (len(matches), 3):
        raise ValueError(
            f"probabilities shape {probabilities.shape} != expected ({len(matches)}, 3)"
        )
    y_true = compute_outcomes(matches)
    return float(log_loss(y_true, probabilities, labels=[0, 1, 2]))
