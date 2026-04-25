"""Walk-forward CV splits (expanding window, mai random).

Conforme a spec §7.2: 3 fold su 2002-2018 con val di 1 anno ciascuno.
"""
from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def walk_forward_splits(
    matches: pd.DataFrame,
    *,
    n_folds: int = 3,
    val_years: int = 1,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Itera `n_folds` fold expanding-window.

    Richiede `matches` con colonna `date` datetime. L'ultimo val window termina
    al `max(date)`; il primo training set inizia al `min(date)` ed espande.

    Args:
        matches: DataFrame con colonna `date`.
        n_folds: numero di fold.
        val_years: ampiezza del validation window per fold, in anni.

    Yields:
        (train_df, val_df) — fold-size crescenti, nessun overlap.
    """
    if matches.empty:
        return
    max_year = matches["date"].dt.year.max()
    for i in range(n_folds):
        val_year_end = max_year - (n_folds - 1 - i) * val_years
        val_year_start = val_year_end - val_years + 1
        train_end = pd.Timestamp(year=val_year_start - 1, month=12, day=31)
        val_start = pd.Timestamp(year=val_year_start, month=1, day=1)
        val_end = pd.Timestamp(year=val_year_end, month=12, day=31)
        train = matches[matches["date"] <= train_end]
        val = matches[(matches["date"] >= val_start) & (matches["date"] <= val_end)]
        yield train, val
