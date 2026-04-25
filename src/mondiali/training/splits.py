"""Walk-forward CV splits (expanding window, mai random).

Default config (spec §7.2): 3 fold con val di 1 anno ciascuno.
Boundaries half-open `[start, end)` — robuste a Timestamp con time-component.
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
        n_folds: numero di fold (>= 1).
        val_years: ampiezza del validation window per fold, in anni (>= 1).

    Yields:
        (train_df, val_df) — fold-size crescenti, nessun overlap.
    """
    if n_folds < 1 or val_years < 1:
        raise ValueError(
            f"n_folds and val_years must be >= 1, got {n_folds=}, {val_years=}"
        )
    if matches.empty:
        return
    max_year = int(matches["date"].dt.year.max())
    for i in range(n_folds):
        val_year_end = max_year - (n_folds - 1 - i) * val_years
        val_year_start = val_year_end - val_years + 1
        train_end = pd.Timestamp(year=val_year_start, month=1, day=1)
        val_end_excl = pd.Timestamp(year=val_year_end + 1, month=1, day=1)
        train = matches[matches["date"] < train_end].copy()
        val = matches[(matches["date"] >= train_end) & (matches["date"] < val_end_excl)].copy()
        yield train, val
