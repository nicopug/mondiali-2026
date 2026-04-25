"""Test per walk-forward CV splits."""
from __future__ import annotations

import pandas as pd

from mondiali.training.splits import walk_forward_splits


def _fake_df(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(dates), "x": range(len(dates))})


def test_walk_forward_splits_produces_expanding_train() -> None:
    """Fold i usa train=[2002-01-01, year_i-12-31], val=[year_i+1, year_i+1-12-31]."""
    dates = [f"{y}-06-15" for y in range(2002, 2019)]
    df = _fake_df(dates)
    folds = list(walk_forward_splits(df, n_folds=3, val_years=1))

    assert len(folds) == 3
    # Fold 1: train 2002-2015, val 2016
    train1, val1 = folds[0]
    assert train1["date"].max().year == 2015
    assert val1["date"].min().year == 2016
    assert val1["date"].max().year == 2016
    # Fold 2: train 2002-2016, val 2017
    train2, val2 = folds[1]
    assert train2["date"].max().year == 2016
    assert val2["date"].min().year == 2017
    # Fold 3
    train3, val3 = folds[2]
    assert train3["date"].max().year == 2017
    assert val3["date"].min().year == 2018


def test_walk_forward_splits_no_overlap_train_val() -> None:
    """Nessuna data di validation cade dentro training."""
    dates = [f"{y}-06-15" for y in range(2002, 2019)]
    df = _fake_df(dates)
    for train, val in walk_forward_splits(df, n_folds=3, val_years=1):
        assert train["date"].max() < val["date"].min()


def test_walk_forward_splits_expands_train() -> None:
    """Il training set cresce fold dopo fold."""
    dates = [f"{y}-06-15" for y in range(2002, 2019)]
    df = _fake_df(dates)
    folds = list(walk_forward_splits(df, n_folds=3, val_years=1))
    sizes = [len(train) for train, _ in folds]
    assert sizes[0] < sizes[1] < sizes[2]
