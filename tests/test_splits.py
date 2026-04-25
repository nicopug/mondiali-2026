"""Test per walk-forward CV splits."""
from __future__ import annotations

import pandas as pd
import pytest

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


def test_walk_forward_splits_empty_dataframe_yields_nothing() -> None:
    """DataFrame vuoto -> generator vuoto, nessuna eccezione."""
    df = pd.DataFrame({"date": pd.to_datetime([]), "x": []})
    folds = list(walk_forward_splits(df, n_folds=3, val_years=1))
    assert folds == []


def test_walk_forward_splits_n_folds_one() -> None:
    """n_folds=1 produce un solo fold con val=ultimo anno."""
    dates = [f"{y}-06-15" for y in range(2002, 2019)]
    df = _fake_df(dates)
    folds = list(walk_forward_splits(df, n_folds=1, val_years=1))
    assert len(folds) == 1
    train, val = folds[0]
    assert train["date"].max().year == 2017
    assert val["date"].min().year == 2018
    assert val["date"].max().year == 2018


def test_walk_forward_splits_val_years_two() -> None:
    """val_years=2: ogni val window copre 2 anni consecutivi, train espande di 2 anni."""
    dates = [f"{y}-06-15" for y in range(2002, 2019)]
    df = _fake_df(dates)
    folds = list(walk_forward_splits(df, n_folds=3, val_years=2))
    assert len(folds) == 3
    # Fold 1: train <= 2012, val 2013-2014
    train1, val1 = folds[0]
    assert train1["date"].max().year == 2012
    assert val1["date"].min().year == 2013
    assert val1["date"].max().year == 2014
    # Fold 2: train <= 2014, val 2015-2016
    train2, val2 = folds[1]
    assert train2["date"].max().year == 2014
    assert val2["date"].min().year == 2015
    assert val2["date"].max().year == 2016
    # Fold 3: train <= 2016, val 2017-2018
    train3, val3 = folds[2]
    assert train3["date"].max().year == 2016
    assert val3["date"].min().year == 2017
    assert val3["date"].max().year == 2018


@pytest.mark.parametrize(("n_folds", "val_years"), [(0, 1), (1, 0), (-1, 1), (3, -1)])
def test_walk_forward_splits_rejects_invalid_params(n_folds: int, val_years: int) -> None:
    """n_folds o val_years < 1 -> ValueError."""
    df = _fake_df([f"{y}-06-15" for y in range(2002, 2019)])
    with pytest.raises(ValueError, match=">= 1"):
        list(walk_forward_splits(df, n_folds=n_folds, val_years=val_years))
