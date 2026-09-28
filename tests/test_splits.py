"""Tests for src/data/splits.py. The central guarantee: no test period precedes any train period."""

import pandas as pd
import pytest

from src.data.loader import DEFAULT_ROOT
from src.data.splits import Splits, assert_chronological, chronological_split, load_splits

SELECTED_DOMAINS = ["Economy", "SocialGood", "Traffic", "Health_US"]


def _monthly(n: int) -> pd.DataFrame:
    start = pd.date_range("2000-01-01", periods=n, freq="MS")
    return pd.DataFrame({
        "start_date": start,
        "end_date": start + pd.offsets.MonthEnd(0),
        "value": range(n),
        "has_text": True,
    })


def _check_no_leakage(s: Splits) -> None:
    # The core property: every test timestamp is strictly after every train timestamp.
    assert s.test["start_date"].min() > s.train["end_date"].max()
    assert s.test["start_date"].min() > s.val["end_date"].max()
    assert s.val["start_date"].min() > s.train["end_date"].max()


def test_no_test_timestamp_precedes_any_train_timestamp():
    _check_no_leakage(chronological_split(_monthly(100)))


def test_sizes_match_mm_tsflib_70_10_20():
    s = chronological_split(_monthly(100))
    assert s.sizes() == {"train": 70, "val": 10, "test": 20}


def test_blocks_are_contiguous_and_cover_every_row():
    df = _monthly(97)
    s = chronological_split(df)
    rejoined = pd.concat([s.train, s.val, s.test], ignore_index=True)
    pd.testing.assert_frame_equal(rejoined, df)


def test_unsorted_input_rejected():
    df = _monthly(50).iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="sorted"):
        chronological_split(df)


def test_duplicate_dates_rejected():
    df = _monthly(50)
    df.loc[10, "start_date"] = df.loc[9, "start_date"]
    with pytest.raises(ValueError):
        chronological_split(df)


@pytest.mark.parametrize("train, test", [(0.9, 0.2), (0, 0.2), (0.7, 0)])
def test_bad_fractions_rejected(train, test):
    with pytest.raises(ValueError):
        chronological_split(_monthly(100), train_frac=train, test_frac=test)


def test_too_few_rows_rejected():
    with pytest.raises(ValueError, match="too few"):
        chronological_split(_monthly(3))


def test_assert_chronological_catches_overlap():
    df = _monthly(30)
    bad = Splits(train=df.iloc[10:], val=df.iloc[5:10], test=df.iloc[:5])
    with pytest.raises(AssertionError):
        assert_chronological(bad)


@pytest.mark.skipif(not (DEFAULT_ROOT / "numerical").exists(), reason="Time-MMD not downloaded")
@pytest.mark.parametrize("domain", SELECTED_DOMAINS)
def test_real_domains_no_leakage(domain):
    s = load_splits(domain)
    _check_no_leakage(s)
    # Split is over the text window, so the test block must contain text to attack.
    assert s.test["has_text"].any()
