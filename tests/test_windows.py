"""Tests for src/data/windows.py. Central guarantee: targets never leave their split."""

import numpy as np
import pandas as pd
import pytest

from src.data.splits import chronological_split
from src.data.windows import LOOKBACK, lookback_for, make_windows, split_windows


def _monthly(n: int) -> pd.DataFrame:
    start = pd.date_range("2000-01-01", periods=n, freq="MS")
    return pd.DataFrame({"start_date": start, "end_date": start + pd.offsets.MonthEnd(0),
                         "value": np.arange(n, dtype=float), "has_text": True})


def test_window_inputs_are_the_seq_len_values_before_the_target():
    w = make_windows(np.arange(10.0), pd.Series(pd.date_range("2000", periods=10, freq="MS")),
                     0, 10, seq_len=3, horizon=1)
    np.testing.assert_array_equal(w.X[0], [0, 1, 2])
    np.testing.assert_array_equal(w.Y[0], [3])
    assert len(w) == 7


def test_split_sample_counts_lose_seq_len_from_train_only():
    s = chronological_split(_monthly(100))  # 70 / 10 / 20
    w = split_windows(s, seq_len=8, horizon=1)
    assert (len(w["train"]), len(w["val"]), len(w["test"])) == (62, 10, 20)


def test_val_and_test_take_context_from_earlier_splits():
    s = chronological_split(_monthly(100))
    w = split_windows(s, seq_len=8, horizon=1)
    # first test target is row 80; its inputs are rows 72..79 (all val + train tail)
    np.testing.assert_array_equal(w["test"].X[0], np.arange(72, 80))
    assert w["test"].Y[0, 0] == 80


def test_targets_never_leave_their_split():
    s = chronological_split(_monthly(100))
    for h in (1, 6):
        w = split_windows(s, seq_len=8, horizon=h)
        assert w["train"].Y.max() < 70           # train targets: rows 0..69
        assert w["val"].Y.min() >= 70 and w["val"].Y.max() < 80
        assert w["test"].Y.min() >= 80


def test_multi_horizon_drops_last_h_minus_1_starts():
    s = chronological_split(_monthly(100))
    w = split_windows(s, seq_len=8, horizon=6)
    assert len(w["test"]) == 20 - 5
    assert w["test"].Y.shape == (15, 6)


def test_target_start_is_the_target_date():
    s = chronological_split(_monthly(100))
    w = split_windows(s, seq_len=8, horizon=1)
    assert w["test"].target_start[0] == s.test["start_date"].iloc[0]


def test_lookbacks_come_from_mm_tsflib_and_time_mmd_not_invented():
    assert LOOKBACK == {"weekly": 24, "monthly": 8}
    assert lookback_for("Health_US") == 24 and lookback_for("Economy") == 8


def test_unknown_domain_raises():
    with pytest.raises(KeyError):
        lookback_for("Atlantis")
