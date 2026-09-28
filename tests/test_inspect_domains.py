"""Tests for src/data/inspect_domains.py — mainly that coverage is windowed correctly."""

import pandas as pd

from src.data.inspect_domains import domain_stats


def _loaded(has_text: list[bool]) -> pd.DataFrame:
    n = len(has_text)
    return pd.DataFrame({
        "start_date": pd.date_range("2020-01-01", periods=n, freq="MS"),
        "text": ["abcd" if h else None for h in has_text],
        "has_text": has_text,
    })


def test_window_starts_at_first_text():
    # 6 periods, text starts at index 2, then 3 of the 4 window periods have text.
    s = domain_stats(_loaded([False, False, True, False, True, True]))
    assert s["periods"] == 6
    assert s["window"] == 4
    assert s["usable"] == 3
    assert s["coverage_window"] == 0.75
    assert s["coverage_all"] == 0.5
    assert str(s["first_text"]) == "2020-03-01"
    assert s["len_median"] == 4


def test_window_ends_at_last_text():
    # Text stops at index 3; the two trailing text-free periods must not count.
    s = domain_stats(_loaded([False, True, False, True, False, False]))
    assert s["window"] == 3
    assert s["usable"] == 2
    assert abs(s["coverage_window"] - 2 / 3) < 1e-9
    assert str(s["last_text"]) == "2020-04-01"


def test_no_text_at_all():
    s = domain_stats(_loaded([False, False, False]))
    assert s["usable"] == 0 and s["window"] == 0
    assert s["first_text"] is None and s["last_text"] is None
