"""Tests for src/data/loader.py.

Synthetic fixtures pin down the alignment rules exactly (a tiny fake Time-MMD on disk);
the final test is a smoke check against the real data, skipped if it isn't downloaded.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import DEFAULT_ROOT, align_text, load_domain, load_numerical, load_text


def _write_domain(root: Path, name: str, numerical: pd.DataFrame, report: pd.DataFrame) -> None:
    (root / "numerical" / name).mkdir(parents=True)
    (root / "textual" / name).mkdir(parents=True)
    numerical.to_csv(root / "numerical" / name / f"{name}.csv", index=False)
    report.to_csv(root / "textual" / name / f"{name}_report.csv")
    report.to_csv(root / "textual" / name / f"{name}_search.csv")


@pytest.fixture
def fake_root(tmp_path):
    # Monthly series stored newest-first, like Climate, to check the loader sorts it.
    num = pd.DataFrame({
        "start_date": ["2020-03-01", "2020-02-01", "2020-01-01"],
        "end_date": ["2020-03-31", "2020-02-29", "2020-01-31"],
        "OT": [3.0, 2.0, 1.0],
    })
    report = pd.DataFrame({
        "start_date": ["2020-01-05", "2020-01-20", "2020-02-10", "2019-06-01", "2020-03-02"],
        "end_date": ["2020-01-05", "2020-01-20", "2020-02-10", "2019-06-01", "2020-03-02"],
        "fact": ["jan a", "jan b", None, "before series", "mar"],
        "preds": ["LLM guess"] * 5,
    })
    _write_domain(tmp_path, "Toy", num, report)
    return tmp_path


def test_sorted_oldest_first(fake_root):
    df = load_domain("Toy", root=fake_root)
    assert df["start_date"].is_monotonic_increasing
    assert df["value"].tolist() == [1.0, 2.0, 3.0]


def test_lag_one_gives_previous_period_text(fake_root):
    df = load_domain("Toy", root=fake_root)
    # Jan has no earlier period; Feb gets Jan's two facts; Mar gets Feb's (whose fact was null).
    assert df.loc[0, "text"] is None
    assert df.loc[1, "text"] == "jan a\njan b"
    assert df.loc[1, "n_texts"] == 2
    assert df.loc[2, "text"] is None
    assert df["has_text"].tolist() == [False, True, False]


def test_no_text_row_precedes_its_period(fake_root):
    """The leakage guard: every text attached to row t started before row t's period."""
    df = load_domain("Toy", root=fake_root)
    text = load_text("Toy", root=fake_root)
    for _, row in df[df["has_text"]].iterrows():
        for fact in row["text"].split("\n"):
            assert (text.loc[text["fact"] == fact, "start_date"] < row["start_date"]).all()


def test_lag_zero_is_same_period(fake_root):
    df = load_domain("Toy", lag=0, root=fake_root)
    assert df.loc[0, "text"] == "jan a\njan b"
    assert df.loc[2, "text"] == "mar"


def test_preds_never_loaded(fake_root):
    df = load_domain("Toy", root=fake_root)
    assert not df["text"].dropna().str.contains("LLM guess").any()


def test_null_fact_dropped_and_counted(fake_root):
    assert load_text("Toy", root=fake_root).attrs["n_empty_text_dropped"] == 1


def test_text_before_series_is_unaligned(fake_root):
    assert load_domain("Toy", root=fake_root).attrs["n_text_unaligned"] == 1


def test_negative_lag_rejected(fake_root):
    with pytest.raises(ValueError):
        load_domain("Toy", lag=-1, root=fake_root)


def test_unknown_domain_and_source_rejected(fake_root):
    with pytest.raises(ValueError):
        load_domain("Nope", root=fake_root)
    with pytest.raises(ValueError):
        load_domain("Toy", source="tweets", root=fake_root)


def test_date_fallback_and_duplicate_dates(tmp_path):
    # Health_AFR has no start_date column; Health_US has duplicate dates (week 53 / week 1).
    num = pd.DataFrame({
        "date": ["2020-01-06", "2020-01-13", "2020-01-13"],
        "end_date": ["2020-01-12", "2020-01-19", "2020-01-19"],
        "OT": [1.0, 53.0, 1.5],
    })
    report = pd.DataFrame({"start_date": [], "end_date": [], "fact": [], "preds": []})
    _write_domain(tmp_path, "Weekly", num, report)
    df = load_numerical("Weekly", root=tmp_path)
    assert len(df) == 2
    assert df["value"].tolist() == [1.0, 1.5]  # later row in the file wins
    assert df.attrs["n_duplicate_dates_dropped"] == 1


@pytest.mark.skipif(not (DEFAULT_ROOT / "numerical").exists(), reason="Time-MMD not downloaded")
@pytest.mark.parametrize("source", ["report", "search"])
def test_real_economy_smoke(source):
    df = load_domain("Economy", source=source)
    assert df["start_date"].is_monotonic_increasing
    assert df["start_date"].is_unique
    assert df["has_text"].any()
    assert df.loc[0, "text"] is None  # first period can never have lagged text
