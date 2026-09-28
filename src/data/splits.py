"""Chronological train / validation / test splits.

Why chronological, never random: a forecaster must only ever learn from the past. A random
split puts later periods in training and earlier ones in test, so the model "forecasts"
periods it has effectively already seen neighbours of — every test number becomes
optimistic and the paper's results would be invalid. So the rows are cut into three
contiguous blocks in time order: train (oldest) -> val -> test (newest).

Why 70 / 10 / 20: matches MM-TSFlib (``num_train = int(0.7n)``, ``num_test = int(0.2n)``,
val = the rest), so P2 baseline numbers are comparable. Validation is used to tune every
threshold (trend window, gate cutoff); **test is never used for tuning.**

Two things this module deliberately does NOT do, and where they belong:
* **Lookback context.** The first test forecasts need input history from before the test
  block. Using train/val rows as *input context* is fine (that past is known at forecast
  time); what must never happen is a *target* inside the test block leaking into training.
  Windowing with context is done when sequences are built (P2).
* **Normalisation.** Scalers must be fit on train only, then applied to val/test (P2).

By default the split is taken over the text window (first -> last period with text), not
the whole series — see ``loader.text_window``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.loader import load_domain, text_window

TRAIN_FRAC = 0.7
TEST_FRAC = 0.2


@dataclass(frozen=True)
class Splits:
    """Three contiguous, time-ordered blocks of the same domain DataFrame."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def assert_chronological(splits: Splits) -> None:
    """Raise if any val/test period starts before the previous split has ended.

    Compares whole periods (``start_date`` of the later split vs ``end_date`` of the
    earlier one), which is stricter than comparing start dates alone: a test period may
    not even overlap the last training period.
    """
    pairs = [("train", splits.train, "val", splits.val),
             ("val", splits.val, "test", splits.test),
             ("train", splits.train, "test", splits.test)]
    for early_name, early, late_name, late in pairs:
        if len(early) and len(late) and late["start_date"].min() <= early["end_date"].max():
            raise AssertionError(
                f"{late_name} starts {late['start_date'].min().date()}, before {early_name} "
                f"ends {early['end_date'].max().date()} — split is not chronological")


def chronological_split(df: pd.DataFrame, train_frac: float = TRAIN_FRAC,
                        test_frac: float = TEST_FRAC) -> Splits:
    """Cut ``df`` into train / val / test blocks in time order.

    ``df`` must be sorted by ``start_date`` with unique dates (``load_domain`` guarantees
    both); this is checked rather than silently fixed, because an unsorted input here means
    something upstream is broken.
    """
    if not (0 < train_frac < 1 and 0 < test_frac < 1 and train_frac + test_frac < 1):
        raise ValueError(f"need 0 < train, test and train + test < 1; got {train_frac}, {test_frac}")
    if not df["start_date"].is_monotonic_increasing or not df["start_date"].is_unique:
        raise ValueError("df must be sorted by start_date with unique dates — use load_domain()")

    n = len(df)
    n_train = int(n * train_frac)
    n_test = int(n * test_frac)
    n_val = n - n_train - n_test
    if min(n_train, n_val, n_test) == 0:
        raise ValueError(f"{n} rows is too few to split into non-empty train/val/test")

    splits = Splits(
        train=df.iloc[:n_train].reset_index(drop=True),
        val=df.iloc[n_train:n_train + n_val].reset_index(drop=True),
        test=df.iloc[n_train + n_val:].reset_index(drop=True),
    )
    assert_chronological(splits)
    return splits


def load_splits(domain: str, source: str = "report", lag: int = 1,
                within_text_window: bool = True, **split_kwargs) -> Splits:
    """Load a domain and split it. The usual entry point for experiments."""
    df = load_domain(domain, source=source, lag=lag)
    if within_text_window:
        df = text_window(df)
    return chronological_split(df, **split_kwargs)


if __name__ == "__main__":
    import logging
    logging.disable(logging.WARNING)
    for d in ("Economy", "SocialGood", "Traffic", "Health_US"):
        s = load_splits(d)
        spans = {k: f"{v['start_date'].min().date()}..{v['end_date'].max().date()}"
                 for k, v in (("train", s.train), ("val", s.val), ("test", s.test))}
        text = {k: int(v["has_text"].sum()) for k, v in (("train", s.train), ("val", s.val), ("test", s.test))}
        print(f"{d:11s} sizes={s.sizes()} with_text={text}")
        print(f"{'':11s} {spans}")
