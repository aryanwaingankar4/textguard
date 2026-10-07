"""Sliding lookback windows over chronological splits (P2).

Why this lives apart from ``splits.py``: a split decides which *targets* belong to train,
val and test; a window decides which *inputs* a forecast may look at. The two rules differ:

* **Targets** stay inside their own split. A val/test target is never used for training.
* **Inputs** may reach back into earlier splits. The first test forecast needs the last
  ``seq_len`` values before the test block — those are past observations, known at
  forecast time, so using them as context is legitimate. This mirrors MM-TSFlib's
  ``border1 = num_train - seq_len`` (val) and ``len - num_test - seq_len`` (test).

Train windows take context from train only — there is no earlier split. Consequence: the
first ``seq_len`` training rows are context-only, so a domain loses ``seq_len`` training
samples (recorded in PROJECT_STATUS §7).

For horizon H > 1, a window's whole target block must sit inside one split, so the last
H − 1 rows of each split are never a target *start*.

**Text per window** (Models 3–5): the text of the window's *first target row*. The loader has
already lagged text by one period (row t carries period t−1's reports, anchored on their end
date), so that text existed before the forecast period starts. This is the same row MM-TSFlib
reads (``text[s_end]`` in its ``Dataset_Custom.get_text``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.splits import Splits

# Per-frequency lookback (seq_len), not invented here:
# * weekly 24 — MM-TSFlib's only shipped config, scripts/week_health.sh (--seq_len 24).
#   The Time-MMD paper (App. K.4) says 36 for weekly; we follow the code that produced
#   the library, and note the discrepancy.
# * monthly 8 — MM-TSFlib ships no monthly script; Time-MMD paper App. K.4 ("fixed
#   lookback window size of 8") is the authors' own setting.
LOOKBACK = {"weekly": 24, "monthly": 8}
DOMAIN_FREQ = {"Economy": "monthly", "SocialGood": "monthly", "Traffic": "monthly",
               "Health_US": "weekly"}
# Shortest horizon in the Time-MMD paper's result tables — used only for the
# reproduction check against published numbers (main experiments use horizon 1).
PUBLISHED_HORIZON = {"weekly": 12, "monthly": 6}


def lookback_for(domain: str) -> int:
    return LOOKBACK[DOMAIN_FREQ[domain]]


@dataclass(frozen=True)
class Windows:
    """Inputs ``X`` (N, seq_len), targets ``Y`` (N, horizon), and where each target sits.

    ``target_start`` is the ``start_date`` of the first target period: the sample id used
    in raw-prediction CSVs, so every prediction can be traced to a date.
    ``text`` (object array, ``str | None`` per window) is the text available for that
    forecast, or ``None`` when the splits carry no text column.
    """
    X: np.ndarray
    Y: np.ndarray
    target_start: pd.DatetimeIndex
    text: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.X)


def make_windows(values: np.ndarray, dates: pd.Series, target_lo: int, target_hi: int,
                 seq_len: int, horizon: int, texts: pd.Series | None = None) -> Windows:
    """Windows whose target blocks start in ``[target_lo, target_hi - horizon]``.

    ``values`` is the full contiguous series; ``target_lo:target_hi`` is one split's row
    range. A target block may not run past ``target_hi`` (it would enter the next split),
    and its input window may not start before row 0.
    """
    starts = range(max(target_lo, seq_len), target_hi - horizon + 1)
    X = np.array([values[t - seq_len:t] for t in starts], dtype=float).reshape(-1, seq_len)
    Y = np.array([values[t:t + horizon] for t in starts], dtype=float).reshape(-1, horizon)
    text = None
    if texts is not None:
        text = np.array([None if pd.isna(v) else v for v in texts.iloc[list(starts)]],
                        dtype=object)
    return Windows(X, Y, pd.DatetimeIndex(dates.iloc[list(starts)]), text)


def split_windows(splits: Splits, seq_len: int, horizon: int,
                  value_col: str = "value") -> dict[str, Windows]:
    """Train / val / test windows; val and test take input context from earlier splits."""
    full = pd.concat([splits.train, splits.val, splits.test], ignore_index=True)
    values = full[value_col].to_numpy(dtype=float)
    n_tr, n_va = len(splits.train), len(splits.val)
    bounds = {"train": (0, n_tr), "val": (n_tr, n_tr + n_va), "test": (n_tr + n_va, len(full))}
    texts = full["text"] if "text" in full.columns else None
    return {name: make_windows(values, full["start_date"], lo, hi, seq_len, horizon, texts)
            for name, (lo, hi) in bounds.items()}
