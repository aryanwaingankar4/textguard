"""R4 held-out test windows — Energy and FNSPID (PROJECT_STATUS §10g "Held-out freeze").

Why this is separate from ``splits.py`` / ``windows.py``: the held-out domains have **no train or
val split**. R4's targets are zero-shot LLMs and TextGuard is training-free (v2 is developed on the
*original* domains' val only), so **every text period is a test period**, and the lookback and σ
come from numbers **before** the text period — they never touch a test target.

One row per forecast window: ``domain, series, sample_id, origin, target_date, lookback (24
values, raw units), y_true, text, has_text, sigma, sigma_alt``.

* **Energy** (weekly, lag 2 — §6 "R4 held-out domains"): every period of the text window is a
  target; ``text`` is the report the loader aligns at lag 2 (with the known leaker dropped);
  lookback = the 24 values before the target (reaching into the pre-text history);
  ``sigma`` = std of the 520 weeks before the text window (Aryan, 2026-10-12); ``sigma_alt`` =
  all pre-text weeks (descriptive only).
* **FNSPID** (weekly, per stock — §6 "R4 FNSPID"): closes **as traded**; week k's origin O_k is its
  last trading day; target = the close of week k+1; lookback = the 24 weekly closes ending at O_k;
  ``text`` = week k's **bundle**: the stock's non-list Benzinga headlines dated O_{k−1} ≤ d < O_k,
  oldest first, one per line, exact duplicates removed. Dates are date-only, so a headline dated
  on the origin day may come after that close — it goes to the next bundle; one dated O_{k−1} was
  surely out before O_k's close. (= the frozen "Fri(k−1) … Thu(k)" whenever Fridays trade.)
  Windows: bundles inside the period (O_{k−1} ≥ 2018-06-01, O_k ≤ 2020-05-31) whose target close
  exists (GOOGL's file ends 2020-04-01). ``sigma`` = std of the 104 weekly closes before the period
  (Aryan, 2026-10-12); ``sigma_alt`` = the 520 weeks before it (the pre-stated sensitivity table).

σ uses ``np.std`` (population), as ``adaptive.train_stats`` does for the original domains.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src.data import fnspid
from src.data.loader import load_domain, text_window

LOOKBACK = 24
ENERGY_LAG = 2
ENERGY_SIGMA_WEEKS = 520
FNSPID_SIGMA_WEEKS, FNSPID_SIGMA_ALT_WEEKS = 104, 520
COLUMNS = ["domain", "series", "sample_id", "origin", "target_date", "lookback", "y_true", "text",
           "has_text", "sigma", "sigma_alt", "history"]
# ``history`` (added 2026-10-16, R4 pre-registration): every value of the series before the target —
# the longer past TextGuard v2 is given explicitly for year-over-year claims. It ends with
# ``lookback``; nothing at or after the target is in it.


def energy_windows() -> pd.DataFrame:
    df = load_domain("Energy", lag=ENERGY_LAG)
    vals = df["value"].to_numpy(float)
    first = int(df["has_text"].to_numpy().argmax())        # first period of the text window
    win = text_window(df)
    pre = vals[:first]
    sigma, sigma_alt = float(np.std(pre[-ENERGY_SIGMA_WEEKS:])), float(np.std(pre))
    rows = []
    for t in range(first, first + len(win)):
        rows.append(dict(domain="Energy", series="Energy",
                         sample_id=df["start_date"].iloc[t].strftime("%Y-%m-%d"),
                         origin=df["start_date"].iloc[t - 1], target_date=df["start_date"].iloc[t],
                         lookback=vals[t - LOOKBACK:t].copy(), y_true=float(vals[t]),
                         text=df["text"].iloc[t] if df["has_text"].iloc[t] else None,
                         has_text=bool(df["has_text"].iloc[t]), sigma=sigma, sigma_alt=sigma_alt,
                         history=vals[:t].copy()))
    return pd.DataFrame(rows, columns=COLUMNS)


def energy_max_step() -> float:
    """Largest weekly |change| of the Energy target **before** its text window — A3's plausibility
    bound (Aryan, 2026-10-12). Uses no test-period value."""
    df = load_domain("Energy", lag=ENERGY_LAG)
    first = int(df["has_text"].to_numpy().argmax())
    return float(np.abs(np.diff(df["value"].to_numpy(float)[:first])).max())


def weekly_closes(daily: pd.Series) -> pd.DataFrame:
    """One row per calendar week (ending Friday): its last trading day and that day's close."""
    d = daily.dropna()
    g = d.groupby(d.index.to_period("W-FRI"))
    return pd.DataFrame({"origin": g.apply(lambda s: s.index[-1]), "close": g.last()}).reset_index(drop=True)


def bundle(headlines: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> str | None:
    h = headlines[(headlines["day"] >= lo) & (headlines["day"] < hi)]
    h = h[~h["title"].str.contains(fnspid.LIST_HEADLINE)]
    titles = list(dict.fromkeys(" ".join(t.split()) for t in h["title"]))   # dedupe, keep order
    return "\n".join(titles) if titles else None


def fnspid_windows(tickers=fnspid.SELECTED) -> pd.DataFrame:
    rows = []
    start, end = map(pd.Timestamp, fnspid.PERIOD)
    for tk in tickers:
        w = weekly_closes(fnspid.load_prices(tk))
        heads = fnspid.load_headlines(tk)
        before = w[w["origin"] < start]["close"].to_numpy(float)
        sigma = float(np.std(before[-FNSPID_SIGMA_WEEKS:]))
        sigma_alt = float(np.std(before[-FNSPID_SIGMA_ALT_WEEKS:]))
        for k in range(LOOKBACK, len(w) - 1):
            o_prev, o, tgt = w["origin"].iloc[k - 1], w["origin"].iloc[k], w["origin"].iloc[k + 1]
            if o_prev < start or o > end:
                continue
            text = bundle(heads, o_prev, o)
            rows.append(dict(domain="FNSPID", series=tk, sample_id=tgt.strftime("%Y-%m-%d"),
                             origin=o, target_date=tgt,
                             lookback=w["close"].iloc[k - LOOKBACK + 1:k + 1].to_numpy(float),
                             y_true=float(w["close"].iloc[k + 1]), text=text, has_text=text is not None,
                             sigma=sigma, sigma_alt=sigma_alt,
                             history=w["close"].iloc[:k + 1].to_numpy(float)))
    return pd.DataFrame(rows, columns=COLUMNS)


@lru_cache(maxsize=None)
def _windows() -> pd.DataFrame:
    return pd.concat([energy_windows(), fnspid_windows()], ignore_index=True)


def windows() -> pd.DataFrame:
    """All R4 held-out windows (Energy + the 10 FNSPID stocks)."""
    return _windows().copy()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    w = windows()
    print(w.groupby(["domain", "series"]).agg(windows=("sample_id", "size"), with_text=("has_text", "sum"),
                                              first=("sample_id", "min"), last=("sample_id", "max"),
                                              sigma=("sigma", "first"), sigma_alt=("sigma_alt", "first")))
