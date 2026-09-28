"""Per-domain text coverage statistics, used to choose which domains to experiment on (P1.3).

Why coverage is measured inside a *window*, not over the whole series: Time-MMD's numbers
often start decades before its text (Energy numbers begin 1993, reports 2011). Over the
whole series, Energy report text covers ~22% of periods, which looks unusable — but the
experiments will only run where text exists, so the question that matters is "once text
starts, how consistently is it there?". The window runs from the first to the last period
that has (lagged) text. Ending at the last text, not the series end, matters because some
sources stop early (Health_US reports end in 2020 while its numbers run to 2024) — those
text-free trailing years can't be used for text experiments and shouldn't dilute coverage.

Columns:
    periods          total periods in the series
    first_text       start_date of the first period with text
    last_text        start_date of the last period with text
    window           periods from first_text to last_text, inclusive
    usable           periods in the window that actually have text  (sort key)
    coverage_window  usable / window     — decides domain selection
    coverage_all     usable / periods    — shown only to contrast; misleading on its own
    len_median/mean  characters of text per period, as the model sees it (all of a period's
                     facts concatenated), so it reflects model input length

Run:  python -m src.data.inspect_domains
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.loader import DEFAULT_ROOT, TEXT_SOURCES, list_domains, load_domain, text_window


def domain_stats(df: pd.DataFrame) -> dict:
    """Coverage statistics for one loaded domain (output of ``load_domain``)."""
    periods = len(df)
    win = text_window(df)
    if win.empty:
        return dict(periods=periods, first_text=None, last_text=None, window=0, usable=0,
                    coverage_window=0.0, coverage_all=0.0, len_median=0.0, len_mean=0.0)

    window = len(win)
    usable = int(win["has_text"].sum())
    lengths = win.loc[win["has_text"], "text"].str.len()
    return dict(
        periods=periods,
        first_text=win["start_date"].iloc[0].date(),
        last_text=win["start_date"].iloc[-1].date(),
        window=window,
        usable=usable,
        coverage_window=usable / window,
        coverage_all=usable / periods,
        len_median=float(lengths.median()),
        len_mean=float(lengths.mean()),
    )


def inspect_all(root: Path | str = DEFAULT_ROOT, sources=TEXT_SOURCES, lag: int = 1) -> pd.DataFrame:
    """One row per domain x source, sorted by usable periods (most first)."""
    rows = []
    for domain in list_domains(root):
        for source in sources:
            df = load_domain(domain, source=source, lag=lag, root=root)
            rows.append(dict(domain=domain, source=source, **domain_stats(df)))
    return (pd.DataFrame(rows)
            .sort_values(["usable", "coverage_window"], ascending=False)
            .reset_index(drop=True))


def main() -> None:
    logging.disable(logging.WARNING)  # loader warnings are per-domain noise here
    table = inspect_all()
    fmt = table.assign(
        coverage_window=table["coverage_window"].map("{:.1%}".format),
        coverage_all=table["coverage_all"].map("{:.1%}".format),
        len_median=table["len_median"].map("{:.0f}".format),
        len_mean=table["len_mean"].map("{:.0f}".format),
        first_text=table["first_text"].fillna("-"),
        last_text=table["last_text"].fillna("-"),
    )
    print(fmt.to_string(index=False))


if __name__ == "__main__":
    main()
