"""P1.5 timing check: is every text fed to a forecast actually available before that forecast?

The loader gives row t the texts assigned to period t-1 (lag 1). When forecasting row t,
the forecast window starts at row t's ``start_date``. Leakage = a text that could not have
existed yet at that moment, i.e. it describes (part of) the period being forecast.

Two independent checks, per text that reaches a forecast:

1. **Metadata** — the text's ``end_date`` must be strictly before the forecast window start.
   This is the requested check. A text dated Jan 1 → Dec 31 (an annual summary) fails it
   when fed to February.

2. **Content** — explicit dates written *inside* the text must be before the window start.
   Metadata can look fine while the text is not: an Energy report dated 12-12 → 12-16
   quotes a price "on December 19". Three granularities are checked, each conservatively
   (a mention is flagged only if its *earliest* possible day is on/after the window start):
     full date  "April 27, 2020"  -> that day
     month-year "March 2012"      -> the 1st of that month
     bare year  "2012"            -> Jan 1 of that year
   The content check is a heuristic: it can miss relative phrasing ("next week") and could
   flag a quoted forecast ("expected to reach X by 2025"). Flagged texts are printed so a
   human can judge them.

Known Time-MMD year typos (``KNOWN_YEAR_TYPOS``) are reported separately and do not fail
the check: e.g. a July 2010 report says "July 2020", but its $42.8B figure is July 2010's.
Their content is correct for their period, so they are kept in the data. Any *new* content
flag still fails the check.

Run:  python -m src.data.check_timing [--anchor start|end]
"""

from __future__ import annotations

import argparse
import logging
import re
import sys

import pandas as pd

from src.data.loader import ANCHORS, assign_periods, load_numerical, load_text

SELECTED_DOMAINS = ("Economy", "SocialGood", "Traffic", "Health_US")

# (domain, start_date) of reports whose text states a wrong year; verified by hand in P1.5
# against the figures they quote. See PROJECT_STATUS.md §6.
KNOWN_YEAR_TYPOS = {
    ("Economy", "2001-03-01"), ("Economy", "2001-04-01"), ("Economy", "2001-11-01"),
    ("Economy", "2010-07-01"), ("Economy", "2010-09-01"), ("Economy", "2011-11-01"),
    ("SocialGood", "2010-08-01"),
}

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
_MONTH_RE = "(" + "|".join(_MONTHS) + ")"
FULL_DATE = re.compile(rf"\b{_MONTH_RE}\s+(\d{{1,2}}),?\s+((?:19|20)\d\d)\b", re.I)
MONTH_YEAR = re.compile(rf"\b{_MONTH_RE}\s+((?:19|20)\d\d)\b", re.I)
YEAR = re.compile(r"(?<![\d-])((?:19|20)\d\d)(?![\d])")


def mentioned_dates(text: str) -> list[tuple[str, pd.Timestamp]]:
    """Every explicit date in ``text`` as (matched string, earliest day it can refer to).

    Spans already matched at a finer granularity are removed before the coarser pass, so
    "April 27, 2020" yields one full date, not also "2020" as a year.
    """
    found, rest = [], text
    for pattern, to_ts in (
        (FULL_DATE, lambda m: pd.Timestamp(int(m[3]), _MONTHS[m[1].lower()], int(m[2]))),
        (MONTH_YEAR, lambda m: pd.Timestamp(int(m[2]), _MONTHS[m[1].lower()], 1)),
        (YEAR, lambda m: pd.Timestamp(int(m[1]), 1, 1)),
    ):
        for m in pattern.finditer(rest):
            try:
                found.append((m[0], to_ts(m)))
            except ValueError:  # e.g. "February 30, 2020"
                continue
        rest = pattern.sub(" ", rest)
    return found


def fed_texts(domain: str, source: str = "report", lag: int = 1, anchor: str = "end") -> pd.DataFrame:
    """Every text that reaches a forecast, with the start of the window it is fed into."""
    num = load_numerical(domain)
    t, _ = assign_periods(num, load_text(domain, source), anchor=anchor)
    t = t[t["period"] + lag < len(num)].copy()  # texts in the last `lag` periods feed nothing
    t["window_start"] = num["start_date"].to_numpy()[t["period"] + lag]
    return t.reset_index(drop=True)


def check_domain(domain: str, source: str = "report", lag: int = 1, anchor: str = "end") -> dict:
    t = fed_texts(domain, source, lag, anchor)

    meta_bad = t[t["end_date"] >= t["window_start"]]

    content_rows = []
    for _, row in t.iterrows():
        future = [s for s, ts in mentioned_dates(row["fact"]) if ts >= row["window_start"]]
        if future:
            known = (domain, str(row["start_date"].date())) in KNOWN_YEAR_TYPOS
            content_rows.append({**row.to_dict(), "future_mentions": future, "known_typo": known})
    flagged = pd.DataFrame(content_rows, columns=[*t.columns, "future_mentions", "known_typo"])
    content_bad = flagged[~flagged["known_typo"].astype(bool)]

    return dict(domain=domain, n_texts=len(t), metadata_violations=len(meta_bad),
                content_violations=len(content_bad), known_typos=int(flagged["known_typo"].sum()),
                meta_bad=meta_bad, content_bad=content_bad)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--anchor", choices=ANCHORS, default="end")
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--examples", type=int, default=3, help="violations to print per domain")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    sys.stdout.reconfigure(encoding="utf-8")  # report text contains non-cp1252 characters

    print(f"Timing check: source=report lag={args.lag} anchor={args.anchor}\n")
    results = [check_domain(d, lag=args.lag, anchor=args.anchor) for d in SELECTED_DOMAINS]

    for r in results:
        print(f"== {r['domain']}: {r['n_texts']} texts fed to forecasts | "
              f"metadata violations {r['metadata_violations']} | content violations {r['content_violations']}"
              f" | known year typos {r['known_typos']}")
        for _, x in r["meta_bad"].head(args.examples).iterrows():
            print(f"   META    [{x.start_date.date()} -> {x.end_date.date()}] fed to window "
                  f"{x.window_start.date()}: {x.fact[:160]!r}")
        for _, x in r["content_bad"].head(args.examples).iterrows():
            print(f"   CONTENT [{x['start_date'].date()} -> {x['end_date'].date()}] fed to window "
                  f"{x['window_start'].date()} mentions {x['future_mentions']}: {x['fact'][:160]!r}")

    total_meta = sum(r["metadata_violations"] for r in results)
    total_content = sum(r["content_violations"] for r in results)
    total = sum(r["n_texts"] for r in results)
    print(f"\nTOTAL: {total} texts | metadata violations {total_meta} | content violations {total_content}")
    print("VERDICT:", "PASS — no text reaches a forecast before it exists" if total_meta + total_content == 0
          else "FAIL — see violations above")
    return 1 if total_meta + total_content else 0


if __name__ == "__main__":
    raise SystemExit(main())
