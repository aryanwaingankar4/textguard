"""FNSPID (R4 held-out domain): fetch only what the frozen spec needs, and re-derive the frozen
stock selection (PROJECT_STATUS §10g "Held-out freeze", §6 "R4 FNSPID").

What is downloaded, and why nothing else:

* **The Benzinga headline block** of ``Stock_news/All_external.csv`` — the ticker-sorted rows at
  bytes 0–331,521,297 (332 MB of FNSPID's ~29 GB), streamed once. Ranking all S&P 500 stocks by
  news days needs every ticker's headlines in that block; nothing else in FNSPID is read. Kept on
  disk: the headlines of the 15 frozen tickers (10 selected + 5 reserve) and the per-ticker counts
  that reproduce the ranking.
* **15 price files** read out of ``Stock_price/full_history.zip`` by HTTP byte ranges (the zip's
  central directory + those members only, ~6 MB).

The selection rule is re-applied to the downloaded block and **asserted** to give the frozen
tickers, so a changed upstream file fails loudly instead of silently changing the test set.

Licence: FNSPID is CC BY-NC-4.0 — academic use only; ``data/`` is gitignored and never released.

Run: ``python -m src.data.fnspid`` → ``data/FNSPID/``.
"""

from __future__ import annotations

import csv
import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "FNSPID"
BASE = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/"
NEWS = "Stock_news/All_external.csv"
PRICES = "Stock_price/full_history.zip"
PRICES_SIZE = 589_525_596
BENZINGA_END = 331_521_297          # last byte of the ticker-tagged Benzinga block (found by probing)
SP500_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
             "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")

PERIOD = ("2018-06-01", "2020-05-31")
SNAPSHOT = "2018-06-01"
SELECTED = ["NFLX", "GOOGL", "MRK", "NVDA", "PFE", "JNJ", "MU", "FDX", "PCG", "LLY"]
RESERVE = ["GILD", "BMY", "QCOM", "BAC", "XOM"]
# list / market-wide headlines (several stocks at once) — the frozen "counted headlines" filter
LIST_HEADLINE = re.compile(
    r"^\d+\s+(?:Biggest|Stocks|Companies)|Stocks (?:Moving|That Hit|To Watch)|Mid-Day|Pre-Market|"
    r"After-Hours|Movers|Earnings Scheduled|Picks|Must Watch|Benzinga's Top|Market Wrap|"
    r"Biggest Price Target", re.I)
DUAL_CLASSES = [("GOOG", "GOOGL"), ("FOX", "FOXA"), ("NWS", "NWSA"), ("DISCK", "DISCA"), ("UA", "UAA")]

# As-traded factors (§10g): FNSPID's ``close`` carries every split-type adjustment up to its
# download; within the period prices are multiplied back so headline price targets and the series
# share units. NVDA 4:1 (2021-07-20), MRK Organon 1.048 (2021-06-03), PFE Viatris 1.054 (2020-11-17).
AS_TRADED = {"NVDA": 4.0, "MRK": 1.048, "PFE": 1.054}


def _stream_benzinga() -> pd.DataFrame:
    """Every ticker-tagged Benzinga headline: Date, ticker, title, publisher, url."""
    req = urllib.request.Request(BASE + NEWS, headers={"Range": f"bytes=0-{BENZINGA_END + 200_000}"})
    csv.field_size_limit(10**9)
    rows = []
    with urllib.request.urlopen(req, timeout=900) as r:
        rd = csv.reader(io.TextIOWrapper(r, encoding="utf-8", errors="replace", newline=""))
        next(rd)
        for row in rd:
            if len(row) < 5:
                continue
            if not row[2]:
                if row[0][:2] in ("19", "20"):
                    break                    # first untagged row: the block has ended
                continue
            rows.append((row[0], row[2], row[1], row[4], row[3]))
    return pd.DataFrame(rows, columns=["Date", "ticker", "title", "publisher", "url"])


def rank(block: pd.DataFrame, members: set[str]) -> pd.DataFrame:
    """The frozen selection rule: S&P 500 members on the snapshot date, counted (non-list)
    headlines in the period, one share class per company, ranked by distinct news days, then
    headlines, then ticker."""
    day = pd.to_datetime(block["Date"].str[:10])
    p = block[(day >= PERIOD[0]) & (day <= PERIOD[1])].assign(day=day)
    p = p[~p["title"].str.contains(LIST_HEADLINE) & p["ticker"].isin(members)]
    g = p.groupby("ticker").agg(days=("day", "nunique"), n=("title", "size"))
    for a, b in DUAL_CLASSES:
        if a in g.index and b in g.index:
            g = g.drop(a if tuple(g.loc[a]) < tuple(g.loc[b]) else b)
    g = g.reset_index().sort_values(["days", "n", "ticker"], ascending=[False, False, True])
    return g.reset_index(drop=True)


def sp500_members(date: str = SNAPSHOT) -> set[str]:
    with urllib.request.urlopen(SP500_URL, timeout=120) as r:
        h = pd.read_csv(io.BytesIO(r.read()))
    h["date"] = pd.to_datetime(h["date"])
    row = h[h["date"] <= date].sort_values("date").iloc[-1]
    return {t.strip().replace(".", "-") for t in row["tickers"].split(",")}


class _HTTPFile(io.RawIOBase):
    """A seekable read-only view of one remote file, fetched by HTTP byte ranges (for ``zipfile``)."""

    def __init__(self, url: str, size: int):
        self.url, self.size, self.pos = url, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch_prices(tickers) -> None:
    z = zipfile.ZipFile(io.BufferedReader(_HTTPFile(BASE + PRICES, PRICES_SIZE), buffer_size=1 << 16))
    lookup = {n.rsplit("/", 1)[-1].rsplit(".", 1)[0].upper(): n for n in z.namelist()
              if n.startswith("full_history/") and n.endswith(".csv")}
    (OUT / "prices").mkdir(parents=True, exist_ok=True)
    for t in tickers:
        (OUT / "prices" / f"{t}.csv").write_bytes(z.read(lookup[t]))


def load_prices(ticker: str) -> pd.Series:
    """Daily closes, **as traded** (× ``AS_TRADED``), indexed by date, oldest first."""
    df = pd.read_csv(OUT / "prices" / f"{ticker}.csv")
    df.columns = [c.lower() for c in df.columns]
    s = pd.Series(df["close"].to_numpy(float), index=pd.to_datetime(df["date"])).sort_index()
    return s * AS_TRADED.get(ticker, 1.0)


def load_headlines(ticker: str) -> pd.DataFrame:
    """The ticker's Benzinga headlines: ``day`` (date as stated), ``title``, oldest first."""
    h = pd.read_parquet(OUT / "benzinga_frozen_tickers.parquet")
    h = h[h["ticker"] == ticker].copy()
    h["day"] = pd.to_datetime(h["Date"].str[:10])
    return h.sort_values(["day", "Date"], kind="stable").reset_index(drop=True)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    print("streaming the Benzinga block (332 MB) ...")
    block = _stream_benzinga()
    ranking = rank(block, sp500_members())
    ranking.to_csv(OUT / "selection_ranking.csv", index=False)
    top = ranking["ticker"].tolist()
    assert top[:10] == SELECTED and top[10:15] == RESERVE, f"ranking changed: {top[:15]}"
    block[block["ticker"].isin(SELECTED + RESERVE)].to_parquet(OUT / "benzinga_frozen_tickers.parquet")
    print(f"{len(block):,} headlines; frozen selection reproduced: {top[:10]} | reserve {top[10:15]}")
    fetch_prices(SELECTED + RESERVE)
    print("prices ->", OUT / "prices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
