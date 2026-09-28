"""Load one Time-MMD domain as a single DataFrame: numerical series + aligned text.

Output: one row per numerical period, sorted oldest-first, with columns

    start_date, end_date   period covered by the row (datetime64)
    value                  the target series (Time-MMD's ``OT`` column)
    text                   concatenated ``fact`` text available *before* this period, or None
    n_texts                how many text rows were concatenated into ``text``
    has_text               n_texts > 0

Why each design choice (details and evidence in PROJECT_STATUS.md §6):

* **Sort by date ourselves.** Time-MMD files are not consistently ordered (Climate is
  newest-first, Economy is unsorted). Every later step — chronological splits, lookback
  windows — assumes row order == time order, so we enforce it here, once.

* **``fact`` only, never ``preds``.** ``preds`` is forecasts written by the Time-MMD authors'
  LLM, not real-world text. Our threat model is an adversary tampering with real reports.

* **Anchor on the end date, then lag 1 period.** A text cannot exist before the end of the
  span it describes, so each text is placed in the period containing its end date
  (``anchor="end"``), and row t receives the texts placed in period t-1. Anchoring on the
  start date instead let annual reports (Jan 1 -> Dec 31) feed February's forecast —
  ``check_timing.py`` (P1.5) found 37 such violations; ``anchor="end"`` has 0.

* **Known report defects are fixed here, once.** Five texts whose stored dates are wrong
  (``KNOWN_LEAKERS``) are dropped, and SocialGood's off-topic influenza texts are removed —
  see ``_clean_reports``. Time-MMD year typos ("July 2020" in a July 2010 report) are kept:
  the content is right for its period, only the written year is wrong.

* **Missing text is explicit.** Periods with no text get ``text=None`` and ``has_text=False``
  instead of an empty string or being dropped, so the ``empty`` attack condition (P3) and
  the "% rows with text" statistic (P1.3) can tell "no report existed" apart from
  "report removed by an attack".

* **``source`` is selectable.** ``report`` (curated, directional) is our default. MM-TSFlib
  uses ``search`` snippets, so ``source="search"`` exists to reproduce its baselines in P2.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "Time-MMD"
TEXT_SOURCES = ("report", "search")
TEXT_SEPARATOR = "\n"


def _to_datetime(col: pd.Series) -> pd.Series:
    """Parse dates at one fixed precision. pandas infers a precision per column (an empty
    column comes out as seconds, a filled one as microseconds), and ``merge_asof`` refuses
    to join columns of different precision — this broke on Security, whose report is empty."""
    return pd.to_datetime(col).astype("datetime64[ns]")


def list_domains(root: Path | str = DEFAULT_ROOT) -> list[str]:
    """Domains present on disk (a domain needs both a numerical and a textual folder)."""
    root = Path(root)
    num = {p.name for p in (root / "numerical").iterdir() if p.is_dir()}
    txt = {p.name for p in (root / "textual").iterdir() if p.is_dir()}
    return sorted(num & txt)


def load_numerical(domain: str, root: Path | str = DEFAULT_ROOT) -> pd.DataFrame:
    """Load a domain's target series as ``start_date, end_date, value``, oldest first.

    Column names vary per domain, so only the fields we need are extracted:
    ``start_date`` (falling back to ``date`` — Health_AFR has no ``start_date``),
    ``end_date`` and the target ``OT``. Missing ``OT`` values are kept as NaN rather than
    imputed: whether to interpolate or drop is a modelling decision, not a loading one.
    """
    path = Path(root) / "numerical" / domain / f"{domain}.csv"
    raw = pd.read_csv(path)

    start_col = "start_date" if "start_date" in raw.columns else "date"
    for col in (start_col, "end_date", "OT"):
        if col not in raw.columns:
            raise ValueError(f"{path}: expected column {col!r}, found {list(raw.columns)}")

    df = pd.DataFrame({
        "start_date": _to_datetime(raw[start_col]),
        "end_date": _to_datetime(raw["end_date"]),
        "value": pd.to_numeric(raw["OT"], errors="coerce"),
    })
    # Stable sort keeps the file's order among equal dates, so keep="last" below picks the
    # row that came later in the file (Health_US: next year's week 1 over week 53).
    df = df.sort_values("start_date", kind="stable")

    n_dup = int(df["start_date"].duplicated().sum())
    if n_dup:
        logger.warning("%s: dropping %d duplicate start_date rows (kept the later one)", domain, n_dup)
        df = df.drop_duplicates("start_date", keep="last")

    n_nan = int(df["value"].isna().sum())
    if n_nan:
        logger.warning("%s: %d periods have a missing target value (kept as NaN)", domain, n_nan)

    df = df.reset_index(drop=True)
    df.attrs.update(domain=domain, n_duplicate_dates_dropped=n_dup, n_missing_values=n_nan)
    return df


def load_text(domain: str, source: str = "report", root: Path | str = DEFAULT_ROOT) -> pd.DataFrame:
    """Load a domain's text as ``start_date, end_date, fact``, oldest first.

    Rows whose ``fact`` is null or blank are dropped (and counted): they carry no text to
    align or attack. ``preds`` is deliberately not loaded — see the module docstring.
    """
    if source not in TEXT_SOURCES:
        raise ValueError(f"source must be one of {TEXT_SOURCES}, got {source!r}")
    path = Path(root) / "textual" / domain / f"{domain}_{source}.csv"
    raw = pd.read_csv(path, usecols=["start_date", "end_date", "fact"])

    fact = raw["fact"].astype("string").str.strip()
    keep = fact.notna() & (fact != "") & (fact.str.upper() != "NA")
    n_dropped = int((~keep).sum())

    df = pd.DataFrame({
        "start_date": _to_datetime(raw.loc[keep, "start_date"]),
        "end_date": _to_datetime(raw.loc[keep, "end_date"]),
        "fact": fact[keep].astype(str),
    }).sort_values("start_date", kind="stable").reset_index(drop=True)

    cleaning = {}
    if source == "report":
        df, cleaning = _clean_reports(domain, df)

    if df.empty:
        logger.warning("%s/%s: no usable text (all %d rows empty)", domain, source, len(raw))
    df.attrs.update(domain=domain, source=source, n_empty_text_dropped=n_dropped, **cleaning)
    return df


# --- Known Time-MMD report defects (found in P1.5; evidence in PROJECT_STATUS.md §6) ---------

# Texts whose content describes the period being forecast even under anchor="end", because
# their stored dates are wrong. Listed individually (not caught by a rule) so each exclusion
# is auditable. Keyed by (start_date, end_date) exactly as stored in Time-MMD.
KNOWN_LEAKERS: dict[str, list[tuple[str, str, str]]] = {
    "Health_US": [
        ("2009-01-01", "2010-01-01", "2009-10 season summary; describes H1N1 through Oct 2010"),
        ("2010-01-01", "2011-01-01", "2010-11 season summary; describes Feb-Apr 2011"),
        ("2011-01-01", "2012-01-01", "2011-12 season summary; describes Feb-Jun 2012"),
    ],
    "Traffic": [
        ("2020-12-28", "2020-01-03", "week 53 to Jan 3 2021; end_date stored a year early"),
        ("2021-12-27", "2021-01-02", "week 52 to Jan 2 2022; end_date stored a year early"),
    ],
}

# SocialGood: Time-MMD's summariser used the influenza prompt on 41% of unemployment reports.
# Most are empty ("NA", "no mention of influenza"), but some hold real unemployment figures
# under the wrong header — including 2020-22 COVID-era reports — so the rule is: drop
# influenza-headed texts unless they state an unemployment figure; strip the header from
# the ones kept, so no model or direction classifier ever sees the word "influenza" there.
_WRONG_HEADER = re.compile(r"^\W*objective facts about the influenza situation\W*", re.I)
_HAS_UNEMPLOYMENT_FIGURE = re.compile(r"unemploy.*?\d+(?:\.\d+)?\s*(?:%|percent)", re.I | re.S)


def _clean_reports(domain: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats = {"n_known_leakers_dropped": 0, "n_off_topic_dropped": 0, "n_wrong_header_stripped": 0}

    leakers = KNOWN_LEAKERS.get(domain, [])
    if leakers:
        bad = pd.MultiIndex.from_tuples([(pd.Timestamp(s), pd.Timestamp(e)) for s, e, _ in leakers])
        is_leaker = pd.MultiIndex.from_frame(df[["start_date", "end_date"]]).isin(bad)
        if is_leaker.sum() != len(leakers):  # the list must match the data exactly, or it is stale
            raise RuntimeError(f"{domain}: expected {len(leakers)} known leakers, found {is_leaker.sum()}")
        df = df[~is_leaker]
        stats["n_known_leakers_dropped"] = int(is_leaker.sum())

    if domain == "SocialGood":
        headed = df["fact"].str.contains(_WRONG_HEADER)
        real = df["fact"].str.contains(_HAS_UNEMPLOYMENT_FIGURE)
        stats["n_off_topic_dropped"] = int((headed & ~real).sum())
        stats["n_wrong_header_stripped"] = int((headed & real).sum())
        df = df[~headed | real].copy()
        df["fact"] = df["fact"].str.replace(_WRONG_HEADER, "", regex=True).str.strip()

    return df.reset_index(drop=True), stats


ANCHORS = ("start", "end")


def assign_periods(numerical: pd.DataFrame, text: pd.DataFrame,
                   anchor: str = "end") -> tuple[pd.DataFrame, int]:
    """Assign each text to the numerical period containing its anchor date.

    ``anchor="start"``: the text's ``start_date``.
    ``anchor="end"``: the later of ``start_date`` and ``end_date`` — the earliest moment the
    whole text could exist. ``max`` rather than ``end_date`` alone because some rows have
    an end before their start (a Traffic report ends 360 days before it starts).
    Why this matters: an annual report dated Jan 1 → Dec 31 is placed in January under
    ``"start"`` and would then feed February's forecast with the full year's numbers —
    ``check_timing.py`` (P1.5) found exactly this in SocialGood and Health_US.

    A text belongs to the last period whose start_date <= the anchor date, provided the
    period's end_date >= it. Texts outside every period are dropped and counted.

    Returns the kept texts with added columns ``period`` (row index into ``numerical``),
    ``p_start`` and ``p_end``, plus the number dropped. Exposed separately so the timing
    check (P1.5) audits exactly the assignment the loader uses.
    """
    if anchor not in ANCHORS:
        raise ValueError(f"anchor must be one of {ANCHORS}, got {anchor!r}")
    periods = numerical[["start_date", "end_date"]].reset_index().rename(
        columns={"index": "period", "start_date": "p_start", "end_date": "p_end"})
    t = text.copy()
    t["anchor_date"] = t["start_date"] if anchor == "start" else t[["start_date", "end_date"]].max(axis=1)
    t = t.sort_values(["anchor_date", "start_date"], kind="stable")
    t = pd.merge_asof(t, periods, left_on="anchor_date", right_on="p_start", direction="backward")
    inside = t["period"].notna() & (t["anchor_date"] <= t["p_end"])
    t = t[inside].copy()
    t["period"] = t["period"].astype(int)
    return t, int((~inside).sum())


def align_text(numerical: pd.DataFrame, text: pd.DataFrame, lag: int = 1,
               anchor: str = "end") -> pd.DataFrame:
    """Attach text to numerical periods, shifted forward by ``lag`` periods.

    Step 1 — ``assign_periods``: each text goes to the period containing its anchor date.

    Step 2 — concatenate each period's texts in time order, then shift by ``lag`` so that
    row t carries period t-lag's text. ``lag=0`` is allowed for experiments that measure
    leakage (P1.5), but is not the default for a reason.
    """
    if lag < 0:
        raise ValueError("lag must be >= 0; a negative lag would feed text from the future")

    t, n_unaligned = assign_periods(numerical, text, anchor=anchor)

    per_period = t.groupby("period")["fact"].agg([TEXT_SEPARATOR.join, "size"])
    per_period.columns = ["text", "n_texts"]
    per_period.index = per_period.index.astype(int)

    out = numerical.copy()
    joined = per_period.reindex(range(len(out)))
    out["text"] = joined["text"].shift(lag).astype(object)
    out["text"] = out["text"].where(out["text"].notna(), None)
    out["n_texts"] = joined["n_texts"].shift(lag).fillna(0).astype(int).to_numpy()
    out["has_text"] = out["n_texts"] > 0

    out.attrs.update(numerical.attrs)
    out.attrs.update(text.attrs)
    out.attrs.update(lag=lag, anchor=anchor, n_text_unaligned=n_unaligned)
    return out


def text_window(df: pd.DataFrame) -> pd.DataFrame:
    """Rows from the first to the last period that has text, inclusive.

    Text experiments can only run where text exists. Time-MMD's numbers usually start
    years before the reports, and some reports stop early (Health_US: 2020, numbers run to
    2024). Splitting the whole series would put Health_US's entire test set in years with
    no text at all, so splits (P1.4) are taken over this window instead.
    Gaps *inside* the window stay, with ``has_text=False`` — they are real missing reports.
    """
    has = df["has_text"].to_numpy()
    if not has.any():
        return df.iloc[0:0].reset_index(drop=True)
    first = int(has.argmax())
    last = len(has) - 1 - int(has[::-1].argmax())
    return df.iloc[first:last + 1].reset_index(drop=True)


def load_domain(domain: str, source: str = "report", lag: int = 1, anchor: str = "end",
                root: Path | str = DEFAULT_ROOT) -> pd.DataFrame:
    """Load one domain's numerical series with aligned text. The main entry point.

    Example::

        df = load_domain("Economy")              # report text, lag 1
        df = load_domain("Economy", "search")    # MM-TSFlib's text source
    """
    root = Path(root)
    available = list_domains(root)
    if domain not in available:
        raise ValueError(f"unknown domain {domain!r}; available: {available}")
    return align_text(load_numerical(domain, root), load_text(domain, source, root),
                      lag=lag, anchor=anchor)
