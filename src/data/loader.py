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

* **Lag of 1 period.** Text timestamps mark the start of the period a text covers, not when
  it was published (an Energy report dated 12-12..12-16 quotes a price "on December 19").
  So row t receives the texts dated within period t-1: by the time period t begins, those
  texts have certainly been published. This is what keeps text from leaking the future.

* **Missing text is explicit.** Periods with no text get ``text=None`` and ``has_text=False``
  instead of an empty string or being dropped, so the ``empty`` attack condition (P3) and
  the "% rows with text" statistic (P1.3) can tell "no report existed" apart from
  "report removed by an attack".

* **``source`` is selectable.** ``report`` (curated, directional) is our default. MM-TSFlib
  uses ``search`` snippets, so ``source="search"`` exists to reproduce its baselines in P2.
"""

from __future__ import annotations

import logging
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

    if df.empty:
        logger.warning("%s/%s: no usable text (all %d rows empty)", domain, source, len(raw))
    df.attrs.update(domain=domain, source=source, n_empty_text_dropped=n_dropped)
    return df


def align_text(numerical: pd.DataFrame, text: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Attach text to numerical periods, shifted forward by ``lag`` periods.

    Step 1 — assign each text to the numerical period containing its ``start_date``
    (the last period whose start_date <= the text's start_date, and whose end_date >= it).
    Texts outside every period, e.g. before the series begins, are dropped.
    ``start_date`` is used rather than ``end_date`` because some rows have unreliable end
    dates (a Traffic report ends 360 days before it starts).

    Step 2 — concatenate each period's texts in time order, then shift by ``lag`` so that
    row t carries period t-lag's text. ``lag=0`` is allowed for experiments that measure
    leakage (P1.5), but is not the default for a reason.
    """
    if lag < 0:
        raise ValueError("lag must be >= 0; a negative lag would feed text from the future")

    periods = numerical[["start_date", "end_date"]].reset_index().rename(columns={"index": "period"})
    t = text.sort_values("start_date", kind="stable")
    t = pd.merge_asof(t, periods.rename(columns={"start_date": "p_start", "end_date": "p_end"}),
                      left_on="start_date", right_on="p_start", direction="backward")
    inside = t["period"].notna() & (t["start_date"] <= t["p_end"])
    n_unaligned = int((~inside).sum())
    t = t[inside]

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
    out.attrs.update(lag=lag, n_text_unaligned=n_unaligned)
    return out


def load_domain(domain: str, source: str = "report", lag: int = 1,
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
    return align_text(load_numerical(domain, root), load_text(domain, source, root), lag=lag)
