"""Shared plumbing for the text conditions: which windows are attacked, where texts come from,
and how a condition is saved.

Why only val + test windows: attacks happen at inference. Training always used real text, so
the attacked conditions are generated only for the windows that P5/P6 evaluate.

Two tables, built once from the same pipeline as the models (``load_splits`` →
``split_windows``), so a condition's ``sample_id`` joins exactly with every predictions CSV:

* ``windows()`` — one row per val/test window: ``domain, split, sample_id, row, text``.
  ``row`` is the position of the forecast period in the domain's full timeline.
* ``timeline(domain)`` — every period of a domain (train + val + test, in time order) with
  its aligned text. Row r's text is period r−1's report (lag 1, end-anchored — P1), so any
  text at a row ≤ r was published before period r starts. Shuffled / time-shifted /
  cross-domain texts are drawn from here.

Saved files: ``texts/<condition>.csv`` (gitignored), one per condition, columns
``domain, split, sample_id, condition, text, has_text, attacked, reason, real_direction,
claimed_direction, variant, source_domain, source_sample_id, synthetic``. ``text`` is empty
when the condition has no text. Fake rows carry ``synthetic=True`` (ethics, CLAUDE.md §14).
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.splits import load_splits
from src.data.windows import lookback_for, split_windows

DOMAINS = ["Economy", "SocialGood", "Traffic", "Health_US"]
ROOT = Path(__file__).resolve().parents[2]
TEXTS_DIR = ROOT / "texts"
CONDITIONS = ["real", "empty", "shuffled", "cross_domain", "time_shifted", "flipped",
              "exaggerated"]
COLUMNS = ["domain", "split", "sample_id", "condition", "text", "has_text", "attacked", "reason",
           "real_direction", "claimed_direction", "variant", "source_domain",
           "source_sample_id", "synthetic"]
SYNTHETIC_NOTE = ("SYNTHETIC TEXT generated for defence research (TextGuard). "
                  "Not a real report. Do not redistribute as genuine.")


@lru_cache(maxsize=None)
def timeline(domain: str) -> pd.DataFrame:
    """All periods of ``domain`` in time order: ``row, start_date, sample_id, split, text``."""
    s = load_splits(domain)
    parts = [df.assign(split=name) for name, df in (("train", s.train), ("val", s.val),
                                                    ("test", s.test))]
    full = pd.concat(parts, ignore_index=True)
    full["text"] = [t if isinstance(t, str) and t.strip() else None for t in full["text"]]
    full["sample_id"] = full["start_date"].dt.strftime("%Y-%m-%d")
    full["row"] = np.arange(len(full))
    return full[["row", "start_date", "sample_id", "split", "text"]]


@lru_cache(maxsize=None)
def _windows(domains: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for d in domains:
        tl = timeline(d)
        row_of = dict(zip(tl.sample_id, tl.row))
        w = split_windows(load_splits(d), lookback_for(d), horizon=1)
        for split in ("val", "test"):
            win = w[split]
            for i in range(len(win)):
                sid = win.target_start[i].strftime("%Y-%m-%d")
                rows.append(dict(domain=d, split=split, sample_id=sid, row=row_of[sid],
                                 target_start=win.target_start[i], text=win.text[i]))
    df = pd.DataFrame(rows)
    df["text"] = [t if isinstance(t, str) and t.strip() else None for t in df["text"]]
    return df


def windows(domains=DOMAINS) -> pd.DataFrame:
    """Val + test windows of ``domains`` with their real (aligned) text."""
    return _windows(tuple(domains)).copy()


def rng_for(*key) -> np.random.Generator:
    """A seeded generator per (condition, domain, sample): reproducible, order-independent."""
    seed = int(hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def frame(w: pd.DataFrame, condition: str, texts, **cols) -> pd.DataFrame:
    """Assemble a condition table with the standard columns (missing ones filled)."""
    out = w[["domain", "split", "sample_id"]].copy()
    out["condition"] = condition
    out["text"] = [t if isinstance(t, str) and t.strip() else "" for t in texts]
    out["has_text"] = out["text"].str.len() > 0
    defaults = dict(attacked=False, reason="", real_direction=0, claimed_direction=0,
                    variant="", source_domain="", source_sample_id="", synthetic=False)
    for k, v in {**defaults, **cols}.items():
        out[k] = v
    return out[COLUMNS]


def save(df: pd.DataFrame, condition: str, out_dir: Path = TEXTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{condition}.csv"
    df.to_csv(path, index=False)
    (out_dir / "README.txt").write_text(
        "TextGuard text conditions (one CSV per condition).\n\n"
        f"Rows with synthetic=True are {SYNTHETIC_NOTE}\n"
        "Generated by: python -m src.attacks.generate\n", encoding="utf-8")
    return path


def load(condition: str, out_dir: Path = TEXTS_DIR) -> pd.DataFrame:
    df = pd.read_csv(out_dir / f"{condition}.csv", keep_default_na=False,
                     dtype={"sample_id": str, "source_sample_id": str})
    df["text"] = df["text"].astype(str)
    return df
