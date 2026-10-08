"""``shuffled`` — a real report from another date of the same domain.

Tests whether a model uses the text's *content* for that date: a model that ignores content
scores the same as with ``real``; a model that reads it gets worse. Same domain, so the
style and vocabulary match — only the date is wrong.

Drawn **only from earlier periods** (rows before the window's own row): every candidate was
published before the forecast period, so shuffling can never leak future information (a
random future report could, e.g. a 2021 report given to a 2019 window). Excluded: the
window's own aligned text, and any text identical to it (Health_US repeats some reports).
Seeded per (domain, sample_id), so the draw is reproducible and order-independent.
"""

from __future__ import annotations

import pandas as pd

from src.attacks.common import frame, rng_for, timeline, windows


def build(w: pd.DataFrame | None = None) -> pd.DataFrame:
    w = windows() if w is None else w
    texts, sources = [], []
    for r in w.itertuples():
        tl = timeline(r.domain)
        own = r.text
        pool = tl[(tl.row < r.row) & tl.text.notna()]
        if own is not None:
            pool = pool[pool.text != own]
        if pool.empty:
            texts.append(None); sources.append("")
            continue
        pick = pool.iloc[int(rng_for("shuffled", r.domain, r.sample_id).integers(len(pool)))]
        texts.append(pick.text); sources.append(pick.sample_id)
    return frame(w, "shuffled", texts, source_domain=w["domain"].values,
                 source_sample_id=sources)
