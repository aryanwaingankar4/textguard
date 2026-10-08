"""``time_shifted`` — the report aligned to the same period one year earlier.

Tests whether a model checks *when* a report is from. The shifted report is genuine, in the
right domain and the same season (12 months / 52 weeks back), so it reads exactly like a
plausible report — only its period is wrong. A model whose error does not change is not
using the timing of the text; one that gets much worse depends on it.

Backward only, never forward: a forward shift would feed reports from the forecast period
or later, i.e. a leakage probe, not a timing control. If the period a year back has no
report, the window gets no text (``has_text=False``) rather than a different offset.
"""

from __future__ import annotations

import pandas as pd

from src.attacks.common import frame, timeline, windows
from src.data.windows import DOMAIN_FREQ

SHIFT = {"monthly": 12, "weekly": 52}  # periods back = one year


def build(w: pd.DataFrame | None = None) -> pd.DataFrame:
    w = windows() if w is None else w
    texts, sources = [], []
    for r in w.itertuples():
        tl = timeline(r.domain)
        src = r.row - SHIFT[DOMAIN_FREQ[r.domain]]
        hit = tl[tl.row == src]
        if src < 0 or hit.empty:
            texts.append(None); sources.append("")
            continue
        texts.append(hit.text.iloc[0]); sources.append(hit.sample_id.iloc[0])
    return frame(w, "time_shifted", texts, source_domain=w["domain"].values,
                 source_sample_id=sources)
