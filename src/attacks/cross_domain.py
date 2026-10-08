"""``cross_domain`` — a genuine report from a different domain, available at the same time.

Tests whether a model can tell relevant from irrelevant text: an unemployment report given
to the traffic forecaster. Timing is held fixed — the source report is the latest one the
other domain had published by the forecast period (its row with start_date ≤ the window's
target start) — so only the topic differs.

Fixed rotation, so each domain receives one other domain:
Economy ← SocialGood ← Traffic ← Health_US ← Economy. When the source domain's reports stopped
earlier (Health_US reports end in 2020), the latest available one is used and its date is
recorded in ``source_sample_id`` so staleness is visible.
"""

from __future__ import annotations

import pandas as pd

from src.attacks.common import DOMAINS, frame, timeline, windows

SOURCE = {d: DOMAINS[(i + 1) % len(DOMAINS)] for i, d in enumerate(DOMAINS)}


def build(w: pd.DataFrame | None = None) -> pd.DataFrame:
    w = windows() if w is None else w
    texts, src_dom, src_id = [], [], []
    for r in w.itertuples():
        sd = SOURCE[r.domain]
        tl = timeline(sd)
        pool = tl[(tl.start_date <= r.target_start) & tl.text.notna()]
        if pool.empty:
            texts.append(None); src_dom.append(sd); src_id.append("")
            continue
        pick = pool.iloc[-1]
        texts.append(pick.text); src_dom.append(sd); src_id.append(pick.sample_id)
    return frame(w, "cross_domain", texts, source_domain=src_dom, source_sample_id=src_id)
