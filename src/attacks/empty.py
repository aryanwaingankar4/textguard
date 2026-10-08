"""``empty`` — the report is removed.

Same input as ``text=None`` / ``has_text=False`` / TextGuard's A (PROJECT_STATUS §6a): one
definition of "no text" everywhere. For Models 3/4 that is the frozen numbers-only backbone,
for Model 5 the prompt without the report block. Empty-vs-real is the Text Usage Gap.
"""

from __future__ import annotations

import pandas as pd

from src.attacks.common import frame, windows


def build(w: pd.DataFrame | None = None) -> pd.DataFrame:
    w = windows() if w is None else w
    return frame(w, "empty", [None] * len(w), reason="empty")
