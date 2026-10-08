"""``exaggerated`` — the report keeps its direction but overstates the magnitude.

Same rule engine as ``flipped`` (``rules.edit_text(..., mode="exaggerate")``): stated changes
×3, dependent numbers recomputed with the same consistency rules, "sharply" after the headline
verb ("slightly" → "sharply"), "well" before above/below a baseline; a level claim's distance
to its baseline is cubed on a ratio scale (V' = B·(V/B)³). A decrease that would reach
≥ 100% is impossible, so that text is not attacked.

This is TextGuard's known weak spot: the text *agrees* with the numerical trend, so a
consistency gate has nothing to object to. We test it honestly (CLAUDE.md §7).
``claimed_direction`` equals ``real_direction``.
"""

from __future__ import annotations

import pandas as pd

from src.attacks import flipped


def build(w: pd.DataFrame | None = None) -> pd.DataFrame:
    return flipped.build(w, mode="exaggerate", condition="exaggerated")
