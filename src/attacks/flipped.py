"""``flipped`` — the core attack: the report's direction is reversed, nothing else.

Deterministic rules (``rules.py``), not an LLM: Model 5 and P6's direction judge are Qwen, so
a Qwen-written fake would bias both (issue #25); rules give a minimal, exact, auditable edit.

Two variants, chosen per text by what its headline asserts (``variant`` column):

* **change-flip** — the headline states a change: increased↔decreased, rose↔fell, up↔down,
  more↔less, signed changes "-1.6%" → "+1.6%"; dependent numbers recomputed so the text
  stays arithmetically consistent ("X, up Y from Z" → "X, down Y from X+Y").
  In practice: **Economy, Traffic and SocialGood** (SocialGood's reports state changes —
  "declined by 0.2 percentage point" — not levels; PROJECT_STATUS §8).
* **level-flip** — the headline states a level vs a reference: above↔below the national
  baseline, with the stated value mirrored to the same distance on the other side
  (2B − V; ratio B²/V only if that would be ≤ 0) so the claim is numerically true and
  plausible. In practice **Health_US only**.

Side sentences that carry no tracked direction word (e.g. "10 states experienced high ILI
activity") are left unchanged: rewriting them risks new contradictions, and leaving them
makes the attack slightly weaker — conservative. A stated limitation.

Every row records ``real_direction`` and ``claimed_direction`` (+1/−1, about the forecast
target) so DSS uses the attacker's known direction, never a classifier. A text that cannot
be flipped cleanly is kept **unchanged** with ``attacked=False`` and the reason; P4/P5 count
only ``attacked`` rows as attacks.
"""

from __future__ import annotations

import pandas as pd

from src.attacks.common import frame, windows
from src.attacks.rules import edit_text

MODE = "flip"
CONDITION = "flipped"


def build(w: pd.DataFrame | None = None, mode: str = MODE, condition: str = CONDITION) -> pd.DataFrame:
    w = windows() if w is None else w
    res = [edit_text(t, d, mode) for t, d in zip(w["text"], w["domain"])]
    return frame(w, condition, [r.text for r in res],
                 attacked=[r.attacked for r in res], reason=[r.reason for r in res],
                 real_direction=[r.real_direction for r in res],
                 claimed_direction=[r.claimed_direction for r in res],
                 variant=[r.variant for r in res], source_domain=w["domain"].values,
                 source_sample_id=w["sample_id"].values,
                 synthetic=[r.attacked for r in res])
