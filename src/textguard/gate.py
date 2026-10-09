"""TextGuard's gate (P6): how much of the text's effect to let through, g ∈ [0, 1].

The rule (CLAUDE.md §8, generalised to "claim vs evidence", PROJECT_STATUS §10b):

* per claim: if the text claims nothing (0), the numbers show nothing (0), or the two
  **agree** → 1 (trust the text fully);
  if they **disagree** → 1 − confidence × strength.
  A confident claim against a strong numerical signal is cut hard; a hesitant claim, or one
  against a weak signal, is barely touched.
* a text that makes several claims (Health_US: change and level) is trusted only as much as
  its least credible claim: **g = min over its claims**. A text with no claims → g = 1.

Why "agree → 1" and not "agree → boost": TextGuard is a defence, not a re-weighting scheme;
genuine text that matches the numbers must pass untouched (the False Block Rate checks this).
Why the gate cannot catch ``exaggerated``: an exaggerated claim has the *same* direction as
the real one, so the gate sees exactly what it sees for genuine text — a known weak spot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.textguard.text_direction import Claim
from src.textguard.trend_detector import Evidence


def branch_gate(text_direction: int, confidence: float, evidence: Evidence) -> float:
    """Gate for one claim: 1 unless claim and evidence point opposite ways."""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must lie in [0, 1]")
    if not 0.0 <= evidence.strength <= 1.0:
        raise ValueError("strength must lie in [0, 1]")
    if text_direction == 0 or evidence.direction == 0 or text_direction == evidence.direction:
        return 1.0
    return 1.0 - confidence * evidence.strength


@dataclass
class GateResult:
    """The gate value plus every claim and the evidence it was checked against."""
    g: float
    checks: list[tuple[Claim, Evidence, float]] = field(default_factory=list)


def combine(checks: list[tuple[Claim, Evidence, float]]) -> GateResult:
    """g = min over the per-claim gates (1 if there are none)."""
    return GateResult(min((g for _, _, g in checks), default=1.0), checks)
