"""A trivial forecaster with hand-checkable behaviour (P2.5).

Why it exists: to prove TextGuard (P6) is genuinely model-agnostic, it must work on a
model it was never designed around — and to unit-test P4's metrics and P6's gate, we need
predictions we can compute on paper. Real models give neither.

Behaviour (everything can be checked with a pencil):

* **Numbers path (A, ``text=None``)** — linear extrapolation: last value + the average
  step of the window, ``(last - first) / (n - 1)``, repeated for each horizon step.
  ``100, 104, 108, 112`` -> ``116``.
* **Text path (B)** — ``A + sensitivity * direction * step * intensity``.
  ``direction`` comes from a tiny keyword lexicon:

  - change-language (Economy, Traffic style): rise/increase/... = +1, fall/decrease/... = -1
  - level-language (Health_US, SocialGood style): "above (the) [word] baseline" = +1,
    "below (the) [word] baseline" = -1 — the fixture for TextGuard's level branch

  Matched cues are summed and the sign taken (a text with one up and one down cue is
  neutral). ``intensity`` is ``intensifier`` if an adverb like "sharply" appears, else 1.
* ``sensitivity`` simulates the two cases the paper cares about:
  ``0`` -> ignores text (Text Usage Gap = 0, "deaf"); ``1`` -> follows text and is fooled
  by flipped text. Values in between are allowed.

Reproduces the PHASES.md worked example with the defaults (step=4, intensifier=2.5)::

    numbers = [100, 104, 108, 112]
    predict(numbers, None)                        -> 116   (A)
    predict(numbers, "Demand will fall sharply.") -> 106   (B, fooled)
    predict(numbers, "demand rising")             -> 120   (y_real used by DSS)

The keyword lexicon is fine *here* because the dummy only needs predictable behaviour.
It is not a text classifier: it ignores negation ("did not rise" counts as up). Real
direction classification in P3/P6 is Qwen-based (the keyword heuristic agreed with
human labels only 72% of the time).
"""

from __future__ import annotations

import re

import numpy as np

from src.models.base import BaseForecaster

_UP = re.compile(
    r"\b(rise|rises|rose|rising|risen|increase|increases|increased|increasing|"
    r"grow|grows|grew|growing|climb|climbs|climbed|climbing|higher)\b", re.I)
_DOWN = re.compile(
    r"\b(fall|falls|fell|falling|fallen|decrease|decreases|decreased|decreasing|"
    r"decline|declines|declined|declining|drop|drops|dropped|dropping|lower)\b", re.I)
# "above baseline", "above the baseline", "below the national baseline"
_ABOVE = re.compile(r"\babove\s+(?:the\s+)?(?:\w+\s+)?baseline\b", re.I)
_BELOW = re.compile(r"\bbelow\s+(?:the\s+)?(?:\w+\s+)?baseline\b", re.I)
_INTENSIFIER = re.compile(r"\b(sharply|steeply|dramatically|significantly)\b", re.I)


def text_direction(text: str) -> int:
    """Net direction claimed by ``text``: +1, -1 or 0. Change and level cues both count."""
    score = (len(_UP.findall(text)) + len(_ABOVE.findall(text))
             - len(_DOWN.findall(text)) - len(_BELOW.findall(text)))
    return int(np.sign(score))


def text_intensity(text: str, intensifier: float) -> float:
    return intensifier if _INTENSIFIER.search(text) else 1.0


class DummyForecaster(BaseForecaster):
    """Linear extrapolation plus a configurable, keyword-driven text shift."""

    def __init__(self, sensitivity: float = 1.0, step: float = 4.0, intensifier: float = 2.5,
                 horizon: int = 1, lookback: int | None = None):
        super().__init__(horizon=horizon, lookback=lookback)
        self.sensitivity = sensitivity
        self.step = step
        self.intensifier = intensifier

    def _predict(self, numbers: np.ndarray, text: str | None) -> np.ndarray:
        slope = (numbers[-1] - numbers[0]) / (len(numbers) - 1) if len(numbers) > 1 else 0.0
        a = numbers[-1] + slope * np.arange(1, self.horizon + 1)
        if text is None:
            return a
        shift = (self.sensitivity * text_direction(text) * self.step
                 * text_intensity(text, self.intensifier))
        return a + shift
