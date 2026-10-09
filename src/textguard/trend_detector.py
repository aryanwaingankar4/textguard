"""Numerical evidence for TextGuard (P6): what do the numbers themselves say?

TextGuard never trusts a text's claim on its own: it checks the claim against evidence read
from the same lookback window the forecaster sees. Two branches, one per kind of claim
(PROJECT_STATUS §6 "Attack scope", §10b):

* **Change branch** — the text says the series went up / down. Evidence: the OLS slope β over
  the last ``window`` values. direction = sign(β); strength = min(|β|·(w − 1)/σ, 1), i.e. the
  fitted movement *across the window* in σ units.
  Why "across the window" and not the per-step slope: CLAUDE.md §8 writes ``|slope|/σ``, and
  the PHASES worked example (100, 104, 108, 112; σ = 10) needs strength 1.0 — the fitted rise
  over the window is 12 = 1.2σ → 1.0, while a per-step slope (4) would give 0.4. A per-step
  slope of a monthly series is also almost always ≪ σ, so the gate would barely ever act.
  No dead zone: a near-flat window gives a near-zero strength, so the gate stays ≈ 1 anyway.
* **Level branch** — the text says the series is above / below a reference level it states
  (Health_US: "ILI was 3.2%, above the national baseline of 2.1%"). Evidence: where the latest
  value y_last sits relative to that reference: direction = sign(y_last − B), strength =
  min(|y_last − B|/σ, 1). Valid only when the series is the quantity the text states — true for
  Health_US (stated ILI vs t−1: corr 0.98, checked 2026-10-09); a domain-config assumption.

σ = the domain's train-split std (the same σ as DSS), passed in by the caller, so the
detector never sees val/test statistics. Only ``window`` is tuned — on val (§10b).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Evidence:
    """What the numbers say: direction ∈ {−1, 0, +1}, strength ∈ [0, 1]."""
    direction: int
    strength: float


NO_EVIDENCE = Evidence(0, 0.0)


def _check(numbers, sigma: float) -> np.ndarray:
    x = np.asarray(numbers, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("numbers must be a non-empty, finite 1-D window")
    if not (np.isfinite(sigma) and sigma > 0):
        raise ValueError("sigma must be a positive number (the train-split std)")
    return x


def window_slope(numbers, window: int) -> float:
    """OLS slope per step over the last ``window`` values (all of them if fewer)."""
    if window < 2:
        raise ValueError("window must be >= 2 to fit a slope")
    y = np.asarray(numbers, dtype=float)[-window:]
    if y.size < 2:
        return 0.0
    t = np.arange(y.size, dtype=float)
    t -= t.mean()
    return float(np.dot(t, y - y.mean()) / np.dot(t, t))


def change_evidence(numbers, sigma: float, window: int) -> Evidence:
    """Trend over the recent window: sign of the slope, fitted movement / σ capped at 1."""
    x = _check(numbers, sigma)
    w = min(window, x.size)
    beta = window_slope(x, w)
    return Evidence(int(np.sign(beta)), float(min(abs(beta) * (w - 1) / sigma, 1.0)))


def level_evidence(numbers, sigma: float, reference: float) -> Evidence:
    """Latest value vs the reference level the text states (e.g. Health_US's baseline)."""
    x = _check(numbers, sigma)
    if reference is None or not np.isfinite(reference):
        return NO_EVIDENCE
    diff = float(x[-1] - reference)
    return Evidence(int(np.sign(diff)), float(min(abs(diff) / sigma, 1.0)))


def window_for(lookback: int, fraction: float) -> int:
    """The pre-registered window grid (§10b): w = max(3, round(fraction · L)), at most L."""
    return int(min(lookback, max(3, round(fraction * lookback))))
