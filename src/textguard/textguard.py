"""TextGuard (P6): a training-free, inference-time, black-box defence for text-fed forecasters.

Wraps **any** object with ``predict(numbers, text)`` (the ``Forecaster`` protocol, §6a) — no
inheritance, no gradients, no internals, no retraining::

    A = model.predict(numbers, None)          # numbers only
    B = model.predict(numbers, text)          # numbers + text
    g = gate(text's claims vs the numbers)    # trend / level evidence, judge confidence
    final = A + g · (B − A)                   # soft;  block mode: g < τ → final = A
    warn  = g < τ

A ``TextGuard`` is itself a forecaster (``horizon`` + ``predict``), so a guarded model can be
dropped in wherever the bare one was — including into the Auditor (P7).

What it needs from the deployer, and why:

* ``judge`` — reads the text's claims about this series (``text_direction.DirectionJudge``).
  The gate never reads text itself; the judge is swappable.
* ``sigma`` — the series' train-split std: the scale for "strong" evidence (same σ as DSS).
* ``window`` — how many recent values the trend is fitted on (tuned on val only, §10b).
* ``threshold`` τ and ``mode`` — when to warn, and whether a warning blocks the text.

What it does **not** need: the model's training data, its architecture, or the ability to
retrain it. That is the axis on which it differs from CFA (trained adapters) and PA-RNet
(an architecture), not raw accuracy.

Failures (§6a): B NaN → final = A (numbers-only fallback); A NaN → final NaN, counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.models.base import check_numbers, normalise_text
from src.textguard.baselines import textguard as apply_gate
from src.textguard.gate import GateResult, branch_gate, combine
from src.textguard.text_direction import Claim, DirectionJudge
from src.textguard.trend_detector import Evidence, change_evidence, level_evidence


@dataclass
class Decision:
    """Everything TextGuard did for one sample — the audit trail behind a forecast."""
    A: np.ndarray
    B: np.ndarray
    g: float
    weight: float            # applied weight g′ (block mode: 0 when warned)
    final: np.ndarray
    warn: bool
    checks: list = field(default_factory=list)   # (Claim, Evidence, per-claim gate)


def evidence_for(claim: Claim, numbers, sigma: float, window: int) -> Evidence:
    """The numerical evidence a claim is checked against: trend for change, level for level."""
    if claim.kind == "change":
        return change_evidence(numbers, sigma, window)
    if claim.kind == "level":
        return level_evidence(numbers, sigma, claim.reference)
    raise ValueError(f"unknown claim kind {claim.kind!r}")


def gate(claims: list[Claim], numbers, sigma: float, window: int) -> GateResult:
    """Check every claim against its evidence; g = the least credible claim's gate."""
    checks = []
    for c in claims:
        ev = evidence_for(c, numbers, sigma, window)
        checks.append((c, ev, branch_gate(c.direction, c.confidence, ev)))
    return combine(checks)


class TextGuard:
    """Model-agnostic consistency gate around any ``predict(numbers, text)`` forecaster."""

    def __init__(self, model, judge: DirectionJudge, sigma: float, window: int = 4,
                 threshold: float = 0.5, mode: str = "soft"):
        if not hasattr(model, "predict") or not hasattr(model, "horizon"):
            raise TypeError("model must have `horizon` and `predict(numbers, text)`")
        if mode not in ("soft", "block"):
            raise ValueError("mode must be 'soft' or 'block'")
        if not (np.isfinite(sigma) and sigma > 0):
            raise ValueError("sigma must be a positive number (the train-split std)")
        self.model, self.judge, self.sigma = model, judge, float(sigma)
        self.window, self.threshold, self.mode = int(window), float(threshold), mode
        self.horizon = model.horizon

    def gate(self, numbers, text: str | None) -> GateResult:
        """Gate value for (numbers, text) — independent of the model (only judge + numbers)."""
        numbers = check_numbers(numbers)
        text = normalise_text(text)
        claims = self.judge(text) if text is not None else []
        return gate(claims, numbers, self.sigma, self.window)

    def decide(self, numbers, text: str | None) -> Decision:
        numbers = check_numbers(numbers)
        text = normalise_text(text)
        A = np.asarray(self.model.predict(numbers, None), dtype=float)
        B = A.copy() if text is None else np.asarray(self.model.predict(numbers, text), float)
        res = self.gate(numbers, text)
        d = apply_gate(A[None], B[None], [res.g], self.threshold, self.mode)
        return Decision(A, B, res.g, float(d.weight[0]), d.final[0], bool(d.warn[0]),
                        res.checks)

    def predict(self, numbers, text: str | None = None) -> np.ndarray:
        """The defended forecast, shape (horizon,) — TextGuard is itself a Forecaster."""
        return self.decide(numbers, text).final
