"""TextGuard v2 (R4): every claim checked against evidence in its own frame, plus a magnitude cap.

Still training-free, inference-time and black-box — it needs ``predict(numbers, text)`` and the
series' own past, nothing else. v1 (``textguard.py``, ``gate.py``, ``trend_detector.py``) stays
exactly as frozen in P6; v2 imports v1's building blocks and never changes them.

    A = model.predict(numbers, None);  B = model.predict(numbers, text)
    claims = judge(every unit of the text)               # change / outlook / level, per unit
    g      = min over claims of 1 − c·s on disagreement  # the most suspicious claim decides
    g′     = 0 if g < τ else g                           # block mode (τ = 0: soft gate)
    final  = A + clamp(g′·(B − A), ±k·σ)                 # magnitude cap (from clip)

What each piece fixes (PROJECT_STATUS §1a "What v2 must address"):

* **Every claim** — the judge reads each unit (``units.py``), so an appended sentence is a claim
  of its own; g is the minimum, so one suspicious claim is enough.
* **Forward-looking claims** (``outlook``) cannot be verified — the future has not happened — so
  they are checked against the only evidence there is: the recent trend (momentum), with the
  same formula as a past change claim. A prediction that the trend will reverse, made with
  confidence against a strong trend, is discounted; one that goes with the trend passes.
  (Not used: "contradicts the report's own past claim" — genuine reports and headline bundles
  mix directions, so it would gate genuine text; and a forward claim against a weak trend is
  left alone on purpose — that is where a genuine "turning point" report lives.)
* **Level vs change** — separate questions (``judge_prompt_v2``); a level claim is checked
  against y_last vs the baseline the unit states (v1's level evidence), a change claim against
  change evidence. A unit can carry both.
* **Year-over-year frame** — a change claim whose unit is year-over-year (``units.frame``) is
  checked against y_last − y_{last−P} (P = 12 monthly, 52 weekly) from the **history**, which
  the deployer passes explicitly (``TextGuardV2.predict(numbers, text, history=…)``). Strength =
  |Δ_P| / std of all past P-differences in that history (a seasonal series' σ is mostly season,
  so it is the wrong scale for a year-over-year change). Too short a history → no evidence.
  ``unchecked`` frames (cumulative, multi-month average) get no evidence → not gated.
* **Magnitude** — the cap borrows clip's rule (clip beat v1 on exaggeration in P6 and was the
  only defence to dent A2 in R3): whatever the gate lets through is limited to k·σ.

Confidence floor c₀: a claim the judge is less sure of than c₀ is ignored. Splitting a report into
units multiplies the claims, and with them the judge's mistakes on genuine text (False Block
Rate); c₀ is tuned on val with the other knobs.

Applied weight (for the False Block Rate, one definition for every defence, §10b): (final − A) /
(B − A), 1 when B = A — so a capped genuine text counts as blocked exactly as under clip.
NaN rules as v1: B NaN → A; A NaN → NaN.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import check_numbers, normalise_text
from src.textguard.baselines import Defended
from src.textguard.gate import branch_gate
from src.textguard.judge_prompt_v2 import ANSWERS, KINDS, build_judge_prompt, judge_sha
from src.textguard.text_direction import POLARITY, stated_baseline
from src.textguard.trend_detector import NO_EVIDENCE, Evidence, change_evidence, level_evidence
from src.textguard.units import frame, units

ROOT = Path(__file__).resolve().parents[2]
QWEN_DIR = ROOT / "results" / "qwen"

PERIOD = {"monthly": 12, "weekly": 52}
MIN_PERIOD_DIFFS = 3     # fewer past P-differences than this → no year-over-year scale → no evidence


@dataclass(frozen=True)
class ClaimV2:
    """One claim of one unit, already in series terms (Economy's deficit mapped to the balance)."""
    kind: str                       # "change" | "outlook" | "level"
    direction: int                  # +1 / −1 / 0
    confidence: float               # in [⅓, 1]
    frame: str = "trend"            # change claims: "trend" | "yoy" | "unchecked"
    reference: float | None = None  # level claims: the baseline the unit states
    unit: str = ""


@dataclass(frozen=True)
class Settings:
    """v2's knobs — tuned on the ORIGINAL domains' val split only (R4, §10g)."""
    window: int                     # trend window (values) for change / outlook claims
    conf_floor: float = 0.0         # ignore claims with confidence < c₀
    tau: float = 0.0                # block threshold (0 = soft gate)
    cap_k: float = np.inf           # magnitude cap in σ (inf = no cap)


def level_reference(unit: str) -> float | None:
    """The national baseline a unit states (v1's regex) — level is asked only where one is stated."""
    return stated_baseline(unit)


def questions_for(unit: str) -> tuple[str, ...]:
    """Which questions a unit gets: change and outlook always; level if it states a baseline."""
    return KINDS if level_reference(unit) is not None else ("change", "outlook")


# --- evidence ---------------------------------------------------------------------------------

def yoy_evidence(history, period: int) -> Evidence:
    """y_last − y_{last−P}, in units of the std of every past P-difference in ``history``."""
    h = np.asarray(history, dtype=float)
    if h.ndim != 1 or h.size <= period:
        return NO_EVIDENCE
    diffs = h[period:] - h[:-period]              # positions kept: a missing value (NaN) stays
    past = diffs[np.isfinite(diffs)]              # one, never shifts the lag
    d = float(diffs[-1])
    if not np.isfinite(d) or past.size < MIN_PERIOD_DIFFS:
        return NO_EVIDENCE
    scale = float(np.std(past))
    if not scale > 0:
        return NO_EVIDENCE
    return Evidence(int(np.sign(d)), float(min(abs(d) / scale, 1.0)))


def evidence_for(claim: ClaimV2, numbers, sigma: float, window: int, history=None,
                 period: int | None = None) -> Evidence:
    if claim.kind == "level":
        return level_evidence(numbers, sigma, claim.reference)
    if claim.kind == "change" and claim.frame == "unchecked":
        return NO_EVIDENCE
    if claim.kind == "change" and claim.frame == "yoy":
        if history is None or period is None:
            return NO_EVIDENCE
        return yoy_evidence(history, period)
    return change_evidence(numbers, sigma, window)          # trend: past change and outlook


@dataclass
class GateV2:
    g: float
    checks: list = field(default_factory=list)   # (ClaimV2, Evidence, per-claim gate)


def gate(claims: list[ClaimV2], numbers, sigma: float, settings: Settings, history=None,
         period: int | None = None) -> GateV2:
    """g = min over the claims at or above the confidence floor (1 if none)."""
    checks = []
    for c in claims:
        if c.direction == 0 or c.confidence < settings.conf_floor:
            continue
        ev = evidence_for(c, numbers, sigma, settings.window, history, period)
        checks.append((c, ev, branch_gate(c.direction, c.confidence, ev)))
    return GateV2(min((g for _, _, g in checks), default=1.0), checks)


def defend(A, B, g, sigma, tau: float = 0.0, cap_k: float = np.inf) -> Defended:
    """Block at τ, then cap the text's effect at k·σ. Vectorised over windows (horizon 1 or more)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    if A.shape != B.shape or A.ndim == 0:
        raise ValueError("A and B must be arrays of the same shape (n,) or (n, horizon)")
    g = np.broadcast_to(np.asarray(g, float), (A.shape[0],))
    if np.any((g < 0) | (g > 1)):
        raise ValueError("gate values must lie in [0, 1]")
    if cap_k < 0:
        raise ValueError("cap_k must be >= 0")
    warn = g < tau
    w = np.where(warn, 0.0, g)
    shape = (-1,) + (1,) * (A.ndim - 1)
    diff = np.where(np.isnan(B), A, B) - A                  # B NaN → no text effect (final = A)
    delta = w.reshape(shape) * diff
    factor = np.ones(A.shape[0])                            # how much the cap leaves of g′(B − A)
    if np.isfinite(cap_k):
        lim = cap_k * np.broadcast_to(np.asarray(sigma, float), (A.shape[0],)).reshape(shape)
        big = np.abs(delta) > lim
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(big, lim / np.abs(delta), 1.0)
        factor = np.nanmin(f.reshape(A.shape[0], -1), axis=1)
        delta = np.clip(delta, -lim, lim)
    return Defended(A + delta, w * factor, warn)


# --- the judge's answers ----------------------------------------------------------------------

def answer_direction(kind: str, answer: str) -> int:
    pos, neg, none = ANSWERS[kind]
    return {pos: 1, neg: -1, none: 0}[answer]


def load_answers(predictions: Path | str, inputs: Path | str, meta: Path | str) -> pd.DataFrame:
    """Every v2 judge answer joined with its unit; fingerprints re-derived; one setup; complete."""
    inp = pd.read_csv(inputs, keep_default_na=False, dtype=str)
    inp = inp[inp.row_type == "judge"]
    out = pd.read_csv(predictions, keep_default_na=False, dtype={"prompt_sha": str})
    run = json.loads(Path(meta).read_text(encoding="utf-8"))
    if out.prompt_sha.duplicated().any():
        raise ValueError("duplicate judge answers for one prompt")
    df = inp.merge(out.drop(columns=["domain", "question"], errors="ignore"), on="prompt_sha",
                   how="left", validate="one_to_one", indicator=True)
    if (df["_merge"] != "both").any():
        raise ValueError(f"{int((df['_merge'] != 'both').sum())} exported questions have no answer")
    if len(out) != len(inp):
        raise ValueError("answers for prompts that were not exported")
    if [judge_sha(build_judge_prompt(d, q, t)) for d, q, t in zip(df.domain, df.question, df.text)] \
            != list(df.prompt_sha):
        raise ValueError("v2 judge prompt changed since the run (fingerprint mismatch)")
    if df.setup.nunique() != 1 or df.setup.iloc[0] != run.get("setup"):
        raise ValueError("answers come from more than one software setup")
    df["confidence"] = df.confidence.astype(float)
    df["literal"] = [answer_direction(q, a) for q, a in zip(df.question, df.answer)]
    return df.drop(columns=["_merge", "row_type"])


class JudgeV2:
    """Claims of every unit of a text, served from a table of answers (one domain)."""

    def __init__(self, domain: str, answers: pd.DataFrame):
        self.domain = domain
        a = answers[answers.domain == domain]
        self._a = {(q, t): (int(d), float(c)) for q, t, d, c in
                   zip(a.question, a.text, a.literal, a.confidence)}

    def __call__(self, text: str | None) -> list[ClaimV2]:
        out = []
        for u in units(text):
            for kind in questions_for(u):
                if (kind, u) not in self._a:
                    raise KeyError(f"{self.domain}/{kind}: unit was not judged: {u[:80]!r}")
                lit, conf = self._a[(kind, u)]
                d = lit if kind == "level" else lit * POLARITY.get(self.domain, 1)
                out.append(ClaimV2(kind, d, conf, frame(u) if kind == "change" else "trend",
                                   level_reference(u) if kind == "level" else None, u))
        return out


# --- the wrapper ------------------------------------------------------------------------------

@dataclass
class DecisionV2:
    A: np.ndarray
    B: np.ndarray
    g: float
    weight: float
    final: np.ndarray
    warn: bool
    checks: list


class TextGuardV2:
    """TextGuard v2 around any ``predict(numbers, text)`` forecaster.

    ``period`` is the seasonal lag for year-over-year claims (12 monthly, 52 weekly). ``history``
    is passed per call: the series up to and including the forecaster's last lookback value
    (it must end with ``numbers``) — the longer past that year-over-year claims need.
    """

    def __init__(self, model, judge, sigma: float, settings: Settings, period: int | None = None):
        if not hasattr(model, "predict") or not hasattr(model, "horizon"):
            raise TypeError("model must have `horizon` and `predict(numbers, text)`")
        if not (np.isfinite(sigma) and sigma > 0):
            raise ValueError("sigma must be a positive number (the train-split std)")
        self.model, self.judge, self.sigma = model, judge, float(sigma)
        self.settings, self.period = settings, period
        self.horizon = model.horizon

    def decide(self, numbers, text: str | None, history=None) -> DecisionV2:
        numbers = check_numbers(numbers)
        if history is not None:
            history = np.asarray(history, dtype=float)
            if history.size < numbers.size or not np.allclose(history[-numbers.size:], numbers):
                raise ValueError("history must end with the forecaster's lookback (numbers)")
        text = normalise_text(text)
        A = np.asarray(self.model.predict(numbers, None), dtype=float)
        B = A.copy() if text is None else np.asarray(self.model.predict(numbers, text), float)
        res = gate(self.judge(text) if text else [], numbers, self.sigma, self.settings,
                   history, self.period)
        d = defend(A[None], B[None], [res.g], self.sigma, self.settings.tau, self.settings.cap_k)
        return DecisionV2(A, B, res.g, float(d.weight[0]), d.final[0], bool(d.warn[0]), res.checks)

    def predict(self, numbers, text: str | None = None, history=None) -> np.ndarray:
        return self.decide(numbers, text, history).final
