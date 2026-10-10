"""R3 — adaptive attacks on TextGuard v1: attackers who know the defence (PROJECT_STATUS §10f).

P3's flip is defence-unaware. These three attackers know TextGuard v1's code and its frozen
settings (``results/summary_p6_frozen.json``) and control only the report text. Like P3 they are
**deterministic rules applied to the real reports** — no LLM writer (Model 5 and the judge are
Qwen; fluent LLM fakes are R5) — and a text that cannot be attacked cleanly is kept unchanged
with ``attacked=False`` and the reason, never emitted incoherent.

* **A1 — selective flip** (``a1_selective``). TextGuard's gate passes a claim that agrees with
  the trend it reads from the lookback numbers. So the attacker flips (P3's flip, byte for byte)
  **only** where the flipped claim agrees with that evidence at TextGuard's frozen window —
  change branch: sign of the OLS slope over the last w values; level branch (a stated national
  baseline B): sign(y_last − B) — and stays silent (real report) everywhere else. Only signs
  matter, so σ plays no role: the attacker needs nothing but the public past numbers. A1 adds a
  *selection*, not new text.
* **A2 — forward-looking lie** (``a2_forward``). The real report is kept as an exact prefix and
  one sentence is appended claiming the **reverse** direction for the next period ("The
  unemployment rate is expected to fall next month."). The judge asks a past-tense question
  ("did … go up or go down?"), so it should read the kept past claim — which looks genuine to the
  gate — while the forecaster can act on the forward claim.
* **A3 — strong magnitude lie** (``a3_magnitude``). P3's exaggeration engine at **K = 10**
  instead of 3 (same consistency rules), plus a plausibility filter fixed in §10f: no percentage
  or percentage-point value ≥ 100 in magnitude, no "by Y … to X" that implies a start ≤ 0, and
  the levels a reader could check against the series itself (Health_US ILI %, SocialGood's
  implied starting unemployment rate) within [0.5·min, 2·max] of the train-split series. Why the
  filter: at K = 10 the engine writes impossible values ("ILI 2252.8%"), and an impossible fake is
  trivially detectable — fair evaluation needs plausible attacks (the P3 level-mirror lesson).

Run: ``python -m src.attacks.adaptive`` → ``texts/a1_selective.csv``, ``a2_forward.csv``,
``a3_magnitude.csv`` (gitignored; same columns as every condition), then
``python -m src.attacks.validate_r3``.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from src.attacks import common
from src.attacks.rules import (NUM, UNIT, EditResult, coherence_problems, edit_text, headline,
                               parse_num, unit_class)
from src.data.splits import load_splits
from src.data.windows import lookback_for, split_windows
from src.textguard.text_direction import POLARITY, stated_baseline
from src.textguard.trend_detector import change_evidence, level_evidence

A1, A2, A3 = "a1_selective", "a2_forward", "a3_magnitude"
R3_CONDITIONS = (A1, A2, A3)
A3_FACTOR = 10.0
FROZEN = common.ROOT / "results" / "summary_p6_frozen.json"


# --- shared inputs ----------------------------------------------------------------------------

def frozen_windows() -> dict[str, int]:
    """TextGuard v1's frozen trend window per domain (P6, val-chosen) — what A1 attacks."""
    return {d: int(w) for d, w in json.loads(FROZEN.read_text(encoding="utf-8"))
            ["textguard"]["window"].items()}


@lru_cache(maxsize=None)
def lookbacks() -> dict[tuple[str, str, str], np.ndarray]:
    """(domain, split, sample_id) → the lookback window in raw units (what TextGuard sees)."""
    out = {}
    for d in common.DOMAINS:
        w = split_windows(load_splits(d), lookback_for(d), horizon=1)
        for split in ("val", "test"):
            win = w[split]
            for i in range(len(win)):
                out[(d, split, win.target_start[i].strftime("%Y-%m-%d"))] = win.X[i].astype(float)
    return out


@lru_cache(maxsize=None)
def train_stats(domain: str) -> tuple[float, float, float]:
    """(min, max, population std) of the domain's train-split series (NaN targets ignored)."""
    v = load_splits(domain).train["value"].to_numpy(float)
    v = v[np.isfinite(v)]
    return float(v.min()), float(v.max()), float(v.std())


# --- A1: selective flip ------------------------------------------------------------------------

def a1_agrees(numbers, fake_text: str, claimed: int, domain: str, window: int) -> bool:
    """Would every branch TextGuard v1 checks agree with ``claimed`` (or see no direction)?

    The attacker's model of the judge: it reads the claim the rules wrote (P6: 94–100% on fakes),
    in both Health_US questions — so both evidence branches must agree.
    """
    sigma = train_stats(domain)[2]          # only the sign is used; any σ > 0 gives the same
    if change_evidence(numbers, sigma, window).direction not in (0, claimed):
        return False
    b = stated_baseline(fake_text)
    if b is not None and level_evidence(numbers, sigma, b).direction not in (0, claimed):
        return False
    return True


def build_a1(w: pd.DataFrame | None = None) -> pd.DataFrame:
    w = common.windows() if w is None else w
    win, lb = frozen_windows(), lookbacks()
    texts, attacked, reason, real_d, claimed, variant, synth = [], [], [], [], [], [], []
    for r in w.itertuples():
        res = edit_text(r.text, r.domain, "flip")
        ok = res.attacked and a1_agrees(lb[(r.domain, r.split, r.sample_id)], res.text,
                                        res.claimed_direction, r.domain, win[r.domain])
        texts.append(res.text if ok else r.text)
        attacked.append(ok)
        reason.append("" if ok else (res.reason if not res.attacked else
                                     "flip would disagree with TextGuard's trend evidence"))
        real_d.append(res.real_direction)
        claimed.append(res.claimed_direction if ok else 0)
        variant.append(res.variant)
        synth.append(ok)
    return common.frame(w, A1, texts, attacked=attacked, reason=reason, real_direction=real_d,
                        claimed_direction=claimed, variant=variant,
                        source_domain=w["domain"].values, source_sample_id=w["sample_id"].values,
                        synthetic=synth)


# --- A2: forward-looking lie -------------------------------------------------------------------

SUBJECT = {"Traffic": "Vehicle travel", "SocialGood": "The unemployment rate",
           "Health_US": "The percentage of outpatient visits for influenza-like illness"}
PERIOD = {"Economy": "month", "Traffic": "month", "SocialGood": "month", "Health_US": "week"}
FORWARD = re.compile(r"(?P<subject>[A-Z][^.]*?) is expected to (?P<verb>rise|fall) "
                     r"next (?P<period>month|week)\.$")


def economy_subject(headline_sentence: str) -> tuple[str, int]:
    """Economy's subject as its headline names it, and that subject's polarity to the balance."""
    surplus = re.search(r"\bsurplus\b", headline_sentence, re.I)
    deficit = re.search(r"\bdeficit\b", headline_sentence, re.I)
    return ("The trade surplus", 1) if surplus and not deficit else ("The trade deficit", -1)


def forward_sentence(domain: str, series_direction: int, headline_sentence: str = "") -> str:
    """The appended claim: ``series_direction`` (+1/−1 about the forecast target) next period."""
    if domain == "Economy":
        subject, pol = economy_subject(headline_sentence)
    else:
        subject, pol = SUBJECT[domain], POLARITY[domain]
    literal = series_direction * pol
    return f"{subject} is expected to {'rise' if literal > 0 else 'fall'} next {PERIOD[domain]}."


def read_forward(sentence: str) -> int:
    """Parse an appended claim back to its series direction (validation; 0 if unreadable)."""
    m = FORWARD.match(sentence.strip())
    if not m:
        return 0
    literal = 1 if m.group("verb") == "rise" else -1
    pol = -1 if re.search(r"\bdeficit\b", m.group("subject"), re.I) else 1
    return literal * pol


def append_claim(text: str, sentence: str) -> str:
    base = text.rstrip()
    return base + (" " if base[-1:] in ".!?" else ". ") + sentence


def a2_edit(text: str | None, domain: str) -> EditResult:
    h = headline(text, domain)
    if not isinstance(text, str) or not text.strip():
        return EditResult(text, False, "no text", 0, 0, "")
    if h.series_direction == 0:
        return EditResult(text, False, "no directional headline", 0, 0, h.variant)
    claimed = -h.series_direction
    return EditResult(append_claim(text, forward_sentence(domain, claimed, h.sentence)), True, "",
                      h.series_direction, claimed, h.variant)


def build_a2(w: pd.DataFrame | None = None) -> pd.DataFrame:
    w = common.windows() if w is None else w
    res = [a2_edit(t, d) for t, d in zip(w["text"], w["domain"])]
    return _frame(w, A2, res)


# --- A3: strong magnitude lie ------------------------------------------------------------------

PCT_AFTER = re.compile(r"\s*(?:%|percent|percentage points?)", re.I)
BY_TO = re.compile(rf"\bby\s+(?P<y>[-+]?{NUM})\s*(?P<yu>{UNIT})?\s+to\s+(?P<x>{NUM})\s*(?P<xu>{UNIT})?",
                   re.I)
UP_VERB = re.compile(r"\b(?:increased|rose|up|gained|grew|climbed|higher)\s*$", re.I)
DOWN_VERB = re.compile(r"\b(?:decreased|fell|declined|down|dropped|lower)\s*$", re.I)


def _edited_values(original: str, res: EditResult) -> list[tuple[float, bool, str]]:
    """(value, is_percent_like, unit_kind) of every number the engine wrote."""
    out = []
    for a, b, new in res.edits:
        m = re.search(rf"[-+]?{NUM}", new)
        if not m:
            continue                                   # a wording edit ("sharply", "well ")
        v = parse_num(m.group().lstrip("+-"))
        after = new[m.end():] + original[b:b + 25]
        pm = PCT_AFTER.match(after)
        kind = "" if not pm else ("pp" if "point" in pm.group().lower() else "percent")
        out.append((v, bool(pm), kind))
    return out


def implied_starts(text: str) -> list[tuple[float, str, str]]:
    """Every "<verb> by Y … to X" in ``text`` → (implied start X ∓ Y, Y's unit, X's unit)."""
    out = []
    for m in BY_TO.finditer(text):
        cy, cx = unit_class(m.group("yu")), unit_class(m.group("xu"))
        if not (cy == cx or (cy == "pp" and cx == "percent")):
            continue
        before = text[max(0, m.start() - 40):m.start()]
        up, down = UP_VERB.search(before), DOWN_VERB.search(before)
        if not (up or down):
            continue
        y, x = parse_num(m.group("y").lstrip("+-")), parse_num(m.group("x"))
        out.append((x - y if up else x + y, cy or "", cx or ""))
    return out


def plausibility_problems(original: str, res: EditResult, domain: str) -> list[str]:
    """§10f's A3 filter (P1–P4). Only problems the edit *created* count (P2, P4)."""
    lo, hi, _ = train_stats(domain)
    bounds = (0.5 * lo, 2 * hi)
    probs = []
    for v, pct, kind in _edited_values(original, res):
        if pct and abs(v) >= 100:
            probs.append(f"P1 percentage {v:g} >= 100")
        if domain == "Health_US" and kind == "percent" and not bounds[0] <= v <= bounds[1]:
            probs.append(f"P3 ILI {v:g}% outside [{bounds[0]:.2f}, {bounds[1]:.2f}]")
    before = set(implied_starts(original))
    for start, yu, xu in implied_starts(res.text):
        if (start, yu, xu) in before:
            continue
        if start <= 0:
            probs.append(f"P2 implied start {start:g} <= 0")
        if domain == "SocialGood" and yu == "pp" and xu == "percent" \
                and not bounds[0] <= start <= bounds[1]:
            probs.append(f"P4 implied starting rate {start:g}% outside "
                         f"[{bounds[0]:.2f}, {bounds[1]:.2f}]")
    return probs


def a3_edit(text: str | None, domain: str) -> EditResult:
    res = edit_text(text, domain, "exaggerate", factor=A3_FACTOR)
    if not res.attacked:
        return res
    probs = plausibility_problems(text, res, domain)
    if probs:
        return EditResult(text, False, f"implausible at x{A3_FACTOR:g}: {probs[0]}",
                          res.real_direction, 0, res.variant)
    return res


def build_a3(w: pd.DataFrame | None = None) -> pd.DataFrame:
    w = common.windows() if w is None else w
    res = [a3_edit(t, d) for t, d in zip(w["text"], w["domain"])]
    return _frame(w, A3, res)


# --- assembly --------------------------------------------------------------------------------

def _frame(w: pd.DataFrame, condition: str, res: list[EditResult]) -> pd.DataFrame:
    return common.frame(w, condition, [r.text for r in res],
                        attacked=[r.attacked for r in res], reason=[r.reason for r in res],
                        real_direction=[r.real_direction for r in res],
                        claimed_direction=[r.claimed_direction for r in res],
                        variant=[r.variant for r in res], source_domain=w["domain"].values,
                        source_sample_id=w["sample_id"].values,
                        synthetic=[r.attacked for r in res])


BUILDERS = {A1: build_a1, A2: build_a2, A3: build_a3}


def build_all() -> dict[str, pd.DataFrame]:
    w = common.windows()
    return {name: fn(w) for name, fn in BUILDERS.items()}


def main() -> None:
    for name, df in build_all().items():
        path = common.save(df, name)
        t = df[df.has_text]
        print(f"{name:<13} {len(df)} rows, with text {len(t)}, attacked {int(df.attacked.sum())} "
              f"-> {path}")


if __name__ == "__main__":
    main()
