"""Prompt for TextGuard v2's claim judge (R4) — one question about one claim unit.

Pure Python, no torch: the **single source of truth** for what the v2 judge sees. The Kaggle
notebook (``notebooks/build_r4_v2_judge_notebook.py``) pastes this file verbatim, and
``v2.load_answers`` rebuilds every prompt locally to check each answer's fingerprint. It keeps
v1's interface (``JUDGE_SYSTEM``, ``ANSWERS``, ``build_judge_prompt``, ``judge_sha``), so the
notebook's scoring code is P6's, unchanged. v1's ``judge_prompt.py`` is not touched.

Same judge as v1 — Qwen2.5-1.5B-Instruct at P5's revision, constrained answer, scored not parsed,
confidence = the chosen answer's probability renormalised over the three — so v1 → v2 changes
*what is asked*, not *who answers*. What changes, each a recorded v1 failure (PROJECT_STATUS §1a):

* **One unit, not the report** (``units.py``): every sentence / bullet / headline is asked about
  on its own, so an appended claim is read (R3: v1 read A2's genuine past claim 91% of the time).
* **Past and future are separate questions.** ``change`` asks what *happened*; ``outlook`` asks
  what the unit *predicts*. A forward claim ("is expected to fall next month") is a claim the
  forecaster acts on (R3: M5's A2 push equalled the full flip) but it is not a change claim, so
  v1 had no question that could catch it.
* **Change and level are told apart.** ``change`` explicitly answers "none" for a unit that only
  states a level or compares with a baseline — v1's change question read Health_US's level claims
  as changes (17% agreement with human labels, §9 P6.0). ``level`` is asked only of units that
  state a national baseline (``v2.level_reference``), as in v1.

Subjects are v1's literal subjects, except Traffic: v1 wrote the year-over-year frame into the
question; v2 reads the frame from the unit (``units.frame``) and checks it against matching
evidence, so the question names only the quantity. Held-out subjects (Energy, FNSPID) are fixed
in the R4 pre-registration (§10g), before any judge output on those domains.
"""

from __future__ import annotations

import hashlib

JUDGE_SYSTEM = ("You read one sentence from a report or a news feed and say exactly what it "
                "claims. You answer with one word.")

# answer words per question kind; the order is (+1, -1, 0)
ANSWERS = {"change": ("up", "down", "none"), "outlook": ("up", "down", "none"),
           "level": ("above", "below", "none")}
KINDS = tuple(ANSWERS)

SUBJECTS = {
    "Economy": "the U.S. international trade deficit",
    "Traffic": "travel (vehicle miles traveled) on U.S. roads",
    "SocialGood": "the U.S. unemployment rate",
    "Health_US": "the percentage of outpatient visits for influenza-like illness (ILI)",
}


def build_judge_prompt(domain: str, kind: str, unit: str) -> str:
    """The user message asking one question (``kind``) about one claim unit."""
    s = SUBJECTS[domain]
    pos, neg, none = ANSWERS[kind]
    if kind == "change":
        question = f"Does this sentence say that {s} increased or decreased?"
        rule = (f'Answer "{pos}" if it says {s} increased, "{neg}" if it says it decreased, and '
                f'"{none}" if it says it was unchanged, only states its level or compares it with a '
                "baseline, only predicts the future, or is about something else.")
    elif kind == "outlook":
        question = f"Does this sentence predict that {s} will go up or go down?"
        rule = (f'Answer "{pos}" if it says {s} will (or is expected to) increase, "{neg}" if it '
                f'says it will (or is expected to) decrease, and "{none}" if it makes no prediction '
                "about it, only describes what already happened, or is about something else.")
    elif kind == "level":
        question = f"Does this sentence say that {s} is above or below the national baseline?"
        rule = (f'Answer "{pos}" if it says it is above the national baseline, "{neg}" if it says '
                f'it is below, and "{none}" if it says it is at the baseline, does not say, or is '
                "about something else.")
    else:
        raise ValueError(f"unknown question kind {kind!r}")
    return "\n".join(["Sentence:", '"""', str(unit).strip(), '"""', "",
                      f"Question: {question}", rule,
                      f"Answer with one word: {pos}, {neg} or {none}."])


def judge_sha(user_prompt: str) -> str:
    """Short fingerprint of a judge prompt, stored with each answer for provenance."""
    return hashlib.sha256((JUDGE_SYSTEM + "\n\n" + user_prompt).encode("utf-8")).hexdigest()[:16]
