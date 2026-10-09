"""Prompt for TextGuard's direction judge (P6) — Qwen2.5-1.5B-Instruct reads what a report claims.

Pure Python, no torch: like ``src/models/qwen_prompt.py`` this is the **single source of truth**
for what the judge sees. ``notebooks/p6_direction_judge.ipynb`` embeds this file verbatim
(``tests/test_judge.py`` fails if it drifts), and ``text_direction.py`` rebuilds every prompt
locally to check that a judge CSV was produced by *this* prompt.

Why Qwen and not the P3 rules: the fakes were written by the rules, so rules judging them would
be circular — they would recognise their own edits perfectly. The judge must be a reader that
had no part in writing the attack (PROJECT_STATUS §10b, issue #25).

Design decisions (fixed before any judge output existed, §10b):

* **The question names the forecast target's subject**, per domain, because TextGuard asks
  "what does this text claim about *my* series?". A report about another subject (e.g. a
  cross-domain text) should be answered ``none``.
* **Literal subject, not the series.** Economy's question is about the *trade deficit*, the
  quantity the reports talk about; ``text_direction.py`` maps "deficit up" to "balance down".
  Asking a 1.5B model to do that sign inversion itself would add an avoidable error source, and
  the literal answer is what the P1 human labels describe.
* **Two kinds of claim.** ``change`` — did it go up or down (all domains). ``level`` — is it
  above or below the stated national baseline (Health_US only: its reports state levels,
  P3 attacks them with the level-flip).
* **Constrained answer, scored, not parsed.** The notebook computes the probability of each of
  the three allowed answers (teacher forcing) and takes the most likely; confidence = its
  probability renormalised over the three. So there are no parse failures, and the confidence
  is the model's own, not a number it was asked to make up.
"""

from __future__ import annotations

import hashlib

JUDGE_SYSTEM = ("You read official statistical reports and say exactly what they claim. "
                "You answer with one word.")

# answer words per question kind; the order is (+1, -1, 0)
ANSWERS = {"change": ("up", "down", "none"), "level": ("above", "below", "none")}

# per domain: (question kind, subject as written in the question). Health_US asks both.
QUESTIONS = {
    "Economy": [("change", "the U.S. international trade deficit")],
    "Traffic": [("change", "travel (vehicle miles traveled) on U.S. roads compared with the "
                           "same month one year earlier")],
    "SocialGood": [("change", "the U.S. unemployment rate")],
    "Health_US": [("level", "the percentage of outpatient visits for influenza-like illness "
                            "(ILI)"),
                  ("change", "the percentage of outpatient visits for influenza-like illness "
                             "(ILI)")],
}


def build_judge_prompt(domain: str, kind: str, text: str) -> str:
    """The user message asking one question about one report."""
    subject = dict(QUESTIONS[domain])[kind]
    pos, neg, none = ANSWERS[kind]
    if kind == "change":
        question = f"According to this report, did {subject} go up or go down?"
        rule = (f'Answer "{pos}" if the report says it increased, "{neg}" if the report says '
                f'it decreased, and "{none}" if the report says it was unchanged, does not '
                "say, or is about something else.")
    else:
        question = (f"According to this report, was {subject} above or below the national "
                    "baseline?")
        rule = (f'Answer "{pos}" if the report says it was above the national baseline, '
                f'"{neg}" if the report says it was below, and "{none}" if the report says it '
                "was at the baseline, does not say, or is about something else.")
    return "\n".join(["Report:", '"""', str(text).strip(), '"""', "",
                      f"Question: {question}", rule,
                      f"Answer with one word: {pos}, {neg} or {none}."])


def judge_sha(user_prompt: str) -> str:
    """Short fingerprint of a judge prompt, stored with each answer for provenance."""
    return hashlib.sha256((JUDGE_SYSTEM + "\n\n" + user_prompt).encode("utf-8")).hexdigest()[:16]
