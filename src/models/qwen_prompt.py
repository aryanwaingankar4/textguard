"""Prompt construction and output parsing for Model 5 (Qwen2.5-1.5B-Instruct, early fusion).

Pure Python, no torch: this module is the **single source of truth** for what Qwen sees and
how its answer becomes a number. ``notebooks/p2_qwen_forecaster.ipynb`` embeds this file's
functions verbatim (``tests/test_qwen.py`` fails if they drift), and ``qwen_forecaster.py``
re-builds the prompts locally to check that a predictions CSV was produced by *this* prompt.

Design decisions (PROJECT_STATUS §7b):

* **Early fusion** — numbers and text in one prompt; the model decides how to combine them.
* **No dates in the numeric part.** Values are labelled ``t-L … t-1`` (relative steps), so
  the prompt never says "March 2020". This reduces Qwen recalling memorised historical values
  from pretraining. The report text is left as published and *can* contain dates — a stated
  limitation, not something we edit away (editing would change the text condition itself).
* **``text=None`` uses the identical prompt with the report block removed** — not a
  placeholder like "no report". So A (no text) is a valid, natural prompt and B − A is the
  effect of adding the report, nothing else.
* **Parsing never drops a sample.** ``parse_number`` returns ``nan`` on failure; the caller
  records ``parse_ok=False``. P4's NaN-pairing rule (PROJECT_STATUS §6a) handles the rest.
"""

from __future__ import annotations

import hashlib
import math
import re

SYSTEM = ("You are an expert time-series forecaster. "
          "You answer with a single number and nothing else.")

PERIOD = {"monthly": "month", "weekly": "week"}


def format_value(v: float) -> str:
    """Compact, lossless-enough rendering: 282558.0 -> '282558', -66029.2 -> '-66029.2'."""
    return f"{float(v):.10g}"


def build_prompt(description: str, unit: str, freq: str, lookback, text: str | None) -> str:
    """The user message. ``text`` None/blank -> the same prompt without the report block."""
    period = PERIOD[freq]
    n = len(lookback)
    lines = [f"Series: {description}.",
             f"Unit: {unit}. Frequency: {freq}.",
             "",
             f"Most recent {n} values, oldest first (t-1 is the latest {period}):"]
    lines += [f"t-{n - i}: {format_value(v)}" for i, v in enumerate(lookback)]
    if text is not None and str(text).strip():
        lines += ["",
                  "Report published before period t:",
                  '"""',
                  str(text).strip(),
                  '"""']
    lines += ["",
              f"What is the value at t, the next {period}? "
              f"Answer with one number in {unit}, no words."]
    return "\n".join(lines)


def prompt_sha(user_prompt: str) -> str:
    """Short fingerprint of a prompt, stored with each prediction for provenance."""
    return hashlib.sha256((SYSTEM + "\n\n" + user_prompt).encode("utf-8")).hexdigest()[:16]


_NUM = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_STEP_LABEL = re.compile(r"\bt\s*[-+]\s*\d+\b", re.IGNORECASE)   # echoed "t-1" labels
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}


def parse_number(output: str, unit: str = "") -> float:
    """First number in Qwen's answer, in the series' unit; ``nan`` if there is none.

    Handles what a chat model typically does: thousands separators ("282,558"), a Unicode
    minus ("−58.4"), echoed step labels ("t-1: …" is skipped, not read as -1), and a scale
    word that differs from the unit ("-66.4 billion" when the unit is "million US dollars"
    -> -66400). Anything else unparseable is ``nan`` — never a guess.
    """
    if not isinstance(output, str):
        return math.nan
    s = output.replace("−", "-").replace("–", "-")
    s = _STEP_LABEL.sub(" ", s)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)                       # 282,558 -> 282558
    m = _NUM.search(s)
    if m is None:
        return math.nan
    value = float(m.group())
    word = re.match(r"\s*(thousand|million|billion|trillion)\b", s[m.end():], re.IGNORECASE)
    unit_word = re.match(r"\s*(thousand|million|billion|trillion)\b", unit, re.IGNORECASE)
    if word:
        said = _SCALE[word.group(1).lower()]
        base = _SCALE[unit_word.group(1).lower()] if unit_word else 1.0
        value *= said / base
    return value if math.isfinite(value) else math.nan
