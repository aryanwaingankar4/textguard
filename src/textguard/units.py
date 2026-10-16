"""TextGuard v2 (R4): split a report into claim units and read each unit's reference frame.

Why units. TextGuard v1 asked one question about the *whole* report, so the judge answered with
the report's main claim and never looked at the rest: R3's A2 appended one forward-looking
sentence ("The trade deficit is expected to fall next month.") and the judge still read the
genuine past claim in 91% of A2 texts (PROJECT_STATUS §9 R3.2). v2 asks its questions about
**every unit** — a sentence, a bullet or a headline — so each claim is checked on its own and the
gate can take the most suspicious one. A unit is judged **without** the rest of the report: given
the whole report, a small judge answers from its main claim again (the A2 failure). Identical
units are judged once, which also keeps the judge job small (A2 = the real units + one).

Why frames. A claim is only checkable against evidence in its own reference frame (§9 P6,
Amendment 1): Traffic reports state **year-over-year** changes ("for July 2017 as compared with
July 2016"), which an 8-month trend cannot verify. The frame is read from the unit's *dates and
phrases* with a fixed rule — a structural reading like v1's baseline regex, not a judgement of
direction, and no attack edits it (the P3/R3/R4 engines change verbs, comparators and numbers).

* ``yoy`` — an explicit year-over-year phrase ("a year ago", "over the year", "year-over-year",
  "same month last year", …) or the same month named with two consecutive years.
* ``unchecked`` — cumulative / year-to-date / multi-month-average claims ("Cumulative travel for
  2017", "for the three months ending in November"): no evidence we compute matches them, so
  they are **not gated** (g = 1) rather than checked against the wrong numbers.
* ``trend`` — everything else: v1's change evidence (the recent lookback trend).
"""

from __future__ import annotations

import re

_MARKDOWN = re.compile(r"\*\*|__")
_BULLET = re.compile(r"^\s*(?:[*\-•]|\d+[.)])\s+")
_HEADER = re.compile(r"^\s*objective facts\b[^:]*:\s*", re.I)
_HEADER_ONLY = re.compile(r"^\W*objective facts\b[^:\d]*$", re.I)
# a sentence that opens with a pronoun refers back to the previous one ("ILI was 1.0%. This
# percentage is below the national baseline …"): it stays in the same unit
_ANAPHOR = re.compile(r"^(?:This|That|These|Those|It|Its|They|Their)\b")
# a sentence ends at . ! ? followed by space and an upper-case letter, digit, quote or bracket
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[$])")
# tokens whose final period is not a sentence end
_ABBREV = re.compile(r"(?:\b(?:[A-Z]\.){1,3}|\b(?:Inc|Corp|Co|Ltd|No|Nos|vs|Mr|Mrs|Ms|Dr|St|Jr|Sr|"
                     r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|approx|est|e\.g|i\.e)\.)$")
_MIN_LETTERS = 3


def _sentences(line: str) -> list[str]:
    parts = _BOUNDARY.split(line)
    out: list[str] = []
    for p in parts:
        if out and _ABBREV.search(out[-1]):
            out[-1] = f"{out[-1]} {p}"          # "the U.S." + "Census Bureau …" is one sentence
        else:
            out.append(p)
    return out


def units(text: str | None) -> list[str]:
    """The report's claim units, in order: lines → bullets/markdown/headers stripped → sentences.

    A headline bundle (FNSPID: one headline per line) gives one unit per headline. A sentence
    opening with a pronoun joins the previous sentence of the same line. Header-only lines
    ("Objective Facts") and fragments with fewer than three letters (stray markdown) are dropped.
    Order and duplicates are kept — the gate takes the minimum, so a duplicate changes nothing.
    """
    if not isinstance(text, str):
        return []
    out: list[str] = []
    for line in text.split("\n"):
        line = _MARKDOWN.sub("", line).strip()
        if _HEADER_ONLY.match(line):
            continue
        line = _BULLET.sub("", line).strip()
        line = _HEADER.sub("", line).strip()
        in_line = False
        for s in _sentences(line):
            s = s.strip()
            if len(re.findall(r"[A-Za-z]", s)) < _MIN_LETTERS:
                continue
            if in_line and _ANAPHOR.match(s):
                out[-1] = f"{out[-1]} {s}"
            else:
                out.append(s)
            in_line = True
    return out


MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december")
_MONTH_ABBR = {m[:3]: m for m in MONTHS} | {"sept": "september"}
_MONTH_YEAR = re.compile(r"\b(" + "|".join(MONTHS) + r"|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)"
                         r"\.?,?\s+(?:\d{1,2},?\s+)?((?:19|20)\d{2})\b", re.I)
_YOY_PHRASE = re.compile(
    r"\b(?:year[- ]over[- ]year|year[- ]on[- ]year|a year (?:ago|earlier|before)|"
    r"(?:one|1) year (?:ago|earlier|before)|over the (?:past |last )?year|from (?:a|one|1) year|"
    r"(?:from|than|over|since|compared (?:to|with)) (?:the same (?:month|week|period|quarter) )?last year|"
    r"same (?:month|week|period|quarter|time) (?:of |in )?(?:last|a|the (?:previous|prior)|one) year|"
    r"(?:previous|prior) year'?s?|annual(?:ly)? (?:rate|basis|change|increase|decrease|decline|rise|gain)|"
    r"12[- ]month (?:change|increase|decrease|decline|rise)|yoy|y/y)\b", re.I)
_UNCHECKED = re.compile(
    r"\b(?:cumulative(?:ly)?|year[- ]to[- ]date|so far (?:this|in) (?:the )?year|ytd|"
    r"(?:average|total) .{0,40}?for the (?:two|three|four|six|twelve|\d+) (?:months|weeks) ending|"
    r"(?:two|three|four|six|twelve|\d+)[- ]month (?:average|moving average))\b", re.I)


def _month_years(unit: str) -> list[tuple[str, int]]:
    out = []
    for m, y in _MONTH_YEAR.findall(unit):
        m = m.lower().rstrip(".")
        out.append((_MONTH_ABBR.get(m, m), int(y)))
    return out


# A unit that also names a short-term comparison ("… $0.099 per gallon less than last week and
# $0.590 under a year ago") is checked against the TREND: its verb states the short-term move.
# Decided by Aryan 2026-10-16 (option a) and pre-registered; prompted by Energy's text format (seen
# in the R4 data checks), with no outcome data used — no original-domain val unit has both.
_SHORT_TERM = re.compile(r"\b(?:(?:last|previous|prior|the preceding)\s+(?:week|month)'?s?|"
                         r"week[- ]over[- ]week|month[- ]over[- ]month|a week (?:ago|earlier)|"
                         r"a month (?:ago|earlier))\b", re.I)


def frame(unit: str) -> str:
    """'yoy' | 'unchecked' | 'trend' — the reference frame of a unit's change claim (fixed rule)."""
    if _UNCHECKED.search(unit):
        return "unchecked"
    my = _month_years(unit)
    yoy = bool(_YOY_PHRASE.search(unit)) or any(m1 == m2 and y2 == y1 - 1 for m1, y1 in my for m2, y2 in my)
    if yoy and not _SHORT_TERM.search(unit):
        return "yoy"
    return "trend"
