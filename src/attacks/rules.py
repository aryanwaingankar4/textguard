"""Deterministic rule engine for the two fake-text attacks: ``flip`` and ``exaggerate``.

Why rules, not an LLM (decided 2026-10-09, PROJECT_STATUS §8): Model 5 is Qwen and P6's
direction judge is Qwen, so a Qwen-written attack would bias both (issue #25). Rules make a
**minimal, exact, reproducible** edit — only the direction (flip) or the magnitude
(exaggerate) changes — and every edit is auditable. In the paper: a conservative,
minimal-edit attacker.

How a text is edited
--------------------
All detectors run on the *original* text and propose span edits; the edits are applied in
one pass at the end, so nothing is flipped twice. Then the result is checked, and if any
check fails the text is returned **not attacked** — never a self-contradictory fake.

Flip (``mode="flip"``):

* direction words swap (increased↔decreased, rose↔fell, declined→rose, up↔down before a
  number / "by" / "from" / "to", more↔less before "than", higher↔lower, above↔below,
  gains↔losses, growth→decline …), case preserved;
* signed change numbers flip sign ("-1.6%" → "+1.6%"); unsigned numbers after
  "changed by" / "change of" (implicitly positive) become negative;
* numbers that would contradict the new direction are recomputed, keeping the *current*
  level fixed (it is the most recent value, which the forecaster also sees as a number):
  "X, up Y from Z" → "X, down Y from X+Y"; "X, down from Z" → "X, up from 2X−Z";
  "rose from A to B" → "fell from 2B−A to B";
* level claims vs a stated baseline: "V%, below the national baseline of B%" → the value is
  mirrored to the same distance on the other side, V' = 2B − V, so the claim "above the
  baseline of B%" is numerically true and the value stays plausible. Only if that would be
  ≤ 0 (a flu peak far above baseline) does it fall back to the ratio mirror V' = B²/V, which
  is always positive (``level_mirror``). Decided 2026-10-09: the ratio mirror alone turned
  0.9% into 5.4% — a flu-peak level in late summer. Implausible fakes are trivially
  detectable and would flatter TextGuard in P6; fair evaluation needs plausible attacks.

Exaggerate (``mode="exaggerate"``): direction kept; stated change magnitudes ×``FACTOR``
(3), the same anchor numbers recomputed with the same rules (Z' = X ∓ 3Y, A' = B − 3(B − A),
V' = B·(V/B)³), "sharply" added after the headline verb ("slightly" → "sharply", "edged
up/down" → "rose/fell sharply"), "well" before above/below a baseline. A decrease of ≥ 100% is impossible → not attacked.

Not attacked when: no text; no directional headline (e.g. "unchanged"); a directional word
the rules do not handle (flip only — it would be left contradicting the rest); an anchor
number cannot be found or would become ≤ 0; or a check on the output fails (headline
direction, arithmetic of "up Y from Z", baseline side, sign agreement in "changed by S% (T)").

Directions
----------
``text_direction``: what the headline literally claims about its subject (+1 up, −1 down).
``series_direction``: the same claim mapped onto the forecast target. Only Economy differs:
its target is the trade *balance* and its headlines are about the *deficit*, so
"deficit increased" = series −1.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

FACTOR = 3.0

# --- lexicon ---------------------------------------------------------------------------------

# word -> opposite (lower case); plain swaps, applied anywhere
SWAP = {
    "increased": "decreased", "decreased": "increased",
    "increase": "decrease", "decrease": "increase",
    "increases": "decreases", "decreases": "increases",
    "increasing": "decreasing", "decreasing": "increasing",
    "rose": "fell", "fell": "rose",
    "rise": "fall", "fall": "rise", "rises": "falls", "falls": "rises",
    "rising": "falling", "falling": "rising",
    "declined": "rose", "decline": "rise", "declines": "rises", "declining": "rising",
    "higher": "lower", "lower": "higher",
    "above": "below", "below": "above",
    "highest": "lowest", "lowest": "highest",
    "gain": "loss", "loss": "gain", "gains": "losses", "losses": "gains",
    "growth": "decline",
}
# Words that are directional only in context (found by scanning every val/test report):
# * up/down: before a number, "by", "from", "to", "less/more than" ("down less than $0.1B from")
# * more/less, higher/lower: only right after a change magnitude ("$2.9 billion more than",
#   "1% lower than") — NOT "unemployed for less than 5 weeks", NOT group comparisons
#   ("5.8 percent, higher than the 4.7-percent rate for non-Hispanics")
# * above/below: only next to a baseline / threshold — NOT "158,000, below the average of 223,000"
# * highest/lowest: only in "… since / in the history / on record" — NOT "the highest rates for
#   American Indians" (a group comparison)
UP_DOWN_CTX = r"(?=\s+(?:by\b|from\b|to\b|less than\b|more than\b|\$?\d))"
MAGNITUDE_BEFORE = re.compile(r"\d(?:[\d,]*\d)?(?:\.\d+)?\s*(?:%|percent|billion|million|thousand)?\s*$")


def directional_here(text: str, start: int, end: int) -> bool:
    """Is the lexicon word at ``text[start:end]`` a directional claim in this context?"""
    w = text[start:end].lower()
    nxt, prev = text[end:end + 50], text[max(0, start - 30):start]
    if w in ("up", "down"):
        return re.match(UP_DOWN_CTX, nxt) is not None
    if w in ("more", "less"):
        return re.match(r"\s+than\b", nxt) is not None and MAGNITUDE_BEFORE.search(prev) is not None
    if w in ("higher", "lower"):
        return MAGNITUDE_BEFORE.search(prev) is not None
    if w in ("above", "below"):
        return re.match(r".{0,40}?(?:baseline|threshold)", nxt, re.S) is not None
    if w in ("highest", "lowest"):
        # wide window: "the highest rate and largest over-the-month increase in the history"
        return re.match(r".{0,80}?(?:\bsince\b|in the history|on record|\bever\b)",
                        text[end:end + 90], re.S) is not None
    return w in SWAP
UP_WORDS = {"increased", "increase", "increases", "increasing", "rose", "rise", "rises",
            "rising", "higher", "above", "gain", "gains", "growth", "up", "more"}
DOWN_WORDS = {"decreased", "decrease", "decreases", "decreasing", "fell", "fall", "falls",
              "falling", "declined", "decline", "declines", "declining", "lower", "below",
              "loss", "losses", "down", "less"}
# directional words we do NOT rewrite: if present, a flip would leave them contradicting
UNHANDLED = ["surged", "surge", "plunged", "plunge", "jumped", "dropped", "drop", "climbed",
             "soared", "slipped", "spiked", "spike", "peaked", "elevated", "improved",
             "worsened", "recovered", "rebounded", "doubled", "halved", "widened", "narrowed",
             "reduced", "lost", "shrank", "grew"]
CHANGE_VERBS = {"increased", "decreased", "rose", "fell", "declined", "increasing",
                "decreasing", "rising", "falling", "declining"}
MILDERS = r"\b(?:slightly|marginally|modestly|somewhat)\b"

YEAR = re.compile(r"^(?:19|20)\d\d$")
NUM = r"\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
UNIT = r"(?:billion|million|thousand|percentage points?|percent|%)"
QTY = re.compile(rf"(?P<num>{NUM})(?:\s*(?P<unit>{UNIT}))?")


# --- numbers ---------------------------------------------------------------------------------

def parse_num(tok: str) -> float:
    return float(tok.replace("$", "").replace(",", ""))


def decimals(tok: str) -> int:
    t = tok.replace("$", "").replace(",", "")
    return len(t.split(".")[1]) if "." in t else 0


def fmt_like(value: float, like: str) -> str:
    """Format ``value`` like the token ``like``: same decimals, $ and thousands commas."""
    d = decimals(like)
    s = f"{abs(value):,.{d}f}" if "," in like else f"{abs(value):.{d}f}"
    return ("$" if like.startswith("$") else "") + s


def unit_class(u: str | None) -> str | None:
    if u is None:
        return None
    u = u.lower()
    return "percent" if u in ("%", "percent") else ("pp" if u.startswith("percentage") else u)


# --- sentences and the headline claim -------------------------------------------------------

# a sentence ends at "." (not a decimal point, not "U.S."), a newline or a bullet "*"
SENT = re.compile(r"(?:U\.S\.|\.(?=\d)|[^.\n*])+(?:\.(?!\d)|$|(?=[\n*]))")

# sentences never used as the headline (they are not about the forecast target itself)
HEADLINE_EXCLUDE = {"Health_US": r"\bregions?\b|region-specific"}

DOMAIN_SUBJECT = {
    # (subject regex, preferred-sentence regex or None, polarity text->series)
    "Economy": (r"\b(?:deficit|surplus)\b", None, None),
    "Traffic": (r"\b(?:travel|VMT|vehicle miles)\b", r"all roads and streets", 1),
    "SocialGood": (r"\bunemployment rate\b", None, 1),
    "Health_US": (r"\b(?:ILI|influenza-like illness|this percentage|patient visits)\b",
                  r"baseline", 1),
}
WORD_RE = re.compile(r"[A-Za-z]+")
CHANGE_ANCHOR = re.compile(r"\bchang(?:e|ed|es|ing)\s+(?:by|of)\s+", re.I)


def sentences(text: str):
    for m in SENT.finditer(text):
        if m.group().strip():
            yield m.start(), m.end()


def _first_cue(s: str, start: int = 0) -> tuple[int, str]:
    """First directional cue in ``s`` at or after ``start``: (+1/-1, kind) or (0, '')."""
    best = (len(s) + 1, 0, "")
    for m in WORD_RE.finditer(s, start):
        w = m.group().lower()
        if (w in UP_WORDS or w in DOWN_WORDS) and not directional_here(s, m.start(), m.end()):
            continue
        if w in UP_WORDS or w in DOWN_WORDS:
            kind = "level" if w in ("above", "below") else "change"
            best = min(best, (m.start(), 1 if w in UP_WORDS else -1, kind))
            break
    for m in CHANGE_ANCHOR.finditer(s, start):  # "changed by -1.6%": the sign is the claim
        num = re.match(rf"([-+]?)({NUM})", s[m.end():])
        if num and parse_num(num.group(2)) != 0:
            best = min(best, (m.start(), -1 if num.group(1) == "-" else 1, "change"))
        break
    return best[1], best[2]


@dataclass
class Headline:
    text_direction: int   # literal claim about the headline subject
    series_direction: int  # mapped to the forecast target
    variant: str           # "change" | "level" | ""
    sentence: str = ""


def headline(text: str | None, domain: str) -> Headline:
    """The main directional claim about the forecast target, or direction 0 if none."""
    if not isinstance(text, str) or not text.strip():
        return Headline(0, 0, "")
    subj, prefer, polarity = DOMAIN_SUBJECT[domain]
    sents = [text[a:b] for a, b in sentences(text)]
    ordered = ([s for s in sents if prefer and re.search(prefer, s, re.I)]
               + [s for s in sents if not (prefer and re.search(prefer, s, re.I))])
    exclude = HEADLINE_EXCLUDE.get(domain)
    for s in ordered:
        m = re.search(subj, s, re.I)
        if not m or (exclude and re.search(exclude, s, re.I)):
            continue
        d, kind = _first_cue(s, m.start())
        if d == 0 and re.search(r"baseline", s, re.I):   # "This percentage is below …"
            d, kind = _first_cue(s, 0)
        if d != 0:
            pol = polarity
            if pol is None:  # Economy: deficit up = balance down; surplus up = balance up
                pol = -1 if re.search(r"\bdeficit\b", s[m.start():], re.I) else 1
            return Headline(d, d * pol, kind, s.strip())
    return Headline(0, 0, "")


# --- result ----------------------------------------------------------------------------------

@dataclass
class EditResult:
    text: str | None                 # attacked text; the ORIGINAL text if not attacked
    attacked: bool
    reason: str                      # why not attacked ("" if attacked)
    real_direction: int              # series direction claimed by the original
    claimed_direction: int           # series direction claimed by the output (0 if not attacked)
    variant: str                     # "change" | "level" | ""
    n_edits: int = 0
    edits: list = field(default_factory=list)


def _not(text, reason, h: Headline) -> EditResult:
    return EditResult(text, False, reason, h.series_direction, 0, h.variant)


# --- the engine -------------------------------------------------------------------------------

def _case_like(new: str, old: str) -> str:
    if old.isupper() and len(old) > 1:
        return new.upper()
    if old[:1].isupper():
        return new[:1].upper() + new[1:]
    return new


def _sentence_bounds(text: str, pos: int) -> tuple[int, int]:
    for a, b in sentences(text):
        if a <= pos < b:
            return a, b
    return 0, len(text)


def _anchor_before(text: str, pos: int, want_unit: str | None):
    """Last quantity before ``pos`` in the same sentence with a matching unit (not a year)."""
    a, _ = _sentence_bounds(text, pos)
    found = None
    for m in QTY.finditer(text, a, pos):
        tok, u = m.group("num"), unit_class(m.group("unit"))
        if YEAR.match(tok) and u is None:
            continue
        if want_unit is None or u == want_unit or (u is None and want_unit in ("percent",)):
            found = m
    return found


class Edits:
    def __init__(self):
        self.spans: list[tuple[int, int, str]] = []

    def add(self, a: int, b: int, new: str) -> None:
        for x, y, _ in self.spans:
            if a < y and x < b:
                if (x, y) == (a, b):
                    return
                raise ValueError("overlap")
        self.spans.append((a, b, new))

    def taken(self, a: int, b: int) -> bool:
        return any(a < y and x < b for x, y, _ in self.spans)

    def apply(self, text: str) -> str:
        out, last = [], 0
        for a, b, new in sorted(self.spans):
            out += [text[last:a], new]
            last = b
        return "".join(out + [text[last:]])


def level_mirror(v: float, b: float) -> float:
    """Flip a level to the other side of baseline ``b``: linear 2b − v, or b²/v if that is ≤ 0."""
    lin = 2 * b - v
    return lin if lin > 0 else b * b / v


def _flip_sign_token(sign: str, num: str, mode: str, factor: float = FACTOR) -> str:
    v = parse_num(num)
    if mode == "exaggerate":
        return sign + fmt_like(v * factor, num)
    if v == 0:
        return sign + num
    return ("+" if sign == "-" else "-") + num


def edit_text(text: str | None, domain: str, mode: str, factor: float = FACTOR) -> EditResult:
    """Flip or exaggerate one report; see the module docstring for every rule.

    ``factor`` is the exaggeration factor (P3: 3; R3's A3 magnitude lie: 10). It is ignored by
    ``flip``. The default keeps P3's output byte-identical.
    """
    if mode not in ("flip", "exaggerate"):
        raise ValueError("mode must be 'flip' or 'exaggerate'")
    if not isinstance(text, str):
        text = None
    h = headline(text, domain)
    if not text or not text.strip():
        return _not(text, "no text", h)
    if h.series_direction == 0:
        return _not(text, "no directional headline", h)
    if mode == "flip":
        bad = [w for w in UNHANDLED if re.search(rf"\b{w}\b", text, re.I)]
        if bad:
            return _not(text, f"unhandled directional word: {bad[0]}", h)
    E = Edits()
    try:
        _numbers(text, E, mode, factor)
        if mode == "flip":
            _words(text, E)
        else:
            _emphasis(text, E, h)
    except _Skip as e:
        return _not(text, str(e), h)
    except ValueError as e:
        return _not(text, f"conflicting edits ({e})", h)
    out = _fix_articles(E.apply(text))
    h2 = headline(out, domain)
    want = -h.series_direction if mode == "flip" else h.series_direction
    if h2.series_direction != want:
        return _not(text, "headline check failed", h)
    new_problems = set(coherence_problems(out)) - set(coherence_problems(text))
    if new_problems:
        return _not(text, f"coherence check failed: {sorted(new_problems)[0]}", h)
    if out == text:
        return _not(text, "no edit made", h)
    return EditResult(out, True, "", h.series_direction, h2.series_direction, h.variant,
                      len(E.spans), sorted(E.spans))


class _Skip(Exception):
    pass


OPPOSITE = {**SWAP, "up": "down", "down": "up", "more": "less", "less": "more"}


def _words(text: str, E: Edits) -> None:
    for m in WORD_RE.finditer(text):
        w = m.group().lower()
        if w in OPPOSITE and directional_here(text, m.start(), m.end()) \
                and not E.taken(m.start(), m.end()):
            E.add(m.start(), m.end(), _case_like(OPPOSITE[w], m.group()))


def _emphasis(text: str, E: Edits, h: Headline) -> None:
    """Exaggerate wording: 'slightly' -> 'sharply', 'sharply' after the headline verb,
    'well' before above/below a baseline, 'edged up/down' -> 'rose/fell sharply' (lexicon
    words, so the headline check can still read the direction)."""
    for m in re.finditer(MILDERS, text, re.I):
        E.add(m.start(), m.end(), _case_like("sharply", m.group()))
    for m in re.finditer(r"\bedged (up|down)\b", text, re.I):
        E.add(m.start(), m.end(), _case_like("rose sharply" if m.group(1).lower() == "up"
                                             else "fell sharply", m.group()))
    for m in re.finditer(r"\b(?<!well )(above|below)(?= (?:the |their )?[\w-]* ?baseline)",
                         text, re.I):
        E.add(m.start(), m.start(), "well ")
    if h.sentence:
        i = text.find(h.sentence)
        for m in WORD_RE.finditer(text, max(i, 0), max(i, 0) + len(h.sentence)):
            if m.group().lower() in CHANGE_VERBS:
                after = text[m.end():m.end() + 14]
                if not re.match(r"\s+(?:sharply|slightly|marginally|modestly|somewhat)", after):
                    E.add(m.end(), m.end(), " sharply")
                break


def _numbers(text: str, E: Edits, mode: str, factor: float = FACTOR) -> None:
    flip = mode == "flip"
    # 1. "X …, up/down [Y] from Z"
    for m in re.finditer(rf"\b(?P<dir>up|down)\s+(?:(?:by\s+)?(?P<y>{NUM})\s*(?P<yu>{UNIT})?\s+)?"
                         rf"from\s+(?P<z>{NUM})\s*(?P<zu>{UNIT})?", text, re.I):
        zu = unit_class(m.group("zu"))
        x = _anchor_before(text, m.start(), zu)
        if x is None:
            raise _Skip("no anchor for 'up/down … from'")
        X, Z = parse_num(x.group("num")), parse_num(m.group("z"))
        up = m.group("dir").lower() == "up"
        if m.group("y"):
            Y = parse_num(m.group("y"))
            Y2 = Y if flip else Y * factor
            # keep X; new direction (flip) or same direction with bigger step (exaggerate)
            new_up = (not up) if flip else up
            Z2 = X - Y2 if new_up else X + Y2
            if not flip:
                E.add(m.start("y"), m.end("y"), fmt_like(Y2, m.group("y")))
        else:
            Z2 = 2 * X - Z if flip else X - factor * (X - Z)
        if Z2 <= 0:
            raise _Skip("recomputed 'from' value would be <= 0")
        E.add(m.start("z"), m.end("z"), fmt_like(Z2, m.group("z")))
    # 2. "<verb> from A to B"
    for m in re.finditer(rf"\b(?P<v>increased|decreased|rose|fell|declined|went (?:up|down))\s+"
                         rf"from\s+(?P<a>{NUM})\s*(?P<au>{UNIT})?\s+to\s+(?P<b>{NUM})", text, re.I):
        A, B = parse_num(m.group("a")), parse_num(m.group("b"))
        A2 = 2 * B - A if flip else B - factor * (B - A)
        if A2 <= 0:
            raise _Skip("recomputed 'from A' value would be <= 0")
        E.add(m.start("a"), m.end("a"), fmt_like(A2, m.group("a")))
    # 3. "changed by S% (T billion …)" and "change of a, b, and c, respectively"
    item = re.compile(rf"(?P<sign>[-+]?)(?P<num>\d[\d,]*(?:\.\d+)?)(?P<pct>\s*%)?"
                      rf"(?:\s*\((?P<psign>[-+]?)(?P<pnum>\d[\d,]*(?:\.\d+)?)\s*(?:billion|million))?")
    sep = re.compile(r",\s*(?:and\s+)?|\s+and\s+")
    for a in CHANGE_ANCHOR.finditer(text):
        pos = a.end()
        while True:
            it = item.match(text, pos)
            if not it or YEAR.match(it.group("num")):
                break
            for s_key, n_key in (("sign", "num"), ("psign", "pnum")):
                if it.group(n_key) is None:
                    continue
                v = parse_num(it.group(n_key))
                if not flip and it.group(s_key) == "-" and v * factor >= 100 and n_key == "num" \
                        and it.group("pct"):
                    raise _Skip("exaggerated decrease would be >= 100%")
                E.add(it.start(s_key), it.end(n_key),
                      _flip_sign_token(it.group(s_key), it.group(n_key), mode, factor))
            pos = it.end()
            nxt = sep.match(text, pos)
            if not nxt:
                break
            pos = nxt.end()
    # 4. other explicitly signed change numbers ("a -0.1% (-0.4 billion vehicle miles) decline")
    for m in re.finditer(rf"(?<![\w.])(?P<sign>[-+])(?P<num>\d[\d,]*(?:\.\d+)?)"
                         rf"(?=\s*(?:%|percent|billion|million))", text):
        if not E.taken(m.start(), m.end()):
            E.add(m.start(), m.end(), _flip_sign_token(m.group("sign"), m.group("num"), mode, factor))
    # 5. level vs a stated baseline: "V% … (above|below) the national baseline of B%"
    for m in re.finditer(rf"\b(?P<cmp>above|below)\s+(?:the\s+)?(?:national\s+)?baseline"
                         rf"(?:\s+level)?\s+of\s+(?P<b>\d+(?:\.\d+)?)%", text, re.I):
        v = None
        for q in re.finditer(r"(?<![\d.])(\d+(?:\.\d+)?)%", text[max(0, m.start() - 400):m.start()]):
            v = q
        if v is None:
            raise _Skip("baseline claim without a stated value")
        V, B = float(v.group(1)), float(m.group("b"))
        V2 = level_mirror(V, B) if flip else B * (V / B) ** factor
        V2r = round(V2, decimals(v.group(1)))
        if V2r <= 0 or (flip and (V2r > B) == (V > B)) or V2r == B:
            raise _Skip("mirrored level does not cross the baseline cleanly" if flip
                        else "exaggerated level rounds onto the baseline")
        old = v.group(1) + "%"
        new = fmt_like(V2r, v.group(1)) + "%"
        for occ in re.finditer(rf"(?<![\d.]){re.escape(old)}", text):
            # every mention of that value (a report may repeat it), never a baseline value
            if re.search(r"baseline(?:\s+level)?\s+of\s*$", text[max(0, occ.start() - 40):occ.start()]):
                continue
            if not E.taken(occ.start(), occ.end()):
                E.add(occ.start(), occ.end(), new)
    # 6. exaggerate: scale magnitudes next to direction words
    if not flip:
        for m in re.finditer(rf"\b(?:increased|decreased|rose|fell|declined|increasing|decreasing|"
                             rf"(?:up|down))\s+(?:by\s+)?(?P<y>{NUM})(?=\s*{UNIT}?\s*(?:to\b|,|\.|and\b|or\b|from\b|"
                             rf"compared|in\b|\(|over\b|$))", text, re.I):
            if not E.taken(m.start("y"), m.end("y")) and not YEAR.match(m.group("y")):
                _scale(text, E, m, "y", factor)
        for m in re.finditer(rf"(?P<y>{NUM})\s*(?:{UNIT})?\s+(?:more|less|higher|lower)\s+than", text, re.I):
            if not E.taken(m.start("y"), m.end("y")):
                _scale(text, E, m, "y", factor)
        for m in re.finditer(rf"\ba\s+(?P<y>{NUM})%\s*(?:\((?P<p>{NUM})\s*billion[^)]*\)\s*)?"
                             rf"(?:increase|decrease|decline|rise)", text, re.I):
            for g in ("y", "p"):
                if m.group(g) and not E.taken(m.start(g), m.end(g)):
                    _scale(text, E, m, g, factor)


def _scale(text: str, E: Edits, m, g: str, factor: float = FACTOR) -> None:
    v = parse_num(m.group(g)) * factor
    after = text[m.end(g):m.end(g) + 3]
    window = text[max(0, m.start() - 30):m.end() + 5].lower()
    if ("%" in after or "percent" in text[m.end(g):m.end(g) + 9]) and v >= 100 and \
            re.search(r"decreas|declin|fell|down|lower|less", window):
        raise _Skip("exaggerated decrease would be >= 100%")
    E.add(m.start(g), m.end(g), fmt_like(v, m.group(g)))


def _fix_articles(text: str) -> str:
    def art(m):
        a, noun = m.group(1), m.group(2)
        want = "an" if noun[0].lower() in "aeiou" else "a"
        return _case_like(want, a) + " " + noun
    return re.sub(r"\b(a|an)\s+(increase|decrease|decline|rise|fall|gain|loss)\b", art, text,
                  flags=re.I)


# --- coherence checks (also used by validate_attacks) ----------------------------------------

def coherence_problems(text: str) -> list[str]:
    """Arithmetic / logical contradictions inside one report. Empty list = coherent."""
    out = []
    for m in re.finditer(rf"\b(?P<dir>up|down)\s+(?:by\s+)?(?P<y>{NUM})\s*(?P<yu>{UNIT})?\s+"
                         rf"from\s+(?P<z>{NUM})\s*(?P<zu>{UNIT})?", text, re.I):
        x = _anchor_before(text, m.start(), unit_class(m.group("zu")))
        if x is None:
            continue
        X, Y, Z = parse_num(x.group("num")), parse_num(m.group("y")), parse_num(m.group("z"))
        tol = 1.01 * 10 ** -min(decimals(m.group("y")), decimals(m.group("z")), decimals(x.group("num")))
        signed = (X - Z) if m.group("dir").lower() == "up" else (Z - X)
        if abs(signed - Y) > tol:
            out.append(f"up/down arithmetic: {x.group('num')} vs {m.group(0)}")
    for m in re.finditer(r"\b(?P<cmp>above|below)\s+(?:the\s+)?(?:national\s+)?baseline"
                         r"(?:\s+level)?\s+of\s+(?P<b>\d+(?:\.\d+)?)%", text, re.I):
        v = None
        for q in re.finditer(r"(?<![\d.])(\d+(?:\.\d+)?)%", text[max(0, m.start() - 400):m.start()]):
            v = q
        if v is not None:
            V, B = float(v.group(1)), float(m.group("b"))
            if (m.group("cmp").lower() == "above") != (V > B) or V == B:
                out.append(f"baseline side: {V}% {m.group('cmp')} {B}%")
    for m in re.finditer(r"chang\w*\s+by\s+(?P<s>[-+]?)(?P<n>\d[\d.]*)%\s*\((?P<ps>[-+]?)(?P<pn>\d[\d.]*)\s*billion",
                         text, re.I):
        if float(m.group("n")) and float(m.group("pn")) and \
                ((m.group("s") == "-") != (m.group("ps") == "-")):
            out.append(f"sign mismatch: {m.group(0)}")
    return out
