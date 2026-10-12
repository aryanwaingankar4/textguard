"""R4 — rule engines for the two held-out text styles: Energy reports and FNSPID headline bundles.

Why a separate module and not more lexicon in ``rules.py``: ``rules.py`` produced P3's and R3's
texts, which are frozen (already reported); any new word there could change them. These engines
reuse its conventions and result type (``EditResult``) but touch only Energy and FNSPID. The P3/R3
principles carry over unchanged: deterministic, minimal edits of the **real** text; every fake
records ``real_direction`` / ``claimed_direction`` (+1/−1 about the forecast target); a text that
cannot be edited cleanly is kept **unchanged** with ``attacked=False`` and the reason — never
emitted self-contradictory.

Energy (one report per week, PROJECT_STATUS §6 "R4 held-out domains")
----------------------------------------------------------------------
The claim about the target is the headline sentence's **week-over-week comparison**:
"… decreased to $3.661 per gallon on September 12, 2011, $0.013 per gallon less than last week
but $0.940 above a year ago." ``real_direction`` = the comparator's sign (more/greater/higher/
above/over = +1); a verb, if present (increased/decreased/rose/fell/declined/dropped), must agree,
else the real report contradicts itself → not attacked.

* **flip:** swap the verb and the comparator only. The current level ($3.661 — it is the latest
  value, also in the numbers), the size of the change and the year-over-year clause stay, so the
  fake is internally consistent (it implies a different last-week price, which the report does not
  state). "for the fifth week in a row" stays: it becomes part of the lie, checkable against the
  numbers. Sentences about other quantities (NYH spot price, inventories) are not edited.
* **exaggerate (A3, ×10):** the change ×10, "sharply" after the verb; **plausibility** (Aryan,
  2026-10-12): the new weekly change must not exceed the largest weekly move of the target before
  the text window (``max_step``), and the implied last-week price must stay > 0.
* **A2:** "The national average retail regular gasoline price is expected to {rise|fall} next week."

FNSPID (a bundle = one week of one stock's headlines, one per line, §6 "R4 FNSPID")
------------------------------------------------------------------------------------------
Headlines are independent statements, often about other companies. A **cue** (a directional word
in context) counts only if it is **about the stock**: the company's name (``ALIASES``) is in the
same clause, or the headline is a Benzinga sector line ("Shares of several technology companies
are trading higher …") that tags this stock. Each target headline's direction = sign of its cues;
the bundle's ``real_direction`` = sign(#up headlines − #down headlines); 0 → not attacked.

* **flip:** every target cue reversed with ``FLIP`` (case kept): trading higher ↔ lower, up ↔ down
  before a number, rises ↔ falls, beats ↔ misses, upgrades ↔ downgrades, raises ↔ lowers a target
  or guidance, 52-week high ↔ low, positive ↔ negative, strong ↔ weak, bullish ↔ bearish …
  Coupled numbers keep the headline coherent: "from $300 to $350" → "from $350 to $300" (targets),
  "$1.23 Beats $1.10" → "$1.10 Misses $1.23", "from Neutral to Buy" → "from Buy to Neutral"; a
  rating change without "from" moves the new rating one notch the other way (``RATING_UP`` /
  ``RATING_DOWN``: "Downgrades Netflix to Neutral" → "Upgrades Netflix to Buy"). Ratings that are
  *levels* ("Maintains Outperform") are not claims about a change and are never edited.
  **A bundle is not attacked if any target headline holds a directional word the engine cannot
  reverse** (``UNHANDLED``), or a quote with one, or a rating it cannot map — headlines in a bundle
  often describe the same event, so one left genuine would contradict the flipped ones.
* **A2:** "{Company} shares are expected to {rise|fall} next week." on a new line.
* No A3 on FNSPID (Aryan, 2026-10-12: only ~2% of headlines state a % move).

**Known limitation (state in the paper).** Lexical and numeric contradictions are prevented here
and scanned in ``consistency_r4``; *semantic* ones are not: a flipped move can keep a cause that no
longer fits it ("… trading higher after the Phase 3 study did not meet its primary endpoint"). The
word list was refined over six by-eye audits of random flips (2026-10-12, before any model
output on FNSPID); each audit's defects were fixed by blocking, never by deleting text.
"""

from __future__ import annotations

import re

from src.attacks.rules import EditResult, Edits, fmt_like, parse_num, sentences

# =============================================================================================
# Energy
# =============================================================================================

ENERGY_SUBJECT = "The national average retail regular gasoline price"
E_CMP = re.compile(
    r"\$(?P<y>\d+(?:\.\d+)?)(?:\s+per gallon)?\s+(?P<cmp>more|less|greater|higher|lower|above|below|over|under)"
    r"(?:\s+than)?\s+(?:last week(?:[’']s)?(?:\s+price)?|the previous week(?:[’']s)?(?:\s+price)?|a week ago)\b",
    re.I)
E_VERB = re.compile(r"\b(increased|decreased|rose|fell|declined|dropped)\b", re.I)
E_LEVEL = re.compile(r"\$(\d+\.\d+)\s+per gallon")
E_UP = {"more", "greater", "higher", "above", "over"}
E_CMP_SWAP = {"more": "less", "less": "more", "greater": "less", "higher": "lower", "lower": "higher",
              "above": "below", "below": "above", "over": "under", "under": "over"}
E_VERB_SWAP = {"increased": "decreased", "decreased": "increased", "rose": "fell", "fell": "rose",
               "declined": "rose", "dropped": "rose"}
E_VERB_DIR = {"increased": 1, "rose": 1, "decreased": -1, "fell": -1, "declined": -1, "dropped": -1}


def _case(new: str, old: str) -> str:
    if old.isupper() and len(old) > 1:
        return new.upper()
    return new[:1].upper() + new[1:] if old[:1].isupper() else new


def energy_headline(text: str | None):
    """(sentence start, sentence end, comparator match, verb match | None, level match | None) of
    the sentence about the national retail price, or None."""
    if not isinstance(text, str) or not text.strip():
        return None
    for a, b in sentences(text):
        s = text[a:b]
        if re.search(r"\bprice\b", s, re.I) and re.search(r"retail|national average", s, re.I):
            cmp_ = E_CMP.search(text, a, b)
            if not cmp_:
                return (a, b, None, None, None)
            verb = E_VERB.search(text, a, cmp_.start())
            level = E_LEVEL.search(text, a, cmp_.start())
            return (a, b, cmp_, verb, level)
    return None


def energy_direction(text: str | None) -> tuple[int, str]:
    """(real direction of the week-over-week claim, reason if 0)."""
    h = energy_headline(text)
    if h is None:
        return 0, "no text" if not (isinstance(text, str) and text.strip()) else "no price sentence"
    _, _, cmp_, verb, _ = h
    if cmp_ is None:
        return 0, "no week-over-week claim"
    if parse_num(cmp_.group("y")) == 0:
        return 0, "no change"
    d = 1 if cmp_.group("cmp").lower() in E_UP else -1
    if verb is not None and E_VERB_DIR[verb.group(1).lower()] != d:
        return 0, "real report's verb contradicts its comparator"
    return d, ""


def energy_edit(text: str | None, mode: str, factor: float = 10.0, max_step: float | None = None) -> EditResult:
    """``mode`` = "flip" or "exaggerate" (A3; ``factor`` ×, ``max_step`` plausibility bound)."""
    d, why = energy_direction(text)
    if d == 0:
        return EditResult(text, False, why, 0, 0, "")
    _, _, cmp_, verb, level = energy_headline(text)
    E = Edits()
    if mode == "flip":
        E.add(cmp_.start("cmp"), cmp_.end("cmp"), _case(E_CMP_SWAP[cmp_.group("cmp").lower()], cmp_.group("cmp")))
        if verb is not None:
            E.add(verb.start(), verb.end(), _case(E_VERB_SWAP[verb.group(1).lower()], verb.group(1)))
        claimed = -d
    elif mode == "exaggerate":
        y = parse_num(cmp_.group("y"))
        y_new = y * factor
        if max_step is not None and y_new > max_step:
            return EditResult(text, False, f"implausible at x{factor:g}: weekly change ${y_new:.3f} > "
                                           f"largest pre-text weekly move ${max_step:.3f}", d, 0, "change")
        if level is not None and parse_num(level.group(1)) - d * y_new <= 0:
            return EditResult(text, False, "implied last-week price <= 0", d, 0, "change")
        E.add(cmp_.start("y"), cmp_.end("y"), fmt_like(y_new, cmp_.group("y")))
        if verb is not None:
            E.add(verb.end(), verb.end(), " sharply")
        claimed = d
    else:
        raise ValueError("mode must be 'flip' or 'exaggerate'")
    out = E.apply(text)
    d2, _ = energy_direction(out)
    if d2 != claimed:
        return EditResult(text, False, "headline check failed", d, 0, "change")
    return EditResult(out, True, "", d, claimed, "change", len(E.spans), sorted(E.spans))


def energy_forward(direction: int) -> str:
    return f"{ENERGY_SUBJECT} is expected to {'rise' if direction > 0 else 'fall'} next week."


# =============================================================================================
# FNSPID
# =============================================================================================

ALIASES = {
    "NFLX": r"Netflix|NFLX", "GOOGL": r"Alphabet|Google|GOOGL?", "MRK": r"Merck(?!\s+KGaA)|MRK",
    "NVDA": r"NVIDIA|Nvidia|NVDA", "PFE": r"Pfizer|PFE", "JNJ": r"Johnson\s*&\s*Johnson|J&J|JNJ",
    "MU": r"Micron|\bMU\b", "FDX": r"FedEx|FDX", "PCG": r"PG&E|PCG", "LLY": r"Eli Lilly|Lilly|LLY",
}
COMPANY = {"NFLX": "Netflix", "GOOGL": "Alphabet", "MRK": "Merck", "NVDA": "NVIDIA", "PFE": "Pfizer",
           "JNJ": "Johnson & Johnson", "MU": "Micron", "FDX": "FedEx", "PCG": "PG&E", "LLY": "Eli Lilly"}
SECTOR_LINE = re.compile(r"^(?:UPDATE:\s*)?Shares of (?:several|some|many|a number of|various)\b|"
                         r"\bcompanies in the [\w\s-]+ (?:space|sector|industry)\b", re.I)
# clause boundaries: a cue belongs to the company named in the same clause
CLAUSE = re.compile(r"[;:]|\s(?:As|After|While|Following|Amid|But|Despite|Ahead Of|On Report|Says|Sees)\s|"
                    r"\swith\s+(?:the\s+)?(?:broader|overall)\s", re.I)
# multi-company roundups: items are comma-separated, each with its own subject
ROUNDUP = re.compile(r"Daily Biotech Pulse|Market Update|Roundup|Week Ahead|Pickup|Barron's Picks|"
                     r"Bulls And Bears|Mid-?Day|Midday|Earnings Preview|Biggest|What's Moving", re.I)
# a subordinate clause about the market / macro has its own subject: never edited (see _target_spans)
FOREIGN = re.compile(r"\b(?:oil|crude|markets?|equities|stocks|indices|index|S&P|Dow|Nasdaq|yields?|treasur\w*|"
                     r"dollar|economy|economic|investors?|futures|rates?|Fed|tariffs?|trade talks|China|"
                     r"coronavirus|COVID\w*|pandemic|vaccine\w*|sector|industry)\b", re.I)

# word -> (opposite, sign of the original). Context rules in ``_cue_ok``.
FLIP = {}
for up, dn in [
        ("higher", "lower"), ("up", "down"), ("rises", "falls"), ("rise", "fall"), ("rose", "fell"),
        ("rising", "falling"), ("gains", "falls"), ("gained", "fell"), ("jumps", "drops"), ("jump", "drop"),
        ("jumped", "dropped"), ("surges", "plunges"), ("surge", "plunge"), ("surged", "plunged"),
        ("surging", "plunging"), ("soars", "plummets"), ("soared", "plummeted"), ("climbs", "slides"),
        ("climb", "slide"), ("climbed", "slid"), ("rallies", "slumps"), ("rally", "slump"), ("rallied", "slumped"),
        ("rebounds", "retreats"), ("rebound", "retreat"), ("rebounded", "retreated"), ("rebounding", "retreating"),
        ("spikes", "tumbles"), ("spike", "tumble"), ("spiked", "tumbled"), ("beats", "misses"), ("beat", "miss"),
        ("tops", "misses"), ("above", "below"), ("upgrades", "downgrades"), ("upgrade", "downgrade"),
        ("upgraded", "downgraded"), ("raises", "lowers"), ("raised", "lowered"),
        ("positive", "negative"), ("bullish", "bearish"), ("strong", "weak"), ("stronger", "weaker"),
        ("strength", "weakness"), ("optimism", "pessimism"), ("optimistic", "pessimistic"),
        ("upbeat", "downbeat"), ("encouraging", "disappointing"), ("risen", "fallen"),
        ("better-than-expected", "worse-than-expected"), ("better", "worse"), ("solid", "weak")]:
    FLIP[up] = (dn, 1)
    FLIP[dn] = (up, -1)
# one-way entries (the pair's other word is already mapped); "lower" stays higher <-> lower
FLIP.update({"cuts": ("raises", -1), "cut": ("raised", -1), "raise": ("cut", 1), "hikes": ("lowers", 1),
             "boosts": ("lowers", 1), "dips": ("rises", -1), "dip": ("rise", -1), "dipped": ("rose", -1),
             "drops": ("jumps", -1), "slips": ("rises", -1),
             "high": ("low", 1), "highs": ("lows", 1), "low": ("high", -1), "lows": ("highs", -1)})
RAISE_WORDS = {"raises", "raise", "raised", "lowers", "lowered", "cuts", "cut", "hikes", "boosts"}
# Directional words the engine does not reverse: a target headline holding one is unflippable.
UNHANDLED = ["disappoint", "disappoints", "disappointment", "pressure", "pressured", "weighs", "weigh",
             "sinks", "sank", "slumping", "tumbling", "plummeting", "soaring", "worsen", "worsens", "improves",
             "improved", "recovers", "recovered", "outpaces", "lags", "crushes", "crushed", "slashes", "slashed",
             "headwinds", "tailwinds", "upside", "downside", "losses", "loses", "lost", "doubles", "halves",
             "boost", "boosted", "hurts", "hurt"]
# Broad net (added after the first by-eye pass, before any model output): any word in a headline
# about the stock that *looks* directional — by stem — and is not a cue the engine reverses in
# this context blocks the bundle. Missing one ("Micron Shares May Have Fallen …" next to a flipped
# cause) leaves the bundle contradicting itself; a false alarm only costs coverage.
BROAD = re.compile(
    r"\b(?:fall\w*|fell|ris(?:e|es|en|ing)|rose|gain\w*|los(?:s|ses|e|es|ing|t)|jump\w*|drop\w*|surg\w*|"
    r"plung\w*|soar\w*|plummet\w*|climb\w*|slid\w*|slip\w*|sink\w*|sank|rall\w*|slump\w*|tumbl\w*|"
    r"spik\w*|dip\w*|rebound\w*|retreat\w*|recover\w*|higher|lower|beat\w*|miss\w*|tops|upgrad\w*|"
    r"downgrad\w*|rais\w*|cuts?|positive|negative|bullish|bearish|strong\w*|weak\w*|optimis\w*|"
    r"pessimis\w*|upbeat|downbeat|disappoint\w*|encourag\w*|pressur\w*|boost\w*|hurt\w*|outpac\w*|"
    r"lag\w*|crush\w*|slash\w*|upside|downside|headwind\w*|tailwind\w*|doubl\w*|halv\w*|worsen\w*|"
    r"improv\w*|declin\w*|decreas\w*|increas\w*|shrink\w*|shrank|"
    r"slow\w*|accelerat\w*|record|concern\w*|fear\w*|worr\w*|hopes?|better|worse|deteriorat\w*|robust|"
    r"soft(?:er|ness)|stellar|dismal|blowout|surprise\w*|momentum|bounce\w*|lift\w*|warn\w*|highs?|lows?|"
    r"sell-?offs?|routs?|crash\w*|bubble|soften\w*|peak\w*|no longer|top\s+(?:[\w-]+\s+){0,3}(?:picks?|ideas?)|"
    r"caution\w*|profit[- ]taking)\b",
    re.I)
# known words whose non-cue use is still directional ("Up 2.5% From Highs") — these block too
STILL_DIRECTIONAL = {"high", "highs", "low", "lows", "above", "below"}
# ("growth", "expand", "contract" are left out: mostly nouns of business news — "Subscriber Growth",
# "a $5B contract" — and not directional on their own.)
RATINGS = ["Strong Buy", "Strong Sell", "Market Perform", "Sector Perform", "Peer Perform", "In-Line",
           "Equal-Weight", "Equal Weight", "Sector Weight", "Market Weight", "Outperform", "Underperform",
           "Overweight", "Underweight", "Accumulate", "Neutral", "Reduce", "Hold", "Buy", "Sell"]
RATING_RE = "|".join(re.escape(r) for r in RATINGS)
# one notch up / down — used when a rating change states only its new rating
RATING_UP = {"Strong Sell": "Sell", "Sell": "Hold", "Reduce": "Hold", "Hold": "Buy", "Neutral": "Buy",
             "Accumulate": "Buy", "Buy": "Strong Buy", "Underperform": "Market Perform",
             "Market Perform": "Outperform", "Sector Perform": "Outperform", "Peer Perform": "Outperform",
             "In-Line": "Outperform", "Underweight": "Equal-Weight", "Equal-Weight": "Overweight",
             "Equal Weight": "Overweight", "Sector Weight": "Overweight", "Market Weight": "Overweight"}
RATING_DOWN = {"Strong Buy": "Buy", "Buy": "Hold", "Accumulate": "Hold", "Hold": "Sell", "Neutral": "Sell",
               "Outperform": "Market Perform", "Market Perform": "Underperform",
               "Sector Perform": "Underperform", "Peer Perform": "Underperform", "In-Line": "Underperform",
               "Overweight": "Equal-Weight", "Equal-Weight": "Underweight", "Equal Weight": "Underweight",
               "Sector Weight": "Underweight", "Market Weight": "Underweight", "Sell": "Strong Sell"}
RATING_RANK = {**{r: 0 for r in ["Strong Sell"]}, **{r: 1 for r in ["Sell", "Underperform", "Underweight", "Reduce"]},
               **{r: 2 for r in ["Hold", "Neutral", "Market Perform", "Sector Perform", "Peer Perform", "In-Line",
                                 "Equal-Weight", "Equal Weight", "Sector Weight", "Market Weight"]},
               **{r: 3 for r in ["Buy", "Outperform", "Overweight", "Accumulate"]}, "Strong Buy": 4}

WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
NUM_AHEAD = re.compile(r"\s+(?:about\s+|nearly\s+|over\s+|more than\s+|roughly\s+)?\$?\d")
TARGETISH = re.compile(r"[^.;]{0,45}?\b(?:price target|PT|target|guidance|outlook|forecast|estimates?|dividend|"
                       r"view|sales|EPS|revenue)\b", re.I)


def _cue_ok(h: str, m: re.Match) -> bool:
    """Is the lexicon word at ``m`` a directional claim in this context?"""
    w = m.group().lower()
    before, after = h[max(0, m.start() - 25):m.start()].lower(), h[m.end():m.end() + 60]
    if w in ("up", "down"):
        return bool(NUM_AHEAD.match(after)) or bool(re.search(r"(?:shares?|stock|trading|moves?|edges?|is|are|now)\s+$", before))
    if w in ("higher", "lower"):      # a move of the stock, or a comparison ("Higher Than Expected")
        return bool(re.search(r"(?:trad(?:ing|es?|ed)|shares?|stock|mov(?:es?|ing|ed)|edg(?:es?|ing|ed)|"
                              r"clos(?:es?|ing|ed)|open(?:s|ing|ed)?|sharply|spik\w+|%)\s+$", before)) \
            or bool(re.match(r"\s+than\b", after, re.I))
    if w in RAISE_WORDS:              # "Raises Price Target …" or "the price target was raised …"
        return bool(TARGETISH.match(after)) or bool(
            re.search(r"\b(?:price target|PT|target|guidance|outlook|forecast|dividend)\s+(?:was|is|has been|were)\s+$",
                      h[max(0, m.start() - 40):m.start()], re.I))
    if w in ("strong", "stronger") and re.match(r"\s+(?:Buy|Sell)\b", after, re.I):
        return False                      # the rating "Strong Buy", not a claim
    if w == "solid" and re.match(r"\s+tumou?rs?\b", after, re.I):
        return False                      # "Advanced Solid Tumors" — a medical term
    if w == "tops":
        return bool(re.match(r"\s+(?:\$|\d|estimates?|expectations|consensus|views?|street)", after, re.I))
    if w in ("above", "below"):
        return bool(re.match(r"\s+(?:the\s+)?(?:(?:analysts?'?|street|wall street|consensus)\s+)?"
                             r"(?:estimates?|consensus|expectations|guidance|views?)", after, re.I))
    if w in ("high", "highs", "low", "lows"):
        return bool(re.search(r"(?:52-week|all-time|record)\s+$", before))
    if w in ("strength",) and re.match(r"\s+(?:of|in)\s+the\s+(?:dollar|market)", after, re.I):
        return False
    return w in FLIP


def _clauses(h: str, ticker: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """(target clauses, foreign clauses) of headline ``h`` as (start, end) offsets.

    * **Roundups** ("The Daily Biotech Pulse: Xeris Jumps …, Lilly's Solid Q1 Beat, …"): split on
      commas too; only items naming the stock are targets — the others belong to other companies.
    * **Main clause about the stock** (it names the company, or it is a Benzinga sector line):
      the main clause and every subordinate clause that has no subject of its own are targets —
      "Netflix Shares Up 5% After Strong Subscriber Growth": the stated cause must flip with the
      claim. A subordinate clause about the market or macro ("… after oil prices fell to historic
      lows", "… with the broader market") is **foreign**: never edited (flipping it would invent a
      false fact about something else), and if it holds a directional word the bundle is not
      attacked (``_unflippable``) — left as is it would contradict the flipped move.
    * Otherwise: only the clauses naming the stock ("Geron Plunges As Johnson & Johnson Unit
      Discontinues …" — the plunge is Geron's).
    """
    if h.rstrip().endswith("?"):
        return [], []                          # a question ("Which Is The Better Stock?") claims nothing
    label = re.match(r"^\s*(?:UPDATE|BREAKING|EXCLUSIVE|REPORT|RUMOR|From Earlier|Earlier)\s*:\s*", h, re.I)
    start = label.end() if label else 0       # "UPDATE:" is a label, not the main clause
    alias = re.compile(ALIASES[ticker])
    roundup = bool(ROUNDUP.search(h))
    sep = re.compile(CLAUSE.pattern + (r"|," if roundup else ""), re.I)
    bounds = [start] + [m.end() for m in sep.finditer(h, start)] + [len(h)]
    spans = [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a and h[a:b].strip()]
    if not spans:
        return [], []
    if roundup:
        return [(a, b) for a, b in spans if alias.search(h, a, b)], []
    if SECTOR_LINE.search(h) or alias.search(h, *spans[0]):
        tgt, foreign = [spans[0]], []
        for a, b in spans[1:]:
            (foreign if FOREIGN.search(h, a, b) and not alias.search(h, a, b) else tgt).append((a, b))
        return tgt, foreign
    return [(a, b) for a, b in spans if alias.search(h, a, b)], []


def _target_spans(h: str, ticker: str) -> list[tuple[int, int]]:
    return _clauses(h, ticker)[0]


def headline_cues(h: str, ticker: str) -> list[re.Match]:
    spans = _target_spans(h, ticker)
    return [m for m in WORD.finditer(h) if any(a <= m.start() < b for a, b in spans)
            and m.group().lower() in FLIP and _cue_ok(h, m)]


def _unflippable(h: str, ticker: str) -> str:
    spans = _target_spans(h, ticker)
    for a, b in spans:
        seg = h[a:b]
        for w in UNHANDLED:
            if re.search(rf"\b{w}\b", seg, re.I):
                return f"unhandled directional word: {w}"
    cues = headline_cues(h, ticker)
    cue_at = {m.start() for m in cues}
    rating_words = {w.lower() for r in RATINGS for w in r.split()}
    for m in cues:                        # "Raises Price Target By 200%" cannot become "Lowers … By 200%"
        if FLIP[m.group().lower()][1] > 0 and m.group().lower() in RAISE_WORDS:
            pct = re.search(r"\bby\s+(\d+(?:\.\d+)?)\s*%", h[m.end():m.end() + 60], re.I)
            if pct and float(pct.group(1)) >= 100:
                return f"a {pct.group(1)}% raise cannot be reversed"
    if cue_at:
        for a, b in _clauses(h, ticker)[1]:
            m = BROAD.search(h, a, b) or next((x for x in WORD.finditer(h, a, b) if x.group().lower() in FLIP
                                               and _cue_ok(h, x)), None)
            if m:
                return f"directional word in a market clause: {m.group().lower()}"
    for a, b in spans:
        for m in BROAD.finditer(h, a, b):
            if m.start() in cue_at or m.group().lower() in rating_words:
                continue
            if (m.group().lower() in FLIP and m.group().lower() not in STILL_DIRECTIONAL
                    and not _cue_ok(h, m)):
                continue                  # a known word used non-directionally here ("Cuts Jobs")
            return f"unhandled directional word: {m.group().lower()}"
    # a quote opens after a space / the start and closes before a space / punctuation / the end, so
    # an apostrophe inside it ("Alphabet's") does not end it
    for q in re.finditer(r"(?:(?<=\s)|^)['\"‘“].{3,}?['\"’”](?=[\s.,;:!?)]|$)", h):
        if any(a <= q.start() < b for a, b in spans) and (
                BROAD.search(h, q.start(), q.end())
                or any(m.group().lower() in FLIP for m in WORD.finditer(h, q.start(), q.end()))):
            return "directional word inside a quote"   # a quote is never edited, so any direction in it blocks
    return ""


def headline_direction(h: str, ticker: str) -> int:
    s = sum(FLIP[m.group().lower()][1] for m in headline_cues(h, ticker))
    return (s > 0) - (s < 0)


def bundle_direction(text: str | None, ticker: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    s = sum(headline_direction(h, ticker) for h in text.split("\n"))
    return (s > 0) - (s < 0)


def _flip_headline(h: str, ticker: str) -> str:
    """Reverse every target cue of one headline, with its coupled numbers / ratings."""
    E = Edits()
    cues = headline_cues(h, ticker)
    for m in cues:
        E.add(m.start(), m.end(), _case(FLIP[m.group().lower()][0], m.group()))
    words = {m.group().lower() for m in cues}
    cue_at = {m.start() for m in cues}
    # coupled numbers: "from $A to $B" next to a target/guidance change; "$A Beats/Misses $B"
    if words & RAISE_WORDS:
        num = r"\$?\d[\d,]*(?:\.\d+)?[BMK]?"                 # a sentence-final "." is not part of it
        val = rf"{num}(?:\s*-\s*{num})?"                     # an amount or a range
        for m in re.finditer(rf"\bfrom\s+({val})\s+to\s+({val})", h, re.I):
            E.add(m.start(1), m.end(1), m.group(2)); E.add(m.start(2), m.end(2), m.group(1))
    amount = r"\$?-?\d(?:[\d,.]*\d)?(?:\s?[BMK](?![A-Za-z]))?"
    for m in re.finditer(rf"({amount})\s+(?P<v>Beats?|Miss(?:es)?|Tops?)\s+({amount})", h, re.I):
        if m.start("v") in cue_at:        # only numbers coupled to a cue about this stock
            E.add(m.start(1), m.end(1), m.group(3)); E.add(m.start(3), m.end(3), m.group(1))
    # ratings: "from A to B" swapped; "… to B" alone moved one notch the other way
    if words & {"upgrades", "upgrade", "upgraded", "downgrades", "downgrade", "downgraded"}:
        up = bool(words & {"upgrades", "upgrade", "upgraded"})
        ft = re.search(rf"\bfrom\s+({RATING_RE})\s+to\s+({RATING_RE})\b", h, re.I)
        if ft:
            E.add(ft.start(1), ft.end(1), ft.group(2)); E.add(ft.start(2), ft.end(2), ft.group(1))
        else:
            to = re.search(rf"\bto\s+({RATING_RE})\b", h, re.I)
            if to:
                key = next(r for r in RATINGS if r.lower() == to.group(1).lower())
                new = (RATING_DOWN if up else RATING_UP).get(key)
                if new is None:
                    raise ValueError(f"no rating one notch {'below' if up else 'above'} {key}")
                E.add(to.start(1), to.end(1), new)
    return E.apply(h)


def fnspid_edit(text: str | None, ticker: str) -> EditResult:
    """Flip a week's headline bundle (see the module docstring)."""
    if not isinstance(text, str) or not text.strip():
        return EditResult(text, False, "no text", 0, 0, "")
    d = bundle_direction(text, ticker)
    if d == 0:
        return EditResult(text, False, "no net direction about the stock", 0, 0, "headlines")
    lines, out, n = text.split("\n"), [], 0
    for h in lines:
        why = _unflippable(h, ticker)     # any headline about the stock, cue or not
        if why:
            return EditResult(text, False, why, d, 0, "headlines")
        if not headline_cues(h, ticker):
            out.append(h)
            continue
        try:
            new = _flip_headline(h, ticker)
        except ValueError as e:
            return EditResult(text, False, f"cannot flip: {e}", d, 0, "headlines")
        n += new != h
        out.append(new)
    fake = "\n".join(out)
    if bundle_direction(fake, ticker) != -d:
        return EditResult(text, False, "headline check failed", d, 0, "headlines")
    return EditResult(fake, True, "", d, -d, "headlines", n)


def fnspid_forward(ticker: str, direction: int) -> str:
    return f"{COMPANY[ticker]} shares are expected to {'rise' if direction > 0 else 'fall'} next week."
