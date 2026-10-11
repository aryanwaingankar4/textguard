"""R3 Amendment 1 — cross-sentence contradiction scan of the A1 / A2 / A3 texts (PROJECT_STATUS §10f).

Why this exists. §10f requires every attack text to keep its numbers internally consistent, and
``rules.coherence_problems`` checks that — but only **inside one statement** ("X, up Y from Z",
the baseline side, the sign in "changed by S% (T)"). Aryan's by-eye review of the 15 pairs found
contradictions **between** statements that it cannot see: an edited percentage that no longer
matches two absolute values stated elsewhere in the same report (Traffic: week 6 = 14.2B, week 7
= 14.5B, "2.1% lower than the previous week"), an edited change that disagrees with an untouched
restatement of the same change ("+19.0%" next to "a 1.50% change"), or a recomputed month that
makes a stated 3-month average impossible. This module finds those, automatically, in every
val + test text of the three attacks. A flagged text is **excluded from the attacked set** in the
R3 analysis (``results/r3_text_exclusions.csv``) — the criterion is text validity only, no model
output is read. The texts and the Kaggle answers are not changed (no re-run): a flagged window is
simply dropped from every attacked-window metric, exactly as if the rules had left it unattacked.

What counts as a contradiction. Each check below returns problems keyed by a **locator** that
survives the edit (check name + which week / sentence), so a problem the *real* report already
has (Time-MMD's summaries are LLM-written and occasionally inconsistent) is not blamed on the
attack: a text is flagged only for problems **new** relative to its real report. Tolerances are
pure rounding (half a unit of each printed digit, interval arithmetic), except the NSA-vs-SA
check, whose tolerance is the largest gap in any genuine report (data-driven, not a choice).

* **C1 week_sequence** — a stated change vs the previous period, against the two absolute
  levels the same report states for those periods (Traffic weekly VMT; Health_US weekly ILI).
* **C2 pct_vs_level** — "S% (T billion …)" next to the level V it refers to: S must equal
  T / (V − T) (Traffic). A flip that keeps V and T but reverses the sign breaks this when the
  change is large (second-order term), as does a ×10 that scales S and T but not V.
* **C3 nsa_vs_sa** — Traffic's unadjusted and seasonally adjusted year-over-year change of the
  same month (two statements of nearly the same quantity): a gap wider than any genuine report
  shows, or opposite signs.
* **C4 economy_avg3** — "the average deficit … to $V for the three months ending in M" with the
  current month X and previous month Z stated: the implied third month 3V − X − Z must be a
  plausible deficit — inside [0.5·min, 2·max] of the deficits stated in genuine reports (the
  §10f plausibility convention).
* **C5 economy_components** — Δdeficit = Δimports − Δexports for the same month, compared with
  the real report's own rounding gap carried through the edit (×10 under A3, reversed by a flip).
* **C6 level_against_direction** — the attack's **own edit** moved a level stated as
  "<verb> … to V" in the direction opposite to the claim ("decreased sharply to 5.3%" where the
  real report said "decreased to 2.4%"): the 3 Health_US A3 texts flagged in §8 "R3" are this
  case. Uses only the fake and its real report (i.e. the edit) — **never the lookback series**
  (Amendment 1 rule, Aryan 2026-10-11): a fake that merely disagrees with the real series is the
  lie itself, which TextGuard exists to catch; excluding it would bias R3. A flip that changes
  only the verb and keeps the genuine level is not flagged.
* **C7 p2_implied_start** — §10f's own A3 rule P2 ("by Y … to X" must imply a start > 0),
  applied as written: the generator's implementation missed verbs followed by the "sharply" it
  had just inserted, "-ing" verbs, and the same statement without "by" ("increased $66.0
  billion to $48.9 billion") — see the comment above ``c7_p2_implied_start``.

Run: ``python -m src.attacks.consistency_r3`` → prints counts per attack × domain × split and
writes ``results/r3_text_exclusions.csv`` (in git; locators and numbers only, no report text).
"""

from __future__ import annotations

import itertools
import re
import sys
from functools import lru_cache

import pandas as pd

from src.attacks import common
from src.attacks.rules import NUM, UNIT, decimals, parse_num, sentences, unit_class

OUT = common.ROOT / "results" / "r3_text_exclusions.csv"
KEYS = ["domain", "split", "sample_id"]
ATTACKS = ("a1_selective", "a2_forward", "a3_magnitude")
REFERENCE = ("flipped", "exaggerated")   # P3's texts: scanned for disclosure only (P5/P6 frozen)

UPW = r"increased|increase|increasing|increases|rose|rise|rising|up|higher|more|gain|growth"
DOWNW = r"decreased|decrease|decreasing|decreases|fell|fall|falling|declined|decline|declining|down|lower|less|loss"
DIRW = rf"(?:{UPW}|{DOWNW})"


def _dir(word: str | None) -> int:
    if not word:
        return 0
    w = word.lower()
    return 1 if re.fullmatch(UPW, w) else (-1 if re.fullmatch(DOWNW, w) else 0)


def _half(tok: str) -> float:
    """Half a unit of the last printed digit — the rounding uncertainty of a stated number."""
    return 0.5 * 10 ** -decimals(tok)


def _interval(f, *args) -> tuple[float, float]:
    """Range of ``f`` over the corners of the rounding box of each (value, half-width) arg."""
    vals = [f(*c) for c in itertools.product(*[(v - h, v + h) for v, h in args])]
    return min(vals), max(vals)


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] + 1e-12 and b[0] <= a[1] + 1e-12


def _lines(text: str) -> list[str]:
    return [ln for ln in re.split(r"\n+", text) if ln.strip()]


# --- levels (absolute amounts, not changes) ----------------------------------------------------

BVM_LEVEL = re.compile(rf"(?:estimated(?:\s+to\s+be|\s+at)?|\bis|\bare|\bwas|\bat)\s+"
                       rf"(?P<v>{NUM})\s*billion\b(?!\s+(?:\w+[- ])?\w*\s*(?:more|less|higher|lower))", re.I)


def _bvm_levels(s: str) -> list[re.Match]:
    """Traffic levels ("estimated to be 14.2 billion vehicle miles", "is 263.0 billion, a …"),
    never a change in parentheses ("(+22.6 billion vehicle miles)") or a difference ("is 1.1
    billion vehicle miles higher")."""
    return [m for m in BVM_LEVEL.finditer(s) if s[:m.start()].count("(") <= s[:m.start()].count(")")]


# --- C1: a stated week-over-week change vs the two stated weekly levels -----------------------

PREV_WEEK = re.compile(
    rf"(?:(?P<d1>{DIRW})\s+(?:sharply\s+)?(?:(?:by|of)\s+)?)?(?P<p>\d+(?:\.\d+)?)\s?%\s*"
    rf"(?:\((?P<abs>[^)]*)\)\s*)?(?P<d2>higher|lower|more|less|increase|decrease)?\s*"
    rf"(?:than|compared\s+(?:to|with)|from|over)\s+the\s+previous\s+week", re.I)
HEALTH_LEVEL = re.compile(r"(?<![\d.])(?P<v>\d+(?:\.\d+)?)%(?!\s*\))")
HEALTH_CHANGE = re.compile(rf"\b(?P<d>increased|decreased|rose|fell|declined)\b"
                           rf"(?:\s+(?:sharply|slightly|marginally|modestly|somewhat))?\s+(?:from\s+[\d.]+%\s+)?to\s+(?P<v>\d+(?:\.\d+)?)%", re.I)


def _week_records(text: str, domain: str) -> dict[int, dict]:
    """week number → {level token, stated pct token + sign, or direction only}, one per line."""
    out = {}
    for ln in _lines(text):
        wk = re.search(r"\b[Ww]eek\s+(\d{1,2})\b", ln)
        if not wk:
            continue
        n, rec = int(wk.group(1)), {}
        if domain == "Traffic":
            lv = _bvm_levels(ln)
            if len(lv) == 1:
                rec["level"] = lv[0].group("v")
            pw = PREV_WEEK.search(ln)
            if pw:
                d = _dir(pw.group("d1")) or _dir(pw.group("d2"))
                if d:
                    rec["pct"], rec["dir"] = pw.group("p"), d
        elif domain == "Health_US":
            # ILI level of that week: the first percentage not stated as a baseline
            for m in HEALTH_LEVEL.finditer(ln):
                if not re.search(r"baseline(?:\s+level)?\s+of\s*$", ln[max(0, m.start() - 30):m.start()]):
                    rec["level"] = m.group("v")
                    break
            ch = HEALTH_CHANGE.search(ln)
            if ch:
                rec["dir"] = _dir(ch.group("d"))
        if rec and n not in out:
            out[n] = rec
    return out


def c1_week_sequence(text: str, domain: str) -> dict[str, str]:
    recs, out = _week_records(text, domain), {}
    for n, cur in recs.items():
        prev = recs.get(n - 1)
        if not prev or "level" not in cur or "level" not in prev or "dir" not in cur:
            continue
        lc, lp = cur["level"], prev["level"]
        impl = _interval(lambda a, b: 100 * (a - b) / b, (parse_num(lc), _half(lc)), (parse_num(lp), _half(lp)))
        if "pct" in cur:
            p = parse_num(cur["pct"]) * cur["dir"]
            hp = _half(cur["pct"])
            stated = (p - hp, p + hp)
        else:   # direction only: the stated sign must be possible
            stated = (0, float("inf")) if cur["dir"] > 0 else (-float("inf"), 0)
        if not _overlap(stated, impl):
            say = f"{'+' if cur['dir'] > 0 else '-'}{cur.get('pct', '')}{'%' if 'pct' in cur else ''}"
            out[f"C1 week {n}"] = (f"week {n} stated {say} vs previous week, levels {lp} -> {lc} "
                                   f"imply {impl[0]:+.2f}..{impl[1]:+.2f}%")
    return out


# --- C2: "S% (T billion …)" vs the level V it refers to ----------------------------------------

PCT_ABS = re.compile(
    rf"(?P<sign>[-+]?)(?P<p>\d+(?:\.\d+)?)\s?%\s*(?P<mid>increase|decrease|decline|rise)?\s*"
    rf"\(\s*(?P<tsign>[-+]?)(?P<t>\d+(?:\.\d+)?)\s*billion"
    rf"(?:\s+vehicle[- ]miles|\s+miles)?\s*\)\s*(?P<after>\w+)?", re.I)


def c2_pct_vs_level(text: str, domain: str) -> dict[str, str]:
    if domain != "Traffic":
        return {}
    out = {}
    for li, ln in enumerate(_lines(text)):
        sents = [ln[a:b] for a, b in sentences(ln)]
        for si, s in enumerate(sents):
            # the level must be in the same sentence: a "cumulative" or seasonally adjusted
            # sentence without its own level is about a different quantity
            levels = _bvm_levels(s)
            if len(levels) != 1:
                continue
            V = levels[0].group("v")
            for k, m in enumerate(PCT_ABS.finditer(s)):
                if m.group("sign") or m.group("tsign"):
                    d = -1 if "-" in (m.group("sign"), m.group("tsign")) else 1
                else:   # unsigned: the direction word around it ("decrease of 25% (4.2 …)",
                    before = re.findall(DIRW, s[max(0, m.start() - 40):m.start()], re.I)
                    d = (_dir(m.group("mid")) or _dir(m.group("after"))
                         or (_dir(before[-1]) if before else 1))
                P, T = parse_num(m.group("p")) * d, parse_num(m.group("t")) * d
                impl = _interval(lambda v, t: 100 * t / (v - t) if v - t > 0 else float("nan"),
                                 (parse_num(V), _half(V)), (T, _half(m.group("t"))))
                hp = _half(m.group("p"))
                if any(x != x for x in impl) or not _overlap((P - hp, P + hp), impl):
                    out[f"C2 line {li} sent {si} #{k}"] = (f"{P:+g}% ({T:+g} billion) with level {V} "
                                                          f"implies {impl[0]:+.2f}..{impl[1]:+.2f}%")
    return out


# --- C3: Traffic NSA vs SA year-over-year change of the same month -----------------------------

NSA = re.compile(rf"(?:chang\w*\s+by|(?P<d>increased|decreased|rose|fell|declined)\s+(?:sharply\s+)?by)\s+"
                 rf"(?P<sign>[-+]?)(?P<p>\d+(?:\.\d+)?)\s?%", re.I)
SA = re.compile(rf"\ban?\s+(?P<sign>[-+]?)(?P<p>\d+(?:\.\d+)?)\s?%\s*(?:\([^)]*\)\s*)?"
                rf"(?P<w>increase|decrease|decline|rise|change|gain|drop)", re.I)


def _nsa_sa(text: str) -> tuple[float, float, str, str] | None:
    nsa = sa = None
    for a, b in sentences(text):
        s = text[a:b]
        if re.search(r"seasonally[- ]adjusted", s, re.I):
            m = SA.search(s)
            if m and sa is None:
                d = -1 if m.group("sign") == "-" else (_dir(m.group("w")) or 1)
                sa = (parse_num(m.group("p")) * d, m.group("p"))
        elif re.search(r"all roads and streets|nationwide|\bVMT\b|vehicle miles traveled", s, re.I) \
                and not re.search(r"\bweek\b|interstate|[Cc]umulative", s):
            m = NSA.search(s)
            if m and nsa is None:
                d = -1 if (m.group("sign") == "-" or _dir(m.group("d")) < 0) else 1
                nsa = (parse_num(m.group("p")) * d, m.group("p"))
    if nsa is None or sa is None:
        return None
    return nsa[0], sa[0], nsa[1], sa[1]


@lru_cache(maxsize=None)
def nsa_sa_tolerance() -> float:
    """Largest |NSA − SA| gap (pp) in any genuine val/test Traffic report: the C3 tolerance."""
    real = common.load("real")
    gaps = [abs(x[0] - x[1]) for t in real[real.domain == "Traffic"].text.dropna()
            if (x := _nsa_sa(t)) is not None]
    return max(gaps)


def c3_nsa_vs_sa(text: str, domain: str) -> dict[str, str]:
    if domain != "Traffic":
        return {}
    x = _nsa_sa(text)
    if x is None:
        return {}
    nsa, sa, tn, ts = x
    gap = abs(nsa - sa) - _half(tn) - _half(ts)
    if gap > nsa_sa_tolerance() or (nsa * sa < 0 and min(abs(nsa), abs(sa)) > 0.5):
        return {"C3 nsa/sa": f"unadjusted {nsa:+g}% vs seasonally adjusted {sa:+g}% "
                             f"(largest genuine gap {nsa_sa_tolerance():.2f} pp)"}
    return {}


# --- Economy: deficit statements ---------------------------------------------------------------

DEF_WAS = re.compile(rf"deficit\s+was\s+(?P<x>{NUM})\s*billion[^.]*?\b(?P<d>up|down)\s+(?:by\s+)?"
                     rf"(?P<y>{NUM})\s*billion\s+from\s+(?P<z>\${NUM})", re.I)   # "$Z": not "from 2020"
DEF_BY = re.compile(rf"deficit\s+(?P<d>increased|decreased|rose|fell|declined)\s+(?:sharply\s+)?(?:by\s+)?"
                    rf"(?P<y>{NUM})\s*billion\s+to\s+(?P<x>{NUM})", re.I)
AVG3 = re.compile(rf"average[^.]*?deficit\s+(?:\w+\s+){{0,2}}?(?:(?P<d>increased|decreased|rose|fell|"
                  rf"declined)\s+(?:sharply\s+)?(?:by\s+)?(?P<y>{NUM})\s*billion\s+)?to\s+(?P<v>{NUM})\s*billion"
                  rf"[^.]*?three months", re.I)
# a clause about exports / imports runs to the next exports / imports mention or sentence end
# (a decimal point is not a sentence end: "$188.5 billion")
COMP = re.compile(rf"\b(?P<what>exports|imports)\b(?P<body>(?:\.(?=\d)|[^.;\n])*?)"
                  rf"(?=\bexports\b|\bimports\b|\.(?!\d)|[;\n]|$)", re.I)
COMP_CHANGE = re.compile(rf"(?P<y>{NUM})\s*billion\s+(?P<d2>more|less)\s+than|"
                         rf"\b(?P<d1>increased|decreased|increasing|decreasing|rose|fell|declined|up|down)\s+"
                         rf"(?:sharply\s+)?(?:by\s+)?(?P<y1>{NUM})\s*billion", re.I)


def _deficit_statements(text: str) -> list[dict]:
    """Every "deficit was X, up/down Y from Z" / "deficit increased by Y to X", in text order."""
    out = []
    for m in DEF_WAS.finditer(text):
        out.append(dict(pos=m.start(), end=m.end(), x=m.group("x"), y=m.group("y"),
                        z=m.group("z"), d=_dir(m.group("d")),
                        gs="goods and services" in text[max(0, m.start() - 40):m.start()].lower()))
    for m in DEF_BY.finditer(text):
        X, Y, d = parse_num(m.group("x")), parse_num(m.group("y")), _dir(m.group("d"))
        out.append(dict(pos=m.start(), end=m.end(), x=m.group("x"), y=m.group("y"),
                        z=f"{X - d * Y:.{decimals(m.group('x'))}f}", d=d,
                        gs="goods and services" in text[max(0, m.start() - 40):m.start()].lower()))
    return sorted(out, key=lambda r: r["pos"])


@lru_cache(maxsize=None)
def deficit_range() -> tuple[float, float]:
    """[0.5·min, 2·max] of every monthly goods-and-services deficit (current and previous) stated
    in a genuine report — the §10f plausibility convention, applied to what a reader can see."""
    real = common.load("real")
    v = [parse_num(s[k]) for t in real[real.domain == "Economy"].text.dropna()
         for s in _deficit_statements(t) if s["gs"] for k in ("x", "z")]
    return 0.5 * min(v), 2 * max(v)


def c4_economy_avg3(text: str, domain: str) -> dict[str, str]:
    if domain != "Economy":
        return {}
    out, stmts = {}, _deficit_statements(text)
    lo, hi = deficit_range()
    for k, m in enumerate(AVG3.finditer(text)):
        before = [s for s in stmts if s["end"] <= m.start() and s["gs"]]   # the average is of goods AND services
        if not before:
            continue
        s = before[-1]
        third = 3 * parse_num(m.group("v")) - parse_num(s["x"]) - parse_num(s["z"])
        if not lo <= third <= hi:
            out[f"C4 avg3 #{k}"] = (f"3-month average {m.group('v')} with months {s['x']} and {s['z']} "
                                    f"implies a third month of {third:.1f} (genuine range {lo:.1f}..{hi:.1f})")
    return out


def _components(text: str) -> list[tuple[float, float, float] | None]:
    """Per deficit statement: (Δdeficit, Δimports − Δexports, rounding tolerance), or None when
    the statement's exports and imports changes are not both stated."""
    out, stmts = [], _deficit_statements(text)
    for k, s in enumerate(stmts):
        end = stmts[k + 1]["pos"] if k + 1 < len(stmts) else len(text)
        seg = text[s["end"]:end]
        seg = seg[:m.start()] if (m := re.search(r"average|three months", seg, re.I)) else seg
        ch = {}
        for c in COMP.finditer(seg):
            cm = COMP_CHANGE.search(c.group("body"))
            if cm and c.group("what").lower() not in ch:
                y = cm.group("y") or cm.group("y1")
                d = 1 if (cm.group("d2") or cm.group("d1")).lower() in ("more", "increased", "increasing",
                                                                         "rose", "up") else -1
                ch[c.group("what").lower()] = (parse_num(y) * d, y)
        if len(ch) < 2:
            out.append(None)
            continue
        (E, te), (I, ti) = ch["exports"], ch["imports"]
        out.append((parse_num(s["y"]) * s["d"], I - E, _half(s["y"]) + _half(te) + _half(ti)))
    return out


def c5_economy_components(fake: str, real: str) -> dict[str, str]:
    """Δdeficit = Δimports − Δexports, **relative to the real report**: the real report's own
    rounding gap (e.g. 3.2 vs 3.1) is carried through the edit — ×10 under A3 (32 vs 31), sign
    reversed under a flip — so only an edit that breaks the identity (one component edited, another
    not) is flagged. Paired by statement index, like C6."""
    f, r = _components(fake), _components(real)
    if len(f) != len(r):
        return {}
    out = {}
    for k, (cf, cr_) in enumerate(zip(f, r)):
        if cf is None or cr_ is None or cr_[0] == 0:
            continue
        scale = cf[0] / cr_[0]                     # 10 under A3, −1 under a flip, 1 if untouched
        gap_f, gap_r = cf[0] - cf[1], cr_[0] - cr_[1]
        if abs(gap_f - scale * gap_r) > cf[2] + abs(scale) * cr_[2] + 1e-9:
            out[f"C5 deficit #{k}"] = (f"deficit change {cf[0]:+g} vs imports - exports = {cf[1]:+g} "
                                       f"(real: {cr_[0]:+g} vs {cr_[1]:+g})")
    return out


# --- C6: an edited level moved against its own sentence's verb ----------------------------------

VERB_TO = re.compile(rf"\b(?P<d>increased|decreased|rose|fell|declined|increasing|decreasing)\b"
                     rf"(?:\s+(?:sharply|slightly|marginally|modestly|somewhat))?(?:\s+by\s+{NUM}\s*(?:%|percent|percentage points?|billion|million)?)?"
                     rf"\s+to\s+(?P<v>{NUM})", re.I)


def c6_level_against_direction(fake: str, real: str) -> dict[str, str]:
    fs, rs = [fake[a:b] for a, b in sentences(fake)], [real[a:b] for a, b in sentences(real)]
    if len(fs) != len(rs):
        return {}
    out = {}
    for i, (f, r) in enumerate(zip(fs, rs)):
        fm, rm = list(VERB_TO.finditer(f)), list(VERB_TO.finditer(r))
        if len(fm) != len(rm):
            continue
        for k, (a, b) in enumerate(zip(fm, rm)):
            vf, vr, d = parse_num(a.group("v")), parse_num(b.group("v")), _dir(a.group("d"))
            if vf != vr and (vf - vr) * d < 0:
                out[f"C6 sent {i} #{k}"] = (f"'{a.group('d')} ... to {a.group('v')}' but the edit moved the "
                                            f"level {'down' if vf < vr else 'up'} from the real {b.group('v')}")
    return out


# --- C7: §10f's own P2 rule, as written -------------------------------------------------------
# §10f: 'every "… by Y … to X" statement (same unit) implies a positive starting value X ∓ Y > 0'.
# ``adaptive.implied_starts`` only recognised a verb *immediately* before "by", so it missed the
# headline sentences where A3 itself had inserted "sharply" ("increased sharply by $64.0 billion
# to $42.6 billion" → a start of −21.4) and the -ing forms ("exports decreasing by …"). The
# generator is left as it ran (its output is what the Kaggle runs answered); C7 applies the rule.

UP_V = r"increased|increasing|rose|rising|up|gained|grew|climbed|higher"
DOWN_V = r"decreased|decreasing|fell|falling|declined|declining|down|dropped|lower"
# "<verb> [adverb] [by] Y to X" — "by" optional: "the average deficit increased $66.0 billion to
# $48.9 billion" is the same statement as "increased by $66.0 billion to …"
VERB_BY_TO = re.compile(rf"\b(?P<v>{UP_V}|{DOWN_V})(?:\s+(?:sharply|slightly|marginally|modestly|somewhat))?"
                        rf"\s+(?:by\s+)?(?P<y>[-+]?{NUM})\s*(?P<yu>{UNIT})?\s+to\s+(?P<x>{NUM})\s*(?P<xu>{UNIT})?",
                        re.I)


def c7_p2_implied_start(text: str, domain: str) -> dict[str, str]:
    out = {}
    for k, m in enumerate(VERB_BY_TO.finditer(text)):
        cy, cx = unit_class(m.group("yu")), unit_class(m.group("xu"))
        if not (cy == cx or (cy == "pp" and cx == "percent")):
            continue
        y, x = parse_num(m.group("y").lstrip("+-")), parse_num(m.group("x"))
        start = x - y if re.fullmatch(UP_V, m.group("v"), re.I) else x + y
        if start <= 0:
            out[f"C7 by-to #{k}"] = (f"'{m.group('v')} ... {m.group('y')} ... to {m.group('x')}' implies a "
                                     f"start of {start:g} <= 0 (10f P2)")
    return out


# --- C8 — found after R3 metrics existed; NOT adopted (Aryan, 2026-10-11): sensitivity only --
# "<verb> from A to B" whose own numbers contradict the verb ("increased from 3.0% to 0.9%").
# P3's flip recomputes A from the *old* B while the Health_US level mirror moves B, so both edits
# together can reverse the stated movement. Rule (a): the text's stated numbers contradict each other.

FROM_TO = re.compile(rf"\b(?P<v>{UP_V}|{DOWN_V})(?:\s+(?:sharply|slightly|marginally|modestly|somewhat))?"
                     rf"\s+from\s+(?P<a>{NUM})\s*(?P<au>{UNIT})?\s+to\s+(?P<b>{NUM})", re.I)


def c8_from_to_direction(text: str, domain: str) -> dict[str, str]:
    out = {}
    for k, m in enumerate(FROM_TO.finditer(text)):
        a, b = parse_num(m.group("a")), parse_num(m.group("b"))
        d = 1 if re.fullmatch(UP_V, m.group("v"), re.I) else -1
        if (b - a) * d < 0:
            out[f"C8 from-to #{k}"] = f"'{m.group('v')} from {m.group('a')} to {m.group('b')}'"
    return out


SINGLE = (c1_week_sequence, c2_pct_vs_level, c3_nsa_vs_sa, c4_economy_avg3, c7_p2_implied_start)
NOT_ADOPTED = (c8_from_to_direction,)   # R3 sensitivity only; R4+ scans include it


def problems(text: str | None, domain: str) -> dict[str, str]:
    """All single-text contradictions (C1–C4, C7), keyed by an edit-stable locator."""
    if not isinstance(text, str) or not text.strip():
        return {}
    out = {}
    for f in SINGLE:
        out.update(f(text, domain))
    return out


def new_problems(fake: str | None, real: str | None, domain: str) -> dict[str, str]:
    """Contradictions the fake has and its real report does not (C1–C4, C7), plus the paired C5, C6."""
    if not isinstance(fake, str) or not isinstance(real, str) or fake == real:
        return {}
    old = problems(real, domain)
    out = {k: v for k, v in problems(fake, domain).items() if k not in old}
    if domain == "Economy":
        out.update(c5_economy_components(fake, real))
    out.update(c6_level_against_direction(fake, real))
    return out


def scan(condition: str) -> pd.DataFrame:
    """One row per (attacked text, new problem) of ``condition`` (val + test)."""
    df, real = common.load(condition).set_index(KEYS), common.load("real").set_index(KEYS)
    rows = []
    for k, r in df[df.attacked].iterrows():
        for loc, detail in new_problems(r.text, real.loc[k, "text"], k[0]).items():
            rows.append(dict(zip(KEYS, k), attack=condition, check=loc.split()[0], locator=loc,
                             detail=detail))
    return pd.DataFrame(rows, columns=KEYS + ["attack", "check", "locator", "detail"])


def exclusions() -> pd.DataFrame:
    """The R3 exclusion list: one row per flagged (attack, window), checks joined."""
    hits = pd.concat([scan(c) for c in ATTACKS], ignore_index=True)
    return (hits.groupby(["attack"] + KEYS, sort=True)
            .agg(checks=("check", lambda s: ",".join(sorted(set(s)))),
                 detail=("detail", " | ".join)).reset_index())


def counts(excl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in ATTACKS:
        df = common.load(c)
        for (dom, split), g in df[df.attacked].groupby(["domain", "split"]):
            n = int(((excl.attack == c) & (excl.domain == dom) & (excl.split == split)).sum())
            rows.append(dict(attack=c, domain=dom, split=split, attacked=len(g), flagged=n,
                             kept=len(g) - n))
    return pd.DataFrame(rows)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    real = common.load("real")
    fp = {d: sum(bool(problems(t, d)) for t in g.text.dropna()) for d, g in real.groupby("domain")}
    print("genuine reports with a C1–C4/C7 problem of their own (not blamed on any attack):", fp)
    print(f"C3 tolerance (largest genuine NSA–SA gap): {nsa_sa_tolerance():.2f} pp; "
          f"C4 genuine deficit range: {deficit_range()[0]:.1f}..{deficit_range()[1]:.1f}")
    excl = exclusions()
    excl.to_csv(OUT, index=False)
    c = counts(excl)
    print("\nflagged / attacked, per attack × domain × split:\n" + c.to_string(index=False))
    print("\nby check:\n" + excl.assign(c=excl.checks.str.split(",")).explode("c")
          .groupby(["attack", "c"]).size().to_string())
    for ref in REFERENCE:
        s = scan(ref)
        print(f"\n(disclosure) P3 {ref}: {s.drop_duplicates(KEYS).shape[0]} attacked texts would be flagged "
              f"(test {s[s.split == 'test'].drop_duplicates(KEYS).shape[0]}) — P5/P6 are frozen, not changed")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
