"""R4 — consistency scan of the held-out attack texts (R3 §10f Amendment 1, extended to new styles).

Same rule as R3 (Aryan, 2026-10-11): exclude a text only if (a) its stated numbers / claims
contradict each other, or (b) the attack's own edit moved a stated number against the claimed
direction. As in R3, a problem the **real** text already has is not blamed on the attack — only
problems **new** relative to the real text count — and nothing here reads a model output or the
target series. A flagged (attack, window) is treated as **not attacked** in every R4 metric.

Checks carried over from ``consistency_r3`` (text-only, domain-free): **C6** an edited level moved
against its own verb, **C7** "by Y … to X" implying a start ≤ 0, **C8** "<verb> from A to B" whose
numbers contradict the verb (adopted for R4+, §10f). C1–C5 are specific to the original domains.

New checks for the held-out text styles:

* **E1 energy_verb_vs_comparator** — the price sentence's verb (increased / decreased …) points
  the other way from its week-over-week comparator ("increased … $0.099 per gallon less than last
  week") — the defect the unextended P3 engine produced in 226 / 328 Energy flips.
* **E2 energy_last_week_price** — the level and the stated change imply a last-week price ≤ 0.
* **F2 target_from_to** — "Raises … target … from $A to $B" needs B > A (Lowers / Cuts: B < A).
* **F3 beat_numbers** — "$A Beats / Tops $B" needs A ≥ B; "$A Misses $B" needs A ≤ B.
* **F4 rating_from_to** — "Upgrades … from R1 to R2" needs R2 above R1 on the rating ladder.
* **F5 leftover_direction** — an edited headline still holds a directional word the engine does
  not reverse (it would contradict the flipped words next to it).
* **F6 near_duplicate** — an edited headline and an unedited one in the same bundle are near
  copies (word overlap ≥ 0.6, e.g. the same analyst action reported twice) and the unedited one
  holds a directional word — the bundle would say the same thing both ways.

Run: ``python -m src.attacks.consistency_r4`` (after ``python -m src.attacks.heldout``).
"""

from __future__ import annotations

import re

import pandas as pd

from src.attacks import consistency_r3 as c3
from src.attacks import heldout_rules as hr
from src.attacks.rules import parse_num

KEYS = ["domain", "series", "sample_id"]


def e1_verb_vs_comparator(text: str) -> dict[str, str]:
    _, why = hr.energy_direction(text)
    return {"E1 price sentence": why} if why == "real report's verb contradicts its comparator" else {}


def e2_last_week_price(text: str) -> dict[str, str]:
    h = hr.energy_headline(text)
    if not h or h[2] is None or h[4] is None:
        return {}
    d = 1 if h[2].group("cmp").lower() in hr.E_UP else -1
    x, y = parse_num(h[4].group(1)), parse_num(h[2].group("y"))
    return {"E2 price sentence": f"level ${x} with change {d * y:+} implies last week <= 0"} if x - d * y <= 0 else {}


_UP_T = {"raises", "raised", "hikes", "boosts", "raise"}
_DN_T = {"lowers", "lowered", "cuts", "cut"}


_VAL = r"\$?\d[\d,.]*\d?[BMK]?(?:\s*-\s*\$?\d[\d,.]*\d?[BMK]?)?"


def _mid(tok: str) -> float:
    parts = [p for p in re.split(r"\s*-\s*", tok) if p.strip()]
    return sum(_amount(p) for p in parts) / len(parts)


def f2_target_from_to(h: str) -> dict[str, str]:
    """A raised / lowered target or guidance stated "from A to B" (amounts or ranges — midpoints)."""
    out = {}
    for m in re.finditer(rf"\b(?P<v>raises?|raised|hikes|boosts|lowers|lowered|cuts?)\b[^\n]{{0,60}}?"
                         rf"\b(?:price target|PT|target|guidance|outlook|forecast)\b[^\n]{{0,15}}?"
                         rf"\bfrom\s+(?P<a>{_VAL})\s+to\s+(?P<b>{_VAL})", h, re.I):
        a, b, v = _mid(m.group("a")), _mid(m.group("b")), m.group("v").lower()
        if (v in _UP_T and b <= a) or (v in _DN_T and b >= a):
            out[f"F2 {m.group('v')}"] = f"'{m.group('v')} … from {m.group('a')} to {m.group('b')}'"
    return out


def _amount(tok: str) -> float:
    t = tok.replace("$", "").replace(",", "").strip()
    mult = {"B": 1e9, "M": 1e6, "K": 1e3}.get(t[-1:].upper(), 1)
    return float(t.rstrip("BMKbmk ").strip()) * mult


def f3_beat_numbers(h: str) -> dict[str, str]:
    out = {}
    amount = r"\$?-?\d(?:[\d,.]*\d)?(?:\s?[BMK](?![A-Za-z]))?"
    for k, m in enumerate(re.finditer(rf"({amount})\s+(?P<v>Beats?|Miss(?:es)?|Tops?)\s+({amount})", h, re.I)):
        a, b, v = _amount(m.group(1)), _amount(m.group(3)), m.group("v").lower()
        if (v.startswith(("beat", "top")) and a < b) or (v.startswith("miss") and a > b):
            out[f"F3 #{k}"] = f"'{m.group(0)}'"
    return out


def f4_rating_from_to(h: str) -> dict[str, str]:
    m = re.search(rf"\b(?P<v>upgrades?|upgraded|downgrades?|downgraded)\b[^\n]*?\bfrom\s+(?P<a>{hr.RATING_RE})\s+to\s+(?P<b>{hr.RATING_RE})\b", h, re.I)
    if not m:
        return {}
    rank = {k.lower(): v for k, v in hr.RATING_RANK.items()}
    a, b = rank.get(m.group("a").lower()), rank.get(m.group("b").lower())
    up = m.group("v").lower().startswith("up")
    if a is None or b is None or a == b or (b > a) == up:
        return {}
    return {"F4 rating": f"'{m.group('v')} from {m.group('a')} to {m.group('b')}'"}


def _directional_words(h: str) -> set[str]:
    words = {w.lower() for w in re.findall(r"[A-Za-z]+", h)}
    return (words & set(hr.FLIP)) | (words & set(hr.UNHANDLED))


def headline_problems(fake: str, real: str, ticker: str) -> dict[str, str]:
    """F2–F6 on a bundle (one headline per line), new relative to the real bundle."""
    fl, rl = fake.split("\n"), real.split("\n")
    out = {}
    for i, h in enumerate(fl):
        old = rl[i] if i < len(rl) else ""
        for f in (f2_target_from_to, f3_beat_numbers, f4_rating_from_to):
            new = {k: v for k, v in f(h).items() if k not in f(old)}
            out.update({f"{k} line {i}": v for k, v in new.items()})
        if i < len(rl) and h != old:
            left = hr._unflippable(h, ticker)
            if left and not hr._unflippable(old, ticker):
                out[f"F5 line {i}"] = left
    edited = [i for i in range(min(len(fl), len(rl))) if fl[i] != rl[i]]
    for i in edited:
        wi = set(re.findall(r"[a-z]+", rl[i].lower()))
        for j in range(min(len(fl), len(rl))):
            if j in edited or j == i:
                continue
            wj = set(re.findall(r"[a-z]+", rl[j].lower()))
            if wi and wj and len(wi & wj) / len(wi | wj) >= 0.6 and _directional_words(rl[j]):
                # locator only — the exclusion list is committed, FNSPID text is CC BY-NC (R3 rule: no text)
                out[f"F6 lines {i},{j}"] = f"edited line {i} has an unedited near-copy (line {j}) with a direction word"
    return out


def new_problems(fake: str | None, real: str | None, domain: str, series: str) -> dict[str, str]:
    if not isinstance(fake, str) or not isinstance(real, str) or fake == real:
        return {}
    out = {}
    for f in (c3.c7_p2_implied_start, c3.c8_from_to_direction):
        old = f(real, domain)
        out.update({k: v for k, v in f(fake, domain).items() if k not in old})
    out.update(c3.c6_level_against_direction(fake, real))
    if domain == "Energy":
        for f in (e1_verb_vs_comparator, e2_last_week_price):
            old = f(real)
            out.update({k: v for k, v in f(fake).items() if k not in old})
    else:
        out.update(headline_problems(fake, real, series))
    return out


def scan(tables: dict[str, pd.DataFrame], real: pd.DataFrame) -> pd.DataFrame:
    """One row per (attack, window, new problem)."""
    r = real.set_index(KEYS)["text"]
    rows = []
    for attack, df in tables.items():
        for _, x in df[df.attacked].iterrows():
            k = (x.domain, x.series, x.sample_id)
            for loc, detail in new_problems(x.text, r.loc[k], x.domain, x.series).items():
                rows.append(dict(zip(KEYS, k), attack=attack, check=loc.split()[0], locator=loc, detail=detail))
    return pd.DataFrame(rows, columns=KEYS + ["attack", "check", "locator", "detail"])


def exclusions(tables: dict[str, pd.DataFrame], real: pd.DataFrame) -> pd.DataFrame:
    hits = scan(tables, real)
    if hits.empty:
        return pd.DataFrame(columns=["attack"] + KEYS + ["checks", "detail"])
    return (hits.groupby(["attack"] + KEYS, sort=True)
            .agg(checks=("check", lambda s: ",".join(sorted(set(s)))), detail=("detail", " | ".join))
            .reset_index())
