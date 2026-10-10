"""Validate R3's adaptive attacks before any GPU run (PROJECT_STATUS §10f, "Validation").

Every check is automatic and **all must pass** — a failure stops R3 before a single forecast is
made (``main`` exits with status 1). As in P3 (``validate_attacks``):

* **direction correct** — A1: claimed = −real, the rules read the flip, and the selection is
  exactly "agrees with TextGuard's frozen evidence" (every selected window agrees, every
  attackable-but-unselected one does not); A2: the appended claim parses back to
  ``claimed_direction`` = −real, the real report is an exact prefix and still the rules'
  headline; A3: the headline keeps the real direction;
* **no new contradictions** — ``rules.coherence_problems`` on the fake ⊆ on the real report;
* **plausible values** — A3's filter (P1–P4) holds on every attacked text; A1 is P3's flip
  (its own checks re-run here); A2 adds no number;
* **bookkeeping** — unattacked rows carry the real text unchanged, ``synthetic`` = ``attacked``,
  ``claimed_direction`` ≠ 0 exactly on attacked rows, same windows as ``texts/real.csv``.

Reported: coverage per attack × domain × split (with reasons), length ratio fake / real, and for
A3 how far the edited numbers moved (median |new| / |old|).
Writes ``results/summary_r3_attack_validation.csv`` (in git) and ``texts/r3_review_pairs.md``
(gitignored: raw report text) — **15 before/after pairs** (5 per attack, seed 0) for review by eye.

Run after ``python -m src.attacks.adaptive``:  ``python -m src.attacks.validate_r3``
"""

from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

from src.attacks import adaptive as ad
from src.attacks import common
from src.attacks.rules import NUM, coherence_problems, headline, parse_num
from src.attacks.validate_attacks import clean, diff

OUT = common.ROOT / "results" / "summary_r3_attack_validation.csv"
PAIRS = common.TEXTS_DIR / "r3_review_pairs.md"
KEYS = ["domain", "split", "sample_id"]


def _load(cond: str) -> pd.DataFrame:
    return common.load(cond).set_index(KEYS)


def _new_contradictions(fake: str, real: str) -> set[str]:
    return set(coherence_problems(fake)) - set(coherence_problems(real))


def check_a1(df: pd.DataFrame, real: pd.DataFrame, flip: pd.DataFrame) -> dict[str, pd.Series]:
    win, lb = ad.frozen_windows(), ad.lookbacks()
    a = df.attacked
    agrees = pd.Series([ad.a1_agrees(lb[k], flip.loc[k, "text"], int(flip.loc[k, "claimed_direction"]),
                                     k[0], win[k[0]]) if flip.loc[k, "attacked"] else False
                        for k in df.index], index=df.index)
    return {
        "text_is_p3_flip_or_real": pd.Series(np.where(a, df.text == flip.text, df.text == real.text), df.index),
        "claimed_is_minus_real": ~a | ((df.claimed_direction == -df.real_direction) & (df.claimed_direction != 0)),
        "rules_read_the_flip": ~a | pd.Series([headline(t, k[0]).series_direction == c
                                               for k, t, c in zip(df.index, df.text, df.claimed_direction)], df.index),
        "selection_matches_evidence": a == (flip.attacked & agrees),
        "no_new_contradiction": ~a | pd.Series([not _new_contradictions(t, real.loc[k, "text"])
                                                for k, t in zip(df.index, df.text)], df.index),
    }


def check_a2(df: pd.DataFrame, real: pd.DataFrame) -> dict[str, pd.Series]:
    a = df.attacked
    rt = real.text.reindex(df.index)
    prefix, parsed, head_same, coh = [], [], [], []
    for k, t, c, r in zip(df.index, df.text, df.claimed_direction, rt):
        base = r.rstrip()
        ok_prefix = t.startswith(base)
        prefix.append(ok_prefix)
        tail = t[len(base):].lstrip(". ") if ok_prefix else ""
        parsed.append(ad.read_forward(tail) == c != 0)
        hr, hf = headline(r, k[0]), headline(t, k[0])
        head_same.append((hf.sentence, hf.series_direction) == (hr.sentence, hr.series_direction))
        coh.append(not _new_contradictions(t, r))
    s = lambda v: pd.Series(v, df.index)
    return {
        "unattacked_is_real": a | (df.text == rt),
        "real_is_exact_prefix": ~a | s(prefix),
        "appended_claim_parses_to_claimed": ~a | s(parsed),
        "claimed_is_minus_real": ~a | ((df.claimed_direction == -df.real_direction) & (df.claimed_direction != 0)),
        "real_headline_kept": ~a | s(head_same),
        "no_new_contradiction": ~a | s(coh),
        "attacked_iff_real_headline": a == (df.real_direction != 0),
    }


def check_a3(df: pd.DataFrame, real: pd.DataFrame) -> dict[str, pd.Series]:
    a = df.attacked
    rt = real.text.reindex(df.index)
    head, coh, plaus = [], [], []
    for k, t, c, r in zip(df.index, df.text, df.claimed_direction, rt):
        head.append(headline(t, k[0]).series_direction == c != 0)
        coh.append(not _new_contradictions(t, r))
        res = ad.edit_text(r if r else None, k[0], "exaggerate", factor=ad.A3_FACTOR)
        plaus.append(res.attacked and res.text == t and not ad.plausibility_problems(r, res, k[0]))
    s = lambda v: pd.Series(v, df.index)
    return {
        "unattacked_is_real": a | (df.text == rt),
        "claimed_is_real_direction": ~a | ((df.claimed_direction == df.real_direction) & (df.claimed_direction != 0)),
        "headline_direction_kept": ~a | s(head),
        "no_new_contradiction": ~a | s(coh),
        "plausible_P1_P4": ~a | s(plaus),
    }


def bookkeeping(df: pd.DataFrame, real: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "same_windows_as_real": pd.Series(df.index.isin(real.index), df.index),
        "has_text_as_real": df.has_text == real.has_text.reindex(df.index),
        "synthetic_iff_attacked": df.synthetic == df.attacked,
        "claimed_nonzero_iff_attacked": (df.claimed_direction != 0) == df.attacked,
    }


def magnitude_ratios(df: pd.DataFrame, real: pd.DataFrame) -> pd.DataFrame:
    """A3: |new| / |old| for every number the engine changed (descriptive)."""
    rows = []
    for k in df.index[df.attacked]:
        r = real.loc[k, "text"]
        res = ad.edit_text(r, k[0], "exaggerate", factor=ad.A3_FACTOR)
        for x, y, new in res.edits:
            mo, mn = re.search(rf"{NUM}", r[x:y]), re.search(rf"{NUM}", new)
            if mo and mn and parse_num(mo.group()):
                rows.append(dict(domain=k[0], ratio=parse_num(mn.group()) / parse_num(mo.group())))
    return pd.DataFrame(rows)


def coverage(df: pd.DataFrame, attack: str) -> pd.DataFrame:
    t = df[df.has_text].reset_index()
    rows = []
    groups = [*t.groupby(["domain", "split"]), *(((d, "val+test"), g) for d, g in t.groupby("domain")),
              (("all", "test"), t[t.split == "test"]), (("all", "val+test"), t)]
    for (dom, split), g in groups:
        rows.append(dict(attack=attack, domain=dom, split=split, texts=len(g),
                         attacked=int(g.attacked.sum()), attacked_pct=round(100 * g.attacked.mean(), 1),
                         claimed_up=int((g.claimed_direction == 1).sum()),
                         claimed_down=int((g.claimed_direction == -1).sum())))
    return pd.DataFrame(rows)


def review_pairs(tables: dict[str, pd.DataFrame], real: pd.DataFrame, n: int = 5) -> str:
    """5 attacked pairs per attack: one per domain where possible, then filled — seed 0."""
    win, lb = ad.frozen_windows(), ad.lookbacks()
    out = ["# R3 — 15 before/after pairs for review by eye (5 per attack, seed 0)", "",
           f"> Fake texts are {common.SYNTHETIC_NOTE}", "",
           "Edits are shown inline as `[old -> new]`; for A2 the appended sentence is shown as",
           "`[ -> …]`. Check: is the lie fluent and coherent, is the claimed direction right,",
           "are the numbers plausible? Write problems next to the pair (they become an amendment).", ""]
    i = 0
    for attack, df in tables.items():
        a = df[df.attacked].reset_index()
        first = a.groupby("domain", sort=False).sample(1, random_state=0)
        rest = a.drop(first.index)
        pick = pd.concat([first, rest.sample(max(0, n - len(first)), random_state=0)]).head(n)
        out += [f"## {attack}", ""]
        for r in pick.itertuples():
            i += 1
            k = (r.domain, r.split, r.sample_id)
            out.append(f"### {i}. {attack} — {r.domain} {r.split} {r.sample_id} | variant "
                       f"{r.variant or '-'} | real direction {r.real_direction:+d} → claimed "
                       f"{r.claimed_direction:+d}")
            if attack == ad.A1:
                w = win[r.domain]
                vals = ", ".join(f"{v:,.4g}" for v in lb[k][-w:])
                out.append(f"*TextGuard's evidence (last {w} values, oldest first): {vals}*")
            out += ["", clean(diff(real.loc[k, "text"], r.text)), ""]
    return "\n".join(out) + "\n"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    real, flip = _load("real"), _load("flipped")
    tables = {c: _load(c) for c in ad.R3_CONDITIONS}
    checks = {ad.A1: check_a1(tables[ad.A1], real, flip),
              ad.A2: check_a2(tables[ad.A2], real),
              ad.A3: check_a3(tables[ad.A3], real)}
    summary, all_ok = [], True
    for attack, df in tables.items():
        cks = {**bookkeeping(df, real), **checks[attack]}
        print("=" * 100 + f"\n{attack}: {int(df.attacked.sum())} attacked of {int(df.has_text.sum())} texts\n")
        for name, ok in cks.items():
            bad = int((~ok.astype(bool)).sum())
            all_ok &= bad == 0
            print(f"   {'PASS' if bad == 0 else 'FAIL'}  {name:<36} {len(ok) - bad}/{len(ok)}")
            if bad:
                for k in ok.index[~ok.astype(bool)][:5]:
                    print("        e.g.", k)
            summary.append(dict(attack=attack, section="check", check=name, n=len(ok), failed=bad))
        cov = coverage(df, attack)
        print("\n   coverage (texts with a report):")
        print("   " + cov.drop(columns="attack").to_string(index=False).replace("\n", "\n   "))
        why = df[df.has_text & ~df.attacked].groupby(["domain", "reason"]).size()
        print("\n   not attacked, by reason:\n   " + why.to_string().replace("\n", "\n   "))
        a = df[df.attacked]
        ratio = a.text.str.len() / real.text.reindex(a.index).str.len()
        print(f"\n   length ratio fake/real: median {ratio.median():.3f}, min {ratio.min():.3f}, "
              f"max {ratio.max():.3f}")
        summary += [dict(attack=attack, section="coverage", **r) for r in cov.drop(columns="attack").to_dict("records")]
        summary.append(dict(attack=attack, section="length_ratio", check="median", n=len(a),
                            value=round(float(ratio.median()), 4)))
    mag = magnitude_ratios(tables[ad.A3], real)
    print("\n   A3 edited numbers, |new| / |old| (median; scaled changes are x10, recomputed anchors"
          " and levels differ):")
    print("   " + mag.groupby("domain").ratio.median().round(2).to_string().replace("\n", "\n   "))
    for dom, g in mag.groupby("domain"):
        summary.append(dict(attack=ad.A3, section="magnitude", domain=dom, check="median |new|/|old|",
                            n=len(g), value=round(float(g.ratio.median()), 4)))

    PAIRS.write_text(review_pairs(tables, real), encoding="utf-8")
    pd.DataFrame(summary).to_csv(OUT, index=False)
    print(f"\n15 review pairs -> {PAIRS}\nsummary -> {OUT}")
    print("\nALL AUTOMATIC CHECKS PASSED" if all_ok else "\nVALIDATION FAILED — do not run the GPU jobs")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
