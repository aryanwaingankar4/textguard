"""Validate the rule-based fake texts (P3.3).

1. **Human labels** — the 50 reports hand-labelled in P1 (``results/summary_directional_sample.csv``,
   ``manual`` = up / down / none: what the text literally claims about its headline subject).
   For each: does the rule engine read the same headline direction, and for the directional
   ones, does the flipped text claim the opposite? These reports predate the test split, so
   they are an independent check on rules written against val/test texts.
2. **Coverage** — per domain, % of val+test texts attacked vs "not attacked", with reasons.
3. **Automatic checks on every attacked text** — headline reversed (flip) / kept (exaggerate),
   no new arithmetic or logical contradiction (``rules.coherence_problems``), length change.
4. **20 before/after pairs** (5 per domain, fixed seed) as inline diffs for review by eye.

Run after ``python -m src.attacks.generate``:  ``python -m src.attacks.validate_attacks``
Writes ``results/summary_p3_attack_validation.csv`` (kept in git).
"""

from __future__ import annotations

import difflib
import re
import sys

import numpy as np
import pandas as pd

from src.attacks import common
from src.attacks.rules import coherence_problems, edit_text, headline

LABELS = common.ROOT / "results" / "summary_directional_sample.csv"
OUT = common.ROOT / "results" / "summary_p3_attack_validation.csv"
LABEL_DIR = {"up": 1, "down": -1, "none": 0}


def human_label_check() -> pd.DataFrame:
    lab = pd.read_csv(LABELS)
    rows = []
    for r in lab.itertuples():
        h = headline(r.fact, r.domain)
        res = edit_text(r.fact, r.domain, "flip")
        fake = headline(res.text, r.domain) if res.attacked else None
        human = LABEL_DIR.get(str(r.manual).strip().lower(), 0)
        rows.append(dict(domain=r.domain, human=human, rules=h.text_direction,
                         attacked=res.attacked, reason=res.reason,
                         fake=fake.text_direction if fake else 0))
    df = pd.DataFrame(rows)
    df["read_agrees"] = df.human == df.rules
    df["flipped_correctly"] = df.attacked & (df.human != 0) & (df.fake == -df.human)
    return df


def diff(a: str, b: str) -> str:
    ta, tb = re.split(r"(\s+)", a), re.split(r"(\s+)", b)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ta, b=tb, autojunk=False).get_opcodes():
        if op == "equal":
            out.append("".join(ta[i1:i2]))
        else:
            old, new = "".join(ta[i1:i2]).strip(), "".join(tb[j1:j2]).strip()
            out.append(f"[{old} -> {new}]" + ("".join(ta[i2 - 1:i2]) if ta[i1:i2] and not ta[i2 - 1].strip() else ""))
    return "".join(out)


def clean(s: str) -> str:
    return re.sub(r"\s*\r?\n\s*", " / ", s).strip()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    summary = []

    # 1. human labels
    hl = human_label_check()
    print("=" * 100 + "\n1. HUMAN-LABELLED REPORTS (P1 sample, 50)\n")
    t = hl.groupby("domain").agg(n=("human", "size"), directional=("human", lambda s: int((s != 0).sum())),
                                 rules_read_same=("read_agrees", "sum"),
                                 flipped_correctly=("flipped_correctly", "sum"))
    t.loc["ALL"] = t.sum()
    print(t.to_string())
    d = hl[hl.human != 0]
    print(f"\ndirectional (human up/down): {len(d)}; flipped correctly: {int(d.flipped_correctly.sum())}"
          f" ({d.flipped_correctly.mean():.0%}); not attacked: {int((~d.attacked).sum())}"
          f"; attacked but wrong direction: {int((d.attacked & ~d.flipped_correctly).sum())}")
    n = hl[hl.human == 0]
    print(f"non-directional (human none): {len(n)}; left unattacked: {int((~n.attacked).sum())}"
          f"; attacked anyway (rules saw a direction): {int(n.attacked.sum())}")
    print("   (P1 labelled up/down CHANGE claims; a Health_US level claim such as 'ILI was 2.1%,"
          " below the national baseline' was labelled none there, but level-flip attacks it by design)")
    for r in hl[~hl.read_agrees].itertuples():
        print(f"   disagreement: {r.domain:<10} human={r.human:+d} rules={r.rules:+d} "
              f"attacked={r.attacked} {r.reason}")
    for dom, g in hl.groupby("domain"):
        dd = g[g.human != 0]
        summary.append(dict(section="human_labels", domain=dom, n=len(g), directional=len(dd),
                            flipped_correctly=int(dd.flipped_correctly.sum()),
                            rules_read_same=int(g.read_agrees.sum())))

    # 2-3. coverage + automatic checks
    for cond in ("flipped", "exaggerated"):
        df = common.load(cond)
        real = common.load("real").set_index(["domain", "split", "sample_id"]).text
        df["orig"] = real.reindex(df.set_index(["domain", "split", "sample_id"]).index).values
        t = df[df.has_text]
        print("\n" + "=" * 100 + f"\n2. COVERAGE — {cond} (val + test texts)\n")
        cov = t.groupby("domain").agg(texts=("attacked", "size"), attacked=("attacked", "sum"))
        cov["attacked_pct"] = (100 * cov.attacked / cov.texts).round(1)
        cov["not_attacked"] = cov.texts - cov.attacked
        print(cov.to_string())
        print("\nnot attacked, by reason:")
        print(t[~t.attacked].groupby(["domain", "reason"]).size().to_string())
        a = t[t.attacked]
        want = (-a.real_direction) if cond == "flipped" else a.real_direction
        head_ok = (a.claimed_direction == want) & (a.claimed_direction != 0)
        new_prob = [set(coherence_problems(x)) - set(coherence_problems(o)) for x, o in zip(a.text, a.orig)]
        ratio = a.text.str.len() / a.orig.str.len()
        print(f"\n3. AUTOMATIC CHECKS — {cond}: {len(a)} attacked texts")
        print(f"   headline {'reversed' if cond == 'flipped' else 'kept'}: {int(head_ok.sum())}/{len(a)}")
        print(f"   new contradictions (arithmetic / baseline side / sign): {sum(bool(p) for p in new_prob)}")
        print(f"   length ratio fake/real: median {ratio.median():.3f}, min {ratio.min():.3f}, max {ratio.max():.3f}")
        for dom, g in t.groupby("domain"):
            ga = g[g.attacked]
            summary.append(dict(section=cond, domain=dom, n=len(g), attacked=int(len(ga)),
                                attacked_pct=round(100 * len(ga) / len(g), 1),
                                headline_ok=int(((ga.claimed_direction == ((-ga.real_direction) if cond == "flipped"
                                                                         else ga.real_direction))).sum()),
                                claimed_up=int((ga.claimed_direction == 1).sum()),
                                claimed_down=int((ga.claimed_direction == -1).sum())))

    # 4. 20 before/after pairs
    df = common.load("flipped")
    real = common.load("real").set_index(["domain", "split", "sample_id"]).text
    a = df[df.attacked].copy()
    a["orig"] = real.reindex(a.set_index(["domain", "split", "sample_id"]).index).values
    pick = pd.concat([a[a.domain == d].sample(5, random_state=0) for d in common.DOMAINS])
    print("\n" + "=" * 100 + "\n4. TWENTY FLIPPED PAIRS (5 per domain, seed 0) — edits shown as [old -> new]\n")
    for i, r in enumerate(pick.itertuples(), 1):
        print(f"--- {i:2d}. {r.domain} {r.split} {r.sample_id} | variant={r.variant} | "
              f"real direction {r.real_direction:+d} -> claimed {r.claimed_direction:+d}")
        print("    " + clean(diff(r.orig, r.text)) + "\n")

    pd.DataFrame(summary).to_csv(OUT, index=False)
    print(f"summary -> {OUT}")


if __name__ == "__main__":
    main()
