"""EXPLORATORY — do P5's and P6's headline conclusions survive dropping the P3 texts that R3's
contradiction scan flags? (PROJECT_STATUS §10f Amendment 1, "Disclosure".)

Why this exists. R3 Amendment 1's scan (``src.attacks.consistency_r3``) also flags 47 / 251
attacked test ``flipped`` and 51 / 250 ``exaggerated`` P3 texts (mostly Traffic: an edited
percentage no longer matches the levels the report states). P5 and P6 are **frozen** — their
pre-registered numbers and verdicts stand and are not re-decided here. This script only asks
whether the headline conclusions would read differently without those texts, so the paper can
state the limitation with a number beside it. Labelled exploratory, outside every family.

What it recomputes, with each attack's flagged windows removed **from that attack only**
(a flagged ``flipped`` window still counts for ``exaggerated`` if that text is clean):

* P5 block A: Model 5 (and Models 3–4 for completeness) DSS + Flip Rate, pooled + per domain;
* P6 operating point, Model 5 ``flipped``: DSS / Flip Rate for always_trust, clip, TextGuard,
  and the T1–T3 comparisons (raw p, and Holm over the same 20 cells as an exploratory replica).
  FBR and real-text MSE are about **genuine** texts, so they are not affected and not recomputed.

The unmasked run is recomputed too and asserted equal to the frozen tables, so any difference
is the exclusion and nothing else.

Run: ``python -m src.explore_p3_exclusions`` → ``results/summary_p3_exclusion_check.csv``.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src import analyze_p5 as P5
from src import analyze_p6 as P6
from src import metrics as M
from src.attacks import consistency_r3 as cr

OUT = P5.RESULTS / "summary_p3_exclusion_check.csv"


def keep_mask(x: pd.DataFrame, attack: str) -> np.ndarray:
    """True for windows whose ``attack`` text is not flagged by the scan (test split)."""
    s = cr.scan(attack)
    bad = set(zip(s[s.split == "test"].domain, s[s.split == "test"].sample_id))
    return np.array([(d, i) not in bad for d, i in zip(x.domain, x.sample_id)])


def p5_rows(w: dict) -> pd.DataFrame:
    rows = []
    for a in P5.ATTACKS:
        keep = keep_mask(w[P5.QWEN], a)
        for label, mask in (("frozen (all texts)", None), ("flagged texts excluded", keep)):
            t = P5.attack(w, mask, models=P5.TEXT_MODELS)
            rows.append(t[t.condition == a].assign(analysis=label))
    return pd.concat(rows, ignore_index=True)


def p6_rows(w: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    fz = P6.load_frozen()
    f = fz["textguard"]["fraction"]
    x5 = w[P5.QWEN]
    gt = P6.gate_table("test", fractions=(f,))
    gt["gL"] = P6.level_only(gt)
    go = P6.gate_table("test", conditions=("real",) + P6.ATTACKS_P6, fractions=(f,), judge_kind="oracle")
    gates = dict(g=P6.gates_wide(gt, x5, f), gL=P6.gates_wide(gt, x5, f, col="gL"),
                 gO=P6.gates_wide(go, x5, f))
    specs = P6._specs(fz, gates["g"], gates["gL"], gates["gO"])
    tg, cl = specs[4][0], specs[3][0]
    ops, tests = [], []
    for attack in P6.ATTACKS_P6:
        keep = keep_mask(x5, attack)
        for label, mask in (("frozen (all texts)", None), ("flagged texts excluded", keep)):
            terms = P6._all_terms(x5, specs, attack, mask)
            for lab in ("none", "always_trust", cl, tg):
                t = terms[lab]
                for r in P6._summ({q: t[q] for q in ("dss", "flip", "damage")}, x5, P6.GROUPS_P6):
                    ops.append(dict(attack=attack, defence=lab, analysis=label, **r))
            if attack == "flipped":
                ht = P6.hypothesis_tests(x5, terms, tg, cl)
                fam = ht[ht.domain.isin(P5.GROUPS_MAIN)].copy()
                fam["p_holm_exploratory"] = M.holm(fam.p_raw)
                tests.append(fam.assign(analysis=label))
    return pd.DataFrame(ops), pd.concat(tests, ignore_index=True)


def _check_frozen(p5: pd.DataFrame, op: pd.DataFrame) -> None:
    """The unmasked recomputation must equal the committed P5 / P6 tables."""
    att = pd.read_csv(P5.RESULTS / "summary_p5_attack.csv")
    mine = p5[(p5.analysis == "frozen (all texts)") & (p5.model == P5.QWEN)]
    ref = att[(att.model == P5.QWEN) & att.condition.isin(P5.ATTACKS)]
    m = mine.merge(ref, on=["condition", "domain"], suffixes=("", "_ref"))
    assert len(m) == len(mine) and np.allclose(m.dss, m.dss_ref) and np.allclose(m.flip_rate, m.flip_rate_ref)
    o = pd.read_csv(P6.RESULTS / "summary_p6_operating.csv")
    o = o[o.model == P5.QWEN]
    mine6 = op[op.analysis == "frozen (all texts)"]
    m6 = mine6.merge(o, on=["attack", "defence", "domain"], suffixes=("", "_ref"))
    assert len(m6) == len(mine6) and np.allclose(m6.dss, m6.dss_ref, equal_nan=True)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    w = P5.wide(P5.load_long("test"), "test")
    p5 = p5_rows(w)
    op, tests = p6_rows(w)
    _check_frozen(p5, op)
    print("unmasked recomputation = the frozen P5 / P6 tables (asserted)\n")
    cols = ["analysis", "short", "condition", "domain", "n", "dss", "dss_ci_low", "dss_ci_high",
            "dss_p_raw", "flip_rate", "toward", "away", "flip_p_raw"]
    print("P5 block A (EXPLORATORY):\n" + p5[cols].round(4).to_string(index=False))
    print("\nP6 operating point, Model 5 (EXPLORATORY):\n"
          + op[["analysis", "attack", "defence", "domain", "n_dss", "dss", "dss_lo", "dss_hi", "flip",
                "damage"]].round(4).to_string(index=False))
    print("\nP6 T1–T4, Model 5 flipped (EXPLORATORY; Holm over the same 20 cells):\n"
          + tests[["analysis", "block", "domain", "estimate", "ci_low", "ci_high", "n", "p_raw",
                   "p_holm_exploratory", "sign_ok"]].round(4).to_string(index=False))
    out = pd.concat([p5.assign(table="p5_attack"), op.assign(table="p6_operating"),
                     tests.assign(table="p6_tests")], ignore_index=True)
    out.insert(0, "label", "EXPLORATORY - P5/P6 are frozen; not a re-decision")
    out.to_csv(OUT, index=False)
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
