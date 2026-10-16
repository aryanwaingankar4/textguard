"""EXPLORATORY — R4 v2 dev round 2: candidate fixes evaluated on VAL only (nothing adopted; Aryan decides).

Same answers, same grid, same approved selection rule; each variant only post-processes the claims:
  V0  as built
  V1  + bug fix: multi-month-average units ('average ... for the three months ending') -> unchecked
  V2  + a unit stating a national baseline carries only its level claim (change/outlook dropped)
  V3  + an outlook claim identical to the same unit's change claim is dropped (judge read the past twice)
  V4  + a unit must name the domain's subject to carry change/outlook claims

Motivated by the V0 diagnosis (20 of 27 genuine val texts blocked by an `outlook` claim that copies
the unit's past reading; the judge answers outlook = change on 94-100% of units). Variants were
chosen AFTER seeing V0's val results — a disclosed development iteration on val, never on test.
Run: ``python -m src.explore_r4_v2_variants`` -> ``results/summary_r4_val_variants_v2.csv``.
"""
import re
from dataclasses import replace

import pandas as pd

from src import analyze_r4_val as R
from src.textguard import v2 as V

AVG = re.compile(r"\b(?:average|total)\b.*\bfor the (?:two|three|four|six|twelve|\d+) (?:months|weeks) ending\b", re.I)
SUBJ = {"Economy": r"\bdeficit\b|\btrade balance\b", "SocialGood": r"\bunemployment rate\b",
        "Traffic": r"\btravel\b|vehicle miles", "Health_US": r"\bILI\b|influenza-like|patient visits|outpatient visits"}


def post(claims, domain, level):
    out = []
    by_unit = {}
    for c in claims:
        by_unit.setdefault(c.unit, []).append(c)
    for u, cs in by_unit.items():
        if level >= 1 and AVG.search(u):
            cs = [replace(c, frame="unchecked") if c.kind == "change" else c for c in cs]
            cs = [c for c in cs if c.kind != "outlook"] if level >= 3 else cs
        if level >= 2 and V.level_reference(u) is not None:
            cs = [c for c in cs if c.kind == "level"]
        if level >= 3:
            ch = next((c for c in cs if c.kind == "change"), None)
            cs = [c for c in cs if not (c.kind == "outlook" and ch is not None and c.direction == ch.direction)]
        if level >= 4 and not re.search(SUBJ[domain], u, re.I):
            cs = [c for c in cs if c.kind == "level"]
        out += cs
    return out


def main():
    w = R.frames(); x = w["M5"]; hist = R.histories(x); tx = R.texts()
    base = R.claims_v2(x, tx)
    groups = {"pooled": R.DOMAINS, **{d: (d,) for d in R.DOMAINS}}
    rows = []
    for lev in range(5):
        cl = {c: [post(base[c][i], d, lev) for i, d in enumerate(x.domain)] for c in R.CONDS}
        gr = R.grid(x, cl, hist)
        best, met = R.select(gr)
        g = R.gate_values(x, cl, hist, best.fraction, best.conf_floor)
        for tgt in ("M5", "Q7"):
            for r in R.metrics_row(w[tgt], R.finals_v2(w[tgt], g, best.tau, best.cap_k), groups):
                rows.append(dict(variant=f"V{lev}", target=tgt, budget_met=met, f=best.fraction, c0=best.conf_floor,
                                 tau=best.tau, k=best.cap_k, **r))
    df = pd.DataFrame(rows)
    df.to_csv(R.RESULTS / "summary_r4_val_variants_v2.csv", index=False)
    p = df[df.cell == "pooled"]
    pd.set_option("display.width", 250)
    print(p[["variant", "target", "budget_met", "f", "c0", "tau", "k", "fbr", "mse_real", "obj", "dss_flipped",
             "dss_a2_forward", "dss_exaggerated", "dss_a3_magnitude", "dss_a1_selective"]].round(3).to_string(index=False))
    m = df[(df.target == "M5")].pivot(index="variant", columns="cell", values="fbr").round(3)
    print(m.to_string())


if __name__ == "__main__":
    main()
