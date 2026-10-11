"""R3 — adaptive attacks on TextGuard v1: the test analysis against the pre-registration
(PROJECT_STATUS §10f + Amendment 1).

**Inputs — nothing is re-run here.** M5, M5-sk, Q7, Q7-sk: the three Kaggle runs, verified by
``models.r3_answers`` (pins, input sha256, one setup per file, fingerprints, parser). M3-F / M4-F
(R1's FinBERT heads): R1's predictions for real / none / flipped / exaggerated and
``run_r3_finbert``'s for A1–A3, seed-mean (§10f). TextGuard v1: P6's frozen settings
(``summary_p6_frozen.json``) with the **R3 judge answers** for real and fake texts alike (one
setup). Clip: k = 0.5 (frozen). The skeptical prompt: the sk answers of the same run.

**What is fixed in advance and how the code enforces it.**

* Test split; P5's windows, σ (train std), y_last; NaN pairing; bootstrap 10,000, seed 0,
  stratified by domain; Wilcoxon signed-rank, two-sided.
* Attacked windows = each attack's own ``attacked`` set **minus Amendment 1's exclusions**
  (``results/r3_text_exclusions.csv``, re-derived from the texts here and asserted equal). The
  same tables are recomputed with all attacked texts as a labelled robustness note.
* **One Holm family per target**, sizes asserted (``FAMILY_SIZE``): blocks U, D0, DΔ for every
  defence × attack × cell (pooled + 4 domains). A cell with < 6 non-zero terms is reported as
  "too few windows" and stays in m (p = 1 if empty).
* **SESOI (R2's):** DSS 0.05σ. Verdicts come from the small functions in "verdicts" below —
  a directional claim needs Holm rejection, the claimed sign and an estimate ≥ the SESOI;
  "bounded" only if the 95% CI upper bound < 0.05σ; otherwise inconclusive.
* Everything else — Flip Rate, damage, gate shares, FBR on attacked windows, TG vs CL, P3's
  within-run reference rows, failures, val, ablations — is descriptive (CIs / raw p).

Run: ``python -m src.analyze_r3`` → ``results/summary_r3_*.csv``, ``summary_r3_reading.json``.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src import analyze_p5 as P5
from src import analyze_p6 as P6
from src import metrics as M
from src.analyze_r2 import SESOI_DSS, verdict_dss
from src.attacks import common
from src.attacks import consistency_r3 as CR
from src.attacks.adaptive import A1, A2, A3, R3_CONDITIONS
from src.run_experiments import RESULTS
from src.textguard import baselines as D

LLM_TARGETS = ("M5", "Q7")
FB_TARGETS = ("DLinear+FinBERT", "PatchTST+FinBERT")
TARGETS = LLM_TARGETS + FB_TARGETS
SHORT = {"M5": "M5", "Q7": "Q7", "DLinear+FinBERT": "M3-F", "PatchTST+FinBERT": "M4-F"}
FAMILY = {"M5": "R3-M5", "Q7": "R3-Q7", "DLinear+FinBERT": "R3-M3F", "PatchTST+FinBERT": "R3-M4F"}
FAMILY_SIZE = {"R3-M5": 105, "R3-Q7": 105, "R3-M3F": 75, "R3-M4F": 75}
ATTACKS = R3_CONDITIONS                       # A1, A2, A3
REFERENCE = ("flipped", "exaggerated")         # P3's attacks, within-run reference (descriptive)
CONDS = ("real", "empty") + REFERENCE + ATTACKS
GROUPS = P5.GROUPS_MAIN                        # 4 domains + pooled
CELLS = tuple(GROUPS)
CHANGE_FLIP = ("Economy", "SocialGood", "Traffic")
MIN_NONZERO = 6
KEYS = ["domain", "sample_id"]
LABEL_ATTACK = {A1: "A1 selective flip", A2: "A2 forward-looking lie", A3: "A3 ×10 magnitude lie",
                "flipped": "P3 flipped", "exaggerated": "P3 exaggerated"}


# =============================================================================================
# data
# =============================================================================================

def exclusions() -> set[tuple[str, str, str, str]]:
    """Amendment 1's (attack, domain, split, sample_id) set — the committed CSV, re-derived."""
    saved = pd.read_csv(RESULTS / "r3_text_exclusions.csv", dtype={"sample_id": str})
    fresh = CR.exclusions()
    key = ["attack", "domain", "split", "sample_id"]
    a = set(map(tuple, saved[key].to_numpy()))
    if a != set(map(tuple, fresh[key].to_numpy())):
        raise ValueError("r3_text_exclusions.csv differs from a fresh scan of the texts")
    return a


def finbert_long(split: str = "test") -> pd.DataFrame:
    from src.run_r1 import PREDICTIONS as R1P
    from src.run_r3_finbert import PREDICTIONS as R3P
    r1 = pd.read_csv(R1P, dtype={"sample_id": str})
    r1 = r1[r1.model.isin(FB_TARGETS) & r1.condition.isin(("real", "empty") + REFERENCE)]
    r3 = pd.read_csv(R3P, dtype={"sample_id": str})
    df = pd.concat([r1, r3[r1.columns]], ignore_index=True)
    return M.seed_mean(df[df.split == split]).drop(columns="n_seeds")


def llm_long() -> pd.DataFrame:
    from src.models.r3_answers import load_all
    return load_all()


def wide(long: pd.DataFrame, split: str, models, excl: set) -> dict[str, pd.DataFrame]:
    """Per model, rows aligned to P5's ``window_index``: a column per condition, y_true, y_last,
    sigma, has_text, and for every attack d_<a> (Amendment 1 set) and dall_<a> (all texts)."""
    idx = P5.window_index(split)
    out = {}
    texts = {a: common.load(a) for a in REFERENCE + ATTACKS}
    for model in models:
        m = long[(long.model == model) & (long.split == split)]
        preds = m.pivot(index=KEYS, columns="condition", values="y_pred")
        base = m[m.condition == "real"].set_index(KEYS)[["y_true", "y_last", "train_std"]]
        w = idx.join(base, on=KEYS).join(preds, on=KEYS).rename(columns={"train_std": "sigma"})
        if set(CONDS) - set(w.columns) or w[["y_true", "y_last", "sigma"]].isna().any().any():
            raise ValueError(f"{model}: conditions or windows missing")
        for a, t in texts.items():
            t = t[t.split == split].set_index(KEYS)
            d = t.claimed_direction.where(t.attacked, 0).astype(int)
            dall = w.set_index(KEYS).index.map(d).to_numpy()
            w[f"dall_{a}"] = dall
            if a in ATTACKS:
                bad = np.array([(a, dom, split, s) in excl for dom, s in zip(w.domain, w.sample_id)])
                w[f"d_{a}"] = np.where(bad, 0, dall)
            else:
                w[f"d_{a}"] = dall
        out[model] = w
    ref = next(iter(out.values()))
    for model, w in out.items():
        for c in ("y_true", "y_last", "sigma"):
            if not np.allclose(w[c], ref[c]):
                raise ValueError(f"{model}: {c} misaligned")
    return out


def load(split: str = "test") -> tuple[dict, dict]:
    """(frames for the 4 targets + 2 sk twins, frames' P5 consistency already asserted)."""
    excl = exclusions()
    w = wide(llm_long(), split, ("M5", "M5-sk", "Q7", "Q7-sk"), excl)
    w.update(wide(finbert_long(split), split, FB_TARGETS, excl))
    p5 = P5.wide(P5.load_long(split), split)[P5.QWEN]       # same windows, σ, y_last as P5
    for k, x in w.items():
        if not (np.allclose(x.y_true, p5.y_true) and np.allclose(x.sigma, p5.sigma)
                and np.allclose(x.y_last, p5.y_last)):
            raise ValueError(f"{k}: windows differ from P5's")
    return w, dict(excl=excl)


# =============================================================================================
# defences — every one at its frozen setting
# =============================================================================================

def gates(split: str = "test") -> tuple[dict, pd.DataFrame]:
    """TextGuard v1's gate table on the R3 judge answers (real + every fake), frozen fraction."""
    from src.models.r3_answers import load_judge
    fz = P6.load_frozen()
    gt = P6.gate_table(split, conditions=("real",) + REFERENCE + ATTACKS,
                       fractions=(fz["textguard"]["fraction"],), j=load_judge())
    return fz, gt


def finals(w: dict, target: str, fz: dict, gt: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """{defence: {condition: final forecast}} for none / TG / CL (+ SK for LLM targets)."""
    x = w[target]
    A, sig = x["empty"].to_numpy(float), x.sigma.to_numpy(float)
    conds = ("real",) + REFERENCE + ATTACKS
    g = P6.gates_wide(gt, x, fz["textguard"]["fraction"], conditions=conds)
    tau, k = fz["textguard"]["tau"], fz["clip"]["k"]
    out = {"none": {c: x[c].to_numpy(float) for c in conds},
           "TG": {c: D.textguard(A, x[c].to_numpy(float), g[c], threshold=tau, mode="block").final
                  for c in conds},
           "CL": {c: D.clip(A, x[c].to_numpy(float), sig, k).final for c in conds}}
    if target in LLM_TARGETS:
        xs = w[f"{target}-sk"]
        if not np.allclose(xs["empty"], x["empty"], equal_nan=True):
            raise ValueError("sk's no-report answers must be its p5 twin's (shared prompt)")
        out["SK"] = {c: xs[c].to_numpy(float) for c in conds}
    out["_g"] = g
    return out


def terms(x: pd.DataFrame, fin: dict, attack: str, dcol: str = "d") -> dict[str, dict]:
    """Per-window DSS / flip / damage terms for every defence on ``attack``'s attacked windows."""
    d = x[f"{dcol}_{attack}"].to_numpy()
    mask = x.has_text.to_numpy() & (d != 0)
    out = {}
    for lab, f in fin.items():
        if lab.startswith("_"):
            continue
        fr, ff = f["real"], f[attack]
        pm = M.paired_mse(x.y_true, {"real": fr, "fake": ff}, x.sigma, mask)
        out[lab] = dict(dss=M.dss(ff, fr, d, x.sigma, mask).terms,
                        flip=M.flip_rate(ff, fr, x.y_last, d, mask).terms,
                        damage=M.paired_difference(pm["fake"], pm["real"]))
    return out


# =============================================================================================
# verdicts — §10f's decision rules, mechanically (unit-tested)
# =============================================================================================

def change_verdict(diff: float, reject: bool) -> str:
    """DΔ: 'reduction' / 'increase' need rejection, the sign and |diff| ≥ SESOI."""
    if reject and diff <= -SESOI_DSS:
        return "reduction"
    if reject and diff >= SESOI_DSS:
        return "increase"
    if reject:
        return "significant but negligible"
    return "no change shown"


def defence_reading(u: str, d0: str, dd: str) -> str:
    """§10f's per-defence reading from the U, D0 and DΔ verdicts."""
    if dd == "increase":
        return "makes the attack worse"
    if u != "fooled":
        return "nothing to defend against"
    if d0 == "fooled" and dd != "reduction":
        return "evaded"
    if dd == "reduction" and d0 == "fooled":
        return "reduces but does not stop"
    if dd == "reduction" and d0 == "bounded":
        return "stops"
    if dd == "reduction":
        return "reduces; residual not shown to be below the SESOI"
    return "not evaded, no reduction shown"


# =============================================================================================
# families
# =============================================================================================

def _cell(t, dom, sel):
    return P5._cell(t, dom, sel)


def _test(t: np.ndarray, sel: np.ndarray) -> dict:
    v = t[sel & np.isfinite(t)]
    r = M.wilcoxon(v) if v.size else dict(p=1.0, n_nonzero=0)
    return dict(p_raw=float(r["p"]), n_nonzero=int(r.get("n_nonzero", 0)))


def family_rows(w: dict, target: str, fin: dict, dcol: str = "d", groups=GROUPS) -> pd.DataFrame:
    """Blocks U, D0, DΔ for every attack × cell (and every defence of the target)."""
    x = w[target]
    dom = x.domain.to_numpy()
    defences = [k for k in fin if k not in ("none",) and not k.startswith("_")]
    rows = []
    for a in ATTACKS:
        t = terms(x, fin, a, dcol)
        for g, sel in P5._groups(x, groups):
            base = t["none"]["dss"]
            c = _cell(base, dom, sel)
            rows.append(dict(target=target, block="U", defence="none", attack=a, cell=g,
                             estimate=c["value"], ci_low=c["ci_low"], ci_high=c["ci_high"], n=c["n"],
                             **_test(base, sel)))
            for dfn in defences:
                td = t[dfn]["dss"]
                c = _cell(td, dom, sel)
                rows.append(dict(target=target, block=f"{dfn}0", defence=dfn, attack=a, cell=g,
                                 estimate=c["value"], ci_low=c["ci_low"], ci_high=c["ci_high"],
                                 n=c["n"], **_test(td, sel)))
                diff = td - base
                c = _cell(diff, dom, sel)
                ok = sel & np.isfinite(diff)
                rows.append(dict(target=target, block=f"{dfn}Δ", defence=dfn, attack=a, cell=g,
                                 estimate=c["value"], ci_low=c["ci_low"], ci_high=c["ci_high"],
                                 n=c["n"], dss_defended=_cell(td, dom, ok)["value"],
                                 dss_none=_cell(base, dom, ok)["value"], **_test(diff, sel)))
    return pd.DataFrame(rows)


def holm(fam: pd.DataFrame, target: str) -> pd.DataFrame:
    fam = fam.copy()
    m = FAMILY_SIZE[FAMILY[target]]
    if len(fam) != m or fam.p_raw.isna().any():
        raise ValueError(f"{FAMILY[target]}: {len(fam)} tests, pre-registered {m}")
    fam["family"] = FAMILY[target]
    fam["p_holm"] = M.holm(fam.p_raw)
    fam["reject"] = fam.p_holm < 0.05
    fam["too_few_windows"] = fam.n_nonzero < MIN_NONZERO
    fam["verdict"] = [
        (verdict_dss(r.estimate, r.ci_low, r.ci_high, r.reject) if not r.block.endswith("Δ")
         else change_verdict(r.estimate, r.reject)) for r in fam.itertuples()]
    fam.loc[fam.too_few_windows, "verdict"] = "too few windows (" + fam.verdict + ")"
    return fam


def _base(v: str) -> str:
    return v.split("(", 1)[1].rstrip(")") if v.startswith("too few windows (") else v


def readings(fam: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, a, cell), g in fam.groupby(["target", "attack", "cell"], sort=False):
        v = g.set_index("block").verdict.map(_base)
        est = g.set_index("block").estimate
        for dfn in [b[:-1] for b in g.block if b.endswith("0")]:
            rows.append(dict(target=target, attack=a, cell=cell, defence=dfn, U=v["U"],
                             dss_none=est["U"], D0=v[f"{dfn}0"], dss_defended=est[f"{dfn}0"],
                             DΔ=v[f"{dfn}Δ"], change=est[f"{dfn}Δ"],
                             reading=defence_reading(v["U"], v[f"{dfn}0"], v[f"{dfn}Δ"])))
    return pd.DataFrame(rows)


def run_families(w: dict, fz: dict, gt: pd.DataFrame, dcol: str = "d") -> tuple[pd.DataFrame, pd.DataFrame]:
    fams = []
    for t in TARGETS:
        fin = finals(w, t, fz, gt)
        fams.append(holm(family_rows(w, t, fin, dcol), t))
    fam = pd.concat(fams, ignore_index=True)
    return fam, readings(fam)


# =============================================================================================
# descriptive
# =============================================================================================

def operating(w: dict, fz: dict, gt: pd.DataFrame, dcol: str = "d") -> pd.DataFrame:
    """DSS, Flip Rate, damage per target × attack × defence × cell, with CIs (+ P3 reference)."""
    rows = []
    for tgt in TARGETS:
        x, fin = w[tgt], finals(w, tgt, fz, gt)
        dom = x.domain.to_numpy()
        for a in ATTACKS + REFERENCE:
            t = terms(x, fin, a, dcol)
            for dfn, q in t.items():
                for g, sel in P5._groups(x, {**GROUPS, "change-flip (Eco+SG+Tra)": CHANGE_FLIP}):
                    r = dict(target=tgt, attack=a, defence=dfn, cell=g)
                    for k, v in q.items():
                        c = _cell(v, dom, sel)
                        r.update({k: c["value"], f"{k}_lo": c["ci_low"], f"{k}_hi": c["ci_high"],
                                  f"n_{k}": c["n"]})
                    rows.append(r)
    return pd.DataFrame(rows)


def tg_vs_clip(w: dict, fz: dict, gt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tgt in TARGETS:
        x, fin = w[tgt], finals(w, tgt, fz, gt)
        dom = x.domain.to_numpy()
        for a in ATTACKS:
            t = terms(x, fin, a)
            diff = t["TG"]["dss"] - t["CL"]["dss"]
            for g, sel in P5._groups(x, GROUPS):
                c = _cell(diff, dom, sel)
                rows.append(dict(target=tgt, attack=a, cell=g, diff_tg_minus_clip=c["value"],
                                 ci_low=c["ci_low"], ci_high=c["ci_high"], n=c["n"],
                                 p_raw=_test(diff, sel)["p_raw"]))
    return pd.DataFrame(rows)


def gate_shares(w: dict, gt: pd.DataFrame, fz: dict) -> pd.DataFrame:
    """The gate on each attack's fakes vs on the genuine report of the same windows (FBR there).
    The gate depends on text + lookback only, so it is the same for every target."""
    x = w["M5"]
    g = P6.gates_wide(gt, x, fz["textguard"]["fraction"], conditions=("real",) + REFERENCE + ATTACKS)
    f = fz["textguard"]["fraction"]
    gtf = gt[gt.fraction == f].set_index(["domain", "sample_id", "condition"])
    rows = []
    for a in ATTACKS + REFERENCE:
        d = x[f"d_{a}"].to_numpy()
        for grp, sel in P5._groups(x, GROUPS):
            att = sel & x.has_text.to_numpy() & (d != 0)
            # what the judge reads on the fake: its change-claim direction vs the claimed one
            jd = np.array([gtf.loc[(dm, s, a), "dir_change"] for dm, s in zip(x.domain[att], x.sample_id[att])],
                          float)
            rows.append(dict(attack=a, cell=grp, n=int(att.sum()),
                             fake_gated_share=float((g[a][att] < 0.5).mean()) if att.any() else np.nan,
                             fake_mean_g=float(g[a][att].mean()) if att.any() else np.nan,
                             fbr_genuine_same_windows=float((g["real"][att] < 0.5).mean()) if att.any() else np.nan,
                             genuine_mean_g=float(g["real"][att].mean()) if att.any() else np.nan,
                             judge_reads_claimed=float(np.mean(jd == d[att])) if att.any() else np.nan,
                             judge_reads_opposite=float(np.mean(jd == -d[att])) if att.any() else np.nan,
                             judge_reads_none=float(np.mean(~np.isfinite(jd) | (jd == 0))) if att.any() else np.nan))
    return pd.DataFrame(rows)


def coverage(w: dict) -> pd.DataFrame:
    x = w["M5"]
    rows = []
    for a in ATTACKS:
        for grp, sel in P5._groups(x, GROUPS):
            has = sel & x.has_text.to_numpy()
            rows.append(dict(attack=a, cell=grp, texts=int(has.sum()),
                             attacked_all=int((has & (x[f"dall_{a}"] != 0)).sum()),
                             attacked_kept=int((has & (x[f"d_{a}"] != 0)).sum())))
    return pd.DataFrame(rows)


def failures(long: pd.DataFrame) -> pd.DataFrame:
    t = long[long.split == "test"]
    return (t.assign(fail=t.y_pred.isna()).groupby(["model", "condition"])
            .agg(n=("fail", "size"), failures=("fail", "sum")).reset_index())


def ablations(w: dict, fz: dict, gt: pd.DataFrame) -> pd.DataFrame:
    """Outside the families (raw p): pooled without 2020; the change-flip domains pooled."""
    out = []
    for tgt in TARGETS:
        fin = finals(w, tgt, fz, gt)
        x = w[tgt]
        keep = P5.year_mask(x, 2020)
        xx = {tgt: x}
        out.append(family_rows(xx, tgt, fin, groups={"change-flip (Eco+SG+Tra)": CHANGE_FLIP})
                   .assign(ablation="Health_US vs change-flip domains"))
        x2 = x.copy()
        for a in ATTACKS:
            x2[f"d_{a}"] = np.where(keep, x2[f"d_{a}"], 0)
        out.append(family_rows({tgt: x2}, tgt, fin, groups={"pooled": tuple(P5.DOMAINS)})
                   .assign(ablation="excluding 2020"))
    return pd.concat(out, ignore_index=True)


def val_u(excl: set) -> pd.DataFrame:
    """VAL, labelled: undefended DSS per target × attack (pooled + domains)."""
    w = wide(llm_long(), "val", ("M5", "Q7"), excl)
    w.update(wide(finbert_long("val"), "val", FB_TARGETS, excl))
    rows = []
    for tgt, x in w.items():
        dom = x.domain.to_numpy()
        for a in ATTACKS:
            d = x[f"d_{a}"].to_numpy()
            t = M.dss(x[a], x.real, d, x.sigma, x.has_text.to_numpy() & (d != 0)).terms
            for g, sel in P5._groups(x, GROUPS):
                c = _cell(t, dom, sel)
                rows.append(dict(split="val", target=tgt, attack=a, cell=g, dss=c["value"],
                                 ci_low=c["ci_low"], ci_high=c["ci_high"], n=c["n"]))
    return pd.DataFrame(rows)


def single_window_sensitivity(w: dict, fz: dict, gt: pd.DataFrame) -> pd.DataFrame:
    """DESCRIPTIVE: every pooled DSS mean with its single most extreme window removed, beside
    the full mean and the median — flags cells whose mean (and so the rule's sign) one window
    decides (found: M5-sk on A2, a copied 2,730.6-billion figure that passes the 10× rule)."""
    rows = []
    for tgt in TARGETS:
        x, fin = w[tgt], finals(w, tgt, fz, gt)
        for a in ATTACKS:
            for dfn, q in terms(x, fin, a).items():
                t = q["dss"][np.isfinite(q["dss"])]
                if not t.size:
                    continue
                i = int(np.argmax(np.abs(t)))
                k = np.flatnonzero(np.isfinite(q["dss"]))[i]
                rest = np.delete(t, i)
                rows.append(dict(target=tgt, attack=a, defence=dfn, n=t.size, mean=t.mean(),
                                 median=float(np.median(t)), extreme_term=t[i],
                                 extreme_window=f"{x.domain.iloc[k]} {x.sample_id.iloc[k]}",
                                 mean_without=rest.mean() if rest.size else np.nan,
                                 p_without=M.wilcoxon(rest)["p"] if rest.size else np.nan))
    return pd.DataFrame(rows)


def a1_genuine_cost(w: dict, fz: dict, gt: pd.DataFrame) -> pd.DataFrame:
    """DESCRIPTIVE: real-text MSE on A1's windows, undefended vs TG / CL — the accuracy cost of
    gating the genuine report where the attacker attacks (§10f's FBR-on-A1 question)."""
    rows = []
    for tgt in TARGETS:
        x, fin = w[tgt], finals(w, tgt, fz, gt)
        dom = x.domain.to_numpy()
        for dc, lab in (("d", "Amendment 1"), ("dall", "all texts")):
            att = x.has_text.to_numpy() & (x[f"{dc}_{A1}"].to_numpy() != 0)
            pm = M.paired_mse(x.y_true, {k: fin[k]["real"] for k in ("none", "TG", "CL")}, x.sigma, att)
            for k in ("TG", "CL"):
                diff = M.paired_difference(pm[k], pm["none"])
                c = _cell(diff, dom, np.ones(len(x), bool))
                rows.append(dict(target=tgt, windows=lab, defence=k, n=c["n"],
                                 mse_real_none=np.nanmean(pm["none"].terms[att]),
                                 mse_real_defended=np.nanmean(pm[k].terms[att]), diff=c["value"],
                                 ci_low=c["ci_low"], ci_high=c["ci_high"], p_raw=M.wilcoxon(diff)["p"]))
    return pd.DataFrame(rows)


def c8_sensitivity(fz: dict, gt: pd.DataFrame, excl: set) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SENSITIVITY only: the families with C8's flags added to the exclusions. C8 was found after
    the metrics existed and is NOT adopted (Aryan, 2026-10-11) — Amendment 1's set stays primary."""
    extra = set()
    real = common.load("real").set_index(["domain", "split", "sample_id"])
    for a in ATTACKS:
        t = common.load(a)
        for _, r in t[t.attacked & (t.split == "test")].iterrows():
            k = (r.domain, r.split, r.sample_id)
            if set(CR.c8_from_to_direction(r.text, r.domain)) - set(CR.c8_from_to_direction(real.loc[k, "text"], r.domain)):
                extra.add((a,) + k)
    ex = excl | extra
    w = wide(llm_long(), "test", ("M5", "M5-sk", "Q7", "Q7-sk"), ex)
    w.update(wide(finbert_long("test"), "test", FB_TARGETS, ex))
    fam, rd = run_families(w, fz, gt, "d")
    return fam.assign(analysis="sensitivity: with C8 (not adopted)"), rd.assign(analysis="sensitivity: with C8 (not adopted)",
                                                                       c8_extra=len(extra))


# =============================================================================================
# run
# =============================================================================================

def headline(rd: pd.DataFrame) -> dict:
    """§10f headline: TextGuard v1 breaks under X if TG is evaded by X or makes X worse, M5 pooled."""
    out = {}
    m5 = rd[(rd.target == "M5") & (rd.cell == "pooled") & (rd.defence == "TG")].set_index("attack")
    for a in ATTACKS:
        r = m5.loc[a]
        out[a] = dict(U=r.U, TG0=r.D0, TGΔ=r.DΔ, reading=r.reading,
                      textguard_breaks=r.reading in ("evaded", "makes the attack worse"))
    return out


def run(save: bool = True) -> dict:
    w, meta = load("test")
    fz, gt = gates("test")
    P6._check_y_last(w["M5"], "test")
    fam, rd = run_families(w, fz, gt, "d")
    fam_all, rd_all = run_families(w, fz, gt, "dall")
    out = dict(family=fam, readings=rd, robustness_family=fam_all, robustness_readings=rd_all,
               operating=operating(w, fz, gt), operating_all_texts=operating(w, fz, gt, "dall"),
               tg_vs_clip=tg_vs_clip(w, fz, gt), gates=gate_shares(w, gt, fz), coverage=coverage(w),
               failures=failures(llm_long()), ablations=ablations(w, fz, gt), val=val_u(meta["excl"]),
               single_window_sensitivity=single_window_sensitivity(w, fz, gt),
               a1_genuine_cost=a1_genuine_cost(w, fz, gt))
    out["sensitivity_c8_family"], out["sensitivity_c8_readings"] = c8_sensitivity(fz, gt, meta["excl"])
    reading = dict(headline_M5_pooled=headline(rd), headline_M5_pooled_all_texts=headline(rd_all),
                   rejections={f: int(g.reject.sum()) for f, g in fam.groupby("family")},
                   family_sizes={f: len(g) for f, g in fam.groupby("family")})
    if save:
        for k, v in out.items():
            v.to_csv(RESULTS / f"summary_r3_{k}.csv", index=False)
        (RESULTS / "summary_r3_reading.json").write_text(json.dumps(reading, indent=1, ensure_ascii=False),
                                                         encoding="utf-8")
    out["reading"] = reading
    return out


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    out = run()
    pd.set_option("display.width", 250)
    rd = out["readings"]
    print("READINGS (pooled):\n" + rd[rd.cell == "pooled"].round(3).to_string(index=False))
    print("\n" + json.dumps(out["reading"], indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
