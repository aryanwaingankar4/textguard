"""P6b analysis: validate the direction judge, tune TextGuard on val, evaluate once on test.

Everything here follows the pre-registered plan (PROJECT_STATUS §10b + Amendment 1). The
module is split into three stages that are run **in order**, and the order is the point:

1. ``judge``  — is the judge any good? Its answers vs the 50 (+25) human labels from P1, and
   vs the **known** ``claimed_direction`` of every rule-written fake (P3). The judge is never
   tuned, so this may use val and test texts. TextGuard cannot be better than its judge.
2. ``tune``   — **val only.** Gate values for every val window, the 33-config TextGuard grid
   and the 5-value clip grid on Model 5 / ``flipped``; the pre-registered selection rule;
   the chosen settings are written to ``results/summary_p6_frozen.json``.
3. ``test``   — reads the frozen settings from that file (it refuses to run without it) and
   evaluates once: the trade-off curve, the operating point, the Holm family (asserted
   m = 20), secondary MSE / damage, Models 3–4 "no harm", ``exaggerated``, COVID FBR,
   ablations.

**Nothing is re-run.** Model predictions come from P5 (``analyze_p5.load_long``), judge
answers from the P6a Colab CSV (``text_direction.load_judgements``), lookback windows from
``results/qwen/qwen_inputs.csv``. The gate only needs (text claims, lookback numbers, σ), so
it is computed once per window × condition and shared by every model.

**Defended forecasts** (``baselines``): final = A + g′·(B − A), A = the ``empty`` forecast,
B = the forecast with that condition's text. Both the real-text and the fake-text forecast
pass through the same defence, so DSS / Flip Rate measure how far the *deployed* output
moves when the true report is replaced by the lie. NaN rules: B NaN → A, A NaN → NaN.
"none" in the tables = the raw model with P5's NaN pairing (no fallback); "always_trust" =
g′ = 1 through the same NaN fallback as every defence, so the two differ only on failed
windows (Model 5 Traffic unit slips).

Run: ``python -m src.analyze_p6 judge | tune | test``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src import metrics as M
from src.analyze_p5 import (BOOT_SEED, GROUPS_MAIN, N_BOOT, QWEN, SHORT, _cell, _groups,
                            load_long, wide, year_mask)
from src.attacks import common
from src.data.windows import lookback_for
from src.models.train_baselines import DOMAINS
from src.run_experiments import RESULTS, ROOT
from src.textguard import baselines as D
from src.textguard.export_judge_inputs import HUMAN_LABELS
from src.textguard.text_direction import Claim, QwenJudge, load_judgements, stated_baseline
from src.textguard.textguard import gate as tg_gate
from src.textguard.trend_detector import window_for

# --- the pre-registered grid and rule (§10b) — do not edit after P6 results exist ----------
FRACTIONS = (0.25, 0.5, 1.0)
TAUS = tuple(round(0.1 * i, 1) for i in range(11))           # 0, 0.1, …, 1.0
CLIP_KS = (0.05, 0.1, 0.25, 0.5, 1.0)
FBR_BUDGET = 0.10
HOLM_M = 20

TEXT_MODELS_P6 = ("DLinear+BERT", "PatchTST+BERT", QWEN)
CHANGE_FLIP = ("Economy", "SocialGood", "Traffic")
COND_P6 = ("real", "flipped", "exaggerated", "shuffled", "cross_domain", "time_shifted")
FROZEN = RESULTS / "summary_p6_frozen.json"
QWEN_INPUTS = RESULTS / "qwen" / "qwen_inputs.csv"
FIGURES = ROOT / "paper" / "figures"


def _save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(RESULTS / f"summary_p6_{name}.csv", index=False, float_format="%.6g")


# =============================================================================================
# 1. Judge validation
# =============================================================================================

LABELS = (-1, 0, 1)
WORD = {"change": {1: "up", -1: "down", 0: "none"}, "level": {1: "above", -1: "below", 0: "none"}}


def _agreement(truth, pred) -> dict:
    """Accuracy, Cohen's κ (3 classes) and the share of 'none' answers."""
    truth, pred = np.asarray(truth, int), np.asarray(pred, int)
    n = truth.size
    kappa = (cohen_kappa_score(truth, pred, labels=list(LABELS))
             if n and len(set(truth) | set(pred)) > 1 else np.nan)
    claimed = truth != 0
    return dict(n=n, accuracy=float((truth == pred).mean()) if n else np.nan, kappa=kappa,
                none_rate=float((pred == 0).mean()) if n else np.nan,
                # among texts that make a claim (truth ≠ 0) and the judge also reads one:
                # is the direction right? Separates "missed the claim" from "reversed it"
                acc_when_answered=float((truth == pred)[claimed & (pred != 0)].mean())
                if (claimed & (pred != 0)).any() else np.nan,
                reversed=int((claimed & (pred == -truth)).sum()))


def _confusion(df: pd.DataFrame, group_cols: list[str], truth: str, pred: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        for t in LABELS:
            row = dict(zip(group_cols, key), truth=t)
            for p in LABELS:
                row[f"judge_{p:+d}" if p else "judge_0"] = int(((g[truth] == t) & (g[pred] == p)).sum())
            if sum(v for k, v in row.items() if k.startswith("judge_")):
                rows.append(row)
    return pd.DataFrame(rows)


def judge_human(j: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(i) Judge vs the P1 human labels (``manual`` ∈ up/down/none, about the literal subject).

    Pre-registered: the **change** answer (``literal``, before Economy's deficit → balance
    mapping, because the humans labelled the deficit) vs ``manual``. The 50 main labels are
    the test; the 25 filtered-SocialGood re-check labels are supplementary. For Health_US the
    **level** answer is also cross-tabulated against the same labels — descriptive only: the
    P1 labels record up/down *change* claims, so a level claim was labelled "none" there.
    """
    key = j.set_index(["domain", "question", "text"])
    rows, conf_rows = [], []
    for path, set_name in zip(HUMAN_LABELS, ("main_50", "socialgood_recheck_25")):
        h = pd.read_csv(path)
        h = h[h.domain.isin(DOMAINS) & h.fact.notna()].copy()
        h["truth"] = h.manual.map({"up": 1, "down": -1, "none": 0}).astype(int)
        for q in ("change", "level"):
            hh = h[h.domain == "Health_US"] if q == "level" else h
            hh = hh.assign(pred=[int(key.loc[(d, q, t.strip()), "literal"])
                                 for d, t in zip(hh.domain, hh.fact)])
            for dom, g in [*hh.groupby("domain", sort=False), ("pooled", hh)]:
                if q == "level" and dom == "pooled":
                    continue
                rows.append(dict(label_set=set_name, question=q, domain=dom,
                                 role="descriptive (labels are change claims)" if q == "level"
                                 else ("supplementary" if set_name != "main_50" else "primary"),
                                 **_agreement(g.truth, g.pred)))
            conf_rows.append(_confusion(hh.assign(label_set=set_name, question=q),
                                        ["label_set", "question", "domain"], "truth", "pred"))
    return pd.DataFrame(rows), pd.concat(conf_rows, ignore_index=True)


def judge_known(j: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(ii) Judge vs P3's **known** ``claimed_direction`` on every attacked fake (val + test).

    The rules wrote these texts, so the direction each one claims is known exactly (about the
    forecast series, Economy already mapped). Compared with the judge's series ``direction``.
    Health_US: the level-variant fakes are matched to the **level** answer, the change-variant
    fakes to the **change** answer (pre-registered); the *other* Health_US question is also
    shown as ``cross_question`` (descriptive — both claims are edited in many of these texts).
    """
    key = j.set_index(["domain", "question", "text"]).direction
    rows, frames = [], []
    for cond in ("flipped", "exaggerated"):
        t = common.load(cond)
        t = t[t.attacked & t.domain.isin(DOMAINS)].copy()
        t["truth"] = t.claimed_direction.astype(int)
        for q in ("change", "level"):
            sel = t if q == "change" else t[t.domain == "Health_US"]
            sel = sel.assign(question=q, condition=cond,
                             pred=[int(key.loc[(d, q, x.strip())]) for d, x in zip(sel.domain, sel.text)])
            sel["role"] = np.where((sel.domain != "Health_US") | (sel.variant == q),
                                   "pre-registered", "cross_question")
            frames.append(sel)
    df = pd.concat(frames, ignore_index=True)
    for (cond, role, q), g in df.groupby(["condition", "role", "question"], sort=False):
        for keys, gg in [*g.groupby(["domain", "variant", "split"], sort=False),
                         *(((d, v, "val+test"), gg) for (d, v), gg in g.groupby(["domain", "variant"], sort=False)),
                         (("pooled", "all", "val+test"), g)]:
            rows.append(dict(condition=cond, role=role, question=q, domain=keys[0],
                             variant=keys[1], split=keys[2], **_agreement(gg.truth, gg.pred)))
    conf = _confusion(df[df.role == "pre-registered"], ["condition", "domain", "variant", "question"],
                      "truth", "pred")
    return pd.DataFrame(rows), conf


def judge_coverage(j: pd.DataFrame) -> pd.DataFrame:
    """(iii) none-rate on ``cross_domain`` texts, and coverage = share of **genuine** texts
    with a non-none claim (any question), per domain × split."""
    key = j.set_index(["domain", "question", "text"]).direction
    rows = []
    for cond in ("cross_domain", "real"):
        t = common.load(cond)
        t = t[t.has_text & t.domain.isin(DOMAINS)]
        for (dom, split), g in t.groupby(["domain", "split"], sort=False):
            qs = ["level", "change"] if dom == "Health_US" else ["change"]
            ans = np.array([[int(key.loc[(dom, q, x.strip())]) for q in qs] for x in g.text])
            for i, q in enumerate(qs):
                rows.append(dict(condition=cond, domain=dom, split=split, question=q,
                                 n=len(g), none_rate=float((ans[:, i] == 0).mean())))
            rows.append(dict(condition=cond, domain=dom, split=split, question="any",
                             n=len(g), none_rate=float((ans == 0).all(1).mean())))
    out = pd.DataFrame(rows)
    out["claim_rate"] = 1 - out.none_rate
    return out


def run_judge(save: bool = True) -> dict[str, pd.DataFrame]:
    j = load_judgements()
    if not j.probe_ok.all():
        raise ValueError("judge reproducibility probe failed")
    hum, hum_conf = judge_human(j)
    known, known_conf = judge_known(j)
    out = dict(judge_human=hum, judge_human_confusion=hum_conf, judge_known=known,
               judge_known_confusion=known_conf, judge_coverage=judge_coverage(j))
    if save:
        for k, v in out.items():
            _save(v, k)
    return out


# =============================================================================================
# Gate table — model-independent: (judge claims, lookback numbers, σ) → g
# =============================================================================================

def lookbacks() -> pd.DataFrame:
    """Lookback windows in raw units (the numbers TextGuard sees), keyed by window."""
    q = pd.read_csv(QWEN_INPUTS, dtype={"sample_id": str}, keep_default_na=False)
    q["numbers"] = [np.asarray(json.loads(x), float) for x in q.lookback]
    return q.set_index(["domain", "split", "sample_id"])[["numbers", "train_std"]]


def oracle_judge(domain: str, cond_texts: pd.DataFrame):
    """Ablation only: claims from P3's **known** directions instead of Qwen's (c = 1).

    The rules know what each text claims: a fake claims ``claimed_direction``, the real report
    of an attacked window claims ``real_direction`` (read by the same rules); the claim's kind
    is the attack variant (Health_US level / change). A text the rules did not read → no claim.
    Uses the rules that wrote the fakes, so it is an upper bound, never a headline.
    """
    known = {}
    for _, r in cond_texts.iterrows():
        d = int(r.direction) if pd.notna(r.direction) else 0
        if d and isinstance(r.text, str) and r.text.strip():
            known[r.text.strip()] = (r.variant or "change", d)

    def judge(text):
        if not isinstance(text, str) or not text.strip() or text.strip() not in known:
            return []
        kind, d = known[text.strip()]
        return [Claim(kind, d, 1.0, stated_baseline(text) if kind == "level" else None)]
    return judge


def _oracle_texts(cond: str) -> pd.DataFrame:
    """Per window: the text of ``cond`` and the direction the rules know it claims."""
    fl = common.load("flipped").set_index(["domain", "split", "sample_id"])
    t = common.load(cond).set_index(["domain", "split", "sample_id"])
    if cond == "real":   # the real report's direction as the rules read it (P3)
        d = fl.real_direction.where(fl.attacked, 0)
    elif cond in ("flipped", "exaggerated"):
        d = t.claimed_direction.where(t.attacked, 0)
    else:
        d = pd.Series(0, index=t.index)
    return t.assign(direction=d, variant=fl.variant.replace("", "change")).reset_index()


def gate_table(split: str, conditions=COND_P6, fractions=FRACTIONS, judge_kind: str = "qwen",
               j: pd.DataFrame | None = None) -> pd.DataFrame:
    """g for every window × condition × window fraction, with each branch's gate.

    No text → no claims → g = 1. Health_US texts carry two claims (level + change);
    g = the minimum. Columns g_change / g_level are the per-branch gates (NaN = no such claim),
    c_* / dir_* the judge's confidence and direction, ev_* the evidence direction.
    """
    j = load_judgements() if (j is None and judge_kind == "qwen") else j
    lb = lookbacks()
    judges = {d: QwenJudge(d, data=j) for d in DOMAINS} if judge_kind == "qwen" else None
    rows = []
    for cond in conditions:
        t = common.load(cond) if judge_kind == "qwen" else _oracle_texts(cond)
        t = t[(t.split == split) & t.domain.isin(DOMAINS)]
        for dom, td in t.groupby("domain", sort=False):
            judge = judges[dom] if judges else oracle_judge(dom, td)
            L = lookback_for(dom)
            for _, r in td.iterrows():
                nums, sigma = lb.loc[(dom, split, r.sample_id)]
                text = r.text if isinstance(r.text, str) and r.text.strip() else None
                claims = judge(text) if text else []
                for f in fractions:
                    w = window_for(L, f)
                    res = tg_gate(claims, nums, float(sigma), w)
                    row = dict(domain=dom, split=split, sample_id=r.sample_id, condition=cond,
                               fraction=f, window=w, has_text=text is not None, g=res.g)
                    for kind in ("change", "level"):
                        ck = [(c, e, gg) for c, e, gg in res.checks if c.kind == kind]
                        c, e, gg = ck[0] if ck else (None, None, np.nan)
                        row.update({f"g_{kind}": gg,
                                    f"dir_{kind}": c.direction if c else np.nan,
                                    f"c_{kind}": c.confidence if c else np.nan,
                                    f"ev_{kind}": e.direction if e else np.nan,
                                    f"s_{kind}": e.strength if e else np.nan})
                    rows.append(row)
    return pd.DataFrame(rows)


def _check_y_last(x: pd.DataFrame, split: str) -> None:
    """The gate's numbers must be the forecaster's lookback: last value == y_last."""
    lb = lookbacks()
    last = np.array([lb.loc[(d, split, s), "numbers"][-1] for d, s in zip(x.domain, x.sample_id)])
    if not np.allclose(last, x.y_last.to_numpy(float)):
        raise ValueError("lookback windows do not match the predictions' y_last")


def level_only(gt: pd.DataFrame) -> pd.Series:
    """TextGuard-L (Amendment 2, exploratory): Health_US gated by its level claim alone.

    g = min over branches, so dropping the change branch leaves the level gate (no level
    claim → 1). Other domains are unchanged.
    """
    return gt.g.where(gt.domain != "Health_US", gt.g_level.fillna(1.0))


def gates_wide(gt: pd.DataFrame, w: pd.DataFrame, fraction: float,
               conditions=("real", "flipped", "exaggerated"), col: str = "g") -> dict[str, np.ndarray]:
    """Gate column ``col`` per condition at one window fraction, aligned to ``wide`` rows."""
    keys = ["domain", "sample_id"]
    g = gt[gt.fraction == fraction].pivot(index=keys, columns="condition", values=col)
    idx = w.set_index(keys).index
    out = {c: g[c].reindex(idx).to_numpy() for c in conditions}
    for c, v in out.items():
        if np.isnan(v).any():
            raise ValueError(f"gate missing for some {c} windows")
    return out


# =============================================================================================
# Defences on saved predictions
# =============================================================================================

def defend(x: pd.DataFrame, defence: str, g: dict | None = None, tau: float | None = None,
           k: float | None = None, conditions=("real", "flipped", "exaggerated")) -> dict:
    """{condition: Defended} for one model's wide frame. A = the ``empty`` forecast."""
    A = x["empty"].to_numpy(float)
    out = {}
    for c in conditions:
        B = x[c].to_numpy(float)
        if defence == "none":           # the raw model, no NaN fallback (P5's numbers)
            out[c] = D.Defended(B, np.ones(len(B)), np.zeros(len(B), bool))
        elif defence == "always_trust":
            out[c] = D.always_trust(A, B)
        elif defence == "never_trust":
            out[c] = D.never_trust(A, B)
        elif defence == "clip":
            out[c] = D.clip(A, B, x.sigma.to_numpy(float), k)
        elif defence == "textguard":
            out[c] = D.textguard(A, B, g[c], threshold=tau, mode="block")
        else:
            raise ValueError(defence)
    return out


def _metric_terms(x: pd.DataFrame, dfd: dict, attack: str, base_mask=None) -> dict:
    """Per-window terms of every P6 quantity for one model × defence × attack."""
    has = x.has_text.to_numpy()
    mask = has if base_mask is None else has & base_mask
    d = x[f"d_{attack}"].to_numpy()
    fr, ff = dfd["real"].final, dfd[attack].final
    dss = M.dss(ff, fr, d, x.sigma, mask)
    flip = M.flip_rate(ff, fr, x.y_last, d, mask)
    fbr = M.false_block_rate(dfd["real"].weight, mask)
    pm = M.paired_mse(x.y_true, {"real": fr}, x.sigma, mask)["real"]
    att = mask & (d != 0)
    pma = M.paired_mse(x.y_true, {"real": fr, "fake": ff}, x.sigma, att)
    return dict(dss=dss.terms, flip=flip.terms, fbr=fbr.terms, mse_real=pm.terms,
                mse_real_att=pma["real"].terms, mse_fake_att=pma["fake"].terms,
                damage=M.paired_difference(pma["fake"], pma["real"]))


def _summ(terms: dict, x: pd.DataFrame, groups=GROUPS_MAIN, ci: bool = True) -> list[dict]:
    dom = x.domain.to_numpy()
    rows = []
    for gname, sel in _groups(x, groups):
        row = dict(domain=gname)
        for q, t in terms.items():
            if ci:
                c = _cell(t, dom, sel)
                row.update({q: c["value"], f"{q}_lo": c["ci_low"], f"{q}_hi": c["ci_high"],
                            f"n_{q}": c["n"]})
            else:
                tt = t[sel & np.isfinite(t)]
                row.update({q: float(tt.mean()) if tt.size else np.nan, f"n_{q}": tt.size})
        rows.append(row)
    return rows


# =============================================================================================
# 2. Tuning — val only
# =============================================================================================

def _select(df: pd.DataFrame, knob_cols: list[str]) -> tuple[pd.Series, bool]:
    """The pre-registered rule (§10b): among FBR ≤ 10%, the lowest DSS; ties → lower FBR, then
    the knobs in order (smaller τ, then smaller w). None within budget → the lowest-FBR config;
    a tie there is broken by the same explicit knob order (Amendment 2, item 2) — **not** by
    DSS, which would be a selection criterion added after seeing val."""
    ok = df[df.fbr <= FBR_BUDGET]
    if len(ok):
        return ok.sort_values(["dss", "fbr", *knob_cols], kind="mergesort").iloc[0], True
    return df.sort_values(["fbr", *knob_cols], kind="mergesort").iloc[0], False


def run_tune(save: bool = True) -> dict[str, pd.DataFrame]:
    """Val grid for TextGuard (f × τ) and clip (k) on Model 5 / flipped, pooled → freeze."""
    wv = wide(load_long("val"), "val")
    x = wv[QWEN]
    _check_y_last(x, "val")
    gt = gate_table("val")
    rows = []
    for f in FRACTIONS:
        g = gates_wide(gt, x, f)
        for tau in TAUS:
            t = _metric_terms(x, defend(x, "textguard", g, tau), "flipped")
            r = _summ({"dss": t["dss"], "flip": t["flip"], "fbr": t["fbr"]}, x,
                      {"pooled": tuple(DOMAINS), **{d: (d,) for d in DOMAINS}}, ci=False)
            for rr in r:
                rows.append(dict(defence="textguard", fraction=f, tau=tau, k=np.nan, **rr))
    for k in CLIP_KS:
        t = _metric_terms(x, defend(x, "clip", k=k), "flipped")
        for rr in _summ({"dss": t["dss"], "flip": t["flip"], "fbr": t["fbr"]}, x,
                        {"pooled": tuple(DOMAINS), **{d: (d,) for d in DOMAINS}}, ci=False):
            rows.append(dict(defence="clip", fraction=np.nan, tau=np.nan, k=k, **rr))
    for dfn in ("always_trust", "never_trust", "none"):
        t = _metric_terms(x, defend(x, dfn), "flipped")
        for rr in _summ({"dss": t["dss"], "flip": t["flip"], "fbr": t["fbr"]}, x,
                        {"pooled": tuple(DOMAINS), **{d: (d,) for d in DOMAINS}}, ci=False):
            rows.append(dict(defence=dfn, fraction=np.nan, tau=np.nan, k=np.nan, **rr))
    grid = pd.DataFrame(rows)
    pooled = grid[grid.domain == "pooled"]
    tg = pooled[pooled.defence == "textguard"]
    cl = pooled[pooled.defence == "clip"]
    tg_pick, tg_ok = _select(tg, ["tau", "fraction"])
    cl_pick, cl_ok = _select(cl, ["k"])
    # is "lowest DSS" ambiguous here? only if an in-budget config pushes *away* from the lie
    neg = tg[(tg.fbr <= FBR_BUDGET) & (tg.dss < 0)]
    frozen = dict(
        fixed_on="val only (Model 5, flipped, pooled)", n_configs_textguard=len(tg),
        n_configs_clip=len(cl), fbr_budget=FBR_BUDGET,
        textguard=dict(fraction=float(tg_pick.fraction), tau=float(tg_pick.tau),
                       window={d: window_for(lookback_for(d), float(tg_pick.fraction))
                               for d in DOMAINS},
                       budget_met=bool(tg_ok), val_dss=float(tg_pick.dss),
                       val_flip=float(tg_pick.flip), val_fbr=float(tg_pick.fbr)),
        clip=dict(k=float(cl_pick.k), budget_met=bool(cl_ok), val_dss=float(cl_pick.dss),
                  val_flip=float(cl_pick.flip), val_fbr=float(cl_pick.fbr)),
        negative_dss_in_budget=int(len(neg)),
        amendment="Amendment 2 (commit 3fbce69): fallback tie -> smaller tau, then smaller w; "
                  "tau = 0.5 exploratory; TextGuard-L (Health_US level claim only) exploratory",
        exploratory=dict(tau_block=0.5, textguard_L="Health_US: level claim only"))
    if save:
        _save(grid, "val_grid")
        gt.to_csv(RESULTS / "p6_gates_val.csv", index=False)   # raw, gitignored
        FROZEN.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    return dict(val_grid=grid, frozen=pd.json_normalize(frozen))


# =============================================================================================
# 3. Test — once, with the frozen settings
# =============================================================================================

ATTACKS_P6 = ("flipped", "exaggerated")
GROUPS_P6 = {**GROUPS_MAIN, "change-flip (Eco+SG+Tra)": CHANGE_FLIP}
QUANTS = ("dss", "flip", "fbr", "mse_real", "d_mse_vs_never", "d_mse_vs_undefended",
          "damage", "mse_real_att", "mse_fake_att")


def load_frozen() -> dict:
    """The val-chosen settings. The test stage refuses to run without them."""
    if not FROZEN.exists():
        raise SystemExit(f"{FROZEN.name} missing — run `python -m src.analyze_p6 tune` first "
                         "and record the frozen settings in PROJECT_STATUS before testing")
    return json.loads(FROZEN.read_text(encoding="utf-8"))


def _specs(fz: dict, g: dict, gL: dict, gO: dict) -> list[tuple[str, str, dict]]:
    """(label, role, defend-kwargs) for the operating-point table."""
    tau, k = fz["textguard"]["tau"], fz["clip"]["k"]
    return [
        ("none", "reference (raw model, P5 NaN pairing)", dict(defence="none")),
        ("always_trust", "baseline", dict(defence="always_trust")),
        ("never_trust", "baseline", dict(defence="never_trust")),
        (f"clip k={k:g}", "baseline (val-chosen)", dict(defence="clip", k=k)),
        (f"TextGuard τ={tau:g}", "MAIN (val-chosen)", dict(defence="textguard", g=g, tau=tau)),
        ("TextGuard block τ=0.5", "exploratory (Amendment 2)",
         dict(defence="textguard", g=g, tau=fz["exploratory"]["tau_block"])),
        (f"TextGuard-L τ={tau:g}", "exploratory (Amendment 2)",
         dict(defence="textguard", g=gL, tau=tau)),
        (f"TextGuard oracle-judge τ={tau:g}", "ablation (oracle judge, upper bound)",
         dict(defence="textguard", g=gO, tau=tau)),
    ]


def _all_terms(x: pd.DataFrame, specs, attack: str, base_mask=None) -> dict[str, dict]:
    """Per-window terms for every defence, plus ΔMSE vs never_trust / the undefended model."""
    terms = {lab: _metric_terms(x, defend(x, **kw), attack, base_mask) for lab, _, kw in specs}
    nt, at = terms["never_trust"]["mse_real"], terms["always_trust"]["mse_real"]
    for t in terms.values():
        t["d_mse_vs_never"] = t["mse_real"] - nt
        t["d_mse_vs_undefended"] = t["mse_real"] - at
    return terms


def operating_point(w: dict, fz: dict, gates: dict, base_mask=None,
                    models=TEXT_MODELS_P6) -> pd.DataFrame:
    """Every model × attack × defence at the frozen settings: DSS, Flip Rate, FBR, real-text
    MSE (and Δ vs never_trust / undefended), damage — pooled, per domain, change-flip pooled."""
    rows = []
    for model in models:
        x = w[model]
        specs = _specs(fz, *(gates[v] for v in ("g", "gL", "gO")))
        for attack in ATTACKS_P6:
            terms = _all_terms(x, specs, attack, base_mask)
            for lab, role, _ in specs:
                t = terms[lab]
                for r in _summ({q: t[q] for q in QUANTS}, x, GROUPS_P6):
                    rows.append(dict(model=model, short=SHORT[model], attack=attack,
                                     defence=lab, role=role, **r))
    return pd.DataFrame(rows)


def curve(w: dict, fz: dict, gates: dict) -> pd.DataFrame:
    """The trade-off curve: TextGuard (and TextGuard-L) at the frozen w over every τ, clip
    over every k, and the two endpoints. (FBR, DSS) and (FBR, Flip Rate), CIs, per domain."""
    rows = []
    for model in TEXT_MODELS_P6:
        x = w[model]
        pts = [("always_trust", np.nan, dict(defence="always_trust")),
               ("never_trust", np.nan, dict(defence="never_trust"))]
        pts += [("clip", k, dict(defence="clip", k=k)) for k in CLIP_KS]
        pts += [("TextGuard", t, dict(defence="textguard", g=gates["g"], tau=t)) for t in TAUS]
        pts += [("TextGuard-L (exploratory)", t, dict(defence="textguard", g=gates["gL"], tau=t))
                for t in TAUS]
        for attack in ATTACKS_P6:
            for fam, knob, kw in pts:
                t = _metric_terms(x, defend(x, **kw), attack)
                ci = model == QWEN and attack == "flipped"   # CIs on the primary curve only
                for r in _summ({q: t[q] for q in ("dss", "flip", "fbr")}, x, GROUPS_MAIN, ci=ci):
                    chosen = ((fam == "TextGuard" and knob == fz["textguard"]["tau"])
                              or (fam == "clip" and knob == fz["clip"]["k"]))
                    rows.append(dict(model=model, short=SHORT[model], attack=attack,
                                     family=fam, knob=knob, val_chosen=chosen, **r))
    return pd.DataFrame(rows)


def _mcnemar(f_a: np.ndarray, f_b: np.ndarray, sel: np.ndarray) -> dict:
    """Exact McNemar on windows flipped under exactly one of two defences (two-sided)."""
    from scipy.stats import binomtest
    ok = sel & np.isfinite(f_a) & np.isfinite(f_b)
    only_a = int((ok & (f_a == 1) & (f_b == 0)).sum())
    only_b = int((ok & (f_a == 0) & (f_b == 1)).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return dict(only_first=only_a, only_second=only_b, p=float(p))


def hypothesis_tests(x: pd.DataFrame, terms: dict, tg: str, clip_lab: str,
                     groups=GROUPS_MAIN) -> pd.DataFrame:
    """T1–T4 (§10b) for one TextGuard variant: estimate (TextGuard − comparator), paired CI,
    raw p. ``supports`` = the sign the claim needs (T1, T3: TextGuard lower DSS; T2: fewer
    flips under TextGuard; T4 has no directional claim)."""
    dom = x.domain.to_numpy()
    T, A, C, N = terms[tg], terms["always_trust"], terms[clip_lab], terms["never_trust"]
    rows = []
    for gname, sel in _groups(x, groups):
        for block, diff in (("T1", T["dss"] - A["dss"]), ("T3", T["dss"] - C["dss"]),
                            ("T4", T["mse_real"] - N["mse_real"])):
            c = _cell(diff, dom, sel)
            rows.append(dict(block=block, domain=gname, estimate=c["value"],
                             ci_low=c["ci_low"], ci_high=c["ci_high"], n=c["n"],
                             p_raw=M.wilcoxon(np.where(sel, diff, np.nan))["p"],
                             sign_ok=bool(c["value"] < 0) if block != "T4" else np.nan))
        mc = _mcnemar(A["flip"], T["flip"], sel)      # only_first = flipped only undefended
        ft, fa = _cell(T["flip"], dom, sel), _cell(A["flip"], dom, sel)
        rows.append(dict(block="T2", domain=gname, estimate=ft["value"] - fa["value"],
                         ci_low=np.nan, ci_high=np.nan, n=ft["n"], p_raw=mc["p"],
                         flips_only_always_trust=mc["only_first"],
                         flips_only_textguard=mc["only_second"],
                         sign_ok=mc["only_first"] > mc["only_second"]))
    out = pd.DataFrame(rows)
    out["comparison"] = out.block.map({"T1": "DSS: TextGuard − always_trust",
                                       "T2": "Flip: TextGuard vs always_trust (McNemar)",
                                       "T3": f"DSS: TextGuard − {clip_lab}",
                                       "T4": "real-text sq. error: TextGuard − never_trust"})
    return out


def holm_family_p6(tests: pd.DataFrame) -> pd.DataFrame:
    """The pre-registered family: T1–T4 × (pooled + 4 domains) = 20. Asserts the count."""
    fam = tests[tests.domain.isin(GROUPS_MAIN)].copy()
    counts = fam.block.value_counts().to_dict()
    if len(fam) != HOLM_M or counts != {b: 5 for b in ("T1", "T2", "T3", "T4")} \
            or fam.p_raw.isna().any():
        raise ValueError(f"family is not the pre-registered {HOLM_M} tests: {counts}")
    fam["p_holm"] = M.holm(fam.p_raw)
    fam["reject"] = fam.p_holm < 0.05
    fam["supports_claim"] = fam.reject & (fam.sign_ok == True)  # noqa: E712
    return fam


def gate_distribution(gt: pd.DataFrame, f: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gate values on genuine texts vs attacked fakes: quantiles + a 0.05-bin histogram
    (the figure is drawn from the histogram, so it is derivable from committed files)."""
    t = gt[gt.fraction == f].copy()
    t["gL"] = level_only(t)
    att = {c: common.load(c).set_index(["domain", "split", "sample_id"]).attacked
           for c in ATTACKS_P6}
    keep = []
    for c, g in t.groupby("condition", sort=False):
        if c == "real":
            keep.append(g[g.has_text].assign(set="genuine (real)"))
        elif c in ATTACKS_P6:
            a = att[c].reindex(pd.MultiIndex.from_frame(g[["domain", "split", "sample_id"]]))
            keep.append(g[a.to_numpy(bool)].assign(set=f"{c} (attacked)"))
    t = pd.concat(keep)
    q, h = [], []
    bins = np.linspace(0, 1, 21)
    for col, variant in (("g", "TextGuard"), ("gL", "TextGuard-L (exploratory)")):
        for (s, dom), g in [*t.groupby(["set", "domain"], sort=False),
                            *(((s, "pooled"), g) for s, g in t.groupby("set", sort=False))]:
            v = g[col].to_numpy()
            q.append(dict(variant=variant, set=s, domain=dom, n=v.size, mean=v.mean(),
                          p10=np.quantile(v, .1), p25=np.quantile(v, .25),
                          median=np.median(v), p75=np.quantile(v, .75),
                          share_g1=float((v == 1).mean()), share_lt_05=float((v < .5).mean())))
            counts = np.histogram(v[v < 1], bins=bins)[0]
            for lo, hi, n in zip(bins[:-1], bins[1:], counts):
                h.append(dict(variant=variant, set=s, domain=dom, bin_lo=lo, bin_hi=hi, count=int(n)))
            h.append(dict(variant=variant, set=s, domain=dom, bin_lo=1.0, bin_hi=1.0,
                          count=int((v == 1).sum())))
    return pd.DataFrame(q), pd.DataFrame(h)


def covid_fbr(gt: pd.DataFrame, f: float) -> pd.DataFrame:
    """Issue #10: FBR and gate values on genuine test texts, 2020 windows vs the rest."""
    t = gt[(gt.fraction == f) & (gt.condition == "real") & gt.has_text
           & gt.domain.isin(CHANGE_FLIP)].copy()
    t["period"] = np.where(t.sample_id.str[:4] == "2020", "2020", "not 2020")
    rows = []
    for (dom, per), g in [*t.groupby(["domain", "period"]),
                          *((("pooled (Eco+SG+Tra)", p), g) for p, g in t.groupby("period"))]:
        b = (g.g < 0.5).astype(float).to_numpy()
        lo, hi = M.bootstrap_ci(b, N_BOOT, seed=BOOT_SEED,
                                strata=g.domain.to_numpy() if dom.startswith("pooled") else None)
        rows.append(dict(domain=dom, period=per, n=len(g), fbr=b.mean(), fbr_lo=lo, fbr_hi=hi,
                         mean_g=g.g.mean(), share_g_lt_1=float((g.g < 1).mean())))
    return pd.DataFrame(rows)


def dummy_check() -> pd.DataFrame:
    """Model-agnostic check: the PHASES worked example on ``DummyForecaster``."""
    from src.models.dummy import DummyForecaster
    from src.textguard.text_direction import fixed_judge
    from src.textguard.textguard import TextGuard
    X = [100.0, 104.0, 108.0, 112.0]
    guard = TextGuard(DummyForecaster(), fixed_judge(Claim("change", -1, 0.9)), sigma=10.0,
                      window=4, threshold=0.5)
    d = guard.decide(X, "Demand will fall sharply.")
    return pd.DataFrame([dict(model="DummyForecaster", A=float(d.A[0]), B=float(d.B[0]),
                              g=d.g, final=float(d.final[0]), warn=d.warn,
                              expected="A 116, B 106, g 0.1, final 115, warn",
                              ok=bool(np.isclose(d.g, 0.1) and np.isclose(d.final[0], 115)
                                      and d.warn))])


def run_test(save: bool = True, split: str = "test") -> dict[str, pd.DataFrame]:
    """``split="val"`` is a dry run to debug this code without touching test (never saved)."""
    fz = load_frozen()
    f = fz["textguard"]["fraction"]
    save = save and split == "test"
    w = wide(load_long(split), split)
    _check_y_last(w[QWEN], split)
    gt = gate_table(split, fractions=(f,))
    gt["gL"] = level_only(gt)
    go = gate_table(split, conditions=("real",) + ATTACKS_P6, fractions=(f,), judge_kind="oracle")
    x5 = w[QWEN]
    gates = dict(g=gates_wide(gt, x5, f), gL=gates_wide(gt, x5, f, col="gL"),
                 gO=gates_wide(go, x5, f))
    op = operating_point(w, fz, gates)
    # the pre-registered family: Model 5, flipped, frozen operating point
    specs = _specs(fz, gates["g"], gates["gL"], gates["gO"])
    tg, cl = specs[4][0], specs[3][0]
    terms5 = _all_terms(x5, specs, "flipped")
    fam = holm_family_p6(hypothesis_tests(x5, terms5, tg, cl))
    # ablations / exploratory — raw p, outside the family
    keep = year_mask(x5, 2020)
    abl = [hypothesis_tests(x5, terms5, tg, cl, {"change-flip (Eco+SG+Tra)": CHANGE_FLIP})
           .assign(analysis="Health_US vs change-flip domains")]
    abl.append(hypothesis_tests(x5, _all_terms(x5, specs, "flipped", keep), tg, cl)
               .assign(analysis="excluding 2020"))
    for lab in (specs[5][0], specs[6][0], specs[7][0]):
        abl.append(hypothesis_tests(x5, terms5, lab, cl).assign(analysis=lab))
    abl = pd.concat(abl, ignore_index=True)
    op_no2020 = operating_point(w, fz, gates, keep, models=(QWEN,))
    dist, hist = gate_distribution(gt, f)
    out = dict(operating=op, holm=fam, ablation_tests=abl, operating_no2020=op_no2020,
               curve=curve(w, fz, gates), gate_dist=dist, gate_hist=hist,
               covid=covid_fbr(gt, f), dummy=dummy_check())
    if save:
        gt.to_csv(RESULTS / "p6_gates_test.csv", index=False)       # raw, gitignored
        for k, v in out.items():
            _save(v, k)
    return out


# =============================================================================================
# CLI
# =============================================================================================

def _print(out: dict) -> None:
    with pd.option_context("display.width", 250, "display.max_rows", 500,
                           "display.max_columns", 40, "display.float_format", "{:.4g}".format):
        for name, df in out.items():
            print(f"\n=== summary_p6_{name}.csv ({len(df)} rows) ===")
            print(df.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["judge", "tune", "test"])
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--dry-run-val", action="store_true",
                    help="run the test stage on val (debugging; nothing saved)")
    args = ap.parse_args()
    if args.stage == "test" and args.dry_run_val:
        _print(run_test(save=False, split="val"))
        return
    fn = {"judge": run_judge, "tune": run_tune, "test": run_test}[args.stage]
    _print(fn(save=not args.no_save))


if __name__ == "__main__":
    main()
