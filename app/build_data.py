"""P7.2 — build the Streamlit demo's data from saved results. No model or judge is run.

Run (project venv, data downloaded): ``python -m app.build_data`` →

* ``app/data/demo.json`` — committed: headline numbers, results tables, and the examples'
  numbers (forecasts, gates, per-claim checks) and FNSPID headlines;
* ``app/data/demo_timemmd_texts.json`` — **gitignored**: the report texts of the Time-MMD examples
  (Energy, SocialGood). Time-MMD states no licence, so the repository does not redistribute its
  texts (DATA_CARD.md §1); the app shows a notice instead when this file is missing;
* ``app/figures/*.png`` — copies of the paper figures the results screen reuses.

**Why precomputed.** Model 5 and the judge need a GPU; the demo only *shows* results that already
exist. Every forecast is a saved Model 5 answer (P5: ``results/qwen/p5_qwen_predictions.csv``; R4:
the verified Kaggle run), every gate comes from the saved judge answers through the same functions
the analyses use, at the frozen settings (``summary_p6_frozen.json``, ``summary_r4_v2_frozen.json``).
Nothing is tuned and no aggregate number is computed here: headline numbers are read from the
``results/summary_*`` tables.

**TextGuard v2 is shown only on the held-out domains (Energy, FNSPID).** v2 was developed on the
original domains' val split and is never evaluated on their test split (PROJECT_STATUS §1a, "Rules
a fresh session must not break"), so the SocialGood example shows v1 only.

**How the examples are chosen (fixed rules, not hand-picked).** Each slot filters windows by a
criterion, sorts the candidates by one number, and takes the **median** candidate (index
(n − 1) // 2; ties by series, then date) — a typical case of its kind, not the most dramatic one.
See :data:`SLOTS`; the rule text is shown in the app next to each example.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import analyze_p5 as P5
from src import analyze_p6 as P6
from src import analyze_r4 as R
from src import metrics as M
from src.attacks import common
from src.data.windows import lookback_for
from src.textguard import v2 as V
from src.textguard.text_direction import QwenJudge, load_judgements
from src.textguard.textguard import gate as tg_gate
from src.textguard.trend_detector import window_for
from src.textguard.units import units

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
OUT = APP / "data" / "demo.json"
OUT_TEXTS = APP / "data" / "demo_timemmd_texts.json"
FIG_DIR = APP / "figures"
RESULTS = ROOT / "results"
FIGURES = ("p5_dss", "p5_flip_rate", "p6_tradeoff", "r3_dss", "r4_dss", "r4_fbr")
WARN = 0.5                 # applied weight < 0.5 = the text was mostly discarded (the FBR definition)
P5_DOMAIN = "SocialGood"   # see SLOTS["p5_fooled"]

SLOTS = {
    "energy_flip": dict(
        title="Energy — a clear flip",
        rule="R4, Energy, `flipped`, Model 5. Candidates: attacked windows (scan exclusions removed) "
             "where the undefended forecast moves toward the lie (DSS > 0) and its up/down direction "
             "reverses (Flip = 1). Picked: the median-DSS candidate."),
    "fnspid_a2": dict(
        title="FNSPID — A2: v2 catches what v1 misses",
        rule="R4, FNSPID, `a2_forward`, Model 5. Candidates: attacked windows with DSS > 0 where "
             "TextGuard v1 lets the fake through (applied weight ≥ 0.5) and v2 does not (< 0.5), "
             "while v2 lets the same window's **genuine** report through (≥ 0.5) — so the block "
             "comes from the added sentence, not from a genuine one. Picked: the median-DSS "
             "candidate."),
    "energy_a2": dict(
        title="Energy — A2: v2 catches what v1 misses",
        rule="Same rule as the FNSPID A2 example (v2 blocks the fake but not the genuine report; "
             "v1 lets the fake through), on Energy."),
    "p5_fooled": dict(
        title=f"{P5_DOMAIN} (original Time-MMD) — Model 5 fooled in P5",
        rule=f"P5 test split, `flipped`, Model 5. Domain: the change-flip domain with the largest "
             f"pooled P5 DSS ({P5_DOMAIN}; Health_US is excluded because its level flip also edits "
             f"a number). Candidates: attacked windows with DSS > 0 and Flip = 1. Picked: the "
             f"median-DSS candidate. TextGuard v1 at its frozen P6 settings; v2 is not shown "
             f"(never evaluated on this test split, by rule)."),
    "fnspid_fbr": dict(
        title="FNSPID — honest failure: v2 blocks a genuine report",
        rule="R4, FNSPID, Model 5, the **genuine** report. Candidates: windows where v2 blocks the "
             "genuine report (applied weight < 0.5 — a False Block) and blocking makes the forecast "
             "worse than the undefended one, among windows that also have an attacked fake. Picked: "
             "the median candidate by extra error (σ). The fake shown is the window's `flipped` "
             "text if it has one, else its A2 text."),
}


def _median_pick(c: pd.DataFrame, by: str) -> pd.Series:
    if c.empty:
        raise ValueError("no candidates")
    c = c.sort_values([by, "series", "sample_id"], kind="mergesort")
    return c.iloc[(len(c) - 1) // 2]


def _f(v) -> float | None:
    v = float(v)
    return None if not np.isfinite(v) else round(v, 6)


# =============================================================================================
# headline numbers + results tables (read from the committed summaries, never recomputed)
# =============================================================================================

def _row(df: pd.DataFrame, **eq) -> pd.Series:
    m = np.ones(len(df), bool)
    for k, v in eq.items():
        m &= (df[k] == v).to_numpy()
    if m.sum() != 1:
        raise ValueError(f"expected one row for {eq}, found {int(m.sum())}")
    return df[m].iloc[0]


def headline() -> dict:
    p5 = pd.read_csv(RESULTS / "summary_p5_attack.csv")
    r3 = pd.read_csv(RESULTS / "summary_r3_operating.csv")
    op = pd.read_csv(RESULTS / "summary_r4_operating.csv")
    fbr = pd.read_csv(RESULTS / "summary_r4_fbr.csv")
    m5 = _row(p5, short="M5", condition="flipped", domain="pooled")
    late = [abs(float(_row(p5, short=s, condition="flipped", domain="pooled").dss)) for s in ("M3", "M4")]
    return dict(
        m5_dss=_f(m5.dss), m5_flip=_f(m5.flip_rate), late_max_abs_dss=_f(max(late)),
        r3_a2_none=_f(_row(r3, target="M5", attack="a2_forward", defence="none", cell="pooled").dss),
        r3_a2_tg1=_f(_row(r3, target="M5", attack="a2_forward", defence="TG", cell="pooled").dss),
        r4_a2_none=_f(_row(op, target="M5", attack="a2_forward", defence="none", cell="pooled").dss),
        r4_a2_tg1=_f(_row(op, target="M5", attack="a2_forward", defence="TG1", cell="pooled").dss),
        r4_a2_tg2=_f(_row(op, target="M5", attack="a2_forward", defence="TG2", cell="pooled").dss),
        r4_fbr_tg1=_f(_row(fbr, target="M5", defence="TG1", cell="pooled").fbr),
        r4_fbr_tg2=_f(_row(fbr, target="M5", defence="TG2", cell="pooled").fbr),
        r4_fbr_tg2_fnspid=_f(_row(fbr, target="M5", defence="TG2", cell="FNSPID").fbr),
    )


def model_table() -> list[dict]:
    """Pooled `flipped` DSS / Flip Rate per model on the original domains (P5, R1, R2)."""
    p5 = pd.read_csv(RESULTS / "summary_p5_attack.csv")
    r1 = pd.read_csv(RESULTS / "summary_r1_attack.csv")
    r2 = pd.read_csv(RESULTS / "summary_r2_attack.csv")
    spec = [  # (table, short, label, how it reads the report, source)
        (p5, "M1", "DLinear (no text)", "does not see text", "P5"),
        (p5, "M2", "PatchTST (no text)", "does not see text", "P5"),
        (p5, "M3", "DLinear + BERT", "late fusion, pooled BERT embedding", "P5"),
        (p5, "M4", "PatchTST + BERT", "late fusion, pooled BERT embedding", "P5"),
        (r1, "M3-Q", "DLinear + Qwen embedding", "late fusion, pooled Qwen embedding", "R1"),
        (r1, "M4-Q", "PatchTST + Qwen embedding", "late fusion, pooled Qwen embedding", "R1"),
        (r1, "M3-F", "DLinear + FinBERT", "late fusion, direction-aware encoder", "R1"),
        (r1, "M4-F", "PatchTST + FinBERT", "late fusion, direction-aware encoder", "R1"),
        (p5, "M5", "Qwen2.5-1.5B (Model 5)", "early fusion: report in the prompt", "P5"),
        (r2, "Q7", "Qwen2.5-7B (4-bit)", "early fusion: report in the prompt", "R2"),
        (r2, "P35", "Phi-3.5-mini", "early fusion: report in the prompt", "R2"),
    ]
    out = []
    for t, s, lab, how, src in spec:
        r = _row(t, short=s, condition="flipped", domain="pooled")
        out.append(dict(model=lab, short=s, reads=how, source=src, dss=_f(r.dss), dss_lo=_f(r.dss_ci_low),
                        dss_hi=_f(r.dss_ci_high), flip=_f(r.flip_rate), n=int(r.n)))
    return out


def tradeoff_table() -> list[dict]:
    """R4 (held-out, Model 5, pooled): DSS on flip / A2 and FBR per defence."""
    op = pd.read_csv(RESULTS / "summary_r4_operating.csv")
    fbr = pd.read_csv(RESULTS / "summary_r4_fbr.csv")
    names = {"none": "No defence", "CL": "clip (k 0.5)", "TG1": "TextGuard v1", "TG2": "TextGuard v2",
             "never_trust": "never_trust (drop all text)"}
    out = []
    for d, lab in names.items():
        out.append(dict(defence=lab,
                        dss_flipped=_f(_row(op, target="M5", attack="flipped", defence=d, cell="pooled").dss),
                        dss_a2=_f(_row(op, target="M5", attack="a2_forward", defence=d, cell="pooled").dss),
                        fbr=_f(_row(fbr, target="M5", defence=d, cell="pooled").fbr)))
    return out


# =============================================================================================
# per-example detail
# =============================================================================================

def _check(c, ev, g) -> dict:
    return dict(kind=c.kind, direction=int(c.direction), confidence=_f(c.confidence),
                evidence=int(ev.direction), strength=_f(ev.strength), g=_f(g),
                frame=getattr(c, "frame", "trend"))


def _v1_detail(claims, nums, sigma, window) -> dict:
    res = tg_gate(claims, nums, sigma, window)
    return dict(g=_f(res.g), checks=[_check(*k) for k in res.checks])


def _v2_detail(key, text, nums, sigma, history) -> tuple[dict, list[str]]:
    """Per unit: the claims v2 checked (above the confidence floor) and the unit's gate."""
    st = R.settings()
    if not text:
        return dict(g=1.0, units=[]), []
    claims = R.v2_claims(key, text)
    res = V.gate(claims, nums, sigma, st["tg2"], history=history, period=st["period"])
    us = units(text)
    rows = []
    for u in us:
        checks = [_check(c, ev, g) for c, ev, g in res.checks if c.unit == u]
        ignored = [dict(kind=c.kind, direction=int(c.direction), confidence=_f(c.confidence))
                   for c in claims if c.unit == u and c.direction != 0 and c.confidence < st["tg2"].conf_floor]
        rows.append(dict(checks=checks, ignored=ignored,
                         g=_f(min((k["g"] for k in checks), default=1.0))))
    return dict(g=_f(res.g), units=rows), us


def _dates(end: pd.Timestamp, n: int, freq: str) -> list[str]:
    step = pd.DateOffset(weeks=1) if freq == "weekly" else pd.DateOffset(months=1)
    return [(end - step * (n - i)).strftime("%Y-%m-%d") for i in range(n)]


R4_DESC = {"Energy": ("US weekly retail gasoline price", "US dollars per gallon"),
           "FNSPID": ("{s} weekly closing share price", "US dollars")}


def r4_examples() -> list[dict]:
    st = R.settings()
    x = R.frame("M5")
    g = R.gates(x)
    fin = R.finals(x, g)
    base = pd.DataFrame(dict(domain=x.domain, series=x.series, sample_id=x.sample_id, i=np.arange(len(x))))
    picks = []
    for slot, dom, a in (("energy_flip", "Energy", "flipped"), ("fnspid_a2", "FNSPID", "a2_forward"),
                         ("energy_a2", "Energy", "a2_forward")):
        t = R.terms(x, fin, a)
        c = base.assign(att=R.attacked(x, a), dss=t["none"]["dss"], flip=t["none"]["flip"],
                        w1=fin["TG1"][a].weight, w2=fin["TG2"][a].weight, w2r=fin["TG2"]["real"].weight)
        c = c[c.att & (c.domain == dom) & (c.dss > 0)]
        c = c[c.flip == 1] if slot == "energy_flip" else c[(c.w1 >= WARN) & (c.w2 < WARN) & (c.w2r >= WARN)]
        picks.append((slot, a, _median_pick(c, "dss"), len(c)))
    # honest failure: v2 blocks a genuine FNSPID report and that costs accuracy
    sig, y = x.sigma.to_numpy(float), x.y_true.to_numpy(float)
    extra = (np.abs(fin["TG2"]["real"].final - y) - np.abs(fin["none"]["real"].final - y)) / sig
    fl, a2 = R.attacked(x, "flipped"), R.attacked(x, "a2_forward")
    c = base.assign(w2=fin["TG2"]["real"].weight, extra=extra, has=x.has_text.to_numpy(), fake=fl | a2)
    c = c[c.has & c.fake & (c.domain == "FNSPID") & (c.w2 < WARN) & (c.extra > 0)]
    r = _median_pick(c, "extra")
    fake = "flipped" if fl[int(r.i)] else "a2_forward"
    picks.append(("fnspid_fbr", fake, r, len(c)))

    out = []
    for slot, a, r, n_cand in picks:
        i = int(r.i)
        w = x.iloc[i]
        nums, sigma, hist = np.asarray(w.lookback, float), float(w.sigma), np.asarray(w.history, float)
        desc, unit = R4_DESC[w.domain]
        texts, unit_txt = {}, {}
        v1, v2, defended = {}, {}, {}
        for cond in ("real", a):
            t = w[f"text_{cond}"]
            texts[cond] = t
            v1[cond] = _v1_detail(R.v1_claims(w.key, t), nums, sigma, st["tg1_window"]) if t else dict(g=1.0, checks=[])
            v2[cond], unit_txt[cond] = _v2_detail(w.key, t, nums, sigma, hist)
            defended[cond] = {d: dict(final=_f(fin[k][cond].final[i]), weight=_f(np.clip(fin[k][cond].weight[i], 0, 1)))
                              for d, k in (("off", "none"), ("v1", "TG1"), ("v2", "TG2"))}
        out.append(dict(
            id=slot, **SLOTS[slot], n_candidates=n_cand, source="R4 (held-out test)",
            domain=w.domain, series=w.series, sample_id=w.sample_id,
            description=desc.format(s=w.series), unit=unit, freq="weekly",
            timemmd=w.domain == "Energy",
            dates=_dates(pd.Timestamp(w.target_date), len(nums), "weekly"), lookback=[_f(v) for v in nums],
            target_date=pd.Timestamp(w.target_date).strftime("%Y-%m-%d"), y_true=_f(w.y_true), sigma=_f(sigma),
            fake_cond=a, synthetic_fake=True, claimed=int(w[f"dall_{a}"]),
            forecast=dict(none=_f(w["none"]), real=_f(w["real"]), fake=_f(w[a])),
            defended=dict(real=defended["real"], fake=defended[a]),
            v1=dict(real=v1["real"], fake=v1[a]), v2=dict(real=v2["real"], fake=v2[a]),
            v2_available=True,
            text=dict(real=texts["real"], fake=texts[a], units=dict(real=unit_txt["real"], fake=unit_txt[a])),
            settings=dict(v1=f"window {st['tg1_window']}, τ {st['tg1_tau']:g} (P6 frozen)",
                          v2=f"window {st['tg2'].window}, c₀ {st['tg2'].conf_floor:g}, τ {st['tg2'].tau:g}, "
                             f"cap {st['tg2'].cap_k:g}σ (R4 frozen)"),
        ))
    return out


def p5_example() -> dict:
    fz = P6.load_frozen()
    f, tau = fz["textguard"]["fraction"], fz["textguard"]["tau"]
    x = P5.wide(P5.load_long("test"), "test")[P5.QWEN]
    gt = P6.gate_table("test", conditions=("real", "flipped"), fractions=(f,))
    gw = P6.gates_wide(gt, x, f, conditions=("real", "flipped"))
    tg = P6.defend(x, "textguard", g=gw, tau=tau, conditions=("real", "flipped"))
    d = x.d_flipped.to_numpy()
    mask = d != 0
    dss = M.dss(x.flipped, x.real, d, x.sigma, mask).terms
    flip = M.flip_rate(x.flipped, x.real, x.y_last, d, mask).terms
    c = pd.DataFrame(dict(domain=x.domain, series=x.domain, sample_id=x.sample_id, dss=dss, flip=flip,
                          i=np.arange(len(x))))
    c = c[(c.domain == P5_DOMAIN) & mask & (c.dss > 0) & (c.flip == 1)]
    r = _median_pick(c, "dss")
    i = int(r.i)
    w = x.iloc[i]
    lb = P6.lookbacks().loc[(P5_DOMAIN, "test", w.sample_id)]
    nums, sigma = np.asarray(lb.numbers, float), float(w.sigma)
    sigma_gate = float(lb.train_std)                   # what P6.gate_table passes to the gate
    q = pd.read_csv(P6.QWEN_INPUTS, dtype={"sample_id": str}, keep_default_na=False)
    q = q[(q.domain == P5_DOMAIN) & (q.split == "test") & (q.sample_id == w.sample_id)].iloc[0]
    texts = {cond: common.load(cond).set_index(["domain", "split", "sample_id"]).loc[(P5_DOMAIN, "test", w.sample_id), "text"]
             for cond in ("real", "flipped")}
    judge = QwenJudge(P5_DOMAIN, data=load_judgements())
    win = window_for(lookback_for(P5_DOMAIN), f)
    v1 = {c_: _v1_detail(judge(texts[c_]), nums, sigma_gate, win) for c_ in ("real", "flipped")}
    for c_ in ("real", "flipped"):
        if abs(v1[c_]["g"] - gw[c_][i]) > 1e-6:
            raise ValueError(f"v1 gate differs from the P6 gate table: {v1[c_]['g']} vs {gw[c_][i]}")
    defended = {c_: {"off": dict(final=_f(w[c_]), weight=1.0),
                     "v1": dict(final=_f(tg[c_].final[i]), weight=_f(np.clip(tg[c_].weight[i], 0, 1)))}
                for c_ in ("real", "flipped")}
    return dict(
        id="p5_fooled", **SLOTS["p5_fooled"], n_candidates=len(c), source="P5 / P6 (original test split)",
        domain=P5_DOMAIN, series=P5_DOMAIN, sample_id=w.sample_id, description=q.description, unit=q.unit,
        freq="monthly", timemmd=True,
        dates=_dates(pd.Timestamp(w.sample_id), len(nums), "monthly"), lookback=[_f(v) for v in nums],
        target_date=w.sample_id, y_true=_f(w.y_true), sigma=_f(sigma),
        fake_cond="flipped", synthetic_fake=True, claimed=int(d[i]),
        forecast=dict(none=_f(w["empty"]), real=_f(w.real), fake=_f(w.flipped)),
        defended=dict(real=defended["real"], fake=defended["flipped"]),
        v1=dict(real=v1["real"], fake=v1["flipped"]), v2=None, v2_available=False,
        text=dict(real=texts["real"], fake=texts["flipped"], units=None),
        settings=dict(v1=f"window {win}, τ {tau:g} (P6 frozen)", v2=None),
    )


# =============================================================================================

def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ex = r4_examples()
    ex.insert(3, p5_example())
    private = {}
    for e in ex:                               # Time-MMD texts stay out of git (no licence)
        if e["timemmd"]:
            private[e["id"]] = e["text"]
            e["text"] = None
    order = ["energy_flip", "energy_a2", "fnspid_a2", "fnspid_fbr", "p5_fooled"]
    ex.sort(key=lambda e: order.index(e["id"]))
    data = dict(
        note="Precomputed by `python -m app.build_data` from saved results; no model was run. "
             "Every fake text is SYNTHETIC (written by the project's deterministic attack rules) — "
             "research use only.",
        headline=headline(), models=model_table(), tradeoff=tradeoff_table(), examples=ex,
        fnspid_attribution="FNSPID — Zihan Dong, Xinyu Fan and Zhiyuan Peng. FNSPID: A Comprehensive "
                           "Financial News Dataset in Time Series. KDD 2024 (arXiv 2402.06698). "
                           "Benzinga headlines, licensed CC BY-NC 4.0 — non-commercial use only. "
                           "Changes: headlines grouped into weekly bundles; fakes are synthetic edits.",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT_TEXTS.write_text(json.dumps(private, ensure_ascii=False, indent=1), encoding="utf-8")
    FIG_DIR.mkdir(exist_ok=True)
    for name in FIGURES:
        shutil.copyfile(ROOT / "paper" / "figures" / f"{name}.png", FIG_DIR / f"{name}.png")
    for e in ex:
        print(f"{e['id']:<12} {e['series']:<10} {e['sample_id']}  candidates {e['n_candidates']:>3}  "
              f"fake {e['fake_cond']}")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB), "
          f"{OUT_TEXTS.relative_to(ROOT)} (gitignored), {len(FIGURES)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
