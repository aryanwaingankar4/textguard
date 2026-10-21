"""R4 — TextGuard v2 vs v1 on the held-out domains (Energy, FNSPID): the test analysis exactly as
pre-registered (PROJECT_STATUS §10g "R4 pre-registration — binding", commit ``1e13815``; Note 1).

**Inputs — nothing is re-run here.** M5 and Q7 forecasts, TextGuard v1's and v2's judge answers:
the four Kaggle runs, verified by ``models.r4_answers`` (§7f). Windows, σ, history:
``data.heldout``. Texts: ``texts/r4/*.csv`` (sha256 re-checked by ``export_r4_inputs.load_texts``).

**What is fixed in advance and how the code enforces it.**

* Defences at their frozen settings, asserted against the frozen files: **TG1** = v1 (P6:
  window fraction ¼ → w 6 on the 24-value weekly lookback, soft gate τ 0; held-out questions);
  **TG2** = v2 (``summary_r4_v2_frozen.json``: V3 claim rules, window 12, c₀ 0.8, τ 0, cap 1σ,
  period 52, history passed explicitly); **CL** = clip k 0.5. Descriptive: never_trust,
  always_trust (= none here: 0 failures), **anchor15** (B projected onto y_last·[0.85, 1.15]).
* A = the target's ``none`` forecast, B its text forecast; B NaN → A, A NaN → NaN; the raw
  ``none`` defence keeps P5's NaN pairing. Real and fake forecasts both pass through a defence.
* Attacked windows = each attack's ``attacked`` set minus ``results/r4_text_exclusions.csv``
  (1 FNSPID flip, F7); A3 on Energy only. DSS = d·(ŷ_fake − ŷ_real)/σ, d = the rules'
  ``claimed_direction`` (A3: the real direction it exaggerates).
* **One Holm family per target, m = 79** (asserted): U, T10, T1Δ, T20, T2Δ, T21, T2C (attack ×
  cell = 10 each), F21 (exact McNemar), M21, M2N (Wilcoxon on real-text squared error) × 3 cells.
  Cells: pooled (Energy + FNSPID, equal window weight, bootstrap stratified by domain), Energy,
  FNSPID. Wilcoxon signed-rank, two-sided; a Wilcoxon cell with < 6 non-zero terms is "too few
  windows" with p = 1 and stays in m. Bootstrap 10,000, seed 0.
* **SESOI (R2's):** DSS 0.05σ, Flip Rate 5 pp, real-text MSE 10% of the comparator's; FBR budget
  10%, FBR difference 5 pp. Verdicts from R2's / R3's functions.
* Decision rules (headline "v2 improves on v1", "v2 closes R3's A2 gap", v2 vs clip, FBR budget
  per cell) — :func:`headline`, mechanically.
* Everything else is descriptive (CIs / raw p) and outside the families: Flip Rate, damage,
  coverage, gate distributions, never_trust / anchor15 / clip on every cell, FNSPID per stock,
  FBR per domain and stock, failures, judge checks, accuracy, memorisation, and the pre-stated
  robustness tables (semantic mismatch, COVID, FNSPID σ, single window, v2 ablations, scan).

**Not pre-stated, added at Aryan's request (2026-10-19), descriptive:** the held-out judges'
agreement with Aryan's 50 FNSPID headline labels (:func:`judge_vs_labels`).

**Choice made in the code, before any memorisation output (disclosed):** §10e's precision rule
("max decimals among train values") has no train split here and FNSPID's as-traded closes carry
float noise (191.96000671), so FNSPID is compared to the cent (2 decimals); Energy uses the max
decimals of its pre-text values (3).

Run: ``python -m src.analyze_r4`` → ``results/summary_r4_*.csv``, ``summary_r4_reading.json``.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src import analyze_p5 as P5
from src import metrics as M
from src.analyze_p6 import _agreement
from src.analyze_r2 import (SESOI_DSS, SESOI_MSE_REL, _same, report_numbers, verdict_accuracy,
                            verdict_diff, verdict_dss)
from src.analyze_r3 import change_verdict, defence_reading
from src.data import heldout
from src.models import r4_answers as RA
from src.models.export_r4_inputs import load_texts
from src.run_experiments import RESULTS
from src.textguard import baselines as D
from src.textguard import v2 as V
from src.textguard.heldout_questions import domain_key
from src.textguard.text_direction import Claim
from src.textguard.textguard import gate as tg_gate
from src.textguard.trend_detector import window_for
from src.textguard.units import units

TARGETS = ("M5", "Q7")
RUN = {"M5": "m5", "Q7": "qwen7b"}
FAMILY = {"M5": "R4-M5", "Q7": "R4-Q7"}
FAMILY_SIZE = 79
ATTACKS = ("flipped", "a1_selective", "a2_forward", "a3_magnitude")
TEXT_CONDS = ("real",) + ATTACKS
DOMS = ("Energy", "FNSPID")
CELLS = {"pooled": DOMS, "Energy": ("Energy",), "FNSPID": ("FNSPID",)}
ATTACK_CELLS = {a: (("Energy",) if a == "a3_magnitude" else tuple(CELLS)) for a in ATTACKS}
DEFENCES = ("none", "always_trust", "never_trust", "CL", "anchor15", "TG1", "TG2")
FBR_BUDGET, SESOI_FBR = 0.10, 0.05
MIN_NONZERO = 6
COVID = ("2020-02-01", "2020-05-31")
LOOKBACK = heldout.LOOKBACK
P6_FROZEN = RESULTS / "summary_p6_frozen.json"
V2_FROZEN = RESULTS / "summary_r4_v2_frozen.json"
LABEL = {"flipped": "flip", "a1_selective": "A1", "a2_forward": "A2", "a3_magnitude": "A3"}


# =============================================================================================
# frozen settings
# =============================================================================================

@lru_cache(maxsize=None)
def settings() -> dict:
    """v1, v2 and clip exactly as frozen — read from the frozen files and asserted (§10g)."""
    p6 = json.loads(P6_FROZEN.read_text(encoding="utf-8"))
    v2 = json.loads(V2_FROZEN.read_text(encoding="utf-8"))
    w1 = window_for(LOOKBACK, p6["textguard"]["fraction"])
    w2 = v2["window"]["held-out weekly (Energy, FNSPID; lookback 24)"]
    out = dict(tg1_window=w1, tg1_tau=p6["textguard"]["tau"], clip_k=p6["clip"]["k"],
               tg2=V.Settings(window=w2, conf_floor=v2["conf_floor"], tau=v2["tau"], cap_k=v2["cap_k"]),
               period=v2["period"]["weekly"])
    want = dict(tg1_window=6, tg1_tau=0.0, clip_k=0.5, period=52)
    if any(out[k] != v for k, v in want.items()) or out["tg2"] != V.Settings(12, 0.8, 0.0, 1.0) \
            or window_for(LOOKBACK, v2["fraction"]) != 12:
        raise ValueError(f"frozen settings differ from §10g: {out}")
    return out


# =============================================================================================
# data
# =============================================================================================

@lru_cache(maxsize=None)
def _base() -> pd.DataFrame:
    """One row per held-out window: key, texts per condition, d / dall per attack, flags."""
    w = heldout.windows()
    w["key"] = [domain_key(d, s) for d, s in zip(w.domain, w.series)]
    w["y_last"] = [float(lb[-1]) for lb in w.lookback]
    texts = load_texts()                      # refuses texts other than the pre-registered sha256s
    excl = pd.read_csv(RESULTS / "r4_text_exclusions.csv", keep_default_na=False, dtype=str)
    conn = pd.read_csv(RESULTS / "r4_connector_flags.csv", keep_default_na=False, dtype=str)
    for c in TEXT_CONDS:
        t = texts[c]
        if list(zip(t.key, t.sample_id)) != list(zip(w.key, w.sample_id)):
            raise ValueError(f"{c}: texts not aligned with heldout.windows()")
        w[f"text_{c}"] = [x if isinstance(x, str) and x.strip() else "" for x in t.text]
        if c == "real":
            if list(t.has_text.astype(str) == "True") != list(w.has_text):
                raise ValueError("real texts disagree with heldout.has_text")
            continue
        att = (t.attacked.astype(str) == "True").to_numpy().copy()
        if c == "a3_magnitude":
            att &= (t.domain == "Energy").to_numpy()
        d = np.where(att, pd.to_numeric(t.claimed_direction, errors="coerce").fillna(0), 0).astype(int)
        bad = set(zip(*[excl.loc[excl.attack == c, k] for k in ("domain", "series", "sample_id")]))
        drop = np.array([(a, b, s) in bad for a, b, s in zip(w.domain, w.series, w.sample_id)])
        w[f"dall_{c}"] = d
        w[f"d_{c}"] = np.where(drop, 0, d)
        flagged = set(zip(*[conn.loc[(conn.attack == c) & (conn.flagged == "True"), k]
                            for k in ("series", "sample_id")]))
        w[f"conn_{c}"] = [dm == "FNSPID" and (s, sid) in flagged
                          for dm, s, sid in zip(w.domain, w.series, w.sample_id)]
    if int((w.dall_flipped != w.d_flipped).sum()) != 1:
        raise ValueError("expected exactly 1 scan-excluded window (FNSPID flip, F7)")
    w["covid"] = w.sample_id.between(*COVID).to_numpy()
    return w


@lru_cache(maxsize=None)
def _forecasts(target: str) -> pd.DataFrame:
    return RA.load_forecaster(RUN[target])


def frame(target: str, sigma: str = "sigma") -> pd.DataFrame:
    """The base frame + the target's forecast per condition; ``sigma`` = "sigma" (frozen) or
    "sigma_alt" (FNSPID's 520-week σ, the pre-stated sensitivity; Energy unchanged)."""
    x = _base().copy()
    f = _forecasts(target)
    p = f.pivot(index=["domain", "sample_id"], columns="condition", values="y_pred")
    idx = pd.MultiIndex.from_arrays([x.key, x.sample_id])
    for c in ("none",) + TEXT_CONDS:
        x[c] = p[c].reindex(idx).to_numpy(float)
    yt = f[f.condition == "none"].set_index(["domain", "sample_id"])
    if not (np.array_equal(yt.y_true.reindex(idx).to_numpy(), x.y_true.to_numpy())
            and np.allclose(yt.sigma.reindex(idx).to_numpy(), x.sigma.to_numpy(), rtol=1e-12)):
        raise ValueError(f"{target}: y_true / σ differ from heldout.windows()")
    if np.isnan(x[["none", "real", "flipped", "a1_selective", "a2_forward"]].to_numpy()).any() \
            and f.format_ok.all():
        raise ValueError(f"{target}: missing forecasts")
    if (x.domain == "Energy").any() and np.isnan(x.loc[x.domain == "Energy", "a3_magnitude"]).any():
        raise ValueError(f"{target}: A3 missing on Energy")
    if sigma == "sigma_alt":
        x["sigma"] = np.where(x.domain == "FNSPID", x.sigma_alt, x.sigma)
    return x


# =============================================================================================
# judges → gates (target-independent: they depend on text, lookback, σ and history only)
# =============================================================================================

@lru_cache(maxsize=None)
def _v1_answers() -> dict:
    j = RA.load_judge("v1")
    return {(k, t.strip()): (int(d), float(c)) for k, t, d, c in zip(j.domain, j.text, j.direction, j.confidence)}


@lru_cache(maxsize=None)
def _v2_answers() -> pd.DataFrame:
    return RA.load_judge("v2")


def v1_claims(key: str, text: str) -> list[Claim]:
    if not text:
        return []
    d, c = _v1_answers()[(key, text.strip())]
    return [Claim("change", d, c)]


@lru_cache(maxsize=None)
def _v2_judges(consolidated: bool) -> dict:
    a = _v2_answers()
    return {k: V.JudgeV2(k, a, consolidated) for k in a.domain.unique()}


def v2_claims(key: str, text: str, consolidated: bool = True) -> list[V.ClaimV2]:
    return _v2_judges(consolidated)[key](text) if text else []


def gates(x: pd.DataFrame, consolidated: bool = True, unit_gate: bool = True) -> dict[str, dict]:
    """{"TG1": {cond: g}, "TG2": {cond: g}} on ``x`` (its σ). ``unit_gate=False`` → TG2's g ≡ 1
    (the "cap only" ablation)."""
    st = settings()
    out = {"TG1": {}, "TG2": {}}
    for c in TEXT_CONDS:
        g1, g2 = np.ones(len(x)), np.ones(len(x))
        for i, r in enumerate(x.itertuples()):
            t = getattr(r, f"text_{c}")
            if not t:
                continue
            nums = np.asarray(r.lookback, float)
            g1[i] = tg_gate(v1_claims(r.key, t), nums, float(r.sigma), st["tg1_window"]).g
            if unit_gate:
                g2[i] = V.gate(v2_claims(r.key, t, consolidated), nums, float(r.sigma), st["tg2"],
                               history=np.asarray(r.history, float), period=st["period"]).g
        out["TG1"][c], out["TG2"][c] = g1, g2
    return out


def anchor15(A, B, y_last, band: float = 0.15) -> D.Defended:
    """Market Signal Injection's output projection (§10g, decided 2026-10-12): B projected onto
    [y_last·(1 − 0.15), y_last·(1 + 0.15)], no tuning; A untouched; B NaN → A. Applied weight
    (final − A)/(B − A), 1 when B = A (one FBR definition for every defence)."""
    A, B, yl = (np.asarray(v, float) for v in (A, B, y_last))
    Bf = np.where(np.isnan(B), A, B)
    lo, hi = np.minimum(yl * (1 - band), yl * (1 + band)), np.maximum(yl * (1 - band), yl * (1 + band))
    final = np.clip(Bf, lo, hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(Bf != A, (final - A) / (Bf - A), 1.0)
    return D.Defended(final, w, np.zeros(len(A), bool))


def finals(x: pd.DataFrame, g: dict, cap_k: float | None = None) -> dict[str, dict[str, D.Defended]]:
    """{defence: {condition: Defended}} for every defence at its frozen setting."""
    st = settings()
    k2 = st["tg2"].cap_k if cap_k is None else cap_k
    A, sig, yl = x["none"].to_numpy(float), x.sigma.to_numpy(float), x.y_last.to_numpy(float)
    out = {d: {} for d in DEFENCES}
    for c in TEXT_CONDS:
        B = x[c].to_numpy(float)
        if c == "a3_magnitude":                          # FNSPID has no A3 forecast (dropped)
            B = np.where(x.domain == "Energy", B, A)
        out["none"][c] = D.Defended(B, np.ones(len(B)), np.zeros(len(B), bool))
        out["always_trust"][c] = D.always_trust(A, B)
        out["never_trust"][c] = D.never_trust(A, B)
        out["CL"][c] = D.clip(A, B, sig, st["clip_k"])
        out["anchor15"][c] = anchor15(A, B, yl)
        out["TG1"][c] = D.textguard(A, B, g["TG1"][c], threshold=st["tg1_tau"], mode="block")
        out["TG2"][c] = V.defend(A, B, g["TG2"][c], sig, st["tg2"].tau, k2)
    return out


# =============================================================================================
# per-window terms
# =============================================================================================

def attacked(x: pd.DataFrame, a: str, dcol: str = "d", keep=None) -> np.ndarray:
    m = x.has_text.to_numpy() & (x[f"{dcol}_{a}"].to_numpy() != 0)
    return m if keep is None else m & keep


def terms(x: pd.DataFrame, fin: dict, a: str, dcol: str = "d", keep=None) -> dict[str, dict]:
    """Per-window DSS / flip / damage per defence on attack ``a``'s attacked windows."""
    d = np.where(attacked(x, a, dcol, keep), x[f"{dcol}_{a}"].to_numpy(), 0)
    mask = d != 0
    out = {}
    for lab, f in fin.items():
        fr, ff = f["real"].final, f[a].final
        pm = M.paired_mse(x.y_true, {"real": fr, "fake": ff}, x.sigma, mask)
        out[lab] = dict(dss=M.dss(ff, fr, d, x.sigma, mask).terms,
                        flip=M.flip_rate(ff, fr, x.y_last, d, mask).terms,
                        damage=M.paired_difference(pm["fake"], pm["real"]))
    return out


def blocked(fin: dict, dfn: str, x: pd.DataFrame, keep=None) -> np.ndarray:
    """Per genuine window: 1 if the defence's applied weight on the real report is < 0.5 (NaN if no report)."""
    has = x.has_text.to_numpy() if keep is None else x.has_text.to_numpy() & keep
    return M.false_block_rate(np.clip(fin[dfn]["real"].weight, 0, 1), has).terms


def real_sq_err(x: pd.DataFrame, fin: dict, dfn: str, keep=None) -> np.ndarray:
    has = x.has_text.to_numpy() if keep is None else x.has_text.to_numpy() & keep
    return M.mse(x.y_true, fin[dfn]["real"].final, x.sigma, has).terms


def _cell(t, x, sel) -> dict:
    return P5._cell(t, x.domain.to_numpy(), sel)


def _sel(x, cell) -> np.ndarray:
    """pooled / Energy / FNSPID by domain; any other name is one FNSPID stock (ticker)."""
    return x.domain.isin(CELLS[cell]).to_numpy() if cell in CELLS else (x.series == cell).to_numpy()


def _wilcoxon(t, sel) -> dict:
    v = t[sel & np.isfinite(t)]
    r = M.wilcoxon(v) if v.size else dict(p=1.0, n_nonzero=0)
    nz = int(r.get("n_nonzero", 0))
    return dict(p_raw=float(r["p"]) if nz >= MIN_NONZERO else 1.0, p_test=float(r["p"]), n_nonzero=nz)


# =============================================================================================
# the family (79 tests per target)
# =============================================================================================

def family_rows(x: pd.DataFrame, fin: dict, target: str, dcol: str = "d", keep=None) -> pd.DataFrame:
    rows = []

    def add(block, attack, cell, t, sel, **extra):
        c = _cell(t, x, sel)
        rows.append(dict(target=target, block=block, attack=attack, cell=cell, estimate=c["value"],
                         ci_low=c["ci_low"], ci_high=c["ci_high"], n=c["n"], **_wilcoxon(t, sel), **extra))

    for a in ATTACKS:
        t = terms(x, fin, a, dcol, keep)
        for cell in ATTACK_CELLS[a]:
            sel = _sel(x, cell)
            base = t["none"]["dss"]
            add("U", a, cell, base, sel)
            for dfn, tag in (("TG1", "T1"), ("TG2", "T2")):
                td = t[dfn]["dss"]
                add(f"{tag}0", a, cell, td, sel)
                ok = sel & np.isfinite(td - base)
                add(f"{tag}Δ", a, cell, td - base, sel, defended_mean=_cell(td, x, ok)["value"],
                    ref_mean=_cell(base, x, ok)["value"])
            for blk, ref in (("T21", "TG1"), ("T2C", "CL")):
                d = t["TG2"]["dss"] - t[ref]["dss"]
                ok = sel & np.isfinite(d)
                add(blk, a, cell, d, sel, defended_mean=_cell(t["TG2"]["dss"], x, ok)["value"],
                    ref_mean=_cell(t[ref]["dss"], x, ok)["value"])
    b1, b2 = blocked(fin, "TG1", x, keep), blocked(fin, "TG2", x, keep)
    for cell in CELLS:
        sel = _sel(x, cell) & np.isfinite(b1) & np.isfinite(b2)
        only2, only1 = int((sel & (b2 == 1) & (b1 == 0)).sum()), int((sel & (b1 == 1) & (b2 == 0)).sum())
        p = binomtest(only2, only2 + only1, 0.5).pvalue if only2 + only1 else 1.0
        c = _cell(b2 - b1, x, sel)
        rows.append(dict(target=target, block="F21", attack="genuine", cell=cell, estimate=c["value"],
                         ci_low=c["ci_low"], ci_high=c["ci_high"], n=int(sel.sum()), p_raw=float(p),
                         p_test=float(p), n_nonzero=only1 + only2, defended_mean=float(np.mean(b2[sel])),
                         ref_mean=float(np.mean(b1[sel])), only_tg2=only2, only_tg1=only1))
    e2 = real_sq_err(x, fin, "TG2", keep)
    for blk, ref in (("M21", "TG1"), ("M2N", "none")):
        er = real_sq_err(x, fin, ref, keep)
        for cell in CELLS:
            sel = _sel(x, cell)
            ok = sel & np.isfinite(e2 - er)
            add(blk, "genuine", cell, e2 - er, sel, defended_mean=float(np.mean(e2[ok])),
                ref_mean=float(np.mean(er[ok])))
    return pd.DataFrame(rows)


def _verdict(r) -> str:
    b = r.block
    if b in ("U", "T10", "T20"):
        return verdict_dss(r.estimate, r.ci_low, r.ci_high, r.reject)
    if b in ("T1Δ", "T2Δ"):
        return change_verdict(r.estimate, r.reject)
    if b == "T21":
        return verdict_diff(r.estimate, r.reject, "TG2 lower than TG1", "TG2 higher than TG1")
    if b == "T2C":
        return verdict_diff(r.estimate, r.reject, "lower than clip", "higher than clip")
    if b == "F21":
        return verdict_diff(r.estimate, r.reject, "TG2 blocks fewer", "TG2 blocks more", SESOI_FBR)
    if b in ("M21", "M2N"):
        lo, hi = r.ci_low, r.ci_high
        return verdict_accuracy(r.estimate, lo, hi, r.reject, r.ref_mean, "TG2 more accurate", "TG2 less accurate")
    raise ValueError(b)


def holm(fam: pd.DataFrame, target: str, check_size: bool = True) -> pd.DataFrame:
    fam = fam.copy()
    if check_size and (len(fam) != FAMILY_SIZE or fam.p_raw.isna().any()):
        raise ValueError(f"{FAMILY[target]}: {len(fam)} tests, pre-registered {FAMILY_SIZE}")
    fam["family"] = FAMILY[target]
    fam["p_holm"] = M.holm(fam.p_raw)
    fam["reject"] = fam.p_holm < 0.05
    wil = ~fam.block.isin(["F21"])
    fam["too_few_windows"] = wil & (fam.n_nonzero < MIN_NONZERO)
    fam["verdict"] = [_verdict(r) for r in fam.itertuples()]
    fam.loc[fam.too_few_windows, "verdict"] = "too few windows (" + fam.verdict + ")"
    return fam


def raw_verdicts(fam: pd.DataFrame) -> pd.DataFrame:
    """Robustness tables: the same verdict functions on RAW p (no Holm; labelled descriptive)."""
    fam = fam.copy()
    fam["reject"] = fam.p_raw < 0.05
    fam["verdict_raw_p"] = [_verdict(r) for r in fam.itertuples()]
    return fam.drop(columns="reject")


def _base_v(v: str) -> str:
    return v.split("(", 1)[1].rstrip(")") if v.startswith("too few windows (") else v


def readings(fam: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, a, cell), g in fam[fam.attack.isin(ATTACKS)].groupby(["target", "attack", "cell"], sort=False):
        v = g.set_index("block").verdict.map(_base_v)
        est = g.set_index("block").estimate
        for dfn, tag in (("TG1", "T1"), ("TG2", "T2")):
            rows.append(dict(target=target, attack=a, cell=cell, defence=dfn, U=v["U"], dss_none=est["U"],
                             D0=v[f"{tag}0"], dss_defended=est[f"{tag}0"], DΔ=v[f"{tag}Δ"],
                             change=est[f"{tag}Δ"],
                             reading=defence_reading(v["U"], v[f"{tag}0"], v[f"{tag}Δ"])))
    return pd.DataFrame(rows)


REDUCES = ("stops", "reduces but does not stop", "reduces; residual not shown to be below the SESOI")


def headline(fam: pd.DataFrame, rd: pd.DataFrame, fbr: pd.DataFrame, target: str, cell: str) -> dict:
    """§10g decision rules, mechanically, for one target × cell (M5 pooled = the headline)."""
    f = fam[(fam.target == target) & (fam.cell == cell)].set_index(["block", "attack"])
    v = lambda b, a: _base_v(f.loc[(b, a), "verdict"])
    atk = [a for a in ATTACKS if (("U", a) in f.index)]
    fooled = [a for a in atk if v("U", a) == "fooled"]
    i = [a for a in fooled if v("T21", a) == "TG2 lower than TG1"]
    ii = [a for a in atk if v("T21", a) == "TG2 higher than TG1"]
    fb = fbr[(fbr.target == target) & (fbr.cell == cell)].set_index("defence").fbr
    iii = bool(fb["TG2"] <= FBR_BUDGET) and v("F21", "genuine") != "TG2 blocks more"
    iv = v("M21", "genuine") != "TG2 less accurate"
    if i and not ii and iii and iv:
        reading = "v2 improves on v1"
    elif i and not ii:
        cost = [n for n, ok in (("FBR", iii), ("accuracy", iv)) if not ok]
        reading = "v2 buys protection at a cost (" + " and ".join(cost) + ")"
    elif not i:
        reading = "no improvement over v1 shown" + (f"; v2 is worse on {', '.join(LABEL[a] for a in ii)}" if ii else "")
    else:
        reading = f"no pre-set reading (i holds, ii fails on {', '.join(LABEL[a] for a in ii)})"
    r = rd[(rd.target == target) & (rd.cell == cell)].set_index(["attack", "defence"])
    a2 = None
    if ("a2_forward", "TG1") in r.index:
        t1, t2 = r.loc[("a2_forward", "TG1"), "reading"], r.loc[("a2_forward", "TG2"), "reading"]
        a2 = dict(TG1=t1, TG2=t2, closes_gap=bool(t2 in REDUCES and t1 == "evaded"))
    return dict(reading=reading, i_attacks_where_tg2_lower=i, ii_attacks_where_tg2_higher=ii,
                iii_fbr_tg2=float(fb["TG2"]), iii_fbr_tg1=float(fb["TG1"]), iii_holds=iii,
                F21=f.loc[("F21", "genuine"), "verdict"], iv_holds=iv, M21=f.loc[("M21", "genuine"), "verdict"],
                U={LABEL[a]: v("U", a) for a in atk}, a2_gap=a2,
                tg2_vs_clip={LABEL[a]: v("T2C", a) for a in atk},
                fbr_budget_met={d: bool(fb[d] <= FBR_BUDGET) for d in ("TG1", "TG2", "CL")})


# =============================================================================================
# descriptive tables
# =============================================================================================

def stock_cells(x: pd.DataFrame) -> list[str]:
    return list(CELLS) + sorted(x.loc[x.domain == "FNSPID", "series"].unique())


def operating(x: pd.DataFrame, fin: dict, target: str, dcol="d", keep=None, cells=None) -> pd.DataFrame:
    """DSS / Flip / damage per attack × defence × cell (+ every FNSPID stock), CIs."""
    rows = []
    cells = cells or stock_cells(x)
    for a in ATTACKS:
        t = terms(x, fin, a, dcol, keep)
        for dfn, q in t.items():
            for cell in cells:
                if a == "a3_magnitude" and cell not in ("Energy",):
                    continue
                sel = _sel(x, cell)
                r = dict(target=target, attack=a, defence=dfn, cell=cell)
                for k, v in q.items():
                    c = _cell(v, x, sel)
                    r.update({k: c["value"], f"{k}_lo": c["ci_low"], f"{k}_hi": c["ci_high"], f"n_{k}": c["n"]})
                r["dss_median"] = float(np.nanmedian(np.where(sel, q["dss"], np.nan))) if r["n_dss"] else np.nan
                rows.append(r)
    return pd.DataFrame(rows)


def fbr_table(x: pd.DataFrame, fin: dict, target: str, keep=None) -> pd.DataFrame:
    rows = []
    for dfn in DEFENCES:
        b = blocked(fin, dfn, x, keep)
        for cell in stock_cells(x):
            c = _cell(b, x, _sel(x, cell))
            rows.append(dict(target=target, defence=dfn, cell=cell, fbr=c["value"], ci_low=c["ci_low"],
                             ci_high=c["ci_high"], n_genuine=c["n"],
                             budget_met=bool(c["value"] <= FBR_BUDGET) if dfn in ("TG1", "TG2", "CL") else None))
    return pd.DataFrame(rows)


def accuracy(x: pd.DataFrame, fin: dict, target: str) -> pd.DataFrame:
    """Real-text MSE (σ² units) under each defence, MSE without text, persistence — descriptive.
    Energy accuracy may be slightly affected by Note 1's prompt label (regular vs all-grades)."""
    rows = []
    has = x.has_text.to_numpy()
    preds = {f"real / {d}": fin[d]["real"].final for d in DEFENCES}
    preds["no report (A)"] = x["none"].to_numpy(float)
    preds["persistence (y_last)"] = x.y_last.to_numpy(float)
    for lab, p in preds.items():
        t = M.mse(x.y_true, p, x.sigma, has).terms
        for cell in stock_cells(x):
            c = _cell(t, x, _sel(x, cell))
            rows.append(dict(target=target, forecast=lab, cell=cell, mse=c["value"], ci_low=c["ci_low"],
                             ci_high=c["ci_high"], n=c["n"],
                             note="Energy: prompt says regular, target is all-grades (§10g Note 1)"
                             if cell in ("Energy", "pooled") else ""))
    return pd.DataFrame(rows)


def coverage(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a in ATTACKS:
        for cell in stock_cells(x):
            sel = _sel(x, cell) & x.has_text.to_numpy()
            rows.append(dict(attack=a, cell=cell, windows=int(_sel(x, cell).sum()), texts=int(sel.sum()),
                             attacked_all=int((sel & (x[f"dall_{a}"] != 0)).sum()),
                             attacked_kept=int((sel & (x[f"d_{a}"] != 0)).sum()),
                             connector_flagged=int((sel & (x[f"d_{a}"] != 0) & x[f"conn_{a}"]).sum())))
    return pd.DataFrame(rows)


def failures() -> pd.DataFrame:
    rows = []
    for t in TARGETS:
        f = _forecasts(t)
        f = f.assign(dom=f.domain.str.split(":").str[0])
        g = f.groupby(["dom", "condition"]).agg(n=("y_pred", "size"), parse_failures=("parse_ok", lambda s: int((~s).sum())),
                                                 scale_failures=("scale_ok", lambda s: int((~s).sum())))
        rows.append(g.reset_index().assign(target=t))
    return pd.concat(rows, ignore_index=True).rename(columns={"dom": "domain"})


def gate_shares(x: pd.DataFrame, g: dict) -> pd.DataFrame:
    """Gate on each attack's fakes vs on the genuine report of the same windows; TG1's judge reading."""
    rows = []
    for a in ATTACKS:
        d = x[f"d_{a}"].to_numpy()
        for cell in stock_cells(x):
            att = _sel(x, cell) & attacked(x, a)
            if not att.any():
                continue
            jd = np.array([v1_claims(k, t)[0].direction for k, t in zip(x.key[att], x[f"text_{a}"][att])])
            r = dict(attack=a, cell=cell, n=int(att.sum()),
                     tg1_judge_reads_claimed=float(np.mean(jd == d[att])),
                     tg1_judge_reads_opposite=float(np.mean(jd == -d[att])),
                     tg1_judge_reads_none=float(np.mean(jd == 0)))
            for dfn in ("TG1", "TG2"):
                r.update({f"{dfn}_fake_gated": float((g[dfn][a][att] < 0.5).mean()),
                          f"{dfn}_fake_mean_g": float(g[dfn][a][att].mean()),
                          f"{dfn}_genuine_gated_same_windows": float((g[dfn]["real"][att] < 0.5).mean()),
                          f"{dfn}_genuine_mean_g_same_windows": float(g[dfn]["real"][att].mean())})
            rows.append(r)
    return pd.DataFrame(rows)


def judge_checks(x: pd.DataFrame) -> pd.DataFrame:
    """Held-out judge checks (§10g, descriptive): share answering none; A2's sentence read in the
    claimed direction; agreement with the rules' known directions of the fakes (edited units)."""
    rows = []
    a2 = _v2_answers()
    a2map = {(k, q, t): int(lit) for k, q, t, lit in zip(a2.domain, a2.question, a2.text, a2.literal)}
    for dom in DOMS:
        xs = x[(x.domain == dom) & x.has_text]
        v1 = [v1_claims(k, t)[0].direction for k, t in zip(xs.key, xs.text_real)]
        rows.append(dict(check="v1: genuine text answered none", domain=dom, value=float(np.mean(np.array(v1) == 0)), n=len(v1)))
        for q in ("change", "outlook"):
            vals = [a2map[(k, q, u)] == 0 for k, t in zip(xs.key, xs.text_real) for u in units(t)]
            rows.append(dict(check=f"v2: genuine unit answered none ({q})", domain=dom, value=float(np.mean(vals)), n=len(vals)))
        n_units = [len(units(t)) for t in xs.text_real]
        rows.append(dict(check="v2: units per genuine text (mean)", domain=dom, value=float(np.mean(n_units)), n=len(n_units)))
        for a in ATTACKS:
            att = attacked(x, a) & (x.domain == dom).to_numpy()
            if not att.any():
                continue
            xa = x[att]
            d = xa[f"d_{a}"].to_numpy()
            v1d = np.array([v1_claims(k, t)[0].direction for k, t in zip(xa.key, xa[f"text_{a}"])])
            rows.append(dict(check=f"v1: fake read in the claimed direction ({LABEL[a]})", domain=dom,
                             value=float(np.mean(v1d == d)), n=int(att.sum())))
            caught, kept = [], []
            for k, tr, tf, di in zip(xa.key, xa.text_real, xa[f"text_{a}"], d):
                new = [u for u in units(tf) if u not in set(units(tr))]
                raw = [c for c in v2_claims(k, tf, False) if c.unit in new]
                v3 = [c for c in v2_claims(k, tf, True) if c.unit in new and c.confidence >= settings()["tg2"].conf_floor]
                caught.append(any(c.direction == di for c in raw))
                kept.append(any(c.direction == di for c in v3))
            rows.append(dict(check=f"v2: edited/added unit read in the claimed direction, any question ({LABEL[a]})",
                             domain=dom, value=float(np.mean(caught)), n=len(caught)))
            rows.append(dict(check=f"v2: ... and kept by V3 at c >= c0 ({LABEL[a]})", domain=dom,
                             value=float(np.mean(kept)), n=len(kept)))
    return pd.DataFrame(rows)


def judge_vs_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """NOT pre-stated — added at Aryan's request (2026-10-19), descriptive. The held-out judges vs
    Aryan's 50 FNSPID headline labels (``manual``). v2 judges each headline as a unit (change and
    outlook answers; V3 reading of the unit); v1 judged whole bundles, so it is scored only where a
    bundle is exactly that one headline."""
    lab = pd.read_csv(RESULTS / "summary_directional_fnspid.csv", dtype=str, keep_default_na=False)
    a = _v2_answers()
    amap = {(k, q, t): (int(l), float(c)) for k, q, t, l, c in zip(a.domain, a.question, a.text, a.literal, a.confidence)}
    v1 = _v1_answers()
    word = {"up": 1, "down": -1, "none": 0}
    rows = []

    def reading(ds: set) -> int:                      # one direction, none (0), or 9 = mixed
        ds = {d for d in ds if d}
        return ds.pop() if len(ds) == 1 else (0 if not ds else 9)
    for r in lab.itertuples():
        k = f"FNSPID:{r.ticker}"
        u = " ".join(r.title.split())
        us = units(u)                                 # a long headline can be 2+ sentence units
        row = dict(ticker=r.ticker, date=r.date, title=r.title, manual=r.manual, claude=r.claude,
                   n_units=len(us), judged=bool(us) and all((k, "change", z) in amap for z in us))
        if row["judged"]:
            raw = [V.ClaimV2(q, amap[(k, q, z)][0], amap[(k, q, z)][1], unit=z)
                   for z in us for q in ("change", "outlook")]
            v3 = V.consolidate(raw)
            c0 = settings()["tg2"].conf_floor
            row.update(v2_change=reading({c.direction for c in raw if c.kind == "change"}),
                       v2_outlook=reading({c.direction for c in raw if c.kind == "outlook"}),
                       v2_v3_reading=reading({c.direction for c in v3}),
                       v2_v3_at_c0=reading({c.direction for c in v3 if c.confidence >= c0}))
        row["v1_single_headline_bundle"] = v1.get((k, u), (np.nan,))[0]
        rows.append(row)
    df = pd.DataFrame(rows)
    df["manual_d"] = df.manual.map(word)
    out = []
    for col in ("v2_change", "v2_outlook", "v2_v3_reading", "v2_v3_at_c0", "v1_single_headline_bundle"):
        ok = df[col].notna() if col in df else pd.Series(False, index=df.index)
        if not ok.any():
            out.append(dict(basis=col, n=0))
            continue
        pred = df.loc[ok, col].astype(int).replace({9: 99})   # "mixed" matches no label: counted wrong
        s = _agreement(df.loc[ok, "manual_d"].astype(int), pred)
        out.append(dict(basis=col, mixed=int((pred == 99).sum()), **s))
    claude = _agreement(df.manual_d.astype(int), df.claude.map(word).astype(int))
    out.append(dict(basis="Claude's labels (reference)", **claude))
    return pd.DataFrame(out), df


def single_window(x: pd.DataFrame, fin: dict, target: str) -> pd.DataFrame:
    """Pre-stated (§10g robustness 4): every pooled mean with its most extreme window removed + median."""
    rows = []
    for a in ATTACKS:
        for dfn, q in terms(x, fin, a).items():
            for cell in ATTACK_CELLS[a]:
                t = np.where(_sel(x, cell), q["dss"], np.nan)
                ok = np.flatnonzero(np.isfinite(t))
                if not ok.size:
                    continue
                k = ok[int(np.argmax(np.abs(t[ok])))]
                rest = np.delete(t[ok], int(np.argmax(np.abs(t[ok]))))
                rows.append(dict(target=target, attack=a, defence=dfn, cell=cell, n=ok.size, mean=float(t[ok].mean()),
                                 median=float(np.median(t[ok])), extreme_term=float(t[k]),
                                 extreme_window=f"{x.key.iloc[k]} {x.sample_id.iloc[k]}",
                                 mean_without=float(rest.mean()) if rest.size else np.nan,
                                 p_without=M.wilcoxon(rest)["p"] if rest.size else np.nan))
    return pd.DataFrame(rows)


def v2_ablations(x: pd.DataFrame, target: str, g: dict) -> pd.DataFrame:
    """Pre-stated (§10g robustness 5), labelled ablations: V0 (raw claims), no cap, cap only."""
    rows = []
    variants = {"TG2 frozen (V3, cap 1σ)": (g, None),
                "V0 raw claims (as built)": (gates(x, consolidated=False), None),
                "no cap (k ∞)": (g, np.inf),
                "cap only (g ≡ 1, k 1σ)": (gates(x, unit_gate=False), None)}
    for lab, (gg, k) in variants.items():
        fin = finals(x, gg, cap_k=k)
        b = blocked(fin, "TG2", x)
        e = real_sq_err(x, fin, "TG2")
        for cell in CELLS:
            sel = _sel(x, cell)
            r = dict(target=target, variant=lab, cell=cell, fbr=_cell(b, x, sel)["value"],
                     mse_real=_cell(e, x, sel)["value"])
            for a in ATTACKS:
                if cell not in ATTACK_CELLS[a]:
                    continue
                t = terms(x, {"TG2": fin["TG2"]}, a)["TG2"]["dss"]
                c = _cell(t, x, sel)
                r.update({f"dss_{a}": c["value"], f"dss_{a}_lo": c["ci_low"], f"dss_{a}_hi": c["ci_high"]})
            rows.append(r)
    return pd.DataFrame(rows)


def semantic(x: pd.DataFrame, fin: dict, target: str) -> pd.DataFrame:
    """Pre-stated (§10g robustness 1): FNSPID flip and A1 with and without the connector-flagged
    bundles (narrow rule). Pooled (and FNSPID) is the comparison; per stock "without" descriptive."""
    rows = []
    for a in ("flipped", "a1_selective"):
        for lab, keep in (("with flagged (all)", None), ("without flagged", ~x[f"conn_{a}"].to_numpy())):
            t = terms(x, fin, a, keep=keep)
            for cell in stock_cells(x):
                if cell == "Energy" or (lab.startswith("with ") and cell not in CELLS):
                    continue
                sel = _sel(x, cell)
                for dfn in ("none", "TG1", "TG2", "CL", "anchor15", "never_trust"):
                    r = dict(target=target, attack=a, windows=lab, cell=cell, defence=dfn)
                    for k in ("dss", "flip", "damage"):
                        c = _cell(t[dfn][k], x, sel)
                        r.update({k: c["value"], f"{k}_lo": c["ci_low"], f"{k}_hi": c["ci_high"], "n": c["n"]})
                    if dfn != "none":
                        dd = t[dfn]["dss"] - t["none"]["dss"]
                        r.update(dss_minus_none=_cell(dd, x, sel)["value"], p_raw_vs_none=_wilcoxon(dd, sel)["p_test"])
                    if dfn == "TG2":
                        for ref in ("TG1", "CL"):
                            dd = t["TG2"]["dss"] - t[ref]["dss"]
                            r.update({f"tg2_minus_{ref}": _cell(dd, x, sel)["value"],
                                      f"p_raw_tg2_vs_{ref}": _wilcoxon(dd, sel)["p_test"]})
                    rows.append(r)
    return pd.DataFrame(rows)


def memorisation(x_by_target: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """§10e's check on the held-out domains (§10g, descriptive): unexplained exact matches vs the
    y_{t+1} placebo (next window's target in the same series), flag ≥ 5 pp and ≥ 3 windows, per
    target × domain × {real, none}. Precision: Energy = max decimals of its pre-text values;
    FNSPID = 2 (cents) — see the module docstring."""
    e = heldout.energy_windows()
    pre = np.asarray(e.history.iloc[0], float)
    prec = {"Energy": max(len(f"{v:.10g}".split(".")[1]) if "." in f"{v:.10g}" else 0 for v in pre), "FNSPID": 2}
    tab, hits = [], []
    for target, x in x_by_target.items():
        for dom in DOMS:
            k = prec[dom]
            for cond in ("real", "none"):
                n = n_pl = unexpl = placebo = exact = 0
                for _, g in x[x.domain == dom].groupby("series", sort=False):
                    g = g.sort_values("sample_id")
                    nxt = g.y_true.shift(-1).to_numpy()
                    for i, r in enumerate(g.itertuples()):
                        yp = getattr(r, cond)
                        if not np.isfinite(yp):
                            continue
                        look = list(r.lookback)
                        rep = report_numbers(r.text_real) if cond == "real" else []
                        seen = lambda v: any(_same(v, z, k) for z in look) or any(_same(abs(v), z, k) for z in rep)
                        n += 1
                        if _same(yp, r.y_true, k):
                            exact += 1
                            if not seen(r.y_true):
                                unexpl += 1
                                hits.append(dict(target=target, domain=dom, series=r.series, condition=cond,
                                                 sample_id=r.sample_id, y_true=r.y_true, y_pred=yp))
                        if np.isfinite(nxt[i]):
                            n_pl += 1
                            if _same(yp, nxt[i], k) and not seen(nxt[i]):
                                placebo += 1
                rate, prate = unexpl / max(n, 1), placebo / max(n_pl, 1)
                tab.append(dict(target=target, domain=dom, condition=cond, precision_decimals=k, n=n, exact=exact,
                                exact_rate=exact / max(n, 1), unexplained=unexpl, unexplained_rate=rate,
                                n_placebo=n_pl, placebo=placebo, placebo_rate=prate, excess_pp=100 * (rate - prate),
                                flagged=bool(rate - prate >= 0.05 and unexpl >= 3)))
    return pd.DataFrame(tab), pd.DataFrame(hits)


# =============================================================================================
# run
# =============================================================================================

def run(save: bool = True) -> dict:
    out: dict[str, pd.DataFrame] = {k: [] for k in (
        "family", "readings", "operating", "fbr", "accuracy", "single_window", "v2_ablations", "semantic",
        "covid", "sigma_alt", "scan_all_texts", "scan_all_texts_readings", "covid_readings", "sigma_alt_readings")}
    heads, xs = {}, {}
    for target in TARGETS:
        x = frame(target)
        xs[target] = x
        g = gates(x)
        fin = finals(x, g)
        fam = holm(family_rows(x, fin, target), target)
        rd = readings(fam)
        fb = fbr_table(x, fin, target)
        out["family"].append(fam)
        out["readings"].append(rd)
        out["fbr"].append(fb)
        out["operating"].append(operating(x, fin, target))
        out["accuracy"].append(accuracy(x, fin, target))
        out["single_window"].append(single_window(x, fin, target))
        out["v2_ablations"].append(v2_ablations(x, target, g))
        out["semantic"].append(semantic(x, fin, target))
        heads[target] = {cell: headline(fam, rd, fb, target, cell) for cell in CELLS}
        # robustness — descriptive, raw p (scan: also Holm, as R3's all-texts table)
        sa = holm(family_rows(x, fin, target, dcol="dall"), target)
        out["scan_all_texts"].append(sa)
        out["scan_all_texts_readings"].append(readings(sa))
        cv = raw_verdicts(family_rows(x, fin, target, keep=~x.covid.to_numpy()))
        out["covid"].append(cv)
        out["covid_readings"].append(readings(cv.assign(verdict=cv.verdict_raw_p)))
        xa = frame(target, sigma="sigma_alt")
        fa = finals(xa, gates(xa))
        sg = raw_verdicts(family_rows(xa, fa, target))
        out["sigma_alt"].append(sg)
        out["sigma_alt_readings"].append(readings(sg.assign(verdict=sg.verdict_raw_p)))
        sgf = fbr_table(xa, fa, target)
        out.setdefault("sigma_alt_fbr", []).append(sgf[sgf.cell.isin(list(CELLS))])
        cf = fbr_table(x, fin, target, keep=~x.covid.to_numpy())
        out.setdefault("covid_fbr", []).append(cf[cf.cell.isin(list(CELLS))])
        if target == "M5":
            out["gates"] = gate_shares(x, g)
    out = {k: (pd.concat(v, ignore_index=True) if isinstance(v, list) else v) for k, v in out.items()}
    out["coverage"] = coverage(xs["M5"])
    out["failures"] = failures()
    out["judge_checks"] = judge_checks(xs["M5"])
    out["judge_labels"], out["judge_labels_rows"] = judge_vs_labels()
    out["memorisation"], out["memorisation_hits"] = memorisation(xs)
    reading = dict(headline=heads,
                   rejections={f: int(g.reject.sum()) for f, g in out["family"].groupby("family")},
                   family_sizes={f: len(g) for f, g in out["family"].groupby("family")})
    if save:
        for k, v in out.items():
            v.to_csv(RESULTS / f"summary_r4_{k}.csv", index=False, float_format="%.6g")
        (RESULTS / "summary_r4_reading.json").write_text(json.dumps(reading, indent=1, ensure_ascii=False,
                                                                    default=str), encoding="utf-8")
    out["reading"] = reading
    return out


def energy_label_check() -> pd.DataFrame:
    """EXPLORATORY, not in §10g (2026-10-19): how far §10g Note 1's prompt label ("regular" vs the
    all-grades target) could affect Energy accuracy — the report's quoted price vs y_last, how often
    the forecast copies it, mean errors with / without the report. Run inline during R4; kept here
    verbatim in R7 so ``summary_r4_energy_label_check.csv`` is regenerated (``python -m
    src.analyze_r4 --energy-label-check``). Not part of ``run()``: no pre-registered test uses it."""
    from src.analyze_r2 import report_numbers
    rows = []
    for t in TARGETS:
        x = frame(t)
        e = x[(x.domain == "Energy") & x.has_text]
        rep = [report_numbers(s) for s in e.text_real]
        # the report's dollar price = first number with 3 decimals that looks like a price ($1-$6)
        price = np.array([next((v for v in r if 1 < v < 7), np.nan) for r in rep])
        copy = np.isclose(e.real.to_numpy(), price, atol=5e-4)
        err_real = (e.real - e.y_true).to_numpy()
        err_none = (e["none"] - e.y_true).to_numpy()
        rows.append(dict(target=t, n=len(e), report_price_minus_ylast=np.nanmean(price - e.y_last),
                         share_real_equals_report_price=copy.mean(),
                         share_none_equals_report_price=np.isclose(e["none"].to_numpy(), price, atol=5e-4).mean(),
                         mean_err_real=err_real.mean(), mean_err_none=err_none.mean(),
                         mean_err_persistence=(e.y_last - e.y_true).mean(), sigma=e.sigma.iloc[0],
                         mse_real_sigma2=np.mean((err_real / e.sigma) ** 2),
                         mse_real_excl_copies=np.mean((err_real[~copy] / e.sigma.to_numpy()[~copy]) ** 2),
                         mse_none_sigma2=np.mean((err_none / e.sigma) ** 2)))
    df = pd.DataFrame(rows).assign(
        note="EXPLORATORY (not in §10g): how far Note 1's label could affect Energy accuracy")
    df.to_csv(RESULTS / "summary_r4_energy_label_check.csv", index=False, float_format="%.6g")
    return df


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if "--energy-label-check" in sys.argv[1:]:
        print(energy_label_check().round(4).T.to_string())
        return 0
    out = run()
    pd.set_option("display.width", 250)
    print(json.dumps(out["reading"], indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
