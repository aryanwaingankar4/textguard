"""R4 — TextGuard v2 development on the ORIGINAL domains' VAL split (nothing here is a test).

v2 is designed and tuned on val only (PROJECT_STATUS §1a, §10g). This module never reads a test
window: every frame is built with ``split="val"`` and :func:`_val_only` asserts it. The held-out
domains (Energy, FNSPID) are not touched.

Two judges:

* ``v2`` — v2's own claim judge (``judge_prompt_v2``; every unit asked change / outlook / level),
  from the Kaggle run ``results/qwen/r4v2_val_judge_kaggle_*``. **This is v2.**
* ``v1j`` — *interim*, before that run exists: v1's whole-report answers (R3's judge run, which
  covers val) dressed as v2 claims — Traffic's change claim in the year-over-year frame (v1's
  Traffic question *is* year-over-year), Health_US gated by its level claim only (v1's change
  question reads level claims). It shows what the judge-free parts (frame evidence, cap, floor,
  block) do; it cannot show "every claim" or forward claims, and it is never frozen.

Grid (fixed before any v2 answer existed): window fraction f ∈ {¼, ½, 1} (v1's grid), confidence
floor c₀ ∈ {0, 0.6, 0.8}, τ ∈ {0, 0.5}, cap k ∈ {0.25, 0.5, 1, ∞} → 72 configs, Model 5, pooled.

Selection rule (approved by Aryan 2026-10-16): among configs with val FBR ≤ 10% (§10b's budget),
the lowest ``obj`` = mean over the four attacks (flipped, a2_forward, exaggerated, a3_magnitude) of
**|pooled DSS|** — each attack's DSS is first pooled (mean over its attacked windows), then its
absolute value is taken, then the four are averaged. Not the mean of per-window |DSS|, which would
count noise as manipulation. Why |·|: the goal is neutralising, not over-correcting (a defended
forecast pushed *away* from the lie, as v1 did on val, is a distortion too; same logic as rejecting
signed-DSS tie-breaks in P6). Ties → lower FBR, smaller τ, larger k (less intervention), smaller c₀,
smaller f. None within budget → the lowest-FBR config, reported as "budget not met". A1 is
descriptive (34 val windows). Q7 descriptive at the chosen config.

Run: ``python -m src.analyze_r4_val [v1j|v2]`` → ``results/summary_r4_val_grid_<judge>.csv``,
``results/summary_r4_val_compare_<judge>.csv`` (+ ``summary_r4_val_judge_human.csv`` for v2).
"""

from __future__ import annotations

import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

from src import analyze_p5 as P5
from src import analyze_p6 as P6
from src import analyze_r3 as R3
from src import metrics as M
from src.attacks import common
from src.data.loader import load_domain
from src.data.windows import lookback_for
from src.run_experiments import RESULTS
from src.textguard import baselines as D
from src.textguard import v2 as V
from src.textguard.export_v2_inputs import V2_VAL_INPUTS
from src.textguard.text_direction import stated_baseline
from src.textguard.trend_detector import window_for

SPLIT = "val"
DOMAINS = tuple(P5.DOMAINS)
WEEKLY = ("Health_US",)
OBJ_ATTACKS = ("flipped", "a2_forward", "exaggerated", "a3_magnitude")
ATTACKS = OBJ_ATTACKS + ("a1_selective",)
CONDS = ("real",) + ATTACKS
FRACTIONS = (0.25, 0.5, 1.0)
FLOORS = (0.0, 0.6, 0.8)
TAUS = (0.0, 0.5)
CAPS = (0.25, 0.5, 1.0, np.inf)
FBR_BUDGET = P6.FBR_BUDGET
V2_ANSWERS = V.QWEN_DIR / "r4v2_val_judge_kaggle_predictions.csv"
V2_META = V.QWEN_DIR / "r4v2_val_judge_kaggle_run_meta.json"
KNOBS = ["tau", "cap_k_desc", "conf_floor", "fraction"]


def _val_only(x: pd.DataFrame) -> None:
    if "split" in x and (x.split != SPLIT).any():
        raise ValueError("R4 v2 development may read VAL rows only")
    if not set(x.domain) <= set(DOMAINS):
        raise ValueError("R4 v2 development may read the original domains only")


# --- data -------------------------------------------------------------------------------------

def frames() -> dict[str, pd.DataFrame]:
    """M5 and Q7 val frames (R3's: real, empty, P3 fakes, A1–A3, R3 Amendment 1 exclusions)."""
    w = R3.wide(R3.llm_long(), SPLIT, ("M5", "Q7"), R3.exclusions())
    for x in w.values():
        _val_only(x.assign(split=SPLIT))
    return w


def histories(x: pd.DataFrame) -> list[np.ndarray]:
    """Per window: every series value before the target period (ends with the lookback)."""
    lb = P6.lookbacks()
    series = {d: load_domain(d).set_index("start_date").value for d in DOMAINS}
    out = []
    for dom, sid in zip(x.domain, x.sample_id):
        s = series[dom]
        h = s[s.index < pd.Timestamp(sid)].to_numpy(float)
        nums = lb.loc[(dom, SPLIT, sid), "numbers"]
        if not np.allclose(h[-len(nums):], nums, equal_nan=True):
            raise ValueError(f"{dom} {sid}: history does not end with the forecaster's lookback")
        out.append(h)
    return out


def texts() -> dict[str, pd.Series]:
    """{condition: text per (domain, sample_id)} for val windows."""
    out = {}
    for c in CONDS:
        t = common.load(c)
        t = t[(t.split == SPLIT) & t.domain.isin(DOMAINS)]
        out[c] = t.set_index(["domain", "sample_id"]).text
    return out


# --- claims per (condition, window) -----------------------------------------------------------

def claims_v2(x: pd.DataFrame, tx: dict, consolidated: bool = True) -> dict[str, list[list[V.ClaimV2]]]:
    """v2's claims per condition and window — consolidated (V3, frozen) unless asked for V0's raw ones."""
    ans = V.load_answers(V2_ANSWERS, V2_VAL_INPUTS, V2_META)
    judges = {d: V.JudgeV2(d, ans, consolidated) for d in DOMAINS}
    return {c: [judges[d](tx[c].get((d, s), "")) for d, s in zip(x.domain, x.sample_id)]
            for c in CONDS}


def claims_v1j(x: pd.DataFrame, tx: dict) -> dict[str, list[list[V.ClaimV2]]]:
    """Interim: v1's whole-report answers as claims (see module docstring)."""
    from src.models.r3_answers import load_judge
    j = load_judge()
    j = j.set_index(["domain", "question", j.text.str.strip()])
    out = {}
    for c in CONDS:
        rows = []
        for d, s in zip(x.domain, x.sample_id):
            t = tx[c].get((d, s), "")
            t = t.strip() if isinstance(t, str) else ""
            cl = []
            if t:
                if d == "Health_US":
                    r = j.loc[(d, "level", t)]
                    cl.append(V.ClaimV2("level", int(r.direction), float(r.confidence),
                                        reference=stated_baseline(t)))
                else:
                    r = j.loc[(d, "change", t)]
                    cl.append(V.ClaimV2("change", int(r.direction), float(r.confidence),
                                        frame="yoy" if d == "Traffic" else "trend"))
            rows.append(cl)
        out[c] = rows
    return out


# --- gates and finals -------------------------------------------------------------------------

def gate_values(x: pd.DataFrame, claims: dict, hist: list, fraction: float, floor: float) -> dict:
    out = {}
    for c in CONDS:
        g = np.ones(len(x))
        for i, (d, nums_sig) in enumerate(zip(x.domain, x.sigma)):
            cl = claims[c][i]
            if not cl:
                continue
            L = lookback_for(d)
            h = hist[i]
            st = V.Settings(window=window_for(L, fraction), conf_floor=floor)
            per = V.PERIOD["weekly" if d in WEEKLY else "monthly"]
            g[i] = V.gate(cl, h[-L:], float(nums_sig), st, history=h, period=per).g
        out[c] = g
    return out


def metrics_row(x: pd.DataFrame, fin: dict[str, D.Defended], groups=None) -> list[dict]:
    """Pooled (and per-domain) DSS / flip / damage per attack, FBR, real-text MSE."""
    groups = groups or {"pooled": DOMAINS}
    has = x.has_text.to_numpy()
    dom = x.domain.to_numpy()
    rows = []
    for gname, ds in groups.items():
        sel = has & np.isin(dom, ds)
        row = dict(cell=gname)
        fr = fin["real"].final
        row["fbr"] = M.false_block_rate(fin["real"].weight, sel).value
        row["n_genuine"] = int(sel.sum())
        row["mse_real"] = M.mse(x.y_true, fr, x.sigma, sel).value
        for a in ATTACKS:
            d = x[f"d_{a}"].to_numpy()
            ff = fin[a].final
            row[f"dss_{a}"] = M.dss(ff, fr, d, x.sigma, sel).value
            row[f"flip_{a}"] = M.flip_rate(ff, fr, x.y_last, d, sel).value
            att = sel & (d != 0)
            pm = M.paired_mse(x.y_true, {"real": fr, "fake": ff}, x.sigma, att)
            row[f"damage_{a}"] = float(np.nanmean(M.paired_difference(pm["fake"], pm["real"])))
            row[f"n_{a}"] = int(np.isfinite(M.dss(ff, fr, d, x.sigma, sel).terms).sum())
        row["obj"] = float(np.mean([abs(row[f"dss_{a}"]) for a in OBJ_ATTACKS]))
        rows.append(row)
    return rows


def finals_v2(x: pd.DataFrame, g: dict, tau: float, k: float) -> dict[str, D.Defended]:
    A, sig = x["empty"].to_numpy(float), x.sigma.to_numpy(float)
    return {c: V.defend(A, x[c].to_numpy(float), g[c], sig, tau, k) for c in CONDS}


def grid(x: pd.DataFrame, claims: dict, hist: list) -> pd.DataFrame:
    rows = []
    for f, c0 in product(FRACTIONS, FLOORS):
        g = gate_values(x, claims, hist, f, c0)
        for tau, k in product(TAUS, CAPS):
            r = metrics_row(x, finals_v2(x, g, tau, k))[0]
            rows.append(dict(fraction=f, conf_floor=c0, tau=tau, cap_k=k, **r))
    df = pd.DataFrame(rows)
    df["cap_k_desc"] = -df.cap_k                 # tie-break: larger k first
    return df


def select(df: pd.DataFrame) -> tuple[pd.Series, bool]:
    ok = df[df.fbr <= FBR_BUDGET]
    if len(ok):
        return ok.sort_values(["obj", "fbr", *KNOBS], kind="mergesort").iloc[0], True
    return df.sort_values(["fbr", *KNOBS], kind="mergesort").iloc[0], False


def comparators(x: pd.DataFrame, gt: pd.DataFrame, fz: dict) -> dict[str, dict]:
    """Undefended, never_trust, clip k = 0.5 (frozen) and TextGuard v1 (frozen) on val."""
    A, sig = x["empty"].to_numpy(float), x.sigma.to_numpy(float)
    g1 = P6.gates_wide(gt, x, fz["textguard"]["fraction"], conditions=CONDS)
    return {
        "undefended": {c: D.always_trust(A, x[c].to_numpy(float)) for c in CONDS},
        "never_trust": {c: D.never_trust(A, x[c].to_numpy(float)) for c in CONDS},
        "clip k=0.5": {c: D.clip(A, x[c].to_numpy(float), sig, fz["clip"]["k"]) for c in CONDS},
        "TextGuard v1": {c: D.textguard(A, x[c].to_numpy(float), g1[c],
                                        threshold=fz["textguard"]["tau"], mode="block")
                         for c in CONDS},
    }


def report_reading(claims: list[V.ClaimV2], kind: str = "change", literal: bool = True,
                   domain: str = "") -> str:
    """A report's overall reading from its unit claims: up / down / mixed / none."""
    pol = 1 if not literal else V.POLARITY.get(domain, 1)
    ds = {c.direction * pol for c in claims if c.kind == kind and c.direction}
    return {frozenset({1}): "up", frozenset({-1}): "down", frozenset(): "none"}.get(
        frozenset(ds), "mixed")


def _sign_reading(claims: list[V.ClaimV2], domain: str, how: str) -> str:
    """Report-level reading (literal subject) from unit change claims, comparable with P1's labels.

    ``majority``: sign of the confidence-weighted sum of the units' change directions (0 → none);
    ``first``: the first unit (in report order) with a non-none change claim — the headline claim.
    """
    pol = V.POLARITY.get(domain, 1)
    ch = [c for c in claims if c.kind == "change" and c.direction]
    if not ch:
        return "none"
    if how == "first":
        s = ch[0].direction
    else:
        s = float(np.sign(sum(c.direction * c.confidence for c in ch)))
    return {1: "up", -1: "down", 0: "none"}[int(s * pol)]


def human_agreement(ans: pd.DataFrame) -> pd.DataFrame:
    """Agreement with P1's report-level human labels, on comparable bases (Aryan, 2026-10-16).

    P1 labels say what a *whole report* claims (up / down / none about the literal subject); v2
    judges *sentences*. ``union`` (any two directions → "mixed", which no label uses) is the basis
    first reported (38.7%) and is not like-for-like. ``majority`` and ``first`` turn the sentence
    claims into one report-level reading; ``v3`` uses the frozen consolidated claims (a baseline-
    stating sentence → level only, so Health_US level reports read "none", as labelled). v1's
    whole-report change answer (P6's judge run) is scored on the same reports.
    """
    from src.textguard.export_judge_inputs import HUMAN_LABELS
    from src.textguard.text_direction import load_judgements
    j = load_judgements()
    j = j[j.question == "change"].assign(key=lambda d: d.text.str.strip()).set_index(["domain", "key"])
    rows = []
    for p in HUMAN_LABELS:
        h = pd.read_csv(p, dtype=str)
        h = h[h.domain.isin(DOMAINS) & h.fact.notna()]
        for d, t, m in zip(h.domain, h.fact, h.manual):
            raw = V.JudgeV2(d, ans, consolidated=False)(t)
            v3 = V.consolidate(raw)
            lit = int(j.loc[(d, t.strip()), "literal"])
            reads = {"v2 raw — union (first reported)": report_reading(raw, domain=d),
                     "v2 raw — majority": _sign_reading(raw, d, "majority"),
                     "v2 raw — first claim": _sign_reading(raw, d, "first"),
                     "v2 V3 — majority": _sign_reading(v3, d, "majority"),
                     "v2 V3 — first claim": _sign_reading(v3, d, "first"),
                     "v1 — whole report (P6 run)": {1: "up", -1: "down", 0: "none"}[lit]}
            for basis, r in reads.items():
                rows.append(dict(basis=basis, domain=d, source=Path(p).stem, agree=float(r == m)))
    df = pd.DataFrame(rows)
    out = [df.groupby("basis", sort=False).agree.agg(["mean", "size"]).reset_index().assign(domain="pooled (75)")]
    p50 = df[df.source == "summary_directional_sample"]
    out.append(p50.groupby("basis", sort=False).agree.agg(["mean", "size"]).reset_index().assign(domain="P1 sample (50)"))
    out.append(df.groupby(["basis", "domain"], sort=False).agree.agg(["mean", "size"]).reset_index())
    return pd.concat(out, ignore_index=True).rename(columns={"mean": "agreement", "size": "n"})


def judge_checks(tx: dict, x: pd.DataFrame, claims: dict) -> pd.DataFrame:
    """v2 judge on VAL (descriptive): A2's forward sentence, forward claims on genuine text.

    * a2_caught — share of attacked A2 texts with a claim (change or outlook) in the claimed
      direction (V3 keeps the change claim of A2's sentence).
    * real_outlook — share of genuine val texts with any non-none outlook claim after V3.
    Human-label agreement: :func:`human_agreement`.
    """
    rows = []
    has = x.has_text.to_numpy()
    d2 = x["d_a2_forward"].to_numpy()
    for i, dom in enumerate(x.domain):
        if d2[i]:
            ok = any(c.kind in ("change", "outlook") and c.direction == d2[i] and "expected to" in c.unit
                     for c in claims["a2_forward"][i])
            rows.append(dict(check="a2_caught", domain=dom, value=float(ok), n=1))
        if has[i]:
            fo = any(c.kind == "outlook" and c.direction for c in claims["real"][i])
            rows.append(dict(check="real_outlook", domain=dom, value=float(fo), n=1))
    df = pd.DataFrame(rows)
    pooled = df.groupby("check").agg(value=("value", "mean"), n=("n", "sum")).reset_index()
    per = df.groupby(["check", "domain"]).agg(value=("value", "mean"), n=("n", "sum")).reset_index()
    return pd.concat([pooled.assign(domain="pooled"), per], ignore_index=True)


def run(judge: str = "v1j", save: bool = True) -> dict[str, pd.DataFrame]:
    w = frames()
    x = w["M5"]
    hist = histories(x)
    tx = texts()
    claims = claims_v2(x, tx) if judge == "v2" else claims_v1j(x, tx)
    gr = grid(x, claims, hist)
    best, met = select(gr)
    fz, gt = R3.gates(SPLIT)
    groups = {"pooled": DOMAINS, **{d: (d,) for d in DOMAINS}}
    comp = []
    for target in ("M5", "Q7"):
        xt = w[target]
        if not np.allclose(xt.y_true, x.y_true):
            raise ValueError("M5 / Q7 val windows differ")
        defs = comparators(xt, gt, fz)
        g = gate_values(xt, claims, hist, best.fraction, best.conf_floor)
        defs[f"v2 ({judge})"] = finals_v2(xt, g, best.tau, best.cap_k)
        for name, fin in defs.items():
            for r in metrics_row(xt, fin, groups):
                comp.append(dict(target=target, defence=name, **r))
    comp = pd.DataFrame(comp)
    out = {"grid": gr.drop(columns="cap_k_desc"), "compare": comp,
           "chosen": pd.DataFrame([dict(judge=judge, budget_met=met, **best.drop("cap_k_desc"))])}
    if judge == "v2":
        out["judge"] = judge_checks(tx, x, claims)
        out["human"] = human_agreement(V.load_answers(V2_ANSWERS, V2_VAL_INPUTS, V2_META))
    if save:
        for k, v in out.items():
            v.to_csv(RESULTS / f"summary_r4_val_{k}_{judge}.csv", index=False)
    return out


FROZEN = RESULTS / "summary_r4_v2_frozen.json"


def git_content_sha(path) -> str:
    """sha256 of a git-tracked text file as git stores it (CRLF -> LF).

    Why: the judge answers came back from Kaggle with LF and were committed as LF, but a fresh
    Windows clone (``core.autocrlf=true``) checks them out as CRLF, so a raw-byte hash depended
    on the checkout and ``freeze`` refused to run (R7 clean room, 2026-10-23). On the LF file
    the frozen value was computed from, this equals the raw sha256, so the frozen file is
    unchanged. Regenerated, untracked files (the judge inputs) are still hashed raw: their
    bytes are what the Kaggle run recorded."""
    import hashlib
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def freeze() -> dict:
    """Write TextGuard v2's frozen settings (V3, val-chosen by the approved rule) — once.

    The R4 held-out analysis refuses to run without this file; it is never rewritten with other
    values (a re-run must reproduce it exactly).
    """
    import hashlib
    import json
    out = run("v2", save=True)
    c = out["chosen"].iloc[0]
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    windows = {d: window_for(lookback_for(d), float(c.fraction)) for d in DOMAINS}
    windows["held-out weekly (Energy, FNSPID; lookback 24)"] = window_for(24, float(c.fraction))
    fz = {"design": "V3 (consolidate: baseline-stating unit -> level claim only; outlook equal to its "
                    "unit's change claim dropped; multi-month-average units unchecked)",
          "decided": "Aryan 2026-10-16, after V0 missed the val FBR budget (disclosed dev iteration)",
          "fixed_on": "val only (original domains, Model 5, pooled)",
          "rule": "FBR <= 0.10, lowest mean over {flipped, a2_forward, exaggerated, a3_magnitude} of |pooled DSS|; "
                  "ties: lower FBR, smaller tau, larger k, smaller c0, smaller f",
          "n_configs": int(len(out["grid"])), "budget_met": bool(c.budget_met),
          "fraction": float(c.fraction), "conf_floor": float(c.conf_floor), "tau": float(c.tau),
          "cap_k": float(c.cap_k), "window": windows, "period": V.PERIOD,
          "val": {k: float(c[k]) for k in ("fbr", "obj", "mse_real", "dss_flipped", "dss_a2_forward",
                                            "dss_exaggerated", "dss_a3_magnitude", "dss_a1_selective")},
          "judge_answers_sha256": git_content_sha(V2_ANSWERS),  # tracked: checkout-independent
          "judge_inputs_sha256": sha(V2_VAL_INPUTS)}             # regenerated: raw, as Kaggle got it
    text = json.dumps(fz, indent=2) + "\n"
    if FROZEN.exists() and FROZEN.read_text(encoding="utf-8") != text:
        raise SystemExit(f"{FROZEN.name} exists with different content — v2's settings are frozen")
    FROZEN.write_text(text, encoding="utf-8")
    return fz


def main() -> None:
    judge = sys.argv[1] if len(sys.argv) > 1 else "v1j"
    if judge == "freeze":
        import json
        print(json.dumps(freeze(), indent=2))
        return
    if judge not in ("v1j", "v2"):
        raise SystemExit("usage: python -m src.analyze_r4_val [v1j|v2|freeze]")
    out = run(judge)
    pd.set_option("display.width", 220)
    print(out["chosen"].T.to_string())
    c = out["compare"]
    cols = ["target", "defence", "cell", "fbr", "mse_real", "obj"] + [f"dss_{a}" for a in ATTACKS]
    print(c[c.cell == "pooled"][cols].round(3).to_string(index=False))
    print(c[(c.target == "M5")][["defence", "cell", "fbr", "n_genuine"]].pivot(
        index="defence", columns="cell", values="fbr").round(3).to_string())


if __name__ == "__main__":
    main()
