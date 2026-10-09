"""P5b analysis: screening, attack, fusion comparison, damage and ablations.

**Inputs — nothing is re-run.** Models 1–4: ``results/p5_predictions.csv`` (P5a). Model 5:
``run_experiments.qwen_rows()`` (P2.4 + P5 Colab answers, one verified software setup).
This module only reads predictions and applies the P4 metrics, so every number below can be
regenerated in seconds from the saved raw predictions.

**What is fixed in advance (PROJECT_STATUS §10a), and why it matters here.**

* **Test split only.** Val is reserved for P6 tuning; reporting on it would leave nothing
  untouched to tune on.
* **Seeds first:** Models 1–4 are averaged over seeds per window (``metrics.seed_mean``)
  before any metric — the unit of analysis is the window, not the (window, seed) pair, so
  three near-identical copies of a window cannot fake a 3× larger sample.
* **Windows:** screening and attack use only test windows whose *real* report is non-empty
  (299 of 326). A window with no report has nothing to attack or compare. Attack metrics
  further keep only windows the P3 rules actually attacked (d ≠ 0) — coverage is reported.
* **NaN pairing** is inside every metric: a Model 5 format failure on either side drops the
  window from both. Because that could hide attack effect (a fake text that causes a unit
  slip vanishes from DSS), ``nan_pairing`` shows exactly which windows were lost and an
  exploratory unit-corrected recomputation.
* **One Holm family of exactly 76 tests** (S 12 + A1 30 + A2 30 + F 4). ``holm_family``
  asserts the count, so a test cannot be silently added or dropped.
* **Ablations** (2020 excluded; dense- vs sparse-text domains) and every other number are
  reported with raw p / CIs only, outside the family.

Outputs: ``results/summary_p5_*.csv`` (small, committable) and the two paper figures in
``paper/figures/`` (``python -m src.analyze_p5``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import metrics as M
from src.attacks import common
from src.models.train_baselines import DOMAINS
from src.run_experiments import PREDICTIONS, RESULTS, ROOT, qwen_rows

TEXT_BLIND = ("DLinear", "PatchTST")
TEXT_MODELS = ("DLinear+BERT", "PatchTST+BERT", "Qwen2.5-1.5B")
MODELS = TEXT_BLIND + TEXT_MODELS
SHORT = {"DLinear": "M1", "PatchTST": "M2", "DLinear+BERT": "M3", "PatchTST+BERT": "M4",
         "Qwen2.5-1.5B": "M5"}
FUSION = {"DLinear": "none", "PatchTST": "none", "DLinear+BERT": "late",
          "PatchTST+BERT": "late", "Qwen2.5-1.5B": "early"}
CONTROLS = ("shuffled", "cross_domain", "time_shifted")
ATTACKS = ("flipped", "exaggerated")
SCREEN_CONDITIONS = ("real", "empty") + CONTROLS
DENSE, SPARSE = ("Economy", "Traffic"), ("SocialGood", "Health_US")

# a "group" = the windows a row is computed over; pooled CIs are stratified by domain
GROUPS_MAIN = {**{d: (d,) for d in DOMAINS}, "pooled": tuple(DOMAINS)}
GROUPS_DENSITY = {"dense (Economy+Traffic)": DENSE, "sparse (SocialGood+Health_US)": SPARSE}

N_BOOT, BOOT_SEED = 10_000, 0
FIGURES = ROOT / "paper" / "figures"
QWEN = "Qwen2.5-1.5B"
UNIT_RESCALE = 1000.0  # Model 5's scale failures answer in billions for a millions series


# --- data ------------------------------------------------------------------------------------

def load_long(split: str = "test") -> pd.DataFrame:
    """Every model's predictions on ``split``, one row per (model, domain, condition, window).

    Models 1–4 are seed-averaged here (a window that failed in any seed stays NaN); Model 5
    is its single greedy run.
    """
    p = pd.read_csv(PREDICTIONS, dtype={"sample_id": str})
    p = M.seed_mean(p[p.split == split]).drop(columns="n_seeds")
    q = qwen_rows()
    q = q[q.split == split][[c for c in p.columns]]
    return pd.concat([p, q], ignore_index=True)


def window_index(split: str = "test") -> pd.DataFrame:
    """The fixed window order every model is aligned to: domain (DOMAINS order), then date."""
    r = common.load("real")
    r = r[(r.split == split) & r.domain.isin(DOMAINS)][["domain", "sample_id", "has_text"]]
    r["_d"] = r.domain.map(DOMAINS.index)
    return r.sort_values(["_d", "sample_id"]).drop(columns="_d").reset_index(drop=True)


def wide(long: pd.DataFrame, split: str = "test") -> dict[str, pd.DataFrame]:
    """One frame per model, rows aligned to ``window_index``: a prediction column per
    condition plus y_true, y_last, sigma (train std), has_text (real report present) and
    the attack direction d_<attack> (P3's ``claimed_direction``; 0 = not attacked)."""
    idx = window_index(split)
    keys = ["domain", "sample_id"]
    out = {}
    for model in MODELS:
        m = long[long.model == model]
        preds = m.pivot(index=keys, columns="condition", values="y_pred")
        base = m[m.condition == "real"].set_index(keys)[["y_true", "y_last", "train_std"]]
        w = idx.join(base, on=keys).join(preds, on=keys).rename(columns={"train_std": "sigma"})
        for a in ATTACKS:
            att = m[m.condition == a].set_index(keys)
            d = att.claimed_direction.where(att.attacked, 0).astype(int)
            w[f"d_{a}"] = w.set_index(keys).index.map(d).to_numpy()
        missing = set(SCREEN_CONDITIONS + ATTACKS) - set(w.columns)
        if missing or w[["y_true", "y_last", "sigma"]].isna().any().any():
            raise ValueError(f"{model}: conditions {missing} or windows missing from the "
                             "predictions")
        out[model] = w
    ref = out[MODELS[0]]
    for model, w in out.items():   # every model must describe the same windows
        for c in ("y_true", "y_last", "sigma", "d_flipped", "d_exaggerated"):
            if not np.allclose(w[c], ref[c]):
                raise ValueError(f"{model}: {c} differs from {MODELS[0]} — misaligned windows")
    for model in TEXT_BLIND:       # Models 1–2: asserted, not tested (§10a)
        w = out[model]
        if not M.is_text_blind(w.real, w["empty"], *(w[c] for c in CONTROLS + ATTACKS)):
            raise ValueError(f"{model} is not text-blind")
    return out


def year_mask(w: pd.DataFrame, exclude: int) -> np.ndarray:
    """True for windows whose target period is not in calendar year ``exclude`` (issue #23)."""
    return w.sample_id.str[:4].astype(int).to_numpy() != exclude


# --- cell helpers ----------------------------------------------------------------------------

def _cell(terms: np.ndarray, domains: np.ndarray, sel: np.ndarray) -> dict:
    """Mean of the valid per-window terms in ``sel`` with a domain-stratified bootstrap CI."""
    t = np.where(sel, terms, np.nan)
    ok = np.isfinite(t)
    lo, hi = M.bootstrap_ci(t, N_BOOT, seed=BOOT_SEED, strata=domains)
    return dict(value=float(t[ok].mean()) if ok.any() else np.nan, n=int(ok.sum()),
                ci_low=lo, ci_high=hi)


def _groups(w: pd.DataFrame, groups: dict) -> list[tuple[str, np.ndarray]]:
    return [(g, w.domain.isin(ds).to_numpy()) for g, ds in groups.items()]


# --- 1. failures -----------------------------------------------------------------------------

def failures() -> pd.DataFrame:
    """Failure (NaN) counts per model × split × domain × condition, from the raw rows.

    Models 1–4 counted per (window, seed) row; Model 5 split into parse and format (unit)
    failures (issue #27). Every cell is listed, including zeros, so "no failures" is visible.
    """
    p = pd.read_csv(PREDICTIONS, dtype={"sample_id": str})
    q = qwen_rows()
    rows = [M.failure_rates(p, by=("model", "split", "domain", "condition")),
            M.failure_rates(q, by=("model", "split", "domain", "condition"))]
    out = pd.concat(rows, ignore_index=True)
    out["fusion"] = out.model.map(FUSION)
    for c in ("parse_failures", "format_failures"):
        out[c] = out[c].fillna(0).astype(int) if c in out else 0
    return out[["model", "fusion", "split", "domain", "condition", "n", "failures",
                "parse_failures", "format_failures", "rate"]]


def nan_pairing(w5: pd.DataFrame) -> pd.DataFrame:
    """Which attacked Model 5 windows NaN pairing removed, and what they would have said.

    Lost windows split into: real answer failed only / fake answer failed only / both.
    **Exploratory** (not pre-registered): every scale failure observed was a value in the
    report's unit (billions) for a millions series, so ×1000 recovers the intended number.
    DSS and Flip Rate are recomputed with those answers rescaled, to bound how much attack
    effect NaN pairing could be hiding.
    """
    from src.models.qwen_forecaster import load_all_predictions
    q = load_all_predictions()
    q = q[q.split == "test"].assign(condition=lambda d: d.condition.replace({"none": "empty"}))
    raw = q.pivot(index=["domain", "sample_id"], columns="condition", values="y_pred_raw")
    bad = q.pivot(index=["domain", "sample_id"], columns="condition", values="scale_ok") == False  # noqa: E712
    fixed = raw.where(~bad, raw * UNIT_RESCALE)
    keys = w5.set_index(["domain", "sample_id"]).index
    rows = []
    for a in ATTACKS:
        d = w5[f"d_{a}"].to_numpy()
        base = w5.has_text.to_numpy() & (d != 0)
        r_ok, f_ok = np.isfinite(w5.real.to_numpy()), np.isfinite(w5[a].to_numpy())
        real_fx = fixed["real"].reindex(keys).to_numpy()
        fake_fx = fixed[a].reindex(keys).to_numpy()
        dss_main = M.dss(w5[a], w5.real, d, w5.sigma, base)
        dss_fx = M.dss(fake_fx, real_fx, d, w5.sigma, base)
        fr_main = M.flip_rate(w5[a], w5.real, w5.y_last, d, base)
        fr_fx = M.flip_rate(fake_fx, real_fx, w5.y_last, d, base)
        for g, sel in _groups(w5, GROUPS_MAIN):
            s = base & sel
            lost = s & ~(r_ok & f_ok)
            rows.append(dict(
                condition=a, domain=g, n_attacked=int(s.sum()),
                lost_real_failed_only=int((lost & ~r_ok & f_ok).sum()),
                lost_fake_failed_only=int((lost & r_ok & ~f_ok).sum()),
                lost_both_failed=int((lost & ~r_ok & ~f_ok).sum()),
                dss=_cell(dss_main.terms, w5.domain, sel)["value"],
                dss_rescaled_exploratory=_cell(dss_fx.terms, w5.domain, sel)["value"],
                flip_rate=_cell(fr_main.terms, w5.domain, sel)["value"],
                flip_rate_rescaled_exploratory=_cell(fr_fx.terms, w5.domain, sel)["value"],
                n_rescaled=int(_cell(dss_fx.terms, w5.domain, sel)["n"])))
    return pd.DataFrame(rows)


def nan_pairing_windows(w5: pd.DataFrame) -> pd.DataFrame:
    """Per-window detail of the attacked windows NaN pairing dropped (Model 5)."""
    from src.models.qwen_forecaster import load_all_predictions
    q = load_all_predictions()
    q = q[q.split == "test"]
    raw = q.pivot(index=["domain", "sample_id"], columns="condition", values="y_pred_raw")
    rows = []
    for a in ATTACKS:
        d = w5[f"d_{a}"].to_numpy()
        lost = w5.has_text.to_numpy() & (d != 0) & ~(np.isfinite(w5.real) & np.isfinite(w5[a]))
        for i in np.flatnonzero(lost):
            r = w5.iloc[i]
            rows.append(dict(condition=a, domain=r.domain, sample_id=r.sample_id,
                             d=int(d[i]), y_last=r.y_last, y_true=r.y_true,
                             real_raw=raw.loc[(r.domain, r.sample_id), "real"],
                             fake_raw=raw.loc[(r.domain, r.sample_id), a]))
    return pd.DataFrame(rows)


# --- 2. screening ----------------------------------------------------------------------------

def screening(w: dict, base_mask=None, groups=GROUPS_MAIN) -> pd.DataFrame:
    """Does text move the forecast, and does *relevant, timely* text beat the controls?

    quantity:
    * ``sensitivity``   mean |ŷ_cond − ŷ_none|/σ — how far that text moves the forecast.
      For ``real`` this is the pre-registered S statistic (Wilcoxon, block S).
    * ``moves_share``   share of windows where the text changes the forecast at all.
    * ``dist_to_real``  mean |ŷ_cond − ŷ_real|/σ — the **content test**: if a control moves
      the forecast as much as real text (sensitivity ≈) *and* lands where real text does
      (dist_to_real ≪ sensitivity), the model reacts to the presence of text, not its content.
    * ``mse``           normalised MSE on the same windows for all five conditions.
    * ``mse_minus_real`` MSE(cond) − MSE(real): > 0 = cond is worse than the real text.
      For ``empty`` this is the Text Usage Gap. For ``time_shifted`` < 0 is flagged (stale
      text beating timely text).
    """
    rows = []
    for model in TEXT_MODELS:
        x = w[model]
        mask = x.has_text.to_numpy() if base_mask is None else x.has_text.to_numpy() & base_mask
        dom = x.domain.to_numpy()
        terms = {}
        for c in ("real",) + CONTROLS:
            terms[("sensitivity", c)] = M.text_sensitivity(x[c], x["empty"], x.sigma, mask)
            terms[("moves_share", c)] = M.text_moves_share(x[c], x["empty"], mask)
        for c in CONTROLS:
            terms[("dist_to_real", c)] = M.text_sensitivity(x[c], x.real, x.sigma, mask)
        pm = M.paired_mse(x.y_true, {c: x[c] for c in SCREEN_CONDITIONS}, x.sigma, mask)
        for c in SCREEN_CONDITIONS:
            terms[("mse", c)] = pm[c]
            if c != "real":
                diff = M.paired_difference(pm[c], pm["real"])
                terms[("mse_minus_real", c)] = M.Metric(np.nanmean(diff), 0, terms=diff)
        for (qty, c), met in terms.items():
            for g, sel in _groups(x, groups):
                row = dict(model=model, short=SHORT[model], fusion=FUSION[model], domain=g,
                           quantity=qty, condition=c, **_cell(met.terms, dom, sel))
                if qty == "sensitivity" and c == "real":
                    row["n_nan"] = int((mask & sel & ~np.isfinite(x.real.to_numpy())).sum())
                    row["p_raw"] = M.wilcoxon(np.where(sel, met.terms, np.nan))["p"]
                rows.append(row)
    return pd.DataFrame(rows)


# --- 3. attack -------------------------------------------------------------------------------

def attack(w: dict, base_mask=None, groups=GROUPS_MAIN, models=MODELS) -> pd.DataFrame:
    """DSS and Flip Rate per model × attack × group (pre-registered blocks A1 and A2).

    DSS: Wilcoxon signed-rank on the per-window d·(ŷ_fake − ŷ_real)/σ. Flip Rate: exact
    McNemar on toward vs away turns (``metrics.mcnemar_flip``). Models 1–2 are listed (they
    are exactly 0 by assertion) but not tested. Coverage = attacked / windows with a report.
    """
    rows = []
    for model in models:
        x = w[model]
        mask = x.has_text.to_numpy() if base_mask is None else x.has_text.to_numpy() & base_mask
        dom = x.domain.to_numpy()
        for a in ATTACKS:
            d = x[f"d_{a}"].to_numpy()
            dss = M.dss(x[a], x.real, d, x.sigma, mask)
            fr = M.flip_rate(x[a], x.real, x.y_last, d, mask)
            for g, sel in _groups(x, groups):
                c_dss, c_fr = _cell(dss.terms, dom, sel), _cell(fr.terms, dom, sel)
                mc = M.mcnemar_flip(x[a], x.real, x.y_last, d, mask & sel)
                tested = model in TEXT_MODELS
                rows.append(dict(
                    model=model, short=SHORT[model], fusion=FUSION[model], condition=a,
                    domain=g, n_text=int((mask & sel).sum()),
                    n_attacked=int((mask & sel & (d != 0)).sum()),
                    coverage=(mask & sel & (d != 0)).sum() / max((mask & sel).sum(), 1),
                    n=c_dss["n"], n_nan=int((mask & sel & (d != 0)).sum()) - c_dss["n"],
                    dss=c_dss["value"], dss_ci_low=c_dss["ci_low"], dss_ci_high=c_dss["ci_high"],
                    dss_p_raw=M.wilcoxon(np.where(sel, dss.terms, np.nan))["p"] if tested else np.nan,
                    flip_rate=c_fr["value"], flip_ci_low=c_fr["ci_low"],
                    flip_ci_high=c_fr["ci_high"], toward=mc["toward"], away=mc["away"],
                    flip_p_raw=mc["p"] if tested else np.nan))
    return pd.DataFrame(rows)


# --- 4. fusion -------------------------------------------------------------------------------

def fusion(w: dict, base_mask=None) -> pd.DataFrame:
    """Block F: per-window DSS_M5 − DSS_Mk (k = 3, 4) on windows valid for both, pooled
    over domains (tested, Wilcoxon) — per-domain rows are descriptive breakdown only."""
    rows = []
    x5 = w[QWEN]
    for a in ATTACKS:
        for k in ("DLinear+BERT", "PatchTST+BERT"):
            terms = []
            for model in (QWEN, k):
                x = w[model]
                mask = x.has_text.to_numpy() if base_mask is None else x.has_text.to_numpy() & base_mask
                terms.append(M.dss(x[a], x.real, x[f"d_{a}"], x.sigma, mask))
            diff = M.paired_difference(*terms)
            for g, sel in _groups(x5, GROUPS_MAIN):
                c = _cell(diff, x5.domain.to_numpy(), sel)
                rows.append(dict(comparison=f"M5 - {SHORT[k]}", other=k, condition=a, domain=g,
                                 dss_m5=_cell(terms[0].terms, x5.domain.to_numpy(),
                                              sel & np.isfinite(diff))["value"],
                                 dss_other=_cell(terms[1].terms, x5.domain.to_numpy(),
                                                 sel & np.isfinite(diff))["value"],
                                 diff=c["value"], ci_low=c["ci_low"], ci_high=c["ci_high"],
                                 n=c["n"],
                                 p_raw=M.wilcoxon(np.where(sel, diff, np.nan))["p"]
                                 if g == "pooled" else np.nan))
    return pd.DataFrame(rows)


# --- 5. damage -------------------------------------------------------------------------------

def damage(w: dict, base_mask=None) -> pd.DataFrame:
    """MSE(fake) − MSE(real) on attacked windows (paired, same windows): how much *worse*
    the forecasts get under attack, not only that they move. Descriptive (CIs only)."""
    rows = []
    for model in MODELS:
        x = w[model]
        dom = x.domain.to_numpy()
        for a in ATTACKS:
            mask = x.has_text.to_numpy() & (x[f"d_{a}"].to_numpy() != 0)
            if base_mask is not None:
                mask &= base_mask
            pm = M.paired_mse(x.y_true, {"real": x.real, a: x[a]}, x.sigma, mask)
            diff = M.paired_difference(pm[a], pm["real"])
            for g, sel in _groups(x, GROUPS_MAIN):
                r, f, dd = (_cell(t, dom, sel) for t in (pm["real"].terms, pm[a].terms, diff))
                rows.append(dict(model=model, short=SHORT[model], fusion=FUSION[model],
                                 condition=a, domain=g, n=dd["n"], mse_real=r["value"],
                                 mse_fake=f["value"], damage=dd["value"],
                                 ci_low=dd["ci_low"], ci_high=dd["ci_high"],
                                 damage_pct=100 * dd["value"] / r["value"] if r["value"] else np.nan))
    return pd.DataFrame(rows)


# --- mechanism: what the late-fusion text heads actually see ---------------------------------

def embedding_similarity(split: str = "test") -> pd.DataFrame:
    """How different is a fake report from the real one *to BERT*? (exploratory, Models 3–4)

    Models 3–4 read text only through one mean-pooled BERT vector, z-scored per dimension with
    the domain's train-text statistics (the ``EmbeddingScaler`` in every M3/M4 checkpoint —
    recomputed here from the train texts, as ``run_experiments.check_embedding_scalers``
    verifies they are equal). If the flipped report's vector is almost the real one's, the
    heads *cannot* react to the direction, whatever they learned.

    A raw cosine alone proves nothing: raw mean-pooled BERT vectors of *any* two reports are
    nearly parallel (§7a). So each attack pair is compared with two baselines on the same
    windows — an unrelated report from the same domain (``shuffled``) and from another domain
    (``cross_domain``) — in raw space and in the standardised space the heads see.
    ``rel_distance`` = ‖z_cond − z_real‖ / ‖z_shuffled − z_real‖: how far the condition moves
    the head's input, as a fraction of swapping in a different report. Embeddings come from
    the on-disk cache only; a cache miss raises (BERT must not be loaded here).
    """
    from src.data.splits import load_splits
    from src.data.windows import lookback_for, split_windows
    from src.models.late_fusion import _clean, _encoder
    from src.models.text_encoder import MODEL_NAME
    enc = _encoder(MODEL_NAME)
    texts = {c: common.load(c) for c in ("real", "shuffled", "cross_domain") + ATTACKS}

    def emb(cond, dom):
        t = texts[cond]
        t = t[(t.split == split) & (t.domain == dom)].set_index("sample_id")
        return t, enc.encode([_clean(x) for x in t.text])

    def cos(a, b):
        return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))

    rows = []
    for dom in DOMAINS:
        train = split_windows(load_splits(dom), lookback_for(dom), horizon=1)["train"]
        tr = enc.encode([t for t in (_clean(x) for x in train.text) if t is not None])
        mean, std = tr.mean(0), tr.std(0) + 1e-6   # = the checkpoints' EmbeddingScaler
        real_t, real_e = emb("real", dom)
        base = {c: emb(c, dom)[1] for c in ("shuffled", "cross_domain")}
        for a in ATTACKS:
            att_t, att_e = emb(a, dom)
            sel = (att_t.attacked & real_t.has_text).to_numpy()
            z = lambda e: (e[sel] - mean) / std  # noqa: E731
            zr = z(real_e)
            d_shuf = np.linalg.norm(z(base["shuffled"]) - zr, axis=1)
            for cond, e in ((a, att_e), ("shuffled", base["shuffled"]),
                            ("cross_domain", base["cross_domain"])):
                rows.append(dict(attack=a, domain=dom, pair=f"real vs {cond}",
                                 n=int(sel.sum()), cos_raw=cos(real_e[sel], e[sel]),
                                 cos_std=cos(zr, z(e)),
                                 rel_distance=np.linalg.norm(z(e) - zr, axis=1) / d_shuf))
    if enc.is_loaded:
        raise RuntimeError("BERT was loaded — an embedding was missing from the cache")
    df = pd.DataFrame(rows)
    out = []
    for (a, pair), g in df.groupby(["attack", "pair"], sort=False):
        for dom, gg in [*((d, g[g.domain == d]) for d in DOMAINS), ("pooled", g)]:
            v = {k: np.concatenate(gg[k].to_list()) for k in ("cos_raw", "cos_std",
                                                               "rel_distance")}
            out.append(dict(attack=a, pair=pair, domain=dom, n=len(v["cos_raw"]),
                            **{f"{k}_mean": float(x.mean()) for k, x in v.items()},
                            **{f"{k}_median": float(np.median(x)) for k, x in v.items()},
                            cos_raw_min=float(v["cos_raw"].min())))
    return pd.DataFrame(out)


# --- Holm over the pre-registered family -----------------------------------------------------

def holm_family(scr: pd.DataFrame, att: pd.DataFrame, fus: pd.DataFrame) -> pd.DataFrame:
    """The 76 pre-registered tests (§10a), Holm-adjusted together. Asserts the count."""
    s = scr[(scr.quantity == "sensitivity") & (scr.condition == "real") & (scr.domain != "pooled")]
    # ``sign`` = the direction a rejection points: S and F by their estimate; A1 by DSS;
    # A2 by toward − away (a Flip Rate is ≥ 0 by construction, so its sign says nothing)
    s = s.assign(block="S", stat="sensitivity", estimate=s.value, sign=s.value)
    a = att[att.model.isin(TEXT_MODELS)]
    a1 = a.assign(block="A1", stat="dss", estimate=a.dss, p_raw=a.dss_p_raw, sign=a.dss)
    a2 = a.assign(block="A2", stat="flip_rate", estimate=a.flip_rate, p_raw=a.flip_p_raw,
                  sign=a.toward - a.away)
    f = fus[fus.domain == "pooled"]
    f = f.assign(block="F", stat="dss_diff", estimate=f["diff"], model=f.comparison,
                 sign=f["diff"])
    cols = ["block", "model", "condition", "domain", "stat", "estimate", "sign", "p_raw"]
    fam = pd.concat([s[cols], a1[cols], a2[cols], f[cols]], ignore_index=True)
    counts = fam.block.value_counts().to_dict()
    if counts != {"S": 12, "A1": 30, "A2": 30, "F": 4} or fam.p_raw.isna().any():
        raise ValueError(f"family is not the pre-registered 76 tests: {counts}")
    fam["p_holm"] = M.holm(fam.p_raw)
    fam["reject"] = fam.p_holm < 0.05
    # a directional claim needs the right sign too (§10a): a significant negative DSS is
    # "pushed away from the claim", not support
    fam["supports_claim"] = fam.reject & (fam.pop("sign") > 0)
    return fam


def attach_holm(fam: pd.DataFrame, scr, att, fus):
    """Copy the adjusted p-values back into the tables they came from."""
    key = ["model", "condition", "domain"]
    s = fam[fam.block == "S"].set_index(key).p_holm
    scr["p_holm"] = [s.get((m, c, d), np.nan) if q == "sensitivity" else np.nan
                     for m, c, d, q in zip(scr.model, scr.condition, scr.domain, scr.quantity)]
    for block, col in (("A1", "dss_p_holm"), ("A2", "flip_p_holm")):
        b = fam[fam.block == block].set_index(key).p_holm
        att[col] = [b.get(k, np.nan) for k in zip(att.model, att.condition, att.domain)]
    f = fam[fam.block == "F"].set_index(key).p_holm
    fus["p_holm"] = [f.get(k, np.nan) for k in zip(fus.comparison, fus.condition, fus.domain)]


# --- 6. ablations ----------------------------------------------------------------------------

def density(w: dict) -> pd.DataFrame:
    """Ablation 2 (issue #9): pooled S, A1, A2 over dense- vs sparse-text domains, raw p."""
    s = screening(w, groups=GROUPS_DENSITY)
    s = s[(s.quantity == "sensitivity") & (s.condition == "real")]
    s = s.assign(block="S", estimate=s.value)[["block", "model", "short", "fusion", "domain",
                                               "condition", "estimate", "ci_low", "ci_high",
                                               "n", "p_raw"]]
    a = attack(w, groups=GROUPS_DENSITY, models=TEXT_MODELS)
    a1 = a.assign(block="A1", estimate=a.dss, ci_low=a.dss_ci_low, ci_high=a.dss_ci_high,
                  p_raw=a.dss_p_raw)
    a2 = a.assign(block="A2", estimate=a.flip_rate, ci_low=a.flip_ci_low,
                  ci_high=a.flip_ci_high, p_raw=a.flip_p_raw)
    cols = list(s.columns)
    return pd.concat([s, a1[cols], a2[cols]], ignore_index=True)


# --- run -------------------------------------------------------------------------------------

def run(save: bool = True) -> dict[str, pd.DataFrame]:
    w = wide(load_long("test"))
    out = {"failures": failures(), "nan_pairing": nan_pairing(w[QWEN]),
           "nan_pairing_windows": nan_pairing_windows(w[QWEN])}
    scr, att, fus = screening(w), attack(w), fusion(w)
    fam = holm_family(scr, att, fus)
    attach_holm(fam, scr, att, fus)
    out.update(screening=scr, attack=att, fusion=fus, damage=damage(w), holm_family=fam)
    keep = year_mask(w[MODELS[0]], 2020)
    out.update(screening_no2020=screening(w, keep), attack_no2020=attack(w, keep),
               fusion_no2020=fusion(w, keep), damage_no2020=damage(w, keep),
               density=density(w), embedding_similarity=embedding_similarity())
    if save:
        for name, df in out.items():
            df.to_csv(RESULTS / f"summary_p5_{name}.csv", index=False, float_format="%.6g")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    out = run(save=not args.no_save)
    if not args.no_figures:
        from src.figures_p5 import make_figures
        make_figures(out["attack"])
    with pd.option_context("display.width", 250, "display.max_rows", 500,
                           "display.max_columns", 40, "display.float_format", "{:.4g}".format):
        for name, df in out.items():
            print(f"\n=== summary_p5_{name}.csv ({len(df)} rows) ===")
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()
