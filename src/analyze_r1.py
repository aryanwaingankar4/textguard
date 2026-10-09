"""R1 — LLM vs fusion: pre-training diagnostics and the test analysis against §10d.

**Before any R1 head is trained:**

* ``embed-finbert``: FinBERT (``ProsusAI/finbert``, pinned revision) embeds every prompt the R1
  heads need — the same set the Qwen export holds (``qwen_embeddings.window_texts`` +
  ``condition_texts``) — into the shared cache, on the CPU. Truncation (512 tokens) counted.
* ``mechanism``: the pre-registered **val** mechanism check (§10d). For each encoder, P5's
  ``embedding_similarity`` on the val split: how far the flip moves the head's input, relative
  to swapping in a different report (``rel_distance``), in the standardised space that
  encoder's head would see. **Rule fixed in §10d:** an encoder is *direction-aware* if its
  pooled val mean ``rel_distance`` (flipped) is ≥ 2× BERT's on the same windows. Only selects
  between readings R1-b and R1-c; it never touches a head or a forecast.
  Qwen is included once its Colab embeddings are imported (``qwen_embeddings import``);
  until then it is reported as pending, never computed from a partial cache.

Why val: the rule was fixed before any number existed, and val is where a pre-training
diagnostic belongs; the test-split table is reported later, beside P5's BERT table.

**After training (``train_r1``) and inference (``run_r1``) — ``test``:** the pre-registered
family on the TEST split, exactly as §10d fixes it, reusing P5's metric code unchanged
(``analyze_p5.screening`` / ``attack`` / ``damage``; seed-mean per window first; windows with a
real report; attacked windows for DSS / Flip Rate; NaN pairing; σ = train std; bootstrap 10,000,
seed 0, stratified by domain):

* **S** (16) sensitivity, **A1** (40) DSS, **A2** (40) Flip-Rate McNemar for the four heads;
* **E** (8) per-window DSS_new − DSS_BERT-twin, pooled (M3-Q − M3, M4-Q − M4, M3-F − M3,
  M4-F − M4) × {flipped, exaggerated};
* **F′** (4) per-window DSS_M5 − DSS_Mk-Q, pooled × {flipped, exaggerated}.

M3, M4, M5 enter only through their saved P5 predictions. ``holm_family`` asserts exactly 108
tests. Descriptive (outside the family): usage gap, content controls, damage, failures, test
mechanism table; ablations (no 2020, dense vs sparse) with raw p.
``reading`` applies the §10d decision rules mechanically and returns R1-a / R1-b / R1-c.

Run: ``python -m src.analyze_r1 embed-finbert``, ``… mechanism``, then (after ``train_r1`` and
``run_r1``) ``python -m src.analyze_r1 test``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.text_encoder import (FINBERT_NAME, MAX_TOKENS, MODEL_NAME, QWEN_NAME,
                                     BertTextEncoder, cache_dir_for)

RESULTS = __import__("pathlib").Path(__file__).resolve().parents[1] / "results"
ENCODERS = {"BERT": MODEL_NAME, "FinBERT": FINBERT_NAME, "Qwen": QWEN_NAME}
AWARE_FACTOR = 2.0  # §10d: direction-aware ⇔ pooled val mean rel_distance ≥ 2× BERT's


def r1_texts() -> list[str]:
    """Every distinct non-blank text the R1 heads read (train/val/test windows + conditions)."""
    from src.models.late_fusion import _clean
    from src.models.qwen_embeddings import condition_texts, window_texts
    t = pd.concat([window_texts(), condition_texts()], ignore_index=True)
    return list(dict.fromkeys(x for x in (_clean(v) for v in t.text) if x is not None))


def embed_finbert() -> dict:
    enc = BertTextEncoder(FINBERT_NAME)
    texts = r1_texts()
    try:
        n_tok = enc.token_counts(texts)
        enc.encode(texts)
    finally:
        enc.unload()
    return dict(encoder=FINBERT_NAME, revision=enc.revision, texts=len(texts),
                median_tokens=int(np.median(n_tok)), max_tokens=int(n_tok.max()),
                truncated=int((n_tok > MAX_TOKENS).sum()))


def qwen_cached() -> bool:
    """All R1 prompts present in the Qwen cache (i.e. the Colab import has run)."""
    from src.models.qwen_embeddings import INPUTS, load_inputs
    if not INPUTS.exists():
        return False
    e = load_inputs()
    e = e[e.row_type == "embed"]
    d = cache_dir_for(QWEN_NAME)
    return all((d / f"{s}.npy").exists() for s in e.prompt_sha)


def mechanism(split: str = "val") -> pd.DataFrame:
    """The §10d mechanism table: one block per available encoder + the direction-aware rule."""
    from src.analyze_p5 import embedding_similarity
    parts = []
    for label, name in ENCODERS.items():
        if name == QWEN_NAME and not qwen_cached():
            print("Qwen: embeddings not imported yet -> pending (run the Colab notebook, §7c)")
            continue
        parts.append(embedding_similarity(split, name).assign(encoder=label, split=split))
    df = pd.concat(parts, ignore_index=True)
    key = (df.attack == "flipped") & (df.pair == "real vs flipped") & (df.domain == "pooled")
    bert = float(df[key & (df.encoder == "BERT")].rel_distance_mean.iloc[0])
    ratio = df[key].set_index("encoder").rel_distance_mean / bert
    df["ratio_to_bert"] = np.where(key, df.encoder.map(ratio), np.nan)
    df["direction_aware"] = np.where(key, df.ratio_to_bert >= AWARE_FACTOR, None)
    cols = ["encoder", "split", "attack", "pair", "domain", "n"]
    return df[cols + [c for c in df.columns if c not in cols]]


# --- test analysis against the pre-registered family (§10d) ----------------------------------

QWEN_M5 = "Qwen2.5-1.5B"
R1_HEADS = ("DLinear+Qwen", "PatchTST+Qwen", "DLinear+FinBERT", "PatchTST+FinBERT")
QWEN_HEADS = R1_HEADS[:2]
BERT_TWIN = {"DLinear+Qwen": "DLinear+BERT", "PatchTST+Qwen": "PatchTST+BERT",
             "DLinear+FinBERT": "DLinear+BERT", "PatchTST+FinBERT": "PatchTST+BERT"}
ALL_MODELS = ("DLinear", "PatchTST", "DLinear+BERT", "PatchTST+BERT", QWEN_M5, *R1_HEADS)
FAMILY_SIZE = {"S": 16, "A1": 40, "A2": 40, "E": 8, "F'": 4}


def load_long(split: str = "test") -> pd.DataFrame:
    """P5's predictions for Models 1–5 (unchanged) + the R1 heads, seed-averaged per window."""
    from src import analyze_p5 as P5
    from src import metrics as M
    from src.run_r1 import PREDICTIONS
    r = pd.read_csv(PREDICTIONS, dtype={"sample_id": str})
    r = M.seed_mean(r[r.split == split]).drop(columns="n_seeds")
    base = P5.load_long(split)
    return pd.concat([base, r[base.columns]], ignore_index=True)


def wide(split: str = "test") -> dict[str, pd.DataFrame]:
    from src import analyze_p5 as P5
    return P5.wide(load_long(split), split, models=ALL_MODELS)


def paired_dss(w: dict, a_model: str, b_model: str, label: str, base_mask=None) -> pd.DataFrame:
    """Per-window DSS_a − DSS_b on windows valid for both (blocks E and F′); pooled row tested
    (Wilcoxon), per-domain rows descriptive — as P5's block F."""
    from src import analyze_p5 as P5
    from src import metrics as M
    rows = []
    for att in P5.ATTACKS:
        terms = []
        for model in (a_model, b_model):
            x = w[model]
            mask = x.has_text.to_numpy() if base_mask is None else x.has_text.to_numpy() & base_mask
            terms.append(M.dss(x[att], x.real, x[f"d_{att}"], x.sigma, mask))
        diff = M.paired_difference(*terms)
        x0 = w[a_model]
        dom = x0.domain.to_numpy()
        for g, sel in P5._groups(x0, P5.GROUPS_MAIN):
            c = P5._cell(diff, dom, sel)
            ok = sel & np.isfinite(diff)
            rows.append(dict(comparison=label, model_a=a_model, model_b=b_model, condition=att,
                             domain=g, dss_a=P5._cell(terms[0].terms, dom, ok)["value"],
                             dss_b=P5._cell(terms[1].terms, dom, ok)["value"],
                             diff=c["value"], ci_low=c["ci_low"], ci_high=c["ci_high"],
                             n=c["n"], p_raw=M.wilcoxon(np.where(sel, diff, np.nan))["p"]
                             if g == "pooled" else np.nan))
    return pd.DataFrame(rows)


def comparisons(w: dict, base_mask=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    from src import analyze_p5 as P5
    enc = pd.concat([paired_dss(w, m, BERT_TWIN[m],
                                f"{P5.SHORT[m]} - {P5.SHORT[BERT_TWIN[m]]}", base_mask)
                     for m in R1_HEADS], ignore_index=True)
    fus = pd.concat([paired_dss(w, QWEN_M5, m, f"M5 - {P5.SHORT[m]}", base_mask)
                     for m in QWEN_HEADS], ignore_index=True)
    return enc, fus


def holm_family(scr: pd.DataFrame, att: pd.DataFrame, enc: pd.DataFrame,
                fus: pd.DataFrame) -> pd.DataFrame:
    """The 108 pre-registered tests (§10d), Holm-adjusted together. Asserts the count."""
    from src import metrics as M
    s = scr[(scr.quantity == "sensitivity") & (scr.condition == "real") & (scr.domain != "pooled")]
    s = s.assign(block="S", stat="sensitivity", estimate=s.value, sign=s.value)
    a = att[att.model.isin(R1_HEADS)]
    a1 = a.assign(block="A1", stat="dss", estimate=a.dss, p_raw=a.dss_p_raw, sign=a.dss)
    a2 = a.assign(block="A2", stat="flip_rate", estimate=a.flip_rate, p_raw=a.flip_p_raw,
                  sign=a.toward - a.away)
    parts = [s, a1, a2]
    for block, df in (("E", enc), ("F'", fus)):
        d = df[df.domain == "pooled"]
        parts.append(d.assign(block=block, stat="dss_diff", estimate=d["diff"],
                              model=d.comparison, sign=d["diff"]))
    cols = ["block", "model", "condition", "domain", "stat", "estimate", "sign", "p_raw"]
    fam = pd.concat([p[cols] for p in parts], ignore_index=True)
    counts = fam.block.value_counts().to_dict()
    if counts != FAMILY_SIZE or fam.p_raw.isna().any():
        raise ValueError(f"family is not the pre-registered 108 tests: {counts}")
    fam["p_holm"] = M.holm(fam.p_raw)
    fam["reject"] = fam.p_holm < 0.05
    fam["supports_claim"] = fam.reject & (fam.pop("sign") > 0)
    return fam


def attach_holm(fam, scr, att, enc, fus) -> None:
    key = ["model", "condition", "domain"]
    s = fam[fam.block == "S"].set_index(key).p_holm
    scr["p_holm"] = [s.get((m, c, d), np.nan) if q == "sensitivity" else np.nan
                     for m, c, d, q in zip(scr.model, scr.condition, scr.domain, scr.quantity)]
    for block, col in (("A1", "dss_p_holm"), ("A2", "flip_p_holm")):
        b = fam[fam.block == block].set_index(key).p_holm
        att[col] = [b.get(k, np.nan) for k in zip(att.model, att.condition, att.domain)]
    for block, df in (("E", enc), ("F'", fus)):
        b = fam[fam.block == block].set_index(key).p_holm
        df["p_holm"] = [b.get(k, np.nan) for k in zip(df.comparison, df.condition, df.domain)]


def reading(fam: pd.DataFrame, mech_val: pd.DataFrame) -> dict:
    """§10d's decision rules, applied mechanically (nothing here is a judgement call)."""
    from src import analyze_p5 as P5
    a1 = fam[(fam.block == "A1") & (fam.condition == "flipped") & (fam.domain == "pooled")]
    fooled = {P5.SHORT[m]: bool(a1[a1.model == m].supports_claim.iloc[0]) for m in R1_HEADS}
    q_fooled = [m for m in ("M3-Q", "M4-Q") if fooled[m]]
    key = ((mech_val.encoder == "Qwen") & (mech_val.attack == "flipped")
           & (mech_val.pair == "real vs flipped") & (mech_val.domain == "pooled"))
    aware = bool(mech_val[key].direction_aware.iloc[0])
    fp = fam[(fam.block == "F'") & (fam.condition == "flipped")]
    amplifies = bool((fp.reject & (fp.estimate > 0)).all())
    if q_fooled:
        path = "R1-a"
        text = ("Qwen-embedding late fusion is fooled -> the vulnerability follows the LLM's "
                "text understanding; early fusion is not necessary. "
                + ("F' rejects for both heads with DSS_M5 > DSS_Q: early fusion amplifies."
                   if amplifies else "F': no difference detected between M5 and the Qwen heads "
                   "for at least one head (not equivalence)."))
    elif aware:
        path, text = "R1-b", ("Not fooled, Qwen embeddings direction-aware -> the vulnerability "
                              "follows early fusion (in-prompt reading).")
    else:
        path, text = "R1-c", ("Not fooled, Qwen embeddings direction-blind (val rule) -> "
                              "inconclusive between the two explanations: mean pooling discards "
                              "the direction, so a pooled-embedding late head cannot separate them.")
    f_fooled = [m for m in ("M3-F", "M4-F") if fooled[m]]
    return dict(path=path, reading=text, fooled=fooled, qwen_heads_fooled=q_fooled,
                qwen_direction_aware_val=aware, early_fusion_amplifies=amplifies,
                finbert_heads_fooled=f_fooled)


def textguard_v1(w: dict, heads) -> pd.DataFrame:
    """§10d, descriptive, only for heads that are fooled: TextGuard v1 **exactly as frozen in
    P6** (``summary_p6_frozen.json`` — soft gate τ = 0, w 3/6) and the P6 baselines, applied to
    the head's saved forecasts. The gate depends only on the text (Qwen judge answers already in
    git), so nothing is re-tuned. Outside the family; per-domain + pooled, CIs."""
    from src import analyze_p6 as P6
    fz = P6.load_frozen()
    f = fz["textguard"]["fraction"]
    gt = P6.gate_table("test", fractions=(f,))
    rows = []
    for model in heads:
        x = w[model]
        g = P6.gates_wide(gt, x, f)
        specs = P6._specs(fz, g, g, g)[:5]   # none, always, never, clip (frozen k), TextGuard
        for att in ("flipped", "exaggerated"):
            terms = P6._all_terms(x, specs, att)
            for lab, role, _ in specs:
                for r in P6._summ({q: terms[lab][q] for q in P6.QUANTS}, x, P6.GROUPS_MAIN):
                    rows.append(dict(model=model, attack=att, defence=lab, role=role, **r))
    return pd.DataFrame(rows)


def failures_r1() -> pd.DataFrame:
    from src import metrics as M
    from src.run_r1 import PREDICTIONS
    p = pd.read_csv(PREDICTIONS, dtype={"sample_id": str})
    return M.failure_rates(p, by=("model", "split", "condition"))


def run_test(save: bool = True) -> dict[str, pd.DataFrame]:
    from src import analyze_p5 as P5
    w = wide("test")
    scr = P5.screening(w, models=R1_HEADS)
    att = P5.attack(w, models=R1_HEADS)
    enc, fus = comparisons(w)
    fam = holm_family(scr, att, enc, fus)
    attach_holm(fam, scr, att, enc, fus)
    keep = P5.year_mask(w[ALL_MODELS[0]], 2020)
    dens = P5.attack(w, groups=P5.GROUPS_DENSITY, models=R1_HEADS)
    ds = P5.screening(w, groups=P5.GROUPS_DENSITY, models=R1_HEADS)
    out = dict(screening=scr, attack=att, encoder=enc, fusion=fus, holm_family=fam,
               damage=P5.damage(w, models=R1_HEADS), failures=failures_r1(),
               screening_no2020=P5.screening(w, keep, models=R1_HEADS),
               attack_no2020=P5.attack(w, keep, models=R1_HEADS),
               density=pd.concat([ds[(ds.quantity == "sensitivity") & (ds.condition == "real")]
                                  .assign(block="S"), dens.assign(block="A")],
                                 ignore_index=True))
    enc2, fus2 = comparisons(w, keep)
    out.update(encoder_no2020=enc2, fusion_no2020=fus2)
    from src.analyze_p5 import embedding_similarity
    out["mechanism_test"] = pd.concat(
        [embedding_similarity("test", n).assign(encoder=lbl, split="test")
         for lbl, n in ENCODERS.items()], ignore_index=True)
    if save:
        for name, df in out.items():
            df.to_csv(RESULTS / f"summary_r1_{name}.csv", index=False, float_format="%.6g")
    return out


def main() -> None:
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["embed-finbert", "mechanism", "test"])
    a = ap.parse_args()
    if a.stage == "embed-finbert":
        print(embed_finbert())
        return
    if a.stage == "test":
        out = run_test()
        mech = pd.read_csv(RESULTS / "summary_r1_mechanism_val.csv")
        r = reading(out["holm_family"], mech)
        (RESULTS / "summary_r1_reading.json").write_text(json.dumps(r, indent=1))
        from src import analyze_p5 as P5
        fooled = [m for m in R1_HEADS if r["fooled"][P5.SHORT[m]]]
        if fooled:
            out["textguard_v1"] = textguard_v1(wide("test"), fooled)
            out["textguard_v1"].to_csv(RESULTS / "summary_r1_textguard_v1.csv", index=False,
                                       float_format="%.6g")
        with pd.option_context("display.width", 250, "display.max_rows", 500,
                               "display.max_columns", 40, "display.float_format", "{:.4g}".format):
            for name, df in out.items():
                print(f"\n=== summary_r1_{name}.csv ({len(df)} rows) ===")
                print(df.to_string(index=False))
        print(json.dumps(r, indent=1))
        return
    df = mechanism("val")
    out = RESULTS / "summary_r1_mechanism_val.csv"
    df.to_csv(out, index=False)
    show = df[(df.domain == "pooled")][["encoder", "attack", "pair", "n", "cos_raw_mean",
                                        "cos_std_mean", "rel_distance_mean",
                                        "rel_distance_median", "ratio_to_bert",
                                        "direction_aware"]]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(show.to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
