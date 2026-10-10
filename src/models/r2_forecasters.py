"""R2 forecasters (pre-registration PROJECT_STATUS §10e) — load and verify the Colab answers.

Five early-fusion LLM forecasters, all answered on a Colab T4 by the notebooks built by
``notebooks/build_r2_notebooks.py`` from P5's input table (``qwen_inputs_p5.csv``):

=========  =======================================  =========  ==============================
name       model @ pinned revision                   precision  prompt
=========  =======================================  =========  ==============================
Q7         Qwen2.5-7B-Instruct @ a09a354             4-bit NF4  M5's (variant ``p5``)
P35        Phi-3.5-mini-instruct @ 2fe1924            fp16       M5's (``p5``)
M5-c1      Qwen2.5-1.5B-Instruct @ 989aa79 (= M5)     fp16       report first (``c1``)
M5-c2      same                                      fp16       reworded (``c2``)
M5-sk      same                                      fp16       skeptical (``sk``)
M5-regen   same — only if the probe failed           fp16       M5's (``p5``), regenerated
=========  =======================================  =========  ==============================

Why a verifier and not a plain ``read_csv``: every R2 comparison is only meaningful if the
answers come from the pre-registered model, revision, precision and prompt. So the loader
refuses a file whose revision differs from the pin, recomputes every prompt fingerprint from
the local inputs with the pre-registered prompt builder, requires one software setup per file
and a complete **test** split (val is optional, §10e), and applies M5's off-scale rule
(``qwen_forecaster.scale_ok``) at load, exactly as for Model 5.

**Model 5 variants and reuse (§2, §10e).** Prompts byte-identical to M5's (the ``none`` rows of
c1 and sk) were not regenerated when the 40-prompt probe reproduced P5 exactly; they are filled
here from M5's answers — and only then. If the probe failed, the run regenerated M5's own prompt
for every row; those rows load as ``M5-regen`` and R2-c / R2-d compare against them instead of
P5's M5.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.export_qwen_inputs import INPUTS_P5, OUT_DIR
from src.models.qwen_forecaster import scale_ok
from src.models.qwen_prompt import build_prompt, prompt_sha
from src.models.r2_prompts import build_variant_prompt, variant_prompt_sha

KEYS = ["domain", "split", "sample_id", "condition"]

# Pinned in §10e — the notebook builder reads these too (one source of truth).
RUN_SPECS = {
    "qwen7b": dict(model_id="Qwen/Qwen2.5-7B-Instruct",
                   revision="a09a35458c702b33eeacc393d103063234e8bc28",
                   quant=("nf4",), variants=("p5",),
                   # §10e Amendment 1: Q7 runs on Kaggle; a Colab file is still recognised, never mixed
                   files=(("nf4_kaggle", "kaggle"), ("nf4", "colab"))),
    "phi35": dict(model_id="microsoft/Phi-3.5-mini-instruct",
                  revision="2fe192450127e6a83f7441aef6e3ca586c338b77",
                  quant=("fp16", "nf4"),             # nf4 only via the pre-registered fallback
                  variants=("p5",), files=(("fp16", "colab"), ("nf4", "colab"))),
    "m5prompts": dict(model_id="Qwen/Qwen2.5-1.5B-Instruct",
                      revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
                      quant=("fp16",), variants=("c1", "c2", "sk"), files=(("fp16", "colab"),)),
}
MODEL_NAME = {("qwen7b", "p5"): "Q7", ("phi35", "p5"): "P35", ("m5prompts", "c1"): "M5-c1",
              ("m5prompts", "c2"): "M5-c2", ("m5prompts", "sk"): "M5-sk",
              ("m5prompts", "p5"): "M5-regen"}


def files_for(run: str, out_dir: Path = OUT_DIR) -> tuple[Path, Path]:
    """The ONE predictions + meta file to use for a run — files are never combined.

    A run can have several candidate files: Phi's fp16 file or its pre-registered nf4 fallback;
    Q7's Kaggle file (§10e Amendment 1) or a Colab file from before the move. The rule: the
    candidate whose meta says the test split is complete; if two are complete, refuse (which
    one counts is a decision, not something to guess); if none is, the first existing one (and
    ``load_run`` then refuses it as incomplete)."""
    found = []
    for tag, _ in RUN_SPECS[run]["files"]:
        p = Path(out_dir) / f"r2_{run}_{tag}_predictions.csv"
        if p.exists():
            found.append((p, p.with_name(f"r2_{run}_{tag}_run_meta.json")))
    if not found:
        tag = RUN_SPECS[run]["files"][0][0]
        raise FileNotFoundError(f"no R2 predictions for {run}: put r2_{run}_{tag}_predictions.csv "
                                f"and r2_{run}_{tag}_run_meta.json in {out_dir}")
    complete = [f for f in found if f[1].exists()
                and json.loads(f[1].read_text()).get("test_complete")]
    if len(complete) > 1:
        raise ValueError(f"{run}: several complete answer files {[f[0].name for f in complete]}; "
                         "one run = one file — move the one that does not count out of results/qwen")
    return complete[0] if complete else found[0]


def _read(path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False,
                       dtype={"sample_id": str, "text": str, "raw_output": str,
                              "expected_raw": str})


def load_run(run: str, predictions: Path | str | None = None, meta: Path | str | None = None,
             inputs: Path | str = INPUTS_P5, m5: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every answer of one R2 run, verified, one row per (model, domain, split, sample_id,
    condition). ``m5`` = Model 5's answers (default: ``qwen_forecaster.load_all_predictions``),
    needed only for the reused rows of the M5-variant run."""
    spec = RUN_SPECS[run]
    if predictions is None:
        predictions, meta = files_for(run)
    m = json.loads(Path(meta).read_text())
    for k in ("model_id", "revision"):
        if m.get(k) != spec[k]:
            raise ValueError(f"{run}: {k} is {m.get(k)!r}, pre-registered {spec[k]!r}")
    if m.get("quant") not in spec["quant"]:
        raise ValueError(f"{run}: quantisation {m.get('quant')!r} not pre-registered")
    platform = m.get("platform", "colab")          # the Colab notebooks do not record one
    tag = Path(predictions).name[len(f"r2_{run}_"):-len("_predictions.csv")]
    if (tag, platform) not in spec["files"]:
        raise ValueError(f"{run}: file {Path(predictions).name} with platform {platform!r} is not "
                         f"a pre-registered combination {spec['files']}")
    if not m.get("test_complete"):
        raise ValueError(f"{run}: the run's meta says the test split is incomplete — re-run "
                         "the notebook (it resumes) until cell 8 prints TEST COMPLETE")
    gen = _read(predictions)
    if gen.empty or set(gen.setup) != {m.get("setup")}:
        raise ValueError(f"{run}: predictions must come from exactly the setup in the meta file")
    if gen.duplicated(KEYS + ["variant"]).any():
        raise ValueError(f"{run}: duplicate rows")
    probe_failed = run == "m5prompts" and not m.get("reuse_m5")
    variants = list(spec["variants"]) + (["p5"] if probe_failed else [])
    if set(gen.variant) - set(variants):
        raise ValueError(f"{run}: unexpected variants {set(gen.variant) - set(variants)}")
    if run == "m5prompts" and m.get("reuse_m5") and not (
            m.get("probe_n", 0) > 0 and m.get("probe_mismatches") == 0):
        raise ValueError("M5 answers reused although the probe did not pass 40/40")
    gen["y_pred"] = pd.to_numeric(gen["y_pred"], errors="coerce")
    gen["parse_ok"] = gen["parse_ok"].astype(str) == "True"
    gen["reused_from"] = ""

    inp = _read(inputs).drop(columns=["reuse", "probe", "expected_raw"], errors="ignore")
    parts = []
    for v in variants:
        want = inp.assign(variant=v)
        want["user"] = [build_variant_prompt(v, r.description, r.unit, r.freq,
                                             json.loads(r.lookback), r.text or None)
                        for r in want.itertuples()]
        want["sha"] = [variant_prompt_sha(v, u) for u in want.user]
        got = want.merge(gen[gen.variant == v].drop(columns="variant"), on=KEYS, how="left",
                         suffixes=("", "_gen"), indicator=True)
        extra = len(gen[gen.variant == v]) - int((got._merge == "both").sum())
        if extra:
            raise ValueError(f"{run}/{v}: {extra} answers for windows not in the inputs file")
        missing = got._merge == "left_only"
        if missing.any() and run == "m5prompts" and m.get("reuse_m5") and v != "p5":
            got = _fill_from_m5(got, missing, m5)
            missing = got.prompt_sha.isna() | (got.prompt_sha == "")
        if (missing & (got.split == "test")).any():
            raise ValueError(f"{run}/{v}: {int((missing & (got.split == 'test')).sum())} test "
                             "answers missing — the run is incomplete")
        if (missing & (got.split == "val")).any():      # val is optional (§10e): all or nothing
            got = got[got.split != "val"]
        got = got.assign(parse_ok=got.parse_ok.map(lambda b: b is True or b == "True"))
        if not np.allclose(got.y_true, got.y_true_gen):
            raise ValueError(f"{run}/{v}: y_true differs between answers and inputs")
        bad = got.prompt_sha.to_numpy() != got.sha.to_numpy()
        if bad.any():
            raise ValueError(f"{run}/{v}: {int(bad.sum())} answers were made with a different "
                             "prompt than r2_prompts builds now")
        if (got.parse_ok & got.y_pred.isna()).any() or (~got.parse_ok & got.y_pred.notna()).any():
            raise ValueError(f"{run}/{v}: parse_ok disagrees with y_pred")
        parts.append(got.assign(model=MODEL_NAME[(run, v)], variant=v))
    df = pd.concat(parts, ignore_index=True)
    df["y_pred_raw"] = df["y_pred"]
    df["scale_ok"] = [scale_ok(y, json.loads(x)) for y, x in zip(df.y_pred, df.lookback)]
    df.loc[~df.scale_ok, "y_pred"] = np.nan
    df["format_ok"] = df.parse_ok & df.scale_ok
    df["prompt_text"] = df["text"]
    return df.drop(columns=["_merge", "y_true_gen", "sha", "user"])


def _fill_from_m5(got: pd.DataFrame, missing: pd.Series, m5: pd.DataFrame | None) -> pd.DataFrame:
    """Fill rows whose variant prompt is byte-identical to M5's from M5's (P5-class) answers.
    Each filled row must carry M5's fingerprint for that prompt — proof it is the same prompt."""
    if m5 is None:
        from src.models.qwen_forecaster import load_all_predictions
        m5 = load_all_predictions()
    src = m5[KEYS + ["y_pred_raw", "parse_ok", "raw_output", "prompt_sha", "setup"]].rename(
        columns={"y_pred_raw": "y_pred"})
    m5_sha = pd.Series([prompt_sha(build_prompt(r.description, r.unit, r.freq,
                                                json.loads(r.lookback), r.text or None))
                        for r in got.loc[missing].itertuples()], index=got.index[missing])
    ident = m5_sha[m5_sha == got.loc[missing, "sha"]].index   # the prompt IS M5's prompt
    got = got.copy()
    if len(ident):
        fill = got.loc[ident, KEYS].merge(src, on=KEYS, how="left", validate="one_to_one")
        fill.index = ident
        if (fill.prompt_sha.to_numpy() != got.loc[ident, "sha"].to_numpy()).any():
            raise ValueError("M5's stored answer for a reused prompt has a different fingerprint")
        for c in ("y_pred", "parse_ok", "raw_output", "prompt_sha", "setup"):
            got.loc[ident, c] = fill[c]
        got.loc[ident, "y_true_gen"] = got.loc[ident, "y_true"]
        got.loc[ident, "reused_from"] = "m5"
    got["parse_ok"] = got["parse_ok"].map(lambda b: b is True or b == "True")
    got["y_pred"] = pd.to_numeric(got["y_pred"], errors="coerce")
    return got


def load_r2(runs=tuple(RUN_SPECS), **kw) -> pd.DataFrame:
    """All R2 answers that exist, one long table (``model`` column)."""
    out = []
    for run in runs:
        try:
            files_for(run)
        except FileNotFoundError:
            continue
        out.append(load_run(run, **kw))
    if not out:
        raise FileNotFoundError("no R2 predictions in results/qwen yet")
    return pd.concat(out, ignore_index=True)
