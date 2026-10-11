"""R3 — load and verify the three Kaggle answer files (PROJECT_STATUS §7e, §10f).

Why a verifier and not a plain ``read_csv`` (as R2's ``r2_forecasters.load_run``): every R3
comparison is only meaningful if the answers come from the pre-registered model, revision and
precision, from the exact input file the attacks were exported to, and from **one** software
setup per file (§10f: real and none are regenerated in the same run, never mixed with Colab's
P5 answers). So the loader refuses a file unless:

* the meta's model id / revision / quantisation are the §10f pins and its ``input_sha256`` is the
  sha256 of today's ``results/qwen/r3_inputs.csv`` (which a re-export reproduces byte for byte);
* every row has the meta's setup string; no duplicate (window, condition, variant);
* **complete**: every exported forecast row answered for variant p5, and for sk every row but
  ``none`` (sk's no-report prompt is M5's own, so A is shared — §10f);
* every prompt fingerprint is recomputed now from the inputs with the R2 prompt builder;
* y_true equals the inputs; y_pred = M5's parser applied to the raw output; parse_ok agrees.

Then M5's off-scale rule (``qwen_forecaster.scale_ok``) is applied, exactly as for Model 5.

The judge file is checked by P6's own loader (``text_direction.load_judgements``) pointed at the
R3 files: all 2411 questions answered once, fingerprints, one setup.

Long schema out (the runner's): model, domain, split, condition, seed, sample_id, y_true, y_pred,
y_last, train_mean, train_std, has_text, attacked, claimed_direction. Model names: ``M5`` / ``Q7``
(M5's prompt, variant p5) and ``M5-sk`` / ``Q7-sk`` (the skeptical prompt). ``none`` → ``empty``,
and the sk models' ``empty`` is their p5 twin's (shared prompt).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.attacks import common
from src.models.export_qwen_inputs import OUT_DIR
from src.models.export_r3_inputs import R3_INPUTS, sha256
from src.models.qwen_forecaster import scale_ok
from src.models.qwen_prompt import parse_number
from src.models.r2_prompts import build_variant_prompt, variant_prompt_sha

KEYS = ["domain", "split", "sample_id", "condition"]
PINS = {
    "m5": dict(model_id="Qwen/Qwen2.5-1.5B-Instruct", revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
               quant="fp16", file="r3_m5_fp16_kaggle", name="M5"),
    "qwen7b": dict(model_id="Qwen/Qwen2.5-7B-Instruct", revision="a09a35458c702b33eeacc393d103063234e8bc28",
                   quant="nf4", file="r3_qwen7b_nf4_kaggle", name="Q7"),
}
JUDGE_FILE = "r3_judge_kaggle"
JUDGE_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def _inputs(path: Path = R3_INPUTS) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, dtype=str)


def load_forecaster(run: str, out_dir: Path = OUT_DIR, inputs: Path = R3_INPUTS) -> pd.DataFrame:
    """One R3 forecaster run (``m5`` or ``qwen7b``), verified, both variants, val + test."""
    pin = PINS[run]
    meta = json.loads((out_dir / f"{pin['file']}_run_meta.json").read_text(encoding="utf-8"))
    for k in ("model_id", "revision", "quant"):
        if meta.get(k) != pin[k]:
            raise ValueError(f"{run}: {k} is {meta.get(k)!r}, pre-registered {pin[k]!r}")
    if meta.get("input_sha256") != sha256(inputs):
        raise ValueError(f"{run}: answered a different input file than results/qwen/r3_inputs.csv")
    if not (meta.get("test_complete") and meta.get("val_complete")):
        raise ValueError(f"{run}: the run's meta says it is incomplete")
    gen = pd.read_csv(out_dir / f"{pin['file']}_predictions.csv", keep_default_na=False,
                      dtype={"sample_id": str, "raw_output": str, "prompt_sha": str})
    if set(gen.setup) != {meta["setup"]}:
        raise ValueError(f"{run}: answers must come from exactly the meta's setup")
    if gen.duplicated(KEYS + ["variant"]).any():
        raise ValueError(f"{run}: duplicate rows")
    inp = _inputs(inputs)
    inp = inp[inp.row_type == "forecast"]
    parts = []
    for v in ("p5", "sk"):
        want = inp if v == "p5" else inp[inp.condition != "none"]
        got = want.merge(gen[gen.variant == v], on=KEYS, how="outer", suffixes=("", "_gen"),
                         indicator=True, validate="one_to_one")
        if (got._merge != "both").any():
            raise ValueError(f"{run}/{v}: {int((got._merge != 'both').sum())} rows missing or extra")
        sha = [variant_prompt_sha(v, build_variant_prompt(v, r.description, r.unit, r.freq,
                                                          json.loads(r.lookback), r.text or None))
               for r in got.itertuples()]
        if list(sha) != list(got.prompt_sha_gen):
            raise ValueError(f"{run}/{v}: prompt fingerprints differ from today's prompt builder")
        if not np.allclose(got.y_true.astype(float), got.y_true_gen.astype(float)):
            raise ValueError(f"{run}/{v}: y_true differs from the inputs")
        parsed = np.array([parse_number(o, u) for o, u in zip(got.raw_output, got.unit)], float)
        y = pd.to_numeric(got.y_pred, errors="coerce").to_numpy(float)
        if not np.allclose(parsed, y, equal_nan=True, rtol=0, atol=0):
            raise ValueError(f"{run}/{v}: y_pred is not the parser's reading of raw_output")
        ok = got.parse_ok.astype(str) == "True"
        if (ok != np.isfinite(y)).any():
            raise ValueError(f"{run}/{v}: parse_ok disagrees with y_pred")
        parts.append(got.assign(variant=v, y_pred_raw=y, parse_ok=ok))
    df = pd.concat(parts, ignore_index=True)
    df["scale_ok"] = [scale_ok(y, json.loads(x)) for y, x in zip(df.y_pred_raw, df.lookback)]
    df["y_pred"] = np.where(df.scale_ok, df.y_pred_raw, np.nan)
    df["format_ok"] = df.parse_ok & df.scale_ok
    df["model"] = np.where(df.variant == "p5", pin["name"], pin["name"] + "-sk")
    return df.drop(columns=["_merge", "y_true_gen", "prompt_sha", "row_type", "question", "sources",
                            "expected_raw"])


def long_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The runner's long schema (``analyze_p5.wide`` reads it). sk's ``empty`` = its p5 twin's."""
    df = df.assign(condition=df.condition.replace({"none": "empty"}))
    base = df[df.variant == "p5"]
    none = base[base.condition == "empty"]
    for name in df.model.unique():
        if name.endswith("-sk"):
            df = pd.concat([df, none.assign(model=name, variant="sk")], ignore_index=True)
    meta = pd.concat([common.load(c).assign(condition=c)
                      for c in ("real", "empty", "flipped", "exaggerated", "a1_selective",
                                "a2_forward", "a3_magnitude")])
    keys = ["domain", "split", "sample_id", "condition"]
    df = df.drop(columns=["attacked"]).merge(
        meta[keys + ["has_text", "attacked", "claimed_direction"]], on=keys, how="left",
        validate="many_to_one")
    df["y_last"] = [json.loads(x)[-1] for x in df.lookback]
    df["seed"] = 0
    df["y_true"] = df.y_true.astype(float)
    df["train_mean"] = df.train_mean.astype(float)
    df["train_std"] = df.train_std.astype(float)
    cols = ["model", "domain", "split", "condition", "seed", "sample_id", "y_true", "y_pred", "y_last",
            "train_mean", "train_std", "has_text", "attacked", "claimed_direction", "format_ok",
            "raw_output"]
    return df[cols]


def load_judge(out_dir: Path = OUT_DIR, inputs: Path = R3_INPUTS) -> pd.DataFrame:
    """The R3 judge answers through P6's own loader (fingerprints, completeness, one setup)."""
    from src.textguard.text_direction import load_judgements
    meta = out_dir / f"{JUDGE_FILE}_run_meta.json"
    m = json.loads(meta.read_text(encoding="utf-8"))
    if JUDGE_REVISION not in m.get("setup", "") or m.get("input_sha256", sha256(inputs)) != sha256(inputs):
        raise ValueError("R3 judge: not the pinned revision or not today's input file")
    return load_judgements(out_dir / f"{JUDGE_FILE}_predictions.csv", inputs, meta)


def load_all() -> pd.DataFrame:
    """M5, M5-sk, Q7, Q7-sk in the runner's long schema (val + test)."""
    return pd.concat([long_rows(load_forecaster(r)) for r in PINS], ignore_index=True)
