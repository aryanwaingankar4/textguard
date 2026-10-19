"""R4 — load and verify the four held-out Kaggle answer files (PROJECT_STATUS §10g "Runs").

R3's verifier (``r3_answers``) applied to R4's single input file. Why a verifier and not a plain
``read_csv``: every R4 comparison is paired within one run, so it is only meaningful if each file
comes from the pre-registered model / revision / precision, answered **today's**
``results/qwen/r4_heldout_inputs.csv`` (sha256 ``cecbdc9b…``; a re-export reproduces it byte for
byte), from **one** software setup, and answered every exported row exactly once. A file is
refused unless:

* **forecasters (M5, Q7):** the meta's model id / revision / quant are §10g's pins and its
  ``input_sha256`` is today's input file; the meta says test complete; every row has the meta's
  setup string; no duplicate (window, condition); every forecast row answered (variant p5 only —
  §10g runs M5's prompt only); every prompt fingerprint recomputed now from the inputs with the
  P5 prompt builder; y_true = inputs; y_pred = M5's parser applied to the raw output; parse_ok
  agrees; byte-identical prompts answered identically. Then M5's off-scale rule
  (``qwen_forecaster.scale_ok``), exactly as for Model 5.
* **judges (v1, v2):** pinned revision and input sha; P6's / v2's own loaders (every exported
  question answered once, fingerprints re-derived with the frozen prompt modules plus the held-out
  questions, one setup = the meta's); probabilities finite, summing to 1, equal to the softmax of
  the stored log-probs; answer = the most likely of the three; confidence = its probability.

The probe (P5's 40 prompts, compared with P2.4 inside the notebook) is **recorded, never gating**
(§10g) — :func:`verify_all` reports it.

Held-out keys: ``Energy`` and ``FNSPID:<TICKER>``. POLARITY is +1 for both (§10g) — P6's loader
maps POLARITY by domain and knows only the original four, so the v1 direction is set here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.export_qwen_inputs import OUT_DIR
from src.models.export_r4_inputs import R4_INPUTS, sha256
from src.models.qwen_forecaster import scale_ok
from src.models.qwen_prompt import parse_number
from src.models.r2_prompts import build_variant_prompt, variant_prompt_sha

KEYS = ["domain", "split", "sample_id", "condition"]
INPUT_SHA256 = "cecbdc9b14086178adc1ccbaa43a5b0c9d3854952d04265a8a8ebd1301dbf231"
M5_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
PINS = {
    "m5": dict(model_id="Qwen/Qwen2.5-1.5B-Instruct", revision=M5_REVISION, quant="fp16",
               file="r4_m5_fp16_kaggle", name="M5"),
    "qwen7b": dict(model_id="Qwen/Qwen2.5-7B-Instruct", revision="a09a35458c702b33eeacc393d103063234e8bc28",
                   quant="nf4", file="r4_qwen7b_nf4_kaggle", name="Q7"),
}
JUDGES = {"v1": dict(file="r4_judge_v1_kaggle", row_type="judge", rows=2913),
          "v2": dict(file="r4_judge_v2_kaggle", row_type="judge_v2", rows=21312)}


def _meta(file: str, out_dir: Path) -> dict:
    return json.loads((out_dir / f"{file}_run_meta.json").read_text(encoding="utf-8"))


def _check_input(meta: dict, who: str, inputs: Path) -> None:
    got = sha256(inputs)
    if got != INPUT_SHA256:
        raise ValueError(f"{who}: today's {Path(inputs).name} is not the pre-registered export (sha {got[:16]})")
    if meta.get("input_sha256") != got:
        raise ValueError(f"{who}: answered a different input file than {Path(inputs).name}")


def load_forecaster(run: str, out_dir: Path = OUT_DIR, inputs: Path = R4_INPUTS) -> pd.DataFrame:
    """One R4 forecaster run (``m5`` or ``qwen7b``), verified; one row per window × condition."""
    pin = PINS[run]
    meta = _meta(pin["file"], out_dir)
    for k in ("model_id", "revision", "quant"):
        if meta.get(k) != pin[k]:
            raise ValueError(f"{run}: {k} is {meta.get(k)!r}, pre-registered {pin[k]!r}")
    _check_input(meta, run, inputs)
    if not meta.get("test_complete") or meta.get("variants") != ["p5"]:
        raise ValueError(f"{run}: the run's meta says it is incomplete or ran other prompt variants")
    gen = pd.read_csv(out_dir / f"{pin['file']}_predictions.csv", keep_default_na=False,
                      dtype={"sample_id": str, "raw_output": str, "prompt_sha": str},
                      float_precision="round_trip")      # exact floats: the checks below are exact
    if set(gen.setup) != {meta["setup"]}:
        raise ValueError(f"{run}: answers must come from exactly the meta's setup")
    if set(gen.variant) != {"p5"} or gen.duplicated(KEYS).any():
        raise ValueError(f"{run}: other variants or duplicate rows")
    inp = pd.read_csv(inputs, keep_default_na=False, dtype=str)
    inp = inp[inp.row_type == "forecast"]
    got = inp.merge(gen, on=KEYS, how="outer", suffixes=("", "_gen"), indicator=True,
                    validate="one_to_one")
    if (got._merge != "both").any():
        raise ValueError(f"{run}: {int((got._merge != 'both').sum())} rows missing or extra")
    sha = [variant_prompt_sha("p5", build_variant_prompt("p5", r.description, r.unit, r.freq,
                                                         json.loads(r.lookback), r.text or None))
           for r in got.itertuples()]
    if sha != list(got.prompt_sha_gen):
        raise ValueError(f"{run}: prompt fingerprints differ from today's prompt builder")
    if not np.allclose(got.y_true.astype(float), got.y_true_gen.astype(float), rtol=0, atol=0):
        raise ValueError(f"{run}: y_true differs from the inputs")
    parsed = np.array([parse_number(o, u) for o, u in zip(got.raw_output, got.unit)], float)
    y = pd.to_numeric(got.y_pred, errors="coerce").to_numpy(float)
    if not np.allclose(parsed, y, equal_nan=True, rtol=0, atol=0):
        raise ValueError(f"{run}: y_pred is not the parser's reading of raw_output")
    ok = got.parse_ok.astype(str) == "True"
    if (ok != np.isfinite(y)).any():
        raise ValueError(f"{run}: parse_ok disagrees with y_pred")
    if (got.groupby("prompt_sha_gen").raw_output.nunique() > 1).any():
        raise ValueError(f"{run}: byte-identical prompts answered differently")
    df = got.assign(y_pred_raw=y, parse_ok=ok, prompt_sha=got.prompt_sha_gen)
    df["scale_ok"] = [scale_ok(v, json.loads(x)) for v, x in zip(df.y_pred_raw, df.lookback)]
    df["y_pred"] = np.where(df.scale_ok, df.y_pred_raw, np.nan)
    df["format_ok"] = df.parse_ok & df.scale_ok
    df["model"] = pin["name"]
    df["y_true"] = df.y_true.astype(float)
    df["sigma"] = df.train_std.astype(float)
    df["attacked"] = df.attacked.astype(str) == "True"
    return df.drop(columns=["_merge", "y_true_gen", "prompt_sha_gen", "row_type", "question", "sources",
                            "expected_raw", "train_mean", "train_std"])


def _check_probs(df: pd.DataFrame, who: str) -> None:
    p = df[["p_pos", "p_neg", "p_none"]].astype(float).to_numpy()
    lp = df[["logp_pos", "logp_neg", "logp_none"]].astype(float).to_numpy()
    if not (np.isfinite(p).all() and np.isfinite(lp).all()):
        raise ValueError(f"{who}: non-finite probabilities")
    if not np.allclose(p.sum(1), 1, atol=1e-9):
        raise ValueError(f"{who}: probabilities do not sum to 1")
    sm = np.exp(lp - lp.max(1, keepdims=True))
    if not np.allclose(p, sm / sm.sum(1, keepdims=True), atol=1e-6):
        raise ValueError(f"{who}: probabilities are not the softmax of the stored log-probs")
    words = np.array([[a for a in ans] for ans in df.question.map(_answer_words)])
    best = words[np.arange(len(df)), p.argmax(1)]
    if (best != df.answer.to_numpy()).any():
        raise ValueError(f"{who}: answer is not the most likely of the three")
    if not np.allclose(df.confidence.astype(float), p.max(1), rtol=0, atol=0):
        raise ValueError(f"{who}: confidence is not the answer's probability")


def _answer_words(question: str) -> tuple[str, str, str]:
    from src.textguard.judge_prompt_v2 import ANSWERS
    return ANSWERS[question]


def load_judge(which: str, out_dir: Path = OUT_DIR, inputs: Path = R4_INPUTS) -> pd.DataFrame:
    """TextGuard v1's (``v1``) or v2's (``v2``) held-out judge answers, verified."""
    from src.textguard import heldout_questions as hq
    from src.textguard import v2 as V
    from src.textguard.text_direction import load_judgements
    spec = JUDGES[which]
    meta_p = out_dir / f"{spec['file']}_run_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    if meta.get("model_id") != PINS["m5"]["model_id"] or meta.get("revision") != M5_REVISION \
            or M5_REVISION not in meta.get("setup", ""):
        raise ValueError(f"judge {which}: not Qwen2.5-1.5B at the pinned revision")
    _check_input(meta, f"judge {which}", inputs)
    if meta.get("rows") != spec["rows"] or meta.get("rows_expected") != spec["rows"]:
        raise ValueError(f"judge {which}: meta says {meta.get('rows')} of {meta.get('rows_expected')} rows")
    hq.register()                                   # v1 QUESTIONS / v2 SUBJECTS for the held-out keys
    pred = out_dir / f"{spec['file']}_predictions.csv"
    if which == "v1":
        df = load_judgements(pred, inputs, meta_p)
        df["direction"] = df["literal"]             # POLARITY +1 for Energy and FNSPID (§10g)
    else:
        df = V.load_answers(pred, inputs, meta_p, row_type="judge_v2")
    _check_probs(df, f"judge {which}")
    return df


def verify_all(out_dir: Path = OUT_DIR, inputs: Path = R4_INPUTS) -> pd.DataFrame:
    """One row per run: what was checked and what it found (descriptive, for PROJECT_STATUS §7f)."""
    rows = []
    for run, pin in PINS.items():
        m = _meta(pin["file"], out_dir)
        df = load_forecaster(run, out_dir, inputs)
        rows.append(dict(run=pin["file"], model=f"{m['model_id']}@{m['revision'][:7]}", quant=m["quant"],
                         setup=m["setup"], input_sha=m["input_sha256"][:16], rows=len(df),
                         rows_expected=m["rows_expected"], distinct_prompts=df.prompt_sha.nunique(),
                         probe=f"{m['probe_n'] - m['probe_mismatches']}/{m['probe_n']}" if "probe_n" in m else "—",
                         smoke_test=(f"{m['smoke_test']['unparseable']}/{m['smoke_test']['n']} unparseable"
                                     if "smoke_test" in m else "—"),
                         determinism=f"{m['determinism_n'] - m['determinism_mismatches']}/{m['determinism_n']}",
                         parse_failures=int((~df.parse_ok).sum()), scale_failures=int((df.parse_ok & ~df.scale_ok).sum()),
                         started=m["started"], finished=m["finished"]))
    for which, spec in JUDGES.items():
        m = _meta(spec["file"], out_dir)
        df = load_judge(which, out_dir, inputs)
        rows.append(dict(run=spec["file"], model=f"{m['model_id']}@{m['revision'][:7]}", quant=m["dtype"],
                         setup=m["setup"], input_sha=m["input_sha256"][:16], rows=len(df),
                         rows_expected=m["rows_expected"], distinct_prompts=df.prompt_sha.nunique(),
                         probe=f"{m['probe_n'] - m['probe_mismatches']}/{m['probe_n']}", smoke_test="—",
                         determinism="—", parse_failures=0, scale_failures=0,
                         started=m["started"], finished=m["finished"]))
    return pd.DataFrame(rows)


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250)
    print(verify_all().T.to_string())


if __name__ == "__main__":
    main()
