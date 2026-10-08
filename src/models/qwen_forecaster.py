"""Model 5 — Qwen2.5-1.5B-Instruct, early fusion — served from precomputed predictions.

Why precomputed: Qwen needs a GPU, which this laptop does not have. It runs once on a Colab
T4 (``notebooks/p2_qwen_forecaster.ipynb``) over every val + test window exported by
``export_qwen_inputs.py``; this class then answers ``predict(numbers, text)`` from that CSV.
P4/P5/P6 therefore treat Model 5 exactly like Models 1–4 — same interface, same windows.

Lookup key: ``(domain, lookback window, text)``. A request that was not precomputed raises
``KeyError`` — it is a pipeline bug (e.g. P3 attacked texts not yet run on Colab), and must
not be confused with a parse failure. A parse failure is ``NaN``, as the contract requires
(PROJECT_STATUS §6a), and is counted by ``parse_failure_rate``.

Integrity check at load: every row's ``prompt_sha`` is recomputed with the *current*
``qwen_prompt.build_prompt``. If the prompt code changed after the Colab run, loading fails
instead of silently serving predictions from a different prompt.

**Scale check (decided 2026-10-09, PROJECT_STATUS §7b).** An answer more than
``SCALE_FACTOR`` (10×) outside the absolute range of its own lookback window is a *format*
failure, like a parse failure: ``y_pred`` becomes NaN (the raw value is kept in
``y_pred_raw``) and ``scale_ok`` is False. Why: on Traffic, Qwen sometimes answered in the
report's unit ("267.0", meaning 267.0 billion) although the prompt asks for million — a unit
slip, not a forecast. Left in, one such row is ~90% of a domain's squared error and would
dominate DSS / Flip Rate, which must measure *direction*. The rule uses only the lookback,
never y_true; it was chosen after seeing these 3 rows, which the paper states.

Determinism: greedy decoding, batch size 1 → one run stands in for 3 seeds (``seed = 0``
in results files).

**P5a — the altered conditions (issue #29).** A second Colab run
(``notebooks/p5_qwen_conditions.ipynb``) forecasts shuffled, cross_domain, time_shifted,
flipped and exaggerated from ``qwen_inputs_p5.csv``, whose rows carry their own condition
``text``. Rows whose prompt is identical to a P2.4 prompt (``reuse`` = real / none) were not
re-generated; ``load_all_predictions`` fills them from P2.4 and re-verifies every prompt
fingerprint, so a reused answer is provably the answer to the same prompt.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster, normalise_text
from src.models.export_qwen_inputs import INPUTS, INPUTS_P5, OUT_DIR
from src.models.qwen_prompt import build_prompt, prompt_sha

PREDICTIONS = OUT_DIR / "p2_qwen_predictions.csv"
PREDICTIONS_P5 = OUT_DIR / "p5_qwen_predictions.csv"
CONDITION_TEXT = {"real": True, "none": False}  # does the condition use the report text?
SCALE_FACTOR = 10.0  # answer > 10x outside the window's |range| -> format failure


def scale_ok(y_pred: float, lookback) -> bool:
    """False if ``y_pred`` is more than ``SCALE_FACTOR`` outside the window's absolute range."""
    if not np.isfinite(y_pred):
        return True  # NaN is already a parse failure; not double-counted
    a = np.abs(np.asarray(lookback, float))
    return a.min() / SCALE_FACTOR <= abs(y_pred) <= a.max() * SCALE_FACTOR


def load_predictions(predictions: Path | str = PREDICTIONS,
                     inputs: Path | str = INPUTS) -> pd.DataFrame:
    """Predictions joined with their inputs, prompt fingerprints verified."""
    preds = pd.read_csv(predictions, keep_default_na=False, dtype={"sample_id": str})
    preds["y_pred"] = pd.to_numeric(preds["y_pred"], errors="coerce")
    preds["parse_ok"] = preds["parse_ok"].astype(str) == "True"
    return _verify(preds, pd.read_csv(inputs, keep_default_na=False, dtype={"sample_id": str}))


def _verify(preds: pd.DataFrame, inp: pd.DataFrame) -> pd.DataFrame:
    """Join predictions to their inputs, recompute every prompt fingerprint, apply the scale
    check. Two input layouts: P2.4 (one row per window; the condition decides real text vs
    none via ``CONDITION_TEXT``) and P5a (one row per window × condition, with its own text)."""
    keys = ["domain", "split", "sample_id"]
    if preds.duplicated(keys + ["condition"]).any():
        raise ValueError("duplicate (domain, split, sample_id, condition) rows in predictions")
    per_condition = "condition" in inp.columns
    if per_condition:
        inp = inp[~inp["probe"].astype(str).eq("True")] if "probe" in inp else inp
        on = keys + ["condition"]
    else:
        unknown = set(preds.condition) - set(CONDITION_TEXT)
        if unknown:
            raise ValueError(f"unknown conditions {unknown}; P3 attack conditions need their "
                             "text (a per-condition inputs file)")
        on = keys
    drop = [c for c in ("reuse", "probe", "expected_raw") if c in inp]
    df = preds.merge(inp.drop(columns=drop), on=on, how="left", suffixes=("", "_input"),
                     indicator=True)
    if (df["_merge"] != "both").any():
        raise ValueError("predictions contain samples that are not in the inputs file")
    if not np.allclose(df["y_true"], df["y_true_input"]):
        raise ValueError("y_true differs between predictions and inputs")
    if (df.parse_ok & df.y_pred.isna()).any() or (~df.parse_ok & df.y_pred.notna()).any():
        raise ValueError("parse_ok disagrees with y_pred (NaN must mean parse failure)")
    df["prompt_text"] = (list(df.text) if per_condition else
                         [t if CONDITION_TEXT[c] else "" for t, c in zip(df.text, df.condition)])
    sha = [prompt_sha(build_prompt(r.description, r.unit, r.freq, json.loads(r.lookback),
                                   r.prompt_text or None)) for r in df.itertuples()]
    bad = df["prompt_sha"].to_numpy() != np.array(sha)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} predictions were made with a different prompt than "
                         "qwen_prompt.build_prompt now builds; re-run the Colab notebook")
    df["y_pred_raw"] = df["y_pred"]
    df["scale_ok"] = [scale_ok(y, json.loads(x)) for y, x in zip(df.y_pred, df.lookback)]
    df.loc[~df.scale_ok, "y_pred"] = np.nan
    df["format_ok"] = df.parse_ok & df.scale_ok
    return df.drop(columns=["_merge", "y_true_input"])


def load_all_predictions(p2_predictions: Path | str = PREDICTIONS, p2_inputs: Path | str = INPUTS,
                         p5_predictions: Path | str = PREDICTIONS_P5,
                         p5_inputs: Path | str = INPUTS_P5) -> pd.DataFrame:
    """P2.4 (real, none) + P5a (5 altered conditions), one verified table.

    P5a rows marked ``reuse`` in the inputs and absent from the Colab output take P2.4's
    answer for the identical prompt (``reused_from`` = real / none). If the Colab run
    generated them anyway (its reproducibility probe failed), the generated answer is used.
    Every row — reused or not — has its prompt fingerprint recomputed from its own text.
    """
    p2 = load_predictions(p2_predictions, p2_inputs).assign(reused_from="")
    if not Path(p5_predictions).exists():
        return p2
    gen = pd.read_csv(p5_predictions, keep_default_na=False, dtype={"sample_id": str})
    inp = pd.read_csv(p5_inputs, keep_default_na=False, dtype={"sample_id": str})
    inp = inp[~inp["probe"].astype(str).eq("True")]
    keys = ["domain", "split", "sample_id"]
    have = set(zip(gen.domain, gen.split, gen.sample_id, gen.condition))
    todo = inp[[k not in have for k in zip(inp.domain, inp.split, inp.sample_id, inp.condition)]]
    if (todo.reuse == "").any():
        raise ValueError(f"{int((todo.reuse == '').sum())} P5 rows were neither generated nor "
                         "reusable — the Colab run is incomplete (re-run its main loop)")
    raw = pd.read_csv(p2_predictions, keep_default_na=False, dtype={"sample_id": str})
    cols = ["y_pred", "parse_ok", "raw_output", "prompt_sha"]
    reused = todo[keys + ["condition", "reuse", "y_true"]].merge(
        raw[keys + ["condition"] + cols].rename(columns={"condition": "reuse"}),
        on=keys + ["reuse"], how="left", validate="many_to_one")
    if reused.prompt_sha.isna().any() or (reused.prompt_sha == "").any():
        raise ValueError("a reused row has no P2.4 prediction")
    reused = reused.rename(columns={"reuse": "reused_from"})
    gen = gen.assign(reused_from="")
    p5_raw = pd.concat([gen, reused[gen.columns]], ignore_index=True)
    p5_raw["y_pred"] = pd.to_numeric(p5_raw["y_pred"], errors="coerce")
    p5_raw["parse_ok"] = p5_raw["parse_ok"].astype(str) == "True"
    if len(p5_raw) != len(inp):
        raise ValueError(f"P5 table has {len(p5_raw)} rows for {len(inp)} inputs")
    p5 = _verify(p5_raw, inp)
    return pd.concat([p2, p5], ignore_index=True)


def parse_failure_rate(df: pd.DataFrame, by=("domain", "split", "condition")) -> pd.DataFrame:
    """Failure counts and rate per group — reported next to every Model 5 metric.

    ``parse_failures`` (no number) and ``scale_failures`` (number on the wrong scale) are
    shown separately; ``rate`` is their union, i.e. the share of NaN predictions.
    """
    g = df.assign(parse_failures=~df.parse_ok, scale_failures=~df.scale_ok,
                  failed=~df.format_ok).groupby(list(by))
    out = g[["parse_failures", "scale_failures"]].sum()
    out["n"] = g.size()
    out["rate"] = g.failed.mean()
    return out


class QwenForecaster(BaseForecaster):
    """Model 5 for one domain, ``predict(numbers, text)`` served from the Colab predictions."""

    name = "Qwen2.5-1.5B"
    uses_text = True

    def __init__(self, domain: str, predictions: Path | str = PREDICTIONS,
                 inputs: Path | str = INPUTS, data: pd.DataFrame | None = None):
        if data is None:  # default files: P2.4 plus the P5a conditions once they exist
            default = Path(predictions) == PREDICTIONS and Path(inputs) == INPUTS
            data = load_all_predictions() if default else load_predictions(predictions, inputs)
        df = data
        df = df[df.domain == domain]
        if df.empty:
            raise ValueError(f"no Qwen predictions for domain {domain!r}")
        lookbacks = [json.loads(s) for s in df.lookback]
        super().__init__(horizon=1, lookback=len(lookbacks[0]))
        self.domain = domain
        self._table: dict[tuple, float] = {}
        for x, text, y in zip(lookbacks, df.prompt_text, df.y_pred):
            key = self._key(np.asarray(x, float), normalise_text(text or None))
            if key in self._table and not _same(self._table[key], y):
                raise ValueError(f"conflicting predictions for one prompt in {domain}")
            self._table[key] = y

    @staticmethod
    def _key(numbers: np.ndarray, text: str | None) -> tuple:
        return tuple(np.asarray(numbers, float).round(9).tolist()), text

    def _predict(self, numbers: np.ndarray, text: str | None) -> np.ndarray:
        try:
            return np.array([self._table[self._key(numbers, text)]])
        except KeyError:
            raise KeyError(f"Qwen has no precomputed prediction for this ({self.domain}) window "
                           "and text; export it and run the Colab notebook") from None

    def predict_windows(self, X: np.ndarray, texts=None) -> np.ndarray:
        """Batched lookup, shape (N, 1) — same interface as the neural wrappers."""
        X = np.asarray(X, float)
        texts = [None] * len(X) if texts is None else list(texts)
        return np.stack([self.predict(x, t) for x, t in zip(X, texts)])


def _same(a: float, b: float) -> bool:
    return (np.isnan(a) and np.isnan(b)) or a == b
