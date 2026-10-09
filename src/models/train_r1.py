"""Train R1's four late-fusion heads (PROJECT_STATUS §10d): M3-Q, M4-Q, M3-F, M4-F.

Why a separate entry point and not ``train_baselines``: the heads must be trained **exactly**
like Models 3/4 — so this calls the same ``protocol`` / ``run_unit`` / unit machinery (P2.2:
head lr ∈ ``LR_GRID`` chosen on val loss with seed 2021, up to 200 epochs, patience 20, then
seeds 2021–2023; frozen Model 1/2 backbone of the same seed; train-only embedding scaler) —
but into **its own unit folder and tables**, so P2's already-reported files
(``summary_p2_*.csv``, ``p2_baselines_predictions.csv``) are never touched.

Each unit (domain × model) saves val + test predictions with the real text and with
``text=None`` (``condition="empty"``). ``check_text_none`` then asserts the critical check of
§7a for all four heads: ``text=None`` reproduces Model 1/2 of the same seed exactly (the
backbone is frozen, so B − A is the text and nothing else).

Crash-safe: a finished unit is skipped on re-run (fingerprinted protocol; ``--fresh`` wipes);
encoders are unloaded and memory collected between units. Qwen embeddings come only from the
cache (``CachedEncoder`` raises on a miss — Qwen is never loaded here).

Outputs: ``results/r1_train_units/`` (gitignored), ``results/r1_train_predictions.csv``
(gitignored), ``results/summary_r1_lr_tuning.csv``, ``results/summary_r1_train.csv`` (MSE
real / empty per domain × model × split, mean ± std over seeds), checkpoints under
``checkpoints/<model>/<domain>/main_seed<s>.pt``.

Run: ``python -m src.models.train_r1`` (``--no-mlflow``, ``--fresh``, ``--models``, ``--domains``).
"""

from __future__ import annotations

import argparse
import gc
import shutil

import numpy as np
import pandas as pd

from src.data.splits import load_splits
from src.data.windows import lookback_for, split_windows
from src.models.late_fusion import unload_encoders
from src.models.train_baselines import (DOMAINS, R1_CLASSES, RESULTS, SEEDS, protocol,
                                        run_unit, save_unit, summarise, unit_done)

R1_MODELS = tuple(R1_CLASSES)
UNIT_DIR = RESULTS / "r1_train_units"
PREDICTIONS = RESULTS / "r1_train_predictions.csv"
P2_PREDICTIONS = RESULTS / "p2_baselines_predictions.csv"


def run(domains=DOMAINS, models=R1_MODELS, seeds=SEEDS, use_mlflow=True, fresh=False):
    if fresh and UNIT_DIR.exists():
        shutil.rmtree(UNIT_DIR)
    if use_mlflow:
        import mlflow
        mlflow.set_experiment("r1_heads")
    for domain in domains:
        splits = load_splits(domain)
        train_values = splits.train["value"].to_numpy(float)
        seq_len = lookback_for(domain)
        w = split_windows(splits, seq_len, 1)
        print(f"{domain:<10} seq_len={seq_len} windows train={len(w['train'])} "
              f"val={len(w['val'])} test={len(w['test'])}", flush=True)
        for name in models:
            proto = protocol(name, "main", seq_len, 1, seeds)
            if unit_done("main", domain, name, proto, UNIT_DIR):
                print(f"    {name:<16} already done, skipped", flush=True)
                continue
            unload_encoders()
            preds, tuning = run_unit("main", domain, name, w, train_values, seq_len, 1, seeds,
                                     use_mlflow)
            save_unit("main", domain, name, proto, preds, tuning, UNIT_DIR)
            del preds, tuning
            unload_encoders()
            gc.collect()
    return collect()


def collect() -> tuple[pd.DataFrame, pd.DataFrame]:
    stems = sorted(d.name.removesuffix(".done.json") for d in UNIT_DIR.glob("*.done.json"))
    if not stems:
        raise RuntimeError(f"no finished units in {UNIT_DIR}")
    preds = pd.concat([pd.read_csv(UNIT_DIR / f"{s}.csv", dtype={"sample_id": str})
                       for s in stems], ignore_index=True)
    tuning = pd.concat([pd.read_csv(UNIT_DIR / f"{s}_tuning.csv") for s in stems],
                       ignore_index=True)
    rank = {d: i for i, d in enumerate(DOMAINS)}
    tuning = (tuning.assign(_r=tuning.domain.map(rank)).sort_values(["_r", "model", "lr"])
              .drop(columns="_r").reset_index(drop=True))
    preds.to_csv(PREDICTIONS, index=False)
    tuning.to_csv(RESULTS / "summary_r1_lr_tuning.csv", index=False)
    print(f"collected {len(stems)} units, {len(preds)} rows", flush=True)
    return preds, tuning


def check_text_none(preds: pd.DataFrame, p2_path=P2_PREDICTIONS) -> pd.DataFrame:
    """``text=None`` (condition ``empty``) must equal Model 1/2 of the same seed — every row.

    Same tolerance as P2.3 (DLinear bit-identical; PatchTST to float32 precision)."""
    p2 = pd.read_csv(p2_path, dtype={"sample_id": str})
    p2 = p2[(p2.config == "main") & p2.model.isin(["DLinear", "PatchTST"])]
    e = preds[preds.condition == "empty"].assign(backbone=lambda d: d.model.str.split("+").str[0])
    keys = ["domain", "seed", "split", "sample_id", "step"]
    m = e.merge(p2[keys + ["model", "y_pred"]].rename(columns={"model": "backbone",
                                                               "y_pred": "y_backbone"}),
                on=keys + ["backbone"], how="left")
    if m.y_backbone.isna().any():
        raise AssertionError("some text=None predictions have no Model 1/2 counterpart")
    m["abs_diff"] = (m.y_pred - m.y_backbone).abs()
    m["rel_diff"] = m.abs_diff / m.y_backbone.abs().clip(lower=1e-12)
    out = m.groupby(["model", "domain"]).agg(rows=("abs_diff", "size"),
                                             max_abs=("abs_diff", "max"),
                                             max_rel=("rel_diff", "max"),
                                             identical=("abs_diff", lambda d: bool((d == 0).all()))
                                             ).reset_index()
    if (out.max_rel > 1e-5).any():
        raise AssertionError(f"text=None does not reproduce Model 1/2:\n{out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--models", nargs="+", default=list(R1_MODELS))
    ap.add_argument("--domains", nargs="+", default=DOMAINS)
    a = ap.parse_args()
    preds, tuning = run(a.domains, a.models, use_mlflow=not a.no_mlflow, fresh=a.fresh)
    summary = summarise(preds.assign(config="main")).drop(columns="published_uni_mse")
    summary.to_csv(RESULTS / "summary_r1_train.csv", index=False)
    chk = check_text_none(preds)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(tuning[tuning.chosen].round(4).to_string(index=False))
        print(summary.round(4).to_string(index=False))
        print("text=None vs Model 1/2:")
        print(chk.to_string(index=False))


if __name__ == "__main__":
    main()
