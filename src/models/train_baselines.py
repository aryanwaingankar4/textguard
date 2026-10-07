"""Train and evaluate the numbers-only baselines (P2.2): Persistence, DLinear, PatchTST.

Why this is the cleanest reproduction check: numbers-only models never see text, so the
text source (``report`` vs MM-TSFlib's ``search``, issue #6) cannot explain a mismatch with
the published numbers. Any gap is our pipeline — split or scaling — or a known, documented
difference (see PROJECT_STATUS §7).

Two configurations per domain:

* ``main`` — horizon 1, PatchTST with ``patching="window"``, **tuned training protocol**
  (below). This is what P5 uses.
* ``repro`` — the shortest horizon in the Time-MMD paper's tables (monthly 6, weekly 12),
  PatchTST exactly as shipped, **MM-TSFlib's original recipe** (10 epochs, patience 5,
  lr 1e-4). Only for comparing with published MSE: horizon-1 MSE is not comparable with a
  6-step average.

Tuned protocol (``main``) — why it exists: under MM-TSFlib's recipe (~70 Adam steps at
lr 1e-4) DLinear barely leaves its initialisation (weights 1/L, i.e. a window average) and
scores the same as a "forecast = window mean" predictor. A baseline that is not trained is
not a baseline. So, per domain and per model:

1. for each lr in ``LR_GRID`` (coarse on purpose — val sets are only 27–68 periods, a fine
   grid would overfit them), train with seed ``TUNE_SEED`` for up to 200 epochs, early
   stopping with patience 20 on val loss;
2. keep the lr with the lowest best-epoch val loss;
3. train all 3 seeds with that lr (the ``TUNE_SEED`` run is reused — training is
   deterministic, so retraining would give the identical model).

The test split is never looked at while choosing. **Models 3 and 4 (P2.3) must reuse this
exact protocol**, so a Text Usage Gap reflects the text, not extra trainable parameters on
an under-trained backbone.

MSE is reported on the **normalised** scale (z-scored with the train split's mean/std), as
in the Time-MMD tables. Equivalently: raw MSE / train variance.

Outputs:

* ``results/p2_baselines/<config>_<domain>_<model>.csv`` (+ ``_tuning.csv``, ``.done.json``)
  — one set per finished unit (config × domain × model), written the moment the unit ends.
  **Resumable:** a re-run skips finished units, so a crash loses only the unit in progress.
  A unit is reused only if its saved protocol matches the current one; ``--fresh`` wipes
  them all. Gitignored.
* ``results/p2_baselines_predictions.csv`` — every (config, model, domain, seed, split,
  sample, step) prediction in raw units, collected from the units. Gitignored; metrics are
  derivable from it.
* ``results/summary_p2_baselines.csv`` — aggregated MSE (kept in git).
* ``results/summary_p2_lr_tuning.csv`` — val loss for every grid point (kept in git).
* ``checkpoints/<model>/<domain>/<config>_seed<s>.pt`` — gitignored.
* one MLflow run per (config, model, domain, seed), plus one per tuning grid point. A unit
  that crashed may leave orphan MLflow runs; the CSVs are the source of truth.

Run: ``python -m src.models.train_baselines`` (``--no-mlflow`` to skip tracking,
``--fresh`` to retrain everything).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.splits import load_splits
from src.data.windows import DOMAIN_FREQ, PUBLISHED_HORIZON, lookback_for, split_windows
from src.models.dlinear import DLinearForecaster
from src.models.patchtst import PatchTSTForecaster
from src.models.persistence import PersistenceForecaster

DOMAINS = ["Economy", "SocialGood", "Traffic", "Health_US"]
SEEDS = [2021, 2022, 2023]  # MM-TSFlib's script uses 2021
TUNE_SEED = 2021

# Tuned protocol (main config) — reuse for Models 3/4 in P2.3
LR_GRID = (1e-4, 1e-3, 1e-2)
TUNED = dict(epochs=200, patience=20)
# MM-TSFlib's original recipe (repro config): run.py defaults, LR decay commented out
MMTSFLIB_RECIPE = dict(epochs=10, patience=5, lr=1e-4)

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"
# one predictions CSV (+ tuning CSV + done marker) per finished unit; gitignored
UNIT_DIR = RESULTS / "p2_baselines"

# Time-MMD paper (Liu et al., NeurIPS 2024), App. O, Tables 11/14/16/17: unimodal ("Uni")
# MSE at the shortest horizon — monthly 6, weekly 12. Normalised scale.
PUBLISHED_UNI_MSE = {
    ("Economy", "DLinear"): 0.08, ("Economy", "PatchTST"): 0.02,
    ("SocialGood", "DLinear"): 1.01, ("SocialGood", "PatchTST"): 0.88,
    ("Traffic", "DLinear"): 0.35, ("Traffic", "PatchTST"): 0.21,
    ("Health_US", "DLinear"): 2.37, ("Health_US", "PatchTST"): 1.23,
}

MODEL_CLASSES = {"DLinear": DLinearForecaster, "PatchTST": PatchTSTForecaster}


def make_model(name: str, config: str, seq_len: int, horizon: int, seed: int,
               lr: float | None = None):
    kw = dict(TUNED, lr=lr) if config == "main" else dict(MMTSFLIB_RECIPE)
    if name == "PatchTST":
        kw["patching"] = "window" if config == "main" else "mmtsflib"
    return MODEL_CLASSES[name](seq_len, horizon, seed=seed, **kw)


def best_val(model) -> float:
    return min(h["val_loss"] for h in model.history)


def tune_lr(name, seq_len, w, train_values, log=None):
    """Grid-search lr on val (seed TUNE_SEED). Returns (best_lr, fitted model, grid rows)."""
    rows, fitted = [], {}
    for lr in LR_GRID:
        m = make_model(name, "main", seq_len, 1, TUNE_SEED, lr).fit(w["train"], w["val"],
                                                                     train_values)
        fitted[lr] = m
        rows.append({"model": name, "lr": lr, "best_val_loss": best_val(m),
                     "epochs_run": len(m.history),
                     "best_epoch": int(np.argmin([h["val_loss"] for h in m.history])) + 1})
        if log:
            log(rows[-1])
    best_lr = min(rows, key=lambda r: r["best_val_loss"])["lr"]
    return best_lr, fitted[best_lr], rows


def _rows(config, model_name, domain, seed, split, w, pred, mean, std, lr=None):
    n, h = w.Y.shape
    return pd.DataFrame({
        "config": config, "model": model_name, "domain": domain, "seed": seed, "lr": lr,
        "horizon": h, "split": split,
        "sample_id": np.repeat(w.target_start.strftime("%Y-%m-%d"), h),
        "step": np.tile(np.arange(1, h + 1), n),
        "y_true": w.Y.ravel(), "y_pred": pred.ravel(),
        "train_mean": mean, "train_std": std,
    })


# --- resumable units -------------------------------------------------------------------
#
# A *unit* is one (config, domain, model): for a neural model in ``main`` that is the lr
# grid plus 3 seeds. Each finished unit is written to ``UNIT_DIR`` at once and skipped on
# the next run, so a crash loses only the unit in progress.

def protocol(name: str, config: str, seq_len: int, horizon: int, seeds=SEEDS) -> dict:
    """Every setting that determines a unit's result. Stored with the unit, so a saved unit
    is only reused under the protocol that produced it, never silently mixed with a new one.
    (It does not fingerprint the code itself: after changing model code, run ``--fresh``.)"""
    p = dict(model=name, config=config, seq_len=seq_len, horizon=horizon)
    if name == "Persistence":
        return p
    p["seeds"] = list(seeds)
    if config == "main":
        p.update(lr_grid=list(LR_GRID), tune_seed=TUNE_SEED, **TUNED)
    else:
        p.update(**MMTSFLIB_RECIPE)
    if name == "PatchTST":
        p["patching"] = "window" if config == "main" else "mmtsflib"
    return p


def unit_paths(config: str, domain: str, name: str) -> dict[str, Path]:
    stem = f"{config}_{domain}_{name}"
    return {"preds": UNIT_DIR / f"{stem}.csv",
            "tuning": UNIT_DIR / f"{stem}_tuning.csv",
            "done": UNIT_DIR / f"{stem}.done.json"}


def unit_done(config: str, domain: str, name: str, proto: dict) -> bool:
    """True if the unit finished under ``proto``. Raises if it finished under another one."""
    paths = unit_paths(config, domain, name)
    if not (paths["done"].exists() and paths["preds"].exists()):
        return False
    saved = json.loads(paths["done"].read_text())
    if saved != proto:
        raise RuntimeError(
            f"{paths['done'].name} was produced under a different protocol:\n"
            f"  saved   {saved}\n  current {proto}\n"
            "Re-run with --fresh to discard saved units.")
    return True


def _atomic_write(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def save_unit(config: str, domain: str, name: str, proto: dict,
              preds: pd.DataFrame, tuning: pd.DataFrame | None) -> None:
    """Write the unit's files. The ``.done.json`` marker goes last, so a crash mid-save
    leaves an unfinished unit (re-run next time), never a half-written one that looks done."""
    paths = unit_paths(config, domain, name)
    UNIT_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write(paths["preds"], lambda f: preds.to_csv(f, index=False))
    if tuning is not None:
        _atomic_write(paths["tuning"], lambda f: tuning.to_csv(f, index=False))
    _atomic_write(paths["done"], lambda f: f.write_text(json.dumps(proto, indent=1)))


def run_unit(config, domain, name, w, train_values, seq_len, horizon, seeds, use_mlflow):
    """Train and evaluate one (config, domain, model). Returns (predictions, tuning | None)."""
    mean, std = float(train_values.mean()), float(train_values.std())
    if name == "Persistence":  # deterministic: one run, seed recorded as 0
        pers = PersistenceForecaster(horizon=horizon, lookback=seq_len)
        return pd.concat([_rows(config, name, domain, 0, split, w[split],
                                pers.predict_windows(w[split].X), mean, std)
                          for split in ("val", "test")], ignore_index=True), None
    if use_mlflow:
        import mlflow
    frames, tuning, lr, reuse = [], None, None, None
    if config == "main":
        def log(r):
            print(f"    tune {name:<9} lr={r['lr']:g} best_val={r['best_val_loss']:.4f} "
                  f"best_epoch={r['best_epoch']} epochs_run={r['epochs_run']}", flush=True)
            if use_mlflow:
                with mlflow.start_run(run_name=f"tune-{name}-{domain}-lr{r['lr']:g}"):
                    mlflow.log_params({"model": name, "domain": domain,
                                       "condition": "numbers_only", "seed": TUNE_SEED,
                                       "config": "tune", "lr": r["lr"],
                                       "horizon": 1, "seq_len": seq_len})
                    mlflow.log_metrics({"best_val_loss": r["best_val_loss"],
                                        "epochs_run": r["epochs_run"]})
        lr, reuse, grid = tune_lr(name, seq_len, w, train_values, log)
        tuning = pd.DataFrame([{"domain": domain, **r, "chosen": r["lr"] == lr} for r in grid])
    for seed in seeds:
        ckpt = CHECKPOINTS / name / domain / f"{config}_seed{seed}.pt"
        if reuse is not None and seed == TUNE_SEED:
            model = reuse
            model.save(ckpt)
        else:
            model = make_model(name, config, seq_len, horizon, seed, lr)
            model.fit(w["train"], w["val"], train_values, checkpoint=ckpt)
        metrics = {}
        for split in ("val", "test"):
            pred = model.predict_windows(w[split].X)
            frames.append(_rows(config, name, domain, seed, split, w[split],
                                pred, mean, std, model.lr))
            metrics[f"{split}_mse_norm"] = float(np.mean(((w[split].Y - pred) / std) ** 2))
        if use_mlflow:
            with mlflow.start_run(run_name=f"{config}-{name}-{domain}-{seed}"):
                mlflow.log_params({"model": name, "domain": domain,
                                   "condition": "numbers_only", "seed": seed,
                                   "config": config, "horizon": horizon,
                                   "seq_len": seq_len, "lr": model.lr,
                                   "max_epochs": model.epochs,
                                   "patience": model.patience,
                                   "patching": getattr(model, "patching", "-"),
                                   "n_train_windows": len(w["train"])})
                mlflow.log_metrics({**metrics, "epochs_run": len(model.history)})
        print(f"    {name:<9} seed={seed} lr={model.lr:g} epochs={len(model.history):>3} "
              f"val={metrics['val_mse_norm']:.4f} test={metrics['test_mse_norm']:.4f}",
              flush=True)
    return pd.concat(frames, ignore_index=True), tuning


def run(domains=DOMAINS, seeds=SEEDS, configs=("main", "repro"), use_mlflow=True,
        fresh=False):
    """Run every unit not already saved, then collect all saved units.

    Returns (raw-predictions frame, lr-tuning frame) over every finished unit.
    ``fresh=True`` deletes saved units first (checkpoints are simply overwritten).
    """
    if fresh and UNIT_DIR.exists():
        shutil.rmtree(UNIT_DIR)
    if use_mlflow:
        import mlflow
        mlflow.set_experiment("p2_baselines")
    for domain in domains:
        splits = load_splits(domain)
        train_values = splits.train["value"].to_numpy(float)
        seq_len = lookback_for(domain)
        for config in configs:
            horizon = 1 if config == "main" else PUBLISHED_HORIZON[DOMAIN_FREQ[domain]]
            w = split_windows(splits, seq_len, horizon)
            print(f"{domain:<10} {config:<5} seq_len={seq_len} h={horizon} windows "
                  f"train={len(w['train'])} val={len(w['val'])} test={len(w['test'])}",
                  flush=True)
            for name in ("Persistence", *MODEL_CLASSES):
                proto = protocol(name, config, seq_len, horizon, seeds)
                if unit_done(config, domain, name, proto):
                    print(f"    {name:<11} already done, skipped", flush=True)
                    continue
                preds, tuning = run_unit(config, domain, name, w, train_values, seq_len,
                                         horizon, seeds, use_mlflow)
                save_unit(config, domain, name, proto, preds, tuning)
    return collect()


def collect() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate every finished unit into the combined predictions and tuning CSVs."""
    stems = sorted(d.name.removesuffix(".done.json") for d in UNIT_DIR.glob("*.done.json"))
    if not stems:
        raise RuntimeError(f"no finished units in {UNIT_DIR}")
    preds = pd.concat([pd.read_csv(UNIT_DIR / f"{s}.csv") for s in stems], ignore_index=True)
    tuning_files = [UNIT_DIR / f"{s}_tuning.csv" for s in stems
                    if (UNIT_DIR / f"{s}_tuning.csv").exists()]
    tuning = pd.DataFrame()
    if tuning_files:  # stable order: domains as in DOMAINS, then model, then lr
        tuning = pd.concat([pd.read_csv(f) for f in tuning_files], ignore_index=True)
        rank = {d: i for i, d in enumerate(DOMAINS)}
        tuning = (tuning.assign(_r=tuning.domain.map(rank))
                  .sort_values(["_r", "model", "lr"]).drop(columns="_r")
                  .reset_index(drop=True))
    RESULTS.mkdir(exist_ok=True)
    preds.to_csv(RESULTS / "p2_baselines_predictions.csv", index=False)
    tuning.to_csv(RESULTS / "summary_p2_lr_tuning.csv", index=False)
    print(f"collected {len(stems)} finished units", flush=True)
    return preds, tuning


def summarise(preds: pd.DataFrame) -> pd.DataFrame:
    """Normalised MSE per (config, domain, model, split): mean ± std over seeds."""
    p = preds.assign(se=((preds.y_true - preds.y_pred) / preds.train_std) ** 2)
    per_seed = p.groupby(["config", "domain", "model", "split", "seed"]).se.mean()
    out = per_seed.groupby(["config", "domain", "model", "split"]).agg(["mean", "std"]).reset_index()
    out["std"] = out["std"].fillna(0.0)
    out["published_uni_mse"] = [PUBLISHED_UNI_MSE.get((d, m)) if c == "repro" and s == "test"
                                else None for c, d, m, s in
                                zip(out.config, out.domain, out.model, out.split)]
    return out.rename(columns={"mean": "mse_norm_mean", "std": "mse_norm_std"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--domains", nargs="+", default=DOMAINS)
    ap.add_argument("--fresh", action="store_true",
                    help="discard saved units and retrain everything")
    args = ap.parse_args()
    preds, tuning = run(domains=args.domains, use_mlflow=not args.no_mlflow,
                        fresh=args.fresh)
    summary = summarise(preds)
    summary.to_csv(RESULTS / "summary_p2_baselines.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(tuning.round(4).to_string(index=False))
        print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
