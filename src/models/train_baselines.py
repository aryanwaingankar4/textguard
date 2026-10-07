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

* ``results/p2_baselines_predictions.csv`` — every (config, model, domain, seed, split,
  sample, step) prediction in raw units. Gitignored; metrics are derivable from it.
* ``results/summary_p2_baselines.csv`` — aggregated MSE (kept in git).
* ``results/summary_p2_lr_tuning.csv`` — val loss for every grid point (kept in git).
* ``checkpoints/<model>/<domain>/<config>_seed<s>.pt`` — gitignored.
* one MLflow run per (config, model, domain, seed), plus one per tuning grid point.

Run: ``python -m src.models.train_baselines`` (``--no-mlflow`` to skip tracking).
"""

from __future__ import annotations

import argparse
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


def run(domains=DOMAINS, seeds=SEEDS, configs=("main", "repro"), use_mlflow=True):
    """Train everything; return (raw-predictions frame, lr-tuning frame)."""
    if use_mlflow:
        import mlflow
        mlflow.set_experiment("p2_baselines")
    frames, tuning = [], []
    for domain in domains:
        splits = load_splits(domain)
        train_values = splits.train["value"].to_numpy(float)
        mean, std = float(train_values.mean()), float(train_values.std())
        seq_len = lookback_for(domain)
        for config in configs:
            horizon = 1 if config == "main" else PUBLISHED_HORIZON[DOMAIN_FREQ[domain]]
            w = split_windows(splits, seq_len, horizon)
            print(f"{domain:<10} {config:<5} seq_len={seq_len} h={horizon} windows "
                  f"train={len(w['train'])} val={len(w['val'])} test={len(w['test'])}", flush=True)
            # Persistence is deterministic: one run, seed recorded as 0
            pers = PersistenceForecaster(horizon=horizon, lookback=seq_len)
            for split in ("val", "test"):
                frames.append(_rows(config, "Persistence", domain, 0, split, w[split],
                                    pers.predict_windows(w[split].X), mean, std))
            for name in MODEL_CLASSES:
                lr, reuse = None, None
                if config == "main":
                    def log(r, name=name):
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
                    tuning += [{"domain": domain, **r, "chosen": r["lr"] == lr} for r in grid]
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
    preds = pd.concat(frames, ignore_index=True)
    RESULTS.mkdir(exist_ok=True)
    preds.to_csv(RESULTS / "p2_baselines_predictions.csv", index=False)
    tuning = pd.DataFrame(tuning)
    tuning.to_csv(RESULTS / "summary_p2_lr_tuning.csv", index=False)
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
    args = ap.parse_args()
    preds, tuning = run(domains=args.domains, use_mlflow=not args.no_mlflow)
    summary = summarise(preds)
    summary.to_csv(RESULTS / "summary_p2_baselines.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(tuning.round(4).to_string(index=False))
        print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
