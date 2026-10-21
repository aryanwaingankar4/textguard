"""Experiment runner (P5a): every model's raw predictions on every text condition.

**What it does.** Models 1–4 (DLinear, PatchTST, DLinear+BERT, PatchTST+BERT) × 4 domains ×
7 text conditions (``real, empty, shuffled, cross_domain, time_shifted, flipped,
exaggerated``) × 3 seeds × val + test windows. Model 5 (Qwen) runs on Colab; its
predictions are converted into the same table by ``qwen_rows`` once the CSV is back.

**Inference only — nothing is retrained.** Each seed's P2 checkpoint is loaded and asked to
``predict_windows(X, texts)``. That is the threat model: the attacker controls the text at
inference time, not the training data. It also means the only thing that differs between
two conditions is the text — same weights, same windows.

**Why the text is joined, not regenerated.** Condition texts come from ``texts/<cond>.csv``
(P3, deterministic), joined on ``(domain, split, sample_id)`` to the windows built by the
same ``load_splits → split_windows`` pipeline as training. ``real`` is cross-checked against
the windows' own text, so a misaligned join fails loudly instead of silently pairing a
window with another period's report.

**Embeddings (Models 3/4).** BERT vectors come from the on-disk cache filled by
``python -m src.models.embed_texts --conditions``; they are standardised by the
``EmbeddingScaler`` *inside* each checkpoint — the train-only statistics fitted in P2.3.
Nothing here refits it. If a text is missing from the cache, BERT loads on demand and is
unloaded after the unit (system RAM is tight).

**No metrics here.** P5a only generates; all screening / attack analysis is P5b, against the
pre-registered family (PROJECT_STATUS §10a). The checks below are integrity checks
(text-blindness of Models 1–2, agreement with P2's saved predictions), not results.

**Resumable.** A unit is one (model, domain, condition): 3 seeds × val + test. Each finished
unit is written at once to ``results/p5_units/`` with a ``.done.json`` fingerprint (texts and
checkpoints hashed); a re-run skips finished units and refuses a unit whose fingerprint no
longer matches (``--fresh`` discards them). A crash loses at most one unit.

Outputs:

* ``results/p5_units/<model>_<domain>_<condition>.csv`` (+ ``.done.json``) — gitignored
* ``results/p5_predictions.csv`` — every unit combined, one row per (model, domain, split,
  condition, seed, sample_id): ``y_true, y_pred, y_last, train_mean, train_std`` plus the
  condition's ``has_text, attacked, claimed_direction``. Gitignored (raw); all P5b metrics
  are derived from it.
* one MLflow run per (model, domain, condition, seed), experiment ``p5_experiments``.

Run: ``python -m src.run_experiments`` (``--no-mlflow``, ``--fresh``, ``--models``,
``--domains``, ``--conditions``).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.attacks import common
from src.attacks.common import CONDITIONS
from src.data.splits import load_splits
from src.data.windows import lookback_for, split_windows
from src.models.late_fusion import _ENCODERS, unload_encoders
from src.models.train_baselines import (CHECKPOINTS, DOMAINS, MODEL_CLASSES, SEEDS,
                                        _atomic_write)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
UNIT_DIR = RESULTS / "p5_units"
PREDICTIONS = RESULTS / "p5_predictions.csv"
P2_PREDICTIONS = RESULTS / "p2_baselines_predictions.csv"
MODELS = ("DLinear", "PatchTST", "DLinear+BERT", "PatchTST+BERT")
TEXT_BLIND = ("DLinear", "PatchTST")
SPLITS = ("val", "test")
CONFIG = "main"  # horizon 1, val-tuned protocol (PROJECT_STATUS §7)

COLUMNS = ["model", "domain", "split", "condition", "seed", "sample_id", "y_true", "y_pred",
           "y_last", "train_mean", "train_std", "has_text", "attacked", "claimed_direction"]


# --- inputs ----------------------------------------------------------------------------------

def checkpoint(model: str, domain: str, seed: int) -> Path:
    return CHECKPOINTS / model / domain / f"{CONFIG}_seed{seed}.pt"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def condition_texts(condition: str, domain: str, split: str, sample_ids) -> pd.DataFrame:
    """The condition's rows for these windows, in window order (``text`` None when blank).

    ``empty`` is the absent text by definition (``text=None`` ≡ TextGuard's A).
    """
    df = common.load(condition)
    df = df[(df.domain == domain) & (df.split == split)].set_index("sample_id")
    missing = set(sample_ids) - set(df.index)
    if missing:
        raise ValueError(f"texts/{condition}.csv lacks {len(missing)} {domain}/{split} windows, "
                         f"e.g. {sorted(missing)[:3]} — re-run python -m src.attacks.generate")
    df = df.loc[list(sample_ids)]
    out = df[["has_text", "attacked", "claimed_direction"]].copy()
    # object dtype: pandas 3 would otherwise infer a string column and turn None into NaN
    out["text"] = pd.Series([t if t.strip() else None for t in df["text"]], index=out.index,
                            dtype=object)
    if condition == "empty" and out["text"].notna().any():
        raise ValueError("texts/empty.csv contains text")
    return out


def _texts_fingerprint(condition: str, domain: str) -> str:
    df = common.load(condition)
    df = df[df.domain == domain][["split", "sample_id", "text", "attacked",
                                  "claimed_direction"]]
    return hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()[:16]


# --- resumable units -------------------------------------------------------------------------

def protocol(model: str, domain: str, condition: str, seeds=SEEDS) -> dict:
    """Everything that determines a unit's predictions: the exact checkpoints and texts."""
    return dict(model=model, domain=domain, condition=condition, config=CONFIG,
                seeds=list(seeds), splits=list(SPLITS),
                checkpoints={str(s): _sha(checkpoint(model, domain, s)) for s in seeds},
                texts=_texts_fingerprint(condition, domain))


def unit_paths(model: str, domain: str, condition: str, unit_dir: Path | None = None
               ) -> dict[str, Path]:
    d = unit_dir or UNIT_DIR
    stem = f"{model}_{domain}_{condition}"
    return {"preds": d / f"{stem}.csv", "done": d / f"{stem}.done.json"}


def unit_done(model: str, domain: str, condition: str, proto: dict,
              unit_dir: Path | None = None) -> bool:
    paths = unit_paths(model, domain, condition, unit_dir)
    if not (paths["done"].exists() and paths["preds"].exists()):
        return False
    saved = json.loads(paths["done"].read_text())
    if saved != proto:
        raise RuntimeError(f"{paths['done'].name} was produced from different checkpoints or "
                           f"texts:\n  saved   {saved}\n  current {proto}\n"
                           "Re-run with --fresh to discard saved units.")
    return True


def run_unit(model: str, domain: str, condition: str, windows: dict, train_std: float,
             train_mean: float, seeds=SEEDS, use_mlflow: bool = True) -> pd.DataFrame:
    """Load each seed's checkpoint and predict every val + test window under ``condition``."""
    frames = []
    texts = {}
    for split in SPLITS:
        w = windows[split]
        ids = w.target_start.strftime("%Y-%m-%d")
        t = condition_texts(condition, domain, split, ids)
        if condition == "real":  # the join must reproduce the windows' own text exactly
            own = [None if v is None or not str(v).strip() else v for v in w.text]
            if list(t["text"]) != own:
                raise ValueError(f"texts/real.csv disagrees with the {domain}/{split} windows")
        texts[split] = (ids, t)
    for seed in seeds:
        m = MODEL_CLASSES[model].load(checkpoint(model, domain, seed))
        n_nan = 0
        for split in SPLITS:
            w = windows[split]
            ids, t = texts[split]
            pred = m.predict_windows(w.X, list(t["text"]))[:, 0]
            n_nan += int(np.isnan(pred).sum())
            frames.append(pd.DataFrame({
                "model": model, "domain": domain, "split": split, "condition": condition,
                "seed": seed, "sample_id": ids, "y_true": w.Y[:, 0], "y_pred": pred,
                "y_last": w.X[:, -1], "train_mean": train_mean, "train_std": train_std,
                "has_text": t["has_text"].to_numpy(bool),
                "attacked": t["attacked"].to_numpy(bool),
                "claimed_direction": t["claimed_direction"].to_numpy(int)}))
        if use_mlflow:
            import mlflow
            with mlflow.start_run(run_name=f"{model}-{domain}-{condition}-{seed}"):
                mlflow.log_params({"model": model, "domain": domain, "condition": condition,
                                   "seed": seed, "config": CONFIG, "horizon": 1,
                                   "mode": "inference_only"})
                mlflow.log_metrics({"n_predictions": sum(len(windows[s]) for s in SPLITS),
                                    "n_nan": n_nan})
        del m
    return pd.concat(frames, ignore_index=True)[COLUMNS]


def run(models=MODELS, domains=DOMAINS, conditions=CONDITIONS, seeds=SEEDS,
        use_mlflow: bool = True, fresh: bool = False, unit_dir: Path | None = None,
        predictions: Path | None = None, experiment: str = "p5_experiments") -> pd.DataFrame:
    """Run every unfinished unit, then collect and check all finished units.

    ``unit_dir`` / ``predictions`` / ``experiment``: R1 (``src/run_r1.py``) runs its heads into
    its own folder and table, so P5's ``p5_predictions.csv`` never changes."""
    unit_dir = unit_dir or UNIT_DIR
    if fresh and unit_dir.exists():
        shutil.rmtree(unit_dir)
    if use_mlflow:
        import mlflow
        mlflow.set_experiment(experiment)
    for domain in domains:
        splits = load_splits(domain)
        train_values = splits.train["value"].to_numpy(float)
        mean, std = float(train_values.mean()), float(train_values.std())
        windows = split_windows(splits, lookback_for(domain), horizon=1)
        for model in models:
            for condition in conditions:
                proto = protocol(model, domain, condition, seeds)
                if unit_done(model, domain, condition, proto, unit_dir):
                    print(f"{domain:<10} {model:<13} {condition:<12} already done, skipped",
                          flush=True)
                    continue
                preds = run_unit(model, domain, condition, windows, std, mean, seeds,
                                 use_mlflow)
                bert = any(e.is_loaded for e in _ENCODERS.values())
                paths = unit_paths(model, domain, condition, unit_dir)
                unit_dir.mkdir(parents=True, exist_ok=True)
                _atomic_write(paths["preds"], lambda f: preds.to_csv(f, index=False))
                _atomic_write(paths["done"],  # marker last: a crash never looks finished
                              lambda f: f.write_text(json.dumps(proto, indent=1)))
                print(f"{domain:<10} {model:<13} {condition:<12} {len(preds):>5} rows, "
                      f"NaN {int(preds.y_pred.isna().sum())}"
                      f"{'  (BERT loaded: cache miss)' if bert else ''}", flush=True)
                unload_encoders()  # free BERT if a cache miss loaded it
                del preds
                gc.collect()
    return collect(unit_dir, predictions)


def collect(unit_dir: Path | None = None, predictions: Path | None = None) -> pd.DataFrame:
    unit_dir, predictions = unit_dir or UNIT_DIR, predictions or PREDICTIONS
    stems = sorted(d.name.removesuffix(".done.json") for d in unit_dir.glob("*.done.json"))
    if not stems:
        raise RuntimeError(f"no finished units in {unit_dir}")
    preds = pd.concat([pd.read_csv(unit_dir / f"{s}.csv", dtype={"sample_id": str})
                       for s in stems], ignore_index=True)
    preds.to_csv(predictions, index=False)
    print(f"collected {len(stems)} units, {len(preds)} rows -> {predictions}", flush=True)
    return preds


# --- integrity checks (not metrics) ----------------------------------------------------------

def check_text_blind(preds: pd.DataFrame) -> pd.DataFrame:
    """Models 1–2 must give bit-identical predictions under every condition (they ignore
    text by construction). Asserted on the actual predictions, not assumed."""
    from src.metrics import is_text_blind
    rows = []
    for (model, domain, seed, split), g in preds[preds.model.isin(TEXT_BLIND)].groupby(
            ["model", "domain", "seed", "split"]):
        by = {c: d.sort_values("sample_id").y_pred.to_numpy() for c, d in g.groupby("condition")}
        ok = is_text_blind(by["real"], *[v for c, v in by.items() if c != "real"])
        rows.append(dict(model=model, domain=domain, seed=seed, split=split,
                         conditions=len(by), identical=ok))
    out = pd.DataFrame(rows, columns=["model", "domain", "seed", "split", "conditions",
                                      "identical"])
    if not out.identical.all():
        raise AssertionError(f"text-blind model changed with the text:\n{out[~out.identical]}")
    return out


def check_against_p2(preds: pd.DataFrame, p2_path: Path = P2_PREDICTIONS) -> pd.DataFrame:
    """P5's ``real`` (and, for Models 3/4, ``empty``) predictions must reproduce P2's saved
    ones: same checkpoints, same windows, same text → same forecast. Max |diff| per model."""
    p2 = pd.read_csv(p2_path, dtype={"sample_id": str})
    p2 = p2[(p2.config == CONFIG) & (p2.split.isin(SPLITS))]
    keys = ["model", "domain", "split", "condition", "seed", "sample_id"]
    m = preds.merge(p2[keys + ["y_pred"]], on=keys, suffixes=("", "_p2"))
    m["abs_diff"] = (m.y_pred - m.y_pred_p2).abs()
    m["rel_diff"] = m.abs_diff / m.y_pred_p2.abs().clip(lower=1e-12)
    out = m.groupby(["model", "condition"]).agg(rows=("abs_diff", "size"),
                                                max_abs=("abs_diff", "max"),
                                                max_rel=("rel_diff", "max")).reset_index()
    expected = p2.merge(preds[keys].drop_duplicates(), on=keys).shape[0]
    if len(m) != expected or (out.max_rel > 1e-5).any():
        raise AssertionError(f"P5 predictions do not reproduce P2:\n{out}")
    return out


def check_embedding_scalers(domains=DOMAINS, seeds=SEEDS,
                            models=("DLinear+BERT", "PatchTST+BERT")) -> pd.DataFrame:
    """Every Model 3/4 checkpoint's ``EmbeddingScaler`` must equal the mean/std of the
    **train** windows' texts recomputed now — proof that attacked text never refit it
    (embeddings come from the cache; BERT is not loaded)."""
    from src.models.late_fusion import _clean, _encoder
    rows = []
    for domain in domains:
        w = split_windows(load_splits(domain), lookback_for(domain), horizon=1)["train"]
        real = [t for t in (_clean(t) for t in w.text) if t is not None]
        for model in models:
            for seed in seeds:
                m = MODEL_CLASSES[model].load(checkpoint(model, domain, seed))
                e = _encoder(m.encoder_name).encode(real)
                mean, std = e.mean(0), e.std(0) + 1e-6
                diff = max(np.abs(m.emb_scaler.mean.numpy() - mean).max(),
                           np.abs(m.emb_scaler.std.numpy() - std).max())
                rows.append(dict(model=model, domain=domain, seed=seed,
                                 train_texts=len(real), max_abs_diff=float(diff)))
    out = pd.DataFrame(rows)
    if (out.max_abs_diff > 1e-5).any():
        raise AssertionError(f"embedding scaler differs from the train-text statistics:\n{out}")
    return out


# --- Model 5 ---------------------------------------------------------------------------------

def qwen_rows() -> pd.DataFrame:
    """Model 5's predictions (P2.4 real/none + P5 Colab run) in this runner's schema.

    ``none`` is renamed ``empty`` (the same thing, PROJECT_STATUS §6a); ``seed`` = 0 (one
    greedy run). ``parse_ok`` / ``scale_ok`` / ``reused_from`` are kept for failure reporting,
    ``setup`` / ``setup_class`` for provenance.

    **Refuses to mix software setups.** Every comparison (fake vs real for DSS / Flip Rate,
    real vs none for sensitivity, M5 vs M3/M4 per window) pairs Model 5 answers across
    conditions, and pooled numbers pair them across windows. If two answers came from setups
    that were not certified equivalent by the reproducibility probe, a software change would
    be measured as a text effect. So the whole table must share one ``setup_class``.
    """
    from src.models.qwen_forecaster import load_all_predictions
    q = load_all_predictions()
    classes = q.groupby(["domain", "split", "sample_id"]).setup_class.nunique()
    if (classes > 1).any() or q.setup_class.nunique() != 1:
        raise ValueError(f"Model 5 answers come from {q.setup_class.nunique()} software setups "
                         f"({int((classes > 1).sum())} windows mix them) — comparisons would "
                         "mix a setup change into the text effect; regenerate on one setup")
    q["condition"] = q["condition"].replace({"none": "empty"})
    q["y_last"] = [json.loads(x)[-1] for x in q.lookback]
    keys = ["domain", "split", "sample_id"]
    meta = pd.concat([common.load(c).assign(condition=c) for c in CONDITIONS])
    q = q.merge(meta[keys + ["condition", "has_text", "attacked", "claimed_direction"]],
                on=keys + ["condition"], how="left", validate="one_to_one")
    q["model"], q["seed"] = "Qwen2.5-1.5B", 0
    extra = [c for c in ("parse_ok", "scale_ok", "reused_from", "setup", "setup_class")
             if c in q]
    return q[COLUMNS + extra]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="discard saved units first")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--domains", nargs="+", default=DOMAINS)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--collect-only", action="store_true",
                    help="no inference: rebuild p5_predictions.csv from the committed units "
                         "(R7 reproduce.py; the scaler check needs the gitignored checkpoints)")
    args = ap.parse_args()
    if args.collect_only:
        preds = collect()
    else:
        preds = run(args.models, args.domains, args.conditions, use_mlflow=not args.no_mlflow,
                    fresh=args.fresh)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        if not args.collect_only:
            scal = check_embedding_scalers(args.domains)
            print(f"embedding scalers: {len(scal)} Model 3/4 checkpoints match the train-text "
                  f"statistics (max |diff| {scal.max_abs_diff.max():.1e})")
        blind = check_text_blind(preds)
        if len(blind):
            print(f"text-blind check: {len(blind)} (model, domain, seed, split) groups, "
                  f"all identical across {blind.conditions.max()} conditions")
        print("agreement with P2 saved predictions:")
        print(check_against_p2(preds).to_string(index=False))
        counts = preds.groupby(["model", "condition"]).size().unstack()
        print(counts.to_string())


if __name__ == "__main__":
    main()
