"""R1 inference: the four heads on all 7 conditions, val + test (PROJECT_STATUS §10d).

The P5a runner (``run_experiments.run``), unchanged, pointed at R1's own unit folder and
predictions table — so ``p5_predictions.csv`` (already reported) never changes. Inference only:
each seed's checkpoint from ``train_r1`` is loaded and fed the condition texts; the only thing
that differs between two conditions is the text.

Integrity checks (not results), as in P5a:

* every head's ``EmbeddingScaler`` equals its encoder's train-text statistics recomputed now
  (attacked text never refit it);
* ``real`` and ``empty`` reproduce the predictions ``train_r1`` saved right after training
  (same checkpoints, windows, texts → same forecast).

Outputs: ``results/r1_units/`` (gitignored), ``results/r1_predictions.csv`` (gitignored; every
R1 metric is derived from it), MLflow experiment ``r1_experiments``.

Run: ``python -m src.run_r1`` (``--no-mlflow``, ``--fresh``).
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.models.train_r1 import PREDICTIONS as TRAIN_PREDICTIONS
from src.models.train_r1 import R1_MODELS
from src.run_experiments import RESULTS, check_against_p2, check_embedding_scalers, collect, run

UNIT_DIR = RESULTS / "r1_units"
PREDICTIONS = RESULTS / "r1_predictions.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--collect-only", action="store_true",
                    help="no inference: rebuild r1_predictions.csv from the committed units (R7)")
    a = ap.parse_args()
    if a.collect_only:
        preds = collect(UNIT_DIR, PREDICTIONS)
    else:
        preds = run(models=R1_MODELS, use_mlflow=not a.no_mlflow, fresh=a.fresh,
                    unit_dir=UNIT_DIR, predictions=PREDICTIONS, experiment="r1_experiments")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        if not a.collect_only:
            scal = check_embedding_scalers(models=R1_MODELS)
            print(f"embedding scalers: {len(scal)} R1 checkpoints match their encoder's "
                  f"train-text statistics (max |diff| {scal.max_abs_diff.max():.1e})")
        print("agreement with the predictions saved at training time:")
        print(check_against_p2(preds, TRAIN_PREDICTIONS).to_string(index=False))
        print(preds.groupby(["model", "condition"]).size().unstack().to_string())
        print(f"NaN predictions: {int(preds.y_pred.isna().sum())}")


if __name__ == "__main__":
    main()
