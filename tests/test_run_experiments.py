"""Tests for the P5a runner's integrity checks and its text join.

The checks are what make the generated predictions trustworthy without looking at any
metric: Models 1–2 identical across conditions, P5 reproducing P2's saved predictions, and
the condition texts joined to the right windows. The full run needs the local checkpoints
and ``texts/`` (both gitignored), so those tests skip when the files are absent.
"""

import numpy as np
import pandas as pd
import pytest

from src import run_experiments as rx
from src.attacks.common import TEXTS_DIR


def _preds(models=("DLinear",), conditions=("real", "empty", "flipped"), shift=None):
    rows = []
    for m in models:
        for c in conditions:
            for seed in (2021, 2022):
                for i in range(4):
                    y = 10.0 + i + (shift.get((m, c), 0.0) if shift else 0.0)
                    rows.append(dict(model=m, domain="Economy", split="test", condition=c,
                                     seed=seed, sample_id=f"2020-0{i + 1}-01", y_pred=y))
    return pd.DataFrame(rows)


def test_text_blind_check_passes_identical_and_ignores_text_models():
    preds = _preds(models=("DLinear", "DLinear+BERT"), shift={("DLinear+BERT", "flipped"): 1.0})
    out = rx.check_text_blind(preds)
    assert out.identical.all() and set(out.model) == {"DLinear"} and (out.conditions == 3).all()


def test_text_blind_check_fails_if_a_numbers_only_model_moves():
    with pytest.raises(AssertionError, match="text-blind"):
        rx.check_text_blind(_preds(shift={("DLinear", "flipped"): 1e-9}))


def test_check_against_p2_detects_a_changed_prediction(tmp_path):
    preds = _preds(models=("DLinear+BERT",), conditions=("real", "empty", "flipped"))
    p2 = preds[preds.condition.isin(["real", "empty"])].assign(config="main")
    p2.to_csv(tmp_path / "p2.csv", index=False)
    out = rx.check_against_p2(preds, tmp_path / "p2.csv")
    assert set(out.condition) == {"real", "empty"} and (out.max_abs == 0).all()
    p2.loc[p2.index[0], "y_pred"] += 1.0
    p2.to_csv(tmp_path / "p2.csv", index=False)
    with pytest.raises(AssertionError, match="do not reproduce P2"):
        rx.check_against_p2(preds, tmp_path / "p2.csv")


needs_texts = pytest.mark.skipif(not (TEXTS_DIR / "flipped.csv").exists(),
                                 reason="texts/ not generated (python -m src.attacks.generate)")


@needs_texts
def test_condition_texts_follow_window_order_and_keep_none():
    ids = ["2018-08-01", "2018-06-01", "2018-07-01"]          # deliberately out of order
    t = rx.condition_texts("flipped", "Economy", "test", ids)
    assert list(t.index) == ids
    e = rx.condition_texts("empty", "Economy", "test", ids)
    assert e.text.dtype == object and all(v is None for v in e.text)   # None, not NaN


@needs_texts
def test_condition_texts_refuse_unknown_windows():
    with pytest.raises(ValueError, match="lacks"):
        rx.condition_texts("flipped", "Economy", "test", ["1900-01-01"])


@needs_texts
@pytest.mark.skipif(not rx.checkpoint("DLinear+BERT", "Economy", 2021).exists(),
                    reason="P2 checkpoints not trained")
def test_unit_reproduces_p2_and_only_text_changes(tmp_path):
    from src.data.splits import load_splits
    from src.data.windows import lookback_for, split_windows
    w = split_windows(load_splits("Economy"), lookback_for("Economy"), horizon=1)
    tv = load_splits("Economy").train["value"].to_numpy(float)
    kw = dict(windows=w, train_std=float(tv.std()), train_mean=float(tv.mean()),
              seeds=[2021], use_mlflow=False)
    real = rx.run_unit("DLinear+BERT", "Economy", "real", **kw)
    flip = rx.run_unit("DLinear+BERT", "Economy", "flipped", **kw)
    blind = rx.run_unit("DLinear", "Economy", "flipped", **kw)
    assert (real.sample_id.to_numpy() == flip.sample_id.to_numpy()).all()
    np.testing.assert_array_equal(real.y_last, flip.y_last)
    moved = flip.y_pred.to_numpy() != real.y_pred.to_numpy()
    assert moved[flip.attacked.to_numpy()].all() and not moved[~flip.attacked.to_numpy()].any()
    np.testing.assert_array_equal(blind.y_pred, rx.run_unit("DLinear", "Economy", "real",
                                                            **kw).y_pred)
