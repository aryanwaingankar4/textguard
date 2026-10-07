"""Tests for the numbers-only baselines: Persistence, DLinear, PatchTST.

The neural tests train for a couple of epochs on a synthetic series: they check the
contract (raw units, text ignored, train-only scaler, checkpoints), not accuracy.
They are skipped if MM-TSFlib is not cloned into external/.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.splits import chronological_split
from src.data.windows import split_windows
from src.models import _mmtsflib
from src.models.base import Forecaster
from src.models.persistence import PersistenceForecaster

needs_mmtsflib = pytest.mark.skipif(not _mmtsflib.available(), reason="MM-TSFlib not cloned")


# --- persistence -------------------------------------------------------------------------

def test_persistence_returns_last_value():
    assert PersistenceForecaster().predict([3.0, 5.0, 9.0], None)[0] == 9.0


def test_persistence_ignores_text_and_repeats_over_horizon():
    m = PersistenceForecaster(horizon=3)
    np.testing.assert_array_equal(m.predict([1.0, 2.0], "prices rose sharply"), [2, 2, 2])


def test_persistence_batched_matches_single():
    m = PersistenceForecaster(horizon=2)
    X = np.array([[1.0, 2.0], [5.0, 4.0]])
    np.testing.assert_array_equal(m.predict_windows(X), [m.predict(x) for x in X])


# --- neural models -----------------------------------------------------------------------

def _data(seq_len=8, horizon=1, n=120, offset=1000.0):
    """Sine + trend at a large offset, so 'forgot to unscale' is obvious."""
    t = np.arange(n)
    start = pd.date_range("2000-01-01", periods=n, freq="MS")
    df = pd.DataFrame({"start_date": start, "end_date": start + pd.offsets.MonthEnd(0),
                       "value": offset + 10 * np.sin(t / 3) + 0.2 * t, "has_text": True})
    s = chronological_split(df)
    return s, split_windows(s, seq_len, horizon)


def _models(seq_len=8, horizon=1):
    from src.models.dlinear import DLinearForecaster
    from src.models.patchtst import PatchTSTForecaster
    return [DLinearForecaster(seq_len, horizon, epochs=2),
            PatchTSTForecaster(seq_len, horizon, epochs=2)]


@needs_mmtsflib
@pytest.mark.parametrize("idx", [0, 1])
def test_neural_predict_contract_raw_units_and_text_ignored(idx, tmp_path):
    s, w = _data()
    m = _models()[idx].fit(w["train"], w["val"], s.train["value"], checkpoint=tmp_path / "m.pt")
    assert isinstance(m, Forecaster)
    x = w["test"].X[0]
    out = m.predict(x, None)
    assert out.shape == (1,)
    assert 900 < out[0] < 1100                # raw units, not z-scores
    np.testing.assert_array_equal(out, m.predict(x, "prices collapsed"))  # text ignored
    np.testing.assert_allclose(m.predict_windows(w["test"].X[:1])[0], out)


@needs_mmtsflib
def test_scaler_is_fit_on_train_split_only():
    s, w = _data()
    m = _models()[0].fit(w["train"], w["val"], s.train["value"])
    assert m.mean == pytest.approx(s.train["value"].mean())
    assert m.std == pytest.approx(s.train["value"].std(ddof=0))


@needs_mmtsflib
@pytest.mark.parametrize("idx", [0, 1])
def test_checkpoint_round_trip_gives_identical_predictions(idx, tmp_path):
    s, w = _data()
    m = _models()[idx].fit(w["train"], w["val"], s.train["value"], checkpoint=tmp_path / "m.pt")
    loaded = type(m).load(tmp_path / "m.pt")
    np.testing.assert_allclose(loaded.predict_windows(w["test"].X), m.predict_windows(w["test"].X))


@needs_mmtsflib
def test_same_seed_is_reproducible():
    s, w = _data()
    a, b = (_models()[0].fit(w["train"], w["val"], s.train["value"]) for _ in range(2))
    np.testing.assert_array_equal(a.predict_windows(w["test"].X), b.predict_windows(w["test"].X))


@needs_mmtsflib
def test_patchtst_window_patching_reads_every_input():
    from src.models.patchtst import PatchTSTForecaster
    for seq_len in (8, 24):
        m = PatchTSTForecaster(seq_len)
        net = m.build_network()
        pl, st = net.patch_embedding.patch_len, net.patch_embedding.stride
        assert pl > 1 and st <= pl                 # overlapping patches, no gaps
        assert net.head_nf // 512 == 4             # four half-window patches


@needs_mmtsflib
def test_patchtst_as_shipped_degenerates_at_horizon_1():
    # Documents why patching="window" exists: MM-TSFlib clamps patch_len to pred_len.
    net = _mmtsflib.model_class("PatchTST")(_mmtsflib.default_configs(8, 1))
    assert net.patch_embedding.patch_len == 1 and net.patch_embedding.stride == 8


@needs_mmtsflib
def test_multi_horizon_shape():
    s, w = _data(horizon=6)
    for m in _models(horizon=6):
        m.fit(w["train"], w["val"], s.train["value"])
        assert m.predict(w["test"].X[0]).shape == (6,)


def test_lookback_is_enforced():
    with pytest.raises(ValueError, match="lookback"):
        PersistenceForecaster(lookback=8).predict(np.ones(5))


# --- tuned protocol ----------------------------------------------------------------------

@needs_mmtsflib
def test_tune_lr_picks_lowest_val_loss_and_never_sees_test(monkeypatch):
    from src.models import train_baselines as tb
    monkeypatch.setattr(tb, "TUNED", dict(epochs=2, patience=2))
    s, w = _data()
    w_no_test = {k: v for k, v in w.items() if k != "test"}  # tuning must not need test
    best_lr, model, rows = tb.tune_lr("DLinear", 8, w_no_test, s.train["value"])
    assert [r["lr"] for r in rows] == list(tb.LR_GRID)
    assert best_lr == min(rows, key=lambda r: r["best_val_loss"])["lr"]
    assert model.lr == best_lr


def test_repro_config_uses_mm_tsflib_recipe():
    from src.models import train_baselines as tb
    m = tb.make_model("DLinear", "repro", 8, 6, seed=2021)
    assert (m.epochs, m.patience, m.lr) == (10, 5, 1e-4)
