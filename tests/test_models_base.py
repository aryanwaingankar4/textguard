"""Tests for src/models/base.py. The central guarantee: every model obeys one predict contract."""

import numpy as np
import pytest

from src.models.base import (BaseForecaster, Forecaster, check_numbers, check_output,
                             normalise_text)
from src.models.dummy import DummyForecaster


class _DuckModel:
    """A third-party model that never imports our base class."""
    horizon = 1

    def predict(self, numbers, text):
        return np.array([float(numbers[-1])])


class _Echo(BaseForecaster):
    """Records what _predict actually receives, and returns a configurable output."""

    def __init__(self, out=0.0, **kw):
        super().__init__(**kw)
        self.out = out
        self.seen_text = "unset"

    def _predict(self, numbers, text):
        self.seen_text = text
        return self.out


# --- the Protocol: TextGuard accepts anything with this shape ---------------------------

def test_duck_typed_model_satisfies_protocol_without_inheritance():
    assert isinstance(_DuckModel(), Forecaster)


def test_our_wrappers_satisfy_protocol():
    assert isinstance(DummyForecaster(), Forecaster)


def test_object_without_predict_is_not_a_forecaster():
    class NoPredict:
        horizon = 1
    assert not isinstance(NoPredict(), Forecaster)


# --- numbers -----------------------------------------------------------------------------

def test_numbers_accepts_list_and_returns_float_array():
    arr = check_numbers([1, 2, 3])
    assert arr.dtype == float and arr.shape == (3,)


def test_numbers_rejects_2d_multivariate_input():
    # Univariate by design: a co-shipped numeric column would confound "text gains".
    with pytest.raises(ValueError, match="1-D"):
        check_numbers(np.ones((4, 2)))


@pytest.mark.parametrize("bad", [[1.0, np.nan], [1.0, np.inf]])
def test_numbers_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="NaN or inf"):
        check_numbers(bad)


def test_numbers_rejects_empty():
    with pytest.raises(ValueError):
        check_numbers([])


def test_lookback_is_enforced_exactly_not_truncated():
    with pytest.raises(ValueError, match="lookback"):
        _Echo(lookback=3).predict([1, 2, 3, 4], None)
    _Echo(lookback=3).predict([1, 2, 3], None)


# --- text --------------------------------------------------------------------------------

@pytest.mark.parametrize("blank", [None, "", "   ", "\n\t"])
def test_blank_text_normalises_to_none(blank):
    assert normalise_text(blank) is None


def test_non_blank_text_passes_through_unchanged():
    assert normalise_text("  rates rose ") == "  rates rose "


def test_non_string_text_raises():
    with pytest.raises(TypeError):
        normalise_text(3.5)


def test_wrapper_receives_none_for_blank_text():
    m = _Echo()
    m.predict([1, 2], "  ")
    assert m.seen_text is None


def test_empty_condition_equals_numbers_only_prediction():
    # `empty`, has_text=False and TextGuard's A must be the same prediction.
    m = DummyForecaster()
    np.testing.assert_array_equal(m.predict([1, 2, 3], ""), m.predict([1, 2, 3], None))


def test_text_defaults_to_none():
    m = DummyForecaster()
    np.testing.assert_array_equal(m.predict([1, 2, 3]), m.predict([1, 2, 3], None))


# --- output ------------------------------------------------------------------------------

def test_scalar_output_is_reshaped_for_horizon_1():
    assert _Echo(out=5.0).predict([1.0], None).shape == (1,)


def test_wrong_output_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        _Echo(out=[1.0, 2.0], horizon=1).predict([1.0], None)


def test_nan_output_is_allowed_as_a_failure_marker():
    out = _Echo(out=np.nan).predict([1.0], "text")
    assert out.shape == (1,) and np.isnan(out[0])


def test_inf_output_is_a_bug_and_raises():
    with pytest.raises(ValueError, match="inf"):
        check_output([np.inf], 1)


def test_multi_horizon_shape():
    assert DummyForecaster(horizon=3).predict([1, 2, 3], None).shape == (3,)


@pytest.mark.parametrize("kw", [{"horizon": 0}, {"lookback": 0}])
def test_invalid_constructor_args(kw):
    with pytest.raises(ValueError):
        _Echo(**kw)


# --- batching ----------------------------------------------------------------------------

def test_predict_batch_matches_single_predictions():
    m = DummyForecaster()
    nums = [[1, 2, 3], [5, 4, 3]]
    texts = ["prices rose", None]
    batch = m.predict_batch(nums, texts)
    for n, t, b in zip(nums, texts, batch):
        np.testing.assert_array_equal(b, m.predict(n, t))


def test_predict_batch_length_mismatch_raises():
    with pytest.raises(ValueError):
        DummyForecaster().predict_batch([[1, 2]], [None, None])
