"""Tests for src/models/dummy.py — the hand-checkable fixture for P4 metrics and P6 TextGuard."""

import numpy as np
import pytest

from src.models.dummy import DummyForecaster, text_direction

# PHASES.md worked example
HISTORY = [100, 104, 108, 112]
FAKE = "Demand will fall sharply."
REAL = "demand rising"


def _p(model, numbers, text):
    return float(model.predict(numbers, text)[0])


# --- worked example ----------------------------------------------------------------------

def test_worked_example_a_numbers_only_is_116():
    assert _p(DummyForecaster(), HISTORY, None) == 116.0


def test_worked_example_b_fake_text_is_106():
    assert _p(DummyForecaster(), HISTORY, FAKE) == 106.0


def test_worked_example_real_text_is_120():
    # The DSS line in PHASES.md uses y_real = 120: -1 * (106 - 120) / 10 = +1.4
    assert _p(DummyForecaster(), HISTORY, REAL) == 120.0


def test_worked_example_dss_is_reproducible_from_dummy_outputs():
    m = DummyForecaster()
    d, sigma = -1, 10.0  # fake text claims "down"
    dss = d * (_p(m, HISTORY, FAKE) - _p(m, HISTORY, REAL)) / sigma
    assert dss == pytest.approx(1.4)


# --- sensitivity: deaf vs text-following -------------------------------------------------

@pytest.mark.parametrize("text", [FAKE, REAL, "ILI was 2.1%, above the national baseline"])
def test_sensitivity_zero_ignores_text(text):
    m = DummyForecaster(sensitivity=0.0)
    assert _p(m, HISTORY, text) == _p(m, HISTORY, None)


def test_sensitivity_scales_the_text_shift_linearly():
    half = DummyForecaster(sensitivity=0.5)
    assert _p(half, HISTORY, FAKE) == 116.0 - 0.5 * 10.0


def test_sensitive_model_is_fooled_by_flipped_text():
    # The real text pushes up, the flipped text pushes down: forecasts move in opposite directions.
    m = DummyForecaster()
    a = _p(m, HISTORY, None)
    assert _p(m, HISTORY, "demand rising") > a > _p(m, HISTORY, "demand falling")


# --- change-language cues ----------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("The trade deficit increased by $4.6 billion.", 1),
    ("Unemployment rose to 5.7%.", 1),
    ("Travel decreased compared with last year.", -1),
    ("Prices fell sharply.", -1),
    ("The rate was unchanged at 8.2%.", 0),
    ("Exports rose while imports fell.", 0),  # one up, one down -> neutral
])
def test_change_language_direction(text, expected):
    assert text_direction(text) == expected


def test_intensifier_applies_only_with_an_adverb():
    m = DummyForecaster()
    assert _p(m, HISTORY, "demand fell") == 112.0
    assert _p(m, HISTORY, "demand fell sharply") == 106.0


def test_neutral_text_equals_numbers_only():
    m = DummyForecaster()
    assert _p(m, HISTORY, "The rate was unchanged.") == _p(m, HISTORY, None)


# --- level-language cues (fixture for TextGuard's level branch) --------------------------

@pytest.mark.parametrize("text,expected", [
    ("ILI was 2.1%, above the national baseline.", 1),
    ("ILI was 2.1%, below the national baseline.", -1),
    ("Activity remained above baseline.", 1),
    ("Visits were below the baseline of 2.2%.", -1),
    ("ABOVE THE REGIONAL BASELINE", 1),
])
def test_level_language_direction(text, expected):
    assert text_direction(text) == expected


def test_level_flip_moves_forecast_in_opposite_directions():
    # Flat ILI history: A = 2.0 exactly, so the text shift is the whole difference.
    m = DummyForecaster(step=0.5)
    flat = [2.0, 2.0, 2.0, 2.0]
    assert _p(m, flat, None) == 2.0
    assert _p(m, flat, "ILI was 2.1%, above the national baseline") == 2.5
    assert _p(m, flat, "ILI was 2.1%, below the national baseline") == 1.5


def test_baseline_without_above_or_below_is_neutral():
    assert text_direction("ILI was at the national baseline.") == 0


# --- numbers path ------------------------------------------------------------------------

def test_single_value_history_extrapolates_flat():
    assert _p(DummyForecaster(), [7.0], None) == 7.0


def test_multi_horizon_extrapolates_each_step_and_shifts_all():
    m = DummyForecaster(horizon=3)
    np.testing.assert_array_equal(m.predict(HISTORY, None), [116, 120, 124])
    np.testing.assert_array_equal(m.predict(HISTORY, "demand rising"), [120, 124, 128])
