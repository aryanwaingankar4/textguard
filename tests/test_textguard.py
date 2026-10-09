"""TextGuard (P6): evidence, gate, defences, and the model-agnostic wrapper.

The headline test is the PHASES.md worked example on ``DummyForecaster`` — a model TextGuard
was never written around — reproduced with no change to TextGuard: g = 0.1, final = 115, warning.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import metrics
from src.models.base import Forecaster
from src.models.dummy import DummyForecaster
from src.textguard.baselines import always_trust, clip, mix, never_trust, textguard
from src.textguard.gate import branch_gate, combine
from src.textguard.text_direction import Claim, fixed_judge, judge_from
from src.textguard.textguard import TextGuard, gate
from src.textguard.trend_detector import (Evidence, change_evidence, level_evidence,
                                          window_for, window_slope)

X = [100.0, 104.0, 108.0, 112.0]
SIGMA = 10.0
FAKE, REAL = "Demand will fall sharply.", "demand rising"


# --- evidence ------------------------------------------------------------------------------

def test_change_evidence_reproduces_worked_example():
    ev = change_evidence(X, SIGMA, window=4)
    assert ev == Evidence(1, 1.0)                       # fitted rise 12 = 1.2σ, capped at 1
    assert window_slope(X, 4) == pytest.approx(4.0)


def test_change_evidence_scales_with_movement_and_is_neutral_when_flat():
    assert change_evidence([5, 5, 5, 5], 1.0, 4) == Evidence(0, 0.0)
    ev = change_evidence([10, 9.5, 9, 8.5], 10.0, 4)      # fitted fall 1.5 = 0.15σ
    assert ev.direction == -1 and ev.strength == pytest.approx(0.15)
    # only the last `window` values count: an old spike does not set the recent trend
    assert change_evidence([0, 100, 1, 2, 3], 10.0, 3).direction == 1


def test_window_grid_matches_preregistration():
    assert [window_for(8, f) for f in (0.25, 0.5, 1.0)] == [3, 4, 8]
    assert [window_for(24, f) for f in (0.25, 0.5, 1.0)] == [6, 12, 24]


def test_level_evidence_compares_latest_value_with_stated_reference():
    assert level_evidence([2.5, 2.9, 3.2], 1.0, 2.1) == Evidence(1, pytest.approx(1.0))
    ev = level_evidence([2.5, 2.9, 1.8], 1.0, 2.1)
    assert ev.direction == -1 and ev.strength == pytest.approx(0.3)
    assert level_evidence([2.5], 1.0, None) == Evidence(0, 0.0)


def test_evidence_rejects_bad_sigma_and_windows():
    with pytest.raises(ValueError):
        change_evidence(X, 0.0, 4)
    with pytest.raises(ValueError):
        change_evidence([1.0, np.nan], 1.0, 2)


# --- gate ----------------------------------------------------------------------------------

def test_gate_rule():
    strong_up = Evidence(1, 1.0)
    assert branch_gate(-1, 0.9, strong_up) == pytest.approx(0.1)   # disagree
    assert branch_gate(1, 0.9, strong_up) == 1.0                   # agree
    assert branch_gate(0, 0.9, strong_up) == 1.0                   # text claims nothing
    assert branch_gate(-1, 0.9, Evidence(0, 0.0)) == 1.0           # numbers show nothing
    assert branch_gate(-1, 0.5, Evidence(1, 0.4)) == pytest.approx(0.8)


def test_several_claims_take_the_least_credible():
    up = Evidence(1, 1.0)
    checks = [(Claim("level", 1, 0.9), up, 1.0), (Claim("change", -1, 0.6), up, 0.4)]
    assert combine(checks).g == 0.4
    assert combine([]).g == 1.0


def test_health_us_change_flip_is_caught_by_the_change_branch():
    """A Health_US fake that keeps "above the baseline" but flips "increased" (P3 change
    variant) agrees on the level and is caught by the change claim — hence g = min."""
    x = [2.0, 2.2, 2.4, 2.6]
    claims = [Claim("level", 1, 0.9, 2.2), Claim("change", -1, 0.8)]
    res = gate(claims, x, sigma=0.5, window=4)
    assert [g for _, _, g in res.checks] == [1.0, pytest.approx(0.2)]
    assert res.g == pytest.approx(0.2)


# --- defences ------------------------------------------------------------------------------

def test_baselines_and_textguard_modes():
    A, B = np.array([116.0, 10.0]), np.array([106.0, 12.0])
    assert np.allclose(always_trust(A, B).final, B)
    assert np.allclose(never_trust(A, B).final, A)
    c = clip(A, B, sigma=np.array([10.0, 10.0]), k=0.5)            # cap 5
    assert np.allclose(c.final, [111.0, 12.0]) and np.allclose(c.weight, [0.5, 1.0])
    g = np.array([0.1, 0.7])
    soft = textguard(A, B, g, threshold=0.5, mode="soft")
    assert np.allclose(soft.final, [115.0, 11.4]) and list(soft.warn) == [True, False]
    block = textguard(A, B, g, threshold=0.5, mode="block")
    assert np.allclose(block.final, [116.0, 11.4]) and np.allclose(block.weight, [0.0, 0.7])
    # tau = 0 is the soft gate; tau > 1 is never_trust
    assert np.allclose(textguard(A, B, g, 0.0, "block").final, soft.final)
    assert np.allclose(textguard(A, B, g, 1.01, "block").final, A)


def test_nan_rules():
    A, B = np.array([1.0, np.nan, 3.0]), np.array([np.nan, 2.0, 4.0])
    d = mix(A, B, [0.5, 0.5, 0.5])
    assert d.final[0] == 1.0                     # B failed -> numbers only
    assert np.isnan(d.final[1])                  # A failed -> nothing to fall back to
    assert d.final[2] == 3.5
    assert clip(A, B, 1.0, 0.1).final[0] == 1.0


def test_multi_horizon_broadcasts_the_gate_per_sample():
    A, B = np.zeros((2, 3)), np.ones((2, 3))
    assert np.allclose(textguard(A, B, [0.25, 1.0]).final, [[0.25] * 3, [1.0] * 3])


# --- the wrapper, model-agnostic -----------------------------------------------------------

def _guard(model, **kw):
    judge = judge_from(lambda t: (-1, 0.9) if "fall" in t else (1, 0.9))
    return TextGuard(model, judge, sigma=SIGMA, window=4, **kw)


def test_worked_example_on_dummy_forecaster():
    """PHASES.md: A 116, B 106, trend up 1.0, text down 0.9 → g 0.1, final 115, warning."""
    guard = _guard(DummyForecaster())
    d = guard.decide(X, FAKE)
    assert d.A[0] == 116 and d.B[0] == 106
    assert d.g == pytest.approx(0.1) and d.warn
    assert d.final[0] == pytest.approx(115.0)
    real = guard.decide(X, REAL)                  # genuine text agrees -> never blocked
    assert real.g == 1.0 and not real.warn and real.final[0] == 120
    # error 11 -> 2; DSS +1.4 -> +0.5 (σ = 10, d = −1), genuine text not falsely blocked
    assert abs(117 - d.B[0]) == 11 and abs(117 - d.final[0]) == pytest.approx(2)
    before = metrics.dss([d.B[0]], [real.B[0]], [-1], SIGMA).value
    after = metrics.dss([d.final[0]], [real.final[0]], [-1], SIGMA).value
    assert before == pytest.approx(1.4) and after == pytest.approx(0.5)
    assert metrics.false_block_rate([real.g]).value == 0.0


def test_textguard_is_itself_a_forecaster_and_needs_no_inheritance():
    class ThirdParty:                              # duck-typed: not one of our classes
        horizon = 1

        def predict(self, numbers, text):
            return np.array([numbers[-1] + (-5.0 if text and "fall" in text else 0.0)])

    guard = _guard(ThirdParty())
    assert isinstance(guard, Forecaster)
    assert guard.predict(X, FAKE)[0] == pytest.approx(112 - 0.1 * 5)
    assert guard.predict(X, None)[0] == 112


def test_no_text_gives_numbers_only_and_block_mode_falls_back_to_A():
    guard = _guard(DummyForecaster(), threshold=0.5, mode="block")
    assert guard.predict(X, None)[0] == 116 and guard.decide(X, "  ").g == 1.0
    d = guard.decide(X, FAKE)
    assert d.warn and d.weight == 0.0 and d.final[0] == 116


def test_failed_predictions_follow_the_nan_rules():
    class Flaky:
        horizon = 1

        def __init__(self, a, b):
            self.a, self.b = a, b

        def predict(self, numbers, text):
            return np.array([self.a if text is None else self.b])

    assert _guard(Flaky(116.0, np.nan)).predict(X, FAKE)[0] == 116     # B failed -> A
    assert np.isnan(_guard(Flaky(np.nan, 106.0)).predict(X, FAKE)[0])  # A failed -> NaN


def test_level_branch_on_dummy_health_style_text():
    """Numbers well above a stated baseline, fake says below → blocked by the level branch."""
    x = [2.6, 2.9, 3.1, 3.2]
    fake = "ILI was 1.0%, below the national baseline of 2.1%."
    judge = fixed_judge(Claim("level", -1, 0.8, 2.1))
    guard = TextGuard(DummyForecaster(step=0.5), judge, sigma=1.0, window=4)
    d = guard.decide(x, fake)
    assert d.B[0] < d.A[0]                          # the dummy is fooled by "below baseline"
    assert d.g == pytest.approx(1 - 0.8 * 1.0) and d.warn


def test_constructor_validates():
    with pytest.raises(TypeError):
        TextGuard(object(), fixed_judge(), 1.0)
    with pytest.raises(ValueError):
        TextGuard(DummyForecaster(), fixed_judge(), 0.0)
    with pytest.raises(ValueError):
        TextGuard(DummyForecaster(), fixed_judge(), 1.0, mode="hard")
