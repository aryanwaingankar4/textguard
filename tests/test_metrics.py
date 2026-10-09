"""Tests for src/metrics.py — anchored on the PHASES.md worked example (DummyForecaster).

Worked example: past demand 100, 104, 108, 112 (σ = 10), true next value 117.
A (no text) = 116, B with fake "Demand will fall sharply." = 106, ŷ_real ("demand rising")
= 120, TextGuard final = 116 + 0.1·(106 − 116) = 115.
Errors 11 → 2; DSS +1.4 → +0.5; real forecast up, fake down (d = −1) → flipped; final 115
up → not flipped.
"""

import numpy as np
import pandas as pd
import pytest

from src import metrics as M
from src.models import _mmtsflib
from src.models.dummy import DummyForecaster

X = [100.0, 104.0, 108.0, 112.0]
SIGMA, Y_TRUE, Y_LAST = 10.0, 117.0, 112.0
FAKE, REAL = "Demand will fall sharply.", "demand rising"


@pytest.fixture(scope="module")
def ex():
    m = DummyForecaster()
    a, b_fake, y_real = (float(m.predict(X, t)[0]) for t in (None, FAKE, REAL))
    g = 1.0 - 0.9 * 1.0                    # gate from the worked example
    final = a + g * (b_fake - a)
    return dict(a=a, b_fake=b_fake, y_real=y_real, final=final, g=g)


# --- the worked example ------------------------------------------------------------------------

def test_dummy_reproduces_the_worked_example(ex):
    assert (ex["a"], ex["b_fake"], ex["y_real"]) == (116.0, 106.0, 120.0)
    assert ex["final"] == pytest.approx(115.0)


def test_errors_11_to_2(ex):
    assert M.mae([Y_TRUE], [ex["b_fake"]], 1.0).value == pytest.approx(11.0)   # raw units
    assert M.mae([Y_TRUE], [ex["final"]], 1.0).value == pytest.approx(2.0)
    assert M.mae([Y_TRUE], [ex["b_fake"]], SIGMA).value == pytest.approx(1.1)  # normalised
    assert M.mse([Y_TRUE], [ex["b_fake"]], SIGMA).value == pytest.approx(1.21)
    assert M.mse([Y_TRUE], [ex["final"]], SIGMA).value == pytest.approx(0.04)


def test_dss_1_4_to_0_5(ex):
    assert M.dss([ex["b_fake"]], [ex["y_real"]], [-1], SIGMA).value == pytest.approx(1.4)
    assert M.dss([ex["final"]], [ex["y_real"]], [-1], SIGMA).value == pytest.approx(0.5)


def test_flip_rate_flipped_then_not(ex):
    assert M.flip_rate([ex["b_fake"]], [ex["y_real"]], [Y_LAST], [-1]).value == 1.0
    assert M.flip_rate([ex["final"]], [ex["y_real"]], [Y_LAST], [-1]).value == 0.0


def test_usage_gap_and_sensitivity_on_the_example(ex):
    # MSE(None) = (116−117)²/100 = 0.01, MSE(real) = (120−117)²/100 = 0.09
    assert M.text_usage_gap([Y_TRUE], [ex["a"]], [ex["y_real"]], SIGMA).value == pytest.approx(-0.08)
    assert M.text_sensitivity([ex["y_real"]], [ex["a"]], SIGMA).value == pytest.approx(0.4)


def test_genuine_text_is_not_blocked_in_the_example(ex):
    # real text agrees with the trend -> g = 1; the fake gets g = 0.1 but is not genuine
    assert M.false_block_rate([1.0]).value == 0.0
    assert M.false_block_rate([1.0, ex["g"]], mask=[True, False]).value == 0.0
    assert M.false_block_rate([ex["g"]]).value == 1.0


# --- definitions -------------------------------------------------------------------------------

def test_dss_only_over_attacked_windows_and_reports_coverage():
    m = M.dss([106, 50, 7], [120, 50, 3], [-1, 0, +1], 10.0)
    assert m.value == pytest.approx((1.4 + 0.4) / 2)
    assert (m.n, m.n_not_attacked) == (2, 1) and m.coverage == pytest.approx(2 / 3)
    assert np.isnan(m.terms[1])


def test_dss_for_exaggerated_is_positive_when_pushed_further():
    # real claim up (d = +1); exaggerated text pushes the forecast higher still
    assert M.dss([130.0], [120.0], [+1], 10.0).value == pytest.approx(1.0)


@pytest.mark.parametrize("real, fake, d, flipped", [
    (120, 106, -1, 1),   # up -> down: toward the lie
    (106, 100, -1, 0),   # already down: cannot flip
    (112, 106, -1, 1),   # flat (= y_last) -> down: counts
    (120, 112, -1, 0),   # fake flat: never counts
    (100, 120, +1, 1),   # down -> up, lie says up
    (120, 125, -1, 0),   # moved, but not across
])
def test_flip_rate_cases_including_zero_change(real, fake, d, flipped):
    assert M.flip_rate([fake], [real], [112.0], [d]).value == flipped


def test_nan_pairing_drops_the_window_from_both_sides():
    y = [10.0, 10.0, 10.0]
    real = [11.0, np.nan, 12.0]
    none = [10.0, 10.0, np.nan]
    gap = M.text_usage_gap(y, none, real, 1.0)
    assert gap.n == 1 and gap.n_nan == 2 and gap.value == pytest.approx(0.0 - 1.0)
    both = M.paired_mse(y, {"real": real, "none": none}, 1.0)
    assert both["real"].n == both["none"].n == 1 and both["real"].n_nan == 2
    d = M.dss([np.nan, 9.0, 9.0], [10.0, 10.0, np.nan], [-1, -1, -1], 1.0)
    assert d.n == 1 and d.n_nan == 2 and d.value == pytest.approx(1.0)


def test_every_metric_accepts_a_window_mask():
    y, p, s = [1.0, 2.0, 100.0], [1.0, 1.0, 0.0], 1.0
    keep = [True, True, False]                      # e.g. exclude 2020
    assert M.mse(y, p, s, keep).value == pytest.approx(0.5) and M.mse(y, p, s, keep).n_masked == 1
    assert M.mae(y, p, s, keep).value == pytest.approx(0.5)
    assert M.text_usage_gap(y, p, y, s, keep).value == pytest.approx(0.5)
    assert M.text_sensitivity(y, p, s, keep).value == pytest.approx(0.5)
    assert M.dss([0, 0, 0], [1, 1, 50], [-1, -1, -1], s, keep).value == pytest.approx(1.0)
    assert M.flip_rate([0, 0, 0], [5, 5, 5], [1, 1, 1], [-1, -1, -1], keep).n == 2
    assert M.false_block_rate([0.1, 0.9, 0.1], keep).value == pytest.approx(0.5)
    with pytest.raises(ValueError, match="mask"):
        M.mse(y, p, s, [True])


def test_per_window_sigma_for_pooled_tables():
    m = M.mse([10.0, 100.0], [11.0, 110.0], [1.0, 10.0])
    assert m.value == pytest.approx(1.0)
    with pytest.raises(ValueError):
        M.mse([1.0], [1.0], 0.0)


def test_direction_must_be_recorded_metadata():
    with pytest.raises(ValueError, match="claimed_direction"):
        M.dss([1.0], [2.0], [0.7], 1.0)            # a classifier score is not allowed


# --- Models 1–2 never see text: sanity -----------------------------------------------------------

def test_text_blind_model_has_zero_dss_and_sensitivity():
    m = DummyForecaster(sensitivity=0.0)
    windows = [X, [5.0, 4.0, 3.0, 2.0], [1.0, 1.0, 1.0, 1.0]]
    real = [m.predict(x, REAL)[0] for x in windows]
    none = [m.predict(x, None)[0] for x in windows]
    fake = [m.predict(x, FAKE)[0] for x in windows]
    assert M.is_text_blind(real, none, fake)
    assert M.dss(fake, real, [-1, -1, -1], 1.0).value == 0.0
    assert M.text_sensitivity(real, none, 1.0).value == 0.0
    assert M.flip_rate(fake, real, [x[-1] for x in windows], [-1, -1, -1]).value == 0.0


@pytest.mark.skipif(not _mmtsflib.available(), reason="MM-TSFlib not cloned")
def test_models_1_and_2_are_exactly_text_blind():
    from src.data.splits import chronological_split
    from src.data.windows import split_windows
    from src.models.dlinear import DLinearForecaster
    from src.models.patchtst import PatchTSTForecaster
    t = np.arange(100)
    start = pd.date_range("2000-01-01", periods=100, freq="MS")
    df = pd.DataFrame({"start_date": start, "end_date": start + pd.offsets.MonthEnd(0),
                       "value": 50 + 5 * np.sin(t / 3) + 0.1 * t, "has_text": True})
    s = chronological_split(df)
    w = split_windows(s, 8, 1)
    for cls in (DLinearForecaster, PatchTSTForecaster):
        m = cls(8, 1, epochs=2).fit(w["train"], w["val"], s.train["value"])
        Xs = w["test"].X
        real = [m.predict(x, "prices rose sharply")[0] for x in Xs]
        fake = [m.predict(x, "prices fell sharply")[0] for x in Xs]
        none = [m.predict(x, None)[0] for x in Xs]
        assert M.is_text_blind(real, none, fake)
        assert M.dss(fake, real, -np.ones(len(Xs)), 1.0).value == 0.0
        assert M.text_sensitivity(real, none, 1.0).value == 0.0


# --- seeds and failures --------------------------------------------------------------------------

def test_seed_mean_averages_and_keeps_failures():
    p = pd.DataFrame({"model": "m", "domain": "d", "sample_id": ["a", "a", "b", "b"],
                      "condition": "real", "seed": [1, 2, 1, 2], "y_pred": [1.0, 3.0, 5.0, np.nan]})
    out = M.seed_mean(p).set_index("sample_id")
    assert out.loc["a", "y_pred"] == 2.0 and np.isnan(out.loc["b", "y_pred"])
    assert (out.n_seeds == 2).all()


def test_failure_rates_break_out_parse_and_format():
    p = pd.DataFrame({"model": "Qwen", "condition": ["real"] * 4,
                      "y_pred": [1.0, np.nan, np.nan, 2.0],
                      "parse_ok": [True, False, True, True], "scale_ok": [True, True, False, True]})
    r = M.failure_rates(p).iloc[0]
    assert (r.n, r.failures, r.parse_failures, r.format_failures) == (4, 2, 1, 1)
    assert r.rate == 0.5


# --- statistics ----------------------------------------------------------------------------------

def test_bootstrap_ci_is_reproducible_and_contains_the_mean():
    rng = np.random.default_rng(1)
    t = rng.normal(1.0, 1.0, 200)
    lo, hi = M.bootstrap_ci(t, n_boot=2000, seed=0)
    assert lo < t.mean() < hi and (lo, hi) == M.bootstrap_ci(t, n_boot=2000, seed=0)
    assert M.bootstrap_ci(np.full(10, 3.0), n_boot=500) == (3.0, 3.0)
    assert all(np.isnan(M.bootstrap_ci([np.nan, np.nan])))


def test_stratified_bootstrap_keeps_domain_shares():
    t = np.r_[np.zeros(50), np.ones(50)]
    strata = np.r_[["A"] * 50, ["B"] * 50]
    assert M.bootstrap_ci(t, n_boot=500, strata=strata) == (0.5, 0.5)   # shares fixed


def test_paired_difference_and_wilcoxon():
    before = M.dss([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0] * 8, [-1] * 8, 1.0)
    after = M.dss([0.9, 0.8, 0.95, 0.7, 0.85, 0.9, 0.6, 0.75], [1.0] * 8, [-1] * 8, 1.0)
    diff = M.paired_difference(before, after)
    assert np.all(diff > 0)
    res = M.wilcoxon(diff)
    assert res["n"] == 8 and res["p"] < 0.01
    assert M.wilcoxon(np.zeros(5))["p"] == 1.0                     # Models 1-2: no difference
    assert M.wilcoxon([1.0, np.nan, 2.0], [0.0, 0.0, np.nan])["n"] == 1


def test_mcnemar_flip_counts_discordant_windows():
    # y_last = 0, lie d = +1. Windows: real down -> fake up (toward), real up -> fake down
    # (away), real flat -> fake up (toward), real up -> fake up (concordant), not attacked,
    # fake NaN (dropped by NaN pairing)
    real = [-1.0, 1.0, 0.0, 1.0, -1.0, -1.0]
    fake = [1.0, -1.0, 2.0, 3.0, 1.0, np.nan]
    d = [1, 1, 1, 1, 0, 1]
    res = M.mcnemar_flip(fake, real, np.zeros(6), d)
    assert (res["toward"], res["away"], res["n"]) == (2, 1, 4)
    assert res["toward"] == M.flip_rate(fake, real, np.zeros(6), d).terms[:4].sum()
    assert res["p"] == pytest.approx(1.0)                          # 2 vs 1: no evidence
    assert M.mcnemar_flip([1.0] * 12, [-1.0] * 12, np.zeros(12), [1] * 12)["p"] < 0.001
    assert M.mcnemar_flip([1.0], [1.0], [0.0], [1])["p"] == 1.0    # no discordant window


def test_holm_adjustment():
    np.testing.assert_allclose(M.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    np.testing.assert_allclose(M.holm([0.5, 0.9]), [1.0, 1.0])
    assert M.holm([0.2]).tolist() == [0.2]


def test_grouped_reports_pooled_headline_and_domains():
    m = M.mse([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 2.0, np.nan], 1.0)
    out = M.grouped(m, ["E", "E", "T", "T"], n_boot=200).set_index("group")
    assert out.loc["E", "value"] == 1.0 and out.loc["T", "n"] == 1
    assert out.loc["pooled", "value"] == pytest.approx(2.0) and out.loc["pooled", "n"] == 3
