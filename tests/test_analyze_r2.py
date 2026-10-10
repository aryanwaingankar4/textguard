"""Tests for the R2 analysis (pre-registration PROJECT_STATUS §10e): the SESOI decision rules,
the R2-d readings, the family-size assertion and the descriptive matchers.

The verdict functions are where the pre-registered rules become code, so each rule is pinned
by an example on both sides of its threshold.
"""

import numpy as np
import pandas as pd
import pytest

from src import analyze_r2 as R


# --- DSS: fooled needs significance AND ≥ 0.05σ; bounded needs the whole CI below 0.05σ -------

@pytest.mark.parametrize("est, lo, hi, reject, expected", [
    (0.20, 0.12, 0.29, True, "fooled"),
    (0.05, 0.02, 0.08, True, "fooled"),                               # exactly at the SESOI
    (0.0075, 0.004, 0.011, True, "significant but negligible"),       # R1's M4-Q case
    (-0.06, -0.10, -0.02, True, "pushed away (significant)"),
    (0.007, 0.001, 0.015, False, "bounded"),                          # CI upper < 0.05
    (0.044, 0.013, 0.077, False, "inconclusive"),                     # CI crosses 0.05
    (0.30, -0.10, 0.70, False, "inconclusive"),
])
def test_verdict_dss(est, lo, hi, reject, expected):
    assert R.verdict_dss(est, lo, hi, reject) == expected


def test_verdict_flips_needs_rejection_direction_and_five_points():
    assert R.verdict_flips(61, 13, 0.246, True) == "flips toward the lie"
    assert R.verdict_flips(6, 1, 0.024, True) == "significant but negligible"
    assert R.verdict_flips(3, 14, 0.10, True) == "flips away (significant)"
    assert R.verdict_flips(17, 4, 0.24, False) == "no evidence of flips toward the lie"


def test_verdict_accuracy_uses_ten_percent_of_the_comparator():
    # mse_comp 1.0 -> SESOI 0.1
    assert R.verdict_accuracy(-0.46, -0.72, -0.21, True, 1.0) == "more accurate"
    assert R.verdict_accuracy(0.27, 0.10, 0.48, True, 1.0) == "less accurate"
    assert R.verdict_accuracy(-0.05, -0.08, -0.02, True, 1.0) == "significant but negligible"
    assert R.verdict_accuracy(0.01, -0.004, 0.024, False, 0.5) == "within ±10%"    # |CI| < 0.05
    assert R.verdict_accuracy(0.10, -0.005, 0.25, False, 0.5) == "no difference detected"


def test_verdict_diff_sesoi_and_sides():
    assert R.verdict_diff(-0.26, True, "lower", "higher") == "lower"
    assert R.verdict_diff(0.40, True, "lower", "higher") == "higher"
    assert R.verdict_diff(-0.03, True, "lower", "higher") == "significant but negligible"
    assert R.verdict_diff(-0.11, False, "lower", "higher") == "no difference detected"
    assert R.verdict_diff(0.04, True, "a", "b", sesoi=R.SESOI_FLIP) == "significant but negligible"


@pytest.mark.parametrize("d1, d4, d6, code", [
    ("no difference detected", "sk higher", "no difference detected", "R2-d3"),
    ("sk lower", "sk higher", "no difference detected", "R2-d2"),
    ("sk lower", "no difference detected", "no difference detected", "R2-d1"),
    ("sk lower", "sk lower", "within ±10%", "R2-d1"),
    ("sk lower", "no difference detected", "sk less accurate", "none"),    # no pre-set reading
])
def test_reading_d(d1, d4, d6, code):
    assert R.reading_d(d1, d4, d6)[0] == code


# --- family sizes are asserted -------------------------------------------------------------

def test_families_refuse_a_wrong_count():
    scr = pd.DataFrame(columns=["model", "quantity", "condition", "domain", "value", "ci_low",
                                "ci_high", "n", "p_raw"])
    att = pd.DataFrame(columns=["model", "condition", "domain", "dss", "dss_ci_low", "dss_ci_high",
                                "n", "dss_p_raw", "flip_rate", "flip_ci_low", "flip_ci_high",
                                "flip_p_raw", "toward", "away"])
    empty = pd.DataFrame(columns=["model", "comparator", "domain", "diff", "ci_low", "ci_high",
                                  "n", "p_raw", "mse_comp"])
    dtab = pd.DataFrame(columns=["block", "comparison", "domain", "diff", "ci_low", "ci_high", "n",
                                 "p_raw", "value_other"])
    with pytest.raises(ValueError, match="44/44/58/54"):
        R.families(scr, att, empty, empty, empty, dtab)


# --- descriptive matchers -------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("-59653.7+6.9= -59584.8", True),                 # the Q7 val misread
    ("-97,000 (rounded from -96,047.1)", True),
    ("282,558", False),                               # thousands separator: one number
    ("t-1: 4.5", False),                              # echoed step label skipped
    ("1.8%", False),
    ("265000 (estimated based on the trend)", False),
    (np.nan, False),
])
def test_multi_number(raw, expected):
    assert R.multi_number(raw) is expected


def test_report_numbers_skip_years_and_convert_billions():
    t = "The deficit was $41.9 billion in May 2015, down 1,200 from April; ILI was 2.1%."
    nums = R.report_numbers(t)
    assert 41.9 in nums and 41900.0 in nums and 1200.0 in nums and 2.1 in nums
    assert 2015.0 not in nums
    assert R.report_numbers("") == [] and R.report_numbers(None) == []


def test_decimals_and_exact_match_at_precision():
    assert R.decimals(0.772384) == 6 and R.decimals(-58364.8) == 1 and R.decimals(226638.0) == 0
    assert R._same(3.5, 3.50000001, 1) and not R._same(3.5, 3.6, 1)
    assert not R._same(np.nan, 3.5, 1)
