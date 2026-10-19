"""R4 analysis (``analyze_r4``): frozen settings, anchor15, and the 79-test family per target."""

import numpy as np
import pytest

from src import analyze_r4 as A
from src.models import r4_answers as RA


def test_frozen_settings_are_the_pre_registered_ones():
    s = A.settings()
    assert (s["tg1_window"], s["tg1_tau"], s["clip_k"], s["period"]) == (6, 0.0, 0.5, 52)
    assert (s["tg2"].window, s["tg2"].conf_floor, s["tg2"].tau, s["tg2"].cap_k) == (12, 0.8, 0.0, 1.0)


def test_anchor15_projects_B_only_and_falls_back_to_A():
    A_, B = np.array([10.0, 10.0, 10.0, 10.0]), np.array([10.5, 14.0, np.nan, 10.0])
    d = A.anchor15(A_, B, np.full(4, 10.0))
    assert np.allclose(d.final, [10.5, 11.5, 10.0, 10.0])
    assert np.allclose(d.weight, [1.0, 1.5 / 4.0, 1.0, 1.0])


@pytest.mark.skipif(not RA.R4_INPUTS.exists(), reason="needs results/qwen/r4_heldout_inputs.csv")
def test_family_has_79_tests_and_the_pre_registered_cells():
    x = A.frame("M5")
    fam = A.holm(A.family_rows(x, A.finals(x, A.gates(x)), "M5"), "M5")
    assert len(fam) == A.FAMILY_SIZE == 79
    assert set(fam[fam.attack == "a3_magnitude"].cell) == {"Energy"}
    assert fam.groupby("block").size().to_dict() == {
        "U": 10, "T10": 10, "T1Δ": 10, "T20": 10, "T2Δ": 10, "T21": 10, "T2C": 10, "F21": 3, "M21": 3, "M2N": 3}
    assert (fam[fam.too_few_windows].p_raw == 1).all()
