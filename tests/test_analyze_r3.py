"""Tests for the R3 analysis (§10f): the decision rules, family sizes, and the loader's refusals."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src import analyze_r3 as R
from src.models import r3_answers as ra


@pytest.mark.parametrize("diff,reject,want", [
    (-0.30, True, "reduction"), (-0.05, True, "reduction"), (-0.02, True, "significant but negligible"),
    (0.20, True, "increase"), (0.30, False, "no change shown"),
])
def test_change_verdict_needs_rejection_sign_and_sesoi(diff, reject, want):
    assert R.change_verdict(diff, reject) == want


@pytest.mark.parametrize("u,d0,dd,want", [
    ("fooled", "fooled", "no change shown", "evaded"),
    ("fooled", "fooled", "significant but negligible", "evaded"),
    ("fooled", "fooled", "reduction", "reduces but does not stop"),
    ("fooled", "bounded", "reduction", "stops"),
    ("fooled", "inconclusive", "reduction", "reduces; residual not shown to be below the SESOI"),
    ("fooled", "inconclusive", "no change shown", "not evaded, no reduction shown"),
    ("bounded", "fooled", "increase", "makes the attack worse"),      # reported whatever D0 says
    ("fooled", "fooled", "increase", "makes the attack worse"),
    ("inconclusive", "fooled", "no change shown", "nothing to defend against"),
])
def test_defence_reading_follows_10f(u, d0, dd, want):
    assert R.defence_reading(u, d0, dd) == want


def test_family_sizes_are_the_preregistered_ones():
    # U 15 + (D0 15 + DΔ 15) per defence: 3 defences for the LLMs, 2 for the FinBERT heads
    assert R.FAMILY_SIZE == {"R3-M5": 105, "R3-Q7": 105, "R3-M3F": 75, "R3-M4F": 75}
    assert len(R.ATTACKS) * len(R.CELLS) * (1 + 2 * 3) == 105
    assert len(R.ATTACKS) * len(R.CELLS) * (1 + 2 * 2) == 75


def test_holm_refuses_a_family_of_the_wrong_size():
    fam = pd.DataFrame(dict(block=["U"] * 10, estimate=0.1, ci_low=0, ci_high=0.2, p_raw=0.01,
                            n_nonzero=10))
    with pytest.raises(ValueError, match="pre-registered 105"):
        R.holm(fam, "M5")


def test_too_few_windows_is_labelled_and_stays_in_the_family():
    n = 105
    fam = pd.DataFrame(dict(block=["U"] * n, estimate=0.2, ci_low=0.1, ci_high=0.3,
                            p_raw=[1e-6] + [1.0] * (n - 1), n_nonzero=[10] + [3] * (n - 1)))
    out = R.holm(fam, "M5")
    assert len(out) == n and out.too_few_windows.sum() == n - 1
    assert out.verdict.iloc[0] == "fooled" and out.verdict.iloc[1].startswith("too few windows")


@pytest.mark.skipif(not ra.R3_INPUTS.exists(), reason="R3 inputs not exported")
def test_loader_refuses_a_file_for_another_input(tmp_path):
    for f in ("r3_m5_fp16_kaggle_predictions.csv", "r3_m5_fp16_kaggle_run_meta.json"):
        shutil.copy(ra.OUT_DIR / f, tmp_path / f)
    meta = tmp_path / "r3_m5_fp16_kaggle_run_meta.json"
    m = json.loads(meta.read_text())
    m["input_sha256"] = "0" * 64
    meta.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="different input file"):
        ra.load_forecaster("m5", out_dir=tmp_path)


@pytest.mark.skipif(not ra.R3_INPUTS.exists(), reason="R3 inputs not exported")
def test_loader_refuses_an_edited_answer(tmp_path):
    for f in ("r3_m5_fp16_kaggle_predictions.csv", "r3_m5_fp16_kaggle_run_meta.json"):
        shutil.copy(ra.OUT_DIR / f, tmp_path / f)
    p = tmp_path / "r3_m5_fp16_kaggle_predictions.csv"
    df = pd.read_csv(p, keep_default_na=False, dtype=str)
    df.loc[0, "y_pred"] = str(float(df.loc[0, "y_pred"]) + 1)
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="parser's reading"):
        ra.load_forecaster("m5", out_dir=tmp_path)
