"""R4 answer verifier (``models.r4_answers``): the backed-up Kaggle files pass, tampered copies fail."""

import shutil

import pandas as pd
import pytest

from src.models import r4_answers as RA
from src.models.export_qwen_inputs import OUT_DIR

pytestmark = pytest.mark.skipif(not RA.R4_INPUTS.exists(), reason="needs results/qwen/r4_heldout_inputs.csv")


def _copy(tmp_path, file):
    for suffix in ("_predictions.csv", "_run_meta.json"):
        shutil.copy(OUT_DIR / f"{file}{suffix}", tmp_path / f"{file}{suffix}")
    return tmp_path / f"{file}_predictions.csv"


def test_all_four_runs_verify():
    v = RA.verify_all()
    assert list(v.rows) == list(v.rows_expected) == [9145, 9145, 2913, 21312]
    assert set(v.input_sha) == {RA.INPUT_SHA256[:16]}


def test_forecaster_refuses_a_changed_answer(tmp_path):
    p = _copy(tmp_path, RA.PINS["m5"]["file"])
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    df.loc[0, "y_pred"] = "999.0"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="parser"):
        RA.load_forecaster("m5", out_dir=tmp_path)


def test_forecaster_refuses_a_missing_row(tmp_path):
    p = _copy(tmp_path, RA.PINS["qwen7b"]["file"])
    pd.read_csv(p, dtype=str, keep_default_na=False).iloc[1:].to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing or extra"):
        RA.load_forecaster("qwen7b", out_dir=tmp_path)


def test_judge_refuses_a_second_setup(tmp_path):
    p = _copy(tmp_path, RA.JUDGES["v1"]["file"])
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    df.loc[0, "setup"] = df.setup[0].replace("kaggle", "colab")
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="setup"):
        RA.load_judge("v1", out_dir=tmp_path)


def test_v1_held_out_direction_has_polarity_plus_one():
    j = RA.load_judge("v1")
    assert (j.direction == j.literal).all() and j.direction.notna().all()
