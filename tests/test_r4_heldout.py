"""R4 held-out side (pre-registration §10g): judge subjects, history, export guards and the four
Kaggle notebooks run with fake models. No GPU, no LLM, no held-out model output."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.attacks import heldout_rules as hr
from src.models import export_r4_inputs as ex
from src.models import qwen_prompt as qp
from src.textguard import heldout_questions as hq
from src.textguard import judge_prompt as jp1
from src.textguard import judge_prompt_v2 as jp2
from src.textguard import v2 as V

ROOT = Path(__file__).resolve().parents[1]
NB = {"m5": "r4_m5_kaggle.ipynb", "qwen7b": "r4_qwen7b_kaggle.ipynb",
      "judge_v1": "r4_judge_v1_kaggle.ipynb", "judge_v2": "r4_judge_v2_kaggle.ipynb"}


def _cells(name):
    nb = json.loads((ROOT / "notebooks" / NB[name]).read_text(encoding="utf-8"))
    return [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]


def test_subjects_match_the_engine_and_the_approved_wording():
    assert hq.COMPANY_NAMES == hr.COMPANY
    assert hq.HELDOUT_QUESTIONS["Energy"] == [("change", "the U.S. national average retail price of regular "
                                                         "gasoline compared with the previous week")]
    assert hq.HELDOUT_SUBJECTS["FNSPID:PFE"] == "the share price of Pfizer stock"
    assert hq.domain_key("FNSPID", "JNJ") == "FNSPID:JNJ" and hq.domain_key("Energy", "Energy") == "Energy"


def test_register_leaves_original_prompts_unchanged():
    before = jp1.judge_sha(jp1.build_judge_prompt("Economy", "change", "The deficit rose."))
    before2 = jp2.judge_sha(jp2.build_judge_prompt("Traffic", "change", "Travel rose."))
    hq.register()
    hq.register()                                                  # idempotent
    assert jp1.judge_sha(jp1.build_judge_prompt("Economy", "change", "The deficit rose.")) == before
    assert jp2.judge_sha(jp2.build_judge_prompt("Traffic", "change", "Travel rose.")) == before2
    assert "the share price of Merck stock" in jp1.build_judge_prompt("FNSPID:MRK", "change", "Merck Shares Up 2%")


def test_frozen_v2_settings_are_the_pre_registered_ones():
    fz = json.loads((ROOT / "results" / "summary_r4_v2_frozen.json").read_text(encoding="utf-8"))
    assert (fz["fraction"], fz["conf_floor"], fz["tau"], fz["cap_k"]) == (0.5, 0.8, 0.0, 1.0)
    assert fz["window"]["held-out weekly (Energy, FNSPID; lookback 24)"] == 12 and fz["budget_met"]


def test_export_refuses_texts_other_than_the_pre_registered(tmp_path, monkeypatch):
    monkeypatch.setitem(ex.TEXT_SHA, "real", "0000000000000000")
    with pytest.raises(ValueError, match="pre-registered"):
        ex.load_texts()


# --- notebooks ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(NB))
def test_notebooks_embed_unchanged_modules_and_the_input_sha(name):
    code = [s for s, t in _cells(name) if t == "code"]
    mods = {"m5": ["models/qwen_prompt.py", "models/r2_prompts.py"],
            "qwen7b": ["models/qwen_prompt.py", "models/r2_prompts.py"],
            "judge_v1": ["models/qwen_prompt.py", "textguard/judge_prompt.py"],
            "judge_v2": ["models/qwen_prompt.py", "textguard/judge_prompt_v2.py"]}[name]
    for m in mods:
        src = (ROOT / "src" / m).read_text(encoding="utf-8").strip("\n")
        assert any(c.endswith(src) for c in code), f"stale: rebuild ({m})"
    if name.startswith("judge"):
        hqsrc = (ROOT / "src" / "textguard" / "heldout_questions.py").read_text(encoding="utf-8").rstrip("\n")
        assert any(hqsrc in c for c in code)
    cfg = next(c for c in code if c.startswith("# 0."))
    if ex.R4_INPUTS.exists():
        assert f'INPUT_SHA256 = "{ex.sha256()}"' in cfg, "inputs changed: re-export and rebuild"
    assert "r4_heldout_inputs.csv" in next(c for c in code if c.startswith("# 3."))


def _write_inputs(tmp_path):
    hq.register()
    f = []
    for key, unit in (("Energy", "US dollars per gallon"), ("FNSPID:PFE", "US dollars")):
        for i in range(3):
            for cond in ("real", "none", "flipped", "a2_forward"):
                text = "" if cond == "none" else f"{key} rose {i}." + (" FAKE" if cond != "real" else "")
                f.append(dict(row_type="forecast", domain=key, split="test", sample_id=f"2019-0{i + 1}-01",
                              condition=cond, freq="weekly", description="series", unit=unit,
                              lookback=json.dumps([1.0, 2.0, 3.0]), text=text, y_true=3.5, train_std=1.0,
                              attacked=cond != "real" and cond != "none"))
    texts = {"Energy": "Gasoline increased. It is expected to fall next week.", "FNSPID:PFE": "Pfizer Shares Up 3%\nPfizer increased"}
    j1 = [dict(row_type="judge", domain=k, question="change", text=t,
               prompt_sha=jp1.judge_sha(jp1.build_judge_prompt(k, "change", t))) for k, t in texts.items()]
    from src.textguard.units import units
    j2 = [dict(row_type="judge_v2", domain=k, question=q, text=u,
               prompt_sha=jp2.judge_sha(jp2.build_judge_prompt(k, q, u)))
          for k, t in texts.items() for u in units(t) for q in V.questions_for(u)]
    probes = [dict(row_type="probe", domain="Economy", split="val", sample_id=f"2015-0{i}-01", condition="none",
                   freq="monthly", description="series", unit="million US dollars", lookback="[1.0, 2.0]", text="",
                   expected_raw="3") for i in range(1, 5)]
    df = pd.DataFrame(f + j1 + j2 + probes)
    for c in ex.COLUMNS:
        if c not in df:
            df[c] = ""
    (tmp_path / "input" / "ds").mkdir(parents=True)
    df[ex.COLUMNS].fillna("").to_csv(tmp_path / "input" / "ds" / "r4_heldout_inputs.csv", index=False)
    return df


def test_forecaster_notebook_answers_every_row_with_m5s_prompt_only(tmp_path):
    df = _write_inputs(tmp_path)
    code = [s for s, t in _cells("m5") if t == "code"]
    pick = lambda key: next(c for c in code if c.startswith(key) or key in c[:80])
    ns = dict(RUN="m5", MODEL_ID="Qwen/x", REVISION="r", QUANT="fp16", RUN_VARIANTS=["p5"], PROBE=True,
              SMOKE=False, MAX_NEW_TOKENS=24, SPLITS=["test"], INPUT_SHA256="x", PLATFORM="kaggle",
              LOCAL_TEST=True, generate=lambda system, user: "3", SETUP="T4 | m5 | kaggle")
    exec(pick("# 3. Paths and inputs").replace("/kaggle", tmp_path.as_posix()), ns)
    ns["meta"] = dict(run="m5", setup=ns["SETUP"])
    exec(pick("# Generated verbatim from src/models/qwen_prompt.py"), ns)
    exec(pick("# Generated verbatim from src/models/r2_prompts.py"), ns)
    for c in ("6", "7", "8"):
        exec(pick(f"# {c}."), ns)
    res = pd.read_csv(tmp_path / "working" / "r4_m5_fp16_kaggle_predictions.csv", keep_default_na=False)
    assert len(res) == (df.row_type == "forecast").sum() and set(res.variant) == {"p5"}
    assert set(res.domain) == {"Energy", "FNSPID:PFE"} and not res.duplicated(["domain", "sample_id", "condition"]).any()
    assert ns["meta"]["probe_ok"]


def _fake_score(prompt, answers):
    s = prompt.split('"""')[1]
    if "fall" in s and "predict" in prompt.split("Question:")[1]:
        return [-3.0, -0.1, -3.0]
    return [-0.1, -3.0, -3.0] if ("increased" in s or "Up" in s) else [-4.0, -4.0, -0.05]


@pytest.mark.parametrize("which", ["v1", "v2"])
def test_judge_notebooks_answer_their_own_rows(tmp_path, which):
    df = _write_inputs(tmp_path)
    code = [s for s, t in _cells(f"judge_{which}") if t == "code"]
    pick = lambda key: next(c for c in code if key in c)
    ns = dict(generate=lambda p, **kw: "3", score=_fake_score, SETUP=f"T4 | {which} | kaggle", P5_SETUP="colab",
              meta={"setup": f"T4 | {which} | kaggle"}, INPUT_SHA256="x", LOCAL_TEST=True, PLATFORM="kaggle")
    exec(pick("# 3. Paths and inputs").replace("/kaggle", tmp_path.as_posix()), ns)
    exec(pick("# Generated verbatim from src/models/qwen_prompt.py"), ns)
    exec(pick("# Generated verbatim from src/textguard/judge_prompt" + ("_v2.py" if which == "v2" else ".py")), ns)
    exec(pick("# Generated verbatim from src/textguard/heldout_questions.py"), ns)
    exec(pick("# 6. Reproducibility probe").replace("len(probes) == 40", "len(probes) == 4"), ns)
    for key in ("# 7. Main loop", "# 8. Completeness"):
        exec(pick(key), ns)
    res = pd.read_csv(tmp_path / "working" / f"r4_judge_{which}_kaggle_predictions.csv", keep_default_na=False)
    want = df[df.row_type == ("judge_v2" if which == "v2" else "judge")]
    assert set(res.prompt_sha) == set(want.prompt_sha) and len(res) == len(want)
