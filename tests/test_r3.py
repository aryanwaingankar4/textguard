"""Tests for R3 (pre-registration PROJECT_STATUS §10f): the three adaptive attacks, the single
input export and the three Kaggle notebooks.

No GPU and no LLM: the notebooks' own cells are executed with a fake model, so the test-first
order, resume, sk/none sharing, probe and smoke-test rules that will run on Kaggle are tested here.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.attacks import adaptive as ad
from src.attacks.rules import edit_text, headline
from src.models import qwen_prompt as qp
from src.models import r2_prompts as rp
from src.models.r2_forecasters import RUN_SPECS
from src.textguard import export_judge_inputs as ej

ROOT = Path(__file__).resolve().parents[1]
NB = {"m5": "r3_m5_kaggle.ipynb", "qwen7b": "r3_qwen7b_kaggle.ipynb", "judge": "r3_judge_kaggle.ipynb"}

ECON = ("The U.S. international trade deficit increased by $2.9 billion to $43.8 billion in June "
        "2015, with exports decreasing by $0.1 billion to $188.6 billion.")
SG = "The unemployment rate declined by 0.2 percentage point to 5.5 percent in February."
SG_COVID = ("The unemployment rate increased by 10.3 percentage points to 14.7 percent in April "
            "2020.")
HUS = "The proportion of outpatient visits for ILI was 2.6%, which is above the national baseline of 2.2%."


# --- A2: forward-looking lie -------------------------------------------------------------------

@pytest.mark.parametrize("domain", ["Economy", "Traffic", "SocialGood", "Health_US"])
@pytest.mark.parametrize("d", [1, -1])
def test_forward_sentence_round_trips_to_its_series_direction(domain, d):
    s = ad.forward_sentence(domain, d, "the trade deficit was $40 billion")
    assert ad.read_forward(s) == d
    assert s.endswith(f"next {'week' if domain == 'Health_US' else 'month'}.")


def test_economy_deficit_rise_means_balance_down():
    assert ad.forward_sentence("Economy", -1, "deficit") == "The trade deficit is expected to rise next month."
    assert ad.forward_sentence("Economy", 1, "the surplus was") == "The trade surplus is expected to rise next month."


def test_a2_keeps_the_real_report_as_prefix_and_reverses_the_claim():
    res = ad.a2_edit(ECON, "Economy")
    assert res.attacked and res.text.startswith(ECON) and res.text != ECON
    assert res.real_direction == -1 and res.claimed_direction == 1        # deficit up = balance down
    assert res.text.endswith("The trade deficit is expected to fall next month.")
    assert headline(res.text, "Economy").sentence == headline(ECON, "Economy").sentence
    assert not ad.a2_edit("The unemployment rate was unchanged at 5.0 percent.", "SocialGood").attacked
    assert not ad.a2_edit(None, "SocialGood").attacked


def test_append_claim_separator():
    assert ad.append_claim("It rose.  ", "X.") == "It rose. X."
    assert ad.append_claim("Facts**", "X.") == "Facts**. X."


# --- A1: selective flip ------------------------------------------------------------------------

def test_a1_selection_follows_the_sign_of_textguards_slope_only():
    rising = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert ad.a1_agrees(rising, "text", 1, "Economy", 3)
    assert not ad.a1_agrees(rising, "text", -1, "Economy", 3)
    assert ad.a1_agrees(np.ones(5), "text", -1, "Economy", 3)                # flat: no evidence


def test_a1_health_us_also_needs_the_level_branch_to_agree():
    x = np.array([3.0, 2.8, 2.6, 2.4, 2.2, 2.0, 1.9, 1.8])                  # falling, below 2.2
    below = "ILI was 1.8%, below the national baseline of 2.2%."
    above = "ILI was 2.6%, above the national baseline of 1.5%."            # level says y_last > B
    assert ad.a1_agrees(x, below, -1, "Health_US", 6)
    assert not ad.a1_agrees(x, above, -1, "Health_US", 6)


# --- A3: magnitude lie ---------------------------------------------------------------------------

def test_engine_default_is_p3s_factor():
    for t, d in ((ECON, "Economy"), (SG, "SocialGood"), (HUS, "Health_US")):
        assert edit_text(t, d, "exaggerate").text == edit_text(t, d, "exaggerate", factor=3.0).text


def test_a3_scales_changes_tenfold_and_keeps_direction():
    res = ad.a3_edit(ECON, "Economy")
    assert res.attacked and "$29.0 billion" in res.text and res.claimed_direction == res.real_direction


def test_a3_plausibility_filter_rejects_impossible_values():
    res = ad.a3_edit(SG_COVID, "SocialGood")
    assert not res.attacked and res.reason.startswith("implausible at x10: P1") and res.text == SG_COVID
    hot = HUS.replace("2.6%", "3.5%")                                      # 2.2·(3.5/2.2)^10 ≈ 229%
    assert not ad.a3_edit(hot, "Health_US").attacked
    assert ad.a3_edit(HUS, "Health_US").attacked                           # 2.6% -> 11.7%: in range
    assert ad.implied_starts("rose by 2.0 percentage points to 3.6 percent") == [(1.6, "pp", "percent")]


# --- export + notebooks -------------------------------------------------------------------------

def _cells(name):
    nb = json.loads((ROOT / "notebooks" / NB[name]).read_text(encoding="utf-8"))
    return [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]


@pytest.mark.parametrize("name", ["m5", "qwen7b"])
def test_forecaster_notebooks_embed_modules_pins_and_the_input_sha(name):
    cells = [s for s, t in _cells(name) if t == "code"]
    for f in ("qwen_prompt.py", "r2_prompts.py"):
        src = (ROOT / "src" / "models" / f).read_text(encoding="utf-8").strip("\n")
        assert any(s.endswith(src) for s in cells), "stale: run python notebooks/build_r3_notebooks.py"
    cfg = cells[0]
    spec = RUN_SPECS["m5prompts" if name == "m5" else "qwen7b"]
    assert f'REVISION = "{spec["revision"]}"' in cfg and f'MODEL_ID = "{spec["model_id"]}"' in cfg
    assert 'RUN_VARIANTS = ["p5", "sk"]' in cfg and 'CUDA_VISIBLE_DEVICES"] = "0"' in cfg
    assert ('QUANT = "fp16"' in cfg and "PROBE = True" in cfg) if name == "m5" else \
        ('QUANT = "nf4"' in cfg and "SMOKE = True" in cfg)
    _assert_sha(cfg)


def test_judge_notebook_is_p6s_with_platform_edits_only():
    cells = [s for s, t in _cells("judge") if t == "code"]
    for f in (ROOT / "src" / "models" / "qwen_prompt.py", ROOT / "src" / "textguard" / "judge_prompt.py"):
        assert any(s.endswith(f.read_text(encoding="utf-8").strip("\n")) for s in cells)
    load = next(s for s in cells if "def score" in s)
    assert 'REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"' in load and "| {PLATFORM}" in load
    p6 = json.loads((ROOT / "notebooks" / "p6_direction_judge.ipynb").read_text(encoding="utf-8"))
    p6_main = next("".join(c["source"]) for c in p6["cells"] if "".join(c["source"]).startswith("# 7."))
    r3_main = next(s for s in cells if s.startswith("# 7."))
    assert r3_main.replace('print(f"===== MAIN LOOP | {SETUP} =====", flush=True)\n', "") \
        .replace('questions to answer", flush=True)', 'questions to answer")') == p6_main
    _assert_sha(cells[0])


def _assert_sha(cfg):
    from src.models.export_r3_inputs import R3_INPUTS, sha256
    if not R3_INPUTS.exists():
        pytest.skip("r3_inputs.csv not exported here")
    assert f'INPUT_SHA256 = "{sha256()}"' in cfg, "inputs changed: re-run the export and the builder"


# --- notebook logic with a fake model ------------------------------------------------------------

CONDS = ("real", "none", "flipped", "exaggerated", "a1_selective", "a2_forward", "a3_magnitude")


def fake(system, user):
    if rp.CAUTION in user:
        return "100.5"
    if "FAKE" in user:
        return "99"
    return "102" if "eport" in user else "101"


def _inputs(tmp_path, n_probe=4):
    rows = []
    for d, freq, unit in (("Economy", "monthly", "million US dollars"), ("Health_US", "weekly", "percent")):
        for split in ("test", "val"):
            for i in range(3):
                x = [100.0 + i + k + 0.5 * (split == "val") for k in range(4)]
                real = "" if i == 0 else f"{d} rose {i}."
                for cond in CONDS:
                    text = "" if cond == "none" else (real + (" FAKE" if cond not in ("real",) and real else ""))
                    rows.append(dict(row_type="forecast", domain=d, split=split, sample_id=f"20{10 + i}-01-01",
                                     condition=cond, freq=freq, description=f"{d} series", unit=unit,
                                     lookback=json.dumps(x), text=text, y_true=104.0 + i,
                                     train_mean=100.0, train_std=2.0,
                                     attacked=bool(real) and cond not in ("real", "none")))
    f = pd.DataFrame(rows)
    pr = f[(f.split == "val") & (f.condition == "real") & (f.text != "")].head(n_probe)
    probes = pr.assign(row_type="probe", expected_raw=[fake(qp.SYSTEM, qp.build_prompt(
        r.description, r.unit, r.freq, json.loads(r.lookback), r.text)) for r in pr.itertuples()])
    texts = pd.DataFrame(dict(domain=["Economy", "Health_US"], text=["The deficit increased.", HUS],
                              source="real"))
    judge = ej.build(texts, probes.drop(columns="row_type"))
    df = pd.concat([f, judge[judge.row_type == "judge"], probes], ignore_index=True).fillna("")
    (tmp_path / "input" / "ds").mkdir(parents=True)
    df.to_csv(tmp_path / "input" / "ds" / "r3_inputs.csv", index=False)
    return f


def _run_forecaster(tmp_path, name, generate, stop_after=None, cells=("4", "6", "7", "8", "9")):
    code = [s for s, t in _cells(name) if t == "code"]
    pick = lambda key: next(c for c in code if c.startswith(key) or key in c[:80])
    spec = RUN_SPECS["m5prompts" if name == "m5" else "qwen7b"]
    calls = []

    def gen(system, user):
        if stop_after is not None and len(calls) >= stop_after:
            raise ConnectionError("Kaggle session ended")
        calls.append((system, user))
        return generate(system, user)

    ns = dict(RUN=name, MODEL_ID=spec["model_id"], REVISION=spec["revision"],
              QUANT="fp16" if name == "m5" else "nf4", RUN_VARIANTS=["p5", "sk"], PROBE=name == "m5",
              SMOKE=name == "qwen7b", MAX_NEW_TOKENS=24, SPLITS=["test", "val"], INPUT_SHA256="x",
              PLATFORM="kaggle", LOCAL_TEST=True, generate=gen, SETUP=f"T4 | {name} | kaggle")
    exec(pick("# 3. Paths and inputs").replace("/kaggle", tmp_path.as_posix()), ns)
    ns["meta"] = dict(run=name, setup=ns["SETUP"])
    exec(pick("# Generated verbatim from src/models/qwen_prompt.py"), ns)
    exec(pick("# Generated verbatim from src/models/r2_prompts.py"), ns)
    for c in cells:
        exec(pick(f"# {c}."), ns)
    return ns, calls


def test_forecaster_answers_every_row_test_first_and_shares_the_none_prompt(tmp_path):
    f = _inputs(tmp_path)
    ns, calls = _run_forecaster(tmp_path, "m5", fake)
    res = pd.read_csv(tmp_path / "working" / "r3_m5_fp16_kaggle_predictions.csv", keep_default_na=False)
    assert len(res) == len(f) + (f.condition != "none").sum()           # p5: all; sk: all but none
    assert set(res[res.condition == "none"].variant) == {"p5"}
    first_val = (res.split == "val").idxmax()
    assert (res.split.iloc[:first_val] == "test").all() and (res.split.iloc[first_val:] == "val").all()
    with_report = res.sample_id.astype(str) != "2010-01-01"              # window 0 has no report
    assert set(res[(res.variant == "sk") & with_report].y_pred) == {100.5}   # caution reaches sk only
    assert 100.5 not in set(res[res.variant == "p5"].y_pred)
    m = json.loads((tmp_path / "working" / "r3_m5_fp16_kaggle_run_meta.json").read_text())
    assert m["test_complete"] and m["val_complete"] and m["determinism_mismatches"] == 0
    assert m["probe_ok"] and m["probe_n"] == 4
    n_distinct = ns["rows"].user.nunique()                               # p5 and sk share M5's system
    assert len(calls) == 4 + n_distinct + 20                             # probe + one per prompt + determinism


def test_probe_mismatch_is_recorded_and_the_run_continues(tmp_path):
    f = _inputs(tmp_path)
    ns, _ = _run_forecaster(tmp_path, "m5", lambda s, u: fake(s, u).replace("1", "7", 1))
    m = json.loads((tmp_path / "working" / "r3_m5_fp16_kaggle_run_meta.json").read_text())
    assert not m["probe_ok"] and m["probe_mismatches"] == 4 and m["test_complete"]


def test_qwen7b_smoke_test_stop_rule(tmp_path):
    _inputs(tmp_path)
    with pytest.raises(RuntimeError, match="Too many unparseable"):
        _run_forecaster(tmp_path, "qwen7b", lambda s, u: "I cannot answer that.", cells=("6",))
    m = json.loads((tmp_path / "working" / "r3_qwen7b_nf4_kaggle_run_meta.json").read_text())
    assert m["smoke_test"]["unparseable"] == m["smoke_test"]["n"] >= 10


def test_forecaster_resumes_without_duplicates(tmp_path):
    f = _inputs(tmp_path)
    with pytest.raises(ConnectionError):
        _run_forecaster(tmp_path, "qwen7b", fake, stop_after=20 + 7)
    ns, _ = _run_forecaster(tmp_path, "qwen7b", fake)
    res = pd.read_csv(tmp_path / "working" / "r3_qwen7b_nf4_kaggle_predictions.csv", keep_default_na=False)
    assert len(res) == len(f) + (f.condition != "none").sum()
    assert not res.duplicated(["domain", "split", "sample_id", "condition", "variant"]).any()


def _fake_score(prompt, answers):
    report = prompt.split('"""')[1]
    if "increased" in report or "above the national" in report:
        return [-0.1, -3.0, -3.0]
    return [-4.0, -4.0, -0.05]


def test_judge_notebook_answers_every_question_and_records_the_probe(tmp_path):
    _inputs(tmp_path)
    code = [s for s, t in _cells("judge") if t == "code"]
    pick = lambda key: next(c for c in code if key in c)
    ns = dict(generate=lambda p, **kw: fake(qp.SYSTEM, p), score=_fake_score, SETUP="T4 | judge | kaggle",
              P5_SETUP="colab", meta={}, INPUT_SHA256="x", LOCAL_TEST=True, PLATFORM="kaggle")
    exec(pick("# 3. Paths and inputs").replace("/kaggle", tmp_path.as_posix()), ns)
    exec(pick("# Generated verbatim from src/models/qwen_prompt.py"), ns)
    exec(pick("# Generated verbatim from src/textguard/judge_prompt.py"), ns)
    exec(pick("# 6. Reproducibility probe").replace("len(probes) == 40", "len(probes) == 4"), ns)
    for key in ("# 7. Main loop", "# 8. Completeness", "# 9. Outputs"):
        exec(pick(key), ns)
    res = pd.read_csv(tmp_path / "working" / "r3_judge_kaggle_predictions.csv", keep_default_na=False)
    assert len(res) == 3 and set(res.answer) == {"up", "above"}
    assert ns["PROBE_OK"]
