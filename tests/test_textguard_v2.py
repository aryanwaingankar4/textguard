"""Tests for TextGuard v2 (R4): claim units, frames, year-over-year evidence, the gate, the cap,
the wrapper, the val-only export and the Kaggle judge notebook (run with a fake model), and the
FNSPID connector flags. No GPU, no LLM."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.attacks.connectors_r4 import connectors_in, edited_lines
from src.models.dummy import DummyForecaster
from src.textguard import export_v2_inputs as ex
from src.textguard import judge_prompt_v2 as jp
from src.textguard import v2 as V
from src.textguard.trend_detector import Evidence
from src.textguard.units import frame, units

ROOT = Path(__file__).resolve().parents[1]

ECON = ("Objective facts about the Monthly international trade balance of U.S. situation: The U.S. "
        "Census Bureau announced that the goods and services deficit was $41.9 billion in May, up "
        "$1.2 billion from $40.7 billion in April. May exports were $188.6 billion. The trade deficit "
        "is expected to fall next month.")
HUS = ("Objective facts:**\n* Nationwide during week 28, 0.8% of patient visits were due to ILI. "
       "This percentage is below the national baseline of 2.1%.\n\n**")


# --- units and frames ---------------------------------------------------------------------------

def test_units_strip_headers_keep_abbreviations_and_split_sentences():
    u = units(ECON)
    assert u[0].startswith("The U.S. Census Bureau announced")        # "U.S." is not an end
    assert u[-1] == "The trade deficit is expected to fall next month."  # A2's sentence: own unit
    assert len(u) == 3 and not any("Objective" in x for x in u)


def test_pronoun_sentence_joins_its_antecedent_and_header_lines_drop():
    u = units(HUS)
    assert u == ["Nationwide during week 28, 0.8% of patient visits were due to ILI. "
                 "This percentage is below the national baseline of 2.1%."]
    assert units("Objective Facts\n* The rate rose.") == ["The rate rose."]
    assert units(None) == [] and units("**\n\n") == []


def test_headline_bundle_gives_one_unit_per_line():
    assert units("Netflix Upgraded To Buy\nNetflix Shares Trading Lower") == [
        "Netflix Upgraded To Buy", "Netflix Shares Trading Lower"]


@pytest.mark.parametrize("unit,expected", [
    ("Travel changed by -0.8% for July 2017 as compared with July 2016.", "yoy"),
    ("a 0.8% increase over October 2016, for October 2017", "yoy"),
    ("The unemployment rate for February 2015 was 5.5 percent, down from 6.7 percent a year earlier.", "yoy"),
    ("Over the year, the unemployment rate was down by 1.3 percentage points.", "yoy"),
    ("Cumulative travel for 2017 changed by 1.3%.", "unchecked"),
    ("The average deficit decreased $2.1 billion for the three months ending in November.", "unchecked"),
    ("The deficit was $42.4 billion in November 2015, down $2.2 billion from October 2015.", "trend"),
    ("ILI was 2.9%, above the national baseline of 2.1%.", "trend"),
    # a unit naming a short-term comparison too is checked against the trend (Aryan, 2026-10-16)
    ("The price decreased to $2.679, $0.099 per gallon less than last week and $0.590 under a year ago.", "trend"),
    ("The price was $1.812, $0.041 lower than last week’s price and $1.029 under a year ago.", "trend"),
    ("a -0.1% decline from February 2018, for February 2019", "yoy"),
])
def test_frame(unit, expected):
    assert frame(unit) == expected


# --- evidence and gate ----------------------------------------------------------------------------

def test_yoy_evidence_reads_the_seasonal_lag_and_survives_missing_values():
    t = np.arange(40, dtype=float)
    h = 10 * np.sin(2 * np.pi * t / 12) + 0.1 * t             # seasonal, slowly rising
    h[-1] -= 5                                                # last month far below a year ago
    ev = V.yoy_evidence(h, 12)
    assert ev.direction == -1 and ev.strength > 0.5
    h2 = h.copy()
    h2[5] = np.nan                                            # a gap does not shift the lag
    assert V.yoy_evidence(h2, 12).direction == -1
    assert V.yoy_evidence(h[:14], 12) == V.NO_EVIDENCE        # < 3 past differences
    h3 = h.copy()
    h3[-13] = np.nan                                          # y_{last−12} missing → no evidence
    assert V.yoy_evidence(h3, 12) == V.NO_EVIDENCE


RISING = np.array([100, 104, 108, 112], float)


def test_forward_claim_against_the_trend_is_gated_like_a_change_claim():
    st = V.Settings(window=4)
    fwd = V.ClaimV2("outlook", -1, 0.9)
    assert V.gate([fwd], RISING, 10.0, st).g == pytest.approx(0.1)    # worked example numbers
    assert V.gate([V.ClaimV2("outlook", 1, 0.9)], RISING, 10.0, st).g == 1.0
    past_ok = V.ClaimV2("change", 1, 0.95)
    assert V.gate([past_ok, fwd], RISING, 10.0, st).g == pytest.approx(0.1)   # most suspicious


def test_floor_unchecked_frame_and_level_branch():
    st = V.Settings(window=4, conf_floor=0.8)
    assert V.gate([V.ClaimV2("outlook", -1, 0.7)], RISING, 10.0, st).g == 1.0   # below c₀
    st0 = V.Settings(window=4)
    assert V.gate([V.ClaimV2("change", -1, 0.9, frame="unchecked")], RISING, 10.0, st0).g == 1.0
    lvl = V.ClaimV2("level", -1, 1.0, reference=100.0)                 # "below 100", y_last 112
    assert V.gate([lvl], RISING, 10.0, st0).g == pytest.approx(0.0)


def test_yoy_claim_uses_history_not_the_lookback_trend():
    t = np.arange(40, dtype=float)
    h = 10 * np.sin(2 * np.pi * t / 12) + 0.5 * t            # every year higher than the last
    nums = h[-8:]
    claim_up = V.ClaimV2("change", 1, 0.9, frame="yoy")
    st = V.Settings(window=3)
    assert V.gate([claim_up], nums, 5.0, st, history=h, period=12).g == 1.0
    ev = V.evidence_for(claim_up, nums, 5.0, 3, history=None, period=12)
    assert ev == V.NO_EVIDENCE                                  # no history → no evidence


# --- defend: block + cap ---------------------------------------------------------------------------

def test_defend_blocks_caps_and_reports_the_applied_weight():
    A, B = np.array([0.0, 0.0, 0.0, 0.0]), np.array([1.0, 3.0, np.nan, -4.0])
    d = V.defend(A, B, [1.0, 1.0, 1.0, 0.4], 1.0, tau=0.5, cap_k=2.0)
    assert d.final.tolist() == [1.0, 2.0, 0.0, 0.0]
    assert d.weight == pytest.approx([1.0, 2 / 3, 1.0, 0.0])
    assert d.warn.tolist() == [False, False, False, True]
    soft = V.defend(A, B, [0.5] * 4, 1.0)
    assert soft.final.tolist() == [0.5, 1.5, 0.0, -2.0]
    assert np.isnan(V.defend(np.array([np.nan]), np.array([1.0]), [1.0], 1.0).final[0])


def test_wrapper_on_a_model_it_has_never_seen():
    judge = lambda text: [V.ClaimV2("outlook", -1, 0.9)] if text else []
    tg = V.TextGuardV2(DummyForecaster(), judge, sigma=10.0, settings=V.Settings(window=4))
    dec = tg.decide(RISING, "Demand will fall sharply.", history=np.r_[90.0, 95.0, RISING])
    assert dec.g == pytest.approx(0.1) and dec.final[0] == pytest.approx(
        dec.A[0] + 0.1 * (dec.B[0] - dec.A[0]))
    with pytest.raises(ValueError, match="history must end"):
        tg.decide(RISING, "x", history=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))


# --- judge prompt, export, notebook ----------------------------------------------------------------

def test_prompt_questions_are_distinct_and_answers_are_single_words():
    ps = {k: jp.build_judge_prompt("Economy", k, "The deficit rose.") for k in jp.KINDS}
    assert len(set(ps.values())) == 3 and all('"""\nThe deficit rose.\n"""' in p for p in ps.values())
    assert "only predicts the future" in ps["change"] and "will" in ps["outlook"]
    assert all(" " not in a for ans in jp.ANSWERS.values() for a in ans)
    assert V.questions_for("ILI was 2.9%, above the national baseline of 2.1%.") == jp.KINDS
    assert V.questions_for("The deficit rose.") == ("change", "outlook")


def test_export_refuses_held_out_domains():
    t = pd.DataFrame(dict(domain=["Energy"], text=["Gasoline prices rose."], source=["real"]))
    with pytest.raises(ValueError, match="original domains"):
        ex.build(t, pd.DataFrame(columns=["domain", "text"] + ex.PROBE_COLS))


def _cells():
    nb = json.loads((ROOT / "notebooks" / "r4_v2_val_judge_kaggle.ipynb").read_text(encoding="utf-8"))
    return [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]


def test_notebook_embeds_v2_module_verbatim():
    code = [s for s, t in _cells() if t == "code"]
    src = (ROOT / "src" / "textguard" / "judge_prompt_v2.py").read_text(encoding="utf-8")
    assert any(c.endswith(src.strip("\n")) for c in code)
    assert not any("Generated verbatim from src/textguard/judge_prompt.py\n" in c for c in code)
    assert any("r4v2_val_inputs.csv" in c for c in code)


def _fake_score(prompt, answers):
    s = prompt.split('"""')[1]
    q = prompt.split("Question:")[1]
    if "predict that" in q:
        if "expected to fall" in s:
            return [-3.0, -0.1, -3.0]
        return [-4.0, -4.0, -0.05]
    if "increased" in s or "above the national" in s or " up " in s:
        return [-0.1, -3.0, -3.0]
    return [-4.0, -4.0, -0.05]


def test_notebook_loop_and_loader_serve_every_claim(tmp_path):
    texts = pd.DataFrame(dict(domain=["Economy", "Health_US"], text=[ECON, HUS], source="real"))
    probes = pd.DataFrame([dict(domain="Economy", text="", split="val", sample_id=f"2015-0{i}-01",
                                condition="none", freq="monthly", description="series",
                                unit="million US dollars", lookback="[1.0, 2.0]", expected_raw="3")
                           for i in range(1, 5)])
    df = ex.build(texts, probes)
    (tmp_path / "input" / "ds").mkdir(parents=True)
    inp = tmp_path / "input" / "ds" / "r4v2_val_inputs.csv"
    df.to_csv(inp, index=False)
    code = [s for s, t in _cells() if t == "code"]
    pick = lambda key: next(c for c in code if key in c)
    ns = dict(generate=lambda p, **kw: "3", score=_fake_score, SETUP="T4 | v2 | kaggle",
              P5_SETUP="colab", meta={"setup": "T4 | v2 | kaggle"}, INPUT_SHA256="x",
              LOCAL_TEST=True, PLATFORM="kaggle")
    exec(pick("# 3. Paths and inputs").replace("/kaggle", tmp_path.as_posix()), ns)
    exec(pick("# Generated verbatim from src/models/qwen_prompt.py"), ns)
    exec(pick("# Generated verbatim from src/textguard/judge_prompt_v2.py"), ns)
    exec(pick("# 6. Reproducibility probe").replace("len(probes) == 40", "len(probes) == 4"), ns)
    for key in ("# 7. Main loop", "# 8. Completeness"):
        exec(pick(key), ns)
    out = tmp_path / "working"
    ans = V.load_answers(out / "r4v2_val_judge_kaggle_predictions.csv", inp,
                         out / "r4v2_val_judge_kaggle_run_meta.json")
    econ = V.JudgeV2("Economy", ans)(ECON)
    fwd = [c for c in econ if c.kind == "outlook" and c.direction]
    assert len(fwd) == 1 and fwd[0].direction == 1          # deficit to fall = balance up
    past = [c for c in econ if c.kind == "change" and c.direction]
    assert past and past[0].direction == -1                 # deficit up = balance down
    hus = V.JudgeV2("Health_US", ans)(HUS)
    assert [c.kind for c in hus] == ["change", "outlook", "level"] and hus[2].reference == 2.1
    with pytest.raises(KeyError):
        V.JudgeV2("Economy", ans)("A sentence that was never exported.")


# --- FNSPID connector flags ---------------------------------------------------------------------

def test_connector_flags_read_only_edited_lines():
    real = "Netflix Shares Trading Lower After Earnings Miss\nNetflix CFO To Leave"
    fake = "Netflix Shares Trading Higher After Earnings Miss\nNetflix CFO To Leave"
    assert edited_lines(real, fake) == ["Netflix Shares Trading Higher After Earnings Miss"]
    assert connectors_in("Shares Higher On Report; Due To Demand, As Seen") == ["on", "as", "due to"]
    assert connectors_in("Raises Guidance; Online Sales Up") == []    # 'Online' is not 'on'
    with pytest.raises(ValueError):
        edited_lines("a\nb", "a")
