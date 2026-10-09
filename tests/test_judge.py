"""TextGuard's direction judge (P6a): prompt, export, Colab notebook logic, CSV-backed judge.

No GPU: the notebook's own cells are executed with a fake ``generate`` / ``score``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.textguard.export_judge_inputs import build
from src.textguard.judge_prompt import QUESTIONS, build_judge_prompt, judge_sha
from src.textguard.text_direction import (Claim, QwenJudge, load_judgements, stated_baseline,
                                          text_key)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "p6_direction_judge.ipynb"


# --- prompt and helpers ----------------------------------------------------------------------

def test_prompt_names_the_domain_subject_and_the_allowed_answers():
    p = build_judge_prompt("Economy", "change", "The deficit was $41.9 billion, up $1.2 billion.")
    assert "trade deficit" in p and "up, down or none" in p and "$41.9 billion" in p
    q = build_judge_prompt("Health_US", "level", "ILI was 3.2%, above the national baseline.")
    assert "national baseline" in q and "above, below or none" in q
    assert [k for k, _ in QUESTIONS["Health_US"]] == ["level", "change"]
    with pytest.raises(KeyError):
        build_judge_prompt("Economy", "level", "x")       # Economy has no level question


@pytest.mark.parametrize("text, expected", [
    ("ILI was 3.2%, which is above the national baseline of 2.1%.", 2.1),
    ("ILI was 1.0%, well below the national baseline of 2.2 percent.", 2.2),
    ("ILI remains below the national baseline (2.0%).", 2.0),
    ("All regions reported ILI at or above region-specific baseline levels.", None),
    (None, None),
])
def test_stated_baseline(text, expected):
    assert stated_baseline(text) == expected


def test_export_dedupes_texts_and_asks_health_us_two_questions():
    texts = pd.DataFrame({"domain": ["Economy", "Economy", "Health_US", "Economy"],
                          "text": ["deficit up", "deficit up", "ILI above", ""],
                          "source": ["real", "time_shifted", "flipped", "empty"]})
    probes = pd.DataFrame([dict(domain="Economy", text="r", split="val", sample_id="2015-01-01",
                                condition="real", freq="monthly", description="d", unit="u",
                                lookback="[1, 2]", expected_raw="3")])
    df = build(texts, probes)
    j = df[df.row_type == "judge"]
    assert len(j) == 3                                   # 1 Economy + 2 Health_US questions
    assert j[j.domain == "Economy"].sources.iloc[0] == "real;time_shifted"
    assert set(j[j.domain == "Health_US"].question) == {"level", "change"}
    assert (df.row_type == "probe").sum() == 1


# --- notebook: in sync with the modules, pinned revision ---------------------------------------

def _cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]


def test_notebook_embeds_both_prompt_modules_and_pins_p5_revision():
    for mod in ("src/models/qwen_prompt.py", "src/textguard/judge_prompt.py"):
        src = (ROOT / mod).read_text(encoding="utf-8").strip("\n")
        assert any(s.endswith(src) and t == "code" for s, t in _cells()), \
            "notebook is stale: run python notebooks/build_p6_judge_notebook.py"
    p5 = json.loads((ROOT / "results" / "qwen" / "p5_qwen_run_meta.json").read_text())
    load = next(s for s, _ in _cells() if "def score" in s)
    assert f'REVISION = "{p5["revision"]}"' in load and "revision=REVISION" in load
    assert "do_sample=False" in load and "torch.float16" in load and "torch.is_tensor(ids)" in load


def test_notebook_setup_string_matches_setup_id():
    from src.models.qwen_forecaster import setup_id
    load = next(s for s, _ in _cells() if "def score" in s)
    expr = load[load.index("SETUP = ("):load.index('meta["setup"]')]
    meta = dict(model_id="Qwen/Qwen2.5-1.5B-Instruct", revision="abc", gpu="Tesla T4",
                torch="2.11.0+cu130", transformers="5.18.0")
    ns = {"meta": meta, "MODEL_ID": meta["model_id"], "REVISION": "abc"}
    exec(expr, ns)
    assert ns["SETUP"] == setup_id(meta)


# --- notebook logic with a fake model ----------------------------------------------------------

TEXTS = {"Economy": ["The goods and services deficit increased $2.0 billion.",
                     "The deficit decreased $1.0 billion.", "Unemployment rose."],
         "Health_US": ["ILI was 3.0%, above the national baseline of 2.1%."]}


def _inputs(tmp_path):
    texts = pd.DataFrame([dict(domain=d, text=t, source="real") for d, ts in TEXTS.items()
                          for t in ts])
    probes = pd.DataFrame([dict(domain="Economy", text="", split="val", sample_id=f"2015-0{i}-01",
                                condition="none", freq="monthly", description="series",
                                unit="million US dollars", lookback="[1.0, 2.0]",
                                expected_raw="3") for i in range(1, 5)])
    df = build(texts, probes)
    df.to_csv(tmp_path / "judge_inputs.csv", index=False)
    return df


def _fake_score(prompt, answers):
    """Reads only the report block: 'increased'/'above' -> first answer, 'decreased' ->
    second, else 'none'."""
    report = prompt.split('"""')[1]
    if "increased" in report or "above the national" in report:
        return [-0.1, -3.0, -3.0]
    if "decreased" in report:
        return [-3.0, -0.2, -3.0]
    return [-4.0, -4.0, -0.05]


def _run(tmp_path, generate=lambda p, **kw: "3", setup="SETUP-A", n_probe_expected=4):
    cells = [s for s, t in _cells() if t == "code"]
    pick = lambda key: next(c for c in cells if key in c)
    meta_path = tmp_path / "p6_judge_run_meta.json"
    ns = {"generate": generate, "score": _fake_score, "SETUP": setup, "P5_SETUP": "SETUP-A",
          "meta": {"setup": setup}}
    exec(pick("# 3. Paths and inputs").replace("/content", tmp_path.as_posix()), ns)
    exec(pick("# Generated verbatim from src/models/qwen_prompt.py"), ns)
    exec(pick("# Generated verbatim from src/textguard/judge_prompt.py"), ns)
    exec(pick("# 6. Reproducibility probe").replace("len(probes) == 40",
                                                    f"len(probes) == {n_probe_expected}"), ns)
    for key in ("# 7. Main loop", "# 8. Completeness"):
        exec(pick(key), ns)
    return ns, meta_path


def test_notebook_answers_every_question_and_judge_serves_claims(tmp_path):
    inp = _inputs(tmp_path)
    ns, meta = _run(tmp_path)
    assert ns["PROBE_OK"]
    df = load_judgements(tmp_path / "p6_judge_predictions.csv", tmp_path / "judge_inputs.csv",
                         meta)
    assert len(df) == (inp.row_type == "judge").sum() and df.probe_ok.all()
    econ = QwenJudge("Economy", data=df)
    up = econ(TEXTS["Economy"][0])
    # "deficit increased" = literal up = series DOWN (balance), confidence renormalised
    assert up == [Claim("change", -1, pytest.approx(up[0].confidence))]
    assert 0.9 < up[0].confidence <= 1.0
    assert econ(TEXTS["Economy"][1])[0].direction == 1
    assert econ(TEXTS["Economy"][2])[0].direction == 0            # off-subject -> none
    assert econ(None) == [] and econ("  ") == []
    with pytest.raises(KeyError):
        econ("a text that was never exported")
    hus = QwenJudge("Health_US", data=df)(TEXTS["Health_US"][0])
    level = next(c for c in hus if c.kind == "level")
    assert level.direction == 1 and level.reference == 2.1


def test_probe_mismatch_is_recorded_not_hidden(tmp_path):
    _inputs(tmp_path)
    ns, meta = _run(tmp_path, generate=lambda p, **kw: "999")
    assert not ns["PROBE_OK"]
    df = load_judgements(tmp_path / "p6_judge_predictions.csv", tmp_path / "judge_inputs.csv",
                         meta)
    assert not df.probe_ok.any()


def test_notebook_resumes_and_refuses_a_second_setup(tmp_path):
    inp = _inputs(tmp_path)
    _run(tmp_path)
    out = tmp_path / "p6_judge_predictions.csv"
    res = pd.read_csv(out, keep_default_na=False, dtype=str)
    res.iloc[:2].to_csv(out, index=False)                          # simulated disconnect
    _run(tmp_path)
    assert len(pd.read_csv(out)) == (inp.row_type == "judge").sum()
    with pytest.raises(AssertionError, match="another software setup"):
        _run(tmp_path, setup="SETUP-B")


def test_loader_rejects_changed_prompts_and_missing_answers(tmp_path, monkeypatch):
    _inputs(tmp_path)
    _, meta = _run(tmp_path)
    pred, inp = tmp_path / "p6_judge_predictions.csv", tmp_path / "judge_inputs.csv"
    import src.textguard.text_direction as td
    monkeypatch.setattr(td, "text_key", lambda d, q, t: judge_sha(build_judge_prompt(d, q, t) + "!"))
    with pytest.raises(ValueError, match="fingerprint"):
        load_judgements(pred, inp, meta)
    monkeypatch.undo()
    res = pd.read_csv(pred, keep_default_na=False, dtype=str)
    res.iloc[1:].to_csv(pred, index=False)
    with pytest.raises(ValueError, match="no judge answer"):
        load_judgements(pred, inp, meta)
    assert text_key("Economy", "change", "x") == judge_sha(build_judge_prompt("Economy", "change", "x"))
