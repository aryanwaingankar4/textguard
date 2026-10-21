"""R7 public toolkit: data format, frozen settings, live judges (stub scorer), audit, notebook template."""

import json
import os
import re

import numpy as np
import pandas as pd
import pytest

import textguard as TG
from src.textguard import heldout_questions as HQ
from src.textguard import judge_prompt as JP1
from src.textguard import judge_prompt_v2 as JP2
from src.textguard.live_judge import LiveJudgeV1, LiveJudgeV2, register_subject
from src.textguard.toolkit import a2_attack, a2_sentence, load_series, v1_window, v2_settings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "examples", "sample_series.csv")


class StubScorer:
    """Stands in for QwenScorer: answers from direction words in the judged sentence."""

    def __init__(self):
        self.calls = []

    def ask(self, system, user_prompt, sha, answers):
        body = user_prompt.split('"""')[1].lower()
        q = user_prompt.split("Question:")[1].splitlines()[0].lower()   # the question line only
        self.calls.append((system, user_prompt))
        if "predict" in q:
            up, down = re.search(r"expected to rise|will rise", body), re.search(r"expected to fall|will fall", body)
        else:
            up = re.search(r"\b(rose|increased)\b", body) and "expected" not in body
            down = re.search(r"\b(fell|decreased)\b", body) and "expected" not in body
            if "expected to rise" in body:
                up = True
            if "expected to fall" in body:
                down = True
        a = answers[0] if up and not down else (answers[1] if down and not up else answers[2])
        return a, 0.9

    def answers(self):
        return pd.DataFrame({"n": [len(self.calls)]})


def _stub_judges(key="widgets-test", subject="the average widget price"):
    try:
        register_subject(key, subject, "compared with the previous week")
    except ValueError:
        pass
    s = StubScorer()
    return LiveJudgeV1(key, s), LiveJudgeV2(key, s)


def test_frozen_settings_match_the_paper():
    p6 = json.load(open(os.path.join(ROOT, "results", "summary_p6_frozen.json")))["textguard"]
    v2 = json.load(open(os.path.join(ROOT, "results", "summary_r4_v2_frozen.json")))
    assert (v1_window(8), v1_window(24)) == (p6["window"]["Economy"], p6["window"]["Health_US"])
    s8, s24 = v2_settings(8), v2_settings(24)
    assert (s8.window, s24.window) == (v2["window"]["Economy"], v2["window"]["Health_US"])
    assert (s24.conf_floor, s24.tau, s24.cap_k) == (v2["conf_floor"], v2["tau"], v2["cap_k"])


def test_registered_subject_asks_r4s_question():
    """The toolkit's prompts for a registered series are byte-identical to R4's held-out prompts."""
    register_subject("toolkit-energy", HQ.HELDOUT_SUBJECTS["Energy"], "compared with the previous week")
    HQ.register()
    text = "The price rose 3 cents."
    assert JP1.build_judge_prompt("toolkit-energy", "change", text) == \
        JP1.build_judge_prompt("Energy", "change", text)
    for kind in ("change", "outlook"):
        assert JP2.build_judge_prompt("toolkit-energy", kind, text) == JP2.build_judge_prompt("Energy", kind, text)
    with pytest.raises(ValueError):
        register_subject("toolkit-energy", "something else")


def test_load_series_format_and_errors(tmp_path):
    d = load_series(SAMPLE, lookback=12, freq="weekly")
    df = pd.read_csv(SAMPLE)
    assert d.sigma == pytest.approx(df[df.split == "train"].value.std(ddof=0))
    w = d.windows[0]
    assert w.numbers.size == 12 and np.allclose(w.history[-12:], w.numbers)
    assert all(isinstance(x.text, (str, type(None))) for x in d.windows)
    bad = df.drop(columns="split")
    with pytest.raises(ValueError, match="train_end"):
        load_series(bad, lookback=12)
    bad2 = df.assign(fake_direction=np.nan)
    with pytest.raises(ValueError, match="fake_direction"):
        load_series(bad2, lookback=12)
    with pytest.raises(ValueError, match="missing column"):
        load_series(df.drop(columns="value"), lookback=12)


def test_a2_reverses_the_recent_trend_and_keeps_the_report():
    up = np.arange(12, dtype=float)
    fake, d = a2_attack("Prices rose.", up, 1.0, 12, "the widget price", "weekly")
    assert d == -1 and fake == "Prices rose. The widget price is expected to fall next week."
    assert a2_attack(None, up, 1.0, 12, "x") == (None, 0)
    assert a2_attack("Flat.", np.ones(12), 1.0, 12, "x") == (None, 0)
    assert a2_sentence("the rate", 1, "monthly") == "The rate is expected to rise next month."


def test_live_judges_read_every_unit():
    j1, j2 = _stub_judges()
    text = "The average widget price rose 1.0% from the previous week. The average widget price is expected to fall next week."
    c1 = j1(text)
    assert len(c1) == 1 and c1[0].kind == "change"          # v1: one question about the whole report
    kinds = sorted((c.kind, c.direction) for c in j2(text))
    assert ("change", -1) in kinds and ("change", 1) in kinds   # v2 reads the appended sentence on its own
    assert j1(None) == [] and j2(None) == []


def test_audit_end_to_end_dummy(tmp_path):
    data = load_series(SAMPLE, lookback=12, freq="weekly")
    model = TG.DummyForecaster(sensitivity=1.0, step=data.sigma / 4, lookback=12)
    j1, j2 = _stub_judges()
    rep = TG.audit(model, data, judge_v1=j1, judge_v2=j2, subject="the average widget price",
                   n_boot=200)
    s = rep.summary.set_index(["attack", "defence"])
    assert set(s.index.get_level_values(1)) == {"none", "never_trust", "clip", "TG1", "TG2"}
    assert s.loc[("a2", "none"), "dss"] > 0.05                   # the dummy follows the lie
    assert s.loc[("a2", "never_trust"), "dss"] == 0
    assert s.loc[("a2", "TG2"), "dss"] < s.loc[("a2", "none"), "dss"]
    assert s.loc[("genuine", "never_trust"), "fbr"] == 1 and s.loc[("genuine", "none"), "fbr"] == 0
    assert 0 <= s.loc[("genuine", "TG2"), "fbr"] <= 1
    assert ("user", "none") in s.index and s.loc[("user", "none"), "n"] == data.frame.fake_text.notna().sum()
    out = rep.save(tmp_path)
    for f in ("audit_summary.csv", "audit_windows.csv", "audit_worst_cases.csv", "audit_report.md",
              "audit_summary.json"):
        assert (out / f).exists()
    assert "| attack | defence |" in (out / "audit_report.md").read_text(encoding="utf-8")


def test_wrappers_are_forecasters():
    data = load_series(SAMPLE, lookback=12, freq="weekly")
    model = TG.DummyForecaster(step=data.sigma / 4, lookback=12)
    j1, j2 = _stub_judges()
    w = next(x for x in data.windows if x.text)
    g1 = TG.TextGuardV1(model, j1, data.sigma, 12)
    g2 = TG.TextGuardV2(model, j2, data.sigma, 12, freq="weekly")
    assert g1.window == 3 and g2.settings.window == 6 and g2.period == 52
    for g in (g1, g2):
        assert isinstance(g, TG.Forecaster)
    assert g1.predict(w.numbers, w.text).shape == (1,)
    assert g2.predict(w.numbers, w.text, history=w.history).shape == (1,)


def test_sample_data_is_regenerable(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("msd", os.path.join(ROOT, "examples", "make_sample_data.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "OUT", tmp_path / "s.csv")
    m.main()
    assert (tmp_path / "s.csv").read_bytes().replace(b"\r\n", b"\n") == \
        open(SAMPLE, "rb").read().replace(b"\r\n", b"\n")


def test_notebook_template_runs_the_worked_example(tmp_path, monkeypatch):
    """Every code cell compiles; cells 2–6 run on the worked example with the stub judge."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("bt", os.path.join(ROOT, "notebooks", "build_toolkit_notebook.py"))
    bt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bt)
    nb = bt.build()
    committed = json.load(open(os.path.join(ROOT, "notebooks", "textguard_your_model.ipynb"), encoding="utf-8"))
    assert committed == json.loads(json.dumps(nb)), "re-run notebooks/build_toolkit_notebook.py"
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for c in cells:
        compile(c, "<cell>", "exec")
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(TG, "judges", lambda *a, **k: _stub_judges("nb-test"))
    ns = {}
    for c in cells[1:6]:                                   # skip clone (1) and download (7)
        c = c.replace('report.save("audit_out")', f'report.save(r"{tmp_path}")')
        c = c.replace('"audit_out/judge_answers.csv"', repr(str(tmp_path / "judge_answers.csv")))
        exec(c, ns)
    assert (tmp_path / "audit_summary.csv").exists() and (tmp_path / "judge_answers.csv").exists()


def test_download_reproduces_the_windows_checkout_bytes():
    """R7 clean-room finding: Time-MMD must be CRLF everywhere, as in the clone the results came from."""
    from src.data.download import to_windows_checkout
    assert to_windows_checkout(b'a,b\n"x\ny",1\r\n') == b'a,b\r\n"x\r\ny",1\r\n'
    assert to_windows_checkout(b"a\r\n") == b"a\r\n"
