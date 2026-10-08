"""Tests for Model 5's prompt, parser, Colab notebook logic and CSV-backed forecaster.

No GPU and no Qwen: the notebook's own cells are executed with a fake ``generate``, so the
resume / incremental-save / parse-failure logic that will run on Colab is tested here.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import qwen_prompt as qp
from src.models.qwen_forecaster import QwenForecaster, load_predictions, parse_failure_rate

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "p2_qwen_forecaster.ipynb"


# --- parser ------------------------------------------------------------------------------

@pytest.mark.parametrize("out, unit, expected", [
    ("282558", "million vehicle miles", 282558.0),
    ("282,558", "million vehicle miles", 282558.0),
    ("-66404.0", "million US dollars", -66404.0),
    ("\u221258365.2", "million US dollars", -58365.2),            # Unicode minus
    ("-66.4 billion", "million US dollars", -66400.0),            # scale word ≠ unit
    ("282.5 billion vehicle miles", "million vehicle miles", 282500.0),
    ("4.5%", "percent", 4.5),
    ("t-1: 4.1, so t = 4.3", "percent", 4.1),                    # echoed label skipped
    ("The value at t is 1.02", "percent", 1.02),
    (".95", "percent", 0.95),
])
def test_parse_number(out, unit, expected):
    assert qp.parse_number(out, unit) == pytest.approx(expected)


@pytest.mark.parametrize("out", ["", "I cannot predict that.", "N/A", None, "t-1"])
def test_parse_failure_is_nan_not_a_guess(out):
    assert math.isnan(qp.parse_number(out, "percent"))


# --- prompt ------------------------------------------------------------------------------

LOOKBACK = [4.1, 4.5, 4.6, 4.0]


def test_prompt_uses_relative_steps_and_no_dates():
    p = qp.build_prompt("US unemployment rate", "percent", "monthly", LOOKBACK, "rate rose")
    assert [l.split(":")[0] for l in p.splitlines() if l.startswith("t-")] == ["t-4", "t-3",
                                                                            "t-2", "t-1"]
    numeric_part = p.split("Report published")[0]
    assert not any(ch.isdigit() and len(tok) == 4 and tok.startswith(("19", "20"))
                   for tok in numeric_part.split() for ch in tok[:1])


def test_none_prompt_is_the_real_prompt_without_the_report_block():
    real = qp.build_prompt("d", "percent", "weekly", LOOKBACK, "ILI was 2.1%, above baseline")
    none = qp.build_prompt("d", "percent", "weekly", LOOKBACK, None)
    block = '\n\nReport published before period t:\n"""\nILI was 2.1%, above baseline\n"""'
    assert real.replace(block, "") == none
    assert qp.build_prompt("d", "percent", "weekly", LOOKBACK, "   ") == none   # blank ≡ None


def test_format_value_is_compact_and_exact_enough():
    assert qp.format_value(282558.0) == "282558"
    assert qp.format_value(-66029.2) == "-66029.2"
    assert qp.format_value(0.992711) == "0.992711"


# --- notebook stays in sync with the module -----------------------------------------------

def _cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]


def test_notebook_embeds_qwen_prompt_verbatim():
    src = (ROOT / "src" / "models" / "qwen_prompt.py").read_text(encoding="utf-8").strip("\n")
    assert any(s.endswith(src) and t == "code" for s, t in _cells()), \
        "notebook is stale: run python notebooks/build_p2_qwen_notebook.py"


def test_notebook_uses_greedy_decoding_and_requests_a_gpu():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["metadata"]["colab"]["gpuType"] == "T4"
    gen = next(s for s, _ in _cells() if "def generate" in s)
    assert "do_sample=False" in gen and "torch.float16" in gen
    assert "torch.is_tensor(ids)" in gen    # transformers >= 5: apply_chat_template -> BatchEncoding


# --- notebook logic, run locally with a fake model ----------------------------------------

def _inputs(path: Path) -> pd.DataFrame:
    rows = []
    for d, freq, unit in (("Economy", "monthly", "million US dollars"),
                          ("Health_US", "weekly", "percent")):
        for split in ("val", "test"):
            for i in range(6):
                x = [100.0 + i + k for k in range(4)]
                rows.append(dict(domain=d, split=split, sample_id=f"20{10 + i}-0{1 + (split == 'test')}-01",
                                 freq=freq, description=f"{d} series", unit=unit,
                                 lookback=json.dumps(x), text="" if i == 0 else f"{d} rose {i}",
                                 y_true=105.0 + i, train_mean=100.0, train_std=2.0))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def _run_notebook(tmp_path, fake_generate):
    cells = [s for s, t in _cells() if t == "code"]
    ns = {"meta": {}, "generate": fake_generate}
    pick = lambda key: next(c for c in cells if key in c)
    exec(pick("# 3. Paths and inputs").replace("/content", tmp_path.as_posix()), ns)
    exec(pick("# Generated verbatim from src/models/qwen_prompt.py"), ns)
    exec(pick("# 7. Main loop"), ns)
    return ns, pick


def test_notebook_loop_resumes_after_disconnect_and_never_drops_failures(tmp_path):
    _inputs(tmp_path / "qwen_inputs.csv")
    calls = []

    def flaky(prompt):
        calls.append(prompt)
        if len(calls) == 15:
            raise ConnectionError("Colab disconnected")
        return "I am not sure" if "rose 3" in prompt else "106.5"

    with pytest.raises(ConnectionError):
        _run_notebook(tmp_path, flaky)
    saved = pd.read_csv(tmp_path / "p2_qwen_predictions.csv")
    assert 0 < len(saved) < 48 and len(saved) % 10 == 0           # flushed in blocks of 10

    ns, pick = _run_notebook(tmp_path, flaky)                       # resume
    exec(pick("# 8. Check completeness"), ns)                       # asserts completeness
    res = ns["res"]
    assert len(res) == 48 and not res.duplicated(["domain", "split", "sample_id", "condition"]).any()
    fails = res[~res.parse_ok]
    assert len(fails) == 4 and fails.y_pred.isna().all()            # kept, as NaN
    assert set(fails.condition) == {"real"}
    # a window with no report: real and none are one prompt, generated once
    no_text = res[res.sample_id == "2010-01-01"]
    assert no_text.prompt_sha.nunique() == 2                        # 2 domains x 1 prompt each


def test_forecaster_serves_notebook_output(tmp_path):
    inp = _inputs(tmp_path / "qwen_inputs.csv")
    _run_notebook(tmp_path, lambda p: "0.105 billion" if "Economy" in p else "104.5")
    df = load_predictions(tmp_path / "p2_qwen_predictions.csv", tmp_path / "qwen_inputs.csv")
    m = QwenForecaster("Economy", data=df)
    r = inp[(inp.domain == "Economy")].iloc[2]
    x = np.array(json.loads(r.lookback))
    assert m.predict(x, r.text)[0] == pytest.approx(105.0)       # billion -> million unit
    assert m.predict(x, None).shape == (1,)
    np.testing.assert_array_equal(m.predict(x, "  "), m.predict(x, None))
    with pytest.raises(KeyError, match="no precomputed"):
        m.predict(x, "a fake report that was never sent to Qwen")
    with pytest.raises(KeyError):
        m.predict(x + 0.5, None)
    rate = parse_failure_rate(df)
    assert rate.parse_failures.sum() == 0 and rate.scale_failures.sum() == 0


def test_forecaster_rejects_predictions_from_a_different_prompt(tmp_path, monkeypatch):
    _inputs(tmp_path / "qwen_inputs.csv")
    _run_notebook(tmp_path, lambda p: "1.0")
    import src.models.qwen_forecaster as qf
    monkeypatch.setattr(qf, "build_prompt", lambda *a: "a changed prompt")
    with pytest.raises(ValueError, match="different prompt"):
        qf.load_predictions(tmp_path / "p2_qwen_predictions.csv", tmp_path / "qwen_inputs.csv")


def test_off_scale_answer_is_a_format_failure_not_a_forecast(tmp_path):
    """Traffic case: answered '267.0' (billions) for a series in millions -> NaN, counted."""
    _inputs(tmp_path / "qwen_inputs.csv")
    _run_notebook(tmp_path, lambda p: "0.5" if "rose 2" in p else "103")  # window ~100-108
    df = load_predictions(tmp_path / "p2_qwen_predictions.csv", tmp_path / "qwen_inputs.csv")
    bad = df[~df.scale_ok]
    assert len(bad) > 0 and bad.y_pred.isna().all() and (bad.y_pred_raw == 0.5).all()
    assert bad.parse_ok.all() and not bad.format_ok.any()
    assert df[df.scale_ok].y_pred.notna().all()
    rate = parse_failure_rate(df, by=("domain", "condition"))
    assert rate.scale_failures.sum() == len(bad) and rate.parse_failures.sum() == 0
    r = df[~df.scale_ok].iloc[0]
    m = QwenForecaster(r.domain, data=df)
    assert np.isnan(m.predict(json.loads(r.lookback), r.text)[0])


def test_scale_ok_bounds():
    from src.models.qwen_forecaster import scale_ok
    window = [250000.0, 280000.0]
    assert scale_ok(260000.0, window) and scale_ok(2_000_000.0, window)
    assert not scale_ok(267.0, window) and not scale_ok(3e6, window)
    assert scale_ok(-60000.0, [-66000.0, -45000.0])       # sign-agnostic (Economy deficits)
    assert scale_ok(float("nan"), window)                 # NaN is a parse failure, not scale


# --- P5a: altered conditions, reuse of P2.4 answers, reproducibility probe -------------------

NOTEBOOK_P5 = ROOT / "notebooks" / "p5_qwen_conditions.ipynb"


def _cells_p5():
    nb = json.loads(NOTEBOOK_P5.read_text(encoding="utf-8"))
    return [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]


def test_p5_notebook_embeds_qwen_prompt_verbatim_and_pins_the_revision():
    src = (ROOT / "src" / "models" / "qwen_prompt.py").read_text(encoding="utf-8").strip("\n")
    assert any(s.endswith(src) and t == "code" for s, t in _cells_p5()), \
        "notebook is stale: run python notebooks/build_p5_qwen_notebook.py"
    load = next(s for s, _ in _cells_p5() if "def generate" in s)
    assert "do_sample=False" in load and "torch.float16" in load
    assert "torch.is_tensor(ids)" in load                  # BatchEncoding fix (P2.4)
    assert "revision=REVISION" in load and "model_info(MODEL_ID).sha" in load   # issue #28


def _p5_files(tmp_path, change=lambda i: i % 2 == 1):
    """P2.4 run (fake model) + a P5 inputs file: 'flipped' alters some texts (others reuse),
    'shuffled' alters all; 2 probe rows per condition from P2.4's val answers."""
    inp = _inputs(tmp_path / "qwen_inputs.csv")
    _run_notebook(tmp_path, lambda p: "101" if "Report" in p else "100")
    p2 = pd.read_csv(tmp_path / "p2_qwen_predictions.csv", keep_default_na=False,
                     dtype={"sample_id": str})
    parts = []
    for cond in ("flipped", "shuffled"):
        d = inp.copy()
        alter = [change(i) or cond == "shuffled" for i in range(len(d))]
        d["text"] = [f"{t} FAKE {cond}" if a and t else t for t, a in zip(d.text, alter)]
        d["reuse"] = ["none" if not t else "" if a else "real" for t, a in zip(d.text, alter)]
        parts.append(d.assign(condition=cond, probe=False, expected_raw=""))
    pr = inp[(inp.split == "val") & (inp.text != "")].head(2)
    for cond in ("real", "none"):
        e = pr.merge(p2[p2.condition == cond][["domain", "split", "sample_id", "raw_output"]],
                     on=["domain", "split", "sample_id"])
        parts.insert(0, e.rename(columns={"raw_output": "expected_raw"}).assign(
            condition=cond, reuse="", probe=True,
            text=e.text if cond == "real" else ""))
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(tmp_path / "qwen_inputs_p5.csv", index=False)
    return df


def _run_p5_notebook(tmp_path, fake_generate):
    cells = [s for s, t in _cells_p5() if t == "code"]
    ns = {"meta": {}, "generate": fake_generate, "META": str(tmp_path / "meta.json")}
    pick = lambda key: next(c for c in cells if key in c)
    exec(pick("# 3. Paths and inputs").replace("/content", tmp_path.as_posix()), ns)
    exec(pick("# Generated verbatim from src/models/qwen_prompt.py"), ns)
    ns["META"] = str(tmp_path / "meta.json")
    for key in ("# 6. Reproducibility probe", "# 7. Main loop", "# 8. Completeness"):
        exec(pick(key), ns)
    return ns


def test_p5_notebook_skips_reusable_rows_when_the_probe_reproduces_p2(tmp_path):
    inp = _p5_files(tmp_path)
    seen = []
    ns = _run_p5_notebook(tmp_path, lambda p: seen.append(p) or ("101" if "Report" in p else "100"))
    assert ns["REUSE_OK"]
    rows = inp[~inp.probe]
    res = pd.read_csv(tmp_path / "p5_qwen_predictions.csv", keep_default_na=False)
    assert len(res) == (rows.reuse == "").sum() < len(rows)        # reused rows not generated
    assert not any("FAKE" not in p and "Report" in p for p in seen[4:])

    from src.models.qwen_forecaster import load_all_predictions
    df = load_all_predictions(tmp_path / "p2_qwen_predictions.csv", tmp_path / "qwen_inputs.csv",
                              tmp_path / "p5_qwen_predictions.csv",
                              tmp_path / "qwen_inputs_p5.csv")
    p5 = df[df.condition.isin(["flipped", "shuffled"])]
    assert len(p5) == len(rows) and set(p5.reused_from) == {"", "real", "none"}
    # a reused row is the P2.4 answer to the byte-identical prompt
    reused = p5[p5.reused_from == "real"].iloc[0]
    p2_row = df[(df.condition == "real") & (df.sample_id == reused.sample_id)
                & (df.domain == reused.domain) & (df.split == reused.split)].iloc[0]
    assert reused.prompt_sha == p2_row.prompt_sha and reused.y_pred == p2_row.y_pred
    # the forecaster serves attacked text, and an unchanged fake = the real prediction
    m = QwenForecaster(reused.domain, data=df)
    x = json.loads(reused.lookback)
    assert m.predict(x, reused.text)[0] == m.predict(x, p2_row.text)[0]
    fake = p5[p5.reused_from == ""].iloc[0]
    assert "FAKE" in fake.text and m.predict(json.loads(fake.lookback), fake.text)[0] == 101


def test_p5_notebook_generates_everything_when_the_probe_differs(tmp_path):
    inp = _p5_files(tmp_path)
    ns = _run_p5_notebook(tmp_path, lambda p: "102")               # a different software stack
    assert not ns["REUSE_OK"] and ns["meta"]["probe_mismatches"] > 0
    res = pd.read_csv(tmp_path / "p5_qwen_predictions.csv", keep_default_na=False)
    assert len(res) == (~inp.probe).sum()                          # nothing reused
    from src.models.qwen_forecaster import load_all_predictions
    df = load_all_predictions(tmp_path / "p2_qwen_predictions.csv", tmp_path / "qwen_inputs.csv",
                              tmp_path / "p5_qwen_predictions.csv",
                              tmp_path / "qwen_inputs_p5.csv")
    assert (df[df.condition == "flipped"].reused_from == "").all()


def test_incomplete_p5_run_is_refused(tmp_path):
    _p5_files(tmp_path)
    _run_p5_notebook(tmp_path, lambda p: "101" if "Report" in p else "100")
    out = tmp_path / "p5_qwen_predictions.csv"
    res = pd.read_csv(out, keep_default_na=False)
    res.iloc[:-3].to_csv(out, index=False)
    from src.models.qwen_forecaster import load_all_predictions
    with pytest.raises(ValueError, match="incomplete"):
        load_all_predictions(tmp_path / "p2_qwen_predictions.csv", tmp_path / "qwen_inputs.csv",
                             out, tmp_path / "qwen_inputs_p5.csv")
