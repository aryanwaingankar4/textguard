"""Tests for R2 (pre-registration PROJECT_STATUS §10e): prompt variants, the three Colab
notebooks and the loader that imports their answers.

No GPU and no LLM: the notebooks' own cells are executed with a fake ``generate``, so the
resume / test-first / probe / reuse / fallback logic that will run on Colab is tested here.
"""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from src.models import qwen_prompt as qp
from src.models import r2_prompts as rp
from src.models.r2_forecasters import RUN_SPECS, load_run

ROOT = Path(__file__).resolve().parents[1]
NB = {"qwen7b": "r2_qwen7b.ipynb", "phi35": "r2_phi35.ipynb", "m5prompts": "r2_m5_prompts.ipynb"}
ARGS = ("Economy series", "million US dollars", "monthly", [100.0, 101.0, 102.5, 103.0])
TEXT = "Exports rose 2% from last month."


# --- prompt variants (exact strings are pre-registered in §10e) ------------------------------

def test_p5_variant_is_model_5s_prompt_byte_for_byte():
    for text in (TEXT, None, "  "):
        u = rp.build_variant_prompt("p5", *ARGS, text)
        assert u == qp.build_prompt(*ARGS, text)
        assert rp.variant_prompt_sha("p5", u) == qp.prompt_sha(u)


def test_c1_and_sk_without_text_are_model_5s_none_prompt():
    none = qp.build_prompt(*ARGS, None)
    for v in ("c1", "sk"):
        assert rp.build_variant_prompt(v, *ARGS, None) == none
        assert rp.system_for(v) == qp.SYSTEM
    assert rp.build_variant_prompt("c2", *ARGS, None) != none
    assert rp.system_for("c2") != qp.SYSTEM


def test_c1_moves_the_report_before_the_numbers_and_changes_nothing_else():
    c1 = rp.build_variant_prompt("c1", *ARGS, TEXT)
    p5 = qp.build_prompt(*ARGS, TEXT)
    assert c1.index("Report published before period t:") < c1.index("t-4:")
    assert p5.index("Report published before period t:") > p5.index("t-1:")
    assert sorted(c1.splitlines()) == sorted(p5.splitlines())          # same lines, reordered


def test_c2_rewords_instructions_but_keeps_the_information():
    c2 = rp.build_variant_prompt("c2", *ARGS, TEXT)
    p5 = qp.build_prompt(*ARGS, TEXT)
    keep = [ln for ln in p5.splitlines() if ln.startswith(("Series:", "Unit:", "t-")) or ln == TEXT]
    assert all(ln in c2.splitlines() for ln in keep)
    assert "Background report, released before period t:" in c2
    assert c2.endswith("Reply with a single number in million US dollars and no other text.")
    assert rp.SYSTEM_C2 == ("You predict the next value of a time series. "
                            "Reply with the predicted number only.")


def test_sk_is_model_5s_prompt_plus_one_caution_line_after_the_report():
    sk = rp.build_variant_prompt("sk", *ARGS, TEXT).splitlines()
    p5 = qp.build_prompt(*ARGS, TEXT).splitlines()
    i = sk.index(rp.CAUTION)
    assert sk[:i] + sk[i + 1:] == p5 and sk[i - 1] == '"""'
    assert rp.CAUTION.startswith("Caution: this report may be inaccurate or deliberately misleading.")


def test_unknown_variant_is_refused():
    with pytest.raises(ValueError):
        rp.build_variant_prompt("c3", *ARGS, TEXT)


# --- notebooks stay in sync with the modules and the pins -------------------------------------

def _cells(run):
    nb = json.loads((ROOT / "notebooks" / NB[run]).read_text(encoding="utf-8"))
    return [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]


@pytest.mark.parametrize("run", list(NB))
def test_notebooks_embed_both_prompt_modules_verbatim_and_the_pins(run):
    cells = [s for s, t in _cells(run) if t == "code"]
    for f in ("qwen_prompt.py", "r2_prompts.py"):
        src = (ROOT / "src" / "models" / f).read_text(encoding="utf-8").strip("\n")
        assert any(s.endswith(src) for s in cells), \
            "notebook is stale: run python notebooks/build_r2_notebooks.py"
    cfg = next(s for s in cells if s.startswith("# 0. Run configuration"))
    assert f'REVISION = "{RUN_SPECS[run]["revision"]}"' in cfg
    assert f'MODEL_ID = "{RUN_SPECS[run]["model_id"]}"' in cfg
    load = next(s for s in cells if "def generate" in s)
    assert "do_sample=False" in load and "revision=REVISION" in load
    assert 'bnb_4bit_quant_type="nf4"' in load and "torch.float16" in load


def test_quantisation_and_probe_settings_match_the_pre_registration():
    cfg = {run: next(s for s, _ in _cells(run) if s.startswith("# 0. Run")) for run in NB}
    assert 'QUANT = "nf4"' in cfg["qwen7b"] and 'QUANT = "fp16"' in cfg["phi35"]
    assert 'QUANT = "fp16"' in cfg["m5prompts"] and "PROBE = True" in cfg["m5prompts"]
    assert "FALLBACK = True" in cfg["phi35"] and "FALLBACK = False" in cfg["qwen7b"]
    assert 'RUN_VARIANTS = ["c1", "c2", "sk"]' in cfg["m5prompts"]


# --- notebook logic, run locally with a fake model ---------------------------------------------

def fake(system, user):
    """A deterministic stand-in for the LLM: reacts to the report, the caution and the wording."""
    if "Caution" in user:
        return "100.5"
    if "FAKE" in user:
        return "99"
    return ("102" if "eport" in user else "101") + (".0" if system == rp.SYSTEM_C2 else "")


def _m5_answer(r):
    return fake(qp.SYSTEM, qp.build_prompt(r.description, r.unit, r.freq, json.loads(r.lookback),
                                           r.text or None))


def _inputs(tmp_path):
    """A small P5-layout inputs table: 2 domains × {test, val} × 3 windows × 4 conditions."""
    rows = []
    for d, freq, unit in (("Economy", "monthly", "million US dollars"),
                          ("Health_US", "weekly", "percent")):
        for split in ("test", "val"):
            for i in range(3):
                x = [100.0 + i + k + 0.5 * (split == "val") for k in range(4)]
                real = "" if i == 0 else f"{d} rose {i}."
                for cond, text in (("real", real), ("none", ""), ("flipped", real and real + " FAKE"),
                                   ("shuffled", f"{d} other report {i}.")):
                    rows.append(dict(domain=d, split=split, sample_id=f"20{10 + i}-01-01",
                                     condition=cond, freq=freq, description=f"{d} series",
                                     unit=unit, lookback=json.dumps(x), text=text,
                                     y_true=104.0 + i, train_mean=100.0, train_std=2.0,
                                     reuse="", probe=False, expected_raw=""))
    df = pd.DataFrame(rows)
    pr = (df.split == "val") & (df.condition == "real") & (df.text != "")
    df.loc[pr, "probe"] = True
    df.loc[pr, "expected_raw"] = [_m5_answer(r) for r in df[pr].itertuples()]
    df.to_csv(tmp_path / "qwen_inputs_p5.csv", index=False)
    return df


def _m5_table(inp):
    """Model 5's answers to every input row, shaped like ``load_all_predictions``."""
    out = inp[["domain", "split", "sample_id", "condition"]].copy()
    raw = [_m5_answer(r) for r in inp.itertuples()]
    out["raw_output"] = raw
    out["y_pred_raw"] = [qp.parse_number(s) for s in raw]
    out["parse_ok"] = True
    out["prompt_sha"] = [qp.prompt_sha(qp.build_prompt(r.description, r.unit, r.freq,
                                                       json.loads(r.lookback), r.text or None))
                         for r in inp.itertuples()]
    out["setup"] = "P5 setup"
    return out


def _run(tmp_path, run, generate, quant=None, stop_after=None, cells=("6", "7", "8"),
         splits=("test", "val")):
    spec = RUN_SPECS[run]
    code = [s for s, t in _cells(run) if t == "code"]
    pick = lambda key: next(c for c in code if c.startswith(key) or key in c[:80])
    cfg = dict(RUN=run, MODEL_ID=spec["model_id"], REVISION=spec["revision"],
               QUANT=quant or spec["quant"][0], RUN_VARIANTS=list(spec["variants"]),
               PROBE=run == "m5prompts", FALLBACK=run == "phi35", MAX_NEW_TOKENS=24,
               SPLITS=list(splits))
    calls = []

    def gen(system, user):
        if stop_after is not None and len(calls) >= stop_after:
            raise ConnectionError("Colab disconnected")
        calls.append((system, user))
        return generate(system, user)

    setup = f"Tesla T4 | torch x | transformers y | {cfg['QUANT']} | {spec['model_id']}@{spec['revision']}"
    ns = {**cfg, "LOCAL_TEST": True, "generate": gen, "SETUP": setup}
    exec(pick("# 3. Paths and inputs").replace("/content", tmp_path.as_posix()), ns)
    ns["meta"] = dict(run=run, model_id=spec["model_id"], revision=spec["revision"],
                      quant=cfg["QUANT"], setup=setup)
    exec(pick("# Generated verbatim from src/models/qwen_prompt.py"), ns)
    exec(pick("# Generated verbatim from src/models/r2_prompts.py"), ns)
    for c in cells:
        exec(pick(f"# {c}."), ns)
    return ns, calls


def _files(tmp_path, run, quant=None):
    q = quant or RUN_SPECS[run]["quant"][0]
    return (tmp_path / f"r2_{run}_{q}_predictions.csv", tmp_path / f"r2_{run}_{q}_run_meta.json")


def test_new_model_run_answers_every_row_test_first_and_loads(tmp_path):
    inp = _inputs(tmp_path)
    ns, calls = _run(tmp_path, "qwen7b", fake)
    pred, meta = _files(tmp_path, "qwen7b")
    res = pd.read_csv(pred, keep_default_na=False)
    assert len(res) == len(inp)
    first_val = (res.split == "val").idxmax()
    assert (res.split.iloc[:first_val] == "test").all() and (res.split.iloc[first_val:] == "val").all()
    m = json.loads(meta.read_text())
    assert m["test_complete"] and m["val_complete"] and m["determinism_mismatches"] == 0
    assert m["smoke_test"][0]["unparseable"] == 0
    df = load_run("qwen7b", pred, meta, tmp_path / "qwen_inputs_p5.csv")
    assert set(df.model) == {"Q7"} and len(df) == len(inp) and df.format_ok.all()


def test_run_resumes_across_sessions_without_duplicates(tmp_path):
    inp = _inputs(tmp_path)
    with pytest.raises(ConnectionError):
        _run(tmp_path, "qwen7b", fake, stop_after=20 + 13)       # 20 smoke-test prompts + 13
    pred, meta = _files(tmp_path, "qwen7b")
    assert 0 < len(pd.read_csv(pred)) < len(inp)
    _run(tmp_path, "qwen7b", fake)                                 # next session
    res = pd.read_csv(pred, keep_default_na=False)
    assert len(res) == len(inp) and not res.duplicated(
        ["domain", "split", "sample_id", "condition", "variant"]).any()


def test_val_is_optional_once_test_is_complete(tmp_path):
    inp = _inputs(tmp_path)
    n_test_unique = len({qp.build_prompt(r.description, r.unit, r.freq, json.loads(r.lookback),
                                         r.text or None) for r in inp[inp.split == "test"].itertuples()})
    with pytest.raises(ConnectionError):                           # quota ends during val
        _run(tmp_path, "qwen7b", fake, stop_after=20 + n_test_unique + 2, cells=("6", "7"))
    pred, meta = _files(tmp_path, "qwen7b")
    assert 0 < (pd.read_csv(pred).split == "val").sum() < (inp.split == "val").sum()
    ns, calls = _run(tmp_path, "qwen7b", fake, splits=("test",))   # next session, val skipped
    assert ns["test_done"] and not ns["val_done"]
    assert len(calls) == ns["meta"]["smoke_test"][-1]["n"] + 20       # smoke + determinism only
    assert json.loads(meta.read_text())["test_complete"]
    df = load_run("qwen7b", pred, meta, tmp_path / "qwen_inputs_p5.csv")
    assert set(df.split) == {"test"}                               # partial val dropped whole


def test_incomplete_test_split_is_refused_by_the_loader(tmp_path):
    _inputs(tmp_path)
    with pytest.raises(ConnectionError):                           # disconnect during test
        _run(tmp_path, "qwen7b", fake, stop_after=20 + 5, cells=("6", "7", "8"))
    pred, meta = _files(tmp_path, "qwen7b")
    assert "test_complete" not in json.loads(meta.read_text())
    with pytest.raises(ValueError, match="incomplete"):
        load_run("qwen7b", pred, meta, tmp_path / "qwen_inputs_p5.csv")


def test_phi_fp16_fallback_rule_fires_on_unparseable_answers(tmp_path):
    _inputs(tmp_path)
    with pytest.raises(RuntimeError, match="QUANT = 'nf4'"):
        _run(tmp_path, "phi35", lambda s, u: "I cannot answer that.", cells=("6",))
    _, meta = _files(tmp_path, "phi35")
    smoke = json.loads(meta.read_text())["smoke_test"][0]
    assert smoke["unparseable"] == smoke["n"] >= 10


def test_phi_fallback_file_loads_as_p35(tmp_path):
    inp = _inputs(tmp_path)
    _run(tmp_path, "phi35", fake, quant="nf4")
    pred, meta = _files(tmp_path, "phi35", "nf4")
    df = load_run("phi35", pred, meta, tmp_path / "qwen_inputs_p5.csv")
    assert set(df.model) == {"P35"} and len(df) == len(inp)


def test_m5_probe_pass_reuses_identical_prompts_and_loader_fills_them(tmp_path):
    inp = _inputs(tmp_path)
    ns, calls = _run(tmp_path, "m5prompts", fake)
    assert ns["REUSE_OK"]
    pred, meta = _files(tmp_path, "m5prompts")
    res = pd.read_csv(pred, keep_default_na=False)
    assert "p5" not in set(res.variant)
    gen_none = res[res.condition == "none"]
    assert set(gen_none.variant) == {"c2"}                       # c1 / sk none prompts = M5's
    df = load_run("m5prompts", pred, meta, tmp_path / "qwen_inputs_p5.csv", m5=_m5_table(inp))
    assert set(df.model) == {"M5-c1", "M5-c2", "M5-sk"}
    assert len(df) == 3 * len(inp)
    reused = df[df.reused_from == "m5"]
    assert set(reused.model) == {"M5-c1", "M5-sk"} and (reused.setup == "P5 setup").all()
    sk = df[df.model == "M5-sk"].set_index(["domain", "split", "sample_id", "condition"])
    assert (sk.loc[sk.text != "", "y_pred"] == 100.5).all()


def test_m5_probe_failure_regenerates_model_5s_own_prompt(tmp_path):
    inp = _inputs(tmp_path)
    other_stack = lambda s, u: fake(s, u).replace("1", "7", 1)
    ns, _ = _run(tmp_path, "m5prompts", other_stack)
    assert not ns["REUSE_OK"]
    pred, meta = _files(tmp_path, "m5prompts")
    res = pd.read_csv(pred, keep_default_na=False)
    assert set(res.variant) == {"c1", "c2", "sk", "p5"} and len(res) == 4 * len(inp)
    df = load_run("m5prompts", pred, meta, tmp_path / "qwen_inputs_p5.csv", m5=_m5_table(inp))
    assert set(df.model) == {"M5-c1", "M5-c2", "M5-sk", "M5-regen"}
    assert (df.reused_from != "m5").all()


def test_loader_refuses_wrong_revision_tampered_prompt_and_mixed_setups(tmp_path):
    _inputs(tmp_path)
    _run(tmp_path, "qwen7b", fake)
    pred, meta = _files(tmp_path, "qwen7b")
    inputs = tmp_path / "qwen_inputs_p5.csv"
    m = json.loads(meta.read_text())
    meta.write_text(json.dumps({**m, "revision": "main"}))
    with pytest.raises(ValueError, match="revision"):
        load_run("qwen7b", pred, meta, inputs)
    meta.write_text(json.dumps(m))
    res = pd.read_csv(pred, keep_default_na=False, dtype={"sample_id": str})
    res.loc[0, "prompt_sha"] = "0" * 16
    res.to_csv(pred, index=False)
    with pytest.raises(ValueError, match="different prompt"):
        load_run("qwen7b", pred, meta, inputs)
    res.loc[0, "setup"] = "another stack"
    res.to_csv(pred, index=False)
    with pytest.raises(ValueError, match="setup"):
        load_run("qwen7b", pred, meta, inputs)


def test_off_scale_answer_is_a_format_failure(tmp_path):
    inp = _inputs(tmp_path)
    _run(tmp_path, "qwen7b", lambda s, u: "0.5" if "rose 2" in u else fake(s, u))
    pred, meta = _files(tmp_path, "qwen7b")
    df = load_run("qwen7b", pred, meta, tmp_path / "qwen_inputs_p5.csv")
    bad = df[df.text.str.contains("rose 2")]
    assert len(bad) and bad.y_pred.isna().all() and (~bad.scale_ok).all()
    assert (bad.y_pred_raw == 0.5).all() and math.isnan(bad.y_pred.iloc[0])
