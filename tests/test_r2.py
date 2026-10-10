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

KAGGLE_NB = "r2_qwen7b_kaggle.ipynb"


def _cells(run):
    name = KAGGLE_NB if run == "qwen7b_kaggle" else NB[run]
    nb = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
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
         splits=("test", "val"), kaggle=False):
    spec = RUN_SPECS[run]
    code = [s for s, t in _cells("qwen7b_kaggle" if kaggle else run) if t == "code"]
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
    ns = {**cfg, "LOCAL_TEST": True, "generate": gen,
          "SETUP": setup + (" | kaggle" if kaggle else "")}
    cell3 = pick("# 3. Paths and inputs")
    exec(cell3.replace("/kaggle", tmp_path.as_posix()) if kaggle
         else cell3.replace("/content", tmp_path.as_posix()), ns)
    ns["meta"] = dict(run=run, model_id=spec["model_id"], revision=spec["revision"],
                      quant=cfg["QUANT"], setup=ns["SETUP"],
                      **({"platform": "kaggle"} if kaggle else {}))
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


# --- Kaggle version of the Qwen-7B run (§10e Amendment 1) ------------------------------------

def _kaggle_inputs(tmp_path, dataset="any-dataset-name/v3"):
    inp = _inputs(tmp_path)
    d = tmp_path / "input" / dataset
    d.mkdir(parents=True)
    (d / "qwen_inputs_p5.csv").write_bytes((tmp_path / "qwen_inputs_p5.csv").read_bytes())
    return inp


def _kfiles(tmp_path):
    w = tmp_path / "working"
    return w / "r2_qwen7b_nf4_kaggle_predictions.csv", w / "r2_qwen7b_nf4_kaggle_run_meta.json"


def test_kaggle_notebook_keeps_the_colab_run_logic_and_changes_only_the_platform():
    colab = {s.split("\n")[0]: s for s, t in _cells("qwen7b") if t == "code"}
    kag = {s.split("\n")[0]: s for s, t in _cells("qwen7b_kaggle") if t == "code"}
    assert len(colab) == len(kag)
    for key in colab:                                  # prompt modules, look, smoke test: identical
        if key.startswith(("# Generated verbatim", "# 4.", "# 6.")):
            assert colab[key] == kag[key], key
    k0 = next(v for k, v in kag.items() if k.startswith("# 0."))
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"' in k0 and "import torch" not in k0
    assert 'PLATFORM = "kaggle"' in k0 and f'REVISION = "{RUN_SPECS["qwen7b"]["revision"]}"' in k0
    assert 'QUANT = "nf4"' in k0
    k1 = next(v for k, v in kag.items() if k.startswith("# 1."))
    assert "device_count() == 1" in k1 and '"T4" in torch.cuda.get_device_name(0)' in k1
    k5 = next(v for k, v in kag.items() if k.startswith("# 5."))
    assert 'kw["torch_dtype"] = torch.float16' in k5 and "{PLATFORM}" in k5
    assert "platform=PLATFORM" in k5 and "input_sha256=INPUT_SHA" in k5
    allsrc = "\n".join(kag.values())
    assert "google.colab" not in allsrc and "files.download" not in allsrc
    assert "/content" not in allsrc
    assert 'glob.glob("/kaggle/input/**/qwen_inputs_p5.csv", recursive=True)' in allsrc
    assert 'OUT_DIR = "/kaggle/working"' in allsrc
    c7, k7 = (next(v for k, v in d.items() if k.startswith("# 7.")) for d in (colab, kag))
    added = [ln for ln in k7.splitlines() if ln not in c7.splitlines()]
    assert added and all("print(" in ln or ln.strip().startswith(("if r.split", 'f"'))
                         for ln in added), added
    removed = [ln for ln in c7.splitlines() if ln not in k7.splitlines()]
    assert not removed, removed                        # main loop: only progress lines added


def test_colab_7b_notebook_pins_fp16_for_unquantised_layers_and_others_are_untouched():
    load = next(s for s, _ in _cells("qwen7b") if "def generate" in s)
    assert 'kw["torch_dtype"] = torch.float16' in load.split("else:")[0]
    for run in ("phi35", "m5prompts"):                 # already run: template unchanged
        load = next(s for s, _ in _cells(run) if "def generate" in s)
        assert load.count('kw["torch_dtype"] = torch.float16') == 1


def test_kaggle_run_finds_input_anywhere_writes_own_file_and_loads(tmp_path, capsys):
    inp = _kaggle_inputs(tmp_path)
    ns, _ = _run(tmp_path, "qwen7b", fake, kaggle=True)
    assert ns["INPUTS"].replace("\\", "/").endswith("any-dataset-name/v3/qwen_inputs_p5.csv")
    pred, meta = _kfiles(tmp_path)
    m = json.loads(meta.read_text())
    assert m["platform"] == "kaggle" and m["test_complete"] and m["determinism_mismatches"] == 0
    log = capsys.readouterr().out
    assert log.count("===== TEST SPLIT DONE") == 1 and "===== MAIN LOOP" in log
    df = load_run("qwen7b", pred, meta, tmp_path / "qwen_inputs_p5.csv")
    assert set(df.model) == {"Q7"} and len(df) == len(inp)
    assert set(df.setup) == {ns["SETUP"]} and ns["SETUP"].endswith("| kaggle")


def test_kaggle_run_refuses_a_different_input_file(tmp_path):
    _kaggle_inputs(tmp_path)
    cell3 = next(s for s, _ in _cells("qwen7b_kaggle") if s.startswith("# 3."))
    with pytest.raises(AssertionError, match="not the exported file"):     # no LOCAL_TEST bypass
        exec(cell3.replace("/kaggle", tmp_path.as_posix()), {"RUN": "qwen7b", "QUANT": "nf4"})
    other = tmp_path / "input" / "second-dataset"
    other.mkdir()
    (other / "qwen_inputs_p5.csv").write_text("domain\nEconomy\n")
    with pytest.raises(AssertionError, match="several different"):
        exec(cell3.replace("/kaggle", tmp_path.as_posix()),
             {"RUN": "qwen7b", "QUANT": "nf4", "LOCAL_TEST": True})


def test_kaggle_run_starts_fresh_and_never_mixes_with_a_partial_colab_file(tmp_path):
    from src.models.r2_forecasters import files_for
    inp = _kaggle_inputs(tmp_path)
    with pytest.raises(ConnectionError):               # a partial Colab 7B run
        _run(tmp_path, "qwen7b", fake, stop_after=20 + 7, cells=("6", "7"))
    colab_pred, colab_meta = _files(tmp_path, "qwen7b")
    n_colab = len(pd.read_csv(colab_pred))
    assert 0 < n_colab < len(inp)
    w = tmp_path / "working"
    w.mkdir()
    for f in (colab_pred, colab_meta):                 # even sitting next to the Kaggle output
        (w / f.name).write_bytes(f.read_bytes())
    ns, _ = _run(tmp_path, "qwen7b", fake, kaggle=True)
    pred, meta = _kfiles(tmp_path)
    assert len(pd.read_csv(pred)) == len(inp)          # every row from the Kaggle setup
    assert len(pd.read_csv(w / colab_pred.name)) == n_colab        # Colab file untouched
    assert files_for("qwen7b", w) == (pred, meta)      # the complete Kaggle file is used
    df = load_run("qwen7b", *files_for("qwen7b", w), inputs=tmp_path / "qwen_inputs_p5.csv")
    assert set(df.setup) == {ns["SETUP"]}


def test_loader_refuses_two_complete_7b_files_and_a_mislabelled_platform(tmp_path):
    from src.models.r2_forecasters import files_for
    _kaggle_inputs(tmp_path)
    _run(tmp_path, "qwen7b", fake)                     # complete Colab run in tmp_path
    _run(tmp_path, "qwen7b", fake, kaggle=True)        # complete Kaggle run in tmp_path/working
    w = tmp_path / "working"
    for f in _files(tmp_path, "qwen7b"):
        (w / f.name).write_bytes(f.read_bytes())
    with pytest.raises(ValueError, match="several complete"):
        files_for("qwen7b", w)
    pred, meta = _kfiles(tmp_path)
    m = json.loads(meta.read_text())
    del m["platform"]                                  # a Kaggle-named file claiming Colab
    meta.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="platform"):
        load_run("qwen7b", pred, meta, tmp_path / "qwen_inputs_p5.csv")
