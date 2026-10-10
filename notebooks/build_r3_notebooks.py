"""Build R3's three Kaggle notebooks (pre-registration PROJECT_STATUS §10f).

* ``r3_m5_kaggle.ipynb``     — Model 5 (Qwen2.5-1.5B @ M5's revision, fp16), prompt variants p5
  (M5's own) and sk (R2's skeptical prompt), P5's 40-prompt probe first (recorded only)
* ``r3_qwen7b_kaggle.ipynb`` — Qwen2.5-7B (R2's revision, 4-bit NF4, fp16 compute), p5 and sk,
  R2's 20-prompt smoke test and stop rule
* ``r3_judge_kaggle.ipynb``  — TextGuard v1's direction judge (P6's prompt, Qwen2.5-1.5B @ M5's
  revision, scored answers) with P5's probe

All three read **one** input file, ``r3_inputs.csv`` (``python -m src.models.export_r3_inputs``),
found anywhere under ``/kaggle/input`` and refused unless its sha256 is the export's (embedded
below at build time). Kaggle, not Colab: the background "Save & Run All" survives a laptop that
sleeps, and R2's Q7 run showed the pattern works (§10e Amendment 1).

Why the forecasters regenerate *everything* (real and none too): Kaggle's software stack is not
P5's Colab stack, so reusing P5's answers would mix two setups inside one comparison — a stack
change would be measured as attack effect (§2). The probe is therefore recorded, not acted on.

Protocol identical to R2 (§10e): greedy, batch size 1, 24 new tokens, M5's prompt modules pasted
verbatim (``qwen_prompt.py`` then ``r2_prompts.py``), M5's parser, test rows first, appends every
10 rows, identical prompts answered once, one setup per file, 20-answer determinism re-check.
The judge notebook is P6's notebook (``build_p6_judge_notebook.CELLS``) with only the platform
plumbing replaced by asserted, exact edits — its scoring code is P6's, byte for byte.

After changing a prompt module or the inputs: ``python -m src.models.export_r3_inputs``, then
``python notebooks/build_r3_notebooks.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT))
import build_p6_judge_notebook as p6  # noqa: E402
from build_p2_qwen_notebook import PROMPT_SRC, code, md  # noqa: E402
from build_r2_notebooks import R2_PROMPT_SRC, _edit, _src  # noqa: E402
from src.models.export_r3_inputs import R3_INPUTS, sha256  # noqa: E402
from src.models.r2_forecasters import RUN_SPECS  # noqa: E402

INPUT_NAME = "r3_inputs.csv"
RUNS = {
    "m5": dict(file="r3_m5_kaggle.ipynb", title="R3 — Model 5 (Qwen2.5-1.5B-Instruct, fp16)",
               model_id=RUN_SPECS["m5prompts"]["model_id"],
               revision=RUN_SPECS["m5prompts"]["revision"], quant="fp16", probe=True, smoke=False),
    "qwen7b": dict(file="r3_qwen7b_kaggle.ipynb",
                   title="R3 — Qwen2.5-7B-Instruct (4-bit NF4)",
                   model_id=RUN_SPECS["qwen7b"]["model_id"],
                   revision=RUN_SPECS["qwen7b"]["revision"], quant="nf4", probe=False, smoke=True),
}
VARIANTS = ["p5", "sk"]
JUDGE_FILE = "r3_judge_kaggle.ipynb"


def input_sha() -> str:
    if not R3_INPUTS.exists():
        raise FileNotFoundError("run python -m src.models.export_r3_inputs first")
    return sha256(R3_INPUTS)


KAGGLE_SETTINGS = """
* **Settings (right panel → Session options):** Accelerator **GPU T4 ×2** (only cuda:0 is used),
  **Internet on**. **+ Add Input** → your dataset holding `r3_inputs.csv`.
* Then **Save Version → Save & Run All (Commit)**. The laptop can sleep; the log prints progress.
* Outputs land in `/kaggle/working` → the finished version's **Output** tab.
* `flipped`, `exaggerated`, `a1_selective`, `a2_forward`, `a3_magnitude` rows are **synthetic
  text for defence research** — not real reports.
"""


def forecaster_intro(cfg: dict) -> str:
    check = ("**Probe (cell 6):** P5's 40 prompts are re-run and compared with P2.4's answers — "
             "recorded only (every condition is regenerated here anyway, so nothing depends on it)."
             if cfg["probe"] else
             "**Smoke test (cell 6):** 20 test answers; ≥ 10 unparseable → the run stops "
             "(tell Claude, change nothing).")
    return f"""
# {cfg['title']} — Kaggle

TextGuard R3 (pre-registration PROJECT_STATUS §10f): adaptive attacks on TextGuard v1. Forecasts
every **test, then val** window under real, none, P3's flipped / exaggerated and R3's
a1_selective / a2_forward / a3_magnitude texts, with M5's prompt (`p5`) and the skeptical prompt
(`sk`; its no-report prompt is M5's, so it is not asked twice). Horizon 1.

* Input: `{INPUT_NAME}` in any attached dataset; its sha256 must match the export.
* Output: `r3_{{RUN}}_{{QUANT}}_kaggle_predictions.csv` + `…_run_meta.json`, appended every 10 rows.
* {check}
{KAGGLE_SETTINGS}"""


CONFIG = """
# 0. Run configuration (pre-registered, §10f) — do not change
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # one GPU (cuda:0) even on T4 x2; before torch is imported
PLATFORM = "kaggle"
RUN = "{run}"
MODEL_ID = "{model_id}"
REVISION = "{revision}"           # pinned Hub commit: loaded and recorded exactly
QUANT = "{quant}"                 # "fp16" or "nf4" (4-bit bitsandbytes NF4, fp16 compute)
RUN_VARIANTS = {variants}         # M5's prompt and the skeptical prompt (R2's sk)
PROBE = {probe}                   # Model 5: P5's reproducibility probe (recorded, gates nothing)
SMOKE = {smoke}                   # Qwen-7B: R2's smoke test and stop rule
MAX_NEW_TOKENS = 24
SPLITS = ["test", "val"]          # test first; val is descriptive in R3 (and R4's dev data)
INPUT_SHA256 = "{sha}"
if QUANT == "nf4":
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=True)
"""

GPU_CELL = """
# 1. GPU check — exactly one visible T4
import torch, transformers
assert torch.cuda.is_available(), "No GPU: Settings > Accelerator > GPU T4 x2, then Save & Run All again"
assert torch.cuda.device_count() == 1, "exactly one GPU must be visible: restart the session and run from cell 0"
assert "T4" in torch.cuda.get_device_name(0), (
    f"pre-registered GPU is a T4, got {torch.cuda.get_device_name(0)}: Settings > Accelerator > GPU T4 x2")
print(torch.cuda.get_device_name(0), "| torch", torch.__version__, "| transformers", transformers.__version__)
"""

PLATFORM_CELL = """
# 2. Platform: Kaggle. No Google Drive and no browser download: outputs are written to
#    /kaggle/working and saved as this notebook's Output when the committed run finishes.
import subprocess
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip(), flush=True)
print("visible to torch:", torch.cuda.device_count(), "GPU(s) ->", torch.cuda.get_device_name(0), flush=True)
"""

FIND_INPUT = """import glob, hashlib
OUT_DIR = "/kaggle/working"
os.makedirs(OUT_DIR, exist_ok=True)
found = sorted(glob.glob("/kaggle/input/**/r3_inputs.csv", recursive=True))
assert found, "r3_inputs.csv not found under /kaggle/input: + Add Input > your dataset, then run again"
shas = {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in found}
assert len(set(shas.values())) == 1, f"several different r3_inputs.csv files attached: {shas}"
INPUTS = found[0]
INPUT_SHA = shas[INPUTS]
assert INPUT_SHA == INPUT_SHA256 or "LOCAL_TEST" in globals(), (
    f"{INPUTS} has sha256 {INPUT_SHA[:16]}, expected {INPUT_SHA256[:16]}: not the exported file")"""

PATHS_CELL = """
# 3. Paths and inputs (one CSV for all three R3 notebooks; forecast + probe rows used here)
import os, json, math, time, datetime
import pandas as pd

__FIND__
TAG = f"r3_{RUN}_{QUANT}_kaggle"
OUT = f"{OUT_DIR}/{TAG}_predictions.csv"
META = f"{OUT_DIR}/{TAG}_run_meta.json"

df = pd.read_csv(INPUTS, keep_default_na=False, dtype=str)    # empty text stays ""
inp = df[df.row_type == "forecast"].copy()
probes = df[df.row_type == "probe"].copy()
inp["y_true"] = inp["y_true"].astype(float)
assert len(inp) == 3451 or "LOCAL_TEST" in globals(), f"expected 3451 forecast rows, got {len(inp)}"
print(f"{len(inp)} forecast rows (windows x 7 conditions) + {len(probes)} probe rows from {INPUTS}", flush=True)
print(inp.groupby(["split", "condition"]).size().unstack().to_string())
""".replace("__FIND__", FIND_INPUT)

LOOK_CELL = """
# 4. Look at the R3 prompts before spending GPU time (one per attack, M5's prompt and sk)
for cond in ("a1_selective", "a2_forward", "a3_magnitude"):
    r = inp[(inp.condition == cond) & (inp.attacked == "True")].iloc[0]
    for v in RUN_VARIANTS:
        print(f"===== {cond} | variant {v} | system: {system_for(v)}")
        print(build_variant_prompt(v, r.description, r.unit, r.freq, json.loads(r.lookback), r.text or None), "\\n")
"""

LOAD_CELL = """
# 5. Load the pinned revision; record the setup. One output file = one software setup.
from transformers import AutoTokenizer, AutoModelForCausalLM
kw = dict(revision=REVISION, device_map="cuda", torch_dtype=torch.float16)   # T4: no bf16
if QUANT == "nf4":
    import bitsandbytes
    from transformers import BitsAndBytesConfig
    kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.float16,
                                                   bnb_4bit_use_double_quant=False)
    QUANT_DESC = f"nf4 bitsandbytes {bitsandbytes.__version__}"
else:
    QUANT_DESC = "fp16"
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw).eval()
loaded = getattr(model.config, "_commit_hash", None)
assert loaded in (None, REVISION), f"loaded {loaded}, pinned {REVISION}"

@torch.no_grad()
def generate(system: str, user_prompt: str) -> str:
    \"\"\"Greedy, batch size 1 (no padding), so the same prompt always gives the same answer.\"\"\"
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(ids):   # transformers >= 5 returns a BatchEncoding, not a tensor
        ids = ids["input_ids"]
    ids = ids.to("cuda")
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAX_NEW_TOKENS,
                         do_sample=False, temperature=None, top_p=None, top_k=None,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

SETUP = (f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | transformers "
         f"{transformers.__version__} | {QUANT_DESC} | {MODEL_ID}@{REVISION} | {PLATFORM}")
meta = dict(run=RUN, model_id=MODEL_ID, revision=REVISION, quant=QUANT, quant_desc=QUANT_DESC,
            variants=RUN_VARIANTS, decoding="greedy", batch_size=1, max_new_tokens=MAX_NEW_TOKENS,
            gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
            transformers=transformers.__version__, setup=SETUP, platform=PLATFORM,
            inputs=os.path.basename(INPUTS), input_path=INPUTS, input_sha256=INPUT_SHA,
            gpu_count_visible=torch.cuda.device_count(),
            started=datetime.datetime.now(datetime.timezone.utc).isoformat())
json.dump(meta, open(META, "w"), indent=1)
print("SETUP:", SETUP, flush=True)
"""

CHECK_CELL = """
# 6. Checks before the main loop. Model 5: P5's probe (recorded only). Qwen-7B: smoke test + stop rule.
def save_meta():
    json.dump(meta, open(META, "w"), indent=1)

if PROBE:
    mismatch = []
    for r in probes.itertuples():
        raw = generate(SYSTEM, build_prompt(r.description, r.unit, r.freq, json.loads(r.lookback), r.text or None))
        if raw != r.expected_raw:
            mismatch.append((r.domain, r.sample_id, r.condition, r.expected_raw, raw))
    print(f"probe: {len(probes) - len(mismatch)}/{len(probes)} identical to P2.4 ->",
          "same setup class as P5" if not mismatch else "different setup class from P5 (recorded; "
          "every R3 condition is regenerated in this run, so nothing is mixed)", flush=True)
    for m in mismatch[:10]:
        print("  ", m)
    meta.update(probe_n=len(probes), probe_mismatches=len(mismatch), probe_ok=not mismatch)
if SMOKE:
    first = inp[(inp.split == "test") & (inp.text != "")].head(20)
    bad = 0
    for r in first.itertuples():
        raw = generate(SYSTEM, build_prompt(r.description, r.unit, r.freq, json.loads(r.lookback), r.text or None))
        bad += math.isnan(parse_number(raw, r.unit))
        print(f"  {r.domain:10s} y_true {r.y_true:<12g} answer {raw!r}")
    print(f"smoke test: {bad}/{len(first)} unparseable", flush=True)
    meta["smoke_test"] = dict(n=len(first), unparseable=bad)
    if bad >= 10:
        save_meta()
        raise RuntimeError("Too many unparseable answers: stop here and tell Claude (do not change the prompt).")
save_meta()
"""

MAIN_CELL = """
# 7. Main loop — test rows first, then val. Appends every 10 rows; identical prompts answered once.
COLS = ["domain", "split", "sample_id", "condition", "variant", "y_true", "y_pred", "parse_ok",
        "raw_output", "prompt_sha", "setup"]
parts = []
for v in RUN_VARIANTS:
    p = inp if v == "p5" else inp[inp.condition != "none"]   # sk's no-report prompt is M5's: asked once
    p = p.assign(variant=v)
    p["user"] = [build_variant_prompt(v, r.description, r.unit, r.freq, json.loads(r.lookback), r.text or None)
                 for r in p.itertuples()]
    parts.append(p)
rows = pd.concat(parts, ignore_index=True)
rows = rows[rows.split.isin(SPLITS)].copy()
rows["order"] = rows.split.map({"test": 0, "val": 1})
rows = rows.sort_values("order", kind="stable")
key = lambda r: (r.domain, r.split, r.sample_id, r.condition, r.variant)

done = set()
if os.path.exists(OUT):
    prev = pd.read_csv(OUT, keep_default_na=False, dtype={"sample_id": str, "raw_output": str})
    other = set(prev.setup) - {SETUP}
    assert not other, f"{OUT} holds rows from another setup {other}; one file = one setup. Tell Claude."
    done = set(zip(prev.domain, prev.split, prev.sample_id, prev.condition, prev.variant))
    print(f"resuming: {len(done)} rows already saved")
else:
    pd.DataFrame(columns=COLS).to_csv(OUT, index=False)
todo = [r for r in rows.itertuples() if key(r) not in done]
n_test = int((rows.split == "test").sum())
print(f"===== MAIN LOOP | {SETUP} =====", flush=True)
print(f"{len(rows)} rows ({n_test} test, {len(rows) - n_test} val; "
      f"{rows[['variant', 'user']].drop_duplicates().shape[0]} distinct prompts); {len(todo)} still to do", flush=True)
meta.update(rows_expected=len(rows), rows_expected_test=n_test)
save_meta()
cache, buf, n_fail, t0 = {}, [], 0, time.time()
for k, r in enumerate(todo, 1):
    system = system_for(r.variant)
    if (system, r.user) not in cache:
        cache[(system, r.user)] = generate(system, r.user)
    raw = cache[(system, r.user)]
    y = parse_number(raw, r.unit)
    ok = not math.isnan(y)
    n_fail += not ok
    buf.append(dict(domain=r.domain, split=r.split, sample_id=r.sample_id, condition=r.condition,
                    variant=r.variant, y_true=r.y_true, y_pred=y, parse_ok=ok, raw_output=raw,
                    prompt_sha=variant_prompt_sha(r.variant, r.user), setup=SETUP))
    if len(buf) >= 10 or k == len(todo):
        pd.DataFrame(buf, columns=COLS).to_csv(OUT, mode="a", header=False, index=False)
        buf = []
    if r.split == "test" and (k == len(todo) or todo[k].split != "test"):
        print(f"===== TEST SPLIT DONE (row {k}/{len(todo)}, {(time.time() - t0) / 60:.1f} min) - val follows =====",
              flush=True)
    if k % 100 == 0 or k == len(todo):
        el = time.time() - t0
        print(f"{k}/{len(todo)} [{r.split}]  parse failures so far: {n_fail}  {el / 60:.1f} min, "
              f"~{el / k * (len(todo) - k) / 60:.1f} min left", flush=True)
"""

FINISH_CELL = """
# 8. Completeness (test required) + determinism re-check of the first 20 answers
res = pd.read_csv(OUT, keep_default_na=False, dtype={"sample_id": str, "raw_output": str})
res["parse_ok"] = res["parse_ok"].astype(str) == "True"
assert not res.duplicated(["domain", "split", "sample_id", "condition", "variant"]).any()
assert set(res.setup) == {SETUP}, "rows from more than one setup"
got = set(zip(res.domain, res.split, res.sample_id, res.condition, res.variant))
miss = rows[[key(r) not in got for r in rows.itertuples()]]
test_done = not (miss.split == "test").any()
val_done = "val" in SPLITS and not (miss.split == "val").any()
print(f"TEST {'COMPLETE' if test_done else 'INCOMPLETE'} | VAL {'complete' if val_done else 'incomplete'}"
      f" | {len(res)} rows saved, {len(miss)} missing", flush=True)
assert test_done, "test split incomplete: Save & Run All again (a committed run starts fresh)"
chk = res.head(20).merge(rows[["domain", "split", "sample_id", "condition", "variant", "user"]],
                         on=["domain", "split", "sample_id", "condition", "variant"])
same = [generate(system_for(r.variant), r.user) == r.raw_output for r in chk.itertuples()]
print(f"determinism: {sum(same)}/{len(same)} re-answered identically", flush=True)
print("parse failures (count):")
print((~res.parse_ok).groupby([res.variant, res.condition, res.domain]).sum().unstack().to_string())
meta.update(finished=datetime.datetime.now(datetime.timezone.utc).isoformat(), rows=len(res),
            test_complete=test_done, val_complete=val_done, determinism_n=len(same),
            determinism_mismatches=len(same) - sum(same), parse_failures=int((~res.parse_ok).sum()))
save_meta()
"""

OUTPUT_CELL = """
# 9. Outputs — in /kaggle/working; after the committed run: the version's Output tab -> download both
for f in (OUT, META):
    print(f"{f}  ({os.path.getsize(f) / 1024:.0f} KB)", flush=True)
print("RUN FINISHED:", "TEST COMPLETE" if test_done else "TEST INCOMPLETE", "| determinism",
      f"{meta['determinism_n'] - meta['determinism_mismatches']}/{meta['determinism_n']}", flush=True)
"""


def build_forecaster(run: str, sha: str) -> dict:
    cfg = RUNS[run]
    cells = [
        md(forecaster_intro(cfg)),
        code(CONFIG.format(run=run, model_id=cfg["model_id"], revision=cfg["revision"],
                           quant=cfg["quant"], variants=json.dumps(VARIANTS), probe=cfg["probe"],
                           smoke=cfg["smoke"], sha=sha)),
        code(GPU_CELL), code(PLATFORM_CELL), code(PATHS_CELL),
        md("## Prompt and parser — generated from `src/models/qwen_prompt.py` and "
           "`src/models/r2_prompts.py` (do not edit here)"),
        code("# Generated verbatim from src/models/qwen_prompt.py\n"
             + PROMPT_SRC.read_text(encoding="utf-8")),
        code("# Generated verbatim from src/models/r2_prompts.py\n"
             + R2_PROMPT_SRC.read_text(encoding="utf-8")),
        code(LOOK_CELL), code(LOAD_CELL), code(CHECK_CELL), code(MAIN_CELL), code(FINISH_CELL),
        code(OUTPUT_CELL),
    ]
    return _notebook(cells)


# ---- the judge: P6's notebook, platform plumbing replaced ----------------------------------------

JUDGE_INTRO = f"""
# R3 — TextGuard v1's direction judge (Qwen2.5-1.5B-Instruct) — Kaggle

TextGuard R3 (PROJECT_STATUS §10f). P6's judge, unchanged — same prompt (`judge_prompt.py`), same
revision, scored answers — on every distinct text of real, flipped, exaggerated, a1_selective,
a2_forward and a3_magnitude (val + test; Health_US asks level and change). Real texts are judged
again so that genuine and fake gates in R3 come from one software setup. P5's 40-prompt probe is
run first and recorded.

* Input: `{INPUT_NAME}` (judge + probe rows) in any attached dataset; sha256 checked.
* Output: `r3_judge_kaggle_predictions.csv` + `r3_judge_kaggle_run_meta.json`, appended every 20.
{KAGGLE_SETTINGS}"""

JUDGE_CONFIG = """
# 0. Run configuration (pre-registered, §10f) — do not change
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # one GPU (cuda:0) even on T4 x2; before torch is imported
PLATFORM = "kaggle"
INPUT_SHA256 = "{sha}"
"""

P6_PATHS = """OUT_DIR = "/content/drive/MyDrive/textguard" if os.path.isdir("/content/drive/MyDrive") else "/content"
os.makedirs(OUT_DIR, exist_ok=True)
INPUTS = next((p for p in ("/content/judge_inputs.csv", f"{OUT_DIR}/judge_inputs.csv") if os.path.exists(p)), None)
assert INPUTS, "Upload judge_inputs.csv to /content (Files panel on the left) or to MyDrive/textguard/"
OUT = f"{OUT_DIR}/p6_judge_predictions.csv"
META = f"{OUT_DIR}/p6_judge_run_meta.json\""""

JUDGE_PATHS = FIND_INPUT + """
OUT = f"{OUT_DIR}/r3_judge_kaggle_predictions.csv"
META = f"{OUT_DIR}/r3_judge_kaggle_run_meta.json\""""


def build_judge(sha: str) -> dict:
    cells = [c for c in p6.CELLS]
    sub = {"__PROMPT_MODULE__": code("# Generated verbatim from src/models/qwen_prompt.py\n"
                                     + PROMPT_SRC.read_text(encoding="utf-8")),
           "__JUDGE_MODULE__": code("# Generated verbatim from src/textguard/judge_prompt.py\n"
                                    + p6.JUDGE_SRC.read_text(encoding="utf-8")),
           "__LOAD_CELL__": code(_p6_load_cell())}
    cells = [sub[c] if isinstance(c, str) else c for c in cells]
    cells[0] = md(JUDGE_INTRO)
    cells.insert(1, code(JUDGE_CONFIG.format(sha=sha)))
    i1 = next(i for i, c in enumerate(cells) if _src(c).startswith("# 1."))
    cells[i1] = code(GPU_CELL)
    i2 = next(i for i, c in enumerate(cells) if _src(c).startswith("# 2."))
    cells[i2] = code(PLATFORM_CELL)
    cells = _edit(cells, "# 3.", P6_PATHS, JUDGE_PATHS)
    cells = _edit(cells, "# 3.", "# 3. Paths and inputs",
                  "# 3. Paths and inputs (one CSV for all three R3 notebooks; judge + probe rows used here)")
    cells = _edit(cells, "# 5.",
                  "            transformers=transformers.__version__, inputs=os.path.basename(INPUTS),",
                  "            transformers=transformers.__version__, inputs=os.path.basename(INPUTS),\n"
                  "            platform=PLATFORM, input_path=INPUTS, input_sha256=INPUT_SHA,\n"
                  "            gpu_count_visible=torch.cuda.device_count(),")
    cells = _edit(cells, "# 5.", "f\"{MODEL_ID}@{REVISION}\")",
                  "f\"{MODEL_ID}@{REVISION} | {PLATFORM}\")")
    cells = _edit(cells, "# 5.", "print(meta)", 'print("SETUP:", SETUP, flush=True)')
    cells = _edit(cells, "# 6.", 'print("setup identical to P5\'s:", SETUP == P5_SETUP, "|", SETUP)',
                  'print("setup:", SETUP, "| P5 (Colab):", P5_SETUP, flush=True)')
    cells = _edit(cells, "# 7.", 'print(f"{len(todo)} questions to answer")',
                  'print(f"===== MAIN LOOP | {SETUP} =====", flush=True)\n'
                  'print(f"{len(todo)} questions to answer", flush=True)')
    i9 = next(i for i, c in enumerate(cells) if _src(c).startswith("# 9."))
    cells[i9] = code("""
# 9. Outputs — in /kaggle/working; after the committed run: the version's Output tab -> download both
for f in (OUT, META):
    print(f"{f}  ({os.path.getsize(f) / 1024:.0f} KB)", flush=True)
print("RUN FINISHED:", len(res), "answers | probe", f"{meta['probe_n'] - meta['probe_mismatches']}/{meta['probe_n']}",
      flush=True)
""")
    return _notebook(cells)


def _p6_load_cell() -> str:
    p5 = json.loads(p6.P5_META.read_text(encoding="utf-8"))
    return (p6.LOAD_CELL.replace("__REVISION__", p5["revision"])
            .replace("__P5_SETUP__", p5["setup"]))


def _notebook(cells: list) -> dict:
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python"},
                         "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True,
                                    "isInternetEnabled": True}},
            "nbformat": 4, "nbformat_minor": 4}


def build_all() -> dict[str, dict]:
    sha = input_sha()
    out = {cfg["file"]: build_forecaster(run, sha) for run, cfg in RUNS.items()}
    out[JUDGE_FILE] = build_judge(sha)
    return out


if __name__ == "__main__":
    for name, nb in build_all().items():
        path = ROOT / "notebooks" / name
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {path}")
