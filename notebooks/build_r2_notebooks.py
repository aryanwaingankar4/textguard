"""Build the three R2 Colab notebooks (pre-registration PROJECT_STATUS §10e).

* ``r2_qwen7b.ipynb``    — Qwen2.5-7B-Instruct, **4-bit NF4** (R2-a, scale), M5's prompt
* ``r2_phi35.ipynb``     — Phi-3.5-mini-instruct, fp16 (R2-b, family), M5's prompt
* ``r2_qwen7b_kaggle.ipynb`` — the same 7B run on Kaggle (§10e Amendment 1), built from the
  Colab cells by asserted replacements (``build_kaggle``)
* ``r2_m5_prompts.ipynb`` — Model 5 (Qwen2.5-1.5B, M5's revision) with the report-first (c1),
  reworded (c2) and skeptical (sk) prompts (R2-c, R2-d)

All three read P5's input table ``qwen_inputs_p5.csv`` (every val + test window × the 7
conditions, each row with its own text; ``python -m src.models.export_qwen_inputs --p5``) — the
same windows and texts M5 answered, so nothing new is exported.

Why one template: the run protocol must be identical across models (§10e) — greedy, batch size
1, 24 new tokens, M5's parser, test rows first, crash-safe appends every 10 rows, resumable
across Colab sessions (the free T4 has a daily limit), one setup per file, a determinism re-check.
Only the config cell differs. The prompt modules are pasted in verbatim (``qwen_prompt.py``,
then ``r2_prompts.py``) and ``tests/test_r2.py`` fails if a notebook drifts from them.

Model 5 variants additionally run P5's 40-prompt reproducibility probe: 40/40 → M5's existing
answers are reused for byte-identical prompts; any mismatch → the run also regenerates M5's
own prompt (variant ``p5``) for every row, so every variant-vs-M5 comparison stays within one
software setup (§2, §10e).

After changing a prompt module: ``python notebooks/build_r2_notebooks.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT))
from build_p2_qwen_notebook import PROMPT_SRC, code, md  # noqa: E402  (same cell helpers)
from src.models.r2_forecasters import RUN_SPECS  # noqa: E402  (pins: one source of truth)

R2_PROMPT_SRC = ROOT / "src" / "models" / "r2_prompts.py"

# Pinned in §10e (src/models/r2_forecasters.RUN_SPECS). The notebook loads exactly these
# commits and records them.
RUNS = {
    "qwen7b": dict(file="r2_qwen7b.ipynb", title="R2-a — Qwen2.5-7B-Instruct (4-bit NF4)",
                   model_id=RUN_SPECS["qwen7b"]["model_id"],
                   revision=RUN_SPECS["qwen7b"]["revision"],
                   quant="nf4", variants=["p5"], probe=False, fallback=False),
    "phi35": dict(file="r2_phi35.ipynb", title="R2-b — Phi-3.5-mini-instruct (fp16)",
                  model_id=RUN_SPECS["phi35"]["model_id"],
                  revision=RUN_SPECS["phi35"]["revision"],
                  quant="fp16", variants=["p5"], probe=False, fallback=True),
    "m5prompts": dict(file="r2_m5_prompts.ipynb",
                      title="R2-c / R2-d — Model 5 with reworded and skeptical prompts",
                      model_id=RUN_SPECS["m5prompts"]["model_id"],
                      revision=RUN_SPECS["m5prompts"]["revision"],
                      quant="fp16", variants=["c1", "c2", "sk"], probe=True, fallback=False),
}


def intro(cfg: dict) -> str:
    probe = ("""
* **Reproducibility probe (cell 6):** P5's 40 val prompts are re-run first. 40/40 identical →
  prompts byte-identical to M5's (the `none` rows of c1 and sk) are skipped and filled from P5
  locally. Any mismatch → M5's own prompt is regenerated for every row too (~3,450 extra prompts,
  ~15 min), so variant-vs-M5 comparisons never mix two software setups.
""" if cfg["probe"] else "")
    fallback = ("""
* **fp16 check (cell 6):** if ≥ 10 of the first 20 test answers are unparseable, the
  pre-registered fallback applies: set `QUANT = "nf4"` in cell 0, **Runtime → Restart session**,
  **Run all** (a separate output file is used).
""" if cfg["fallback"] else "")
    return f"""
# {cfg['title']}

TextGuard R2 (pre-registration PROJECT_STATUS §10e). Forecasts every **test, then val** window
under all 7 text conditions (real, none, shuffled, cross_domain, time_shifted, flipped,
exaggerated), horizon 1. `flipped` / `exaggerated` rows are **synthetic text for defence
research** — not real reports.

* Input: `qwen_inputs_p5.csv` (from `python -m src.models.export_qwen_inputs --p5`), uploaded to
  `/content` — or placed in `MyDrive/textguard/` (recommended).
* Output: `r2_<run>_predictions.csv` + `r2_<run>_run_meta.json`, appended **every 10 prompts**
  to Drive (if mounted). **Re-running the notebook resumes** where it stopped — also in a new
  Colab session on another day. Test rows come first: once the test split is done (cell 8 says
  so) the run is usable; val is optional — if the GPU quota runs out during val, set
  `SPLITS = ["test"]` in cell 0 and Run all in the next session to finish (cell 8).
* Greedy, batch size 1, 24 new tokens; prompt and parser generated from
  `src/models/qwen_prompt.py` + `src/models/r2_prompts.py` — do not edit them here.
{probe}{fallback}
Runtime: **Runtime → Change runtime type → T4 GPU**. Then **Runtime → Run all**.
"""


CONFIG = """
# 0. Run configuration (pre-registered, §10e) — do not change except the Phi fp16 fallback
RUN = "{run}"
MODEL_ID = "{model_id}"
REVISION = "{revision}"           # pinned Hub commit: loaded and recorded exactly
QUANT = "{quant}"                 # "fp16" or "nf4" (4-bit bitsandbytes NF4, fp16 compute)
RUN_VARIANTS = {variants}         # prompt variants answered by this run
PROBE = {probe}                   # Model 5 only: reproducibility probe against P5
FALLBACK = {fallback}             # Phi-3.5 only: fp16 -> nf4 fallback rule
MAX_NEW_TOKENS = 24
SPLITS = ["test", "val"]          # test first; set to ["test"] to skip val (optional, §10e)
if QUANT == "nf4":
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=True)
"""

CELLS = [
    "__INTRO__",
    "__CONFIG__",
    code("""
# 1. GPU check — fails fast on a CPU runtime
import torch, transformers
assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > T4 GPU, then Run all again"
print(torch.cuda.get_device_name(0), "| torch", torch.__version__, "| transformers", transformers.__version__)
"""),
    code("""
# 2. Google Drive (recommended): results survive a disconnect and a new session. Set False to skip.
USE_DRIVE = True
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
"""),
    code("""
# 3. Paths and inputs
import os, json, math, time, datetime
import pandas as pd

OUT_DIR = "/content/drive/MyDrive/textguard" if os.path.isdir("/content/drive/MyDrive") else "/content"
os.makedirs(OUT_DIR, exist_ok=True)
INPUTS = next((p for p in ("/content/qwen_inputs_p5.csv", f"{OUT_DIR}/qwen_inputs_p5.csv") if os.path.exists(p)), None)
assert INPUTS, "Upload qwen_inputs_p5.csv to /content (Files panel on the left) or to MyDrive/textguard/"
TAG = f"r2_{RUN}_{QUANT}"
OUT = f"{OUT_DIR}/{TAG}_predictions.csv"
META = f"{OUT_DIR}/{TAG}_run_meta.json"

inp = pd.read_csv(INPUTS, keep_default_na=False, dtype={"sample_id": str, "text": str, "raw_output": str, "expected_raw": str})   # empty text stays ""
inp["probe"] = inp["probe"].astype(str) == "True"
assert len(inp) == 3451 or "LOCAL_TEST" in globals(), f"expected P5's 3451 rows, got {len(inp)}"
probes = inp[inp.probe]
print(f"{len(inp)} input rows (windows x 7 conditions) from {INPUTS}\\nsaving to {OUT}")
print(inp.groupby(["split", "condition"]).size().unstack().to_string())
"""),
    md("""
## Prompt and parser — generated from `src/models/qwen_prompt.py` and `src/models/r2_prompts.py` (do not edit here)
"""),
    "__PROMPT_MODULE__",
    "__R2_PROMPT_MODULE__",
    code("""
# 4. Look at one flipped prompt per variant before spending GPU time
r = inp[(inp.condition == "flipped") & (inp.text != "")].iloc[0]
for v in RUN_VARIANTS:
    print(f"===== variant {v} | system: {system_for(v)}")
    print(build_variant_prompt(v, r.description, r.unit, r.freq, json.loads(r.lookback), r.text), "\\n")
"""),
    code("""
# 5. Load the pinned revision; record the setup. One output file = one software setup.
from transformers import AutoTokenizer, AutoModelForCausalLM
kw = dict(revision=REVISION, device_map="cuda")
if QUANT == "nf4":
    import bitsandbytes
    from transformers import BitsAndBytesConfig
    kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.float16,
                                                   bnb_4bit_use_double_quant=False)
    QUANT_DESC = f"nf4 bitsandbytes {bitsandbytes.__version__}"
else:
    kw["torch_dtype"] = torch.float16
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
         f"{transformers.__version__} | {QUANT_DESC} | {MODEL_ID}@{REVISION}")
session = dict(started=datetime.datetime.now(datetime.timezone.utc).isoformat(), setup=SETUP)
meta = json.load(open(META)) if os.path.exists(META) else {}
assert meta.get("setup", SETUP) == SETUP, (
    f"{META} was written by another setup:\\n  {meta.get('setup')}\\nthis session:\\n  {SETUP}\\n"
    "One file = one setup. Tell Claude before doing anything else (do not delete the file).")
meta.update(run=RUN, model_id=MODEL_ID, revision=REVISION, quant=QUANT, quant_desc=QUANT_DESC,
            variants=RUN_VARIANTS, decoding="greedy", batch_size=1, max_new_tokens=MAX_NEW_TOKENS,
            gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
            transformers=transformers.__version__, inputs=os.path.basename(INPUTS), setup=SETUP)
meta.setdefault("sessions", []).append(session)
json.dump(meta, open(META, "w"), indent=1)
print(SETUP)
"""),
    code("""
# 6. Checks before the main loop: Model 5 -> reproducibility probe vs P5; others -> parse smoke test
def save_meta():
    json.dump(meta, open(META, "w"), indent=1)

REUSE_OK = False
if PROBE:
    mismatch = []
    for r in probes.itertuples():
        up = build_prompt(r.description, r.unit, r.freq, json.loads(r.lookback), r.text or None)
        raw = generate(SYSTEM, up)
        if raw != r.expected_raw:
            mismatch.append((r.domain, r.sample_id, r.condition, r.expected_raw, raw))
    REUSE_OK = len(probes) > 0 and not mismatch
    print(f"probe: {len(probes) - len(mismatch)}/{len(probes)} identical to P5 ->",
          "SAME SETUP CLASS AS P5: M5's answers reused for byte-identical prompts" if REUSE_OK
          else "MISMATCH: M5's own prompt is regenerated for every row as well (no reuse)")
    for m in mismatch[:10]:
        print("  ", m)
    assert meta.get("reuse_m5", REUSE_OK) == REUSE_OK, "probe result differs from an earlier session"
    meta.update(probe_n=len(probes), probe_mismatches=len(mismatch), reuse_m5=REUSE_OK)
else:
    first = inp[(inp.split == "test") & (inp.text != "")].head(20)
    bad = 0
    for r in first.itertuples():
        raw = generate(system_for(RUN_VARIANTS[0]), build_variant_prompt(
            RUN_VARIANTS[0], r.description, r.unit, r.freq, json.loads(r.lookback), r.text or None))
        bad += math.isnan(parse_number(raw, r.unit))
        print(f"  {r.domain:10s} y_true {r.y_true:<12g} answer {raw!r}")
    print(f"smoke test: {bad}/{len(first)} unparseable")
    meta.setdefault("smoke_test", []).append(dict(n=len(first), unparseable=bad))
    if bad >= 10:
        save_meta()
        raise RuntimeError(
            "PRE-REGISTERED FALLBACK: set QUANT = 'nf4' in cell 0, Runtime > Restart session, Run all."
            if FALLBACK and QUANT == "fp16" else
            "Too many unparseable answers: stop here and tell Claude (do not change the prompt).")
save_meta()
"""),
    code("""
# 7. Main loop — resumable. Test rows first, then val. Appends every 10 prompts.
#    After a disconnect / a new session: Run all again; it continues where it stopped.
COLS = ["domain", "split", "sample_id", "condition", "variant", "y_true", "y_pred", "parse_ok",
        "raw_output", "prompt_sha", "setup"]
run_variants = list(RUN_VARIANTS) + (["p5"] if PROBE and not REUSE_OK else [])
parts = []
for v in run_variants:
    p = inp.copy()
    p["variant"] = v
    p["user"] = [build_variant_prompt(v, r.description, r.unit, r.freq, json.loads(r.lookback),
                                      r.text or None) for r in p.itertuples()]
    # byte-identical to Model 5's prompt for this row? (then P5 already answered it)
    p["same_as_m5"] = [system_for(v) == SYSTEM and u == build_prompt(
        r.description, r.unit, r.freq, json.loads(r.lookback), r.text or None)
        for u, r in zip(p.user, p.itertuples())]
    parts.append(p)
rows = pd.concat(parts, ignore_index=True)
rows = rows[rows.split.isin(SPLITS)].copy()
rows["order"] = rows.split.map({"test": 0, "val": 1})
rows = rows.sort_values("order", kind="stable")
needed = rows[~rows.same_as_m5] if (PROBE and REUSE_OK) else rows

done = set()
if os.path.exists(OUT):
    prev = pd.read_csv(OUT, keep_default_na=False, dtype={"sample_id": str, "text": str, "raw_output": str, "expected_raw": str})
    other = set(prev.setup) - {SETUP}
    assert not other, f"{OUT} holds rows from another setup {other}; one file = one setup. Tell Claude."
    done = set(zip(prev.domain, prev.split, prev.sample_id, prev.condition, prev.variant))
    print(f"resuming: {len(done)} rows already saved")
else:
    pd.DataFrame(columns=COLS).to_csv(OUT, index=False)

key = lambda r: (r.domain, r.split, r.sample_id, r.condition, r.variant)
todo = [r for r in needed.itertuples() if key(r) not in done]
n_test = int((needed.split == "test").sum())
print(f"{len(needed)} rows needed ({n_test} test, {len(needed) - n_test} val; "
      f"{len(rows) - len(needed)} reused from P5); {len(todo)} still to do")
meta.update(rows_expected=len(needed), rows_expected_test=n_test)
save_meta()
cache, buf, n_fail, t0 = {}, [], 0, time.time()
for k, r in enumerate(todo, 1):
    system = system_for(r.variant)
    if (system, r.user) not in cache:     # identical prompts (e.g. an unattacked fake = real) run once
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
    if k % 100 == 0 or k == len(todo):
        el = time.time() - t0
        print(f"{k}/{len(todo)} [{r.split}]  parse failures so far: {n_fail}  {el / 60:.1f} min, "
              f"~{el / k * (len(todo) - k) / 60:.1f} min left", flush=True)
"""),
    code("""
# 8. Completeness (test required, val optional) + determinism re-check of the first 20 answers
res = pd.read_csv(OUT, keep_default_na=False, dtype={"sample_id": str, "text": str, "raw_output": str, "expected_raw": str})
res["parse_ok"] = res["parse_ok"].astype(str) == "True"
assert not res.duplicated(["domain", "split", "sample_id", "condition", "variant"]).any()
assert set(res.setup) == {SETUP}, "rows from more than one setup"
got = set(zip(res.domain, res.split, res.sample_id, res.condition, res.variant))
miss = needed[[key(r) not in got for r in needed.itertuples()]]
test_done = not (miss.split == "test").any()
val_done = "val" in SPLITS and not (miss.split == "val").any()
print(f"TEST {'COMPLETE' if test_done else 'INCOMPLETE'} | VAL "
      f"{'complete' if val_done else 'skipped (optional)' if 'val' not in SPLITS else 'incomplete (optional)'}"
      f" | {len(res)} rows saved, {len(miss)} missing")
assert test_done, "test split incomplete: Run all again (it resumes) — in a new session if the GPU quota ran out"
chk = res.head(20).merge(rows[["domain", "split", "sample_id", "condition", "variant", "user"]],
                         on=["domain", "split", "sample_id", "condition", "variant"])
same = [generate(system_for(r.variant), r.user) == r.raw_output for r in chk.itertuples()]
print(f"determinism: {sum(same)}/{len(same)} re-answered identically")
print("parse failures (count):")
print((~res.parse_ok).groupby([res.variant, res.condition, res.domain]).sum().unstack().to_string())
meta.update(finished=datetime.datetime.now(datetime.timezone.utc).isoformat(), rows=len(res),
            test_complete=test_done, val_complete=val_done, determinism_n=len(same),
            determinism_mismatches=len(same) - sum(same), parse_failures=int((~res.parse_ok).sum()))
save_meta()
"""),
    code("""
# 9. Download both files (they are also in MyDrive/textguard/ if Drive was mounted)
from google.colab import files
files.download(OUT)
files.download(META)
"""),
]


def build(run: str) -> dict:
    cfg = RUNS[run]
    src = PROMPT_SRC.read_text(encoding="utf-8")
    r2 = R2_PROMPT_SRC.read_text(encoding="utf-8")
    sub = {
        "__INTRO__": md(intro(cfg)),
        "__CONFIG__": code(CONFIG.format(run=run, model_id=cfg["model_id"],
                                         revision=cfg["revision"], quant=cfg["quant"],
                                         variants=json.dumps(cfg["variants"]),
                                         probe=cfg["probe"], fallback=cfg["fallback"])),
        "__PROMPT_MODULE__": code("# Generated verbatim from src/models/qwen_prompt.py\n" + src),
        "__R2_PROMPT_MODULE__": code("# Generated verbatim from src/models/r2_prompts.py\n" + r2),
    }
    cells = [sub.get(c, c) if isinstance(c, str) else c for c in CELLS]
    if run == "qwen7b":
        # The layers bitsandbytes leaves unquantised (embeddings, norms, lm_head) otherwise get
        # the installed transformers' default dtype — fp32 on 4.x, the checkpoint's bf16 on 5.x,
        # and a T4 has no bf16. §10e says "fp16 compute": pin it. (Only the 7B notebooks: the
        # M5-variant and Phi notebooks ran in fp16 already and stay byte-identical.)
        cells = _edit(cells, "# 5.", 'QUANT_DESC = f"nf4 bitsandbytes {bitsandbytes.__version__}"',
                      'QUANT_DESC = f"nf4 bitsandbytes {bitsandbytes.__version__}"\n'
                      '    kw["torch_dtype"] = torch.float16   # unquantised layers in fp16 (T4: no bf16)')
    return {"cells": cells,
            "metadata": {"accelerator": "GPU",
                         "colab": {"gpuType": "T4", "provenance": []},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def _src(cell: dict) -> str:
    return "".join(cell["source"])


def _edit(cells: list, key: str, old: str, new: str) -> list:
    """Replace ``old`` by ``new`` in the one code cell starting with ``key``; the text must occur
    exactly once, so a template change can never silently skip a platform edit."""
    hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and _src(c).startswith(key)]
    assert len(hits) == 1, f"{len(hits)} cells start with {key!r}"
    src = _src(cells[hits[0]])
    assert src.count(old) == 1, f"{old!r} occurs {src.count(old)}x in cell {key!r}"
    out = list(cells)
    out[hits[0]] = code(src.replace(old, new))
    return out


# ---- Kaggle version of the Qwen-7B run (§10e Amendment 1) -------------------------------------
# Built FROM the Colab cells by exact, asserted replacements, so cells 4, 6, 7, 8 (prompts,
# smoke test, main loop, completeness + determinism) are the Colab code — only the platform
# plumbing changes: input found by glob under /kaggle/input, outputs in /kaggle/working, no
# Drive / files.download, one visible GPU, platform recorded, progress lines for the log.

KAGGLE_FILE = "r2_qwen7b_kaggle.ipynb"
# sha256 prefix of results/qwen/qwen_inputs_p5.csv (a fresh export reproduces it byte for byte,
# §7d): the run refuses any other file before spending GPU time.
INPUT_SHA256 = "89260e2a627a6e94"

KAGGLE_INTRO = """
# R2-a — Qwen2.5-7B-Instruct (4-bit NF4) — **Kaggle** version

TextGuard R2 (pre-registration PROJECT_STATUS §10e, **Amendment 1**: Q7 runs on Kaggle). Same
model, pinned revision, 4-bit NF4, prompt, parser, test-first order, incremental saving and
checks as the Colab notebook `r2_qwen7b.ipynb`; only the platform plumbing differs.
`flipped` / `exaggerated` rows are **synthetic text for defence research** — not real reports.

* Input: `qwen_inputs_p5.csv` in **any** Kaggle dataset attached to this notebook (found by
  searching `/kaggle/input`); its sha256 must match the export.
* Output: `/kaggle/working/r2_qwen7b_nf4_kaggle_predictions.csv` + `…_run_meta.json`, appended
  every 10 prompts. After a background run they are in the notebook's **Output** tab.
* Settings: **Accelerator: GPU T4 ×2** (only cuda:0 is used), **Internet: on**.
  Then **Save Version → Save & Run All (Commit)**. The log prints a progress line every 100 rows.
* A committed run always starts fresh in `/kaggle/working`; it never reads a Colab Q7 file.
"""

KAGGLE_CELL2 = """
# 2. Platform: Kaggle. No Google Drive and no browser download: outputs are written to
#    /kaggle/working and saved as this notebook's Output when the committed run finishes.
import subprocess
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip(), flush=True)
print("visible to torch:", torch.cuda.device_count(), "GPU(s) ->", torch.cuda.get_device_name(0), flush=True)
"""

COLAB_PATHS = """OUT_DIR = "/content/drive/MyDrive/textguard" if os.path.isdir("/content/drive/MyDrive") else "/content"
os.makedirs(OUT_DIR, exist_ok=True)
INPUTS = next((p for p in ("/content/qwen_inputs_p5.csv", f"{OUT_DIR}/qwen_inputs_p5.csv") if os.path.exists(p)), None)
assert INPUTS, "Upload qwen_inputs_p5.csv to /content (Files panel on the left) or to MyDrive/textguard/"
TAG = f"r2_{RUN}_{QUANT}\""""

KAGGLE_PATHS = """import glob, hashlib
OUT_DIR = "/kaggle/working"
os.makedirs(OUT_DIR, exist_ok=True)
found = sorted(glob.glob("/kaggle/input/**/qwen_inputs_p5.csv", recursive=True))
assert found, "qwen_inputs_p5.csv not found under /kaggle/input: + Add Input > your dataset, then run again"
shas = {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in found}
assert len(set(shas.values())) == 1, f"several different qwen_inputs_p5.csv files attached: {shas}"
INPUTS = found[0]
INPUT_SHA = shas[INPUTS]
assert INPUT_SHA.startswith("__SHA__") or "LOCAL_TEST" in globals(), (
    f"{INPUTS} has sha256 {INPUT_SHA[:16]}, expected __SHA__: not the exported file")
TAG = f"r2_{RUN}_{QUANT}_kaggle"     # own file: never appends to a Colab Q7 file""".replace(
    "__SHA__", INPUT_SHA256)

KAGGLE_CELL9 = """
# 9. Outputs — in /kaggle/working; after a committed run: the notebook's Output tab -> download both
for f in (OUT, META):
    print(f"{f}  ({os.path.getsize(f) / 1024:.0f} KB)", flush=True)
print("RUN FINISHED:", "TEST COMPLETE" if test_done else "TEST INCOMPLETE", "| determinism",
      f"{meta['determinism_n'] - meta['determinism_mismatches']}/{meta['determinism_n']}", flush=True)
"""


def build_kaggle() -> dict:
    nb = build("qwen7b")
    cells = nb["cells"]
    cells[0] = md(KAGGLE_INTRO)
    cells = _edit(cells, "# 0.",
                  "# 0. Run configuration (pre-registered, §10e) — do not change except the Phi "
                  "fp16 fallback\n",
                  "# 0. Run configuration (pre-registered, §10e + Amendment 1: Kaggle) — do not "
                  "change\nimport os\n"
                  'os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # one GPU (cuda:0) even on T4 x2; '
                  "before torch is imported\n"
                  'PLATFORM = "kaggle"\n')
    cells = _edit(cells, "# 1.",
                  'assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > T4 '
                  'GPU, then Run all again"',
                  'assert torch.cuda.is_available(), "No GPU: Settings > Accelerator > GPU T4 x2, '
                  'then Save & Run All again"\n'
                  'assert torch.cuda.device_count() == 1, "exactly one GPU must be visible: '
                  'restart the session and run from cell 0"\n'
                  'assert "T4" in torch.cuda.get_device_name(0), (\n'
                  '    f"pre-registered GPU is a T4, got {torch.cuda.get_device_name(0)}: '
                  'Settings > Accelerator > GPU T4 x2")')
    i2 = next(i for i, c in enumerate(cells) if _src(c).startswith("# 2."))
    cells[i2] = code(KAGGLE_CELL2)
    cells = _edit(cells, "# 3.", COLAB_PATHS, KAGGLE_PATHS)
    cells = _edit(cells, "# 5.",
                  'f"{transformers.__version__} | {QUANT_DESC} | {MODEL_ID}@{REVISION}")',
                  'f"{transformers.__version__} | {QUANT_DESC} | {MODEL_ID}@{REVISION} | {PLATFORM}")')
    cells = _edit(cells, "# 5.",
                  "transformers=transformers.__version__, inputs=os.path.basename(INPUTS), setup=SETUP)",
                  "transformers=transformers.__version__, inputs=os.path.basename(INPUTS), setup=SETUP,\n"
                  "            platform=PLATFORM, input_path=INPUTS, input_sha256=INPUT_SHA,\n"
                  "            gpu_count_visible=torch.cuda.device_count())")
    cells = _edit(cells, "# 5.", "print(SETUP)", 'print("SETUP:", SETUP, flush=True)')
    cells = _edit(cells, "# 7.", "    if k % 100 == 0 or k == len(todo):",
                  '    if r.split == "test" and (k == len(todo) or todo[k].split != "test"):\n'
                  '        print(f"===== TEST SPLIT DONE (row {k}/{len(todo)} of this run, "\n'
                  '              f"{(time.time() - t0) / 60:.1f} min) - val follows (optional) =====",'
                  " flush=True)\n"
                  "    if k % 100 == 0 or k == len(todo):")
    cells = _edit(cells, "# 7.", 'print(f"{len(needed)} rows needed',
                  'print(f"===== MAIN LOOP | {SETUP} =====", flush=True)\n'
                  'print(f"{len(needed)} rows needed')
    cells = _edit(cells, "# 8.",
                  'assert test_done, "test split incomplete: Run all again (it resumes) — in a new '
                  'session if the GPU quota ran out"',
                  'assert test_done, "test split incomplete: Save & Run All again (a committed run '
                  'starts fresh)"')
    i9 = next(i for i, c in enumerate(cells) if _src(c).startswith("# 9."))
    cells[i9] = code(KAGGLE_CELL9)
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                     "name": "python3"},
                      "language_info": {"name": "python"},
                      "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True,
                                 "isInternetEnabled": True}}
    return nb


if __name__ == "__main__":
    nbs = {cfg["file"]: build(run) for run, cfg in RUNS.items()}
    nbs[KAGGLE_FILE] = build_kaggle()
    for name, nb in nbs.items():
        out = ROOT / "notebooks" / name
        out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {out}")
