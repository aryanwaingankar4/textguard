"""Build ``notebooks/p5_qwen_conditions.ipynb`` — Model 5 on the 5 altered text conditions.

Second Colab run (issue #29): shuffled, cross_domain, time_shifted, flipped, exaggerated, from
``qwen_inputs_p5.csv`` (``python -m src.models.export_qwen_inputs --p5``). Same prompt, parser
and decoding as P2.4 — the prompt module is pasted in verbatim, as in the P2 notebook, and
``tests/test_qwen.py`` fails if either notebook drifts from ``src/models/qwen_prompt.py``.

What is new compared with P2.4:

* **Rows carry their own text** (one row per window × condition), so the loop sends exactly
  ``text`` — no real/none switch.
* **Reuse with a reproducibility probe.** Rows whose prompt is identical to a P2.4 prompt
  (``reuse`` = real / none) are skipped and filled locally from P2.4. That is only valid if
  today's Colab stack (torch / transformers / GPU) reproduces P2.4's greedy answers, so the
  notebook first re-runs 40 P2.4 val prompts and compares the raw text. Any mismatch → it
  generates the reuse rows too.
* **The model revision is pinned and recorded** (issue #28): the Hub commit is resolved
  first and passed to ``from_pretrained(revision=...)``, so the recorded hash is the one loaded.
* No accuracy printout: all analysis is P5b, against the pre-registered family.

After changing the prompt or parser: ``python notebooks/build_p5_qwen_notebook.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from build_p2_qwen_notebook import PROMPT_SRC, code, md  # noqa: E402  (same cell helpers)

OUT = ROOT / "notebooks" / "p5_qwen_conditions.ipynb"

CELLS = [
    md("""
# P5a — Model 5 (Qwen2.5-1.5B-Instruct) on the altered text conditions

Forecasts every **val + test** window under `shuffled`, `cross_domain`, `time_shifted`,
`flipped` and `exaggerated` (horizon 1). `flipped` / `exaggerated` rows are **synthetic text
for defence research** — not real reports.

* Input: `qwen_inputs_p5.csv` (from `python -m src.models.export_qwen_inputs --p5`), uploaded to
  `/content` — or placed in `MyDrive/textguard/` if Drive is mounted.
* Output: `p5_qwen_predictions.csv` + `p5_qwen_run_meta.json`, written **incrementally**
  (every 10 prompts) to Drive if mounted, else `/content`. Re-running the loop **resumes**.
* Rows whose prompt is identical to a P2.4 prompt (`reuse` = real / none) are skipped and filled
  from P2.4 locally — **only if** the reproducibility probe (cell 6) reproduces P2.4 exactly.
* Greedy decoding, batch size 1, fp16 — identical to P2.4. Prompt and parser are generated from
  `src/models/qwen_prompt.py` — do not edit them here.

Runtime: **Runtime → Change runtime type → T4 GPU**. Then **Runtime → Run all**.
"""),
    code("""
# 1. GPU check — fails fast on a CPU runtime
import torch, transformers
assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > T4 GPU, then Run all again"
print(torch.cuda.get_device_name(0), "| torch", torch.__version__, "| transformers", transformers.__version__)
"""),
    code("""
# 2. Google Drive (recommended): results survive a Colab disconnect. Set False to skip.
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
OUT = f"{OUT_DIR}/p5_qwen_predictions.csv"
META = f"{OUT_DIR}/p5_qwen_run_meta.json"

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 24

df = pd.read_csv(INPUTS, keep_default_na=False, dtype={"sample_id": str})   # empty text stays ""
df["probe"] = df["probe"].astype(str) == "True"
probes, rows = df[df.probe], df[~df.probe]
print(f"{len(rows)} condition rows + {len(probes)} probe rows from {INPUTS}\\nsaving to {OUT}")
print(rows.assign(skip=rows.reuse != "").groupby(["condition", "split"])
      .agg(rows=("sample_id", "size"), reusable=("skip", "sum")).to_string())
"""),
    md("""
## Prompt and parser — generated from `src/models/qwen_prompt.py` (do not edit here)
"""),
    "__PROMPT_MODULE__",
    code("""
# 4. Look at one attacked prompt before spending GPU time
r = rows[rows.condition == "flipped"].iloc[0]
print(build_prompt(r.description, r.unit, r.freq, json.loads(r.lookback), r.text))
"""),
    code("""
# 5. Resolve the exact Hub commit, then load THAT revision in fp16 (issue #28: record what ran)
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from huggingface_hub import model_info
    REVISION = model_info(MODEL_ID).sha
except Exception as e:   # offline: load the cached snapshot and record its folder name
    print("could not query the Hub:", type(e).__name__, e)
    REVISION = None
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, torch_dtype=torch.float16,
                                             device_map="cuda").eval()
if REVISION is None:
    from huggingface_hub import try_to_load_from_cache
    p = try_to_load_from_cache(MODEL_ID, "config.json")
    REVISION = os.path.basename(os.path.dirname(p)) if isinstance(p, str) else "unknown"
print("revision:", REVISION)

@torch.no_grad()
def generate(user_prompt: str) -> str:
    \"\"\"Greedy, batch size 1 (no padding), so the same prompt always gives the same answer.\"\"\"
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(ids):   # transformers >= 5 returns a BatchEncoding, not a tensor
        ids = ids["input_ids"]
    ids = ids.to("cuda")
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAX_NEW_TOKENS,
                         do_sample=False, temperature=None, top_p=None, top_k=None,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

meta = dict(model_id=MODEL_ID, revision=REVISION,
            dtype="float16", decoding="greedy", batch_size=1, max_new_tokens=MAX_NEW_TOKENS,
            gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
            transformers=transformers.__version__, inputs=os.path.basename(INPUTS),
            started=datetime.datetime.now(datetime.timezone.utc).isoformat())
json.dump(meta, open(META, "w"), indent=1)
print(meta)
"""),
    code("""
# 6. Reproducibility probe: re-run P2.4 val prompts; reuse P2.4 answers only if ALL match exactly
mismatch = []
for r in probes.itertuples():
    raw = generate(build_prompt(r.description, r.unit, r.freq, json.loads(r.lookback), r.text or None))
    if raw != r.expected_raw:
        mismatch.append((r.domain, r.sample_id, r.condition, r.expected_raw, raw))
REUSE_OK = len(probes) > 0 and not mismatch
print(f"probe: {len(probes) - len(mismatch)}/{len(probes)} identical to P2.4 ->",
      "reuse P2.4 answers for unchanged prompts" if REUSE_OK else "GENERATE EVERYTHING (no reuse)")
for m in mismatch[:10]:
    print("  ", m)
meta.update(probe_n=len(probes), probe_mismatches=len(mismatch), reuse_p2=REUSE_OK)
json.dump(meta, open(META, "w"), indent=1)
"""),
    code("""
# 7. Main loop — resumable. Appends every 10 prompts; a disconnect loses at most 10 rows.
#    Re-run this cell (after cells 1-6) to continue where it stopped.
COLS = ["domain", "split", "sample_id", "condition", "y_true", "y_pred", "parse_ok", "raw_output", "prompt_sha"]
done = set()
if os.path.exists(OUT):
    prev = pd.read_csv(OUT, keep_default_na=False, dtype={"sample_id": str})
    done = set(zip(prev.domain, prev.split, prev.sample_id, prev.condition))
    print(f"resuming: {len(done)} rows already saved")
else:
    pd.DataFrame(columns=COLS).to_csv(OUT, index=False)

needed = rows if not REUSE_OK else rows[rows.reuse == ""]
todo = [r for r in needed.itertuples() if (r.domain, r.split, r.sample_id, r.condition) not in done]
print(f"{len(todo)} predictions to make ({len(rows) - len(needed)} reused from P2.4)")
cache, buf, n_fail, t0 = {}, [], 0, time.time()
for k, r in enumerate(todo, 1):
    up = build_prompt(r.description, r.unit, r.freq, json.loads(r.lookback), r.text or None)
    if up not in cache:            # identical prompts (e.g. two conditions with one text) run once
        cache[up] = generate(up)
    raw = cache[up]
    y = parse_number(raw, r.unit)
    ok = not math.isnan(y)
    n_fail += not ok
    buf.append(dict(domain=r.domain, split=r.split, sample_id=r.sample_id, condition=r.condition,
                    y_true=r.y_true, y_pred=y, parse_ok=ok, raw_output=raw, prompt_sha=prompt_sha(up)))
    if len(buf) >= 10 or k == len(todo):
        pd.DataFrame(buf, columns=COLS).to_csv(OUT, mode="a", header=False, index=False)
        buf = []
    if k % 50 == 0 or k == len(todo):
        el = time.time() - t0
        print(f"{k}/{len(todo)}  parse failures so far: {n_fail}  {el / 60:.1f} min elapsed, "
              f"~{el / k * (len(todo) - k) / 60:.1f} min left", flush=True)
"""),
    code("""
# 8. Completeness and parse failures only — accuracy and attack metrics are P5b, not here
res = pd.read_csv(OUT, keep_default_na=False, dtype={"sample_id": str})
res["parse_ok"] = res["parse_ok"].astype(str) == "True"
got = set(zip(res.domain, res.split, res.sample_id, res.condition))
missing = [r for r in needed.itertuples() if (r.domain, r.split, r.sample_id, r.condition) not in got]
assert not missing, f"{len(missing)} rows missing: re-run cell 7"
assert not res.duplicated(["domain", "split", "sample_id", "condition"]).any()
print(f"{len(res)} rows generated; parse failures (count):")
print((~res.parse_ok).groupby([res.condition, res.domain]).sum().unstack().to_string())
meta.update(finished=datetime.datetime.now(datetime.timezone.utc).isoformat(), rows=len(res),
            parse_failures=int((~res.parse_ok).sum()))
json.dump(meta, open(META, "w"), indent=1)
"""),
    code("""
# 9. Download both files (they are also in Drive if it was mounted)
from google.colab import files
files.download(OUT)
files.download(META)
"""),
]


def build() -> dict:
    src = PROMPT_SRC.read_text(encoding="utf-8")
    cells = [code("# Generated verbatim from src/models/qwen_prompt.py\n" + src)
             if c == "__PROMPT_MODULE__" else c for c in CELLS]
    return {"cells": cells,
            "metadata": {"accelerator": "GPU",
                         "colab": {"gpuType": "T4", "provenance": []},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
