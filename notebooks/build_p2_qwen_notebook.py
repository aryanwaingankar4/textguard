"""Build ``notebooks/p2_qwen_forecaster.ipynb`` (Model 5 on a Colab T4).

Why generated rather than hand-edited: the notebook must use *exactly* the prompt and parser
in ``src/models/qwen_prompt.py``. The builder pastes that file in verbatim as one cell, and
``tests/test_qwen.py`` fails if the committed notebook falls out of sync. After changing the
prompt or parser: ``python notebooks/build_p2_qwen_notebook.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPT_SRC = ROOT / "src" / "models" / "qwen_prompt.py"
OUT = ROOT / "notebooks" / "p2_qwen_forecaster.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


CELLS = [
    md("""
# P2.4 — Model 5: Qwen2.5-1.5B-Instruct forecaster (early fusion)

Forecasts every **val + test** window of the 4 TextGuard domains at horizon 1, twice:
with the real report (`condition = real`) and with the report block removed (`condition = none`).

* Input: `qwen_inputs.csv` (from `python -m src.models.export_qwen_inputs`), uploaded to `/content`
  — or placed in `MyDrive/textguard/` if Drive is mounted.
* Output: `p2_qwen_predictions.csv` + `p2_qwen_run_meta.json`, written **incrementally**
  (every 10 prompts) to Drive if mounted, else `/content`. Re-running the loop **resumes**.
* Greedy decoding, batch size 1 → deterministic: one run replaces 3 seeds.
* Prompt and parser are generated from `src/models/qwen_prompt.py` — do not edit them here.

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
INPUTS = next((p for p in ("/content/qwen_inputs.csv", f"{OUT_DIR}/qwen_inputs.csv") if os.path.exists(p)), None)
assert INPUTS, "Upload qwen_inputs.csv to /content (Files panel on the left) or to MyDrive/textguard/"
OUT = f"{OUT_DIR}/p2_qwen_predictions.csv"
META = f"{OUT_DIR}/p2_qwen_run_meta.json"

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 24
CONDITIONS = ("real", "none")

df = pd.read_csv(INPUTS, keep_default_na=False)   # keep_default_na: empty text stays ""
print(f"{len(df)} windows from {INPUTS}\\nsaving to {OUT}")
print(df.groupby(["domain", "split"]).size().to_string())
"""),
    md("""
## Prompt and parser — generated from `src/models/qwen_prompt.py` (do not edit here)
"""),
    "__PROMPT_MODULE__",
    code("""
# 4. Look at one prompt pair before spending GPU time: identical except the report block
r = df.iloc[0]
print(build_prompt(r.description, r.unit, r.freq, json.loads(r.lookback), r.text))
print("\\n" + "=" * 80 + "\\n")
print(build_prompt(r.description, r.unit, r.freq, json.loads(r.lookback), None))
"""),
    code("""
# 5. Load Qwen in fp16 on the GPU (~3 GB download the first time)
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cuda").eval()

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

def model_revision():
    \"\"\"Hub commit of the weights actually loaded (transformers 5 dropped config._commit_hash).\"\"\"
    rev = getattr(model.config, "_commit_hash", None)
    if rev is None:
        try:
            from huggingface_hub import model_info
            rev = model_info(MODEL_ID).sha
        except Exception as e:  # offline etc. — record why, never fail the run over metadata
            rev = f"unknown ({type(e).__name__})"
    return rev

meta = dict(model_id=MODEL_ID, revision=model_revision(),
            dtype="float16", decoding="greedy", batch_size=1, max_new_tokens=MAX_NEW_TOKENS,
            gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
            transformers=transformers.__version__, inputs=os.path.basename(INPUTS),
            started=datetime.datetime.now(datetime.timezone.utc).isoformat())
json.dump(meta, open(META, "w"), indent=1)
print(meta)
"""),
    code("""
# 6. Smoke test on 2 windows (not saved): raw output -> parsed number
for r in df.head(2).itertuples():
    for cond in CONDITIONS:
        up = build_prompt(r.description, r.unit, r.freq, json.loads(r.lookback), r.text if cond == "real" else None)
        raw = generate(up)
        print(f"{r.domain} {r.sample_id} {cond:<4} y_true={r.y_true:<10g} raw={raw!r} -> {parse_number(raw, r.unit)}")
"""),
    code("""
# 7. Main loop — resumable. Appends every 10 prompts; a disconnect loses at most 10 rows.
#    Re-run this cell (after cells 1-5) to continue where it stopped.
COLS = ["domain", "split", "sample_id", "condition", "y_true", "y_pred", "parse_ok", "raw_output", "prompt_sha"]
done = set()
if os.path.exists(OUT):
    prev = pd.read_csv(OUT, keep_default_na=False, dtype={"sample_id": str})
    done = set(zip(prev.domain, prev.split, prev.sample_id, prev.condition))
    print(f"resuming: {len(done)} rows already saved")
else:
    pd.DataFrame(columns=COLS).to_csv(OUT, index=False)

todo = [(r, c) for r in df.itertuples() for c in CONDITIONS
        if (r.domain, r.split, r.sample_id, c) not in done]
print(f"{len(todo)} predictions to make")
cache, buf, n_fail, t0 = {}, [], 0, time.time()
for k, (r, cond) in enumerate(todo, 1):
    up = build_prompt(r.description, r.unit, r.freq, json.loads(r.lookback), r.text if cond == "real" else None)
    if up not in cache:            # windows without a report: real and none are the same prompt
        cache[up] = generate(up)
    raw = cache[up]
    y = parse_number(raw, r.unit)
    ok = not math.isnan(y)
    n_fail += not ok
    buf.append(dict(domain=r.domain, split=r.split, sample_id=r.sample_id, condition=cond,
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
# 8. Check completeness, parse failures, and a first look at normalised MSE
res = pd.read_csv(OUT, keep_default_na=False, dtype={"sample_id": str})
res["y_pred"] = pd.to_numeric(res["y_pred"], errors="coerce")
res["parse_ok"] = res["parse_ok"].astype(str) == "True"
assert len(res) == len(CONDITIONS) * len(df), f"{len(res)} rows, expected {len(CONDITIONS) * len(df)}: re-run cell 7"
assert not res.duplicated(["domain", "split", "sample_id", "condition"]).any()
print("parse failures (count):")
print((~res.parse_ok).groupby([res.domain, res.split, res.condition]).sum().unstack().to_string())

m = res.merge(df[["domain", "split", "sample_id", "train_std"]], on=["domain", "split", "sample_id"])
m["se"] = ((m.y_true - m.y_pred) / m.train_std) ** 2
print("\\nnormalised MSE on parsed rows (P4 applies the NaN-pairing rule properly):")
print(m.groupby(["domain", "split", "condition"]).se.mean().unstack().round(3).to_string())
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
