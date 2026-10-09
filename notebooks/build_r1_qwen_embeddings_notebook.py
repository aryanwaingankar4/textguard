"""Build ``notebooks/r1_qwen_embeddings.ipynb`` — Qwen2.5-1.5B text embeddings on Colab (R1).

R1 (PROJECT_STATUS §10d) turns Model 5's own LLM into a frozen text encoder for the late-fusion
heads of Models 3/4. The embedding recipe is fixed in the pre-registration and fixed here:

* **Same model, revision and setup class as Model 5:** revision pinned to P5's
  (``p5_qwen_run_meta.json``); P5's 40-prompt **reproducibility probe** runs first. 40/40 →
  same setup class as P5, certified; otherwise the embeddings still come from one setup, and
  the outcome is recorded in the meta file — reported, not hidden.
* **Embedding = last-layer hidden state (after the final norm) mean-pooled over all tokens**,
  fp16 forward, pooled in float32, **batch size 1** (no padding, so every token is real), no
  chat template — the encoder reads the same MM-TSFlib template string BERT reads. No
  truncation; token counts recorded.
* **Determinism re-check:** the first 20 prompts are embedded again at the end and must be
  bit-identical.

The forecasting-prompt module (for the probe) is pasted in verbatim; ``tests/test_r1.py`` fails
if the committed notebook drifts from it or from the pinned revision.
After changing either: ``python notebooks/build_r1_qwen_embeddings_notebook.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from build_p2_qwen_notebook import PROMPT_SRC, code, md  # noqa: E402

P5_META = ROOT / "results" / "qwen" / "p5_qwen_run_meta.json"
OUT = ROOT / "notebooks" / "r1_qwen_embeddings.ipynb"

CELLS = [
    md("""
# R1 — Qwen2.5-1.5B-Instruct text embeddings (LLM vs fusion)

Model 5's own LLM as a **frozen text encoder** for the late-fusion heads of Models 3/4
(PROJECT_STATUS §10d). One 1536-d vector per prompt: the last-layer hidden state (after the
final norm), **mean-pooled over all tokens**. Some prompts wrap `flipped` / `exaggerated` texts —
**synthetic text for defence research**, not real reports.

* Input: `r1_embed_inputs.csv` (from `python -m src.models.qwen_embeddings export`), uploaded to
  `/content` — or placed in `MyDrive/textguard/` if Drive is mounted.
* Output: `r1_qwen_embeddings.npz` + `r1_qwen_embed_meta.json`, saved **every 200 prompts** to
  Drive if mounted, else `/content`. Re-running the loop **resumes**.
* Same model **revision** as P5 (pinned), fp16, batch size 1. A 40-prompt **probe** first checks the
  stack reproduces P5's setup class.
* No training, no forecasts here.

Runtime: **Runtime → Change runtime type → T4 GPU**. Then **Runtime → Run all** (~5 min).
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
# 3. Paths and inputs; every fingerprint is re-checked against its prompt
import os, json, time, hashlib, datetime
import numpy as np
import pandas as pd

OUT_DIR = "/content/drive/MyDrive/textguard" if os.path.isdir("/content/drive/MyDrive") else "/content"
os.makedirs(OUT_DIR, exist_ok=True)
INPUTS = next((p for p in ("/content/r1_embed_inputs.csv", f"{OUT_DIR}/r1_embed_inputs.csv") if os.path.exists(p)), None)
assert INPUTS, "Upload r1_embed_inputs.csv to /content (Files panel on the left) or to MyDrive/textguard/"
OUT = f"{OUT_DIR}/r1_qwen_embeddings.npz"
META = f"{OUT_DIR}/r1_qwen_embed_meta.json"

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 24          # forecaster probe: identical to P2.4 / P5
DIM = 1536

df = pd.read_csv(INPUTS, keep_default_na=False, dtype=str)
probes, rows = df[df.row_type == "probe"], df[df.row_type == "embed"]
bad = [r.prompt_sha for r in rows.itertuples() if hashlib.sha256(r.prompt.encode("utf-8")).hexdigest() != r.prompt_sha]
assert not bad, f"{len(bad)} prompts changed in transit (fingerprint mismatch) - re-upload the CSV"
print(f"{len(rows)} prompts to embed + {len(probes)} probe prompts from {INPUTS}\\nsaving to {OUT}")
print(rows.iloc[0].prompt[:400])
"""),
    md("""
## Probe prompt module — generated from `src/models/qwen_prompt.py`. Do not edit here.
"""),
    "__PROMPT_MODULE__",
    "__LOAD_CELL__",
    code("""
# 5. Reproducibility probe: P5's 40 forecasting prompts must reproduce P2.4's raw answers exactly
mismatch = []
for r in probes.itertuples():
    raw = generate(build_prompt(r.description, r.unit, r.freq, json.loads(r.lookback), r.text or None))
    if raw != r.expected_raw:
        mismatch.append((r.domains, r.sample_id, r.condition, r.expected_raw, raw))
PROBE_OK = len(probes) == 40 and not mismatch
print(f"probe: {len(probes) - len(mismatch)}/{len(probes)} identical to P2.4 ->",
      "SAME SETUP CLASS AS P5 (certified)" if PROBE_OK
      else "DIFFERENT SETUP CLASS from P5 - embeddings still come from one setup; this is recorded")
print("setup identical to P5's:", SETUP == P5_SETUP, "|", SETUP)
for m in mismatch[:10]:
    print("  ", m)
meta.update(probe_n=len(probes), probe_mismatches=len(mismatch), probe_ok=PROBE_OK,
            same_setup_string_as_p5=SETUP == P5_SETUP)
json.dump(meta, open(META, "w"), indent=1)
"""),
    code("""
# 6. Main loop — resumable. Saves every 200 prompts; a disconnect loses at most 200 (~20 s of work).
#    Re-run this cell (after cells 1-5) to continue where it stopped.
def save(shas, embs, ntok):
    tmp = OUT + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, sha=np.array(shas), emb=np.stack(embs).astype(np.float32), n_tokens=np.array(ntok))
    os.replace(tmp, OUT)

shas, embs, ntok = [], [], []
if os.path.exists(OUT):
    assert meta.get("setup") == SETUP
    with np.load(OUT, allow_pickle=False) as z:
        shas, embs, ntok = [str(s) for s in z["sha"]], list(z["emb"]), [int(n) for n in z["n_tokens"]]
    print(f"resuming: {len(shas)} embeddings already saved")
done = set(shas)
todo = [r for r in rows.itertuples() if r.prompt_sha not in done]
print(f"{len(todo)} prompts to embed")
t0 = time.time()
for k, r in enumerate(todo, 1):
    v, n = embed(r.prompt)
    shas.append(r.prompt_sha); embs.append(v); ntok.append(n)
    if k % 200 == 0 or k == len(todo):
        save(shas, embs, ntok)
        el = time.time() - t0
        print(f"{k}/{len(todo)}  {el / 60:.1f} min elapsed, ~{el / k * (len(todo) - k) / 60:.1f} min left", flush=True)
"""),
    code("""
# 7. Checks: complete, finite, 1536-d, and deterministic (first 20 prompts embedded again, bit-identical)
with np.load(OUT, allow_pickle=False) as z:
    z = {k: z[k] for k in z.files}
got = [str(s) for s in z["sha"]]
missing = set(rows.prompt_sha) - set(got)
assert not missing, f"{len(missing)} prompts missing: re-run cell 6"
assert len(got) == len(set(got)) == len(rows)
E = z["emb"]
assert E.shape == (len(rows), DIM) and E.dtype == np.float32 and np.isfinite(E).all()
pos = {s: i for i, s in enumerate(got)}
again = [embed(r.prompt)[0] for r in rows.head(20).itertuples()]
DETERMINISTIC = all(np.array_equal(a, E[pos[r.prompt_sha]]) for a, r in zip(again, rows.head(20).itertuples()))
print("deterministic re-embedding (20 prompts):", DETERMINISTIC)
print(f"{len(got)} embeddings, tokens median {int(np.median(z['n_tokens']))}, max {int(z['n_tokens'].max())}")
print("vector norm: median %.1f, min %.1f, max %.1f" % tuple(np.percentile(np.linalg.norm(E, axis=1), [50, 0, 100])))
meta.update(finished=datetime.datetime.now(datetime.timezone.utc).isoformat(), rows=len(got),
            deterministic_first20=DETERMINISTIC, tokens_median=int(np.median(z["n_tokens"])),
            tokens_max=int(z["n_tokens"].max()))
json.dump(meta, open(META, "w"), indent=1)
"""),
    code("""
# 8. Download both files (they are also in Drive if it was mounted)
from google.colab import files
files.download(OUT)
files.download(META)
"""),
]

LOAD_CELL = '''
# 4. Load the SAME revision P5 loaded (pinned), fp16; define the probe's generate() and embed()
from transformers import AutoTokenizer, AutoModelForCausalLM
REVISION = "__REVISION__"     # P5's revision (p5_qwen_run_meta.json) - pinned, not "latest"
P5_SETUP = "__P5_SETUP__"
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, torch_dtype=torch.float16,
                                             device_map="cuda").eval()
BASE = model.model            # the decoder stack; its last_hidden_state is after the final norm
assert type(BASE).__name__ == "Qwen2Model", type(BASE)

@torch.no_grad()
def generate(user_prompt: str, system: str = SYSTEM, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Greedy, batch size 1 (no padding) - identical to P2.4 / P5, for the probe only."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(ids):   # transformers >= 5 returns a BatchEncoding, not a tensor
        ids = ids["input_ids"]
    ids = ids.to("cuda")
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                         do_sample=False, temperature=None, top_p=None, top_k=None,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

@torch.no_grad()
def embed(prompt: str):
    """(1536-d float32 mean of the last-layer hidden states over all tokens, token count).
    Plain tokenizer call (no chat template, no truncation), batch size 1 -> no padding."""
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    h = BASE(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0]
    return h.float().mean(0).cpu().numpy().astype(np.float32), int(ids.shape[1])

meta = dict(model_id=MODEL_ID, revision=REVISION, dtype="float16", batch_size=1,
            embedding="last_hidden_state of Qwen2Model (after final norm), mean over all tokens, float32",
            template="MM-TSFlib prompt (src.models.text_encoder.make_prompt), no chat template",
            truncation=None, dim=DIM, max_new_tokens_probe=MAX_NEW_TOKENS,
            gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
            transformers=transformers.__version__, inputs=os.path.basename(INPUTS),
            started=datetime.datetime.now(datetime.timezone.utc).isoformat())
# the software setup that produces every row of this run — same format as qwen_forecaster.setup_id
SETUP = (f"{meta['gpu']} | torch {meta['torch']} | transformers {meta['transformers']} | "
         f"{MODEL_ID}@{REVISION}")
meta["setup"] = SETUP
if os.path.exists(META):      # resuming: keep what an earlier session recorded (probe, start time)
    old = json.load(open(META))
    assert old.get("setup") == SETUP, f"{META} is from another software setup {old.get('setup')}; one file = one setup"
    meta = {**meta, **old}
json.dump(meta, open(META, "w"), indent=1)
print(meta)
'''


def build() -> dict:
    p5 = json.loads(P5_META.read_text(encoding="utf-8"))
    load = (LOAD_CELL.replace("__REVISION__", p5["revision"])
            .replace("__P5_SETUP__", p5["setup"]))
    sub = {"__PROMPT_MODULE__": code("# Generated verbatim from src/models/qwen_prompt.py\n"
                                     + PROMPT_SRC.read_text(encoding="utf-8")),
           "__LOAD_CELL__": code(load)}
    cells = [sub[c] if isinstance(c, str) else c for c in CELLS]
    return {"cells": cells,
            "metadata": {"accelerator": "GPU",
                         "colab": {"gpuType": "T4", "provenance": []},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
