"""Build ``notebooks/p6_direction_judge.ipynb`` — TextGuard's Qwen direction judge on Colab (P6a).

Same model, revision and software setup class as P5's Model 5 run, so the paper can say one
Qwen2.5-1.5B-Instruct snapshot did both jobs:

* **Revision pinned** to the one P5 loaded (``p5_qwen_run_meta.json``), not "latest on the Hub".
* **Reproducibility probe first:** the 40 forecasting prompts P5 probed with are re-run and
  compared with P2.4's raw answers. 40/40 identical → same setup class as P5, certified. If not,
  the judge still runs (every answer comes from one setup, and the judge is never paired with a
  forecast from another stack), and the outcome is recorded in the meta file and on every row
  when loaded — reported, not hidden.
* **Answers are scored, not parsed** (``src/textguard/judge_prompt.py``): the probability of
  each allowed answer given the prompt; answer = the most likely; confidence = its probability
  renormalised over the three. A short greedy answer is kept in ``raw_output`` for auditing.

Both prompt modules (forecaster for the probe, judge) are pasted in verbatim;
``tests/test_judge.py`` fails if the committed notebook drifts from them.
After changing either: ``python notebooks/build_p6_judge_notebook.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from build_p2_qwen_notebook import PROMPT_SRC, code, md  # noqa: E402

JUDGE_SRC = ROOT / "src" / "textguard" / "judge_prompt.py"
P5_META = ROOT / "results" / "qwen" / "p5_qwen_run_meta.json"
OUT = ROOT / "notebooks" / "p6_direction_judge.ipynb"

CELLS = [
    md("""
# P6a — TextGuard's direction judge (Qwen2.5-1.5B-Instruct)

For every distinct text TextGuard will see (val + test windows, real **and** every altered
condition, plus the human-labelled P1 reports) the judge answers what the text claims about the
forecast series: **change** (up / down / none) for every domain, and for Health_US also **level**
(above / below / none the national baseline). `flipped` / `exaggerated` texts are **synthetic text
for defence research** — not real reports.

* Input: `judge_inputs.csv` (from `python -m src.textguard.export_judge_inputs`), uploaded to
  `/content` — or placed in `MyDrive/textguard/` if Drive is mounted.
* Output: `p6_judge_predictions.csv` + `p6_judge_run_meta.json`, written **incrementally** (every
  20 questions) to Drive if mounted, else `/content`. Re-running the loop **resumes**.
* Same model **revision** as P5 (pinned), fp16, batch size 1. A 40-prompt **probe** first checks the
  stack reproduces P5's setup class.
* Answers are **scored**: probability of each allowed answer → answer + confidence. No parsing.
  Prompts are generated from `src/textguard/judge_prompt.py` — do not edit them here.
* No accuracy here: judge validation is P6b.

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
INPUTS = next((p for p in ("/content/judge_inputs.csv", f"{OUT_DIR}/judge_inputs.csv") if os.path.exists(p)), None)
assert INPUTS, "Upload judge_inputs.csv to /content (Files panel on the left) or to MyDrive/textguard/"
OUT = f"{OUT_DIR}/p6_judge_predictions.csv"
META = f"{OUT_DIR}/p6_judge_run_meta.json"

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 24          # forecaster probe: identical to P2.4 / P5
JUDGE_NEW_TOKENS = 4         # audit-only greedy answer of the judge

df = pd.read_csv(INPUTS, keep_default_na=False, dtype=str)
probes, rows = df[df.row_type == "probe"], df[df.row_type == "judge"]
print(f"{len(rows)} judge questions + {len(probes)} probe prompts from {INPUTS}\\nsaving to {OUT}")
print(rows.groupby(["domain", "question"]).size().to_string())
"""),
    md("""
## Prompts — generated from `src/models/qwen_prompt.py` (probe) and `src/textguard/judge_prompt.py` (judge). Do not edit here.
"""),
    "__PROMPT_MODULE__",
    "__JUDGE_MODULE__",
    code("""
# 4. Look at one judge prompt before spending GPU time
r = rows[rows.domain == "Health_US"].iloc[0]
print(build_judge_prompt(r.domain, r.question, r.text))
"""),
    "__LOAD_CELL__",
    code("""
# 6. Reproducibility probe: P5's 40 forecasting prompts must reproduce P2.4's raw answers exactly
mismatch = []
for r in probes.itertuples():
    raw = generate(build_prompt(r.description, r.unit, r.freq, json.loads(r.lookback), r.text or None))
    if raw != r.expected_raw:
        mismatch.append((r.domain, r.sample_id, r.condition, r.expected_raw, raw))
PROBE_OK = len(probes) == 40 and not mismatch
print(f"probe: {len(probes) - len(mismatch)}/{len(probes)} identical to P2.4 ->",
      "SAME SETUP CLASS AS P5 (certified)" if PROBE_OK
      else "DIFFERENT SETUP CLASS from P5 - the judge still runs on one setup; this is recorded")
print("setup identical to P5's:", SETUP == P5_SETUP, "|", SETUP)
for m in mismatch[:10]:
    print("  ", m)
meta.update(probe_n=len(probes), probe_mismatches=len(mismatch), probe_ok=PROBE_OK,
            same_setup_string_as_p5=SETUP == P5_SETUP)
json.dump(meta, open(META, "w"), indent=1)
"""),
    code("""
# 7. Main loop — resumable. Appends every 20 questions; a disconnect loses at most 20 rows.
#    Re-run this cell (after cells 1-6) to continue where it stopped.
COLS = ["domain", "question", "prompt_sha", "answer", "confidence", "p_pos", "p_neg", "p_none",
        "logp_pos", "logp_neg", "logp_none", "raw_output", "setup"]
done = set()
if os.path.exists(OUT):
    prev = pd.read_csv(OUT, keep_default_na=False, dtype=str)
    other = set(prev.setup) - {SETUP}
    assert not other, (f"{OUT} holds rows from another software setup {other}; one file = one setup. "
                       "Rename or delete it and run again from cell 1.")
    done = set(prev.prompt_sha)
    print(f"resuming: {len(done)} answers already saved")
else:
    pd.DataFrame(columns=COLS).to_csv(OUT, index=False)

todo = [r for r in rows.itertuples() if r.prompt_sha not in done]
print(f"{len(todo)} questions to answer")
meta.update(rows_expected=len(rows))
json.dump(meta, open(META, "w"), indent=1)
buf, t0 = [], time.time()
for k, r in enumerate(todo, 1):
    up = build_judge_prompt(r.domain, r.question, r.text)
    assert judge_sha(up) == r.prompt_sha, "judge prompt differs from the exported one"
    answers = ANSWERS[r.question]
    lp = score(up, answers)
    m = max(lp)
    p = [math.exp(x - m) for x in lp]
    p = [x / sum(p) for x in p]
    best = max(range(3), key=lambda i: (p[i], -i))    # ties -> earlier answer (never happens in practice)
    buf.append(dict(domain=r.domain, question=r.question, prompt_sha=r.prompt_sha,
                    answer=answers[best], confidence=p[best], p_pos=p[0], p_neg=p[1], p_none=p[2],
                    logp_pos=lp[0], logp_neg=lp[1], logp_none=lp[2],
                    raw_output=generate(up, system=JUDGE_SYSTEM, max_new_tokens=JUDGE_NEW_TOKENS),
                    setup=SETUP))
    if len(buf) >= 20 or k == len(todo):
        pd.DataFrame(buf, columns=COLS).to_csv(OUT, mode="a", header=False, index=False)
        buf = []
    if k % 100 == 0 or k == len(todo):
        el = time.time() - t0
        print(f"{k}/{len(todo)}  {el / 60:.1f} min elapsed, ~{el / k * (len(todo) - k) / 60:.1f} min left",
              flush=True)
"""),
    code("""
# 8. Completeness and answer counts only — judge accuracy is P6b, not here
res = pd.read_csv(OUT, keep_default_na=False, dtype=str)
missing = set(rows.prompt_sha) - set(res.prompt_sha)
assert not missing, f"{len(missing)} questions missing: re-run cell 7"
assert not res.prompt_sha.duplicated().any()
assert set(res.setup) == {SETUP}, "rows from more than one software setup"
print(f"{len(res)} answers from one setup")
print(res.groupby(["domain", "question", "answer"]).size().unstack(fill_value=0).to_string())
meta.update(finished=datetime.datetime.now(datetime.timezone.utc).isoformat(), rows=len(res))
json.dump(meta, open(META, "w"), indent=1)
"""),
    code("""
# 9. Download both files (they are also in Drive if it was mounted)
from google.colab import files
files.download(OUT)
files.download(META)
"""),
]

LOAD_CELL = '''
# 5. Load the SAME revision P5 loaded (pinned), fp16; define the probe's generate() and the judge's score()
from transformers import AutoTokenizer, AutoModelForCausalLM
REVISION = "__REVISION__"     # P5's revision (p5_qwen_run_meta.json) - pinned, not "latest"
P5_SETUP = "__P5_SETUP__"
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, torch_dtype=torch.float16,
                                             device_map="cuda").eval()

def chat_ids(system: str, user_prompt: str):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(ids):   # transformers >= 5 returns a BatchEncoding, not a tensor
        ids = ids["input_ids"]
    return ids.to("cuda")

@torch.no_grad()
def generate(user_prompt: str, system: str = SYSTEM, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Greedy, batch size 1 (no padding) - identical to P2.4 / P5 for the forecaster probe."""
    ids = chat_ids(system, user_prompt)
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                         do_sample=False, temperature=None, top_p=None, top_k=None,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

ANSWER_IDS = {a: tok(a, add_special_tokens=False).input_ids for k in ANSWERS for a in ANSWERS[k]}
print("answer tokens:", ANSWER_IDS)

@torch.no_grad()
def score(user_prompt: str, answers) -> list:
    """log P(answer | judge prompt) for each allowed answer (teacher forcing, summed over tokens)."""
    ids = chat_ids(JUDGE_SYSTEM, user_prompt)
    if all(len(ANSWER_IDS[a]) == 1 for a in answers):        # one forward pass is enough
        lsm = torch.log_softmax(model(ids).logits[0, -1].float(), -1)
        return [lsm[ANSWER_IDS[a][0]].item() for a in answers]
    out = []
    for a in answers:                                         # multi-token answer: full forward each
        a_ids = torch.tensor([ANSWER_IDS[a]], device="cuda")
        logits = model(torch.cat([ids, a_ids], 1)).logits[0].float()
        lsm = torch.log_softmax(logits[ids.shape[1] - 1:-1], -1)
        out.append(lsm.gather(1, a_ids[0][:, None]).sum().item())
    return out

meta = dict(model_id=MODEL_ID, revision=REVISION, dtype="float16", batch_size=1,
            judge="scored answers (teacher forcing), confidence renormalised over 3",
            answer_tokens=ANSWER_IDS, max_new_tokens_probe=MAX_NEW_TOKENS,
            max_new_tokens_judge=JUDGE_NEW_TOKENS,
            gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
            transformers=transformers.__version__, inputs=os.path.basename(INPUTS),
            started=datetime.datetime.now(datetime.timezone.utc).isoformat())
# the software setup that produces every row of this run — same format as qwen_forecaster.setup_id
SETUP = (f"{meta['gpu']} | torch {meta['torch']} | transformers {meta['transformers']} | "
         f"{MODEL_ID}@{REVISION}")
meta["setup"] = SETUP
json.dump(meta, open(META, "w"), indent=1)
print(meta)
'''


def build() -> dict:
    p5 = json.loads(P5_META.read_text(encoding="utf-8"))
    load = (LOAD_CELL.replace("__REVISION__", p5["revision"])
            .replace("__P5_SETUP__", p5["setup"]))
    sub = {"__PROMPT_MODULE__": code("# Generated verbatim from src/models/qwen_prompt.py\n"
                                     + PROMPT_SRC.read_text(encoding="utf-8")),
           "__JUDGE_MODULE__": code("# Generated verbatim from src/textguard/judge_prompt.py\n"
                                    + JUDGE_SRC.read_text(encoding="utf-8")),
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
