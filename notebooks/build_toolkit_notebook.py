"""Build ``notebooks/textguard_your_model.ipynb`` — audit YOUR model with TextGuard on a free GPU (R7).

The template a deployer opens on Colab (Runtime → T4 GPU) or Kaggle (Accelerator → GPU T4,
Internet on). It gets this repository's code (the toolkit zip, or a clone), loads a CSV in the toolkit's format
(``src/textguard/toolkit.py``), wraps a model, runs the Qwen2.5-1.5B judge live on the GPU and
writes the audit. The worked example — the DummyForecaster on ``examples/sample_series.csv`` —
runs as is; to audit your own model, edit only cells 3 and 4.

Cells (``tests/test_toolkit.py`` compiles every code cell and runs the worked example with a
stub judge):

1. get the code: ``textguard_toolkit.zip`` (``build_toolkit_zip.py``) attached as a Kaggle dataset
   or uploaded to Colab, hashes checked against its manifest; otherwise ``git clone``;
2. imports and the GPU check;
3. **your data** — path, lookback, frequency, what the series is (``SUBJECT``), split;
4. **your model** — any object with ``horizon`` and ``predict(numbers, text)``;
5. the judges (one model shared by v1 and v2; pinned revision);
6. the audit (DSS / flip rate / False Block Rate / MSE / worst cases), saved to ``audit_out/``,
   plus every judge answer (``audit_out/judge_answers.csv``) so the run can be re-scored offline,
   and ``audit_out/run_info.json`` (toolkit commit, Python / torch / transformers, GPU);
7. download.

Run: ``python notebooks/build_toolkit_notebook.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).with_name("textguard_your_model.ipynb")
REPO = "https://github.com/aryanwaingankar4/textguard"


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


CELLS = [
    md(f"""
# TextGuard — audit your own text-fed forecaster

Measures how far misleading report text can push **your** forecaster (DSS, flip rate), and what
TextGuard v1 / v2, clipping and never_trust cost on genuine reports (False Block Rate, MSE).
Paper code and settings: [{REPO}]({REPO}).

* **Runtime:** a GPU (Colab: Runtime → Change runtime type → T4 GPU; Kaggle: Accelerator GPU T4,
  Internet on). The judge is Qwen2.5-1.5B-Instruct; on a CPU it is ~50× slower.
* **Worked example:** runs as is (a dummy model on a small synthetic series). For your model, edit
  cells 3 and 4 only.
* **Honest limits** (README "Use TextGuard on your own model"): the judge reads a direction into
  about 1 in 3 neutral financial headlines, so v2 can block many genuine headline bundles (18–22% in
  our held-out test); neither version helped Qwen2.5-7B there; v1 is evaded by an appended
  forward-looking sentence. Treat the audit as a measurement, not a guarantee.
* **Ethics:** the attack texts this notebook writes are synthetic, for testing defences only.
"""),
    code(f"""
# 1. Get the code: from textguard_toolkit.zip (notebooks/build_toolkit_zip.py) if one is attached
#    (Kaggle: a dataset under /kaggle/input; Colab: uploaded next to this notebook), else git clone.
import glob, hashlib, json, os, shutil, subprocess, sys, zipfile
if not os.path.exists("textguard"):
    unpacked = sorted(glob.glob("/kaggle/input/**/TOOLKIT_MANIFEST.json", recursive=True))
    zips = sorted(glob.glob("/kaggle/input/**/textguard_toolkit*.zip", recursive=True)
                  + glob.glob("textguard_toolkit*.zip"))
    if unpacked:                       # Kaggle unpacks an uploaded zip; /kaggle/input is read-only
        shutil.copytree(os.path.dirname(unpacked[0]), "textguard")
    elif zips:
        zipfile.ZipFile(zips[0]).extractall("textguard")
    else:
        subprocess.run(["git", "clone", "--depth", "1", "{REPO}"], check=True)
os.chdir("textguard")
if os.path.exists("TOOLKIT_MANIFEST.json"):  # refuse a stale or edited upload
    TOOLKIT = json.load(open("TOOLKIT_MANIFEST.json"))
    bad = [f for f, h in TOOLKIT["files"].items()
           if hashlib.sha256(open(f, "rb").read()).hexdigest() != h]
    assert not bad, f"files differ from the manifest: {{bad}}"
    print("toolkit zip from commit", TOOLKIT["commit"], "-", len(TOOLKIT["files"]), "files verified")
else:
    TOOLKIT = {{"commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                       text=True).stdout.strip()}}
# Colab / Kaggle already ship torch, transformers, pandas and numpy; only add what is missing
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scipy", "pyarrow"], check=True)
"""),
    code("""
# 2. Imports and the GPU check
import sys, torch
sys.path.insert(0, ".")
from textguard import DummyForecaster, audit, judges, load_series
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "(no GPU: the judge will be slow)")
"""),
    code("""
# 3. YOUR DATA — a CSV with columns date, value, text (+ split or TRAIN_END; optional fake_text,
#    fake_direction). `text` = the report(s) available BEFORE this row's value is known.
DATA = "examples/sample_series.csv"      # or upload: from google.colab import files; files.upload()
LOOKBACK = 12                            # values the model sees per forecast
FREQ = "weekly"                          # "weekly" / "monthly" (sets v2's year-over-year lag) or None
SUBJECT = "the average widget price"     # what the series is, in words (the judge's question)
FRAME_V1 = "compared with the previous week"   # v1 only: the frame of a typical claim ("" if none)
KEY = "my-series"                        # any name for this series
TRAIN_END = None                         # e.g. "2023-06-30" if the CSV has no `split` column
data = load_series(DATA, lookback=LOOKBACK, freq=FREQ, train_end=TRAIN_END)
print(len(data.windows), "evaluation windows; sigma (train) =", data.sigma)
"""),
    code("""
# 4. YOUR MODEL — any object with `horizon` and `predict(numbers, text) -> array of shape (horizon,)`;
#    numbers = the last LOOKBACK values (raw units), text = str or None (None = numbers only).
model = DummyForecaster(sensitivity=1.0, step=data.sigma / 4, lookback=LOOKBACK)
# class MyModel:
#     horizon = 1
#     def predict(self, numbers, text):
#         ...                      # call your forecaster here
# model = MyModel()
"""),
    code("""
# 5. The judges (Qwen2.5-1.5B-Instruct @ 989aa79, fp16 on the GPU; one model for v1 and v2)
judge_v1, judge_v2 = judges(KEY, SUBJECT, frame_v1=FRAME_V1, device=DEVICE)
"""),
    code("""
# 6. The audit: undefended, never_trust, clip (k 0.5), TextGuard v1 and v2 (frozen paper settings)
report = audit(model, data, judge_v1=judge_v1, judge_v2=judge_v2, subject=SUBJECT)
report.save("audit_out")
judge_v1.scorer.answers().to_csv("audit_out/judge_answers.csv", index=False)
import json, platform, transformers            # what this run used, saved beside the audit
json.dump(dict(toolkit_commit=TOOLKIT["commit"], python=platform.python_version(),
               torch=torch.__version__, transformers=transformers.__version__, device=DEVICE,
               gpu=torch.cuda.get_device_name(0) if DEVICE == "cuda" else None, data=DATA),
          open("audit_out/run_info.json", "w"), indent=1)
try:
    from IPython.display import Markdown, display
    display(Markdown(report.to_markdown()))
except ImportError:
    print(report.to_markdown())
"""),
    code("""
# 7. Download the results
import shutil
shutil.make_archive("audit_out", "zip", "audit_out")
try:
    from google.colab import files
    files.download("audit_out.zip")
except ImportError:
    print("Kaggle / local: audit_out.zip is in", os.getcwd())
"""),
]


def build() -> dict:
    return {"cells": CELLS,
            "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"-> {OUT}")
