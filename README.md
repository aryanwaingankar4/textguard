# TextGuard

**Detecting and Defending Against Misleading Text Attacks on Multimodal Time-Series Forecasting Models**

Aryan Manoj Waingankar · B.Tech AI & ML, Symbiosis Institute of Technology, Pune

---

## What this is

Some forecasters read text next to the numbers: an LLM prompted with a series and the latest
report, or a fusion model with a text encoder. For them the report is an **unverified input
channel**. This project measures how far **visible, minimal, meaning-inverting edits of genuine
reports** move such forecasters, and tests a defence:

* **Attacks** made by deterministic rules from each dataset's own reports: a direction flip,
  exaggeration, and three adaptive attacks against the defence (selective flip, an appended
  forward-looking lie, a ×10 magnitude lie). They are minimal, auditable and labelled synthetic.
* **DSS (Directional Shift Score)**: how far a forecast moves *toward the lie*, in units of the
  series' σ. MSE cannot tell "wrong" from "wrong in the direction the attacker wanted".
* **TextGuard**: a **training-free, inference-time, black-box claim-vs-evidence consistency
  gate**. It calls the forecaster with and without the text, asks a small LLM judge what the text
  claims, checks the claim against the series' own recent numbers, and lets through less of the
  text's effect when they disagree. v1 (P6) judges the whole report; v2 (R4) judges every sentence,
  checks year-over-year claims against the history, and caps the text's effect at 1σ.
* **An audit toolkit** for your own model and data (below).

This is an attack + defence study. It does not propose a new forecasting architecture, and it is
not the first work on text attacks against price forecasters (see `CLAUDE.md` §4a for related work).

## Headline findings

All on Time-MMD (Economy, SocialGood, Traffic, Health_US; chronological splits) unless noted.
Each extension was pre-registered before its results existed (`PROJECT_STATUS.md` §10), with
smallest effects of interest (DSS 0.05σ). Every number is in `PROJECT_STATUS.md` §9.

| | Finding |
|---|---|
| **Use = vulnerability** (P5) | The rule-based flip moves **Qwen2.5-1.5B** (early fusion, report in the prompt) **+0.46σ** toward the lie and reverses its forecast direction in 24% of attacked windows. Late-fusion BERT models are moved by text but not by its direction (DSS < 0.02σ). |
| **Fusion is the confound** (R1) | Feeding the same LLM's report as a pooled embedding to a late-fusion head removes ~98% of the manipulation. A direction-aware encoder (FinBERT) makes late heads attackable again. |
| **Generality** (R2) | Qwen2.5-7B (4-bit) is fooled too (+0.21σ). Phi-3.5-mini is not fooled because it ignores report content. A skeptical prompt does not help. |
| **TextGuard v1** (P6) | Cuts Qwen2.5-1.5B's pooled DSS +0.45σ → −0.04σ, but blocks 14.4% of genuine reports (budget 10%), and clipping is better on accuracy and exaggeration. A trade-off. |
| **Adaptive attack** (R3) | **One appended sentence predicting a reversal evades v1** (+0.46σ on Qwen2.5-1.5B, +0.40σ on Qwen2.5-7B), because v1's judge reads the genuine past claim. No defence stops it. |
| **TextGuard v2, held out** (R4) | Tested once on two unseen domains (Energy; FNSPID stock headlines). v2 **cuts the appended-lie attack by 88%** on Qwen2.5-1.5B (+0.180σ → +0.021σ, lower than v1), but **blocks 18.4% of genuine reports** (v1 7.0%; FNSPID 22.4%), reversing its validation advantage: the judge reads directions into neutral headlines. **Nothing transfers to Qwen2.5-7B.** Validation overstated v2, as stated before the test. |

Paper sections: `paper/sections/`; figures: `paper/figures/`.

---

## Install

Python **3.14.3** (`.python-version`). Everything that reproduces the paper runs on a CPU.

```bash
git clone https://github.com/aryanwaingankar4/textguard.git
cd textguard
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements-lock.txt   # exact environment (requirements.txt = direct deps)
```

## Reproduce every table and figure

```bash
python -m src.data.download        # Time-MMD @00281e2 + FNSPID subset @bf9189c, sha256-checked (~340 MB read)
python reproduce.py                # P1 → R4, then a byte-for-byte comparison with the committed files
```

`reproduce.py` regenerates every `results/summary_*` table, the R4 inputs and every figure from the
committed outputs. No GPU is used, and nothing is retrained. At the end it compares each committed
file under `results/` and `paper/figures/` byte for byte with the version checked out before the
run. It also checks that each regenerated GPU input file has the sha256 its Kaggle run recorded.
Details, runtime, retraining and the clean-room test: **`docs/REPRODUCE.md`**. How to rerun the
Colab / Kaggle GPU jobs (pinned revisions, setup strings): **`docs/GPU_RUNS.md`**.

Tests: `python -m pytest -q`.

---

## Use TextGuard on your own model

Any object with a `horizon` attribute and `predict(numbers, text)` works, with no inheritance and
no access to weights. `numbers` is the last *L* values in raw units; `text` is a string or `None`
(numbers only); it returns an array of shape `(horizon,)`.

**Data format** (CSV, one row per period, oldest first):

| column | |
|---|---|
| `date`, `value` | the series |
| `text` | the report(s) available **before** this row's value is known (blank = none) |
| `split` | `train` / `val` / `test` (or pass `train_end=`). σ comes from `train` only |
| `fake_text`, `fake_direction` | optional: your own attack texts and the direction (+1 / −1) each claims |

Rows before the first evaluated one serve as history. v2 uses it for year-over-year claims, so a
longer past costs nothing.

```python
from textguard import TextGuardV1, TextGuardV2, audit, judges, load_series

data = load_series("my_series.csv", lookback=24, freq="weekly")
judge_v1, judge_v2 = judges("my-series", subject="the weekly price of gasoline",
                            frame_v1="compared with the previous week")   # Qwen2.5-1.5B, GPU

report = audit(MyModel(), data, judge_v1=judge_v1, judge_v2=judge_v2,
               subject="the weekly price of gasoline")
report.save("audit_out")      # DSS, flip rate, False Block Rate, MSE, worst cases (CSV / JSON / markdown)

guarded = TextGuardV2(MyModel(), judge_v2, sigma=data.sigma, lookback=24, freq="weekly")
y_hat = guarded.predict(numbers, text, history=past_values)   # frozen paper settings
```

* **Attacks in the audit:** `a2` (keeps the genuine report and appends "*⟨Subject⟩ is expected to
  fall next week.*", predicting the reverse of the recent trend) works on any text. Our rule-based
  **flip** was written for the Time-MMD / FNSPID report styles and often fails on other text, so it
  is not offered. **Supply your own fakes** in `fake_text` / `fake_direction`.
* **Notebook:** `notebooks/textguard_your_model.ipynb` (Colab or Kaggle; free T4 GPU recommended,
  CPU works slowly) runs the judge
  with your model and CSV. Its worked example (the `DummyForecaster` on
  `examples/sample_series.csv`, a small **synthetic** series) runs as is. Without GitHub access
  (a private copy, or no Internet for `git clone`), run `python notebooks/build_toolkit_zip.py`
  and attach `dist/textguard_toolkit.zip` (23 files, ~60 KB, sha256 manifest) as a Kaggle dataset
  or upload it to Colab. Cell 1 finds it and checks every hash. Locally:
  `python examples/audit_dummy.py` (no judge) or `--judge qwen` (CPU, slow).

**Honest limits — read before trusting a number:**

* **A GPU is recommended for the judge, but a CPU works, slowly.** The judge is Qwen2.5-1.5B
  scoring three allowed answers per question. In R7 the notebook's worked example (26 windows,
  110 judge questions) ran on a Kaggle **CPU** session in ~47 min end to end, including the model
  download, with no errors (`results/r7_toolkit_kaggle/`). Time grows with the number of distinct
  texts and sentences, so use a GPU (free Colab / Kaggle T4) for real data.
* **Judge weakness on neutral financial news:** on 50 hand-labelled stock headlines it gave a
  direction to 11 of 33 neutral ones (it never reversed one). This is why v2 blocked **22% of
  genuine headline bundles** in R4. Expect high False Block Rates on text made of many short,
  mostly neutral items.
* **No transfer to Qwen2.5-7B:** on held-out data neither v1 nor v2 reduced the appended-lie attack
  on the larger model. Audit *your* model and do not assume the paper's numbers.
* **v1 is evaded by an appended forward-looking sentence.** Use v2 if that threat matters.
* The gate checks claims against the series' own past, not against the world. A lie that agrees
  with the recent trend passes, and genuine reports of a regime change (COVID) get blocked.
* Settings were frozen on our validation data; validation overstated v2 on new domains.

Full list: **`MODEL_CARD.md`**.

---

## Repository layout

| Path | Contents |
|---|---|
| `textguard/` | Public API (re-exports `src/`) |
| `src/data/` | Loaders, splits, timing / directional checks, downloads (`download.py`, `fnspid.py`) |
| `src/models/` | Forecaster interface, Models 1–5 wrappers, GPU-run exports and verifiers |
| `src/attacks/` | Rule-based attacks, validation and consistency scans |
| `src/textguard/` | TextGuard v1 / v2, judges (served and live), defences, toolkit |
| `src/metrics.py`, `src/analyze_*.py`, `src/figures_*.py` | Metrics, pre-registered analyses, figures |
| `notebooks/` | Colab / Kaggle GPU notebooks (+ their builders) |
| `results/` | Committed raw outputs (GPU answers, CPU per-unit predictions) and summary tables |
| `paper/` | LaTeX sections and figures |
| `reproduce.py`, `docs/` | Reproduction entry point and documentation |
| `PROJECT_STATUS.md`, `PHASES.md`, `CLAUDE.md` | Full log, plan, pre-registrations and project context |

## Data, ethics, licence

* **Data:** the raw datasets are **downloaded from their owners, not redistributed**:
  Time-MMD (Liu et al., NeurIPS 2024; no licence stated by its authors) and FNSPID. See
  `DATA_CARD.md`.
* **FNSPID attribution:** Zihan Dong, Xinyu Fan, Zhiyuan Peng, *FNSPID: A Comprehensive Financial
  News Dataset in Time Series*, KDD 2024 (arXiv 2402.06698), licensed **CC BY-NC 4.0**. The
  FNSPID-derived extracts in this repository and its git history are redistributed under that
  licence and are for **non-commercial use only**. They are 50 labelled headlines, plus the R4
  held-out input file of headline bundles, which is in history only. Details are in
  `DATA_CARD.md` §1a.
* **Synthetic texts:** every attack text (flipped, exaggerated, A1–A3, and the fakes in the R4
  input file in git history) is a **synthetic research artefact**. Each one is rule-generated,
  labelled, and for defence research only, and none is genuine news or a genuine report. Do not
  publish or deploy it as genuine reporting. See `ETHICS.md`.
* **Licence:** code MIT (`LICENSE`). Datasets and their extracts keep their own terms.
