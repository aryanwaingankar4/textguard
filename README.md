# TextGuard

Fake-report attacks on forecasting models that read text, and a defence against them.

Aryan Manoj Waingankar · B.Tech AI & ML, Symbiosis Institute of Technology, Pune

---

## In short

Some forecasting models get a written report along with the numbers (for example an LLM given a
series and last week's news). Nobody checks that report. We edit real reports so they say the
opposite ("rose" → "fell"), measure how much the forecast follows the lie, and test a defence.

* **Attacks** are made by fixed rules, not by an LLM: flip the direction, exaggerate the change,
  and three attacks aimed at the defence (flip only where it looks consistent, add one sentence
  about next week, multiply the change by ten). Every fake is marked synthetic.
* **DSS** (Directional Shift Score) is the move of the forecast toward the lie, in units of the
  series' σ. Error metrics like MSE can't tell you which way the forecast was pushed.
* **TextGuard** asks your model for a forecast with and without the report, asks a small LLM what
  the report claims, compares that claim with the recent numbers, and lets less of the report
  through when they disagree. No training, nothing inside the model is touched. v1 reads the whole
  report; v2 reads each sentence, checks year-over-year claims against older data and caps the
  report's effect at 1σ.
* **An audit toolkit** to run the same checks on your own model and data.

This is an attack-and-defence study, not a new forecasting model. Related work: `CLAUDE.md` §4a.

**Paper:** `paper/TextGuard_paper.docx` (IEEE format; PDF beside it). Every number in it comes from
`results/summary_*.csv` or `PROJECT_STATUS.md` §9; `paper/check_numbers.py` checks this.

## Main results

Time-MMD (Economy, SocialGood, Traffic, Health_US) unless noted. Every step had a written plan
committed before its results (`PROJECT_STATUS.md` §10).

| Step | Result |
|---|---|
| P5 attack | Qwen2.5-1.5B with the report in its prompt moves **+0.46σ** toward the lie; its predicted direction flips in 24% of attacked windows. BERT-based models barely move (< 0.02σ). |
| R1 | Same LLM, but the report given as a pooled embedding: ~98% of the effect disappears. With FinBERT, the late-fusion heads can be pushed again. |
| R2 | Qwen2.5-7B (4-bit) is pushed too (+0.21σ). Phi-3.5-mini isn't, because it ignores the report. Asking the model to be careful doesn't help. |
| P6 TextGuard v1 | +0.45σ → −0.04σ, but it blocks 14.4% of real reports (limit was 10%). Clipping does better on accuracy and exaggeration. |
| R3 | Adding one sentence that predicts the opposite gets past v1 (+0.46σ on the 1.5B model, +0.40σ on the 7B). |
| R4 TextGuard v2 | Tested once on unseen data (gasoline prices; FNSPID stock headlines): that attack drops +0.180σ → +0.021σ, but v2 blocks 18.4% of real reports (v1: 7.0%). No gain on Qwen2.5-7B. |

Figures: `paper/figures/`.

---

## Install

Python 3.14.3 (see `.python-version`). A CPU is enough to reproduce everything.

```bash
git clone https://github.com/aryanwaingankar4/textguard.git
cd textguard
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements-lock.txt
```

## Reproduce

```bash
python -m src.data.download        # Time-MMD + FNSPID files, hash-checked
python reproduce.py                # rebuilds all tables and figures, then compares with the repo
```

No GPU and no retraining: the LLM answers are stored in `results/` and checked on load.
`docs/REPRODUCE.md` has the details; `docs/GPU_RUNS.md` explains how to redo the GPU runs.
Tests: `python -m pytest -q`.

---

## Use TextGuard on your own model

Your model needs a `horizon` attribute and a `predict(numbers, text)` method. `numbers` is the last
*L* values; `text` is a string or `None`; the result has shape `(horizon,)`.

Input CSV, one row per period, oldest first:

| column | meaning |
|---|---|
| `date`, `value` | the series |
| `text` | the report available before this value is known (blank if none) |
| `split` | `train` / `val` / `test` (or pass `train_end=`); σ uses `train` only |
| `fake_text`, `fake_direction` | optional: your own fakes and the direction (+1 / −1) each one claims |

Earlier rows act as history; v2 uses them for year-over-year claims.

```python
from textguard import TextGuardV1, TextGuardV2, audit, judges, load_series

data = load_series("my_series.csv", lookback=24, freq="weekly")
judge_v1, judge_v2 = judges("my-series", subject="the weekly price of gasoline",
                            frame_v1="compared with the previous week")   # Qwen2.5-1.5B

report = audit(MyModel(), data, judge_v1=judge_v1, judge_v2=judge_v2,
               subject="the weekly price of gasoline")
report.save("audit_out")      # DSS, flip rate, false blocks, MSE, worst cases

guarded = TextGuardV2(MyModel(), judge_v2, sigma=data.sigma, lookback=24, freq="weekly")
y_hat = guarded.predict(numbers, text, history=past_values)
```

* The audit's built-in attack is `a2`: it keeps the real report and adds "*⟨Subject⟩ is expected to
  fall next week.*" (the opposite of the recent trend). Our flip rules only fit the Time-MMD and
  FNSPID report styles, so for other text bring your own fakes (`fake_text`, `fake_direction`).
* Notebook: `notebooks/textguard_your_model.ipynb` (Colab or Kaggle). Its example uses the
  `DummyForecaster` on `examples/sample_series.csv`, a small synthetic series. No GitHub access?
  Run `python notebooks/build_toolkit_zip.py` and upload `dist/textguard_toolkit.zip`; the first
  cell checks its hashes. Locally: `python examples/audit_dummy.py` (add `--judge qwen` to use the
  real judge on a CPU).

**Before you trust a number:**

* The judge runs on a CPU but slowly (the notebook example took about 47 minutes on a Kaggle CPU,
  including the download). Use a GPU for real data.
* The judge sees a direction in some neutral headlines (11 of 33 in our check), which is why v2
  blocked 22% of real stock-headline bundles. Text made of many short neutral items will see many
  false blocks.
* On Qwen2.5-7B neither version helped on new data. Audit your own model.
* v1 can be beaten by one added sentence about the future; use v2 if that matters.
* The check compares claims with the series' own past. A lie that matches the trend passes; true
  news of a sudden change can be blocked.
* Settings were tuned on our validation data, which made v2 look better than it was on new data.

More in `MODEL_CARD.md`.

---

## Where things are

| Path | What |
|---|---|
| `textguard/` | public API |
| `src/data/` | loading, splits, data checks, downloads |
| `src/models/` | model wrappers, GPU-run inputs and checks |
| `src/attacks/` | attack rules and their checks |
| `src/textguard/` | TextGuard v1 / v2, judges, baselines, toolkit |
| `src/metrics.py`, `src/analyze_*.py`, `src/figures_*.py` | metrics, planned analyses, figures |
| `notebooks/` | GPU notebooks and their builders |
| `results/` | stored outputs and summary tables |
| `paper/` | the paper (.docx / .pdf), its build and check scripts, figures, older LaTeX drafts |
| `reproduce.py`, `docs/` | reproduction |
| `PROJECT_STATUS.md`, `PHASES.md`, `CLAUDE.md` | project log, plan, plans fixed in advance |

## Data, ethics, licence

* The datasets are downloaded from their owners, not copied here: Time-MMD (Liu et al., NeurIPS
  2024; its authors state no licence) and FNSPID. See `DATA_CARD.md`.
* FNSPID: Zihan Dong, Xinyu Fan, Zhiyuan Peng, *FNSPID: A Comprehensive Financial News Dataset in
  Time Series*, KDD 2024 (arXiv 2402.06698), licence **CC BY-NC 4.0**. The FNSPID extracts here and
  in git history (50 labelled headlines; the R4 input file of headline bundles, history only) are
  shared under that licence, **non-commercial use only**. `DATA_CARD.md` §1a.
* Every attack text is **synthetic**, made by rules for defence research. None is real news. Don't
  publish or use one as if it were. See `ETHICS.md`.
* Code: MIT (`LICENSE`). Data keeps its own terms.
