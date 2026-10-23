# TextGuard demo (P7.2)

A four-screen Streamlit app: **the problem → the attack → the defence → the results**. It shows saved
results only: no model or judge runs, and there is no custom-text input.

## Run locally

The app has its own small environment, so the project's pinned R7 environment (`.venv`) is left
untouched:

```powershell
python -m venv .venv-app
.venv-app\Scripts\python -m pip install -r app/requirements.txt
.venv-app\Scripts\streamlit run app/app.py
```

The app opens at <http://localhost:8501>.

## Rebuild the data (only when results change)

```powershell
.venv\Scripts\python -m app.build_data
```

This needs the project environment and the downloaded data (`python -m src.data.download`). It writes:

| File | In git? | What |
|---|---|---|
| `app/data/demo.json` | yes | headline numbers, results tables, the 5 examples' forecasts / gates / per-claim checks, FNSPID headlines |
| `app/data/demo_timemmd_texts.json` | **no** | the report texts of the Time-MMD examples (Energy, SocialGood). Time-MMD states no licence, so the repo does not redistribute its texts (`DATA_CARD.md` §1). Without this file the app still shows every number, plus a notice where the text would be |
| `app/figures/*.png` | yes | copies of the paper figures |

Every number comes from a saved Model 5 answer and the saved judge answers, run through the analysis
code at the frozen settings, or from `results/summary_*.csv`. Nothing is tuned. TextGuard v2 is
shown only on the held-out domains, because it is never evaluated on the original domains' test
split.

## How the examples are chosen

Each slot filters windows by a fixed criterion and takes the **median** candidate (index (n − 1) // 2;
ties broken by series, then date). The rule text appears in the app next to each example and in
`SLOTS` in `build_data.py`.

| Slot | Domain | Criterion (Model 5) |
|---|---|---|
| clear flip | Energy | `flipped`: DSS > 0 and direction reversed. Median DSS |
| A2: v2 catches, v1 misses | Energy, FNSPID | `a2_forward`: DSS > 0; v1 lets the fake through (weight ≥ 0.5); v2 blocks the fake (< 0.5) but not the window's genuine report. Median DSS |
| P5 fooled | SocialGood | P5 `flipped`: DSS > 0 and direction reversed. Median DSS. SocialGood is the change-flip domain with the largest pooled P5 DSS (Health_US is excluded because its level flip also edits a number) |
| honest failure | FNSPID | v2 blocks the **genuine** report (weight < 0.5), blocking worsens the forecast, and the window has an attacked fake. Median extra error |

## Ethics and attribution

Every fake text is **synthetic**: our deterministic attack rules wrote it, for defence research only
(`ETHICS.md`). The app labels every fake. FNSPID headlines: Dong, Fan & Peng, *FNSPID*, KDD 2024
(arXiv 2402.06698), **CC BY-NC 4.0, non-commercial use only**. Changes: headlines are grouped into
weekly bundles, and the fakes are synthetic edits.

## Deployment

Not done yet. It is planned for Streamlit Community Cloud once the repository is public (P8). The
free tier needs only `app/requirements.txt`. Free apps sleep after ~12 h idle, so open the app before
any demo.
