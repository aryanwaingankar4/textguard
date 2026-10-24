# Data card

Where the data comes from, how it was cleaned and split, and what the fake texts are. Full detail:
`PROJECT_STATUS.md` §6 (data), §8 (attacks), §10 (plans fixed in advance).

**The raw data is not stored here.** `python -m src.data.download` fetches each source at a fixed
revision and checks every file against `src/data/manifest.json` (sha256 of the raw bytes); a changed
file fails. A few small FNSPID extracts *are* in the repository and its history, shared under
FNSPID's CC BY-NC 4.0 licence (§1a).

**Line endings matter.** All results were computed from a Windows clone of Time-MMD, which turned
every LF into CRLF, also inside the report texts. So the prompts the GPU models answered contain
`\r\n` inside the reports. With an LF copy, 166 Model 5 prompts change and the loaders reject the
stored answers (fingerprints differ). `download.py` writes Time-MMD exactly as that clone had it on
any OS (each lone LF becomes CRLF), byte-identical for all 30 files. Use the script, not `git clone`;
a clone on Linux / macOS gives LF and fails the check.

## 1. Sources and licences

| Source | Used for | Revision | Licence |
|---|---|---|---|
| **Time-MMD** (Liu et al., NeurIPS 2024 Datasets & Benchmarks), `github.com/AdityaLab/Time-MMD` | Economy, SocialGood, Traffic, Health_US (P1–R3) and Energy (R4) | commit `00281e2` (2025-01-14) | **None stated** (no LICENSE file, no terms in the README). Not redistributed; get it from the authors and cite arXiv 2406.08627. |
| **FNSPID** (Dong et al., KDD 2024), `huggingface.co/datasets/Zihan1004/FNSPID` | R4's FNSPID domain (10 S&P 500 stocks, Benzinga headlines, daily closes) | `bf9189c` | **CC BY-NC 4.0**, non-commercial, with credit. Raw files not redistributed; the extracts in §1a are, under the same licence. |
| S&P 500 membership history, `github.com/fja05680/sp500` | picking the FNSPID stocks (members on 2018-06-01) | `a2430f2` | MIT |
| MM-TSFlib (Liu et al.), `github.com/AdityaLab/MM-TSFlib` | DLinear / PatchTST code for Models 1–4 (only needed to retrain) | `e789ce7` | MIT |

Derived material in the repository: 150 hand-labelled excerpts (50 + 50 Time-MMD reports in
`results/summary_directional_*.csv`; 50 FNSPID headlines in `summary_directional_fnspid.csv` and
`summary_r4_judge_labels_rows.csv`), plus model and judge answers, which hold numbers and answer
words only; each prompt's report appears as a fingerprint.

## 1a. FNSPID credit and what is shared

> **FNSPID**: Zihan Dong, Xinyu Fan and Zhiyuan Peng. *FNSPID: A Comprehensive Financial News
> Dataset in Time Series.* KDD 2024 (arXiv 2402.06698). `huggingface.co/datasets/Zihan1004/FNSPID`,
> revision `bf9189c`. Licence **CC BY-NC 4.0**
> (<https://creativecommons.org/licenses/by-nc/4.0/>). **Non-commercial use only.**

CC BY-NC 4.0 allows non-commercial sharing with credit and a note of changes. The material below is
shared on those terms: non-commercial only, keep the credit above. The code's MIT licence does not
cover it.

| Where | What | Changed how |
|---|---|---|
| `results/summary_directional_fnspid.csv`, `results/summary_r4_judge_labels_rows.csv` | 50 Benzinga headlines (ticker, date, title) | our direction labels and judge answers added |
| `app/data/demo.json` (P7.2 demo) | real weekly headline bundles for 3 windows (JNJ 2018-11-30, NFLX 2018-11-23, GOOGL 2019-06-21) and their **synthetic** A2 fakes (`synthetic_fake: true`) | grouped by week; split into headlines; fakes made by our rules |
| `results/qwen/r4_heldout_inputs.csv`, **git history only** (added `801c3f6`, present until `d93f98a`, removed in `24bd16d`) | the R4 GPU input: 15 MB, 33,410 rows of weekly FNSPID headline bundles for 10 stocks, Time-MMD Energy reports and **synthetic fakes** of both | grouped by week; prompts and judge questions built around them; fakes made by our rules |

**Git history is kept (Aryan, 2026-10-23).** Rewriting it would change the commit hashes that show
each pre-registration came before its results. The old file is shared under CC BY-NC 4.0 as above.
`python -m src.models.export_r4_inputs` rebuilds it and checks its sha256 (`cecbdc9b…`, the value
the Kaggle runs recorded).

**Which texts in that old file are fake.** It has no `synthetic` column and can't get one without
rewriting history, so this is its label. A text is a synthetic fake, not real news or a real
report, exactly when:

* `row_type = forecast` and `attacked = True`: 454 `flipped`, 133 `a1_selective`, 883
  `a2_forward`, 218 `a3_magnitude` rows (Energy and FNSPID). Where `attacked = False` the text
  equals the `real` row of that window or is empty (checked row by row); `real` is the real text,
  `none` is no text.
* `row_type = judge` / `judge_v2` and `real` is not in `sources`: 1,555 / 1,526 texts. If
  `sources` include `real`, the text is real.
* `row_type = probe`: 20 real Time-MMD reports, no fakes.

## 2. Domains

**Original domains**: the `fact` field only (`preds` is forecast text written by Time-MMD's own LLM,
not a real report).

| Domain | Freq | Target (`OT`) | Lookback | Text window | Train / val / test |
|---|---|---|---|---|---|
| Economy | monthly | U.S. goods trade balance | 8 | 1995-02 → 2024-03 | 244 / 36 / 70 |
| SocialGood | monthly | U.S. unemployment rate | 8 | 1994-02 → 2023-01 | 243 / 36 / 69 |
| Traffic | monthly | U.S. vehicle miles travelled | 8 | 2002-02 → 2024-03 | 186 / 27 / 53 |
| Health_US | weekly | % of outpatient visits for influenza-like illness | 24 | 2007-02 → 2020-01 | 470 / 68 / 134 |

Six other Time-MMD domains were looked at in P1 and dropped (too little text, empty reports, or no
up/down claims; §6).

**Held-out domains (R4)**, fixed on 2026-10-12 before any model output on them (§10g):

| Domain | What | Windows (with text) | Notes |
|---|---|---|---|
| **Energy** | U.S. weekly all-grades retail gasoline price (Time-MMD) | 665 (352) | Two-period lag (timing below). All text periods are test periods. σ from the 520 weeks before the text. *Easy:* the reports restate the target (same sign 99.4% of weeks). The prompt says "regular gasoline" but the target is all-grades (about $0.09 apart); disclosed in §10g Note 1; tests compare answers to the same prompt, so none changed. |
| **FNSPID** | Friday closes of NFLX, GOOGL, MRK, NVDA, PFE, JNJ, MU, FDX, PCG, LLY; text = that week's Benzinga headlines about the stock | 1,031 (1,006) | 2018-06 → 2020-05, lookback 24. Stocks chosen by a rule written first (most news days among S&P 500 members on 2018-06-01; `src/data/fnspid.py` checks it); 5 reserves. Prices as traded (split factors for NVDA, MRK, PFE). A bundle has the headlines from Friday to Thursday before the forecast. *Hard:* weak weekly trends, short noisy headlines. |

## 3. Splits

* Always in time order: train, then val, then test (70 / 10 / 20 over the text window;
  `src/data/splits.py`). Never random.
* Every threshold was tuned on val (v1 window and τ, clip k, v2's settings). Each test split was
  used once per plan. Nothing is tuned on the held-out domains.
* σ = the train split's population std (held-out: the 520 weeks before the text). It is the unit of
  DSS and of the gate's evidence.

## 4. Cleaning (`src/data/loader.py`, P1)

* Files are sorted by date (some Time-MMD files are newest-first or unsorted).
* Timing: a text belongs to the period of its **end** date, and row t only sees texts from t−1
  (Energy: t−2). Using the start date let yearly reports reach the same year's forecasts (37
  cases). After the fix, `check_timing` finds 0 problems in 1,497 texts (a regression test).
* 5 texts with wrong stored dates (3 Health_US season summaries, 2 Traffic week-53 reports) and
  Energy 2020-08-03 (under the two-period lag) are dropped by name (`loader.KNOWN_LEAKERS`; the
  loader stops if the list no longer matches).
* 7 reports with a mistyped year but correct figures are kept (`check_timing.KNOWN_YEAR_TYPOS`).
  Energy's 2023-12-25 report says "January 1, 2023" for January 1, 2024; also kept.
* SocialGood: 144 of 347 reports (41%) carry the heading "Objective facts about the influenza
  situation" because Time-MMD's summariser used the health prompt. They are dropped unless they
  give an unemployment figure; the 13 that do (8 from the COVID years) lose the wrong heading.
  347 → 216 reports. Using SocialGood as shipped means training on 41% off-topic text.
* A missing report is `has_text=False`, never an empty string, so it can't be confused with the
  `empty` condition.
* Health_US has 5 doubled dates (ISO week 53 vs week 1); the later row is kept.

## 5. Fake texts

Every fake is **synthetic**, for defence research only, and marked (`synthetic=True`, `attacked`,
`claimed_direction`; `texts/` is gitignored and rebuilt). Fakes come from **fixed rules applied to
the real report**, never from an LLM, because Model 5 and the judge are both Qwen. If a report can't
be edited cleanly it is marked *not attacked* with a reason; no self-contradicting fake is written.

| Condition | Phase | What changes |
|---|---|---|
| `empty`, `shuffled`, `cross_domain`, `time_shifted` | P3 | controls: no text / another date / another domain / a year earlier |
| `flipped` | P3 | direction words swapped, signed changes reversed, dependent numbers redone; Health_US levels mirrored around the baseline (2B − V) |
| `exaggerated` | P3 | same direction, change × 3 |
| A1 `a1_selective` | R3 | the P3 flip, only where it agrees with v1's trend check |
| A2 `a2_forward` | R3, R4 | real report plus one sentence predicting the opposite next period |
| A3 `a3_magnitude` | R3, R4 (Energy) | change × 10, implausible results removed |
| R4 versions | R4 | rules extended to Energy's weekly comparisons and FNSPID headlines (`src/attacks/heldout_rules.py`) |

Checks:
* P3: all 29 human-labelled directional reports flipped correctly; no new contradictions; 84% of
  reports could be flipped.
* R3 found inconsistent numbers in about 19% of P3's fakes. A scan (`src/attacks/consistency_r3.py`,
  §10f Amendment 1, run before any metric) removes flagged texts (`results/r3_text_exclusions.csv`);
  P5/P6 conclusions hold without them. A later check (C8), found after the metrics, is only a
  sensitivity analysis.
* R4: the scan was extended (`src/attacks/consistency_r4.py`; 1 text removed,
  `results/r4_text_exclusions.csv`). After it: Energy flip / A1 / A2 / A3 = 345 / 102 / 345 / 218;
  FNSPID flip / A1 / A2 = 108 / 31 / 538. FNSPID bundles whose edited headline keeps a "because"
  word (after, on, as …) are flagged for a planned sensitivity check (`results/r4_connector_flags.csv`).
* Not caught: meaning-level mismatches, such as a reason that no longer fits the reversed verb.

## 6. Hand labels

| File | What | By |
|---|---|---|
| `summary_directional_sample.csv` | 50 Time-MMD reports from before each test split, up / down / none | Aryan (P1) |
| `summary_directional_socialgood_energy.csv` | 25 SocialGood (filtered) + 25 Energy reports | Aryan (P1) |
| `summary_directional_fnspid.csv` | 50 FNSPID headlines, direction about the stock; Claude's labels beside them (96% agreement) | Aryan (R4, before the analysis) |
