# Data card — TextGuard

What data every result in this repository comes from, how it was cleaned and split, and what the
synthetic attack texts are. Full evidence and every number: `PROJECT_STATUS.md` §6 (data), §8
(attacks), §10 (pre-registrations).

**The raw datasets in `data/` are not redistributed.** `python -m src.data.download` fetches each
source at a pinned revision and checks every file against `src/data/manifest.json` (sha256 of the
raw bytes). A changed upstream file fails the check. Small FNSPID-derived extracts *are* in the
repository and its history. They are redistributed under FNSPID's CC BY-NC 4.0 licence (§1a).

**Line endings are part of the data (found by R7's clean-room test).** Every result was computed
from a Windows `git clone` of Time-MMD (`core.autocrlf=true`). That clone turned every LF into CRLF,
including the line breaks *inside* report texts. **The exact prompts the GPU models answered
therefore contain Windows line endings (`\r\n`) inside the report text.** On an LF copy of
Time-MMD, 166 Model 5 prompts differ, and the loaders refuse the committed answers because the
prompt fingerprints no longer match. `download.py` writes Time-MMD in exactly the original form on
every OS: it converts each lone LF to CRLF, and the result is **byte for byte identical to all 30
files of the original clone** (the manifest hashes raw bytes; an earlier version normalised line
endings before hashing and hid the difference). Get Time-MMD with the script, not with
`git clone`. On Linux / macOS a clone gives LF and fails the check.

## 1. Sources and licences

| Source | Used for | Pinned at | Licence |
|---|---|---|---|
| **Time-MMD** (Liu et al., NeurIPS 2024 Datasets & Benchmarks), `github.com/AdityaLab/Time-MMD` | P1–R3 (Economy, SocialGood, Traffic, Health_US) and R4's held-out **Energy** | commit `00281e2` (2025-01-14) | **None stated.** The repository has no LICENSE file and its README gives no terms. We therefore do not redistribute it and ask users to obtain it from the authors and cite the paper (arXiv 2406.08627). |
| **FNSPID** (Dong et al., KDD 2024), `huggingface.co/datasets/Zihan1004/FNSPID` | R4's held-out **FNSPID** domain (10 S&P 500 stocks, Benzinga headlines + daily closes) | revision `bf9189c` | **CC BY-NC 4.0**: non-commercial use only, with attribution. Raw files are not redistributed; the derived extracts listed in §1a are, under the same licence. |
| S&P 500 historical membership, `github.com/fja05680/sp500` | FNSPID stock selection (membership on 2018-06-01) | commit `a2430f2` | MIT (repository) |
| MM-TSFlib (Liu et al.), `github.com/AdityaLab/MM-TSFlib` | DLinear / PatchTST network code for Models 1–4 (retraining only) | commit `e789ce7` | MIT |

**What the repo does contain that is derived from these sources** (all small, for audit and
verification):
* 150 hand-labelled report excerpts: the P1 directional-claim samples, 50 + 50 Time-MMD reports
  (`results/summary_directional_*.csv`); and 50 FNSPID headlines with Aryan's labels
  (`summary_directional_fnspid.csv`, `summary_r4_judge_labels_rows.csv`);
* model outputs and judge answers, which hold numbers and answer words, not the texts. The report
  text in each prompt is identified only by its fingerprint.

## 1a. FNSPID attribution and what is redistributed

> **FNSPID** — Zihan Dong, Xinyu Fan and Zhiyuan Peng. *FNSPID: A Comprehensive Financial News
> Dataset in Time Series.* KDD 2024 (arXiv 2402.06698). `huggingface.co/datasets/Zihan1004/FNSPID`,
> revision `bf9189c`. Licensed under **CC BY-NC 4.0**
> (<https://creativecommons.org/licenses/by-nc/4.0/>). **Non-commercial use only.**

CC BY-NC 4.0 permits non-commercial redistribution with attribution and an indication of changes.
This repository redistributes the following FNSPID-derived material on those terms. It may be used
for **non-commercial purposes only**, and the attribution above must be kept. The MIT licence of the
code does not apply to it.

| Where | What | Changes made |
|---|---|---|
| `results/summary_directional_fnspid.csv`, `results/summary_r4_judge_labels_rows.csv` (current tree) | 50 Benzinga headlines (ticker, date, title) | added our direction labels and judge readings |
| `results/qwen/r4_heldout_inputs.csv` in **git history only**: added in `801c3f6`, present through `d93f98a`, removed from the tree in `24bd16d` (R7) | the R4 held-out GPU input, 15 MB, 33,410 rows: weekly bundles of FNSPID headlines for 10 stocks, Time-MMD Energy reports, and the **synthetic fakes** made from both | headlines grouped into weekly bundles; prompts and judge questions built around them; fakes written by our rules (below) |

**Decision (Aryan, 2026-10-23): git history is not rewritten.** A rewrite would change the commit
hashes cited throughout `PROJECT_STATUS.md` and the pre-registrations, which are the evidence that
each pre-registration preceded its results. The historical file is redistributed under CC BY-NC 4.0
as above. The current tree no longer holds it: `python -m src.models.export_r4_inputs` regenerates
it and asserts its sha256 (`cecbdc9b…`, the value the Kaggle runs recorded).

**Synthetic texts in that historical file.** The file predates the `synthetic` column of `texts/`
and cannot be relabelled without rewriting history, so this paragraph is its label (`ETHICS.md`
repeats the rule). A text in it is a **synthetic research artefact**, written by our
deterministic rules and not genuine news or a genuine report, exactly when:

* `row_type = forecast` and **`attacked = True`**: 454 `flipped`, 133 `a1_selective`,
  883 `a2_forward` and 218 `a3_magnitude` rows (Energy and FNSPID). On every forecast row with
  `attacked = False`, the text is identical to the `real` row of the same window or is empty
  (checked row by row). `real` = the genuine text and `none` = no text.
* `row_type = judge` / `judge_v2` and **`real` is not among the `sources`**: 1,555 / 1,526 texts.
  A text whose `sources` include `real` is genuine.
* `row_type = probe`: 20 genuine Time-MMD reports (original domains), no fakes.

## 2. Domains

**Original domains** (`source="report"`, the curated `fact` text; never `preds`, which is text
written by the Time-MMD authors' LLM, not real reports):

| Domain | Freq | Target (`OT`) | Lookback | Text window | Train / val / test periods |
|---|---|---|---|---|---|
| Economy | monthly | U.S. trade balance (goods) | 8 | 1995-02 → 2024-03 | 244 / 36 / 70 |
| SocialGood | monthly | U.S. unemployment rate | 8 | 1994-02 → 2023-01 | 243 / 36 / 69 |
| Traffic | monthly | U.S. vehicle miles travelled | 8 | 2002-02 → 2024-03 | 186 / 27 / 53 |
| Health_US | weekly | % outpatient visits for influenza-like illness | 24 | 2007-02 → 2020-01 | 470 / 68 / 134 |

Six other Time-MMD domains were checked and dropped in P1 (low text coverage, empty reports or
no directional claims; §6).

**Held-out domains (R4)**, chosen and **frozen on 2026-10-12, before any model output existed on
them** (the pre-registration is §10g "Held-out freeze"):

| Domain | What | Windows (with text) | Design |
|---|---|---|---|
| **Energy** | U.S. weekly all-grades retail gasoline price (Time-MMD) | 665 (352) | Lag 2 (see timing). Every text period is a test period (zero-shot targets, training-free defences). σ = std of the 520 weeks before the text window. *Easy case:* reports restate the target (same sign 99.4% of weeks). The prompt calls the series "regular gasoline" while the target is all-grades (≈ $0.09 apart). This wording error is disclosed in §10g Note 1; every comparison is paired within one prompt, so no test changed. |
| **FNSPID** | Weekly Friday closes of NFLX, GOOGL, MRK, NVDA, PFE, JNJ, MU, FDX, PCG, LLY; text = that week's Benzinga headlines about the stock | 1,031 (1,006) | 2018-06 → 2020-05, lookback 24. Stocks picked by a pre-stated rule (most news days among S&P 500 members on 2018-06-01; `src/data/fnspid.py` re-derives and asserts it); 5 reserves. Prices are as traded (split factors for NVDA, MRK, PFE). A bundle holds headlines dated Fri(k−1) … Thu(k), all before the forecast origin. *Hard case:* weak weekly trends, short noisy headlines. |

## 3. Splits

* **Always chronological**: train → val → test, 70 / 10 / 20 over each domain's text window
  (`src/data/splits.py`); never random.
* **All thresholds were tuned on val only** (TextGuard v1 window/τ, clip k, TextGuard v2's V3
  settings). The test split was evaluated once per pre-registration. The held-out domains have no
  tuning split: nothing is tuned on them.
* σ = population std of the train split (held out: the 520 weeks before the text window). It is
  the unit of DSS and of the gate's evidence strength.

## 4. Cleaning (`src/data/loader.py`, decided in P1)

* **Sorting**: Time-MMD files are not consistently ordered (Climate newest-first, Economy
  unsorted), so the loader sorts by date once.
* **Timing**: each text is anchored on its **end** date and row t receives only texts placed in
  period t−1 (Energy: t−2). Anchoring on the start date let annual reports feed the same year's
  February forecast (37 violations). After the fix `check_timing` finds **0 metadata and 0 content
  violations** in 1,497 texts (a regression test).
* **Known leakers**: 5 texts whose stored dates are wrong (3 Health_US season summaries, 2 Traffic
  week-53 reports), plus Energy 2020-08-03 under lag 2, are dropped by name
  (`loader.KNOWN_LEAKERS`; the loader raises if the list stops matching the data).
* **Year typos kept**: 7 reports state the wrong year in the text, but the figures are right for
  their period (`check_timing.KNOWN_YEAR_TYPOS`).
* **SocialGood influenza headers**: 144 / 347 (41%) of SocialGood reports are headed "Objective
  facts about the influenza situation". Time-MMD's summariser was run with the Health prompt on
  unemployment reports. Influenza-headed texts are dropped unless they state an unemployment
  figure; the wrong header is stripped from the 13 that do (8 of them COVID-era). Result:
  347 → 216 reports. Anyone using SocialGood text as-is trains on 41% off-topic input.
* **Missing text is explicit** (`has_text=False`), never an empty string, so "no report existed"
  is never confused with the `empty` attack.
* **Health_US**: 5 duplicate dates (ISO week 53 vs week 1); the later row is kept.

## 5. Synthetic attack texts (fake reports)

**Every fake is synthetic, for defence research only, and labelled** (`synthetic=True`,
`attacked`, `claimed_direction` in every generated file; `texts/` is gitignored and regenerated).
All fakes are written by **deterministic rules applied to the genuine report**, never by an LLM,
because Model 5 and the judge are both Qwen (circularity). Each makes a minimal, auditable edit.
A text that cannot be edited coherently is marked **not attacked** with a reason; a
self-contradictory fake is never emitted.

| Condition | Phase | Edit |
|---|---|---|
| `empty`, `shuffled`, `cross_domain`, `time_shifted` | P3 | controls: no text / another date's / another domain's / one year earlier |
| `flipped` | P3 | direction words swap, signed changes flip, dependent numbers recomputed; Health_US levels mirrored across the stated baseline (2B − V) |
| `exaggerated` | P3 | same direction, change × 3 |
| A1 `a1_selective` | R3 | P3's flip, only where the flipped claim agrees with TextGuard v1's trend evidence |
| A2 `a2_forward` | R3, R4 | genuine report kept + one appended sentence predicting the reverse next period |
| A3 `a3_magnitude` | R3, R4 (Energy) | change × 10, with a plausibility filter |
| R4 flip / A1 / A2 / A3 | R4 | extended engines for Energy's week-over-week comparators and FNSPID headline cues (`src/attacks/heldout_rules.py`) |

**Validation and exclusions:**
* P3: the 29 / 29 human-labelled directional reports were flipped correctly; 0 new contradictions
  by the arithmetic checks; coverage 84% (flip).
* R3 found that P3's engine left internally inconsistent numbers in ~19% of fakes. A text-validity
  scan (`src/attacks/consistency_r3.py`, §10f Amendment 1, made before any metric) excludes flagged
  texts from the attacked sets (`results/r3_text_exclusions.csv`). An exploratory recheck shows
  P5/P6's conclusions do not depend on them. A later check (C8) found *after* metrics was not
  adopted and is reported as a sensitivity analysis only.
* R4: scan extended to the new styles (`src/attacks/consistency_r4.py`; 1 exclusion,
  `results/r4_text_exclusions.csv`). Coverage after the scan: Energy flip / A1 / A2 / A3 =
  345 / 102 / 345 / 218; FNSPID flip / A1 / A2 = 108 / 31 / 538. FNSPID bundles whose edited
  headline keeps a causal connector are flagged for a pre-stated sensitivity analysis
  (`results/r4_connector_flags.csv`).
* Known limitation: lexical and numeric contradictions are prevented or scanned for; *semantic*
  ones (a reason that no longer fits the flipped verb) are not fully caught.

## 6. Human labels

| File | What | By |
|---|---|---|
| `summary_directional_sample.csv` | 50 Time-MMD reports (before each test split), up / down / none | Aryan (P1) |
| `summary_directional_socialgood_energy.csv` | 25 SocialGood (filtered) + 25 Energy reports | Aryan (P1) |
| `summary_directional_fnspid.csv` | 50 FNSPID headlines, direction about the stock; Claude's labels beside, 96% agreement | Aryan (R4, before the analysis) |
