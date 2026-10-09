# PROJECT_STATUS.md — TextGuard

> **Living document. Update at the end of every session, and always at a phase boundary.**
> Claude Code reads this to recover context after a session reset.
> Keep it factual: what is actually done, what actually broke, what the numbers actually are.

**Current phase:** P6b — tune on val, evaluate on test (P6a complete: TextGuard built, judge run back)
**Last updated:** 2026-10-09

> 👤 **Solo project.** All work by Aryan.
> 📌 **Phase-gated, no deadlines.** See `PHASES.md`.

---

## 1. Phase board

| Phase | Builds | Required? | Status |
|---|---|---|---|
| P0 | Environment, repo, folder skeleton | ✅ | ✅ Done |
| P1 | Data pipeline — loader, splits, domain selection | ✅ | ✅ Done |
| P2 | Model wrappers — 5 models, one interface | ✅ | ✅ Done |
| P3 | Attack module — 6 text conditions | ✅ | ✅ Done |
| P4 | Metrics module — DSS, Flip Rate, etc. | ✅ | ✅ Done |
| P5 | Experiment runner — screening + attack | ✅ | ✅ Done (P5a + P5b) |
| P6 | TextGuard defense | ✅ Core | 🔵 P6a done (built + judge run); P6b next |
| P7 | Auditor, demo, extras | ⬜ Optional | ⚪ Pending |
| P8 | Paper & release | ✅ | ⚪ Pending |

Legend: ⚪ pending · 🔵 in progress · ✅ done · ⚠️ blocked · ❌ cut

**P0–P6 + P8 = complete paper. P7 is upside.**

---

## 1a. ▶ START HERE — handoff for the next session (P6b)

> Written 2026-10-09 at the end of P6a. Replace this section at the end of P6.

**P6 is split:** **P6a** = build TextGuard + run the Qwen direction judge (✅ done);
**P6b** = tune on **val only**, evaluate on test — **exactly** as pre-registered in **§10b**
(commit `9838239`, alone, before any TextGuard output) **+ Amendment 1** (`01184db`, before any
judge output: per-domain FBR, expected higher on Traffic and Economy). Read §10b in full before
writing code: it fixes every definition, the grid, the selection rule, the Holm family (m = 20,
T1–T4) and the ablations. Anything not in §10b is exploratory and must be labelled so.

**P5 result in one line:** the rule-based flip attack moves Model 5 (Qwen, early fusion)
+0.46σ toward the lie and flips its direction in 24% of attacked windows (MSE +92%); Models
3–4 (late fusion) are moved by text but not by its direction (DSS < 0.02σ). §9 "P5 results".

### P6a — done (2026-10-09)

* ✅ Pre-registration §10b (`9838239`) + Amendment 1 (`01184db`), each committed alone.
* ✅ `src/textguard/` built and tested (259 tests): `trend_detector.py`, `gate.py`,
  `baselines.py`, `textguard.py`, `text_direction.py`, `judge_prompt.py`,
  `export_judge_inputs.py`. Worked example reproduced on `DummyForecaster` (g = 0.1, final 115,
  warning) and on a duck-typed third-party class.
* ✅ CFA (issue #4): **not run** — decided before any test result → T5 dropped, **m = 20**.
* ✅ **Judge run (Colab T4, 2026-10-09 01:03–01:11 UTC, 8 min):** 2490 answers, **probe 40/40**,
  setup string **identical** to P5's (`Tesla T4 | torch 2.11.0+cu130 | transformers 5.18.0 |
  Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306`). All five answer words are
  single tokens (up 454, down 2923, none 6697, above 48432, below 53177) → one forward pass per
  question. `load_judgements()` verified locally: 2490 rows, every fingerprint, one setup,
  `probe_ok` True. Confidence range 0.38–1.00. Backed up with `git add -f`.
* Answer counts (descriptive only — **no accuracy computed yet**, that is P6b):

  | Domain / question | up / above | down / below | none |
  |---|---|---|---|
  | Economy change | 222 | 186 | 64 |
  | SocialGood change | 59 | 179 | 122 |
  | Traffic change | 176 | 146 | 18 |
  | Health_US level | 311 | 298 | 50 |
  | Health_US change | 274 | 333 | 52 |

  (Counts over all distinct exported texts — real, every altered condition and the human-labelled
  reports — so they are not a class balance of any one condition.)

### P6b — what to build (suggested: `src/analyze_p6.py` + `src/figures_p6.py`, tests in `tests/test_analyze_p6.py`)

**Inputs — all on disk, nothing to re-run:**

| What | Where | Join keys / notes |
|---|---|---|
| Judge answers | `load_judgements()` (`results/qwen/p6_judge_predictions.csv` + `p6_judge_run_meta.json` ✅ in git; `results/qwen/judge_inputs.csv` gitignored — regenerate with `python -m src.textguard.export_judge_inputs`, deterministic; the loader re-verifies every fingerprint) | `QwenJudge(domain, data=df)(text)` → `list[Claim]`; key = (domain, question, text.strip()). Economy already mapped deficit → balance (`direction`; `literal` = about the subject, for the human-label check) |
| Texts per condition | `texts/<condition>.csv` (`src.attacks.common.load`) | domain, split, sample_id; `attacked`, `claimed_direction`, `real_direction`, `variant` |
| Lookback windows (for the gate) | `results/qwen/qwen_inputs.csv` (`lookback` JSON, raw units; same 493 val + test windows) or `src.data.windows.split_windows` | domain, split, sample_id |
| Models 3–4 A / B | `results/p5_predictions.csv` (seed-mean first: `metrics.seed_mean`); A = `empty` | model, domain, split, condition, sample_id; `y_true, y_last, train_std, attacked, claimed_direction` |
| Model 5 A / B | `run_experiments.qwen_rows()` (3451 rows; `none` → `empty` already) | same schema; NaN = parse / scale failure |
| P5 analysis helpers | `analyze_p5.load_long`, `wide`, `year_mask`, `_cell`, `_groups` | reuse, do not copy |
| Human labels | `results/summary_directional_sample.csv` (50, `manual` ∈ up/down/none, about the literal subject) + `summary_directional_socialgood_energy.csv` (SocialGood rows only = supplementary 25) | judge text = `fact.strip()`, domain = row's domain, question `change`, compare `literal` |

**Steps, in order:**

1. **Gate table** (raw, save to `results/p6_gates.csv`, gitignored): for every window × condition
   (real, flipped, exaggerated, + the controls) × split × w ∈ grid: claims (judge), evidence,
   g. σ = `train_std`. Gate is model-independent. No text → g = 1.
2. **Defended forecasts** per model: `baselines.textguard / clip / always_trust / never_trust`
   on A, B. Save per-window finals (`results/p6_defended.csv`, gitignored).
3. **Tune on val only** (§10b): M5, flipped, pooled; (w, τ) with val FBR ≤ 10% → min val DSS;
   ties → lower FBR, smaller τ, smaller w. clip k likewise. Record the choice before opening test.
4. **Test:** curve + operating point (pooled + per domain, Amendment 1), T1–T4 + Holm (assert
   m = 20), secondary MSE / damage vs never_trust per domain, M3/M4 no-harm, exaggerated, COVID
   2020 FBR, judge accuracy (human labels; known `claimed_direction` of attacked
   flipped/exaggerated by domain × variant — Health_US level variant ↔ `level` answer, change
   variant ↔ `change` answer; none-rate on cross_domain), ablations.
5. Figures (curve), tables `results/summary_p6_*.csv`, fill the §9 "P6" tables, issue #10.

**Definitions to get right (all in §10b):** FBR = share of **genuine** (real-text, has-report)
windows with **applied** weight g′ < 0.5; DSS / Flip Rate with **both** forecasts through the
same defence; block mode = g < τ → A; B NaN → A, A NaN → NaN (failure, reported).

### What exists and where (P5 artifact map, still valid)

| Artifact | Path | In git? | How to regenerate |
|---|---|---|---|
| Splits / windows | `src/data/splits.py`, `src/data/windows.py` (h=1; lookback 8 monthly / 24 weekly) | code | — |
| Models 1–4 predictions, **real + empty only**, val + test, 3 seeds | `results/p2_baselines_predictions.csv` (`config=main`; `condition` = real / empty; Models 1–2 real only) | ❌ (raw) | `python -m src.models.train_baselines` (resumable; ~1 h CPU; units in `results/p2_baselines/`) |
| Models 1–4 checkpoints | `checkpoints/<model>/<domain>/main_seed{2021,2022,2023}.pt` (model ∈ DLinear, PatchTST, DLinear+BERT, PatchTST+BERT) | ❌ | same runner (deterministic) |
| BERT embedding cache | `cache/text_emb/` | ❌ | `python -m src.models.embed_texts` (real texts); new texts are embedded on first use |
| Model 5 predictions, **real + none only**, val + test | `results/qwen/p2_qwen_predictions.csv` + `qwen_inputs.csv` | ✅ predictions (backed up), inputs ❌ | Colab notebook (§7b); inputs: `python -m src.models.export_qwen_inputs` |
| All 7 text conditions, val + test (493 windows each) | `texts/<condition>.csv` (real, empty, shuffled, cross_domain, time_shifted, flipped, exaggerated) | ❌ | `python -m src.attacks.generate` (deterministic) |
| Metrics | `src/metrics.py` (§9) | code | — |
| **P5a: Models 1–4 × 7 conditions**, val + test, 3 seeds (41,412 rows) | `results/p5_predictions.csv` (units in `results/p5_units/`) | ❌ (raw) | `python -m src.run_experiments` (resumable; ~15 min CPU incl. MLflow) |
| **P5a: Qwen inputs** — 986 real/none baseline rows (40 flagged as probes) + 2465 condition rows | `results/qwen/qwen_inputs_p5.csv` | ❌ | `python -m src.models.export_qwen_inputs --p5` |
| **P5b: analysis tables + figures** | `results/summary_p5_*.csv` (14), `paper/figures/p5_dss.*`, `p5_flip_rate.*` | ✅ | `python -m src.analyze_p5` (~20 s; `--no-figures`) |
| **P6a: judge answers** (2490, probe 40/40) | `results/qwen/p6_judge_predictions.csv` + `p6_judge_run_meta.json`; inputs `results/qwen/judge_inputs.csv` | ✅ answers backed up (`git add -f`); inputs ❌ | `notebooks/p6_direction_judge.ipynb` (§1a); inputs: `python -m src.textguard.export_judge_inputs` |
| **P5a: Qwen predictions, 5 altered conditions** (2215 generated) | `results/qwen/p5_qwen_predictions.csv` + `p5_qwen_run_meta.json`; all 3451 Model 5 rows via `run_experiments.qwen_rows()` | ✅ backed up (`git add -f`) | `notebooks/p5_qwen_conditions.ipynb` (steps below) |

### What P5a must build

0. **Pre-registered comparison family** (§10a) — committed on its own, before any P5 prediction.
1. **Embeddings** for every condition text (val + test) into `cache/text_emb/`. The cache holds
   *raw* BERT vectors; standardisation happens inside each Model 3/4 checkpoint with the
   train-only mean/std fitted in P2.3, so attacked text can never refit it. BERT is unloaded
   afterwards.
2. **Runner** `src/run_experiments.py` (resumable unit = model × domain; save after each unit;
   skip finished ones — the `train_baselines.run` pattern). Models 1–4 × 7 conditions (real,
   empty, shuffled, cross_domain, time_shifted, flipped, exaggerated) × 3 seeds × val + test,
   **inference only** (load the P2 checkpoints, never retrain). Every raw prediction:
   `model, domain, split, condition, seed, sample_id, y_true, y_pred, y_last, train_std`.
   MLflow run per (model, domain, condition, seed). Assert Models 1–2 identical across all
   conditions (`metrics.is_text_blind`) and Models 3–4 `real` / `empty` identical to P2.
3. **Model 5 (issue #29):** export the 5 altered conditions (one row per window × condition,
   same format as P2.4), regenerate the Colab notebook (BatchEncoding fix, `revision` recorded
   — issue #28), same prompt / parser / off-scale rule. Windows whose condition text equals the
   real text (or is empty) reuse P2.4's `real` (or `none`) prediction and are skipped on Colab.
   `QwenForecaster` learns the new conditions. Then **wait for the Colab CSV**.

**No screening or attack metrics in P5a** — that is P5b.

### P5a status (2026-10-09)

* ✅ Step 0 — family pre-registered and pushed **alone** (commit `5792daf`, 04:22 IST),
  before any P5 prediction existed.
* ✅ Step 1 — 732 new embeddings (flipped 361, exaggerated 371; shuffled / cross_domain /
  time_shifted were already cached — they reuse real reports). 0 truncated (max 488 tokens).
  All 24 Model 3/4 checkpoints' scalers equal the train-text mean/std recomputed now
  (max |diff| 0.0) → nothing was refit.
* ✅ Step 2 — 112 units, 41,412 predictions, 0 NaN, 336 MLflow runs (`p5_experiments`).
  BERT never loaded (all cache hits). Models 1–2 bit-identical across all 7 conditions
  (48 model × domain × seed × split groups). `real` (and Models 3/4 `empty`) reproduce P2's
  saved predictions exactly (max |diff| 0.0). **No metric has been computed.**
* ✅ Step 3 — Colab T4 run 2026-10-09 (23:45–00:00 UTC, 14 min): **probe 40/40 identical to
  P2.4 → reuse**; 2215 prompts generated, 0 parse failures. Same stack as P2.4 (torch
  2.11.0+cu130, transformers 5.18.0, T4); revision **989aa7980e4cf806f80c7fef2b1adb7bc71aa306**
  (issue #28). `qwen_rows()`: **3451 rows (986 real/empty + 2465 conditions), one setup class**
  (P2.4's, probe-certified): 1236 rows carry P2.4's answer (986 baseline + 151 `real` + 99
  `none` reuses), 2215 the P5 run's. All fingerprints verified. 16 scale (unit) failures in
  total, 3 of them the known P2.4 Traffic ones — broken down in P5b, not here.
* **P5a complete. Next: P5b** (analysis against §10a)

### Colab run (Model 5, issue #29)

1. colab.research.google.com → File → Upload notebook → `notebooks/p5_qwen_conditions.ipynb`.
2. Runtime → Change runtime type → **T4 GPU**.
3. Upload `results/qwen/qwen_inputs_p5.csv` to `/content` (or `MyDrive/textguard/`).
4. Runtime → Run all (allow Drive access in cell 2). Cell 6 prints the probe result:
   * `40/40 identical to P2.4` → reuse; 2215 prompts, ~15 min;
   * anything else → **"REGENERATE EVERYTHING, real and none included"**; 3451 prompts,
     ~22 min. Nothing to change by hand — the notebook does it.
   On a disconnect: reconnect, run cells 1–6, then cell 7 resumes. If the runtime came back
   with a different torch / transformers / GPU, cell 7 refuses to append to the old file
   (one file = one setup): rename or delete `p5_qwen_predictions.csv` and Run all again.
5. Put `p5_qwen_predictions.csv` and `p5_qwen_run_meta.json` into `results/qwen/`.
6. Locally: `python -c "from src.run_experiments import qwen_rows; print(len(qwen_rows()))"`
   — verifies every fingerprint, fills the reused rows and checks there is one setup class;
   expect 3451 (986 real/empty + 2465 conditions) either way. Then P5b.

### Rules a fresh session must not break

* **Never tune on test.** Any P5 choice (thresholds, which models to attack) is made on val
  or fixed in advance; test is for reporting.
* **Seeds:** `metrics.seed_mean` before any metric (Models 1–4); Model 5 = one greedy run.
* **Condition names:** Models 3–4 call no-text `empty`; Model 5 calls it `none`. Same thing
  (§6a) — map `none → empty` when joining.
* **NaN pairing** is inside every metric; report `failure_rates` next to every metric
  (Model 5: 3 Traffic real-text scale failures, issue #27).
* **σ** = `train_std` per domain (already in every predictions file). **y_last** = last
  lookback value (`windows.X[:, -1]` or `json.loads(lookback)[-1]` in `qwen_inputs.csv`).
* **Not every window has text:** compare text conditions only on windows where `real` has
  text (SocialGood 46/69 and Health_US 130/134 test windows do), else report separately
  (§6 caveat).

### Interpretation notes already decided

* DSS ≈ 0 / gap ≈ 0 on Models 3–4 is a **finding, not a failure** (CLAUDE.md §4a).
* Traffic: report **with and without 2020** (issue #23) — *hypothesis was* that genuine
  COVID-era text hurts Model 3 on test. **P5b: not supported** — the harm persists without
  2020 (§9). Traffic is the likeliest genuine text user (#22).
* SocialGood / Health_US are sparse-text domains (#9) — don't attribute a low gap there to
  fusion type alone.
* Model 5 is a weak forecaster but text moves it in 65–78% of windows → expected attack target.

---

## 2. Decisions locked in

| Decision | Value |
|---|---|
| Project | TextGuard (replaces QRShield) |
| Category | Cybersecurity, SDG 16 |
| Team | **Solo — Aryan only** |
| Pacing | **Phase-gated, no deadlines** |
| Framing | **Attack + defense paper** |
| Novelty claim | Semantic directional text attack + DSS + training-free defense |
| Primary dataset | Time-MMD |
| Secondary dataset | FNSPID — optional, P7 |
| Models | 1–5 required (early + late fusion); Model 6 optional in P7 |
| Text encoder | `bert-base-uncased`, **frozen, all 12 layers, mean-pooled over real tokens**; one 768-d vector per text, cached in `/cache/` (gitignored); never saved in checkpoints |
| Late fusion (Models 3/4) | `ŷ = TS(x) + w·MLP(z(e_text))`, w = 0.1 — MM-TSFlib's fusion **minus `norm()` and `prior_y`**; **TS = Model 1/2 of the same seed, frozen**; only the text head trains, under the P2.2 protocol (§7a) |
| `text=None` (Models 3/4) | Text branch absent → `ŷ = TS(x)` = **Model 1/2 exactly**. So the Usage Gap = MSE(M1/2) − MSE(M3/4, real) and B − A = w·MLP(z(e)) (§7a) |
| CFA (Lee et al. 2026) | Primary defense baseline |
| Distribution | GitHub toolkit + Streamlit demo (demo = P7) |
| Streamlit CSV upload | ❌ Cut from scope |
| Audit report generator | ✅ In scope — P7 |
| Text field | **`fact` only** — `preds` is LLM-generated by the Time-MMD authors; attacking machine-written forecasts is a different threat model (see §6) |
| Text source | `report` by default, `search` selectable — MM-TSFlib uses `search` (see §6) |
| Text alignment | `anchor="end"` + lag 1: each text placed in the period containing its end date; row t gets period t−1's texts (see §6) |
| Known report defects | 5 leakers dropped (Health_US 3, Traffic 2); SocialGood influenza junk filtered; 7 year-typos **kept** (see §6) |
| Attack variants | **change-flip** in practice for Economy, Traffic **and SocialGood** (its reports state changes, not levels); **level-flip only for Health_US** (§8) |
| Attack generator | **Deterministic rules** (`src/attacks/rules.py`), **not Qwen** — avoids Qwen-on-Qwen bias (Model 5) and same-model circularity with the P6 judge (#25); minimal, exact, auditable edit. Paper: a conservative, minimal-edit attacker. LLM-written attack = optional P7 extra (§8) |
| Attack direction for DSS | `claimed_direction` recorded per fake (+1/−1, about the forecast target) — **DSS uses the known attack direction, never a classifier** |
| TextGuard gate | Generalised: "text claim vs numerical evidence", with a **change branch** and a **level branch** (P6) |
| Text direction classifier | **Qwen-based**, not rule-based — keyword heuristic agreed with humans only 72% |
| Forecaster interface | `predict(numbers, text) -> ndarray (horizon,)`; `Forecaster` Protocol (duck-typed) + optional `BaseForecaster` ABC — see §6a |
| Model input | **Univariate** 1-D window in raw units — avoids Sridhar et al.'s co-shipped-column confound; scalers fit on train only |
| Blank text | `""` / whitespace → `None`: `empty` ≡ `has_text=False` ≡ TextGuard's A |
| Prediction failures | **NaN**, never raise or drop; NaN-pairing rule for P4/P6 (§6a) |
| Model 5 software setup (P5) | **One setup class per comparison.** P2.4 answers are reused only if a 40-prompt probe reproduces them exactly on today's Colab stack; otherwise real, none and all conditions are regenerated together. Every row records `setup`; `qwen_rows()` refuses mixed classes — a stack change must never be measured as attack effect (decided 2026-10-09) |
| Scoring horizon | **horizon = 1** for main experiments; DSS and Flip Rate scored at h=1 |
| Attack-target selection (P5) | **by text sensitivity, not Text Usage Gap** — sensitivity is what makes a model attackable (§9) |
| Training protocol (main) | **Val-tuned:** up to 200 epochs, patience 20, lr ∈ {1e-4, 1e-3, 1e-2} chosen per domain × model by val loss (seed 2021), then 3 seeds. **Models 3 and 4 (P2.3) must reuse it exactly** — see §7 |
| Training protocol (repro) | MM-TSFlib's original recipe (10 epochs, patience 5, lr 1e-4), published horizon, PatchTST as shipped — only for the reproduction column |
| PatchTST patching (main) | `patching="window"`: patch_len = L/2, stride = L/4 — bypasses MM-TSFlib's patch_len clamp (§7) |

---

## 3. Novelty claim

> We present the first study of **semantically misleading text attacks on numerical
> multimodal time-series forecasters**, introduce **DSS** to measure directional
> manipulation, and propose **TextGuard** — a **training-free, inference-time, black-box**
> defense.

Every word is load-bearing:
- "semantically misleading" excludes PA-RNet (zero-mean noise only)
- "numerical multimodal TSF" excludes Corroboration Illusion (RAG events) and AutoRedTrader (trading agents)
- "training-free, inference-time, black-box" excludes CFA (trained) and PA-RNet (architecture change)

Competing-work detail lives in **CLAUDE.md §4a**. Confirm the claim holds when writing
Related Work in P8, and re-run the arXiv search then.

---

## 4. Phase checklist — P0 (complete)

- [x] Folder `C:\Users\Aryan\textguard` created, opened in VS Code
- [x] `git init` + `.gitignore`
- [x] Remote set to `https://github.com/aryanwaingankar4/textguard.git`, first push done
- [x] Python venv created and activated
- [x] Dependencies installed, `requirements.txt` pinned (torch 2.14.0+cpu, mlflow 3.16.1)
- [~] Colab extension installed but unreliable — superseded: use colab.research.google.com directly for GPU work (see open issue #5)
- [x] `src/` folder skeleton created (docstring-only stubs; no experiment code)
- [x] `mlflow ui` runs locally

---

## 5. Build status

| Module | Phase | Status |
|---|---|---|
| `src/data/loader.py` | P1 | ✅ (12 tests) |
| `src/data/splits.py` | P1 | ✅ (14 tests) |
| `src/data/check_directional.py` | P1 | ✅ run, 58% |
| `src/data/check_timing.py` | P1 | ✅ run — FAILS with current default |
| `src/data/inspect_domains.py` | P1 | ✅ (3 tests) |
| `src/models/base.py` | P2 | ✅ (27 tests) |
| `src/data/windows.py` | P2 | ✅ (lookback windows; val/test take context from earlier splits) |
| `src/models/_mmtsflib.py` | P2 | ✅ (imports MM-TSFlib networks only, not its data pipeline) |
| `src/models/neural.py` | P2 | ✅ (shared train/scale/checkpoint for DLinear + PatchTST) |
| `src/models/persistence.py` | P2 | ✅ (sanity floor) |
| `src/models/dlinear.py` | P2 | ✅ trained (§7) |
| `src/models/patchtst.py` | P2 | ✅ trained (§7) |
| `src/models/train_baselines.py` | P2 | ✅ resumable runner — saves each config × domain × model unit, skips finished ones |
| `src/models/text_encoder.py` | P2 | ✅ frozen BERT + on-disk embedding cache |
| `src/models/embed_texts.py` | P2 | ✅ pre-embeds all texts; truncation report (0 / 1305) |
| `src/models/late_fusion.py` | P2 | ✅ fusion mixin shared by Models 3/4 (15 tests) |
| `src/models/dlinear_bert.py` | P2 | ✅ Model 3 (thin module → `late_fusion`) |
| `src/models/patchtst_bert.py` | P2 | ✅ Model 4 (thin module → `late_fusion`) |
| `src/models/qwen_prompt.py` | P2 | ✅ prompt + parser, single source of truth |
| `src/models/export_qwen_inputs.py` | P2 | ✅ 493 val + test windows → one CSV |
| `notebooks/p2_qwen_forecaster.ipynb` | P2 | ✅ run on Colab T4 2026-10-08 (986 rows, 6 min) |
| `src/models/qwen_forecaster.py` | P2 | ✅ serves the Colab CSV; prompt fingerprints + scale check (26 tests) |
| `src/models/dummy.py` | P2 | ✅ (26 tests) |
| `src/attacks/common.py` | P3 | ✅ windows (val + test), timelines, save/load `texts/<condition>.csv` |
| `src/attacks/empty.py`, `shuffled.py`, `time_shifted.py`, `cross_domain.py` | P3 | ✅ simple conditions |
| `src/attacks/rules.py` | P3 | ✅ rule engine: headline reader, flip / exaggerate, coherence checks |
| `src/attacks/flipped.py`, `exaggerated.py` | P3 | ✅ 384 / 381 attacked texts |
| `src/attacks/generate.py`, `validate_attacks.py` | P3 | ✅ all 7 conditions → `texts/`; validation report (26 tests in `test_attacks.py`) |
| `src/metrics.py` | P4 | ✅ pure functions; worked example reproduced (27 tests) |
| `src/run_experiments.py` | P5a | ✅ resumable inference runner (unit = model × domain × condition); integrity checks: text-blind M1/M2, P2 agreement, embedding scalers; `qwen_rows` for Model 5 (6 tests) |
| `src/models/embed_texts.py --conditions` | P5a | ✅ altered-condition texts embedded (732 new, 0 truncated) |
| `src/models/export_qwen_inputs.py --p5` | P5a | ✅ 2465 condition rows (2215 to generate, 250 reused) + 40 probe rows |
| `notebooks/p5_qwen_conditions.ipynb` | P5a | ✅ built (`build_p5_qwen_notebook.py`); reproducibility probe, revision pinned — **awaiting Colab run** |
| `qwen_forecaster.load_all_predictions` | P5a | ✅ P2.4 + P5 conditions, reused rows filled and re-fingerprinted (4 new tests) |
| `src/analyze_p5.py`, `src/figures_p5.py` | P5b | ✅ screening / attack / fusion / damage / ablations → `results/summary_p5_*.csv`, `paper/figures/p5_*`; Holm family asserted = 76 (3 tests); `metrics.mcnemar_flip` (1 test) |
| `src/textguard/trend_detector.py` | P6a | ✅ change evidence (OLS slope, fitted movement / σ) + level evidence (y_last vs stated baseline) |
| `src/textguard/judge_prompt.py` | P6a | ✅ judge prompt, single source of truth (pasted into the notebook) |
| `src/textguard/text_direction.py` | P6a | ✅ `DirectionJudge` protocol, `QwenJudge` (served from the Colab CSV, fingerprints verified), `stated_baseline` |
| `src/textguard/gate.py` | P6a | ✅ per-claim gate, min over claims |
| `src/textguard/baselines.py` | P6a | ✅ always / never / clip / TextGuard soft + block; NaN rules (vectorised, for P6b) |
| `src/textguard/textguard.py` | P6a | ✅ wraps any `predict(numbers, text)`; itself a Forecaster |
| `src/textguard/export_judge_inputs.py` | P6a | ✅ 2490 questions / 1831 texts + 40 probes |
| `notebooks/p6_direction_judge.ipynb` | P6a | ✅ run on Colab T4 2026-10-09 (2490 answers, 8 min, probe 40/40, P5's exact setup) |
| `src/auditor.py` | P7 | ⚪ |
| `app/` (Streamlit) | P7 | ⚪ |

---

## 6. Data findings (fill in during P1)

**Selected domains:** Economy, SocialGood, Traffic, Health_US — all `source="report"` (decided 2026-09-28)

**Per-domain stats** (`python -m src.data.inspect_domains`, anchor end, lag 1, after cleaning; window = first → last period with text):

| Domain | Freq | Text window | Window periods | Usable (with text) | Coverage in window | Median text chars |
|---|---|---|---|---|---|---|
| Health_US | weekly | 2007-02 → 2020-01 | 672 | 482 | 71.7% | 229 |
| Economy | monthly | 1995-02 → 2024-03 | 350 | 344 | 98.3% | 398 |
| SocialGood | monthly | 1994-02 → 2023-01 | 348 | 213 | 61.2% | 437 |
| Traffic | monthly | 2002-02 → 2024-03 | 266 | 266 | 100.0% | 308 |

SocialGood's coverage was 99.4% before its off-topic influenza texts were filtered — that figure was inflated by junk.

Coverage is measured **inside the text window**, not over the whole series: the numbers
start years before the reports, so whole-series coverage (Traffic 41%, SocialGood 37%)
understates what is usable. The window ends at the last text, not the series end, because
some reports stop early (Health_US: 2020, while its numbers run to 2024).

**Why the other six were dropped:**

| Domain | Window coverage | Usable | Reason dropped |
|---|---|---|---|
| Energy | 53.2% | 354 | Weekly reports cover only half the weeks — half the windows would be text-free |
| Climate | 23.1% | 291 | Monthly reports on a weekly series: coverage can never exceed ~25% |
| Health_AFR | 44.7% | 156 | Low coverage and few usable periods; 49% of report `fact` fields are null |
| Environment | 9.5% | 80 | Daily series with reports only 2021–2023; almost no aligned text |
| Agriculture | 81.5% | 53 | Only 65 periods of text (from 2018); median 2614 chars — above BERT's 512-token limit |
| Security | 0% | 0 | `report` text is 100% null |

**Health_US / SocialGood caveat:** 28% (Health_US) and 39% (SocialGood) of periods in the window have no text. Those periods are real
missing reports, not attacks — P3/P5 must keep `has_text=False` samples out of attack metrics
(or report them separately), otherwise `empty` and "no report existed" get conflated.

**⚠️ Risk — small test sets.** These domains have 266–483 usable periods. After a chronological
70/10/20 split and lookback windows, each domain's test set is small (~30–60 sequences).
Per-domain DSS will be noisy. **Plan:**
- **Headline:** DSS pooled across all four domains
- **Breakdown:** per-domain DSS, reported with its CI, not as a headline
- **P4:** bootstrap confidence intervals on every reported metric (already in P4 scope)

**Actual splits** (70/10/20 over the text window, `python -m src.data.splits`):

| Domain | Train | Val | Test | Test span |
|---|---|---|---|---|
| Economy | 244 | 36 | 70 | 2018-06 → 2024-03 |
| SocialGood | 243 (136 with text) | 36 (31) | 69 (46) | 2017-05 → 2023-01 |
| Traffic | 186 | 27 | 53 | 2019-11 → 2024-03 |
| Health_US | 470 (286 with text) | 68 (66) | 134 (130) | 2017-06 → 2020-01 |

- **Validation sets are also small (27–68 periods).** Tuning thresholds on them risks
  overfitting to validation — prefer few tunable knobs and coarse grids in P6.3
- **Test sets for Economy, SocialGood and Traffic include COVID-19 (2020),** a large regime shift.
  Expect high MSE. See the COVID reframe below — this is an opportunity, not only a nuisance

**⚠️ Health_US confound (flag when interpreting P5):** 183 of 470 Health_US *training* periods
lack text, while its *test* set is 97% covered. A text-consuming model trained there saw text
only ~61% of the time, so a low Text Usage Gap on Health_US may reflect **training sparsity**,
not the architecture ignoring text. Don't attribute a low Health_US gap to fusion type without
comparing against Economy and Traffic (≥98% train coverage).
**After the influenza filter, SocialGood has the same confound:** only 136/243 train periods (56%)
have text, and 46/69 test periods (67%). Treat Health_US and SocialGood as the "sparse-text"
domains and Economy and Traffic as the "dense-text" domains when interpreting P5.

**COVID reframe → dedicated P6 analysis:** 2020 sits in three test sets (Economy, SocialGood,
Traffic). It is a natural stress test for TextGuard's **False Block Rate**: genuine reports
("unemployment surged", "traffic collapsed") that contradict the pre-COVID numerical trend.
That is exactly the case where TextGuard's gate wrongly distrusts *true* text. Plan a dedicated
**"TextGuard under genuine regime change"** analysis in P6 — gate values, False Block Rate and
forecast error on 2020 periods specifically — not just an ablation that excludes 2020.

**Directional-claim check** (`python -m src.data.check_directional`, 50 reports, stratified
13/13/12/12, seed 0, drawn only from before each test split; labels in
`results/summary_directional_sample.csv`):

| Domain | Clear up/down claim (manual) | Typical claim |
|---|---|---|
| Economy | 12/13 = 92% | "trade deficit increased by $4.6B" (month-over-month) |
| Traffic | 12/12 = 100% | "travel changed by −1.6% vs same month last year" (year-over-year) |
| SocialGood | 3/13 = 23% | mostly "rate unchanged at 8.2%"; 5/13 were off-topic influenza text |
| Health_US | 2/12 = 17% | mostly levels: "ILI was 2.1%, below the national baseline" |
| **Overall** | **29/50 = 58%** | |

→ **Verdict: viable overall (58% ≥ 30%), but only because of Economy and Traffic.**
SocialGood and Health_US are each below the 30% bar on their own. *(This sample predates the
influenza filter — see the re-check below.)*

**Re-check after filtering, and Energy as a swap candidate** (25 reports each, seed 0, before each
test split; `results/summary_directional_socialgood_energy.csv`):

| Domain | Clear up/down claim (manual) | Typical claim |
|---|---|---|
| SocialGood (filtered) | 14/25 = **56%** | "unemployment rate rose to 5.7%"; still ~44% "unchanged at X%" |
| Energy | 25/25 = **100%** | "price decreased to $2.993, $0.063 less than last week" (week-over-week) |

SocialGood's earlier 23% was dragged down by the junk (5/13 of that sample) and small-sample noise.

**⚠️ Energy has a severe timing problem if swapped in:** `check_timing` flags **351/354** Energy
reports. Each weekly report quotes "the price on Monday X", and Monday X is the first day of the
week being forecast — the text contains the forecast target itself. Energy would need `lag=2`
(losing a week of text freshness) before it could be used. **Swap decision pending (issue #14).** Keyword heuristic agrees with
manual labels on claim/no-claim only 72% of the time — it misses signed changes ("changed by −0.9%",
"$5.4B less") and fires on sub-clauses — so P3's `validate_attacks.py` needs a better direction check.

**📄 Paper-worthy dataset finding — SocialGood off-topic text:** 144/347 = **41% of Time-MMD's
SocialGood reports are headed "Objective facts about the influenza situation"** — Time-MMD's LLM
summariser was run with the influenza (Health) prompt on unemployment reports. 131 are empty
("NA", "There is no mention of the influenza situation in this report"); 13 contain real
unemployment facts under the wrong header (8 of them from 2020–22, the COVID period).
**Handling (`loader._clean_reports`):** drop influenza-headed texts unless they state an
unemployment figure; strip the wrong header from the 13 kept. A blanket filter would have deleted
genuine COVID-era reports. Worth reporting in the paper: anyone using Time-MMD SocialGood text
as-is trains on 41% off-topic input. Result: 347 → 216 SocialGood reports.

**Timing check** (`python -m src.data.check_timing`; first run, before cleaning, 1633 texts):

| Alignment | Metadata violations (end_date ≥ forecast start) | Content violations (dates in text ≥ forecast start) |
|---|---|---|
| `anchor="start"` (old default) | **37** (SocialGood 9, Traffic 25, Health_US 3) | 24 |
| `anchor="end"` | **0** | 12 |

→ **Verdict: the 1-period lag alone was NOT sufficient.** Anchoring by `start_date` lets annual
reports (SocialGood: "unemployment rate in 2015 was 5.3%", dated Jan 1 → Dec 31) feed the
*February* forecast of the same year. Anchoring by the later of start/end date fixes all metadata
violations. Of the 12 content flags left under `anchor="end"`:
- 7 are **year typos** in Time-MMD text, not leakage (text dated July 2010 says "July 2020", but the
  $42.8B figure is July 2010's) — Economy 6, SocialGood 1
- 2 Traffic: week-53 report with end_date stored a year early (2020-01-03 for 2021-01-03); 3 days of overlap
- 3 Health_US: season summaries stored as ending Jan 1 but describing Feb–Oct peaks — **real leakage**

**Resolution (decided 2026-09-28):** `anchor="end"` is now the default; the 5 real leakers are
dropped via `loader.KNOWN_LEAKERS` (listed individually, with reasons; the loader raises if the
list stops matching the data). **Final check: 1497 texts, 0 metadata violations, 0 content
violations → PASS.** Now a regression test (`tests/test_check_timing.py`).

**Known Time-MMD quirk — year typos (kept, not dropped):** 7 reports state the wrong year in their
text — Economy 2001-03, 2001-04, 2001-11, 2010-07, 2010-09, 2011-11 and SocialGood 2010-08 (e.g.
a July 2010 report says "$42.8 billion in July **2020**"; $42.8B is July 2010's figure, July 2020
was ~$63B). The content is correct for the period it is filed under; only the written year is
wrong, so dropping them would lose valid data to fix a metadata error. They are listed in
`check_timing.KNOWN_YEAR_TYPOS` and reported separately, so any *new* content flag still fails.
**Implication for P3:** the flip attack must leave dates untouched, and must not "correct" a
typo'd year — otherwise the flipped text differs from the real one in two ways, not one.

### Attack scope (decided 2026-09-28)

All four domains stay. The directional check showed two text styles, so the attack has
**two variants**:

| Variant | Domains | Real text | Attacked text |
|---|---|---|---|
| **change-flip** | Economy, Traffic | "deficit increased by $4.6B" / "travel changed by −1.6%" | "deficit decreased by $4.6B" / "travel changed by +1.6%" |
| **level-flip** | Health_US, SocialGood | "ILI was 2.1%, below the national baseline" | "ILI was 2.1%, above the national baseline" |

**Consequence for P6 — TextGuard's gate must generalise** from "text *direction* vs numerical
*trend*" to **"text *claim* vs numerical *evidence*"**, with two branches:
- **change branch** — as in CLAUDE.md §8: the text's claimed direction vs the slope of recent values
- **level branch** — the text's claimed level (above/below a reference such as a baseline or a
  prior value) vs where the recent numbers actually sit relative to that reference
The gate formula `g = 1 − confidence × evidence_strength` stays; what changes is how the text
claim and the numerical evidence are extracted. CLAUDE.md §8 still describes only the change branch.

**P3 requirement — the direction classifier must be Qwen-based.** The keyword heuristic agreed
with human labels only **72%** of the time: it missed signed changes ("changed by −0.9%",
"$5.4B less") and fired on sub-clauses about other quantities. `validate_attacks.py` (P3) and
`text_direction.py` (P6) must use Qwen classification, validated against the 100 hand labels in
`results/summary_directional_*.csv`, not rules.

**MM-TSFlib fusion mechanism (from reading the code):** late additive fusion; at h=1 its text term is identically 0 — see §7a.

### Time-MMD layout & text decisions (P1.1–1.2, 2026-09-28)

**Layout (verified by inspection, not from the README):**
- `numerical/<D>/<D>.csv` — target column `OT`; date columns vary by domain
  (Health_AFR has no `start_date`, only `date`/`date.1`/`end_date`)
- `textual/<D>/<D>_{report,search}.csv` — columns `start_date, end_date, fact, preds`
  (README says `pred`; the real column is `preds`)
- Row order is not consistent: Climate is newest-first, Economy is unsorted → the loader sorts
- Security `report` is 100% null — unusable with `source='report'`
- Health_US has 5 duplicate dates (ISO week 53 and next year's week 1 share dates) → keep the later row

**Why `fact` only, not `fact + preds`:** `preds` is not source text — it is forecasts
written by the Time-MMD authors' LLM ("In the long term (next 4-18 months), I predict…").
Our threat model is an adversary manipulating *real-world reports* feeding a forecaster.
Attacking or defending machine-written forecasts is a different threat model, and would
also let an upstream LLM's guesses leak into the "text" signal we measure.

**Why lag 1 period:** text dates mark the start of the period a text covers, not when it was
published. Example: an Energy report dated `2016-12-12 → 12-16` quotes the price "on December 19".
Giving row t only texts from period t−1 was meant to guarantee the text existed before period t.
*P1.5 showed this alone is not enough — see the timing check above; `anchor="end"` is now the default.*

**⚠️ Confound for P2 baseline reproduction:** MM-TSFlib does **not** read Time-MMD's raw text.
It ships its own preprocessed CSVs whose `Final_Search_k` column is **`search`** snippets from
the k weeks before each row. We default to **`report`**. **If our P2 numbers don't match
published MM-TSFlib numbers, check this first** — rerun with `source='search'` before debugging
anything else. Also: MM-TSFlib's Economy CSV is unsorted (starts 2024, ends 1989) and it splits
by row position, so its Economy train/test split is **not chronological**.

### 6a. Forecaster interface (P2.1, decided 2026-10-02)

`src/models/base.py`. TextGuard and the Auditor only see the `Forecaster` Protocol: any object
with `horizon` and `predict(numbers, text)`, no inheritance required. Our wrappers subclass
`BaseForecaster`, which validates inputs/outputs and calls `_predict`.

- **`numbers`:** 1-D univariate lookback window, raw data units, finite. 2-D input is rejected,
  not flattened. Univariate is deliberate: Sridhar et al. (2026) found "text gains" can come from
  a co-shipped numeric column, not the text. Raw units keep DSS (÷σ) and `A + g(B−A)` meaningful.
  **Wrappers fit any scaler on the train split only.**
- **`text`:** `str | None`; blank → `None`. So `empty`, `has_text=False` and TextGuard's `A` are
  one and the same prediction.
- **Output:** always shape `(horizon,)`. NaN = failure (e.g. Qwen parse failure); inf raises (a bug).
- **Horizon:** the shape stays general, but **main experiments use horizon = 1, and DSS and Flip
  Rate are scored at horizon 1.**

**NaN pairing rule — binding on P4 and P6:**
- **P4 metrics:** in any paired comparison (ŷ_fake vs ŷ_real for DSS / Flip Rate; MSE(empty) vs
  MSE(real) for the Usage Gap), if *either* side is NaN, drop that sample from *both* sides.
  **Report the parse-failure rate per model × condition** alongside every metric. Otherwise a
  model that fails more often on fake text looks artificially robust: its worst cases vanish.
- **P6 TextGuard:** if `B` is NaN, `final = A` (fall back to numbers-only). If `A` is NaN there is
  nothing to fall back to: the sample stays NaN and counts as a failure.

**Dummy model (`src/models/dummy.py`)** — the hand-checkable fixture for P4/P6:
- A = linear extrapolation; B = A + `sensitivity × direction × step × intensity`
- Defaults reproduce the worked example: A = 116, B(fake "fall sharply") = 106, B(real
  "demand rising") = 120 → DSS = +1.4
- `sensitivity=0` → deaf (gap 0); `sensitivity=1` → fooled by flipped text
- Change cues (rise/fall…) and **level cues ("above/below [the] [word] baseline" = ±1)**, so
  P6's level branch has a fixture too. Flat `[2,2,2,2]`, step 0.5 → above 2.5 / below 1.5
- Keyword lexicon is fine for a fixture; it ignores negation and is **not** a text classifier

---

## 7. Baselines (P2.2)

`python -m src.models.train_baselines` → `results/summary_p2_baselines.csv`,
`results/summary_p2_lr_tuning.csv`; raw predictions in `results/p2_baselines_predictions.csv`.
Normalised MSE (÷ train variance), mean ± std over seeds 2021–2023. Lookback 8 (monthly) / 24 (weekly).

### Main config — horizon 1, val-tuned protocol (what P5 uses)

| Domain | Split | Persistence | DLinear | PatchTST |
|---|---|---|---|---|
| Economy | val | 0.150 | 0.101 ± 0.000 | **0.080 ± 0.011** |
| Economy | test | 0.280 | 0.198 ± 0.002 | **0.183 ± 0.011** |
| SocialGood | val | 0.033 | 0.019 ± 0.000 | **0.009 ± 0.001** |
| SocialGood | test | **0.573** | 0.740 ± 0.007 | 0.883 ± 0.092 |
| Traffic | val | 1.331 | 1.021 ± 0.008 | **0.041 ± 0.008** |
| Traffic | test | 1.582 | 1.043 ± 0.005 | **0.885 ± 0.032** |
| Health_US | val | 0.057 | **0.042 ± 0.001** | 0.062 ± 0.015 |
| Health_US | test | 0.150 | **0.109 ± 0.004** | 0.159 ± 0.032 |

Chosen lr (val loss, seed 2021): Economy 1e-3 / 1e-3 · SocialGood 1e-2 / 1e-3 ·
Traffic 1e-2 / 1e-4 · Health_US 1e-2 / 1e-3 (DLinear / PatchTST). Every chosen run stopped
early (best epoch 15–60), well inside the 200-epoch cap.

Reading the table:
- **Val numbers are optimistic** — lr was selected on val. Test is the honest column.
- **SocialGood test: both models lose to persistence.** Its test span (2017-05 → 2023-01)
  contains the 2020 unemployment spike; no 8-month window predicts it. Expected, and the
  same regime shift as the P6 COVID analysis (issue #10).
- **Traffic: DLinear val 1.02 vs PatchTST 0.04 is real, not a bug.** Traffic has a strong
  12-month cycle and the lookback is 8, so lag 12 is never visible. PatchTST infers the
  seasonal phase from the window's shape (e.g. predicts the March rebound after the February
  trough); a linear map cannot (issue #22). Traffic reports state year-over-year changes
  ("−1.6% vs January 2009") — exactly what the 8-month window lacks — so Traffic is the
  likeliest domain for genuine text use and the strongest P5 attack target. Lookback stays 8.
- **Health_US: PatchTST does not beat persistence** (test 0.159 vs 0.150, high seed variance).
  DLinear does. Keep in mind when Model 4's gap is computed on Health_US.
- Training is deterministic: an independent re-run on 2026-10-07 reproduced the 2026-10-02
  run to ≤1.3e-7 (CSV round-off).

### ⚠️ Why the tuned protocol exists — DLinear under MM-TSFlib's recipe ≈ window mean

Under MM-TSFlib's recipe (10 epochs, patience 5, lr 1e-4 ≈ 70 Adam steps) DLinear barely
leaves its initialisation (weights 1/L = a window average). At h=1, 3 seeds:

| Domain | Split | DLinear (MM-TSFlib recipe) | Window-mean predictor | corr(DLinear pred, window mean) |
|---|---|---|---|---|
| Economy | test | 0.348 ± 0.032 | 0.309 | 0.9999 |
| SocialGood | test | 1.341 ± 0.023 | 1.322 | 0.9999 |
| Traffic | test | 2.547 ± 0.031 | 2.561 | 0.9996 |
| Health_US | test | 2.245 ± 0.037 | 2.823 | 0.9758 |

On the monthly domains the "trained" model *is* the window mean (corr ≥ 0.9996; on val
≥ 0.9984). An untrained backbone would make any text gain in Model 3 meaningless — text would
be compensating for under-training. Hence the tuned protocol, and hence:

> **Models 3 and 4 (P2.3) must reuse the tuned protocol exactly** — `LR_GRID`, `TUNED`
> (200 epochs, patience 20), `TUNE_SEED`, the same 3 seeds, and `patching="window"` for
> PatchTST — tuned per domain on val. Otherwise a Text Usage Gap mixes the effect of text
> with the effect of extra trainable parameters / different training (issue #19).

### ⚠️ PatchTST patch_len clamp (MM-TSFlib bug at h=1)

MM-TSFlib's `models/PatchTST.py` has `if patch_len > pred_len: patch_len = pred_len` (not in
the original PatchTST). At `pred_len=1` this gives patch_len 1, stride 8: on a lookback of 8
the encoder sees **2 of 8 inputs** — `x[0]` and the padded copy of `x[7]`. Six inputs are never
read. Our main config uses `patching="window"` (patch_len L/2, stride L/4 → 4 overlapping
patches covering every input); the as-shipped network is used only in the repro config at
h=6/12, where the clamp does not fire. Tested in `tests/test_baseline_models.py`.
**Model 4 (PatchTST + BERT) inherits the same constructor → must use `patching="window"`** (issue #20).

### Reproduction config — MM-TSFlib recipe, published horizon (monthly 6, weekly 12)

| Domain | Model | Our MSE (test) | Published Uni MSE | Close? |
|---|---|---|---|---|
| Economy | DLinear | 0.493 ± 0.044 | 0.08 | ❌ |
| Economy | PatchTST | 0.285 ± 0.003 | 0.02 | ❌ |
| SocialGood | DLinear | 2.101 ± 0.035 | 1.01 | ❌ |
| SocialGood | PatchTST | 2.299 ± 0.172 | 0.88 | ❌ |
| Traffic | DLinear | 2.869 ± 0.034 | 0.35 | ❌ |
| Traffic | PatchTST | 2.248 ± 0.015 | 0.21 | ❌ |
| Health_US | DLinear | 3.107 ± 0.008 | 2.37 | ❌ (closest) |
| Health_US | PatchTST | 1.607 ± 0.042 | 1.23 | ❌ (closest) |

Same networks, same recipe, numbers-only — so text source (issue #6) cannot explain the gap.
**Cause:** different test periods and splits. Ours is chronological over the text window,
and the monthly test sets contain 2020; MM-TSFlib splits its own CSVs by row position over the
whole series, and its Economy CSV is unsorted (issue #7), so its test set is not the future.
Health_US — the one domain whose test span (2017–2020) ends before COVID — is the closest.
**Closed (2026-10-07): not pursued further** — listed as a limitation in the paper. The repro
column only shows the networks and recipe are MM-TSFlib's; it is not a claim of matching their numbers (issue #21).

---

## 7a. Late-fusion text models (P2.3)

### How MM-TSFlib fuses text (issue #3 — read 2026-10-07)

`exp/exp_long_term_forecasting.py`, with the shipped `scripts/week_health.sh` settings:

```
prompt = "<|start_prompt|Make predictions about the future based on the following information: {text}<|<end_prompt>|>"
E      = BERT.get_input_embeddings()(tokens)       # use_fullmodel=0: word-embedding table only, no transformer
t      = avg_pool_tokens( MLP(E) )                 # MLP 768→96→pred_len per token; padding included in the average
ŷ      = (1 − w)·TS(x) + w·( norm(t) + prior_y )   # w = prompt_weight: 0.01 default, 0.1 in the script
norm(t)= (t − mean_h t) / std_h t                  # z-score ACROSS THE FORECAST HORIZON, per sample
```

→ **Late, additive fusion** (our assumption holds). But, as shipped:

1. **At h=1 the text term is identically 0.** `norm()` subtracts the mean over the horizon;
   with one step that is the value itself. Verified numerically and locked in by
   `tests/test_late_fusion.py::test_mmtsflib_norm_is_identically_zero_at_horizon_1`. At h>1
   the text can only reshape the forecast (zero-mean across steps), never shift its level —
   a "rose → fell" attack cannot move it by construction.
2. **`prior_y` is a numeric seasonal prior in the "text" branch.** It is the
   `prior_history_avg` column at the target date: same month last year (Economy, 97% of rows)
   or the mean of the same month over the last 2 years (Traffic, SocialGood, 95%). That is
   lag-12 information the 8-month window lacks — a co-shipped numeric column, the confound
   Sridhar et al. (2026) warn about.
3. Smaller: padding tokens are averaged into the text vector (batch-dependent); early
   stopping restores only the backbone, not the text MLP; missing text becomes the string
   `"nan"`; `mlp_proj`, `weight1`, `weight2` are built and never used.

📄 **Paper finding A — MM-TSFlib's text term is exactly zero at horizon 1.** It is
normalised across the forecast horizon (`norm()`), and a single value minus its own mean is
0. At h=1, MM-TSFlib's text models are effectively numbers-only (plus `prior_y`, below).
Evidence: `tests/test_late_fusion.py::test_mmtsflib_norm_is_identically_zero_at_horizon_1`.

📄 **Paper finding B — MM-TSFlib's text branch carries a co-shipped numeric value.**
`prior_y` = `prior_history_avg` = the same month 1 year (Economy, 97% of rows) or the mean of
the same month 1–2 years earlier (Traffic, SocialGood, 95%). This **independently confirms
the mechanism behind Sridhar et al. (2026)'s finding** that "text gains" in multimodal TSF
can come from a numeric column rather than the text.

### What we implemented (decided 2026-10-07)

`ŷ = (1 − w)·TS(x) + w·MLP(e_text)`, `src/models/late_fusion.py`:

| Part | MM-TSFlib | Ours | Why |
|---|---|---|---|
| Fusion | late, additive, w = 0.1 | same (`TS + w·MLP` ≡ `(1−w)·TS + w·MLP` up to a backbone constant) | faithful |
| Text MLP | 768→96→h, ReLU, dropout 0.3 | same | faithful |
| Backbone | DLinear / PatchTST | same networks; PatchTST `patching="window"` | §7 clamp bug |
| `norm()` | z-score across horizon | **dropped** | ≡ 0 at h=1 |
| `prior_y` | numeric seasonal prior | **dropped** | not text; Models 1/2 lack it |
| BERT | word-embedding table only | **all 12 layers, frozen, mean-pooled over real tokens** | must read context to test meaning-inverting attacks |
| Embedding scaling | none | **per-dim z-score, fit on train texts only** | see below |
| Training | joint (TS + MLP), 10 epochs, TS lr 1e-4, MLP lr 1e-2 | **two-stage: TS = trained Model 1/2 (same seed), frozen + eval mode; text head trained under the P2.2 protocol** (head lr ∈ {1e-4, 1e-3, 1e-2} on val, 200 epochs, patience 20, 3 seeds) | critical check — below |
| Missing text | string `"nan"` | text branch absent | well-defined `text=None` |

**Embedding scaling — found while building.** Mean-pooled BERT vectors are nearly identical
across texts (a large shared direction; per-dimension spread across texts only 0.03–0.25).
Fed raw into the MLP at MM-TSFlib's lr 1e-2, every ReLU unit died within 1–3 epochs on
Economy (PatchTST+BERT: 0/96 alive) and the model gave every text the same output — text-deaf
through optimisation, not architecture. Standardising each dimension with train-split
statistics keeps all 96 units alive. BERT itself is untouched.

**Truncation:** 0 of 1305 period texts exceed 512 tokens (max 488, Traffic; medians 84–128).
`python -m src.models.embed_texts`. Re-check for P3's attacked texts (exaggerated may be longer).

### What `text=None` means for Models 3/4 — and why the backbone is frozen

`predict(x, None)` (also blank text, and periods with no report) **drops the text branch**:
`ŷ = TS(x)`, which is Model 1/2's forecast exactly (same seed, frozen weights). Then

- Text Usage Gap = MSE(Model 1/2) − MSE(Model 3/4, real text): the effect of text and nothing else
- TextGuard's A = Model 1/2, B − A = w·MLP(z(e_text)); `A + g·(B − A)` interpolates between
  the numbers-only and the text-informed forecast

**How we got here (val only, seed 2021, P2.2 lrs).** First implementation: MM-TSFlib's joint
training, `text=None` = a "null prompt" (empty template), kept in-distribution by text
dropout. The no-text path stayed worse than the numbers-only model:

| val MSE | numbers-only | joint, None, p=0.2 | joint, None, p=0.5 |
|---|---|---|---|
| Economy DLinear / PatchTST | 0.101 / 0.080 | 0.123 / 0.089 | 0.111 / 0.082 |
| SocialGood DLinear / PatchTST | 0.019 / 0.009 | 0.018 / 0.012 | 0.020 / 0.013 |
| Traffic DLinear / PatchTST | 1.021 / 0.041 | 1.199 / **0.121** | 1.165 / 0.047 |
| Health_US DLinear / PatchTST | 0.043 / 0.062 | 0.049 / 0.064 | 0.048 / 0.055 |

The full joint run (p=0.5) on Economy DLinear+BERT gave `text=None` val 0.111 / 0.109 / 0.128
over 3 seeds vs Model 1's 0.101, with real text 0.093–0.095: most of the "text gain" was the
backbone co-adapting with text and becoming a weaker numbers-only model — an inflated Usage
Gap. Stopped; the frozen-backbone design replaces it. Val comparison, real text:

| val MSE, seed 2021 | `text=None` (= M1/M2) | frozen + real text | joint p=0.5 + real text |
|---|---|---|---|
| Economy DLinear / PatchTST | 0.101 / 0.076 | **0.071 / 0.063** | 0.095 / 0.069 |
| SocialGood DLinear / PatchTST | 0.019 / 0.009 | **0.017 / 0.008** | 0.019 / 0.014 |
| Traffic DLinear / PatchTST | 1.012 / 0.034 | **0.467 / 0.029** | 0.641 / 0.035 |
| Health_US DLinear / PatchTST | 0.043 / 0.050 | **0.043 / 0.048** | 0.049 / 0.055 |

Frozen wins all 8 cells with real text and passes the critical check by construction.
(Val numbers include lr selection on val, for both variants alike.)

📄 **Method note (accepted 2026-10-07) — Models 3 and 4 are "frozen backbone + additive
text head".** This differs from MM-TSFlib, which trains backbone and text MLP jointly. With
`text=None` they reproduce Models 1/2 **exactly** (all 1479 main-config predictions per model:
DLinear+BERT bit-identical to DLinear; PatchTST+BERT identical to float32 precision, ≤2.4e-7
relative), so **B − A = w·MLP(z(e_text)) isolates the text effect** — nothing else changes
between A and B. Rationale for the paper: jointly trained late-fusion models can show a
positive Text Usage Gap that is really a weakened no-text path (Economy above).

📄 **Method note (accepted 2026-10-07) — BERT embeddings standardised per dimension with
train-only statistics**, because raw mean-pooled BERT vectors share a dominant direction and
killed the text MLP's ReLUs (above).

### Results — test MSE, real text vs `text=None` (h=1, normalised, mean ± std, 3 seeds)

| Domain | Persistence | M1 DLinear | M2 PatchTST | M3 real | M3 None | M4 real | M4 None |
|---|---|---|---|---|---|---|---|
| Economy | 0.280 | 0.198 ± 0.002 | 0.183 ± 0.011 | **0.169 ± 0.012** | 0.198 ± 0.002 | **0.177 ± 0.018** | 0.183 ± 0.011 |
| SocialGood | **0.573** | 0.740 ± 0.007 | 0.883 ± 0.092 | 0.783 ± 0.012 | 0.740 ± 0.007 | 0.886 ± 0.103 | 0.883 ± 0.092 |
| Traffic | 1.582 | 1.043 ± 0.005 | 0.885 ± 0.032 | 1.599 ± 0.146 | 1.043 ± 0.005 | 0.911 ± 0.066 | 0.885 ± 0.032 |
| Health_US | 0.150 | 0.109 ± 0.004 | 0.159 ± 0.032 | 0.112 ± 0.007 | 0.109 ± 0.004 | **0.145 ± 0.023** | 0.159 ± 0.032 |

Val (selection split), real vs None: Economy 0.076 vs 0.101 (M3), 0.070 vs 0.080 (M4) ·
SocialGood 0.018 vs 0.019, 0.010 vs 0.009 · **Traffic 0.585 vs 1.021**, 0.038 vs 0.041 ·
Health_US 0.042 vs 0.042, 0.063 vs 0.062. Chosen head lr (M3 / M4): Economy 1e-2 / 1e-4 ·
SocialGood 1e-2 / 1e-3 · Traffic 1e-3 / 1e-2 · Health_US 1e-4 / 1e-3.

Reading it (preliminary — P5 does the proper screening with shuffled/cross-domain controls):
- **Economy:** text helps both models on val and test — the cleanest "text is used" case.
- **Traffic:** text halves Model 3's val error (year-over-year info, issue #22) but **hurts on
  test** (1.04 → 1.60; M4 0.885 → 0.911). The test span contains 2020; genuine extreme
  COVID-era reports are far outside the text head's training range → issue #23.
  *(P5b: this explanation is **not supported** — real text still hurts M3 on Traffic with all
  2020 windows removed, usage gap −0.59 [−0.88, −0.31]; §9 P5.6a.)*
- **SocialGood:** text slightly hurts on test; val gain ≈ 0. Sparse-text domain (issue #9).
- **Health_US:** ≈ no text effect on val; M4 test improves (0.159 → 0.145) within noise.
- Several heads early-stop at epoch 1–7 (SocialGood M4 best epoch 1, Health_US M4 2):
  little learnable text signal there on val. Expected for level-language domains.

## 7b. Model 5 — Qwen2.5-1.5B-Instruct, early fusion (P2.4)

**Workflow** (Colab web UI, issue #5):
`python -m src.models.export_qwen_inputs` → `results/qwen/qwen_inputs.csv` (493 val + test
windows, 4 domains; lookback in raw units, aligned report text, y_true, train scaler; 329 KB,
gitignored) → upload with `notebooks/p2_qwen_forecaster.ipynb` to colab.research.google.com
(T4) → `p2_qwen_predictions.csv` back into `results/qwen/` → `QwenForecaster(domain)` serves
`predict(numbers, text)` from it, so P4–P6 treat Model 5 like Models 1–4.

| Decision | Value | Why |
|---|---|---|
| Fusion | **early**: numbers + report in one chat prompt | the model decides how to combine them |
| Precision / hardware | fp16 on a Colab T4 | free tier; 1.5B fits easily |
| Decoding | **greedy, batch size 1** (no padding) | deterministic: **one run replaces 3 seeds** (recorded as seed 0). Seed variance for Model 5 is not reported because there is none |
| Numeric prompt | values labelled `t-L … t-1`, **no dates** | reduces recall of memorised historical values |
| `text=None` | **identical prompt with the report block removed** | A is a natural, valid prompt; B − A = the effect of adding the report |
| Parsing | first number; handles "282,558", Unicode minus, echoed `t-1` labels, "billion" vs a "million" unit | `nan` + `parse_ok=False` on failure, **never dropped**; raw output kept in the CSV |
| Scale check (decided 2026-10-09) | answer > 10× outside its own window's absolute range → NaN, `scale_ok=False`, raw kept in `y_pred_raw`; applied at load in `qwen_forecaster.py`, the Colab CSV stays raw | a unit slip is a format failure, not a forecast; it must not dominate DSS / Flip Rate. Uses only the lookback, never y_true — but was chosen after seeing the 3 rows below; **state this in the paper** |
| Source of truth | `src/models/qwen_prompt.py`, pasted verbatim into the notebook by `notebooks/build_p2_qwen_notebook.py` | a test fails if the notebook drifts; `QwenForecaster` recomputes every `prompt_sha` at load and refuses a CSV from a different prompt |
| Crash safety | appends to Drive (if mounted) every 10 prompts; re-running the loop resumes | a Colab disconnect loses ≤ 10 rows |

Output CSV: `domain, split, sample_id, condition (real / none), y_true, y_pred, parse_ok`,
plus `raw_output` and `prompt_sha` for auditing.

**⚠️ Limitation — memorisation leakage.** Qwen2.5's pretraining very likely includes these
public statistics (US trade balance, unemployment, VMT, ILI) and even the reports themselves.
Removing dates from the numeric prompt makes recall harder, but **the report text keeps its
dates** (e.g. "…for March 2020 as compared with March 2019") — editing them would change the
`real` condition itself. Model 5's absolute accuracy may therefore be optimistic; DSS (a
*difference* between two prompts that share everything but the report) is less exposed than
MSE. **List as a limitation in the paper.**

**⚠️ Data caveat — report figures are not the series values.** Economy's target is the
**goods** balance (May 2015: −58,365 M$) while its reports quote the goods-**and-services**
deficit ("$41.9 billion"); Traffic's reports quote a different VMT estimate (March 2020:
"221.0 billion" vs series 226,638 M). Qwen must not copy the report's headline number. The
prompt names the series precisely (e.g. "trade balance in goods"). Check Qwen's errors for
copying in P5.

**⚠️ Risk for P3/P6 — same-model circularity.** Qwen will (a) generate the fake text in P3 and
(b) judge text direction inside TextGuard in P6. A model may find its *own* rewrites easy to
classify, so detection could look easier than it is against human-written or other-model
manipulations. **P6 must validate the direction judge against the human labels from
`check_directional`** (`results/summary_directional_*.csv`, 100 hand-labelled reports), and
ideally test on fakes from a second generator (rule-based word flip, issue #16 / CLAUDE.md §15).

### Model 5 run (Colab T4, 2026-10-08)

`results/qwen/p2_qwen_run_meta.json`: Qwen/Qwen2.5-1.5B-Instruct, fp16, greedy, batch 1,
max 24 new tokens, Tesla T4, **torch 2.11.0+cu130, transformers 5.18.0**; 986 rows in 6 min.
`revision` was recorded as **null** — transformers 5 dropped `config._commit_hash`; the
notebook now falls back to the Hub's `model_info(...).sha` (fixed for future runs, this run's
exact weight snapshot is unrecorded).

**Bug fixed during the run:** transformers 5 returns a `BatchEncoding` from
`apply_chat_template(..., return_tensors="pt")`, so `torch.ones_like(ids)` failed. `generate()`
now uses `ids["input_ids"]` when the result is not a tensor (patched on Colab, then in
`build_p2_qwen_notebook.py`; regression-tested).

**Integrity:** all 986 prompt fingerprints match the current `qwen_prompt.build_prompt`; 986 =
2 × 493 windows; `QwenForecaster.predict` round-trips every row.

**Failures** (`results/summary_p2_qwen.csv`):

| | Economy | SocialGood | Traffic | Health_US |
|---|---|---|---|---|
| parse failures (real / none) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| scale failures (real / none) | 0 / 0 | 0 / 0 | **3 / 0** (3.75% of 80) | 0 / 0 |

The 3 scale failures are all Traffic test, real text: Qwen answered "267.0", "269.3", "259.7" —
the report's unit (billion vehicle miles) for a prompt asking for million. They were 90% of
Traffic's real-text squared error (MSE 18.3 → 1.905 without them). **Text-induced unit
confusion: 3/80 with the report vs 0/80 without** — itself a small text-channel failure mode.

### Results — h=1 test MSE, all 5 models (normalised)

| Domain | Persistence | M1 DLinear | M2 PatchTST | M3 real | M3 None | M4 real | M4 None | M5 real | M5 None |
|---|---|---|---|---|---|---|---|---|---|
| Economy | 0.280 | 0.198 ± 0.002 | 0.183 ± 0.011 | 0.169 ± 0.012 | 0.198 ± 0.002 | 0.177 ± 0.018 | 0.183 ± 0.011 | 0.302 | 0.311 |
| SocialGood | 0.573 | 0.740 ± 0.007 | 0.883 ± 0.092 | 0.783 ± 0.012 | 0.740 ± 0.007 | 0.886 ± 0.103 | 0.883 ± 0.092 | 0.633 | 0.648 |
| Traffic | 1.582 | 1.043 ± 0.005 | 0.885 ± 0.032 | 1.599 ± 0.146 | 1.043 ± 0.005 | 0.911 ± 0.066 | 0.885 ± 0.032 | 1.905* | 2.002 |
| Health_US | 0.150 | 0.109 ± 0.004 | 0.159 ± 0.032 | 0.112 ± 0.007 | 0.109 ± 0.004 | 0.145 ± 0.023 | 0.159 ± 0.032 | 1.348 | 1.054 |

Model 5 is one deterministic run (no ±). *Traffic M5 real over 50 of 53 test windows (3 scale
failures NaN); M5 None over all 53 is 2.002, over the same 50 windows 1.880 — P4 applies
the NaN-pairing rule. Model 5 val (real / none): Economy 0.205 / 0.209 · SocialGood 0.033 /
0.038 · Traffic 1.883 / 2.130 · Health_US 0.590 / 0.297.

Reading it:
- **Model 5 is a weak forecaster** — worse than persistence on Economy and Traffic, and far
  worse on Health_US — but it is not what we need it for. It mostly **copies a recent value**:
  its answer equals some lookback value exactly in 35–87% of windows (t-1 itself in 24–51%).
- **It does read the text:** the real and none answers differ in 65–78% of windows that have a
  report. That makes it an attack surface, which is its role in P5.
- **Health_US is heavy-tailed, not broken:** median squared error 0.018 (none) / 0.052 (real);
  the mean comes from a few decimal slips ("7.72384" for t-1 = 0.772384) and copies of the
  wrong lookback value during flu peaks.

**Issue #26 check — does Qwen copy the report's number?** Rarely. Exact match between Qwen's
answer and a number quoted in the report (billion → million converted; years excluded), in
windows that have a report: **real 5.9%, none 4.1%** (chance baseline: the same comparison
when the report was never shown).

| | Economy | SocialGood | Traffic | Health_US |
|---|---|---|---|---|
| real | 0.0% | 18.2% | 1.2% | 6.1% |
| none (chance) | 0.0% | 24.7% | 0.0% | 0.0% |

SocialGood's matches are coincidences on round values (5.0, 6.0) — more frequent *without*
the report. Health_US's 6.1% are genuine copies, and harmless: its reports quote last week's
value of this same series ("ILI was 2.1%"), so those predictions score test MSE 0.029. The
Economy/Traffic risk (copying a goods-and-services or differently-estimated figure) did not
materialise (0% / 1.2%). → issue #26 closed.

## 8. Attack module and fake-text validation (P3)

`python -m src.attacks.generate` → `texts/<condition>.csv` (gitignored; 7 files: real, empty,
shuffled, cross_domain, time_shifted, flipped, exaggerated; 493 val + test windows each).
`python -m src.attacks.validate_attacks` → report + `results/summary_p3_attack_validation.csv`.
Fake rows carry `synthetic=True`; `texts/README.txt` labels them as synthetic research text.

### Design change (decided 2026-10-09) — fakes are written by RULES, not Qwen

Reasons: (1) Model 5 is Qwen — a Qwen-written attack on Qwen biases the result; (2) P6's
direction judge is Qwen — same-model circularity (issue #25); (3) rules make a **minimal,
exact, reproducible** edit — only the direction (flip) or magnitude (exaggerate) changes —
and every edit is auditable. **Paper framing: a conservative, minimal-edit attacker.** An
LLM-written attack is an optional P7 extra. Also recorded in CLAUDE.md §7.

### The conditions

| Condition | Definition | Why this way |
|---|---|---|
| `empty` | no text | identical to `text=None` / TextGuard's A |
| `shuffled` | a real report from an **earlier** date, same domain (seeded per window; never its own text) | earlier-only: a random *future* report could leak information |
| `time_shifted` | same domain, the period **one year earlier** (12 months / 52 weeks) | same season and style, only the timing is wrong; backward only (forward = a leakage probe, not a control) |
| `cross_domain` | the latest report another domain had published by the forecast date (Economy ← SocialGood ← Traffic ← Health_US ← Economy) | only the topic differs, timing held fixed |
| `flipped` | direction reversed, numbers kept coherent | the core attack |
| `exaggerated` | direction kept, stated change ×3, "sharply"/"well" | TextGuard's known weak spot |

**Variants in practice:** change-flip for **Economy, Traffic and SocialGood** (SocialGood's
reports state changes — "declined by 0.2 percentage point to 3.6 percent" — not levels);
level-flip **only for Health_US** ("ILI was 1.8%, below the national baseline of 2.1%").

**Flip rules** (`src/attacks/rules.py`): direction words swap; signed changes flip sign and
unsigned "changed by 1.0%" becomes "-1.0%"; dependent numbers are recomputed keeping the
*current* level fixed ("X, up Y from Z" → "X, down Y from X+Y"; "X, down from Z" → "X, up from
2X−Z"); Health_US levels are mirrored to the **same distance on the other side of the
baseline (2B − V)**, falling back to the ratio mirror B²/V only if 2B − V ≤ 0. Each fake
records `real_direction` and `claimed_direction` (+1/−1 about the forecast target; Economy's
"deficit up" = balance −1). A text that cannot be flipped cleanly is kept unchanged with
`attacked=False` and a reason — **never emitted self-contradictory**.

**Level mirror — decided 2026-10-09:** the first version used the ratio mirror B²/V for all
Health_US levels. It produced implausible values (0.9% → 5.4%, a flu-peak level in late
summer; 26 of 133 above 5%). Implausible fakes are trivially detectable and would make
TextGuard's P6 results look better than they are — **fair evaluation needs plausible
attacks.** Now linear first: **0 of 133 flipped values above 5%** (range 0.5–3.8%, median
2.9%); the ratio fallback fires 16 times, all for flu-peak values 4.4–7.5% → 0.6–1.1%.

### Validation results

**1. Human labels** — the 50 reports hand-labelled in P1 (written before the test period):

| Domain | Texts | Directional (human) | Rules read the same direction | Flipped correctly |
|---|---|---|---|---|
| Economy | 13 | 12 | 13 | **12 / 12** |
| Traffic | 12 | 12 | 12 | **12 / 12** |
| SocialGood | 13 | 3 | 13 | **3 / 3** |
| Health_US | 12 | 2 | 3 | **2 / 2** |
| **All** | 50 | 29 | 41 | **29 / 29 (100%)** |

The 9 "read differently" are all Health_US *level* claims ("ILI was 2.1%, below the national
baseline"): P1 labelled only up/down *change* claims, so they were "none" there; level-flip
attacks them by design.

**2. Coverage** (val + test texts):

| Domain | Texts | Flipped | Not attacked (reason) | Exaggerated |
|---|---|---|---|---|
| Economy | 105 | **105 (100%)** | — | 105 |
| Traffic | 80 | **79 (99%)** | 1 no directional headline | 77 (2: decrease would exceed 100%) |
| Health_US | 196 | **157 (80%)** | 38 no directional headline ("at the baseline"), 1 unhandled word "elevated" | 156 |
| SocialGood | 77 | **43 (56%)** | 33 no directional headline ("unchanged", "little changed"), 1 unhandled word "lost" | 43 |
| **All** | 458 | **384 (84%)** | 74 | 381 |

SocialGood's 56% matches its P1 directional rate (56%).

**3. Automatic checks on every attacked text:** headline reversed 384/384 (flip) / kept
381/381 (exaggerate); **0 new contradictions** (arithmetic of "up Y from Z", side of the
baseline, sign agreement in "changed by S% (T billion)"); length fake/real 0.95–1.02 (flip),
1.00–1.11 (exaggerate).

**4. Review by eye:** 20 before/after pairs (5 per domain, seed 0) reviewed by Aryan
2026-10-09 → the level-mirror change above.

### Edge cases found and fixed during review (each now a test in `tests/test_attacks.py`)

| # | Trap | Example | Fix |
|---|---|---|---|
| 1 | "less than" as a duration, not a direction | "persons unemployed for **less than 5 weeks**" was flipped to "more than" | more/less flip only right after a change magnitude ("$2.9 billion more than") |
| 2 | record phrases | "declined … to 3.6 percent, the **lowest** rate since 1969" left unflipped (contradicts "rose"); "the **highest** rate and largest … increase in the history" | highest↔lowest swap only in "… since / in the history / on record" phrases (80-char window) |
| 3 | group comparisons | "5.8 percent, **higher than** the 4.7-percent rate for non-Hispanics" would become false | higher/lower flip only right after a change magnitude ("1% lower than") |
| 4 | above/below a stated number | "averaged 158,000, **below** the average monthly gain of 223,000" | above/below flip only next to a baseline / threshold |
| 5 | "less than" as an amount after up/down | "**down less than** $0.1 billion from September" — "down" was missed, leaving that clause unflipped | up/down also flips before "less/more than" |
| 6 | regional sentence as headline | national ILI exactly *at* baseline, direction taken from "regions … at or above region-specific baseline" | Health_US headline comes from the national value only; regional sentences excluded → not attacked |
| 7 | rewrite outside the lexicon | exaggerate turned "edged down" into "plunged", which the headline check cannot read (4 SocialGood texts rejected) | "edged up/down" → "rose/fell sharply" (lexicon words) |

### Limitations (state in the paper)

- **Leftover descriptive side sentences are kept unchanged** — e.g. "10 states experienced
  high ILI activity" next to a flipped "ILI was 1.2%, below the national baseline". Flipping
  side sentences without a tracked direction word risks new contradictions; leaving them
  makes the attack slightly **weaker, i.e. conservative**.
- **Coverage is partial where reports make no directional claim** (SocialGood 56%, Health_US
  80%): those windows are not attacked, and DSS / Flip Rate are computed on attacked windows
  only — report the attacked count next to every metric.
- **Rules are tuned to Time-MMD's report styles** (the lexicon was built by scanning every
  val/test report). A new domain needs its headline subject and a lexicon scan.
- **Exaggerated level claims use the ratio scale** (V' = B·(V/B)³); not changed with the flip
  mirror — check plausibility if exaggeration matters in P6.

---

## 9. Results

### Metric definitions (P4, `src/metrics.py`)

Every metric is the **mean of a per-window term** over the *valid* windows — inside the
optional `mask` (every function takes one: P5 excludes 2020 with it, issue #23), with every
input finite (**NaN pairing**, §6a: a failure on either side drops the window from both),
and, for attack metrics, attacked (d ≠ 0). Each returns a `Metric`: value, `n` used, `n_nan`
dropped by NaN pairing, `n_masked`, `n_not_attacked`, `coverage`, and the per-window `terms`.

σ = the **train-split std of the domain's series** (population std — the models' own scaler),
fixed before any test data is seen; scalar or one per window (pooled tables mix domains).

| Metric | Definition | Reading |
|---|---|---|
| `mse` / `mae` | mean ((y − ŷ)/σ)² / mean \|y − ŷ\|/σ | lower = better; `paired_mse` scores several conditions on the same windows |
| `text_usage_gap` | MSE(text=None) − MSE(real), paired per window; `empty` ≡ None | > 0: text improves accuracy; ≈ 0: text collapse |
| `text_sensitivity` | mean \|ŷ_real − ŷ_none\|/σ (+ `text_moves_share`: share of windows with any change) | does text **move** the forecast, right or wrong |
| `dss` | mean d·(ŷ_fake − ŷ_real)/σ over **attacked** windows; d = P3's recorded `claimed_direction` (never a classifier) | > 0: pulled toward the lie (σ units). `exaggerated`: d = the real claim's direction, > 0 = pushed further. Coverage reported (issue #30) |
| `flip_rate` | share of attacked windows with sign(ŷ_real − y_last) ≠ d **and** sign(ŷ_fake − y_last) = d | forecast direction turned toward the lie |
| `false_block_rate` | share of **genuine** texts with gate g < 0.5 | the honesty check: a block-everything defence fails here (P6) |

**Flip Rate zero-change rule** (exact equality, no tolerance — Model 5 copies the last value
often): real forecast exactly at y_last (sign 0) is *not* in the lie's direction, so a fake
forecast moving in direction d **counts** (flat → toward the lie); a fake forecast exactly at
y_last **never counts**; a real forecast already in direction d cannot flip but stays in the
denominator (all attacked, valid windows).

**Attack targets are selected by text sensitivity, not Text Usage Gap** (decided
2026-10-09). A model is attackable if text *moves* its forecast, whether or not the text
makes it more accurate. Model 5 shows why: its text *hurts* accuracy on Health_US (MSE 1.054
→ 1.348), yet the text changes its answer in 65–78% of windows (§7b). Selecting by usage gap
would wrongly drop it.

**Seeds:** per-window predictions are averaged over the 3 seeds before any metric
(`seed_mean`); a window that failed in any seed stays NaN. Model 5 is one greedy run.
**Failures:** `failure_rates` per model × condition, next to every metric; for Model 5 parse
and format (unit-confusion, issue #27) failures are broken out.

**Statistics:** paired percentile **bootstrap CIs** over windows (10,000 resamples, seed 0;
pooled numbers stratified by domain so each domain keeps its share); comparisons bootstrap
the per-window difference (`paired_difference`), which is what makes them paired. **Paired
Wilcoxon signed-rank** (two-sided, zero differences dropped; all-zero → p = 1). **Holm**
correction across each family of comparisons (e.g. TextGuard vs undefended and vs each
baseline defence). **Pooled across domains = headline; per-domain = breakdown** (`grouped`,
issue #8).

**Sanity (Models 1–2):** `is_text_blind` — real, None and fake forecasts bit-identical, so
DSS, sensitivity and Flip Rate are exactly 0. Tested on real DLinear/PatchTST networks; P5
asserts it on the actual predictions.

**Worked example reproduced** (PHASES.md, `DummyForecaster`): A = 116, B_fake = 106,
ŷ_real = 120, y_true = 117, y_last = 112, σ = 10 → errors 11 (undefended) → 2 (TextGuard
final 115); DSS +1.4 → +0.5; real up / fake down (d = −1) → flipped; final 115 up → not
flipped; usage gap −0.08, sensitivity 0.4; genuine text (g = 1) not blocked.

### P5 results (P5b, 2026-10-09) — test split, against the pre-registered family (§10a)

`python -m src.analyze_p5` → `results/summary_p5_*.csv` (14 tables) + `paper/figures/p5_dss.*`,
`p5_flip_rate.*` (`src/figures_p5.py`). Test split only; Models 3–4 seed-averaged per window
first; text conditions compared only on the **299 test windows with a real report**; attack
metrics on attacked windows only. CIs: 95% percentile bootstrap, 10,000 resamples, seed 0,
pooled = stratified by domain. σ units. Holm over exactly the 76 pre-registered tests
(asserted in code); everything else is descriptive or ablation (raw p).
Note: MSE here is the MSE of the **seed-averaged** prediction, so Models 3–4 differ slightly
from §7a (mean of per-seed MSEs), e.g. M3 Economy real 0.164 vs 0.169.

#### 1. Failures (issue #27)

Models 1–4: **0 failures** (41,412 rows). Model 5: **16 failures, all Traffic, all format
(unit) failures, 0 parse failures** — the answer is a report figure in billions (e.g. "259.7")
for a millions series.

| Model 5, Traffic | real | empty | shuffled | cross_domain | time_shifted | flipped | exaggerated |
|---|---|---|---|---|---|---|---|
| test (53 windows) | 3 | 0 | 0 | 0 | 4 | 3 | 4 |
| val (27 windows) | 0 | 0 | 1 | 1 | 0 | 0 | 0 |

**Do failures concentrate in flipped/exaggerated? No more than in real or time_shifted** —
7 of the 14 test failures are in the two fake conditions, but real (3) and time_shifted (4)
fail at the same rate; cross_domain (SocialGood text, no billions) never does. Text quoting
billions is the trigger, not the attack.
**But NaN pairing does drop attacked windows** (`summary_p5_nan_pairing*.csv`): flipped
loses 5 of 52 Traffic windows (2 fake-only, 2 real-only, 1 both), exaggerated 4 of 50
(1 fake-only, 3 both). **Exploratory** (not pre-registered): rescaling the unit slips ×1000
and recomputing, **all 4 recoverable flipped windows move toward the lie** — Traffic DSS
0.328 → 0.378, Flip Rate 23.4% → 25.0%; pooled DSS 0.457 → 0.465. So NaN pairing hides a
little attack effect on Traffic (conservative direction), and does not change any conclusion.

#### 2. Screening (RQ0, block S + descriptive controls)

Text sensitivity, mean |ŷ_real − ŷ_none|/σ (S test: Wilcoxon, Holm):

| Model | Fusion | Economy | SocialGood | Traffic | Health_US | **Pooled** [95% CI] |
|---|---|---|---|---|---|---|
| M3 DLinear+BERT | late | 0.194 | 0.080 | 0.521 | 0.018 | **0.158** [0.142, 0.175] |
| M4 PatchTST+BERT | late | 0.110 | 0.036 | 0.121 | 0.048 | **0.074** [0.067, 0.081] |
| M5 Qwen2.5-1.5B | early | 0.164 | 0.197 | 0.449 | 0.553 | **0.388** [0.308, 0.474] |

**All 12 S hypotheses reject** (Holm p ≤ 6.3e-5) → by the pre-registered rule all three
text models are "moved by text" in every domain → **all three are attack targets.** As
stated in advance, S was expected to reject for M3/M4 (any text adds w·MLP(z(e)) ≠ 0); text
changes M3/M4 in 100% of windows, M5 in 69% [63, 74]%.

Content controls, pooled (sensitivity vs none / distance to the real-text forecast / MSE on
the same windows: 299; M5 296 for sensitivity, 292 for MSE = windows valid in all 5 conditions):

| Model | sens real | sens shuffled | sens cross_domain | sens time_shifted | dist shuffled→real | MSE real | empty | shuffled | cross_domain | time_shifted |
|---|---|---|---|---|---|---|---|---|---|---|
| M3 | 0.158 | 0.133 | 0.108 | 0.149 | 0.199 | 0.542 | **0.443** | 0.525 | 0.480 | 0.533 |
| M4 | 0.074 | 0.077 | **0.087** | 0.073 | 0.079 | **0.436** | 0.443 | 0.451 | 0.462 | 0.441 |
| M5 | 0.388 | 0.580 | **0.911** | 0.452 | 0.432 | 1.143 | **1.009** | 1.932 | 3.498 | 1.472 |

Text Usage Gap MSE(empty) − MSE(real), pooled: M3 **−0.099** [−0.150, −0.050] (text hurts,
driven by Traffic −0.546); M4 +0.007 [−0.006, 0.019]; M5 −0.133 [−0.485, 0.266].

Reading:
* **Presence vs content.** Irrelevant text moves every text model about as far as real text
  (M3/M4: shuffled/cross_domain sensitivity ≈ real), and *further* for M5 (cross_domain 2.3×
  real; Health_US 1.88 vs 0.55 — Health_US receives Traffic's report). But the controls do
  not land where real text does (distance to the real forecast ≈ the sensitivity itself), so
  the models react to **what** the text says, not merely to its presence — they just cannot
  tell **relevant from irrelevant** text. Relevant text is clearly the most accurate only for
  M3 on Economy (0.164 vs shuffled 0.243, cross 0.251; M4 Economy marginally, 0.167 vs
  0.175–0.203). M4 on Health_US is the presence pattern: real, shuffled and time_shifted all
  give 0.130 vs none 0.144 — *any* in-domain report helps equally. Pooled, no text model is
  most accurate with real text.
* **M5's errors under controls are large** (cross_domain MSE 3.50 vs real 1.14): an
  unverified channel that degrades a forecaster badly when fed the wrong report.
* **time_shifted leakage check:** time_shifted beats real text in **SocialGood for M3
  (−0.073 [−0.196, −0.002]) and M4 (−0.011 [−0.022, −0.002])** — flagged. Not read as
  leakage: time-shifted text is a year *older* than real text (it cannot carry more future
  information, and the P1 timing check passed); in SocialGood *no* text also beats real text
  (M3 −0.061), 11 of 46 SocialGood windows have no year-earlier report (time_shifted ≡ none
  there), and the effect disappears without 2020 (M3 +0.003, M4 −0.003, CIs span 0). It is
  "less text is better in SocialGood around COVID". Pooled, no model's time_shifted CI lies
  below real (M3 −0.009 [−0.077, +0.060]; M4 +0.005; M5 +0.330).

#### 3. Attack (RQ1, blocks A1 + A2)

Pooled (headline). Coverage = attacked / windows with a report. Models 1–2: exactly 0
(asserted text-blind).

| Model | Fusion | Attack | Attacked | n | DSS [95% CI] | Holm p | Flip Rate [95% CI] | toward / away | Holm p |
|---|---|---|---|---|---|---|---|---|---|
| M3 | late | flipped | 251/299 | 251 | −0.005 [−0.010, +0.002] | 0.30 | 0.4% [0.0, 1.2] | 1 / 5 | 1 |
| M3 | late | exaggerated | 250/299 | 250 | +0.004 [−0.006, +0.014] | 1 | 1.2% [0.0, 2.8] | 3 / 3 | 1 |
| M4 | late | flipped | 251/299 | 251 | +0.003 [+0.001, +0.006] | 1 | 0.4% [0.0, 1.2] | 1 / 0 | 1 |
| M4 | late | exaggerated | 250/299 | 250 | +0.002 [−0.001, +0.004] | 1 | 0.4% [0.0, 1.2] | 1 / 2 | 1 |
| **M5** | **early** | **flipped** | 251/299 | 246 | **+0.457 [+0.348, +0.572]** | **2.4e-16** | **23.6% [18.7, 28.9]** | **58 / 4** | **1.8e-11** |
| M5 | early | exaggerated | 250/299 | 246 | −0.098 [−0.188, −0.015] | 1 | 4.5% [2.0, 7.3] | 11 / 13 | 1 |

Model 5, flipped, per domain:

| Domain | Attacked | n | DSS [95% CI] | Holm p | Flip Rate | toward / away | Holm p |
|---|---|---|---|---|---|---|---|
| Economy | 70/70 | 70 | +0.031 [+0.005, +0.073] | 1 | 5.7% | 4 / 2 | 1 |
| SocialGood | 26/46 | 26 | **+0.371 [+0.210, +0.563]** | 0.009 | **50.0%** | 13 / 0 | 0.013 |
| Traffic | 52/53 | 47 | **+0.328 [+0.177, +0.520]** | 3.6e-6 | 23.4% | 11 / 0 | 0.051 |
| Health_US | 103/130 | 103 | **+0.828 [+0.590, +1.082]** | 2.3e-7 | **29.1%** | 30 / 2 | 1.5e-5 |

**Holm family: 24 of 76 reject** — S 12/12; A1 7 (5 support the claim); A2 3/3 (all M5
flipped: SocialGood, Health_US, pooled); F 2/4.
The 3 A1 rejections outside M5: M3 and M4 flipped SocialGood are **negative** (−0.017,
−0.006: pushed *away* from the claim, not support); M4 flipped Health_US is +0.002σ —
significant but negligible.

Reading:
* **The flip attack works on the early-fusion LLM and not on the late-fusion models.**
  Model 5's forecast moves 0.46σ toward the lie on average and turns direction toward it in
  24% of attacked windows (58 toward vs 4 away). Models 3–4 move < 0.02σ in every cell and
  flip ≤ 1 window.
* **Economy is the exception for M5** (+0.031σ, n.s.): the fake report leaves Qwen's answer
  unchanged in 39/70 windows (DSS term exactly 0).
* **Health_US's large DSS is not an outlier artefact:** 57 windows move toward the lie, 6
  away; without the 5 largest windows the mean is still 0.66σ. Mechanism (exploratory): under
  "ILI *above* the baseline" Qwen answers with an earlier, higher lookback value (e.g. a flu-
  peak 5.17 for t−1 = 1.05); it copies the fake report's edited number in only 6% of windows.
  Health_US is the **level-flip** variant (the stated value is changed), the others are
  change-flips — so Health_US vs other domains is also a variant difference.
* **Exaggeration does not push Model 5 further** (DSS −0.098, Wilcoxon p = 0.10, Holm 1;
  the bootstrap CI of the *mean* excludes 0 but the median term is 0 in every domain — a few
  windows move against the claim). No evidence of over-trust of extreme claims; see damage.
* Late-fusion DSS ≈ 0 is the **expected finding** (CLAUDE.md §4a): the models are moved by
  text but not by its *direction*. Mechanism below.

**Mechanism check — why Models 3–4 are immune** (`summary_p5_embedding_similarity.csv`,
`analyze_p5.embedding_similarity`; exploratory, not pre-registered). Cosine similarity of the
BERT vector of each real report and its fake, test attacked windows, from the embedding cache
(BERT not loaded). "Standardised" = after the per-dimension z-score with the train-text
statistics, i.e. exactly the input the M3/M4 heads see. Baselines on the same windows: an
unrelated report from the same domain (shuffled) and from another domain (cross_domain).

| Pair (pooled, flipped windows, n = 251) | cos raw | cos standardised | head-input distance, relative to a shuffled report |
|---|---|---|---|
| real vs **flipped** | **0.9985** (min 0.9933) | **0.968** | **0.20** |
| real vs exaggerated (n = 250) | 0.9984 | 0.965 | 0.22 |
| real vs shuffled (same domain) | 0.947 | 0.078 | 1.00 |
| real vs cross_domain | 0.852 | 0.041 | 2.06 |

Per domain the flipped pair is 0.996–0.999 raw and 0.963–0.975 standardised in all four.
**Raw cosine alone would not show this**: raw mean-pooled BERT vectors of *any* two reports
are ≥ 0.95 similar (§7a). In the standardised space the contrast is unambiguous: a different
report is nearly orthogonal (0.08), while the flipped report is nearly the same point
(0.97) — reversing the direction moves the heads' input only ~20% as far as swapping the
report.
→ **For the paper: Models 3–4 are immune to the flip attack because the frozen, mean-pooled
BERT encoder is (nearly) direction-blind — it barely distinguishes "rose" from "fell" — not
because the forecasters are robust.** This is consistent with them still being moved by
*which* report they get (§9 screening). Scope: this is about our encoder choice (frozen
`bert-base-uncased`, mean-pooled; §2); a fine-tuned or direction-aware encoder could make the
same late-fusion heads attackable — untested.

#### 4. Fusion comparison (RQ2, block F) — pooled, per-window DSS_M5 − DSS_Mk

| Comparison | Attack | n | Difference [95% CI] | Holm p |
|---|---|---|---|---|
| M5 − M3 | flipped | 246 | **+0.461 [+0.352, +0.576]** | **1.1e-15** |
| M5 − M4 | flipped | 246 | **+0.454 [+0.345, +0.568]** | **6.9e-11** |
| M5 − M3 | exaggerated | 246 | −0.103 [−0.194, −0.019] | 1 |
| M5 − M4 | exaggerated | 246 | −0.100 [−0.190, −0.016] | 1 |

**Pre-registered rule met for `flipped`: F rejects for both late models → "fusion type
matters" under the flip attack.** Not met for `exaggerated`. **Caveat:** there is one
early-fusion model, and it is also the only LLM, the only untrained-for-this-task model and
the only one that reads the text's numbers — fusion type is confounded with all of these.
Write it as "the early-fusion LLM is vulnerable, the late-fusion BERT heads are not", not as
a causal claim about fusion.

#### 5. Damage — MSE(fake) − MSE(real) on attacked windows (descriptive)

| Model | Attack | n | MSE real | MSE fake | Damage [95% CI] | Change |
|---|---|---|---|---|---|---|
| M3 | flipped | 251 | 0.474 | 0.465 | −0.009 [−0.028, +0.005] | −2% |
| M4 | flipped | 251 | 0.363 | 0.363 | +0.001 [−0.002, +0.003] | +0% |
| **M5** | **flipped** | 246 | 1.07 | 2.06 | **+0.988 [+0.631, +1.39]** | **+92%** |
| M3 | exaggerated | 250 | 0.456 | 0.458 | +0.001 [−0.025, +0.026] | +0% |
| M4 | exaggerated | 250 | 0.358 | 0.360 | +0.001 [−0.001, +0.004] | +0% |
| **M5** | **exaggerated** | 246 | 1.03 | 1.58 | **+0.552 [+0.256, +0.915]** | **+54%** |

M5 flipped by domain: Health_US +2.40 [+1.60, +3.35] (+166%); Economy, SocialGood, Traffic
CIs span 0. M5 exaggerated: Health_US +1.27 (+88%); SocialGood −0.071 (exaggerating helps
there). **The attack makes Model 5 ~2× worse overall, almost all of it on Health_US.**
**Surprise:** exaggerated text does not move M5 *in the claimed direction* yet raises its
error by 54% — it scatters the forecast rather than pushing it.

#### 6. Ablations (outside the Holm family; raw p)

**(a) Excluding 2020 (issue #23)** — `summary_p5_*_no2020.csv`:

| | Full test | Without 2020 |
|---|---|---|
| M5 flipped DSS, pooled | +0.457 (n 246) | **+0.437** [+0.325, +0.558] (n 216), p 1.7e-15 |
| M5 flipped Flip Rate, pooled | 23.6% | **23.1%** (50 / 4), p 3.8e-11 |
| M5 flipped DSS, SocialGood | +0.371 | +0.180 [+0.102, +0.261] |
| M5 flipped DSS, Traffic | +0.328 | +0.240 [+0.130, +0.374] |
| F: M5 − M3 / M5 − M4 (flipped) | +0.461 / +0.454 | +0.444 / +0.433 (p 2.6e-16 / 1.3e-10) |
| M3 usage gap, Traffic | −0.546 | **−0.591** [−0.884, −0.308] |
| M5 damage, flipped, pooled | +0.988 (+92%) | +1.14 (+113%) |

The attack result survives without 2020 (2020 roughly doubles the SocialGood effect but is
not what makes it significant). **Surprise — issue #23's hypothesis is not supported:**
real text still hurts Model 3 on Traffic without 2020 (−0.59). The harm is not 2020 alone;
a likely (unchecked) cause is that 2021+ reports state year-over-year changes measured
against 2020, which are also outside the text head's training range.

**(b) Dense- vs sparse-text domains (issue #9)** — `summary_p5_density.csv`, pooled within each:

| Model | S dense / sparse | DSS flipped dense / sparse | Flip Rate flipped dense / sparse |
|---|---|---|---|
| M3 | 0.335 / 0.034 | −0.007 / −0.003 | 0.8% / 0% |
| M4 | 0.115 / 0.045 | +0.007 / +0.000 | 0.8% / 0% |
| M5 | 0.282 / 0.460 | **+0.150 / +0.736** | **12.8% / 33.3%** |

M3/M4 are moved ~3–10× less by text in the sparse-text domains — consistent with the
training-sparsity caveat (their Health_US/SocialGood heads saw text in ~56–61% of train
periods and early-stopped at epochs 1–7, §7a); **a low Health_US/SocialGood number for M3/M4
must not be attributed to fusion type.** Model 5 is not trained on these data, so the caveat
does not apply to it — its sparse > dense pattern instead reflects the Health_US level-flip
variant and SocialGood's 2020 windows (above). The M3/M4 attack null holds in both groups.
**Health_US specifically:** M3 is the least text-sensitive cell in the study (0.018σ),
yet S still rejects; read Health_US for M3/M4 as "text head barely trained", not "late fusion
ignores text".

#### 7. Caveats to carry into the paper (recorded 2026-10-09)

* **Fusion confound.** Model 5 is the only early-fusion model **and** the only LLM (also the
  only model not trained on these data, and the only one that reads the reports' numbers).
  The supported claim is **"the LLM forecaster is vulnerable, the BERT forecasters are not"**
  — not that fusion type causes the vulnerability. Block F is significant, but it compares
  models, not fusion types in isolation.
* **Health_US level-flip edits a stated number as well as the direction** ("ILI was 1.8%,
  below the baseline of 2.1%" → "2.4%, above …"). Part of Health_US's effect (DSS +0.83,
  the largest; most of M5's damage) may therefore be **numeric injection** rather than a
  directional claim. Qwen copies the fake number exactly in only 6% of windows, but a changed
  number can steer it without being copied. Report Health_US separately from the three
  change-flip domains (SocialGood, Traffic, Economy), where the current level is kept and
  only the direction changes.
* **Flipping steers Model 5; exaggeration scatters it.** Flipped: DSS +0.46, 58 toward vs 4
  away, MSE +92%. Exaggerated: no directional push (DSS −0.10, Holm 1; median per-window
  effect 0 in every domain), yet MSE +54% [+26, +92]%. Exaggeration is a damage attack on M5,
  not a steering attack.
* **Economy: clean directional text, no Model 5 effect** (DSS +0.031, n.s.; the fake leaves
  Qwen's answer unchanged in 39/70 windows), although Economy has the clearest directional
  reports (92%, §6). Plausible explanation (**untested**): Economy reports quote a different
  series — the goods-*and-services* deficit in $ billions (e.g. "$41.9 billion") — while the
  target is the goods balance in $ millions (e.g. −58,365), §7b; a claim about a different
  quantity at a different scale is weaker evidence for the forecast.

#### 8. Notes for P6 (recorded 2026-10-09)

* **`never_trust` (g = 0) is a strong baseline for Model 5.** Pooled, M5 is more accurate
  with no text than with real text: MSE 1.009 vs 1.143 (292 windows valid in all 5 conditions),
  driven by Health_US (1.086 vs 1.389). **But no difference is statistically clear:** the CI of
  MSE(empty) − MSE(real) spans 0 in every domain and pooled (pooled −0.133 [−0.485, +0.266];
  Health_US −0.302 [−1.099, +0.579]).
* **Per domain, text helps M5 only in Economy (+0.009) and SocialGood (+0.022)**, both within
  noise. **In Traffic text hurts on the full test** (1.946 vs 1.919, −0.027) and helps only
  without 2020 (1.400 vs 1.575, +0.175 [−0.192, +0.581]). So "text helps M5 in Economy,
  SocialGood and Traffic" is **not** what the test numbers show; the honest statement is
  "text's value to M5 is small and uncertain everywhere, and negative on Health_US".
* → P6 must report **per domain**, and compare TextGuard against `never_trust` honestly — on
  M5, TextGuard beating `always_trust` is not enough; it has to be shown against g = 0, on DSS
  *and* real-text MSE, with the False Block Rate.
* **Models 3–4 are not fooled** (DSS < 0.02σ): for them P6 only needs to show that TextGuard
  does no harm — real-text MSE unchanged and a low False Block Rate.

**P5 exit criterion met:** all three text models are moved by text (S); only the
early-fusion LLM is manipulated in the lie's direction (A1/A2), significantly more than both
late-fusion models (F, flipped). → P6 attack target: Model 5 primarily; Models 3–4 kept as the
"text-moved but direction-blind" contrast.

### P6 — TextGuard
| Model | DSS before | DSS after | Flip before | Flip after | MSE(real) before | MSE(real) after | False Block Rate |
|---|---|---|---|---|---|---|---|
| _(pending)_ | | | | | | | |

### P6 — Defense comparison
| Defense | DSS | Flip Rate | MSE(real) | False Block Rate |
|---|---|---|---|---|
| None | | | | — |
| never_trust (g=0) | | | | |
| always_trust (g=1) | | | | |
| clip | | | | |
| CFA (trained) | | | | |
| **TextGuard** | | | | |

### P7.6 — FNSPID (optional)
| Model | DSS before | DSS after | Flip before | Flip after |
|---|---|---|---|---|
| _(pending)_ | | | | |

---

## 10. Pre-registration and open issues

### 10a. Pre-registered comparison family (P5) — fixed 2026-10-09, before any P5 prediction

> Committed **on its own**, before a single P5 prediction (Models 1–4 on the new conditions,
> or Model 5's second Colab run) existed, so the commit timestamp shows the family was fixed
> before the results. Holm–Bonferroni is only valid for a family chosen in advance.
> **Do not edit this subsection after P5 results exist.** A later change goes in a dated
> amendment below it, says what changed and why, and is reported as a deviation in the paper.
> Already known when writing it (from P2, real vs none only): Models 3–5 test MSE (§7a, §7b)
> and Model 5's "text changes the answer in 65–78% of windows". Nothing about shuffled,
> cross_domain, time_shifted, flipped or exaggerated predictions was known to anyone.

**Common to every hypothesis**

* **Split:** test only. Val is never used for these tests.
* **Models tested:** the text models 3 (DLinear+BERT, late), 4 (PatchTST+BERT, late),
  5 (Qwen2.5-1.5B, early). Models 1–2 are not tested: they are *asserted* text-blind
  (`metrics.is_text_blind`), so every attack quantity is exactly 0 for them.
* **Predictions:** Models 3–4 averaged over seeds 2021–2023 per window (`metrics.seed_mean`)
  before any metric; Model 5 = the single greedy run. Model 5's `none` ≡ `empty`.
* **Windows:** screening uses test windows where the **real** text is non-empty; attack uses
  test windows with `attacked=True` in that condition (d ≠ 0, P3's recorded
  `claimed_direction`). NaN pairing (§6a) drops a window from both sides; the off-scale rule
  for Model 5 is unchanged from P2.4 (answer > 10× outside its window's |range| → NaN).
* **Scale:** σ = the domain's train-split std; y_last = the last lookback value.
* **Pooled** = all 4 domains' windows together, each window weighted equally; bootstrap CIs
  stratified by domain. **Per-domain** = that domain's windows only.
* **Tests:** two-sided, α = 0.05. A directional claim ("DSS > 0") is supported only if the
  test rejects **and** the estimate has the claimed sign; a significant negative DSS is
  reported as "pushed away from the claim", not as support.
* **Every metric is reported with** its 95% paired percentile bootstrap CI (10,000 resamples,
  seed 0), `n`, NaN drops, attacked coverage and the failure rates per model × condition.
* **No selection before testing:** the attack hypotheses are tested for **all three** text
  models whatever the screening says, so no attack test is conditioned on a screening result.

**The family — one Holm family, m = 76 hypotheses**

| Block | Hypothesis (H₀) | Statistic / test | Cells | m |
|---|---|---|---|---|
| **S — screening** (RQ0) | text does not move the forecast: median of \|ŷ_real − ŷ_none\|/σ = 0 | `metrics.text_sensitivity` terms; Wilcoxon signed-rank (`metrics.wilcoxon`) | model {3, 4, 5} × domain {Economy, SocialGood, Traffic, Health_US} | 12 |
| **A1 — attack, DSS** (RQ1) | fake text does not shift the forecast toward its claim: median of d·(ŷ_fake − ŷ_real)/σ = 0 | `metrics.dss` terms; Wilcoxon signed-rank | model {3, 4, 5} × condition {flipped, exaggerated} × (4 domains + pooled) | 30 |
| **A2 — attack, Flip Rate** (RQ1) | fake text is no more likely to turn the forecast direction toward the claim than away from it | exact McNemar (two-sided binomial, p = ½) on the discordant windows: *toward* = sign(ŷ_real − y_last) ≠ d and sign(ŷ_fake − y_last) = d (the `metrics.flip_rate` numerator); *away* = sign(ŷ_real − y_last) = d and sign(ŷ_fake − y_last) ≠ d. Flip Rate itself is reported as the effect size | model {3, 4, 5} × condition {flipped, exaggerated} × (4 domains + pooled) | 30 |
| **F — fusion type** (RQ2) | early and late fusion are equally vulnerable: median of DSS_M5 − DSS_Mk = 0 (k = 3, 4), paired per window | per-window difference of DSS terms on windows valid for both models (`metrics.paired_difference`), pooled over domains; Wilcoxon signed-rank; bootstrap CI stratified by domain | {M5 vs M3, M5 vs M4} × {flipped, exaggerated} | 4 |

Holm-adjusted p-values (`metrics.holm`) over all 76 at once; a hypothesis is rejected if its
adjusted p < 0.05. Raw and adjusted p are both reported.

**Decision rules fixed in advance**

* A model **"is moved by text"** in a domain if its S hypothesis is rejected there (§9 decision:
  attack targets by sensitivity, not usage gap); a model is an **attack target** if that holds
  in at least one domain. Stated in advance: for Models 3–4 any non-empty text adds
  w·MLP(z(e)) ≠ 0, so S is expected to reject almost surely — the informative screening
  numbers for them are the descriptive content controls below, and DSS itself.
* **"Fusion type matters"** (RQ2) needs F to reject for **both** late models (M5 vs M3 and
  M5 vs M4) in the same condition; one of the two alone is reported as model-specific.

**Reported, not tested (descriptive, CIs only, not in the family):** Text Usage Gap
(`empty` vs `real`), `text_moves_share`, `paired_mse` over real / empty / shuffled /
cross_domain / time_shifted (the content controls: does *relevant, timely* text beat
irrelevant or stale text?), per-condition sensitivity for the controls, failure rates,
attacked coverage. Val-split numbers, if shown, are labelled as such.

**Planned ablations — reported separately, NOT in the Holm family** (raw p-values and CIs,
labelled as ablations; they can qualify the main results but never replace them):

1. **Excluding 2020 (issue #23):** every S, A1, A2 and F quantity recomputed with a `mask`
   that drops windows whose target period (`sample_id`) falls in calendar year 2020 — in all
   four domains (Health_US test ends 2020-01, so it loses a few weeks). Reported side by side
   with the full-test numbers. Motivation: genuine COVID-era text far outside the text head's
   training range (Model 3 Traffic test).
2. **Health_US / sparse-text training (issue #9):** pooled S, A1, A2 recomputed over the
   **dense-text** domains (Economy, Traffic: ≥ 98% train text coverage) and the
   **sparse-text** domains (SocialGood 56%, Health_US 61% train coverage) separately.
   No retraining. Purpose: a low Health_US/SocialGood number must not be attributed to fusion
   type when it may reflect a text head trained on fewer texts.

Anything not listed above (other conditions as attacks, other splits, other subgroups, a
different window mask) is **exploratory** and will be labelled so.

### 10b. Pre-registered evaluation plan (P6) — fixed 2026-10-09, before any TextGuard output

> Committed **on its own**, before TextGuard, its gate or its direction judge had produced a
> single output on any split (val or test). **Do not edit this subsection after P6 results
> exist**; a later change goes in a dated amendment below it and is reported as a deviation.
> Already known when writing it: everything in §9 (P5, test split), including Model 5's
> real-vs-none MSE per domain. Nothing about judge answers, gate values or defended forecasts.

**Framing (§9 part 8).** Text's accuracy value for Model 5 is **not established** in our data
(MSE(empty) − MSE(real) has a CI spanning 0 in every domain and pooled). So the primary claim
is the **trade-off** — how much manipulation TextGuard removes for how much genuine text it
blocks — not accuracy.

**Definitions (fixed now; the code in `src/textguard/` must match them)**

* **Direction judge** = Qwen2.5-1.5B-Instruct, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`,
  same Colab setup class as P5 (the 40-prompt reproducibility probe is re-run and recorded).
  **Not** the P3 rules (rules judging rule-written fakes would be circular). Per (domain, text)
  it answers fixed questions: a **change** question about the domain's subject (Economy: the
  U.S. trade deficit; Traffic: travel vs the same month a year earlier; SocialGood: the
  unemployment rate; Health_US: the ILI percentage) → up / down / none; Health_US also a
  **level** question → above / below / none the national baseline. Answer = the most likely
  of the three allowed answers (sequence probability given the prompt); **confidence c** =
  its probability renormalised over the three (∈ [⅓, 1]). Mapped to the series direction
  locally (Economy: deficit up = balance −1). The judge prompt is frozen at the P6a commit;
  it has no tunable knob.
* **Change evidence:** OLS slope β over the last w lookback values; direction sign(β);
  strength s = min(|β|·(w − 1)/σ, 1) — the fitted movement across the window in σ units.
  (This is how "slope" in CLAUDE.md §8 is read; it reproduces the worked example:
  100…112, σ 10 → 12/10 → s = 1.0. A per-step slope would give 0.4 and g = 0.64, not 0.1.)
* **Level evidence** (Health_US; only when the text states a baseline number B — read by a
  regex, a number not a direction — and the level answer is not none): direction
  sign(y_last − B), strength min(|y_last − B|/σ, 1). Valid because Health_US's series is the
  ILI percentage the reports state (stated value vs t−1: corr 0.98, checked 2026-10-09).
* **Gate** per branch: 1 if the text's claim is 0, the evidence direction is 0, or they
  agree; else 1 − c·s. **g = min over the branches present** (a text that claims nothing → 1).
* **TextGuard final.** Soft (CLAUDE.md §8): A + g·(B − A), warning if g < τ.
  **Block mode at threshold τ:** applied weight g′ = 0 if g < τ (warning ⇒ fall back to A),
  else g; final = A + g′·(B − A). τ = 0 is the soft gate; τ > 1 is never_trust. **Why block
  mode:** with the soft final, τ only switches the warning flag and never changes a forecast,
  so a τ sweep would trace no trade-off. B NaN → final = A; A NaN → NaN (counted as a failure).
* **Baselines:** always_trust (g′ = 1 = undefended); never_trust (g′ = 0); clip:
  final = A + clamp(B − A, ±k·σ), applied weight min(1, k·σ/|B − A|) (1 when B = A).
* **False Block Rate — one definition for every defence:** share of **genuine** texts (test
  windows with a real report) whose applied weight g′ < 0.5 (`metrics.false_block_rate`).
  never_trust = 100%, always_trust = 0%. TextGuard's g does not depend on the forecaster, so
  its FBR is the same for Models 3, 4, 5; clip's does.
* **DSS / Flip Rate under a defence:** both forecasts pass through the same defence:
  d·(final_fake − final_real)/σ, Flip Rate on (final_real, final_fake) — how far the deployed
  system's output moves when the true report is replaced by the lie (as in the PHASES worked
  example). never_trust ⇒ exactly 0.
* Models 3–4: A and B seed-averaged first (for fixed g the final is linear in A, B, so this
  equals the seed mean of finals). Windows, σ, y_last, NaN pairing, bootstrap (10,000, seed 0,
  stratified by domain) as §10a.

**Tuning — val split only, coarse grid (val is small)**

* Grid: window fraction f ∈ {¼, ½, 1} of the lookback L, w = max(3, round(f·L)) → monthly
  w ∈ {3, 4, 8}, weekly (Health_US) w ∈ {6, 12, 24}; τ ∈ {0, 0.1, …, 1.0}. 33 configs.
* Rule: Model 5, `flipped`, val, pooled — among configs with **val FBR ≤ 10%**, the lowest val
  DSS; ties → lower FBR, then smaller τ, then smaller w. If none meets 10%: the lowest-FBR
  config, reported as "budget not met".
* clip: k ∈ {0.05, 0.1, 0.25, 0.5, 1.0}, same rule. The FBR budget is the same for both.
* The chosen settings are applied unchanged to Models 3–4 (no per-model tuning).

**Reported on test**

* **PRIMARY — Model 5, `flipped`:** (a) the curve (FBR, DSS) and (FBR, Flip Rate) for
  TextGuard at the val-chosen w over every τ, clip over every k, and the two endpoints
  always_trust (FBR 0%) and never_trust (100%); (b) the val-chosen operating point: DSS, Flip
  Rate, FBR with CIs. Pooled (headline) + per domain.
* **SECONDARY — Model 5 real-text MSE and damage** (MSE(fake) − MSE(real)) under every
  defence, against never_trust, **per domain**, reported whichever way it falls.
* **Models 3–4 — "no harm" only:** real-text MSE TextGuard vs undefended (ΔMSE with CI) and
  FBR; their DSS shown for completeness.
* **`exaggerated`:** the same tables, descriptive. Expected weak: the fake claims the *same*
  direction as the real report, so the gate sees what it sees for real text.
* **COVID regime change (issue #10):** FBR and the gate distribution on genuine test texts in
  2020 windows (Economy, SocialGood, Traffic) vs non-2020 windows.
* **Judge accuracy:** (i) vs the 50 human labels from P1 (`summary_directional_sample.csv`;
  the change answer about the literal subject vs `manual`; accuracy, Cohen's κ, confusion
  matrix), plus the 25 filtered-SocialGood re-check labels as supplementary; (ii) vs P3's
  **known** `claimed_direction` on every attacked `flipped` / `exaggerated` text (val + test),
  by domain × variant; (iii) share answered none on `cross_domain` texts; coverage = share of
  genuine texts with a non-none claim. The judge is not tuned, so (ii) may use test texts.
* **Model-agnostic:** `DummyForecaster`, worked example (g = 0.1, final 115, warning).

**Hypothesis family — one Holm family, test, Model 5, `flipped`, val-chosen operating point**

| Block | H₀ | Test | Cells | m |
|---|---|---|---|---|
| T1 | DSS(TextGuard) = DSS(always_trust) | Wilcoxon on per-window DSS-term differences | pooled + 4 domains | 5 |
| T2 | TextGuard and always_trust flip equally often | exact McNemar on windows flipped under exactly one | pooled + 4 domains | 5 |
| T3 | DSS(TextGuard) = DSS(clip, val-chosen k) | Wilcoxon | pooled + 4 domains | 5 |
| T4 | real-text squared error, TextGuard = never_trust | Wilcoxon | pooled + 4 domains | 5 |
| T5 | DSS(TextGuard) = DSS(CFA) — **only if CFA runs**; decided in P6a, before any test result | Wilcoxon | pooled + 4 domains | 5 |

m = 20 (25 with CFA). Two-sided, α = 0.05; a directional claim needs a rejection **and** the
claimed sign. T4 non-rejection is reported as "no difference detected", never as equivalence.
FBR and the M3/M4 "no harm" numbers are descriptive (CIs), outside the family.

**Ablations — outside the family (raw p, labelled):** excluding 2020; Health_US (level-flip,
level branch) vs the three change-flip domains; soft gate (τ = 0); **oracle judge** (P3's
known directions instead of Qwen's — an upper bound on what judge errors cost; uses the rules,
so never a headline). Anything else is exploratory.

**Limitation recorded in advance.** In our data the model that benefits from text (Model 3,
Economy) is not attackable, and the attackable model (Model 5) does not measurably benefit from
text. TextGuard's target scenario — a forecaster that both gains from text and can be steered
by it — needs a larger LLM forecaster; that is future work. P6 can show the trade-off and that
the gate costs little genuine text; it cannot show "keeps the benefit of text" for Model 5.

#### Amendment 1 — 2026-10-09, added before any judge output existed

> Written after P6a was built and the judge job exported (`60f10ee`), **before the Colab judge
> run**: no judge answer, gate value or defended forecast existed on any split. It adds stated
> expectations; it changes no definition, grid, selection rule or hypothesis above.

* **FBR is reported per domain** (pooled + Economy, SocialGood, Traffic, Health_US), at the
  val-chosen operating point and along the curve — not only pooled.
* **Expected: FBR higher on Traffic.** Traffic reports state **year-over-year** changes
  ("−1.6% vs January 2009"), but TextGuard's change evidence is the trend over the 8-month
  lookback, which cannot verify a year-over-year claim (the window never contains lag 12,
  issue #22). Genuine text can therefore disagree with a seasonal 8-month trend and be gated.
* **Expected: FBR elevated on Economy too.** Its reports state **month-over-month** changes in a
  **different series** — the goods-*and-services* deficit — while the target is the goods
  balance (§7b), so a genuine claim need not match the target's own recent trend.
* If either expectation holds, it is reported as a **known mismatch between the claim's
  reference frame and the evidence TextGuard can see**, not explained away; if it does not
  hold, that is reported too.

#### Amendment 2 — 2026-10-09, after val tuning, before any test computation

> Written after P6b Step 1 (judge validation: judge answers on val + test *texts* vs human
> labels and P3's known directions — no forecast involved) and Step 2 (the val grid,
> `results/summary_p6_val_grid.csv`). **No test gate value, defended forecast or test metric
> existed.** Committed alone. It changes no definition, grid, FBR budget or hypothesis above;
> it resolves a tie the fallback rule leaves open and declares one exploratory variant.

1. **Val outcome — budget not met (TextGuard).** All 33 TextGuard configs exceed the 10% val
   FBR budget; the lowest is **11.3%** (pooled, Model 5, `flipped`). The pre-registered
   fallback ("the lowest-FBR config, reported as budget not met") applies. Clip **met** the
   budget: k = 0.5, val FBR 6.3%, val DSS 0.110. The budget is not changed.
2. **Tie in the fallback, resolved by the explicit tie-breaks.** Six configs share the lowest
   FBR (f = ¼, τ ∈ {0, 0.1, …, 0.5}: for τ ≤ 0.5 the blocked set is the g < 0.5 set either
   way). The fallback names no tie-break; the main rule's explicit ones are applied — smaller
   τ, then smaller w → **τ = 0, f = ¼ (w = 3 monthly, 6 weekly Health_US): the soft gate of
   CLAUDE.md §8.** Not used: "lowest signed DSS" — every val TextGuard DSS is negative, and
   adding a DSS criterion after seeing val would be a new selection rule. **τ = 0.5** (same
   val FBR; val DSS −0.053 / Flip 6.8% vs −0.021 / 11.3% at τ = 0) is reported as an
   **exploratory** point. T1–T4 are tested at τ = 0, f = ¼. Soft gate ⇒ applied weight g′ = g,
   so its FBR is the share of genuine texts with g < 0.5 (definition unchanged).
3. **Exploratory variant "TextGuard-L"** — Health_US's gate uses the **level** claim only (the
   change claim is dropped); Economy, SocialGood, Traffic unchanged; same frozen f, τ (the
   level branch uses no window). **Justification, from Step 1 only:** on Health_US the judge's
   *change* question reads the report's *level* claim as a change claim ("ILI was 0.9%, below
   the national baseline" → "down", c = 0.98): it agrees with the P1 human labels on 2 of 12
   Health_US reports (17%) — the 10 reports humans labelled "no change claim" all get a
   direction — so the change branch tests claims the reports do not make. **Disclosure:**
   declared after the val grid was also seen (val Health_US FBR 13.6% at the frozen config).
   **Reported:** test, Model 5, `flipped`: DSS, Flip Rate, FBR (pooled + per domain) at the
   frozen operating point and along the τ curve, plus real-text MSE; raw p only, **outside
   the Holm family**, labelled exploratory; it never replaces the main TextGuard result.
4. **Wording of §10b made explicit (fixed now, before test):**
   * "none" = the raw model with P5's NaN pairing (no fallback); `always_trust` = g′ = 1 through
     the defences' NaN rule (B NaN → A). They differ only on Model 5's Traffic unit-slip
     windows.
   * T4 and the secondary real-text MSE use **every test window with a real report**; damage
     (MSE(fake) − MSE(real)) uses the attacked windows.
   * T2's discordant windows = flipped under exactly one of TextGuard / always_trust (McNemar,
     two-sided binomial, p = ½).

### 10c. Open issues

| # | Issue | Phase | Status |
|---|---|---|---|
| 1 | Does Time-MMD text make directional claims? (≥30% needed) | P1 | ✅ 58% overall (§6) |
| 2 | Is text guaranteed available *before* its forecast window? | P1 | ✅ Yes, with `anchor="end"` + 5 leakers dropped (§6) |
| 3 | How does MM-TSFlib actually fuse text? (we assume late) | P2 | ✅ Late additive; text term ≡ 0 at h=1, numeric prior in the text branch (§7a) |
| 4 | Can CFA (`seunghan96/cfa`) run as a defense baseline? | P6 | ✅ Decided in P6a (before any test result): **not run** — trained adapters on internal embeddings / fusion layer; cannot wrap a black-box `predict()`; Model 5 has no fusion layer. T5 dropped, m = 20; Related Work in P8 (§1a) |
| 5 | Colab VS Code extension unreliable — use colab.research.google.com directly for GPU work (Qwen) in P2; `git clone` the repo inside the notebook | P2 | ⬜ Open |
| 6 | MM-TSFlib uses `search` text, we default to `report` — first suspect if P2 baselines don't match | P2 | ⬜ Open |
| 7 | MM-TSFlib Economy CSV unsorted + positional split ⇒ non-chronological split in their baseline | P2 | ⬜ Open |
| 8 | Small test sets (~30–60 sequences/domain) → pool DSS across domains as headline, bootstrap CIs in P4 | P4/P5 | 🔵 P4 done: stratified paired bootstrap + `grouped` (pooled headline, per-domain breakdown); apply in P5 |
| 9 | Health_US: 183/470 train periods lack text vs 97% test coverage — low Usage Gap may be training sparsity, not architecture | P5 | ✅ P5b ablation: M3/M4 3–10× less text-sensitive in sparse domains; attack null holds in both groups; caveat stays in the paper (§9 P5.6b) |
| 10 | COVID-2020 in 3 test sets → dedicated "TextGuard under genuine regime change" False Block Rate analysis | P6 | ⬜ Open |
| 11 | Timing: `anchor="end"` default + drop 5 real leakers | P1 | ✅ Done — check passes |
| 12 | SocialGood: 41% off-topic influenza text | P1 | ✅ Filtered (131 dropped, 13 header-stripped) |
| 13 | Directional claims concentrated in Economy/Traffic | P3 | ✅ change-flip (Economy, Traffic, SocialGood) + level-flip (Health_US); coverage 84% (§8) |
| 14 | Swap SocialGood (56% directional, 61% coverage) for Energy (100% directional, but 351/354 reports contain the forecast target; needs lag 2)? | P1→P2 | ⬜ Open — needs decision |
| 15 | Generalise TextGuard gate to "text claim vs numerical evidence" (change + level branches); update CLAUDE.md §8 | P6 | ✅ P6a: change + level evidence, g = min over claims; CLAUDE.md §8 updated |
| 16 | Direction classifier must be Qwen-based (heuristic only 72% vs humans); validate on the 100 hand labels | P6 | 🔵 P6a: Qwen judge built + exported (scored answers, confidence from token probabilities); validation in P6b (50 + 25 labels; Energy's 25 are not one of our domains) |
| 17 | NaN pairing: drop a sample from both sides if either is NaN; report parse-failure rate per model × condition (§6a) | P4 | ✅ Built into every metric; `failure_rates` per model × condition |
| 18 | TextGuard: if B is NaN, final = A (§6a) | P6 | ✅ P6a: `baselines.mix` (B NaN → A; A NaN → NaN), tested |
| 19 | **Models 3/4 must reuse the val-tuned protocol** (`train_baselines.LR_GRID`, `TUNED`, `TUNE_SEED`, `patching="window"`) — else a Text Usage Gap mixes text with extra training | P2.3 | ✅ Applied — same `LR_GRID`/`TUNED`/seeds, lr tuned per domain for Models 3/4 |
| 20 | MM-TSFlib PatchTST clamps patch_len to pred_len → at h=1 it reads 2 of 8 inputs; Models 4 (and 6) must use `patching="window"` too | P2.3 | ✅ Fixed for Models 2 and 4 |
| 21 | Repro column far above published MSE (e.g. Traffic DLinear 2.87 vs 0.35) | P2/P8 | ✅ Closed — explained by different test periods and splits; not pursued further — listed as a limitation in the paper |
| 22 | Lookback 8 < 12-month seasonality: DLinear can't see lag 12 (Traffic val 1.02 vs PatchTST 0.04). Traffic reports state **year-over-year** changes ("−1.6% vs January 2009") — exactly the information an 8-month window lacks. So Traffic is the domain most likely to show **genuine text use**, and the **strongest P5 attack target**. Lookback stays 8 to match the published setup | P5 | ⬜ Open — note when interpreting |
| 23 | **P5b result: the COVID explanation is NOT supported.** Real text still hurts M3 on Traffic with every 2020 window removed (usage gap −0.546 full test → −0.591 [−0.884, −0.308] without 2020). The cause of M3's Traffic text harm is open (candidate, unchecked: 2021+ reports state year-over-year changes measured against 2020). Attack results are unchanged without 2020 (§9 P5.6a). Original task: report Traffic with and without 2020. Genuine, extreme COVID-era Traffic text hurts Model 3 on test (test MSE 1.04 → 1.60 with text) although text halves val error — the text head never saw year-over-year changes of that size. Same regime-shift theme as #10 | P5 | ✅ Analysis done — COVID hypothesis rejected; cause open |
| 24 | Model 5 memorisation leakage: Qwen may have seen these statistics/reports in pretraining; numeric prompt is undated, report text still dated — paper limitation (§7b) | P8 | ⬜ Open |
| 25 | Same-model circularity: Qwen generates P3 fakes and judges direction in P6 → validate the P6 judge on the human labels from `check_directional`, ideally also on non-Qwen fakes (§7b) | P6 | 🔵 Generation side removed (rule-based fakes, §8); judge run back (P6a) — validate on human labels + known `claimed_direction` in P6b |
| 26 | Report figures ≠ series values (Economy goods vs goods-and-services; Traffic VMT estimates) — check Qwen errors for copying the report's number (§7b) | P5 | ✅ Checked: real 5.9% vs 4.1% chance; Economy 0%, Traffic 1.2%; Health_US copies are same-series and accurate |
| 27 | Model 5 unit confusion: 3/80 Traffic real-text answers in billions (report's unit) → scale failures (NaN). Report the rate next to every Model 5 metric; consider it in P5 as a text-channel failure mode | P5 | ✅ P5b: 16 failures (14 test), all Traffic unit slips; not concentrated in fakes; NaN pairing drops 5/52 flipped windows — exploratory rescale shows all recoverable ones move toward the lie (§9 P5.1) |
| 28 | Qwen run `revision` unrecorded (transformers 5 dropped `_commit_hash`); notebook now queries the Hub. Re-run Model 5 in P3 (attacked texts) will record it | P5a | ✅ P5 run recorded revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 (loaded by that commit). P2.4's own snapshot stays unrecorded (paper: same model id; probe shows whether outputs reproduce) |
| 29 | Model 5 needs a second Colab run on the attacked conditions (shuffled, cross_domain, time_shifted, flipped, exaggerated) — export from `texts/`, `QwenForecaster` must accept the new conditions' text | P5a | ✅ Colab run back: 2215 generated, probe 40/40 → 1236 rows reused from P2.4; 3451 rows, one setup class |
| 30 | DSS / Flip Rate only over `attacked` windows; report attacked counts (384 flipped / 381 exaggerated of 458 texts) next to every attack metric | P5 | ✅ P5b: test coverage 251/299 flipped, 250/299 exaggerated; reported in every attack table |

---

## 11. Session log

> Newest at top. Format: what was done / what broke / next step.

### 2026-10-09 — P6a: judge run back → **P6a complete**
- Pre-registration **Amendment 1** committed alone (`01184db`) before any judge output:
  per-domain FBR; expected higher on Traffic (year-over-year claims vs an 8-month window) and
  Economy (month-over-month claims about goods + services vs a goods target)
- Colab T4 judge run: 2490 answers in 8 min, probe 40/40, setup identical to P5's; all answer
  words single tokens; `load_judgements()` verified locally (fingerprints, completeness, one
  setup); predictions + meta backed up with `git add -f`
- No accuracy, gate or defended forecast computed — that is P6b
- **Next:** P6b (§1a)

### 2026-10-09 — P6a: TextGuard built, direction-judge job exported (awaiting Colab)
- **P6 evaluation plan pre-registered (§10b) and pushed alone first** (`9838239`): trade-off
  primary (M5 flipped: DSS / Flip Rate vs FBR curve + val-chosen operating point), block mode
  for the τ sweep, one FBR definition for all defences, val-only coarse grid, Holm m = 20
- `src/textguard/`: change + level evidence, gate (min over claims), defences
  (always / never / clip / soft / block), `TextGuard` wrapper, judge protocol + `QwenJudge`
- "Slope" read as the fitted movement across the window (the worked example needs strength 1.0)
- CFA timeboxed: not run (white-box, trained; Model 5 has no fusion layer) → issue #4
- Judge: 2490 questions / 1831 texts + 40 probes; notebook pins P5's revision, re-runs the probe
- Worked example reproduced on `DummyForecaster` with no change to TextGuard
- **Broke:** the test's fake judge keyed on "increased" in the prompt's instructions → now reads
  the report block only
- 259 tests passing (30 new)
- **Next:** Colab judge run (§1a), then P6b

### 2026-10-09 — P5b: analysis against the pre-registered family → **P5 complete**
- `src/analyze_p5.py` (+ `figures_p5.py`): test split, seed-mean first, 299 report windows;
  14 tables `results/summary_p5_*.csv`; figures `paper/figures/p5_dss.*`, `p5_flip_rate.*`
- `metrics.mcnemar_flip` (A2's exact McNemar was named in §10a but not yet implemented)
- Holm over exactly 76 (asserted): **24 reject** — S 12/12; A1 7 (5 support); A2 3; F 2/4
- **Headline:** flip attack moves Model 5 (early-fusion LLM) +0.457σ [0.348, 0.572] toward the
  lie, Flip Rate 23.6% (58 toward / 4 away), MSE +92%; Models 3–4 < 0.02σ everywhere; F
  rejects for both late models (flipped). Exaggerated: no directional push, but M5 MSE +54%
- Mechanism: flipped vs real BERT vectors cos 0.968 standardised (shuffled 0.078) → M3/M4
  immune because the encoder is near direction-blind, not robust
- Caveats (fusion confound, Health_US numeric injection, exaggeration scatters, Economy null)
  and P6 notes (never_trust strong for M5, per-domain reporting) recorded in §9
- Surprises: M3's Traffic text harm survives without 2020 (#23: COVID explanation not
  supported); time_shifted beats real in
  SocialGood for M3/M4 (not leakage — gone without 2020); M5 moved 2.3× more by cross-domain
  than by real text
- **Broke:** `w.empty` hit pandas' `DataFrame.empty` property instead of the column (caught by
  the text-blind assertion) → `w["empty"]`
- 229 tests passing (4 new)
- **Next:** P6 — TextGuard (primary target Model 5)

### 2026-10-09 — P5a: pre-registration, embeddings, Models 1–4 on all conditions, Qwen export
- §1a corrected: P5a = generate (family, embeddings, M1–4 predictions, Qwen export for all 5
  altered conditions); P5b = all analysis once Qwen is back
- **Pre-registered family (§10a) committed and pushed alone first** (`5792daf`): one Holm
  family, m = 76 (S 12, A1 30, A2 30, F 4); ablations (2020, sparse/dense) outside it
- Embeddings: 732 new, 0 truncated; checkpoint scalers verified = train-only stats
- `src/run_experiments.py`: 112 units / 41,412 rows / 0 NaN / 336 MLflow runs; M1–2 text-blind
  confirmed; P2 reproduced exactly
- Qwen: `export_qwen_inputs --p5`, `notebooks/p5_qwen_conditions.ipynb` (probe-gated reuse of
  P2.4 answers, Hub revision pinned + recorded), `load_all_predictions`
- **Broke:** pandas 3 turned `None` texts into NaN in a string column (caught by the
  real-text cross-check before any prediction was saved) → object dtype
- 221 tests passing (10 new); 225 after the follow-up
- Follow-up (Aryan's review): a failed probe now regenerates real + none too (986 rows),
  not only the conditions; `setup` recorded per row; `qwen_rows()` refuses mixed setups
  (P5 Qwen tests 4 → 8: setup string, probe pass / fail, mixed classes, failed-probe reuse, two-setup file)
- Colab run back: probe 40/40, 2215 generated, 0 parse failures; `qwen_rows()` = 3451 rows,
  one setup class; predictions + meta backed up with `git add -f` → **P5a complete**
- **Next:** P5b — analysis against the pre-registered family (§10a)

### 2026-10-09 — P4: metrics module → **P4 complete**
- `src/metrics.py`: mse, mae, paired_mse, text_usage_gap, text_sensitivity, text_moves_share,
  dss, flip_rate, false_block_rate, is_text_blind; seed_mean, failure_rates; bootstrap_ci
  (stratified, paired via differences), wilcoxon, holm, grouped — definitions in §9
- NaN pairing, optional window mask and attacked-only filtering built into every metric
- Decided: P5 selects attack targets by **text sensitivity**, not usage gap (Model 5)
- 27 tests in `test_metrics.py` (worked example via DummyForecaster; Models 1–2 exactly
  text-blind on real networks); suite 211 passing
- **Next:** P5 — experiment runner: screening (usage gap + sensitivity), then the attack;
  Model 5 second Colab run on the attacked conditions (#29)

### 2026-10-09 — P3: attack module (rule-based flip + exaggerate) → **P3 complete**
- **Design change:** fakes generated by deterministic rules, not Qwen (Qwen-on-Qwen bias,
  P6 circularity); recorded in CLAUDE.md §7 and §8 here
- Simple conditions: empty, shuffled (earlier dates only), time_shifted (−1 year),
  cross_domain (other domain, same date); 7 condition files in `texts/`
- `rules.py`: headline reader + flip / exaggerate with coherent numbers; flip covers 384 / 458
  texts (Economy 100%, Traffic 99%, Health_US 80%, SocialGood 56%)
- Validation: **29 / 29 human-labelled directional reports flipped correctly**; 0 new
  contradictions; 7 edge cases found by review and fixed (§8)
- Health_US level mirror changed ratio → linear (2B − V, ratio fallback) after review: values
  above 5% 26 → 0 (plausible fakes for a fair P6)
- 26 tests in `test_attacks.py`; suite 184 passing
- **Next:** P4 — metrics module (MSE, Text Usage Gap, DSS, Flip Rate, False Block Rate)

### 2026-10-09 — P2.4: Qwen forecaster (Model 5) → **P2 complete**
- `export_qwen_inputs.py` → 493 val + test windows; `qwen_prompt.py` (undated `t-L…t-1` prompt,
  text=None = same prompt minus the report block, robust parser); notebook built from it
- Colab T4 run (2026-10-08): 986 rows in 6 min, **0 parse failures**, all prompt fingerprints
  verified; transformers 5.18.0, torch 2.11.0+cu130
- **Broke:** transformers 5 `apply_chat_template` returns a `BatchEncoding` → fixed in the
  builder (`ids["input_ids"]`), regression test added; run `revision` came out null → Hub fallback
- 3 Traffic real-text answers in billions (the report's unit) → **scale check**: >10× outside
  the window's range = NaN format failure (decided with Aryan). Traffic M5 real MSE 18.3 → 1.905
- Issue #26: Qwen copies a report number in 5.9% of real answers vs 4.1% chance — closed
- Model 5 is weak (mostly copies recent values) but text changes its answer in 65–78% of windows
- P2 exit criterion met: all 5 models answer `predict(numbers, text)` and `predict(numbers, None)`
  (Model 5 for every exported window)
- **Next:** P3 — attack module (6 text conditions); Model 5 needs a second Colab run for them

### 2026-10-07 — P2.3: text models 3 and 4 (frozen backbone + BERT text head)
- Read MM-TSFlib's fusion (issue #3): late additive; **text term ≡ 0 at h=1** (`norm()` across
  the horizon) and a **co-shipped numeric prior** (`prior_y`) in the text branch → paper
  findings A and B (§7a). Implemented its fusion minus `norm()` and `prior_y`, full frozen BERT
- `text_encoder.py`: frozen `bert-base-uncased`, mean-pooled, on-disk cache (`/cache/`); BERT
  loaded only on a cache miss, `unload()` before every training unit. 0/1305 texts truncated
- Found: raw BERT vectors killed the head's ReLUs (0/96 alive) → per-dim z-score, train-only
- Critical check failed with joint training (Economy M3 text=None val 0.109–0.128 vs M1 0.101)
  → **frozen Model 1/2 backbone + trained text head**; text=None ≡ Model 1/2 (verified on all
  predictions). Both changes accepted. Val probes for both designs in §7a
- Runner: Models 3/4 in the same resumable runner, P2.2 protocol, predictions saved with
  `condition` = real / empty; `NeuralForecaster` refactored to hooks (Models 1/2 bit-identical)
- Results §7a: text helps on Economy; on Traffic it halves val error but hurts test (2020) → #23
- 133 tests passing (18 new in `test_late_fusion.py`)
- **Broke:** first full run killed by Claude Code under *system* memory pressure (15.7 GB box,
  4.7 GB free before training; the training process itself peaks ~0.54 GB, BERT never
  loaded). Resumed: 5 finished units skipped, 3 run
- **Next:** P2.4 — Qwen2.5-1.5B forecaster (Model 5) on Colab

### 2026-10-07 — P2.2: numbers-only baselines trained (val-tuned protocol)
- Persistence, DLinear, PatchTST trained on all 4 domains: lr grid {1e-4, 1e-3, 1e-2} on val,
  up to 200 epochs / patience 20, 3 seeds; repro column on MM-TSFlib's recipe. Results §7
- Found: DLinear under MM-TSFlib's recipe ≈ window-mean predictor (corr ≥ 0.9996 monthly) →
  tuned protocol, **binding on Models 3/4** (issue #19)
- Found: MM-TSFlib PatchTST clamps patch_len to pred_len → reads 2 of 8 inputs at h=1;
  `patching="window"` bypasses it (issue #20)
- Repro column far above published MSE — explained by different test periods/splits; closed as a limitation (issue #21)
- `train_baselines.py` made resumable: each config × domain × model unit saved to
  `results/p2_baselines/` on completion with a protocol fingerprint; finished units skipped;
  `--fresh` retrains all. 115 tests passing (2 new: simulated crash/resume, protocol mismatch)
- **Broke:** laptop shut down mid-session. The 2026-10-02 tuned run had in fact completed
  (MLflow: 72 finished runs, nothing after); its CSVs were verified complete, then reproduced
  by an independent re-run with the new runner (tuning identical, MSE ≤1.3e-7)
- **Next:** P2.3 — read MM-TSFlib fusion code (issue #3), then DLinear+BERT / PatchTST+BERT
  with the tuned protocol

### 2026-10-02 — P2.1 + P2.5: forecaster interface + dummy model
- `src/models/base.py`: `Forecaster` Protocol + `BaseForecaster` ABC; univariate raw-unit input,
  blank text → `None`, output `(horizon,)`, NaN on failure (§6a)
- `src/models/dummy.py`: linear extrapolation + keyword text shift with configurable
  sensitivity; reproduces the worked example (116 / 106 / 120); change and level cues
- Recorded for later phases: NaN pairing rule (P4, P6), DSS/Flip Rate scored at horizon 1
- 90 tests passing (53 new)
- **Broke:** nothing
- **Next:** P2.2 — DLinear and PatchTST wrappers on the selected domains

### 2026-09-28 — P1.5 checks done → **P1 complete**
- `check_timing.py`: 37 leaks with start-anchored alignment → loader default now `anchor="end"`;
  5 genuine leakers dropped (`KNOWN_LEAKERS`); 7 year-typos kept and documented. Final: **PASS**
- SocialGood: 41% of reports were off-topic influenza text → filtered (131 dropped, 13 real reports
  kept with the header stripped). Paper-worthy dataset finding
- `check_directional.py`: 58% overall (Economy 92%, Traffic 100%, SocialGood 23% → 56% after filter,
  Health_US 17%). Energy 100%, but 351/354 of its reports contain the forecast target
- Decided: two attack variants (change-flip, level-flip); TextGuard gate generalises to
  claim-vs-evidence; direction classifier must be Qwen-based
- 37 tests passing
- **Broke:** nothing unresolved. **Open:** SocialGood-vs-Energy swap (#14)
- **Next:** P2 — `src/models/base.py` and the `predict(numbers, text)` interface

### 2026-09-28 — P1.3–1.4: domains selected, splits built
- `inspect_domains.py`: window = first → last text; selected Economy, SocialGood, Traffic, Health_US (report)
- `loader.text_window()` shared by inspection and splits
- `splits.py`: 70/10/20 chronological split over the text window (matches MM-TSFlib ratios);
  rejects unsorted/duplicate input; leakage test passes on all 4 real domains
- **Broke:** nothing
- **Next:** P1.5 `check_directional.py` and `check_timing.py`

### 2026-09-28 — P1.1–1.2: data cloned, loader built
- Cloned Time-MMD → `data/Time-MMD`, MM-TSFlib → `external/MM-TSFlib` (both gitignored)
- Inspected real layout (README is wrong in places) — findings in §6
- `src/data/loader.py`: `load_domain(domain, source='report'|'search', lag=1)`; `fact` only;
  sorts by date; drops duplicate dates; missing text explicit (`text=None`, `has_text`)
- `tests/test_loader.py`: 12 passing, incl. a no-future-text leakage test
- **Broke:** empty Security report crashed `merge_asof` (datetime precision mismatch) — fixed
- **Finding:** `report` text covers only part of most series (Agriculture 10%, Energy 22%,
  Climate 23%, Economy 77% of rows) because reports start years after the numbers.
  Feeds directly into domain selection (P1.3)
- **Next:** P1.3 `inspect_domains.py` → pick 4–5 domains

### 2026-09-28 — P0 complete
- Created `src/` skeleton: `data/`, `models/`, `attacks/`, `textguard/` packages, plus
  `metrics.py` and `run_experiments.py` placeholders; `notebooks/`, `results/`, `tests/`,
  `paper/`, `app/` folders
- venv + pinned deps in place (torch 2.14.0+cpu, mlflow 3.16.1)
- **Broke:** Colab VS Code extension unreliable → use colab.research.google.com directly (issue #5)
- **Next:** P1 — clone Time-MMD + MM-TSFlib, write `loader.py`

### 2026-09-28 — Plan restructured for solo build
- Rewrote PHASES.md as a pure build plan: P0–P8, every phase produces code
- Literature work moved into P8 (Related Work), where it's actually needed;
  competing-work reference kept in CLAUDE.md §4a
- Model 6 and FNSPID confirmed optional (P7)
- Repo: `https://github.com/aryanwaingankar4/textguard`
  Local: `C:\Users\Aryan\textguard`
- **Broke:** nothing — no code yet
- **Next:** P0 — venv, dependencies, Colab check, folder skeleton, first push

---

## 12. Standing reminders

- **Commit and push at every phase boundary**, with this file updated in the same commit
- Never tune on the test split — validation split only
- Save every raw prediction to CSV, not just aggregate metrics
- Keep `predict(numbers, text)` uniform across all models — it's what makes TextGuard model-agnostic
- Write paper sections as you go; don't leave all writing to P8
- Understand every module well enough to explain it in the viva, alone
- Post to arXiv as soon as results are solid — it timestamps the work
- DSS ≈ 0 on late-fusion models is an expected finding, not a failure (see PHASES.md P5)
