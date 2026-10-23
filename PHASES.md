# PHASES.md — TextGuard

**Phase-gated build plan.** No deadlines. Every phase produces working code.
Finish a phase → tick its exit criterion → update `PROJECT_STATUS.md` → commit → push.

**Solo project.** All work by Aryan.

---

## Build order

```
P0 Setup ─► P1 Data ─► P2 Models ─► P3 Attacks ─► P4 Metrics
                                                      │
                                                      ▼
                                               P5 Experiments
                                                      │
                                                      ▼
                                               P6 TextGuard
                                                      │
                                                      ▼
              R1 ─► R2 ─► R3 ─► R4 ─► R7   (research extensions; R5, R6 = future work)
                                                      │
                                    ┌─────────────────┴─────────────┐
                                    ▼                               ▼
                             P7 Auditor + Demo                 P8 Paper
                                (optional)                     (required)
```

| Phase | Builds | Required? |
|---|---|---|
| P0 | Environment, repo, folder skeleton | ✅ |
| P1 | Data pipeline — loader, splits, domain selection | ✅ |
| P2 | Model wrappers — 5 models behind one interface | ✅ |
| P3 | Attack module — 6 text conditions incl. fake text | ✅ |
| P4 | Metrics module — MSE, Text Usage Gap, DSS, Flip Rate | ✅ |
| P5 | Experiment runner — run everything, save raw predictions | ✅ |
| P6 | TextGuard — trend detector, gate, defense wrapper | ✅ **Core** |
| R1–R7 | Research extensions — see "Research extensions" below | ✅ (decided 2026-10-09) |
| P7 | Auditor, audit report, Streamlit demo, drift monitor | ⬜ Optional |
| P8 | Paper and release | ✅ |

**P0–P6 + P8 = complete paper.** R1–R4 and R7 run before P8 (decided 2026-10-09; R5 and R6 moved
to future work, not run — Aryan 2026-10-19). **Order after R4: R7 (done 2026-10-24) → P7.2 (demo, built 2026-10-24) → P8.** P7 is
upside; P7.4 (drift monitor) stays future work with R6.

---

# Phase 0 — Setup

**Builds:** a working environment and an empty but complete repo skeleton.

- [ ] Create `C:\Users\Aryan\textguard`, open in VS Code
- [ ] `git init`, add `.gitignore`, connect to GitHub, first push
- [ ] Python venv + install: torch, transformers, pandas, numpy, scikit-learn,
      matplotlib, mlflow, pytest
- [ ] `pip freeze > requirements.txt`
- [ ] Install the official **Google Colab** extension for VS Code (verified badge)
- [ ] Verify a Colab T4 notebook can `git clone` the repo and `import torch`
- [ ] Create the folder skeleton:

```
src/
  data/          # loaders, splits
  models/        # model wrappers
  attacks/       # text condition generators
  textguard/     # defense
  metrics.py
  run_experiments.py
notebooks/       # Colab GPU runs
results/         # raw predictions + summaries
tests/
paper/
app/
```

- [ ] `mlflow ui` runs locally

**✅ Exit:** env works, Colab reachable, skeleton pushed.
**➡️ Commit & push.**

---

# Phase 1 — Data Pipeline

**Builds:** `src/data/` — everything that turns raw Time-MMD into clean train/val/test splits.

### 1.1 Get the data
- [ ] Clone `AdityaLab/Time-MMD` and MM-TSFlib into the workspace
- [ ] Download the Time-MMD data

### 1.2 Loader
- [ ] `loader.py` — load a domain's numerical series + aligned text into a DataFrame
- [ ] Handle missing text (some periods have no report) explicitly

### 1.3 Domain selection
- [ ] `inspect_domains.py` — script that reports per domain: number of samples,
      % of rows with text, average text length
- [ ] Pick **4–5 text-rich domains**, record the choice in PROJECT_STATUS

### 1.4 Splits
- [ ] `splits.py` — **chronological** train/val/test. Never random
- [ ] Test asserting no test timestamp precedes any train timestamp

### 1.5 Two data checks (these decide whether the attack is even possible)
- [ ] `check_directional.py` — sample 50 random reports, print them, and record what % make
      a clear up/down claim. **If <30%, the flipped-text attack has nothing to grab onto** —
      switch domains or retarget the attack to whatever the text does assert
- [ ] `check_timing.py` — verify each report's timestamp precedes its forecast window.
      Catches the leakage case where a report describes what already happened

**✅ Exit:** any domain loads into clean chronological splits; directional-claim % recorded;
timing verified.
**➡️ Commit & push.**

---

# Phase 2 — Model Wrappers

**Builds:** `src/models/` — all models behind one uniform interface.

### 2.1 The interface
- [ ] `base.py` — every model exposes `predict(numbers, text) -> prediction`.
      Passing `text=None` must give a valid numbers-only prediction.
      **This interface is what makes TextGuard model-agnostic.** Get it right here

### 2.2 Numbers-only baselines
- [ ] `dlinear.py` — wrap DLinear (Model 1)
- [ ] `patchtst.py` — wrap PatchTST (Model 2)
- [ ] Train on selected domains, sanity-check the MSE is reasonable

### 2.3 Late-fusion models
- [ ] `dlinear_bert.py` — DLinear + BERT (Model 3)
- [ ] `patchtst_bert.py` — PatchTST + BERT (Model 4)
- [ ] **Read the MM-TSFlib fusion code and document how text is actually combined.**
      We assume late fusion — confirm it and write the finding into PROJECT_STATUS

### 2.4 LLM forecaster (early fusion)
- [ ] `qwen_forecaster.py` — Qwen2.5-1.5B-Instruct (Model 5)
- [ ] Prompt template: numbers + text → next value
- [ ] **Robust output parser.** LLMs return messy text; handle parse failures explicitly
      rather than silently dropping samples
- [ ] Runs on Colab T4

### 2.5 Dummy model (for testing TextGuard later)
- [ ] `dummy.py` — a trivial model with known behaviour, used to prove TextGuard works
      on something outside our five

**✅ Exit:** all 5 models respond to `predict(numbers, text)` and `predict(numbers, None)`.
**➡️ Commit & push.**

---

# Phase 3 — Attack Module

**Builds:** `src/attacks/` — generators for all six text conditions.

### 3.1 Simple conditions
- [ ] `empty.py` — remove text
- [ ] `shuffled.py` — text from other dates, same domain
- [ ] `cross_domain.py` — text from a different domain
- [ ] `time_shifted.py` — text from the wrong period
- [ ] Unit tests: shuffled text genuinely comes from other dates; the time-shift offset is correct

### 3.2 Fake text (the core attack)
- [ ] `flipped.py` — **deterministic rules** (not Qwen — see CLAUDE.md §7) invert each
      report's directional polarity ("rose" → "fell"), keeping everything else identical and
      the numbers coherent; record `claimed_direction`
- [ ] `exaggerated.py` — text agrees with the trend but overstates the magnitude.
      This is TextGuard's known weak spot; we test it honestly

### 3.3 Validate the fake text
- [ ] `validate_attacks.py` — automatic checks: length similarity, direction actually flipped
- [ ] Manually review ~100 flipped samples. Record the pass rate in PROJECT_STATUS
- [ ] Save all generated text to `texts/`, one file per condition

**✅ Exit:** all six conditions generate from any domain; flipped text validated.
**➡️ Commit & push.**

---

# Phase 4 — Metrics Module

**Builds:** `src/metrics.py` — every number the paper reports.

- [ ] `mse`, `mae`
- [ ] `text_usage_gap` = MSE(empty) − MSE(real). Near zero ⇒ the model ignores text
- [ ] `dss` = `d × (ŷ_fake − ŷ_real) / σ`, where `d` is the direction the fake text claims
      and σ is the series standard deviation. Positive ⇒ pulled toward the lie
- [ ] `flip_rate` = % of forecasts whose up/down direction reverses under fake text
- [ ] `false_block_rate` = % of **genuine** texts that TextGuard gates below 0.5
- [ ] Unit tests using the worked example below — every metric must reproduce those numbers
- [ ] Bootstrap confidence intervals + paired Wilcoxon signed-rank test

**✅ Exit:** all metrics implemented and unit-tested against the worked example.
**➡️ Commit & push.**

---

# Phase 5 — Experiments

**Builds:** `src/run_experiments.py` and the results in `results/`.

### 5.1 Runner
- [ ] Loop over model × domain × condition × seed (3 seeds)
- [ ] **Save every raw prediction** — one CSV row per
      (model, domain, condition, seed, sample_id, y_true, y_pred).
      New metrics and plots must be derivable without re-running anything
- [ ] Log each run to MLflow
- [ ] Checkpoint/resume, so a crashed run doesn't restart from zero

### 5.2 Screening — which models actually use text
- [ ] Run models 1–5 on `real`, `empty`, `shuffled`, `cross_domain`, `time_shifted`
- [ ] Compute Text Usage Gap per model
- [ ] Record in PROJECT_STATUS: **which models genuinely consume text**.
      Only those can be attacked *through* text — they are the attack targets

### 5.3 Attack
- [ ] Run the attack targets on `flipped` and `exaggerated`
- [ ] Compute DSS and Flip Rate per model, per fusion type, per domain
- [ ] Significance tests

**✅ Exit:** we can state which models use text and which fusion type is most vulnerable,
with numbers and statistics.
**➡️ Commit & push.**

> **Expected outcome:** the late-fusion models may barely move across conditions, i.e. they
> ignore text. If so, Model 5 (LLM, early fusion) becomes the primary attack target and the
> thesis sharpens to the **use-vs-vulnerability trade-off**: vulnerability is the price of
> actually reading text. That is a better story, not a setback. Do not treat DSS ≈ 0 on the
> fusion models as a failed experiment.

---

# Phase 6 — TextGuard · **core contribution**

**Builds:** `src/textguard/` — the defense.

### 6.1 Components
- [ ] `trend_detector.py` — fit a slope over the recent window →
      direction ∈ {−1, 0, +1} and strength = `min(|slope|/σ, 1)`
- [ ] `text_direction.py` — Qwen classifies the text's claimed direction ∈ {−1, 0, +1}
      with a confidence in [0, 1]
- [ ] `gate.py` —
      agree, or either side neutral → `g = 1`;
      disagree → `g = 1 − (confidence × strength)`
- [ ] `textguard.py` — wrapper taking **any** object with `predict(numbers, text)`:

```python
A = model.predict(numbers, None)
B = model.predict(numbers, text)
final = A + g * (B - A)
warn  = g < 0.5
```

### 6.2 Prove it's model-agnostic
- [ ] Run TextGuard on `dummy.py` from P2 — it must work on a model it has never seen

### 6.3 Tune on the **validation split only**
- [ ] Trend window length
- [ ] Gate warning threshold (default 0.5)

### 6.4 Baseline defenses to compare against
- [ ] `never_trust` — g = 0 always
- [ ] `always_trust` — g = 1 always
- [ ] `clip` — cap `|B − A|` at a fixed multiple of σ
- [ ] **CFA** (Lee et al. 2026, code `seunghan96/cfa`) — trained constrained fusion.
      Strongest baseline. If it won't run, say so explicitly in the paper

### 6.5 Evaluate
- [ ] All attack targets, with and without TextGuard
- [ ] DSS, Flip Rate, MSE on real text, False Block Rate
- [ ] Significance tests vs each baseline defense

**✅ Exit:** TextGuard beats the baselines on the manipulation-vs-false-block trade-off,
and works on a model outside our five.
**➡️ Commit & push.**

> 🎉 **You now have a complete paper.** Everything in P7 is upside.

---

# Research extensions — R1–R7 (before P8, decided 2026-10-09)

P0–P6 answer the core questions; R1–R7 close the gaps P5/P6 left open (PROJECT_STATUS §9:
fusion confound, one small LLM, a non-adaptive attacker, TextGuard v1's known failure modes,
rule-written fakes only, no drift monitor, reproducibility). Done in this order; each one is
its own phase: pre-register → commit → build → run → results → paper section → commit & push.

### Rules for every extension (also in PROJECT_STATUS §1a and CLAUDE.md §14)

- **Pre-registered and committed alone** (its own commit, pushed) before any of its test
  results exist. Changes after that are dated amendments, reported as deviations.
- **New models and new attacks may be evaluated on the existing test set.**
  **Already-reported methods (Models 1–5, TextGuard v1 at its frozen P6 settings) are never
  changed** — they may be re-run on new inputs, never re-tuned or redefined.
- **R4's held-out domains are chosen and frozen before any model output exists on them.**
  Their data checks (directional %, timing) are allowed before freezing.
- **Each extension's results section is written into `paper/` as it finishes** — not left to P8.
- **Null results are reported, not dropped.**
- **From R2 on, every pre-registration defines a smallest effect size of interest (SESOI)** for
  each decision rule — significance alone is not enough (R1 lesson).

### R1 — LLM vs fusion: late-fusion heads on Qwen embeddings (+ FinBERT)
Model 5 is the only early-fusion model **and** the only LLM (PROJECT_STATUS §9 P5.7), so P5
cannot say which one makes it vulnerable.
- [x] Pre-register (PROJECT_STATUS §10d), commit alone
- [x] Mechanism check on **val** first: flip-vs-swap ratio (P5's embedding similarity) for
      BERT, FinBERT and Qwen embeddings — how direction-aware each encoder is, before training
- [x] Qwen2.5-1.5B-Instruct embeddings (P5's pinned revision) on Colab; FinBERT
      (`ProsusAI/finbert`) on CPU
- [x] Models 3/4 with the encoder swapped, everything else identical (frozen M1/M2 backbone,
      two-stage, train-only standardisation, P2.2 protocol, 3 seeds); all 7 conditions
- [x] Analysis against the R1 family; results section in `paper/`
- **✅ R1 done (2026-10-09):** pre-registered reading R1-a; in-prompt reading carries ~98% of
  the manipulation; a direction-aware encoder (FinBERT) makes late heads attackable

### R2 — generality: other LLM forecasters and prompt wording
- [x] Qwen2.5-7B-Instruct (4-bit; on Kaggle, §10e Amendment 1) and Phi-3.5-mini-instruct as
      early-fusion forecasters
- [x] 2–3 prompt-wording variants for Model 5 (Model 5's own prompt stays as reported) —
      report-first, reworded, and a skeptical "be careful" prompt
- [x] Same conditions and metrics as P5; pre-registered families (44 / 44 / 58 / 54) with SESOI
- **✅ R2 done (2026-10-11):** Qwen2.5-7B fooled (+0.206σ); Phi-3.5 bounded because it ignores
  report content; M5's conclusions prompt-robust; skeptical prompt does not help (TextGuard
  0.40σ lower)

### R3 — adaptive attacks on TextGuard v1
Attackers who know the defence. TextGuard v1 is run exactly as frozen in P6.
- [x] Pre-register (PROJECT_STATUS §10f), commit alone
- [x] Trend-consistent lies (claim direction agrees with the lookback trend, so the gate passes it)
- [x] Forward-looking lies ("will reverse next period" — a claim about the future, not the past)
- [x] Magnitude lies (right direction, wrong size)
- [x] DSS / Flip Rate / damage undefended vs TextGuard v1 vs clip; pre-registered family
- **✅ R3 done (2026-10-11):** one appended forward-looking sentence (A2) evades TextGuard v1 on M5
  (+0.456σ) and Qwen2.5-7B (+0.403σ); no defence stops it (clip leaves +0.12σ); A1 on M5 small and
  not robust (evades TG on M3-F); A3 pushes nothing. Amendment 1: text-validity exclusions

### R4 — TextGuard v2
- [x] **Judge evaluates every claim in a report, including forward-looking ones** (R3: A2's
      appended forward claim was never checked — v1 asks only about the main past claim)
- [x] Level/change-aware judge; longer history for year-over-year claims; a magnitude cap
      (borrowed from clip, which beat v1 on exaggeration)
- [x] **Developed on VAL only.** Held-out domains chosen and frozen before any model output
      existed on them — **Energy and FNSPID** (Health_AFR, Climate dropped, PROJECT_STATUS §6)
- [x] Tested **once** on the held-out domains: v2 and v1 both, and v2 also against R3's
      adaptive attacks (pre-registered §10g)
- [x] ~~v2 also evaluated on R1's FinBERT heads~~ — dropped before the pre-registration (2026-10-12)
- **✅ R4 done (2026-10-19):** v2 generalises its A2 protection on M5 (pooled +0.180 → +0.021σ,
  lower than v1) but not its FBR improvement (18.4% vs v1 7.0%, val ordering reversed — the judge
  reads directions into neutral FNSPID headlines); nothing transfers to Qwen2.5-7B. Pre-registered
  reading: "v2 buys protection at a cost (FBR)". PROJECT_STATUS §9 "R4",
  `paper/sections/r4_textguard_v2.tex`.

### R5 — realistic attacks — ❌ FUTURE WORK, not run (Aryan 2026-10-19; goes in the paper's future work, P8)
- [ ] Fluent fakes written by an LLM from a **different family** than Qwen
- [ ] Small human plausibility study (real vs rule-written vs LLM-written fakes)

### R6 — drift monitor for sustained fake-report campaigns (RQ4; replaces P7.4) — ❌ FUTURE WORK, not run (Aryan 2026-10-19)
- [ ] Embedding centroid + per-batch distance, threshold from historical variation (val)
- [ ] Detection delay and false-alarm rate on streams seeded with fake reports

### R7 — reproducibility — ✅ DONE 2026-10-24 (PROJECT_STATUS §9 "R7")
- [x] One-command reproduce script; pinned environment (`reproduce.py`, `requirements-lock.txt`)
- [x] Data card, model card, ethics statement
- [x] Clean-room test: fresh clone + fresh venv, 47 / 47 steps, every committed file byte-identical
      (`docs/REPRODUCE.md` §5); toolkit tested on Kaggle with the live judge

**✅ Exit (each R):** pre-registration commit precedes its results; results in PROJECT_STATUS §9
and in `paper/`; nulls reported. **➡️ Commit & push after each.**

---

# Phase 7 — Auditor, Demo & Extras (optional)

Independent items. Do as many as you want, in this order. Commit & push after each.

### 7.1 Auditor + audit report *(answers "what does a model provider get?")*
- [ ] `auditor.py` — plug in any model, run all conditions, return metrics
- [ ] Report generator producing:
      1. Health check — traffic-light table
      2. Fault breakdown by domain, horizon, attack type
      3. Worst 10 failure cases — real vs fake text, prediction before/after, true value
      4. Diagnosis → recommended fix per fault
      5. With-TextGuard before/after comparison
      6. Drift monitor setup
- [ ] Output as HTML/PDF + CSV (raw predictions) + JSON (for CI gating)

### 7.2 Streamlit demo *(what your panel sees)* — ✅ BUILT 2026-10-24 (local; `app/README.md`)
- [x] Domain selector; real / fake text. **Custom text dropped** (Aryan 2026-10-24: precomputed only)
- [x] Side-by-side: undefended forecast vs TextGuard forecast (off / v1 / v2, per-sentence trust)
- [x] Gate value + warning display
- [x] Results tab with DSS / Flip Rate charts (DSS per model + the paper figures)
- [x] Qwen results **precomputed** — the free tier has ~2.7 GB RAM and no GPU
- [ ] Deploy to Streamlit Community Cloud — **in P8**, once the repo is public (needs the Time-MMD
      licence decision: the demo's Time-MMD texts are not in git)
- Free apps sleep after ~12 h idle and wake on click (~1 min). Open it before any demo

### 7.3 Model 6 — cross-attention *(completes the 3-fusion-type claim)*
- [ ] Look for public TTCA-style code first
- [ ] Otherwise build it: PatchTST numbers + BERT text + a cross-attention layer
- [ ] Re-run P5 and P6 for this model only

### 7.4 Drift monitor
- [ ] BERT-embed incoming text → reference centroid from normal history
- [ ] Per-batch cosine distance + threshold from historical variation
- [ ] Test it fires on a stream seeded with fake reports

### 7.5 CI hardening
- [ ] GitHub Actions: pytest + a mini-attack check that fails the build if DSS regresses

### 7.6 FNSPID second dataset *(lowest priority)*
- [ ] Download a **subset only** — 5–10 stocks, not all 29.7M rows
- [ ] Align news with prices chronologically
- [ ] Top 2 models × core conditions ± TextGuard
- [ ] **Frame it right:** absolute stock accuracy will be poor. The finding is that the
      *vulnerability pattern and TextGuard's protection transfer* to a different text style
- [ ] CC BY-NC-4.0 attribution in the README

---

# Phase 8 — Paper & Release

**Builds:** the paper and the public repo.

### 8.1 Related Work
- [ ] Write it now, from CLAUDE.md §4a (competing work) plus the Time-MMD, PA-RNet,
      text-collapse and TimesX papers
- [ ] One explicit **"how we differ"** sentence per competing paper
- [ ] Re-run the arXiv search before finalizing — this field publishes fast

### 8.2 Paper
- [ ] Abstract, Intro, Related Work, Method, Experiments, Results, Discussion,
      Limitations, Conclusion
- [ ] Figures: architecture diagram, DSS/Flip Rate charts, fusion comparison, gate distribution
- [ ] Error analysis — 6–8 concrete failure cases with real text shown
- [ ] Limitations — be explicit about the exaggeration attack and False Block Rate

### 8.3 Release
- [ ] README: install, 5-minute example, how to plug in your own model, results, license
- [ ] MIT license for the code
- [ ] Publish the counterfactual/fake-text benchmark, clearly labelled as research-only
- [ ] **Linux CI reproduction check (R7 open point a):** a GitHub Actions job on `ubuntu-latest`
      that downloads the data, runs `reproduce.py` and requires `REPRODUCED` (9 / 9 GPU inputs byte
      for byte; tables may show `identical_up_to_line_endings`). It tests R7's CRLF fix to the
      hash-checked exports, which is not yet tested off Windows (`docs/REPRODUCE.md` §5). Needs ~1 h
      CPU; check the Actions time limits
- [ ] **Time-MMD licence check before the repo goes public (R7 open point b):** git history
      (`801c3f6`…`d93f98a`) holds Time-MMD Energy report texts, and Time-MMD states no licence
      (`DATA_CARD.md` §1a). Decide: ask the authors, accept and document, or rewrite history
      (which changes cited commit hashes)
- [ ] Tag a release
- [ ] Post to arXiv — **as soon as results are solid.** It timestamps the work
- [ ] Submit to the target venue

### 8.4 Course deliverables
- [ ] PPT updated with real numbers
- [ ] Project report
- [ ] Viva prep — walk through the worked example cold

**➡️ Final commit & push.**

---

## Worked example — the metrics unit tests must reproduce this

Past demand: `100, 104, 108, 112` (rising). σ = 10. True next value = **117**.
Fake text: *"Demand will fall sharply."*

| Step | Value |
|---|---|
| A (no text) | 116 |
| B (with fake text) | 106 ← fooled |
| Trend direction | up, strength 1.0 |
| Text direction | down, confidence 0.9 |
| Gate `g = 1 − 0.9×1.0` | **0.1** |
| Final `= 116 + 0.1×(106−116)` | **115** ⚠ warning |

Error without TextGuard: 11 → with TextGuard: 2.
DSS: `−1 × (106−120)/10 = +1.4` → `−1 × (115−120)/10 = +0.5`.

With **real** text ("demand rising"), directions agree → `g = 1` → text fully used.
Good text is never blocked.

---

## Commit convention

Prefix every commit with its phase:

```
P0: setup repo and environment
P1: chronological splits + directional claim check
P2: model wrappers behind uniform predict interface
P3: flipped text generator
P4: DSS and flip rate metrics
P5: screening results for models 1-5
P6: textguard gate implementation
R1: pre-register LLM-vs-fusion heads
P7: streamlit demo
P8: paper draft
```

Commit as you go. **Always** commit at a phase boundary, with `PROJECT_STATUS.md`
updated in the same commit.
