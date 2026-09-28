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
| P7 | Auditor, audit report, Streamlit demo, drift monitor | ⬜ Optional |
| P8 | Paper and release | ✅ |

**P0–P6 + P8 = complete paper.** P7 is upside.

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
- [ ] `flipped.py` — Qwen rewrites each report inverting directional polarity
      ("rose" → "fell"), keeping everything else identical
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

### 7.2 Streamlit demo *(what your panel sees)*
- [ ] Domain selector; real / fake / custom text input
- [ ] Side-by-side: undefended forecast vs TextGuard forecast
- [ ] Gate value + warning display
- [ ] Results tab with DSS / Flip Rate charts
- [ ] Qwen results **precomputed** — the free tier has ~2.7 GB RAM and no GPU
- [ ] Deploy to Streamlit Community Cloud
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
P7: streamlit demo
P8: paper draft
```

Commit as you go. **Always** commit at a phase boundary, with `PROJECT_STATUS.md`
updated in the same commit.
