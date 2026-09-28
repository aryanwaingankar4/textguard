# PHASES.md — TextGuard

**Phase-gated, not date-gated.** There are no deadlines here. Finish a phase, tick its exit
criterion, commit, push, move on. Work at whatever pace fits around college.

**Solo project.** Aryan does all of it.

**Rule: at the end of every phase — update `PROJECT_STATUS.md`, commit, push.** No exceptions.

---

## Phase order and dependencies

```
P0 Setup ──► P1 Foundation ──► P2 Screening ──► P3 Attack ──► P4 Defense ──► P6 Paper
                                                                   │
                                                                   └──► P5 Extras (optional)
```

Phase 5 is optional. Phase 6 can start once Phase 4 is done.

| Phase | Name | Required? |
|---|---|---|
| P0 | Setup & repo | ✅ Required |
| P1 | Foundation: data, baselines, competing literature | ✅ Required |
| P2 | Screening: which models use text (RQ0) | ✅ Required |
| P3 | Attack: misleading text + DSS (RQ1, RQ2) | ✅ **Core contribution** |
| P4 | Defense: TextGuard (RQ3) | ✅ **Core contribution** |
| P5 | Extras: Model 6, FNSPID, drift monitor, MLOps, demo | ⬜ Optional |
| P6 | Paper & release | ✅ Required |

**If you only ever finish P0–P4 and P6, you still have a complete paper.**
Everything in P5 is upside.

---

# Phase 0 — Setup & Repo

### 0.1 Admin
- [ ] Guide approval: topic counts as multimodal ✓, Cybersecurity category ✓, paper expected?
- [ ] Clarify Devansh's role on the registration (co-author or not) — decide deliberately

### 0.2 Local folder & repo
- [ ] Create project folder (see README of this repo / setup notes)
- [ ] `git init`, add `.gitignore`, create GitHub repo, push
- [ ] Add `CLAUDE.md`, `PHASES.md`, `PROJECT_STATUS.md` to the repo root

### 0.3 Environment
- [ ] Python 3.10+ venv
- [ ] Install: torch, transformers, pandas, numpy, scikit-learn, matplotlib, mlflow, pytest
- [ ] Install the official **Google Colab** extension for VS Code (verified badge — not third-party)
- [ ] Verify: a Colab T4 notebook can `git clone` your repo and `import torch`

**✅ Exit:** repo is live on GitHub, env works, Colab reachable.
**➡️ Commit & push.**

---

# Phase 1 — Foundation

## 1A — Competing literature (do this BEFORE writing experiment code)

These four decide whether the project survives as designed. Read them in full.

- [ ] **Sridhar et al. (2026)** *Semantics or Structure? Auditing Text Sensitivity in
      Multimodal TSF* — scooped our original RQ1. Extract: exact conditions, models covered,
      scope of their released toolkit. **Decide: replicate, or build on theirs?**
- [ ] **Lee et al. (2026)** *CFA / Constrained Fusion* — code `seunghan96/cfa`.
      **Decide: can we run it as a defense baseline?**
- [ ] **The Corroboration Illusion** (arXiv 2609.22246) — confirm it covers RAG *event*
      forecasting, NOT numerical multimodal TSF
- [ ] **Plug-In Adversarial Detection for Time-Series LLMs** (arXiv 2512.12154) —
      ⚠️ **confirm their attack is numeric, not textual.** If textual, stop and reposition
- [ ] Liu & Jiang (2024) arXiv 2412.08099 — confirm it attacks numbers, not text
- [ ] Zhu et al. (2025) PA-RNet — quote the *zero-mean perturbation* assumption exactly

**Then:**
- [ ] Re-run arXiv search: "adversarial text multimodal time series forecasting",
      "misleading text forecasting attack", "semantic perturbation time series text"
- [ ] Write the novelty claim as **one defensible paragraph** in PROJECT_STATUS
- [ ] Tier 1 papers (parallel, lighter): Time-MMD, text collapse, TimesX, Context is Key,
      PatchTST, DLinear
- [ ] Shared notes file, 3 lines per paper: *what they did / what they found / what they did NOT do*
- [ ] Draft Related Work

## 1B — Data

- [ ] Clone `AdityaLab/Time-MMD` and MM-TSFlib
- [ ] Count text volume per domain; pick **4–5 text-rich domains**
- [ ] Implement chronological train/val/test splits
- [ ] ⚠️ **CHECK A — directional claims.** Read ~50 random reports. Count how many make a
      clear up/down claim. Record the % in PROJECT_STATUS.
      **If <30% → stop and replan** (switch domains, or retarget the attack)
- [ ] ⚠️ **CHECK B — timing.** Verify each report was available *before* its forecast period.
      Document how Time-MMD aligns dates

## 1C — Baselines

- [ ] Get Models 1–4 running via MM-TSFlib
- [ ] Reproduce published baseline numbers (numbers-only vs +text)
- [ ] **Read the MM-TSFlib fusion code.** We *assume* late fusion — confirm it
- [ ] Set up MLflow logging

**✅ Exit:** all 4 Tier-0 papers read, novelty claim restated in one paragraph, directional-claim
% recorded, timing verified, fusion mechanism documented, baselines reproduce.
**➡️ Commit & push.**

> 🛑 **Hard gate:** if a Tier-0 paper already does semantic directional text attacks on
> numerical multimodal forecasters, stop and replan with your guide before Phase 2.

---

# Phase 2 — Screening (RQ0)

> **This is replication, not a contribution.** It reproduces Sridhar et al. (2026).
> Purpose: find which models actually consume text, since only those can be attacked
> *through* text. Cite them. Keep this phase lean — it's a gate, not the paper.

### 2.1 Condition generators
- [ ] Implement `empty`, `shuffled`, `time_shifted`
- [ ] (`cross_domain` optional — Sridhar covers it; cite instead if short on effort)
- [ ] Unit tests: shuffled text genuinely from other dates; time_shifted offset correct
- [ ] Save generated text to `texts/`

### 2.2 Qwen forecaster (Model 5)
- [ ] Prompt template: numbers + text → forecast
- [ ] Output parser — LLMs return messy text; handle parse failures explicitly
- [ ] Run on Colab T4, save raw predictions to CSV

### 2.3 Run the screen
- [ ] Models 1–5 × conditions × domains × 3 seeds
- [ ] **Save every raw prediction** to `results/`
- [ ] Compute Text Usage Gap per model
- [ ] Record in PROJECT_STATUS: **which models passed the screen** → these are Phase 3's targets

**✅ Exit:** a named list of attack targets, backed by numbers.
**➡️ Commit & push.**

> **Expected per Sridhar:** fusion models may move <0.5% across conditions, i.e. they ignore
> text. If so, Model 5 (LLM) becomes the primary attack target and the thesis shifts to the
> **use-vs-vulnerability trade-off**. That's a sharper story, not a setback. See CLAUDE.md §4a.

---

# Phase 3 — Attack (RQ1, RQ2) · **core contribution**

- [ ] **Re-run the competing-work arXiv search before starting.** This field publishes weekly

### 3.1 Generate misleading text
- [ ] Qwen prompt: invert directional polarity, keep everything else identical
- [ ] Generate `flipped` for all test samples
- [ ] Generate `exaggerated` (agrees with trend, overstates magnitude)

### 3.2 Validate (do NOT skip — reviewers will ask)
- [ ] Manually check ~100 rewrites: only the direction changed?
- [ ] Automatic length-similarity check
- [ ] Automatic direction-classifier check
- [ ] Record agreement rates in PROJECT_STATUS

### 3.3 Attack experiments
- [ ] Run attack targets on `flipped` + `exaggerated`
- [ ] Implement `metrics.py`: DSS, Flip Rate (+ MSE/MAE)
- [ ] Results: DSS + Flip Rate per model, per fusion type, per domain
- [ ] Wilcoxon tests

### 3.4 Writing
- [ ] Draft Method section (attack + metrics)

**✅ Exit:** we can state which fusion type is most vulnerable, with statistics.
**➡️ Commit & push.**

---

# Phase 4 — Defense (RQ3) · **core contribution**

### 4.1 Build TextGuard
- [ ] `trend_detector.py` — slope fit, direction, normalized strength
- [ ] `text_direction.py` — Qwen classifier → direction + confidence
- [ ] `gate.py` — consistency check → g
- [ ] `textguard.py` — wrapper taking any object with `predict(numbers, text)`
- [ ] `auditor.py` — runs all conditions on any model, returns metrics

**Model-agnostic from day one.** Test on a dummy model to prove it works outside our 5.

### 4.2 Tune (validation split ONLY)
- [ ] Trend window length
- [ ] Gate warning threshold (default 0.5)

### 4.3 Evaluate
- [ ] All attack targets, with and without TextGuard
- [ ] vs `never_trust` (g=0), `always_trust` (g=1), `clip`
- [ ] vs **CFA** (`seunghan96/cfa`) — strongest baseline, the one reviewers ask about.
      If it won't run, discuss in Related Work and justify the omission explicitly
- [ ] False Block Rate on genuine text
- [ ] Wilcoxon signed-rank tests
- [ ] **State positioning precisely:** training-free, inference-time, black-box.
      NOT "first plug-in fusion control" — CFA exists

### 4.4 Writing
- [ ] Draft TextGuard section + first results tables

**✅ Exit:** TextGuard beats the baseline defenses on the manipulation-vs-false-block
trade-off, with significance tests.
**➡️ Commit & push.**

> 🎉 **At this point you have a complete paper.** Everything below is upside.

---

# Phase 5 — Extras (optional, pick in this order)

Each item is independent. Do as many as you have appetite for. Commit & push after each.

### 5.1 Streamlit demo *(highest value — this is what your panel sees)*
- [ ] Domain selector, real/fake/custom text input
- [ ] Side-by-side: undefended forecast vs TextGuard forecast
- [ ] Gate value + warning display
- [ ] Results tab with DSS / Flip Rate charts
- [ ] Qwen results **precomputed** (free tier ≈2.7 GB RAM, no GPU)
- [ ] Deploy to Streamlit Community Cloud
- Note: free apps sleep after ~12 h idle and wake on click (~1 min). Open it before demos.

### 5.2 Audit Report generator *(answers your teacher's question directly)*
- [ ] HTML/PDF report with the 6 sections from CLAUDE.md §11
- [ ] CSV export (raw predictions) + JSON summary

### 5.3 Model 6 — cross-attention *(completes the 3-fusion-type claim)*
- [ ] Search for public TTCA-style code first
- [ ] If none: PatchTST numbers + BERT text + a cross-attention layer, built yourself
- [ ] Re-run Phases 2–4 for this model only

### 5.4 Drift monitor (RQ4)
- [ ] BERT embeddings → reference centroid from normal history
- [ ] Per-batch cosine distance + threshold from historical variation
- [ ] Test: does it fire on a stream seeded with fake reports?

### 5.5 MLOps hardening
- [ ] GitHub Actions: pytest + mini-attack check that fails the build if DSS regresses
- [ ] DVC for data versioning

### 5.6 FNSPID validation *(lowest priority)*
- [ ] Download a **subset only** (5–10 stocks), not the full 29.7M rows
- [ ] ⚠️ First check AutoRedTrader (2605.09185) + the financial-headline attack paper —
      they may pre-empt novelty here
- [ ] Align news with prices chronologically
- [ ] Top 2 models × core conditions ± TextGuard
- [ ] **Frame it right:** absolute stock accuracy will be poor. The finding is that the
      *vulnerability pattern and TextGuard's protection* transfer to a different text style
- [ ] Add CC BY-NC-4.0 attribution to README

---

# Phase 6 — Paper & Release

### 6.1 Paper
- [ ] Assemble: Abstract, Intro, Related Work, Method, Experiments, Results, Discussion,
      Limitations, Conclusion
- [ ] Figures: architecture diagram, DSS/Flip Rate charts, fusion comparison, gate distribution
- [ ] Error analysis: 6–8 concrete failure cases
- [ ] Limitations — be explicit about the exaggeration-attack weakness and False Block Rate
- [ ] **Re-run the competing-work search one final time**

### 6.2 Release
- [ ] README: install, 5-minute example, how to plug in your own model, results table, license
- [ ] MIT license for our code
- [ ] Tag a release; publish the counterfactual/fake-text benchmark
- [ ] Post to arXiv — **do this as soon as results are solid.** It timestamps the work
- [ ] Submit to target venue (IEEE India conference, or a subscription journal)

### 6.3 Course deliverables
- [ ] Final PPT updated with real numbers
- [ ] Project report
- [ ] Viva prep — be able to walk through the worked example cold

**➡️ Final commit & push.**

---

## Worked example (memorize for the viva)

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

Error without TextGuard: 11. With TextGuard: 2.
DSS: `−1 × (106−120)/10 = +1.4` → `−1 × (115−120)/10 = +0.5`.

With **real** text ("demand rising"), directions agree → g = 1 → text fully used.
Good text is not blocked.

---

## Commit convention

```
P0: setup repo and environment
P1: reproduce Time-MMD baselines
P1: add directional-claim check results
P2: implement text condition generators
P2: screening results for models 1-5
P3: DSS metric + flipped text attack
P4: TextGuard gate implementation
P5: streamlit demo
P6: paper draft
```

Prefix every commit with its phase. Commit as you go, not just at phase boundaries —
but **always** at a phase boundary, with PROJECT_STATUS updated in the same commit.
