# CLAUDE.md — TextGuard

> Persistent context for Claude Code. Read this first in every new session.
> For current progress, read `PROJECT_STATUS.md`. For the work plan, read `PHASES.md`.

---

## 1. Project identity

**Title:** TextGuard: Detecting and Defending Against Misleading Text Attacks on Multimodal Time-Series Forecasting Models

**Team:** Aryan Manoj Waingankar (PRN 23070126022) — **solo project, all work by one person**
(Devansh Kumar appears on the course registration; clarify his role before submission)
**Course:** Final-year MMAI project, B.Tech AI & ML, Symbiosis Institute of Technology, Pune
**Category:** Cybersecurity | **Modalities:** Time series + Text | **SDG 16**
**Goal:** Working system + research paper (arXiv preprint, then conference/journal)
**Pacing:** Phase-gated, not date-gated. See `PHASES.md`. No deadlines — finish a phase,
commit, push, move on.

---

## 2. What this project is (one paragraph)

Multimodal time-series forecasters combine numerical history with aligned text reports
(news, government reports) to predict future values. Where these models genuinely consume
text, that text is an **unverified input channel** — an attack surface. Prior robustness
work on this modality tests only random, zero-mean perturbations; semantic attacks that
invert the *meaning* of a report are untested on numerical forecasters. This project
(a) screens 6 models across 3 fusion types to find which actually use text, (b) attacks
those models with direction-flipped and exaggerated text, (c) introduces DSS, a metric for
directional manipulation, (d) proposes TextGuard, a **training-free, inference-time**
model-agnostic consistency gate, and (e) ships an open-source toolkit, audit report
generator, drift monitor, and Streamlit demo.

**This is an ATTACK + DEFENSE paper, NOT an audit paper and NOT a new-architecture paper.**
We do not try to beat SOTA accuracy. Our contribution is the threat model, its measurement,
and a deployable defense.

> **Repositioned 2026-09-28** after verifying the literature. The original framing was
> "audit whether models use text" — that was scooped by Sridhar et al. (2026). See §4a.

---

## 3. Research questions

- **RQ0 (screening, NOT a contribution)** — Which of our models actually consume text?
  Replicates Sridhar et al. (2026). Cite them; do not claim novelty. Purpose: a model that
  ignores text cannot be attacked *through* text, so RQ0 selects the attack surface.
- **RQ1** — Can semantically misleading text manipulate forecasts of models that DO use text?
- **RQ2** — Does fusion type (early / intermediate / late) determine vulnerability?
- **RQ3** — Can TextGuard reduce manipulation without losing the benefit of genuine text,
  and does it beat trained alternatives (CFA) and architectural ones (PA-RNet)?
- **RQ4** (stretch) — Can embedding-drift monitoring detect a sustained fake-text campaign?

---

## 4. Research gaps we fill (VERIFIED 2026-09-28)

1. **Semantic directional text attacks on numerical multimodal forecasters are untested.**
   PA-RNet proves error reduction under *zero-mean* perturbations only. Liu & Jiang
   (arXiv 2412.08099) attack the **numerical** input (2% of dataset mean), not text.
   Corroboration Illusion attacks **RAG event forecasters** (probability judgments), not
   numerical TSF. AutoRedTrader attacks **trading agents**. Our target — Time-MMD-style
   numeric forecasters under meaning-inverted text — is unoccupied.
2. **No directional manipulation metric.** Existing work reports MSE degradation, which
   cannot distinguish "wrong" from "wrong in the direction the attacker wanted". DSS does.
3. **No cross-fusion-type vulnerability comparison** under semantic text attack.
4. **No training-free, inference-time defense.** CFA trains low-rank adapters; PA-RNet is
   an architecture. TextGuard requires neither retraining nor model access.
5. Embedding-drift detection exists for LLM prompt injection, not for text streams feeding
   forecasters.

---

## 4a. Competing work — READ THESE FIRST (Phase 1 blocker)

These four determine whether the project survives as designed. Read in full before writing
any experiment code.

| Paper | Overlap | How we differ |
|---|---|---|
| **Sridhar et al. (2026)** *Semantics or Structure? Auditing Text Sensitivity in Multimodal TSF* | **Scoops our original RQ1.** Same conditions (empty, constant, shuffled, cross-domain), same dataset (Time-MMD), same lib (MM-TSFlib), and they released a diagnostic toolkit. Found <0.5% MSE change across three architectures | We **cite and replicate** them as screening. Our contribution starts where theirs ends: adversarial text + defense |
| **Lee et al. (2026)** *Constrained Fusion / CFA* | Plug-in fusion control without modifying the backbone | CFA is **trained** (low-rank adapters) and targets *irrelevant* text. TextGuard is **training-free**, inference-time, and targets *adversarial* text. **Use CFA as a defense baseline** |
| **Corroboration Illusion** (arXiv 2609.22246, Sep 2026) | Adversary publishes news to move forecasts in a chosen direction, opposite the clean forecast | They attack **RAG event forecasters** (ForecastBench probabilities). We attack **numerical multimodal TSF**. Very recent — read carefully |
| **Plug-In Adversarial Detection for Time-Series LLMs in Energy Forecasting** (arXiv 2512.12154) | Plug-in adversarial detection, our exact positioning | **Verify whether their attack is numeric or textual.** If textual, we must reposition again |

**⚠️ THE BIG RISK:** if Sridhar's finding holds on our setup, models 3/4/6 barely read text,
so fake text will produce near-zero DSS — not because they are robust, but because they are
deaf. Then the attack only lands on Model 5 (LLM, early fusion).

**If that happens, the finding is still publishable** and becomes the paper's thesis:
*"Vulnerability to text attacks is the price of actually using text. Models that resist
attack do so only by ignoring the modality entirely — and the one architecture that genuinely
reads text is the one that can be weaponized. TextGuard lets you have both."*
That is a sharper story than the original. Do not panic if DSS ≈ 0 for the fusion models;
report it as the trade-off finding and pivot the narrative to Model 5 + TextGuard.

---

## 5. Datasets

| Dataset | Role | Notes |
|---|---|---|
| **Time-MMD** (NeurIPS 2024) | **Primary.** All 6 models, all 7 conditions | 9 domains; use 4–5 text-rich ones. Repo: `AdityaLab/Time-MMD`, lib: MM-TSFlib |
| **FNSPID** (arXiv 2402.06698) | **Secondary validation** (Week 5) | HF: `Zihan1004/FNSPID`. Use only 5–10 stocks. **License CC BY-NC-4.0 — academic use only, must be stated in README** |

**Splitting rule:** ALWAYS chronological (train → val → test in time order). Never random.
Random splits cause future-information leakage and invalidate everything.

**Tuning rule:** All thresholds (trend window, gate cutoff, warning threshold) tuned on
**validation split only**. Never touch the test split when choosing settings.

---

## 6. Models under test

| # | Model | Text? | Fusion | Source | Required? |
|---|---|---|---|---|---|
| 1 | DLinear | No | — (baseline) | MM-TSFlib | ✅ |
| 2 | PatchTST | No | — (baseline) | MM-TSFlib | ✅ |
| 3 | DLinear + BERT | Yes | Late | MM-TSFlib | ✅ |
| 4 | PatchTST + BERT | Yes | Late | MM-TSFlib | ✅ |
| 5 | Qwen2.5-1.5B-Instruct | Yes | Early (one prompt) | HuggingFace | ✅ |
| 6 | PatchTST + BERT + cross-attention | Yes | Intermediate | **Self-built** if no public code | ⬜ Phase 5 |

Models 1–2 are the control: they isolate how much the text contributes.
Use **BERT only** as the text encoder (not BERT + GPT-2) to keep the experiment grid small.

**Solo-scope note:** Models 1–5 cover early and late fusion, which is enough to support the
fusion-type comparison. Model 6 (intermediate) completes the picture but is optional — it
lives in Phase 5. Do not block the core paper on it.

---

## 7. The 7 text conditions

| Condition | What it is | What failure means |
|---|---|---|
| `real` | Original text | Reference |
| `empty` | Text removed | Text contributes nothing |
| `shuffled` | Text from other dates, same domain | Model ignores text *content* |
| `cross_domain` | Text from a different domain | Can't tell relevant from irrelevant |
| `time_shifted` | Text from the wrong period | Data leakage / timing bug |
| `flipped` | Direction reversed ("rose" → "fell") | Vulnerable to fake news |
| `exaggerated` | Agrees with trend but overstates it | Over-trusts extreme claims — TextGuard's known weak spot |

**Minimum viable set if time-constrained:** `real`, `empty`, `shuffled`, `flipped`.
Drop `cross_domain` first. The others are cheap (same code, different text).

**Why all of them:** `flipped` alone is uninterpretable. If fake text changes nothing, that
could mean the model resists lies (good) OR ignores all text (bad). `empty` and `shuffled`
are the controls that distinguish these.

---

## 8. TextGuard algorithm

Model-agnostic. Only uses model **outputs** — never touches internals. No retraining.

```
A = model.predict(numbers, text=None)      # without text
B = model.predict(numbers, text=text)      # with text

# Trend detector (from raw numbers)
slope        = linear_fit(recent_window)
trend_dir    = sign(slope)                 # -1, 0, +1
trend_str    = min(abs(slope) / sigma, 1)  # sigma = std of series

# Text direction (via Qwen classification)
text_dir, conf = classify_direction(text)  # dir in {-1, 0, +1}, conf in [0,1]

# Gate
if trend_dir == 0 or text_dir == 0 or trend_dir == text_dir:
    g = 1.0
else:
    g = 1.0 - (conf * trend_str)

final = A + g * (B - A)
warn  = (g < 0.5)
```

**Baseline defenses to compare against** (required for the paper):
- `never_trust`: g = 0 always
- `always_trust`: g = 1 always
- `clip`: cap |B − A| at a fixed multiple of sigma
- **CFA** (Lee et al. 2026) — trained constrained fusion. Code at `seunghan96/cfa`.
  This is the strongest baseline and the one reviewers will ask about. If running it is
  too costly, at minimum discuss it in Related Work and justify the omission.

**TextGuard's positioning claim (say this precisely):** training-free, inference-time,
requires only black-box `predict()` access. CFA requires training; PA-RNet requires
changing the architecture. That is the axis on which we win, not raw accuracy.

---

## 9. Metrics

| Metric | Formula | Good |
|---|---|---|
| MSE / MAE | mean (y−ŷ)² / mean abs(y−ŷ) | Low |
| Text Usage Gap | MSE(empty) − MSE(real) | High (≈0 ⇒ text collapse) |
| **DSS** (ours) | `d * (ŷ_fake − ŷ_real) / sigma`, where d = direction the fake text claims | Low (≈0) |
| Flip Rate | % forecasts whose up/down direction reverses under fake text | Low |
| False Block Rate | % of **genuine** texts with g < 0.5 | Low |

DSS and Flip Rate measure the attack. False Block Rate is the honesty check — a defense
that blocks all text would score perfectly on the first two and fail here.

**Statistics:** 3 seeds per config; report mean ± std; paired Wilcoxon signed-rank for
key comparisons (TextGuard vs undefended, and vs each baseline defense).

---

## 10. Deliverables

1. **GitHub toolkit** — `Auditor` + `TextGuard`, usable on any model via `predict(numbers, text)`
2. **Audit Report generator** — PDF/HTML + CSV + JSON (see §11)
3. **DSS metric** + benchmark of counterfactual/fake text versions
4. **Drift monitor** — BERT-embedding cosine distance vs reference centroid
5. **Streamlit demo** — live attack vs defense (Qwen results precomputed; free tier has no GPU)
6. **Research paper** — arXiv preprint + conference/journal submission

**Dropped from scope:** CSV upload feature in Streamlit (decided against).

---

## 11. Audit Report contents (answers "what does a model provider get?")

The provider supplies their model (`predict(numbers, text)`) + their dataset. We return:

1. **Health check** — traffic-light table: uses text? fooled? notices irrelevant text?
2. **Fault breakdown** — scores split by domain, forecast horizon, and attack type
3. **Worst failure cases** — top 10 samples with real vs fake text, prediction before/after, true value (reproducible bug reports)
4. **Diagnosis + fixes** — each fault mapped to a recommended correction
5. **With-TextGuard comparison** — before/after table including False Block Rate
6. **Drift monitor setup** — reference centroid + alert threshold for their data

**Key framing for the viva:** a developer only sees accuracy on their own clean test data.
That cannot reveal text collapse, leakage, or adversarial vulnerability, because those
situations never appear in the test set. This is a crash test, not a speed test.

---

## 12. Tech stack

Python 3.10+ · PyTorch · HuggingFace Transformers · scikit-learn · pandas · NumPy
MM-TSFlib (DLinear, PatchTST) · BERT · Qwen2.5-1.5B-Instruct
MLflow (tracking) · DVC (data versioning) · GitHub Actions (CI) · pytest
Streamlit (demo) · Matplotlib/Plotly · Overleaf/LaTeX (paper)

**Compute:** laptop CPU for small models; free Google Colab T4 for Qwen (via the official
Google Colab extension for VS Code). Kaggle is the fallback if Colab throttles.
**Colab gotcha:** the Colab runtime cannot read local files — `git clone` the repo inside
the notebook instead.

**Budget: ₹0** until publication fees.

---

## 13. Repo structure

```
textguard/
├── CLAUDE.md               # this file
├── PROJECT_STATUS.md       # living progress log — UPDATE AFTER EVERY SESSION
├── PHASES.md               # detailed 6-week plan
├── data/                   # Time-MMD + FNSPID (DVC-tracked, not in git)
├── texts/                  # generated counterfactual & fake text versions
├── src/
│   ├── data/               # loaders, chronological splits, domain selection
│   ├── models/             # baselines, MM-TSFlib wrappers, Qwen forecaster, cross-attn
│   ├── attacks/            # the 7 text condition generators
│   ├── textguard/          # trend detector, text direction, gate, baseline defenses
│   ├── metrics.py          # MSE, MAE, Text Usage Gap, DSS, Flip Rate, False Block Rate
│   ├── report/             # audit report generator
│   ├── drift/              # embedding drift monitor
│   └── run_audit.py        # main experiment runner
├── notebooks/              # Colab GPU runs (Qwen)
├── results/                # raw predictions (CSV) + aggregated metrics
├── app/                    # Streamlit demo
├── tests/                  # pytest
├── paper/                  # LaTeX draft, figures
└── README.md
```

---

## 14. Working conventions for Claude Code

- **This is a solo project.** Do not suggest splitting work between people. Do not assume
  a second person is reviewing, testing, or writing anything.
- **Pacing is phase-gated, never date-gated.** Do not invent deadlines or ask how much time
  is left. Read `PHASES.md` for the current phase and its exit criterion.
- **At the end of every phase: update `PROJECT_STATUS.md`, `git add`, `git commit`, `git push`.**
  Remind the user if they forget. Prefix commits with the phase: `P3: DSS metric`.
- **Always save raw predictions**, not just aggregate metrics. Store every
  (model, domain, condition, seed, sample_id, prediction) row to CSV in `results/`.
  New metrics and plots must be derivable without re-running experiments.
- **Log every run to MLflow** with params: model, domain, condition, seed.
- **Build TextGuard as a reusable plug-in from day one** — `Auditor` and `TextGuard` as
  two independent classes taking any object with `predict(numbers, text)`. Do not hardcode
  it to our models.
- **Never tune on the test split.**
- **Write docstrings explaining *why*, not just what** — Aryan must be able to explain
  every module in the viva, alone.
- **Prefer small, runnable increments** over large refactors.
- **Ethics:** generated fake text is for defense research only. Label it clearly in any
  released files, and state this in the README.

---

## 15. Known risks & fallbacks

| Risk | Fallback |
|---|---|
| MM-TSFlib code broken | Max 3–4 days fixing; then write a simple PatchTST + BERT-concat model ourselves |
| Time-MMD text lacks directional claims | **CHECK IN PHASE 1.** Read ~50 reports, count directional claims. If <30%, switch domains or retarget the attack to what the text does assert |
| **Fusion models ignore text ⇒ DSS ≈ 0** (Sridhar's finding replicates) | **Expected, not a failure.** Pivot the narrative: vulnerability is the price of actually using text. Focus attack results on Model 5 (LLM); report the fusion models' immunity as deafness, not robustness. See §4a |
| arXiv 2512.12154 turns out to attack text, not numbers | Reposition TextGuard as the *gated-fusion* defense vs their *detection* approach, and cite them as the closest baseline |
| Colab GPU throttled | Move Qwen runs to Kaggle |
| Similar paper published mid-project | Post to arXiv as soon as results exist — it timestamps the work |
| Qwen rewrites low quality | Stricter prompt + manual filter, or rule-based word flip for simple reports |
| Behind on effort / losing momentum | Nothing is time-bound. Phase 5 is entirely optional. **P0–P4 + P6 is a complete paper.** Drop Phase 5 items in this order: FNSPID → MLOps → drift monitor → Model 6 → audit report → demo |

---

## 16. Key literature

**Tier 0 — competing work, read in full FIRST (see §4a):**
1. Sridhar et al. (2026) *Semantics or Structure? Auditing Text Sensitivity in Multimodal TSF*
2. Lee et al. (2026) *Rethinking Multimodal Fusion: Auxiliary Modalities Need Constrained Fusion (CFA)* — code: `seunghan96/cfa`
3. *The Corroboration Illusion* — arXiv 2609.22246
4. *Plug-In Adversarial Detection for Time-Series LLMs in Energy Forecasting* — arXiv 2512.12154

**Tier 1 — must-read in full:**
5. Liu et al. (2024) *Time-MMD* — NeurIPS 37. Our dataset + baselines to reproduce.
6. Zhu et al. (2025) *PA-RNet* — proves error reduction under **zero-mean** perturbations.
   Our core gap. Quote the zero-mean assumption precisely in Related Work.
7. Liu & Jiang (2024) *Adversarial Vulnerabilities in LLMs for Time Series Forecasting* —
   arXiv 2412.08099. Attacks the **numerical** channel, not text. Establishes that
   adversarial > random noise, which supports our motivation.
8. Nguyen et al. (2026) *Does Text Actually Help? / text collapse*
9. Liu et al. (2026) *TimesX* — current evaluation is too weak.

**Tier 2 — method reference:** Williams et al. (2024) *Context is Key*; Anh et al. (2026)
*TTCA* (Model 6); Nie et al. (2023) *PatchTST*; Zeng et al. (2023) *DLinear*;
Su et al. (2025) *TeR* (text reinforcement — the inverse of our attack).

**Tier 3 — adjacent threat models (Related Work, abstracts enough):**
*AutoRedTrader* (arXiv 2605.09185, misinformation injection into trading agents);
*Adversarial manipulation of LLM-based trading agents via financial news headlines*.
**Both are directly relevant to the FNSPID phase — check them before claiming novelty there.**

**Drift monitor:** Sekar et al. (2026) *ZEDD*; Greco et al. (2024) *DriftLens*.

**Context papers (abstracts only):** TEA-Mamba, SpecTF, Dual-Forecaster, TriAlignNet —
all chase accuracy, none test adversarial robustness. Cite as evidence the field is
accuracy-focused.

> Gap claims verified 2026-09-28 via academic + arXiv search. **Re-run the search at the
> start of Phase 3 and Phase 6** — this field publishes fast, and Corroboration Illusion
> appeared three weeks before our check.

---

## 17. Things Claude should NOT assume

- Do not assume MM-TSFlib runs out of the box. Research code often breaks.
- Do not assume Time-MMD text makes directional claims until Phase 1 verifies it.
- Do not assume the cross-attention paper has public code.
- Do not propose new forecasting architectures — that is out of scope.
- Do not write results into the paper before the experiments have actually run.
- **Do not claim the text-usage audit as a novel contribution.** Sridhar et al. (2026) did
  it. We replicate and cite. Novelty = adversarial text + DSS + training-free defense.
- **Do not describe TextGuard as "the first plug-in fusion control."** CFA exists. The
  precise claim is "training-free, inference-time, black-box."
- Do not treat DSS ≈ 0 on fusion models as a failed experiment — see §4a.
