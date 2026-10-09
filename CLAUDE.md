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

- **RQ0 (screening, P5.2 — replication, not a contribution)** — Which models actually
  consume text? A model that ignores text cannot be attacked *through* text, so this
  selects the attack surface. Reproduces Sridhar et al. (2026); cite them.
- **RQ1 (P5.3)** — Can semantically misleading text manipulate forecasts of models that
  DO use text?
- **RQ2 (P5.3)** — Does fusion type (early / late) determine vulnerability?
- **RQ3 (P6)** — Can TextGuard reduce manipulation without losing the benefit of genuine
  text, and does it beat trained alternatives (CFA) and architectural ones (PA-RNet)?
- **RQ4 (R6; was P7.4)** — Can embedding-drift monitoring detect a sustained fake-text
  campaign?

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

## 4a. Competing work — REFERENCE, not a phase task

> This section exists so the novelty claim stays accurate. It is **reference material for
> writing Related Work in P8**, not a task blocking any build phase. Build first.

| Paper | Overlap | How we differ |
|---|---|---|
| **Sridhar et al. (2026)** *Semantics or Structure? Auditing Text Sensitivity in Multimodal TSF* | Ran empty/constant/shuffled/cross-domain perturbations on Time-MMD across three architectures; found <0.5% MSE movement; released a diagnostic toolkit | Our screening phase (P5.2) reproduces this — **cite them, don't claim it**. Our contribution starts after: adversarial text + defense |
| **Lee et al. (2026)** *Constrained Fusion / CFA* — code `seunghan96/cfa` | Plug-in fusion control without modifying the backbone | CFA is **trained** (low-rank adapters) and targets *irrelevant* text. TextGuard is **training-free**, inference-time, and targets *adversarial* text. Use as the primary defense baseline |
| **Corroboration Illusion** (arXiv 2609.22246) | Adversary publishes news to move forecasts in a chosen direction | They attack **RAG event forecasters** (ForecastBench probabilities). We attack **numerical multimodal TSF** |
| **Plug-In Adversarial Detection for Time-Series LLMs** (arXiv 2512.12154) | Plug-in adversarial detection in our space | Check whether their attack is numeric or textual when writing P8. If textual, reposition TextGuard as *gated fusion* vs their *detection* |
| **Liu & Jiang (2024)** arXiv 2412.08099 | Adversarial attacks on LLM time-series forecasters | Attacks the **numerical** channel (2% of dataset mean), not text. Supports our motivation that adversarial > random noise |
| **Zhu et al. (2025)** *PA-RNet* | Robustness to textual perturbation | Proves error reduction under **zero-mean** perturbations only. Meaning-inverted text is a different threat model — **this is our core gap** |

Also relevant to the optional FNSPID phase: *AutoRedTrader* (arXiv 2605.09185) and
*Adversarial manipulation of LLM-based trading agents via financial news headlines* —
check these before claiming novelty on financial data.

**⚠️ Expected result:** if Sridhar's finding replicates, models 3/4 barely read text, so
fake text will produce near-zero DSS — not because they are robust, but because they are
deaf. **That is still publishable**, and it sharpens the thesis:

> *Vulnerability to text attacks is the price of actually using text. The architecture that
> genuinely reads text is the one that can be weaponized. TextGuard lets you have both.*

Do not treat DSS ≈ 0 on the fusion models as a failed experiment.

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

## 7. The 6 text conditions

| Condition | What it is | What failure means | Phase |
|---|---|---|---|
| `real` | Original text | Reference | P3 |
| `empty` | Text removed | Text contributes nothing | P3 |
| `shuffled` | Text from other (earlier) dates, same domain | Model ignores text *content* | P3 |
| `cross_domain` | Text from a different domain, available at the same time | Can't tell relevant from irrelevant | P3 |
| `time_shifted` | Same domain, the period one year earlier | Model ignores text *timing* | P3 |
| `flipped` | Direction reversed ("rose" → "fell"), numbers kept coherent | **Vulnerable to fake news — the core attack** | P3 |
| `exaggerated` | Agrees with trend but overstates it (change ×3) | Over-trusts extreme claims — TextGuard's known weak spot | P3 |

> **Design change (2026-10-09): `flipped` and `exaggerated` are generated by deterministic
> RULES (`src/attacks/rules.py`), not by Qwen.** Reasons: (1) Model 5 is Qwen — a
> Qwen-written attack on Qwen biases the result; (2) P6's direction judge is Qwen —
> same-model circularity (PROJECT_STATUS issue #25); (3) rules make a minimal, exact,
> reproducible edit — only the direction (or magnitude) changes — and are fully auditable.
> **Paper framing: a conservative, minimal-edit attacker.** Every fake records its
> `claimed_direction`, so DSS uses the known attack direction, not a classifier. An
> LLM-written attack is an optional P7 extra. Fakes that cannot be made coherent are marked
> "not attacked", never emitted self-contradictory.
>
> **Variants in practice:** change-flip applies to **Economy, Traffic and SocialGood**
> (SocialGood's reports state changes, not levels); level-flip applies **only to Health_US**,
> mirroring the value to the same distance on the other side of the baseline (2B − V; ratio
> B²/V only if that would be ≤ 0) so fakes stay plausible.

**Why not just test `flipped`:** it is uninterpretable alone. If fake text changes nothing,
that could mean the model resists lies (good) OR ignores all text (bad). `empty` and
`shuffled` are the controls that distinguish these. That is why P5 runs screening before
the attack.

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

**As built in P6a (`src/textguard/`, pre-registered in PROJECT_STATUS §10b):**
- `slope` = OLS slope per step over the last w values; `trend_str = min(|slope|·(w−1)/σ, 1)`
  — the fitted movement across the window (this is what reproduces the worked example).
- **Level branch** (Health_US): the text's above/below-baseline claim vs `sign(y_last − B)`,
  strength `min(|y_last − B|/σ, 1)`, B = the baseline the text states. A text with several
  claims gets `g = min` over them.
- **Judge** = Qwen2.5-1.5B (P5's revision), scored answers; confidence = renormalised answer
  probability. Not the P3 rules (circular).
- **Block mode** at threshold τ: `g < τ → final = A`; τ = 0 is the soft gate, τ > 1 is
  never_trust — the τ sweep is the trade-off curve. B NaN → A; A NaN → NaN.

**Baseline defenses to compare against** (required for the paper):
- `never_trust`: g = 0 always
- `always_trust`: g = 1 always
- `clip`: cap |B − A| at a fixed multiple of sigma
- **CFA** (Lee et al. 2026) — trained constrained fusion. Code at `seunghan96/cfa`.
  **P6a decision: not run** (white-box, trained; our attacked model has no fusion layer) —
  discussed in Related Work instead; see PROJECT_STATUS issue #4.
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
├── CLAUDE.md               # this file — project context
├── PHASES.md               # build plan, P0–P8
├── PROJECT_STATUS.md       # living progress log — UPDATE AT EVERY PHASE BOUNDARY
├── README.md
├── requirements.txt
├── data/                   # Time-MMD (+ FNSPID if P7.6) — gitignored
├── texts/                  # generated text conditions — gitignored
├── src/
│   ├── data/               # loader.py, splits.py, inspect_domains.py, check_*.py
│   ├── models/             # base.py, dlinear.py, patchtst.py, *_bert.py,
│   │                       # qwen_forecaster.py, dummy.py
│   ├── attacks/            # empty, shuffled, cross_domain, time_shifted,
│   │                       # flipped, exaggerated, validate_attacks.py
│   ├── textguard/          # trend_detector.py, text_direction.py, gate.py, textguard.py
│   ├── metrics.py          # MSE, MAE, Text Usage Gap, DSS, Flip Rate, False Block Rate
│   ├── auditor.py          # P7
│   └── run_experiments.py
├── notebooks/              # Colab GPU runs (Qwen)
├── results/                # raw predictions (CSV) + aggregated metrics
├── app/                    # Streamlit demo (P7)
├── tests/                  # pytest
└── paper/                  # LaTeX draft, figures
```

**Local:** `C:\Users\Aryan\textguard`
**Remote:** `https://github.com/aryanwaingankar4/textguard`

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
- **Research extensions R1–R7 run before P8** (decided 2026-10-09; PHASES.md "Research
  extensions", PROJECT_STATUS §1a). Rules for every extension:
  - **pre-registered and committed alone** (pushed) before any of its test results exist;
    later changes are dated amendments, reported as deviations;
  - **new models and new attacks may be evaluated on the existing test set; already-reported
    methods (Models 1–5, TextGuard v1 at its frozen P6 settings) are never changed** —
    re-run on new inputs, never re-tuned or redefined;
  - **R4's held-out domains are chosen and frozen before any model output exists on them**
    (their data checks — directional %, timing — are allowed);
  - **write each extension's results section into `paper/` as it finishes**;
  - **null results are reported, not dropped.**
- **Write docstrings explaining *why*, not just what** — Aryan must be able to explain
  every module in the viva, alone.
- **Prefer small, runnable increments** over large refactors.
- **Ethics:** generated fake text is for defense research only. Label it clearly in any
  released files, and state this in the README.

---

## 15. Known risks & fallbacks

| Risk | Fallback |
|---|---|
| MM-TSFlib code broken | Spend a bounded effort fixing it; then write a simple PatchTST + BERT-concat model yourself. Don't let it block P3–P6 |
| Time-MMD text lacks directional claims (<30%) | Found in P1 by `check_directional.py`. Switch domains, or retarget the attack to whatever the text does assert |
| Late-fusion models ignore text ⇒ DSS ≈ 0 | **Expected, not a failure.** See §4a. Pivot the narrative to Model 5 and the use-vs-vulnerability trade-off |
| Colab GPU throttled | Move Qwen runs to Kaggle |
| Rule-based flip can't handle a text | Mark it "not attacked" (never emit a contradictory fake); report coverage per domain. LLM-written attack only as an optional P7 extra |
| Qwen output unparseable | Handle parse failures explicitly in P2.4 — never silently drop samples |
| Losing momentum | Nothing is time-bound. **P0–P6 + P8 is a complete paper.** Drop P7 items in this order: FNSPID → CI → drift monitor → Model 6 → audit report → demo |

---

## 16. Key literature — for writing Related Work in P8

**Directly competing** (see §4a for the "how we differ" table):
Sridhar et al. (2026) *Semantics or Structure?* · Lee et al. (2026) *CFA* ·
*Corroboration Illusion* (arXiv 2609.22246) ·
*Plug-In Adversarial Detection for Time-Series LLMs* (arXiv 2512.12154) ·
Liu & Jiang (2024) arXiv 2412.08099 · Zhu et al. (2025) *PA-RNet*

**Foundational:**
Liu et al. (2024) *Time-MMD* — NeurIPS 37, our dataset ·
Nie et al. (2023) *PatchTST* — ICLR · Zeng et al. (2023) *DLinear* — AAAI ·
Yang et al. (2024) *Qwen2.5 Technical Report* — arXiv:2412.15115

**Supporting the motivation:**
Nguyen et al. (2026) *Does Text Actually Help? / text collapse* ·
Zhang et al. (2025) *When Does Multimodality Lead to Better Forecasting?* ·
Liu et al. (2026) *TimesX* · Williams et al. (2024) *Context is Key* ·
Kim et al. (2024) *Hybrid-MMF* (negative result)

**Field-is-accuracy-focused evidence** (abstracts enough):
TEA-Mamba · SpecTF · Dual-Forecaster · TriAlignNet · TTCA · TeR

**Drift monitor (only if P7.4):**
Sekar et al. (2026) *ZEDD* · Greco et al. (2024) *DriftLens*

**Financial attacks (only if P7.6):**
*AutoRedTrader* (arXiv 2605.09185) · *Adversarial manipulation of LLM-based trading agents
via financial news headlines*

> Gap claims verified 2026-09-28. **Re-run the arXiv search in P8 before submitting** —
> this field publishes weekly.

---

## 17. Things Claude should NOT assume

- Do not assume MM-TSFlib runs out of the box. Research code often breaks.
- Do not assume Time-MMD text makes directional claims until `check_directional.py` (P1) says so.
- Do not propose new forecasting architectures — out of scope.
- Do not write results into the paper before the experiments have actually run.
- **Do not claim the text-usage screening as a novel contribution.** Sridhar et al. (2026)
  did it. We replicate and cite. Novelty = adversarial text + DSS + training-free defense.
- **Do not describe TextGuard as "the first plug-in fusion control."** CFA exists. The
  precise claim is "training-free, inference-time, black-box."
- Do not treat DSS ≈ 0 on late-fusion models as a failed experiment — see §4a.
- Do not block a build phase on reading papers. Literature work lives in P8.
