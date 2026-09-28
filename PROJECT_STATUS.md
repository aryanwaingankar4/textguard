# PROJECT_STATUS.md — TextGuard

> **Living document. Update at the end of every session, and always at a phase boundary.**
> Claude Code reads this to recover context after a session reset.
> Keep it factual: what is actually done, what actually broke, what the numbers actually are.

**Current phase:** P1 — Data Pipeline
**Last updated:** 2026-09-28

> 👤 **Solo project.** All work by Aryan.
> 📌 **Phase-gated, no deadlines.** See `PHASES.md`.

---

## 1. Phase board

| Phase | Builds | Required? | Status |
|---|---|---|---|
| P0 | Environment, repo, folder skeleton | ✅ | ✅ Done |
| P1 | Data pipeline — loader, splits, domain selection | ✅ | ⚪ Pending |
| P2 | Model wrappers — 5 models, one interface | ✅ | ⚪ Pending |
| P3 | Attack module — 6 text conditions | ✅ | ⚪ Pending |
| P4 | Metrics module — DSS, Flip Rate, etc. | ✅ | ⚪ Pending |
| P5 | Experiment runner — screening + attack | ✅ | ⚪ Pending |
| P6 | TextGuard defense | ✅ Core | ⚪ Pending |
| P7 | Auditor, demo, extras | ⬜ Optional | ⚪ Pending |
| P8 | Paper & release | ✅ | ⚪ Pending |

Legend: ⚪ pending · 🔵 in progress · ✅ done · ⚠️ blocked · ❌ cut

**P0–P6 + P8 = complete paper. P7 is upside.**

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
| Text encoder | BERT only — keeps the experiment grid small |
| CFA (Lee et al. 2026) | Primary defense baseline |
| Distribution | GitHub toolkit + Streamlit demo (demo = P7) |
| Streamlit CSV upload | ❌ Cut from scope |
| Audit report generator | ✅ In scope — P7 |

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
| `src/data/loader.py` | P1 | ⚪ |
| `src/data/splits.py` | P1 | ⚪ |
| `src/data/inspect_domains.py` | P1 | ⚪ |
| `src/models/base.py` | P2 | ⚪ |
| `src/models/dlinear.py` | P2 | ⚪ |
| `src/models/patchtst.py` | P2 | ⚪ |
| `src/models/dlinear_bert.py` | P2 | ⚪ |
| `src/models/patchtst_bert.py` | P2 | ⚪ |
| `src/models/qwen_forecaster.py` | P2 | ⚪ |
| `src/models/dummy.py` | P2 | ⚪ |
| `src/attacks/*.py` (6 conditions) | P3 | ⚪ |
| `src/metrics.py` | P4 | ⚪ |
| `src/run_experiments.py` | P5 | ⚪ |
| `src/textguard/trend_detector.py` | P6 | ⚪ |
| `src/textguard/text_direction.py` | P6 | ⚪ |
| `src/textguard/gate.py` | P6 | ⚪ |
| `src/textguard/textguard.py` | P6 | ⚪ |
| `src/auditor.py` | P7 | ⚪ |
| `app/` (Streamlit) | P7 | ⚪ |

---

## 6. Data findings (fill in during P1)

**Selected domains:** ______

**Per-domain stats:**

| Domain | Samples | % rows with text | Avg text length |
|---|---|---|---|
| _(pending)_ | | | |

**Directional-claim check:** ____% of 50 sampled reports make a clear up/down claim
→ verdict: ______ (≥30% = the flipped attack is viable)

**Timing check:** ______

**MM-TSFlib fusion mechanism (from reading the code):** ______

---

## 7. Baseline reproduction (P2)

| Domain | Model | Our MSE | Published MSE | Close? |
|---|---|---|---|---|
| _(pending)_ | | | | |

---

## 8. Fake text validation (P3)

- Samples manually checked: ___ / 100
- Direction correctly flipped: ___%
- Length similarity passed: ___%
- Automatic classifier agreement: ___%
- Failure modes seen: _(pending)_

---

## 9. Results

### P5 — Screening: which models use text
| Model | Fusion | MSE(real) | MSE(empty) | MSE(shuffled) | Usage Gap | Uses text? |
|---|---|---|---|---|---|---|
| _(pending)_ | | | | | | |

**Attack targets identified:** _(pending)_

### P5 — Attack
| Model | Fusion | DSS (flipped) | Flip Rate | DSS (exaggerated) |
|---|---|---|---|---|
| _(pending)_ | | | | |

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

## 10. Open issues

| # | Issue | Phase | Status |
|---|---|---|---|
| 1 | Does Time-MMD text make directional claims? (≥30% needed) | P1 | ⬜ Open |
| 2 | Is text guaranteed available *before* its forecast window? | P1 | ⬜ Open |
| 3 | How does MM-TSFlib actually fuse text? (we assume late) | P2 | ⬜ Open |
| 4 | Can CFA (`seunghan96/cfa`) run as a defense baseline? | P6 | ⬜ Open |
| 5 | Colab VS Code extension unreliable — use colab.research.google.com directly for GPU work (Qwen) in P2; `git clone` the repo inside the notebook | P2 | ⬜ Open |

---

## 11. Session log

> Newest at top. Format: what was done / what broke / next step.

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
