# PROJECT_STATUS.md — TextGuard

> **Living document. Update at the end of every session, and always at a phase boundary.**
> Claude Code reads this to recover context after a session reset.
> Keep it factual: what is actually done, what actually broke, what the numbers actually are.

**Current phase:** P0 — Setup & Repo
**Last updated:** 2026-09-28 (kickoff + literature verification — no code written yet)

> ⚠️ **Project repositioned.** Literature verification found our original RQ1 (auditing
> whether models use text) was published by Sridhar et al. (2026). This is now an
> **attack + defense** paper; the audit became a screening step we cite and replicate.
> See CLAUDE.md §4a for the competing-work table and contingency plan.

> 👤 **Solo project.** All work by Aryan.

---

## 1. Phase board

| Phase | Name | Required? | Status |
|---|---|---|---|
| P0 | Setup & repo | ✅ | 🔵 In progress |
| P1 | Foundation: data, baselines, competing literature | ✅ | ⚪ Pending |
| P2 | Screening: which models use text (RQ0) | ✅ | ⚪ Pending |
| P3 | Attack: misleading text + DSS (RQ1, RQ2) | ✅ Core | ⚪ Pending |
| P4 | Defense: TextGuard (RQ3) | ✅ Core | ⚪ Pending |
| P5 | Extras: demo, report gen, Model 6, drift, MLOps, FNSPID | ⬜ Optional | ⚪ Pending |
| P6 | Paper & release | ✅ | ⚪ Pending |

Legend: ⚪ pending · 🔵 in progress · ✅ done · ⚠️ blocked · ❌ cut

**P0–P4 + P6 = a complete paper. P5 is upside.**

---

## 2. Decisions locked in

| Decision | Value |
|---|---|
| Project topic | TextGuard (replaces QRShield) |
| Category | Cybersecurity, SDG 16 |
| Team | **Solo — Aryan only.** Devansh on registration; role to be clarified |
| Pacing | **Phase-gated, no deadlines** |
| Framing | **Attack + defense paper** (was: audit + defense) |
| RQ1 (text-usage audit) | Demoted to RQ0 screening; cite Sridhar et al. 2026 |
| Novelty claim | Semantic directional text attack + DSS + training-free defense |
| Primary dataset | Time-MMD |
| Secondary dataset | FNSPID (public, CC BY-NC-4.0) — **Phase 5, optional** |
| Models | 1–5 required (early + late fusion); Model 6 optional in P5 |
| Text encoder | BERT only — keeps the grid small |
| CFA (Lee et al. 2026) | Primary defense baseline |
| Distribution | GitHub toolkit + Streamlit demo (demo = P5) |
| Streamlit CSV upload | ❌ Cut from scope |
| Audit Report generator | ✅ In scope — P5 (from teacher feedback) |

---

## 3. Open questions / blockers

| # | Question | Status |
|---|---|---|
| 1 | Guide approval: multimodal ✓, Cybersecurity ✓, paper expected? | ⬜ Open — **P0** |
| 2 | Devansh's role — co-author or registration only? | ⬜ Open — **P0** |
| 3 | Does Time-MMD text make directional claims? (target ≥30%) | ⬜ Open — **P1 blocker** |
| 4 | Is text guaranteed available *before* the forecast period? | ⬜ Open — **P1 blocker** |
| 5 | How does MM-TSFlib actually fuse text? (we assume late — unverified) | ⬜ Open — **P1** |
| 6 | Does arXiv 2512.12154 attack **text** or **numbers**? If text → reposition | ⬜ Open — **P1 blocker** |
| 7 | Can we run CFA (`seunghan96/cfa`) as a defense baseline? | ⬜ Open — P1 |
| 8 | Does Sridhar's released toolkit overlap enough that we build on it instead? | ⬜ Open — P1 |
| 9 | Anyone published adversarial-text attacks on numerical multimodal forecasters? | 🟡 Searched 2026-09-28 — none found for our exact target. Re-check at P3 and P6 |
| 10 | Do AutoRedTrader / financial-headline papers pre-empt the FNSPID phase? | ⬜ Open — before P5.6 |

---

## 4. Novelty claim (rewrite after P1 literature)

> We present the first study of **semantically misleading text attacks on numerical
> multimodal time-series forecasters**, introduce **DSS** to measure directional
> manipulation, and propose **TextGuard** — a **training-free, inference-time, black-box**
> defense.

Every word is load-bearing:
- "semantically misleading" excludes PA-RNet (zero-mean noise)
- "numerical multimodal TSF" excludes Corroboration Illusion (RAG events) and AutoRedTrader (trading agents)
- "training-free, inference-time, black-box" excludes CFA (trained) and PA-RNet (architecture)

**Status:** ⬜ Unverified — confirm against full texts in P1.

---

## 5. Current phase checklist — P0

- [ ] Guide approval obtained
- [ ] Devansh's role clarified
- [ ] Project folder created
- [ ] `git init` + `.gitignore`
- [ ] GitHub repo created and pushed
- [ ] CLAUDE.md / PHASES.md / PROJECT_STATUS.md in repo root
- [ ] Python env + dependencies installed
- [ ] Colab extension working (T4 reachable, repo clones inside notebook)

---

## 6. Verification results (fill in during P1)

**Directional-claim check:** ____% of ~50 sampled reports make a clear up/down claim
→ verdict: ______ (≥30% = proceed)

**Timing check:** ______

**MM-TSFlib fusion mechanism:** ______

**arXiv 2512.12154 attack type:** ☐ numeric ☐ textual → implication: ______

**Selected domains:** ______

---

## 7. Reproduction results (P1)

| Domain | Model | Our MSE | Paper MSE | Match? |
|---|---|---|---|---|
| _(pending)_ | | | | |

---

## 8. Key experimental results

### P2 — Screening: Text Usage Gap
| Model | Fusion | MSE(real) | MSE(empty) | MSE(shuffled) | Gap | Uses text? |
|---|---|---|---|---|---|---|
| _(pending)_ | | | | | | |

**Attack targets identified:** _(pending)_

### P3 — Attack
| Model | Fusion | DSS (flipped) | Flip Rate | DSS (exaggerated) |
|---|---|---|---|---|
| _(pending)_ | | | | |

### P4 — Defense
| Model | DSS before | DSS after | Flip before | Flip after | MSE(real) before | MSE(real) after | False Block Rate |
|---|---|---|---|---|---|---|---|
| _(pending)_ | | | | | | | |

### P4 — Defense comparison
| Defense | DSS | Flip Rate | MSE(real) | False Block Rate |
|---|---|---|---|---|
| None | | | | — |
| never_trust (g=0) | | | | |
| always_trust (g=1) | | | | |
| clip | | | | |
| CFA (trained) | | | | |
| **TextGuard** | | | | |

### P5.6 — FNSPID validation (optional)
| Model | DSS before | DSS after | Flip before | Flip after |
|---|---|---|---|---|
| _(pending)_ | | | | |

---

## 9. Fake text validation log (P3)

- Samples manually checked: ___ / target 100
- Correct direction flips: ___%
- Length similarity passed: ___%
- Automatic classifier agreement: ___%
- Failure modes observed: _(pending)_

---

## 10. Session log

> One entry per session. Newest at top.
> Format: what was done / what broke / next step.

### 2026-09-28 (3) — Solo restructure
- Confirmed project is solo; removed all two-person assumptions
- Switched from date-gated to **phase-gated** planning (P0–P6)
- Model 6 and FNSPID moved to optional Phase 5
- Added commit-and-push-at-every-phase rule to all three files
- **Broke:** nothing (no code yet)
- **Next:** P0 — guide approval, repo creation, environment

### 2026-09-28 (2) — Literature verification & repositioning
- Verification search on the gap claims. **Result: partial scoop.**
- **Sridhar et al. (2026)** does our original RQ1: same conditions, dataset, lib;
  three architectures; <0.5% MSE movement; released a toolkit
- **Lee et al. (2026) CFA** occupies "plug-in fusion control" — but trained, not training-free
- **Corroboration Illusion** (2609.22246) — directional news attacks on RAG *event* forecasters
- **arXiv 2412.08099** attacks the numerical channel, not text → our text-attack gap holds
- **Surviving novelty:** semantic directional text attack on numerical multimodal
  forecasters, DSS, training-free inference-time defense, fusion-type comparison
- **Action:** repositioned to attack+defense. Added §4a competing-work table and the
  DSS≈0 contingency to CLAUDE.md
- **Next:** read the 4 Tier-0 papers in full

### 2026-09-28 (1) — Kickoff
- Scope finalized after ~1 week of topic selection and literature scoping
- FNSPID confirmed public (HF `Zihan1004/FNSPID`, CC BY-NC-4.0)
- CLAUDE.md, PHASES.md, PROJECT_STATUS.md created

---

## 11. Standing reminders

- **Commit and push at every phase boundary**, with this file updated in the same commit
- Never tune on the test split
- Save every raw prediction to CSV, not just aggregate metrics
- Write paper sections as you go — don't leave writing to P6
- Understand every module well enough to explain it in the viva, alone
- Post to arXiv as soon as results are solid (timestamps the work)
- Re-run the competing-work search at the start of P3 and again in P6
- P5 is optional. Don't let it block P6.
