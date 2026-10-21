# Model card — TextGuard v1 and v2

TextGuard is a **training-free, inference-time, black-box claim-vs-evidence consistency gate** for
forecasters that read text alongside a numeric series. It checks what a report claims against what
the series' own recent numbers show, and lets through less of the text's effect when the two
disagree. It is not a forecaster and is not trained. It wraps one.

```
A = model.predict(numbers, None)       # numbers only
B = model.predict(numbers, text)       # numbers + text
g = min over the text's claims of (1 − confidence × evidence strength) where claim and evidence disagree, else 1
final = A + g·(B − A)                  # v2: then the text's effect is capped at 1σ
```

Code: v1 `src/textguard/textguard.py`, v2 `src/textguard/v2.py`; public wrappers `textguard.TextGuardV1`
/ `TextGuardV2` (frozen settings). Every number below comes from `PROJECT_STATUS.md` §9. The tests
were pre-registered (§10b, §10f, §10g), with smallest effects of interest DSS 0.05σ, FBR 5 pp,
MSE 10%.

## Versions and frozen settings

| | TextGuard v1 (P6) | TextGuard v2 (R4) |
|---|---|---|
| Judge | Qwen2.5-1.5B-Instruct @ `989aa79`, one question about the **whole report**: "did ⟨subject⟩ go up or down?" Scored answers; confidence = renormalised probability | Same model. **Every sentence / headline** is judged on its own, with separate *change*, *outlook* and *level* questions (V3 consolidation rules) |
| Evidence | OLS trend over the last w values (change claims); latest value vs the stated baseline (level claims) | v1's, plus **year-over-year** claims checked against y − y₋P from the series' **history**; cumulative / multi-month claims left unchecked |
| Settings | soft gate τ 0; w = max(3, ¼ · lookback) | w = max(3, ½ · lookback), confidence floor 0.8, τ 0, **magnitude cap 1σ** |
| Tuned on | original domains' val split (Model 5, flip) | original domains' val split only (72 configs; V3 chosen after the as-built V0 missed the FBR budget, disclosed) |
| Frozen file | `results/summary_p6_frozen.json` | `results/summary_r4_v2_frozen.json` |

## Requirements (black box)

* Two calls per forecast: `predict(numbers, None)` and `predict(numbers, text)`. No gradients,
  weights, embeddings or retraining.
* σ: the std of the series on its training period, the scale of "strong evidence".
* **v2 also needs `history`**: the series before the lookback window, for year-over-year claims (≥ P
  periods plus 3 past P-differences; P = 52 weekly, 12 monthly). Without it, year-over-year claims
  get no evidence and are not gated.
* A judge: Qwen2.5-1.5B on a **GPU** (a free T4 handles a few questions per second; a CPU is ~50×
  slower). Any object returning claims can replace it (`DirectionJudge` protocol).

## Intended use

* Defending a forecaster that reads report text (an LLM prompted with the series and the report,
  or a fusion model) against **visible, minimal edits that invert or bend a report's meaning**,
  where the series has meaningful recent trends to check claims against.
* **Auditing** a text-fed forecaster: `textguard.audit` measures how far lies move it (DSS, flip
  rate) and what each defence costs on genuine reports (False Block Rate, MSE).
* Research on text attacks and defences for multimodal forecasting.

## Results summary

**v1** (original domains, test split, Model 5 = Qwen2.5-1.5B early fusion):
* Against P3's direction flip, pooled DSS **+0.451σ → −0.042σ**, significantly lower than
  always_trust and clip. **But** the FBR was **14.4%** (over the 10% budget), the flip-rate
  reduction was not significant, and clip was better on accuracy, FBR and exaggeration. It is a
  trade-off, not a win.
* It did not reduce the push on R1's FinBERT late-fusion head (+0.049σ → +0.053σ). On Qwen2.5-7B:
  −0.112σ (Holm n.s.).

**v2 vs v1 on held-out Energy + FNSPID** (R4, tested once; Model 5 primary). Pre-registered reading:
**"v2 buys protection at a cost (False Block Rate)"**.

| Model 5, pooled | Undefended | v1 | v2 |
|---|---|---|---|
| A2 (appended forward-looking lie), DSS σ | +0.180 | +0.096 | **+0.021** (88% cut; lower than v1 by 0.075σ, Holm p < 0.001) |
| Flip, DSS σ | +0.054 | +0.029 | +0.016 |
| False Block Rate on genuine reports | 0 | **7.0%** | **18.4%** (Energy 7.1%, FNSPID 22.4%) |

On validation v2 had blocked fewer genuine reports than v1 (6.3% vs 11.3%). Held out, that ordering
reversed. Real-text MSE was lower under v2, partly because genuine FNSPID headlines *hurt* Model 5
(MSE 1.06 vs 0.48σ² without them), so damping text helps there whatever the claims say.

## Known failures (read these before deploying)

1. **v1 is evaded by A2.** One appended sentence predicting a reversal ("… is expected to fall next
   month") moved Model 5 +0.456σ and Qwen2.5-7B +0.403σ, as much as the full flip. v1's judge read
   the genuine past claim (91%), so the gate passed the report (R3). On held-out Energy, v1 is again
   evaded (the judge reads the genuine claim 99.4% of the time).
2. **v2 blocks many genuine FNSPID headline bundles: 22.4%.** On 50 hand-labelled FNSPID headlines
   the judge gives a direction to **11 of 33 neutral** ones. It never reversed a direction. With ~10
   judged units per bundle and g = the minimum, one spurious claim against a weak weekly trend
   blocks the bundle. Expect the same on any text made of many short, mostly neutral items.
3. **No improvement transfers to Qwen2.5-7B.** On the held-out domains, A2 still fools it under
   both v1 (+0.064σ) and v2 (+0.054σ). Both are *evaded*, and v2 still blocks 16.3% of genuine
   reports.
4. **v2 does not "stop" A2 by the pre-registered vocabulary.** Its pooled residual (+0.021σ, CI
   0.004–0.037) is below the 0.05σ smallest effect but significantly above zero. It reads "reduces;
   residual not shown to be below the SESOI".
5. **Exaggeration**: a claim in the right direction but overstated is invisible to the gate (it
   agrees with the trend). v2's 1σ cap limits the damage but does not detect it.
6. **Genuine regime changes** (COVID 2020): true reports that contradict the pre-shock trend are the
   gate's false blocks by design.
7. **Validation overstated v2**, as stated before the test. Expect held-out performance below
   validation performance on any new domain.

## Out of scope / not for

* **Not a fact-checker.** It checks claims against the series' own past, not against the world. A
  lie consistent with the recent trend passes (R3's A1 is built on this).
* **Not a guarantee.** Adaptive attackers who know the defence were only partly tested (R3: A1–A3).
  Fluent LLM-written fakes and coordinated campaigns (R5, R6) were **not run** (future work).
* Not for imperceptible attacks (homoglyphs, hidden text). That is a different threat model
  (Rizvani et al., 2026).
* Not for numeric-channel attacks on the series itself.
* Not a reason to deploy a text-fed forecaster that has not been audited. Not for decisions about
  individuals, or for trading without a human in the loop.
* The judge's prompts name one series at a time. Multi-target text needs one judge per target.

## Ethics

Attack texts are synthetic and research-only (ETHICS.md). The judge is an LLM with the biases and
failure modes of a 1.5B model. Its known failure on neutral financial headlines is measured above.
