# Model card: TextGuard v1 and v2

TextGuard sits in front of a forecaster that reads text with its numbers. It is not a forecaster
and it is never trained. It asks the wrapped model for two forecasts, one without the report (A)
and one with it (B), asks a small LLM what the report claims, compares the claim with the recent
numbers, and keeps less of B − A when they disagree.

```
A = model.predict(numbers, None)
B = model.predict(numbers, text)
g = 1 if every claim fits the numbers, else the smallest (1 − confidence × evidence strength)
final = A + g·(B − A)              # v2 then limits the change to ±1σ
```

Code: `src/textguard/textguard.py` (v1), `src/textguard/v2.py` (v2); wrappers with the frozen
settings: `textguard.TextGuardV1`, `textguard.TextGuardV2`. Numbers below: `PROJECT_STATUS.md` §9.
Tests were planned in advance (§10b, §10f, §10g); smallest effects that count: DSS 0.05σ, block
rate 5 points, MSE 10%.

## Settings

| | v1 (P6) | v2 (R4) |
|---|---|---|
| Judge | Qwen2.5-1.5B-Instruct @ `989aa79`; one question about the whole report ("did ⟨subject⟩ go up or down?"); the three answers are scored and the best one's share is the confidence | same model; each sentence or headline asked separately, with *change*, *outlook* and *level* questions (V3 rules) |
| Evidence | straight-line trend over the last w values; for level claims, latest value vs the stated baseline | as v1, plus year-over-year claims compared with y − y₋P from older data; cumulative and multi-month claims not checked |
| Settings | τ 0 (soft); w = max(3, ¼ of the lookback) | w = max(3, ½ of the lookback); ignore claims with confidence < 0.8; τ 0; cap 1σ |
| Tuned on | validation split of the original domains (Model 5, flip) | the same split only; 72 settings tried; V3 picked after the first build (V0) blocked too much (disclosed) |
| Saved in | `results/summary_p6_frozen.json` | `results/summary_r4_v2_frozen.json` |

## What it needs

* Two forecasts per window: `predict(numbers, None)` and `predict(numbers, text)`. Nothing else
  from the model.
* σ, the series' standard deviation over its training period.
* v2 only: `history`, the values before the lookback (at least P + 3 past yearly differences;
  P = 52 weekly, 12 monthly). Without it, year-over-year claims are not checked.
* The judge, Qwen2.5-1.5B. A GPU is best; a CPU works slowly (about 47 min for the 110-question
  notebook example on Kaggle, download included; `results/r7_toolkit_kaggle/`). Anything that
  follows the `DirectionJudge` protocol can replace it.

## Good uses

* Protecting a text-reading forecaster (an LLM given the series and a report, or a fusion model)
  from small, visible edits that reverse or bend what a report says, when the series has recent
  trends worth checking against.
* Auditing such a forecaster with `textguard.audit`: how far fakes move it (DSS, flip rate) and
  what each defence costs on real reports (block rate, MSE).
* Research on text attacks and defences in forecasting.

## Results

**v1**, Time-MMD test split, Model 5 (Qwen2.5-1.5B with the report in the prompt):

* Direction flip: DSS +0.451σ → −0.042σ, significantly below always trusting the text and below
  clipping. But 14.4% of real reports were blocked (limit 10%), fewer flipped forecasts was not
  significant, and clipping did better on accuracy, blocks and exaggeration.
* FinBERT late-fusion head (R1): no reduction (+0.049σ → +0.053σ). Qwen2.5-7B: −0.112σ, not
  significant after correction.

**v2 vs v1, unseen Energy and FNSPID data** (R4, run once). Planned reading: *v2 buys protection at
a cost (FBR)*.

| Model 5, pooled | No defence | v1 | v2 |
|---|---|---|---|
| Added-sentence attack (A2), DSS σ | +0.180 | +0.096 | **+0.021** (below v1 by 0.075σ, Holm p < 0.001) |
| Flip, DSS σ | +0.054 | +0.029 | +0.016 |
| Real reports blocked | 0 | **7.0%** | **18.4%** (Energy 7.1%, FNSPID 22.4%) |

On validation v2 had blocked less than v1 (6.3% vs 11.3%); on new data it was the other way round.
v2 had lower MSE on real text, partly because real FNSPID headlines make Model 5 worse (MSE 1.06
with them, 0.48σ² without), so damping the text helps whatever it says.

## Known failures

1. **v1 falls to one added sentence.** "… is expected to fall next month" moved Model 5 +0.456σ and
   Qwen2.5-7B +0.403σ; v1's judge answered from the real past claim (91%) and let the report
   through. On Energy the judge did the same 99.4% of the time.
2. **v2 blocks 22.4% of real FNSPID headline bundles.** In a 50-headline hand check the judge found
   a direction in 11 of 33 neutral headlines (it never reversed one). A bundle has about ten
   judged pieces and the gate takes the lowest, so one bad reading can block it.
3. **Nothing carried over to Qwen2.5-7B** on new data: the A2 attack still worked under v1
   (+0.064σ) and v2 (+0.054σ), and v2 blocked 16.3% of real reports.
4. **v2 does not count as stopping A2** under the planned rules: +0.021σ (CI 0.004–0.037) is under
   the 0.05σ threshold but above zero, so the label is *reduces; residual not shown to be below the
   SESOI*.
5. **Exaggeration** (right direction, size overstated) looks fine to the gate. v2's cap limits the
   harm but doesn't flag it.
6. **Real shocks** (COVID 2020): true reports that break the old trend get blocked.
7. **Validation made v2 look better than it was.** Expect worse numbers on any new domain.

## Not for

* Fact-checking. It compares claims with the series' own past only; a lie that fits the trend
  passes (that is how A1 works).
* Guarantees. Only three adaptive attacks were tried. LLM-written fakes and coordinated campaigns
  (R5, R6) were not run.
* Invisible attacks (homoglyphs, hidden text; Rizvani et al., 2026) or attacks on the numbers.
* Skipping an audit, decisions about individuals, or trading without a person in the loop.
* Text about several series at once: use one judge per series.

## Ethics

Attack texts are synthetic and for research only (`ETHICS.md`). The judge is a 1.5B LLM with the
usual biases; its weakness on neutral financial headlines is measured above.
