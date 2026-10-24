"""Paper content: Section V (defence design and calibration) and Section VI (results).

Table cells come from results/summary_*.csv through V(); prose numbers from PROJECT_STATUS §9
and the same tables (checked by check_numbers.py). Pre-registered verdict wording is quoted as
the analysis code wrote it.
"""

from build_paper import FIG, MINUS, V, J, ci, pct, pick, plain, signed


def pval(p: float) -> str:
    if p >= 0.995:
        return "1"
    if p >= 0.01:
        return f"{p:.2f}"
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"{m}×10^{{{str(int(e)).replace('-', MINUS)}}}"


def n0(x):
    return f"{x:.0f}"


def write(d):
    _section_v(d)
    _section_vi(d)


# ===========================================================================================
def _section_v(d):
    d.h1("V. Defence Design and Calibration Evidence")
    d.h2("A. TextGuard v1 (P6)")
    d.p("All tuning used the validation split; the test split was opened once, at the frozen "
        "settings. The grid crossed the trend-window fraction f ∈ {1/4, 1/2, 1} of the "
        "lookback (w ∈ {3, 4, 8} months or {6, 12, 24} weeks) with the block threshold "
        "τ ∈ {0, 0.1, …, 1}, 33 configurations in all; clip was tuned over "
        "k ∈ {0.05, 0.1, 0.25, 0.5, 1}. The pre-registered rule took, for Model 5 under the "
        "flip on pooled validation windows, the lowest DSS among configurations with FBR at most 10%. "
        f"No TextGuard configuration met that budget: the lowest validation FBR was "
        f"{J('summary_p6_frozen.json', 'textguard/val_fbr', pct)}. The pre-registered fallback "
        "(lowest FBR) left a tie that Amendment 2 resolved with the rule’s own tie-breaks, "
        "smaller τ and then smaller w, giving the soft gate (τ = 0) with w = 3 months or "
        f"6 weeks, at a validation DSS of {J('summary_p6_frozen.json', 'textguard/val_dss')}. Clip met "
        f"the budget at k = 0.5 (FBR {J('summary_p6_frozen.json', 'clip/val_fbr', pct)}, DSS "
        f"{J('summary_p6_frozen.json', 'clip/val_dss')}). Amendment 1, written before any judge "
        "output existed, predicted a higher FBR on Traffic and Economy, whose claims use reference "
        "frames that a short trend cannot verify (year-over-year changes; a different series).")
    d.p("Before tuning, the judge was compared with 50 reports labelled by hand before the test "
        f"period. Its change answer agreed on {V('summary_p6_judge_human.csv', 'accuracy', pct, label_set='main_50', question='change', domain='pooled')} "
        f"(κ = {V('summary_p6_judge_human.csv', 'kappa', lambda x: plain(x, 2), label_set='main_50', question='change', domain='pooled')}), "
        f"but only {V('summary_p6_judge_human.csv', 'accuracy', lambda x: pct(x, 0), label_set='main_50', question='change', domain='Health_US')} "
        "on Health_US, where it reads statements of level (“ILI was 0.9%, below the national "
        "baseline”) as changes. In every report where the annotator and the judge each saw a "
        "direction, the two matched, and against the known direction of the fakes it was right on 94% of change claims.")
    d.h2("B. TextGuard v2 (R4): From V0 to V3 on Validation")
    d.p("v2 was developed on the validation split of the original domains only; no held-out and no "
        "original test data was read. Its grid (fixed before any v2 answer existed) crossed "
        "f ∈ {1/4, 1/2, 1}, c₀ ∈ {0, 0.6, 0.8}, τ ∈ {0, 0.5} and "
        "k ∈ {0.25, 0.5, 1, ∞}, 72 configurations. The selection rule kept configurations "
        "with FBR at most 10% and minimised the mean, over flip, A2, exaggeration and A3, of the "
        "absolute pooled DSS, so that over-correction counts as error. {T:v2dev} shows the path. As "
        "built (V0), no configuration met the budget. Of V0’s 27 false blocks, 20 came from an "
        "outlook claim that merely restated a sentence’s past fact and, checked against the "
        "short trend, reintroduced the level and year-over-year errors that frame matching had "
        "removed; 3 came from a pattern bug, 3 from off-subject Economy sentences and 1 from a level "
        "read as a change. V3 drops an outlook claim identical to its unit’s change claim and "
        "met the budget in every domain. V4, which also requires a unit to name the domain’s "
        "subject, scored slightly better on the objective but blocked 18.5% of genuine Traffic "
        "reports, so V3 was frozen: f = 1/2 (w = 4 months, 12 weeks), c₀ = 0.8, τ = 0, "
        "k = 1σ. The judge log explains where the gain comes from: the 1.5B judge answers the "
        "outlook question like the change question on 94–100% of units, so A2 is caught "
        "because the appended sentence is judged separately, not because of the new question.")
    _v2dev_table(d)
    d.p("This iteration is disclosed as development, not as evidence: V3 was chosen after V0 had "
        "missed the budget, and five variants were compared on 159 genuine validation texts, so "
        "validation overstates v2. We wrote that expectation into the R4 plan before the held-out "
        "test, together with the frozen settings, the held-out domains and the attack texts.",
        runin="Disclosure")


def _v2dev_table(d):
    F = "summary_r4_val_variants_v2.csv"
    C = "summary_r4_val_compare_v2.csv"
    names = {"V0": "V0 as built", "V1": "+ pattern bug fixed", "V2": "+ baseline sentence = level only",
             "V3": "+ duplicate outlook dropped (frozen)", "V4": "+ unit must name the subject"}
    body = []
    for v, label in names.items():
        q = dict(variant=v, target="M5", cell="pooled")
        body.append([label, V(F, "fbr", pct, **q),
                     V(F, "fbr", lambda x: pct(x), variant=v, target="M5", cell="Traffic"),
                     V(F, "dss_flipped", **q), V(F, "dss_a2_forward", **q),
                     V(F, "dss_exaggerated", **q), V(F, "dss_a3_magnitude", **q),
                     V(F, "obj", plain, **q)])
    for dfn, label in (("undefended", "Undefended (reference)"), ("TextGuard v1", "TextGuard v1 (reference)")):
        q = dict(target="M5", defence=dfn, cell="pooled")
        body.append([label, V(C, "fbr", pct, **q), V(C, "fbr", pct, target="M5", defence=dfn, cell="Traffic"),
                     V(C, "dss_flipped", **q), V(C, "dss_a2_forward", **q),
                     V(C, "dss_exaggerated", **q), V(C, "dss_a3_magnitude", **q), "–"])
    d.table("v2dev", "TextGuard v2 Development Path on Validation (Model 5, Pooled)",
            ["Variant (cumulative)", "FBR", "FBR Traffic", "DSS flip", "DSS A2", "DSS exagg.",
             "DSS A3", "Objective"], body,
            widths=[2.2, 0.75, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8], wide=True,
            note="Each variant at the configuration its own grid search selects. Objective = mean "
                 "absolute pooled DSS over the four attacks (lower is better). FBR budget 10%.")


# ===========================================================================================
def _section_vi(d):
    d.h1("VI. Results")
    _p5(d)
    _p6(d)
    _r1(d)
    _r2(d)
    _r3(d)
    _r4(d)


def _p5(d):
    P = "summary_p5_attack.csv"
    d.h2("A. Which Forecasters the Flip Fools (P5)")
    d.p("All twelve pre-registered screening tests reject: text moves Models 3, 4 and 5 in every "
        "domain, Models 3 and 4 in every window with a report and Model 5 in 69% of them, so all "
        "three are attack targets. The controls show what kind of reading this is. An irrelevant "
        "report moves each text model about as far as the real one, and Model 5 further (pooled "
        "sensitivity 0.911σ for another domain’s report against 0.388σ for its own), "
        "yet the controls do not land where the real report does. The models respond to what a "
        "report says but cannot tell a relevant report from an irrelevant one. Text does not "
        "improve accuracy measurably: the pooled usage gap MSE(empty) − MSE(real) is "
        "−0.099 for Model 3 (text hurts, mostly on Traffic), +0.007 for Model 4 and "
        "−0.133 [−0.485, +0.266] for Model 5.")
    body = []
    for short, model, fusion in (("M3", "DLinear+BERT", "late"), ("M4", "PatchTST+BERT", "late"),
                                 ("M5", "Qwen2.5-1.5B", "early")):
        for cond, lab in (("flipped", "flip"), ("exaggerated", "exagg.")):
            q = dict(short=short, condition=cond, domain="pooled")
            body.append([f"{short} ({fusion})", lab, V(P, "n", n0, **q),
                         V(P, "dss", **q) + " " + ci(P, "dss_ci_low", "dss_ci_high", **q),
                         V(P, "dss_p_holm", pval, **q), V(P, "flip_rate", pct, **q),
                         V(P, "toward", n0, **q) + " / " + V(P, "away", n0, **q)])
    d.table("p5", "Attack on the Original Domains (Test, Pooled)",
            ["Model", "Attack", "n", "DSS (σ) [95% CI]", "Holm p", "Flip Rate",
             "Toward / away"], body, widths=[1.15, 0.6, 0.45, 2.0, 0.95, 0.75, 0.9], wide=True,
            note="Models 1–2 are text-blind by construction (DSS and Flip Rate exactly 0). Holm "
                 "over the 76-test P5 family. Toward / away: windows whose forecast direction turned "
                 "toward / away from the claim.")
    d.p("The reversal steers Model 5 and nothing else ({T:p5}, {F:p5dss}). Model 5 moves "
        "+0.457σ toward the lie and turns direction in 23.6% of attacked windows, 58 toward the "
        "claim against 4 away. The BERT models move by at most 0.005σ and flip at most three "
        "windows. Model 5’s effect is +0.371σ on SocialGood, +0.328σ on Traffic and "
        "+0.828σ on Health_US; on Economy it is +0.031σ and not significant, because the "
        "Qwen gives the identical answer with the fake in 39 of 70 windows. The pre-registered fusion "
        "contrast rejects against both late-fusion models (+0.461σ and +0.454σ), the "
        "rule for “fusion type matters” under the flip. Model 5 is, however, also the only "
        "LLM, the only one used without training on these series and the only one that sees the "
        "figures inside the report. What the data support is that the LLM forecaster is "
        "vulnerable and the BERT-based ones are not; they do not show that the fusion type is the "
        "cause (Section VI-C tests this).", runin="Attack")
    d.figure("p5dss", FIG / "p5_dss.png",
             "DSS of Models 1–5 under the flip and the exaggeration (test, pooled over four "
             "domains, attacked windows). Whiskers: 95% stratified bootstrap CI.")
    d.p("The flip makes Model 5’s error roughly twice as large (MSE +92%, almost all on "
        "Health_US). Exaggeration pushes no model in the claimed direction (Model 5: −0.098σ, "
        "Holm p = 1, median per-window effect zero in every domain), yet it raises Model 5’s "
        "error by 54%, spreading the forecasts out instead of pushing them one way. The BERT heads are immune "
        "for a mechanical reason. After the per-dimension standardisation their heads see, the "
        "cosine between a report and its reversal is 0.968, against 0.078 for an unrelated report "
        "from the same domain; the reversal moves the head input only about a fifth as far as "
        "swapping the report.", runin="Damage and mechanism")
    d.figure("p5flip", FIG / "p5_flip_rate.png",
             "Flip Rate under the two attacks (test, pooled): the share of attacked windows whose "
             "forecast direction turned toward the claim. Whiskers: 95% bootstrap CI.")
    d.p("Without 2020 the Model 5 effect is +0.437σ; on the three change-reversal domains "
        "alone, where no stated number is edited, Model 5 still moves +0.190σ. Removing the "
        "attacked texts that a later consistency scan flags (Section VI-E) gives +0.503σ "
        "(Appendix C).", runin="Robustness")


def _p6(d):
    P = "summary_p6_operating.csv"
    d.h2("B. TextGuard v1 on the Flip (P6)")
    names = [("none", "Undefended (raw)"), ("always_trust", "always_trust"),
             ("never_trust", "never_trust"), ("clip k=0.5", "clip, k = 0.5"),
             ("TextGuard τ=0", "**TextGuard v1 (soft gate)**"),
             ("TextGuard block τ=0.5", "TextGuard, block τ = 0.5*"),
             ("TextGuard-L τ=0", "TextGuard-L*"),
             ("TextGuard oracle-judge τ=0", "Oracle judge†")]
    body = []
    for dfn, lab in names:
        q = dict(short="M5", attack="flipped", defence=dfn, domain="pooled")
        body.append([lab, V(P, "dss", **q) + " " + ci(P, "dss_lo", "dss_hi", **q),
                     V(P, "flip", pct, **q), V(P, "fbr", pct, **q), V(P, "mse_real", plain, **q)])
    d.table("p6", "TextGuard v1 and Baselines on Model 5 (Flip, Test, Pooled)",
            ["Defence", "DSS (σ) [95% CI]", "Flip Rate", "FBR", "Real-text MSE"], body,
            widths=[1.9, 1.9, 0.8, 0.7, 0.9], wide=True,
            note="*Exploratory (Amendment 2). †Ablation using the rules’ known directions. "
                 "Undefended uses P5’s NaN pairing (n = 246); the others n = 251.")
    d.p("TextGuard removes Model 5’s pooled manipulation ({T:p6}): DSS falls from "
        "+0.451σ under always_trust to −0.042σ, and the pre-registered contrasts "
        "against always_trust (−0.493σ, Holm p = 1.4×10^{−14}) and against clip "
        "(−0.167σ, Holm p = 4.1×10^{−8}) both reject; 7 of the 20 tests in the "
        "family reject, all with the claimed sign. It does so at an FBR of 14.4%, above the 10% "
        "budget it had already missed on validation. Its reduction of the Flip Rate (23.1% to 17.9%) "
        "does not survive Holm correction, and on genuine-text accuracy no difference from "
        "never_trust is detected. By the pre-registered exit criterion the result is *partly met*: "
        "no defence dominates, and {F:p6trade} shows the trade-off along each defence’s "
        "threshold.")
    d.figure("p6trade", FIG / "p6_tradeoff.png",
             "Manipulation against genuine text blocked for Model 5 under the flip (test). Top: DSS; "
             "bottom: Flip Rate; both against the FBR, pooled and per domain. Curves sweep "
             "TextGuard’s τ and clip’s k; large markers are the validation-chosen "
             "operating points with 95% CIs; diamonds are always and never trusting the text.",
             wide=True)
    d.p("The pooled gain sits mostly in Health_US (−0.210σ against +0.828σ "
        "undefended). There the negative value means that the fake has stopped influencing the "
        "output; it is not an additional gain. Genuine Health_US reports already shift Model 5 in "
        "the direction the fake asserts (+0.29σ), so once the fake is held back and the genuine "
        "report let through, the difference turns negative. On the three change-reversal domains TextGuard leaves "
        "+0.074σ [+0.010, +0.142].", runin="Where the gain comes from")
    d.p("False blocks have two separable sources ({F:p6gate}). On Health_US they come from the "
        "judge: all 15 false blocks there arise from the change question applied to level "
        "statements, and the exploratory TextGuard-L, which keeps only the level claim, removes "
        "them (pooled FBR 9.4%). On Traffic (32.1%) and Economy (14.3%) they come from the "
        "evidence: with the rules’ own directions in place of the judge, the oracle blocks as "
        "much or more (30.2% and 18.6%), because year-over-year claims and claims about a "
        "different series cannot be checked against a three-month trend. Amendment 1 predicted "
        "both domains in advance.", runin="Why genuine reports are blocked")
    d.figure("p6gate", FIG / "p6_gate_distribution.png",
             "Cumulative distribution of TextGuard v1’s gate on genuine (solid) and reversed "
             "(dashed) test reports. The height of the solid curve at g = 0.5 is the False Block "
             "Rate; on Traffic the two curves nearly coincide.", wide=True)
    d.p("Of all defences, only clip beats never_trust on accuracy with genuine text (0.965 against "
        "1.045), and it removes most of the exaggeration’s damage (+0.057 against TextGuard’s "
        "+0.422), which TextGuard cannot see because an exaggeration keeps the true direction. On "
        "the direction-blind Model 3, TextGuard leaves accuracy intact (real-text MSE −0.035) "
        "but itself introduces a slight directional shift (Flip Rate 0.4% to 5.2%), since a genuine "
        "report and its reversal receive different gates.", runin="Other costs")


def _r1(d):
    P5, R1, M = "summary_p5_attack.csv", "summary_r1_attack.csv", "summary_r1_mechanism_val.csv"
    d.h2("C. Is It the LLM or Early Fusion? (R1)")
    d.p("To separate fusion type from “is an LLM” we hold the LLM fixed: the same "
        "Qwen2.5-1.5B snapshot becomes a frozen, mean-pooled text encoder for late-fusion heads "
        "identical to Models 3 and 4 (M3-Q, M4-Q). FinBERT, a BERT fine-tuned for financial "
        "sentiment, gives two more heads (M3-F, M4-F). A pre-registered check on validation, made "
        "before any head was trained, measured how far a reversal moves each encoder’s "
        "standardised vector relative to swapping the report: Qwen’s pooled embedding is as "
        "direction-blind as BERT’s, while FinBERT’s moves about three times as far "
        "({T:r1}).")
    ratio = {"BERT": V(M, "ratio_to_bert", lambda x: f"{x:.2f}", encoder="BERT", split="val",
                       attack="flipped", pair="real vs flipped", domain="pooled"),
             "Qwen": V(M, "ratio_to_bert", lambda x: f"{x:.2f}", encoder="Qwen", split="val",
                       attack="flipped", pair="real vs flipped", domain="pooled"),
             "FinBERT": V(M, "ratio_to_bert", lambda x: f"{x:.2f}", encoder="FinBERT", split="val",
                          attack="flipped", pair="real vs flipped", domain="pooled")}
    body = []
    for f_, short, enc in ((P5, "M3", "BERT"), (P5, "M4", "BERT"), (R1, "M3-Q", "Qwen"),
                           (R1, "M4-Q", "Qwen"), (R1, "M3-F", "FinBERT"), (R1, "M4-F", "FinBERT"),
                           (P5, "M5", None)):
        q = dict(short=short, condition="flipped", domain="pooled")
        p = "(P5)" if f_ == P5 else V(f_, "dss_p_holm", pval, **q)
        body.append([short, (enc + " (pooled)") if enc else "Qwen (in prompt)",
                     ratio[enc] if enc else "–",
                     V(f_, "dss", **q) + " " + ci(f_, "dss_ci_low", "dss_ci_high", **q), p,
                     V(f_, "flip_rate", pct, **q),
                     V(f_, "toward", n0, **q) + " / " + V(f_, "away", n0, **q)])
    d.table("r1", "Encoder Swap: Flip Attack (Test, Pooled)",
            ["Model", "Encoder", "Flip move × BERT (val)", "DSS (σ) [95% CI]", "Holm p",
             "Flip Rate", "Toward / away"], body,
            widths=[0.65, 1.2, 0.95, 2.0, 0.95, 0.7, 0.85], wide=True,
            note="Holm over the 108-test R1 family; M3, M4 and M5 rows are the P5 predictions, "
                 "unchanged.")
    d.p("By the pre-registered rule M4-Q is fooled and the early-versus-late contrast rejects for "
        "both Qwen heads, which yields reading R1-a, “early fusion amplifies”; we report "
        "it as specified. Its magnitude needs to travel with it. The rule fires on +0.0075σ, "
        "about a sixtieth of Model 5’s effect, with a non-significant Flip Rate, and the plan "
        "fixed no smallest effect of interest. Read with the mechanism check, R1 compares reading a "
        "report inside the prompt with reading a pooled vector of the same LLM: moving the report "
        "out of the prompt removes about 98% of the manipulation. The FinBERT heads are the "
        "positive control. When the embedding carries direction, late-fusion heads trained only for "
        "accuracy follow it (M3-F +0.049σ, 19 flips toward against 1 away), roughly ten times "
        "less than Model 5 but clearly steered. Whether late fusion resists depends on the encoder "
        "rather than on the architecture. TextGuard v1, applied as frozen, did not reduce M3-F "
        "(+0.049σ to +0.053σ). From R2 on, every plan states a smallest effect of interest.")


def _r2(d):
    A, M = "summary_r2_attack.csv", "summary_r2_accuracy_all.csv"
    d.h2("D. Does It Generalise Across Models and Prompts? (R2)")
    verdict = {"M5": "(reference)", "M5-c1": "fooled", "M5-c2": "fooled", "M5-sk": "fooled",
               "Q7": "fooled", "P35": "bounded"}
    label = {"M5": "Model 5 (Qwen2.5-1.5B)", "M5-c1": " + report first", "M5-c2": " + reworded",
             "M5-sk": " + sceptical line", "Q7": "Qwen2.5-7B (4-bit)", "P35": "Phi-3.5-mini"}
    body = []
    for short in ("M5", "M5-c1", "M5-c2", "M5-sk", "Q7", "P35"):
        q = dict(short=short, condition="flipped", domain="pooled")
        body.append([label[short],
                     V(M, "mse", plain, short=short, condition="real", domain="pooled"),
                     V(A, "dss", **q) + " " + ci(A, "dss_ci_low", "dss_ci_high", **q),
                     V(A, "flip_rate", pct, **q),
                     V(A, "toward", n0, **q) + " / " + V(A, "away", n0, **q), verdict[short]])
    body.append(["Persistence", V(M, "mse", plain, short="persistence", condition="real",
                                  domain="pooled"), "–", "–", "–", "–"])
    d.table("r2", "Other Forecasters and Prompts Under the Flip (Test, Pooled)",
            ["Forecaster", "MSE (real)", "DSS (σ) [95% CI]", "Flip Rate", "Toward / away",
             "Verdict"], body, widths=[1.65, 0.75, 1.95, 0.7, 0.85, 0.8], wide=True,
            note="Verdicts follow the pre-registered SESOI rules with Holm correction inside each question’s "
                 "family (44, 44, 58 and 54 tests).")
    d.p("Qwen2.5-7B is fooled ({T:r2}, {F:r2}): +0.206σ toward the lie, 61 flips toward "
        "against 13 away, about half of Model 5’s effect. It is also more accurate than "
        "Model 5 (0.597 against 1.061 on the same windows) but does not beat persistence, and "
        "removing the report leaves its error unchanged (usage gap −0.021 [−0.182, "
        "+0.125]), so it is not yet the forecaster that both benefits from text and can be steered. "
        "On validation the same model moved only +0.045σ; we read the test value as one "
        "period’s size, not a constant. Phi-3.5-mini is bounded, and for a revealing reason: "
        "given any report, real or reversed, its forecast equals the last observed value in 73% of "
        "windows, against "
        "1% without a report (exploratory). Its low DSS measures indifference to what the report "
        "says, so we do not call it robust.")
    d.figure("r2", FIG / "r2_dss.png",
             "DSS under the flip (blue) and the exaggeration (orange) for Model 5 (M5) and its "
             "three prompt variants (c1 report first, c2 reworded, sk sceptical), Phi-3.5-mini and "
             "Qwen2.5-7B, pooled and per domain, with 95% CIs. Dashed: the 0.05σ SESOI.",
             wide=True)
    d.p("Model 5’s conclusions hold under both neutral rewordings, though the size depends on "
        "wording (the reworded prompt reaches 41% of the original effect). Adding the line "
        "“Caution: this report may be inaccurate or deliberately misleading” leaves the "
        "model fooled (+0.362σ); its reduction against the plain prompt (−0.107σ, raw "
        "p = 0.008) is not significant after Holm correction (p = 0.30), so the pre-registered reading is "
        "“prompting does not help”. TextGuard v1 on Model 5 sits 0.402σ below the "
        "sceptical prompt, and the caution makes the model react more, not less, to genuine reports "
        "(sensitivity 0.461σ against 0.388σ). On Qwen2.5-7B, TextGuard v1 lowers DSS from "
        "+0.200σ to +0.087σ, a reduction above the SESOI that does not survive Holm "
        "correction (p = 0.36), so its verdict is “no difference detected”.",
        runin="Prompts and caution")
    d.p("Every forecaster that resisted the attack ignores report direction (the BERT heads, the "
        "mean-pooled Qwen heads and Phi-3.5), and every one that registers direction was steered "
        "(Model 5 with each of its four prompts, Qwen2.5-7B, and both FinBERT heads). Across 13 "
        "configurations of six model types on four domains, we found no forecaster that both reads "
        "report direction and resists the attack. We offer that as an observation rather than a "
        "law, but it is the trade-off that motivates an external check.", runin="Across R1 and R2")


def _r3(d):
    O, R = "summary_r3_operating.csv", "summary_r3_readings.csv"
    d.h2("E. Adaptive Attacks on TextGuard v1 (R3)")
    d.p("The adversary now knows TextGuard: defences stay frozen and only the attacker changes "
        "(A1–A3, Section III-C). The targets are the four models the flip steered: Model 5, "
        "Qwen2.5-7B and the two FinBERT heads. Reading the generated texts by eye revealed "
        "contradictions between sentences that per-sentence checks could not see, so after the model "
        "runs and before any metric we added an automatic scan (Amendment 1) that removes a text "
        "only if its own numbers contradict each other or the edit moved a stated number against "
        "the claim. No genuine report trips it. It drops 18 of the 60 A1 test texts and 30 of the "
        "128 A3 texts, and none of the A2 texts.")
    body = []
    tgt = {"M5": "Model 5", "Q7": "Qwen2.5-7B", "DLinear+FinBERT": "M3-F",
           "PatchTST+FinBERT": "M4-F"}
    for t in ("M5", "Q7", "DLinear+FinBERT", "PatchTST+FinBERT"):
        for att, lab in (("a1_selective", "A1"), ("a2_forward", "A2"), ("a3_magnitude", "A3")):
            if t in ("DLinear+FinBERT", "PatchTST+FinBERT") and att != "a1_selective":
                continue
            q = dict(target=t, attack=att, cell="pooled")
            row = [tgt[t], lab, V(O, "n_dss", n0, defence="none", **q),
                   V(O, "dss", defence="none", **q) + " " + ci(O, "dss_lo", "dss_hi", defence="none", **q),
                   V(O, "dss", defence="TG", **q), V(O, "dss", defence="CL", **q),
                   V(O, "dss", defence="SK", **q) if t in ("M5", "Q7") else "–",
                   pick(R, target=t, attack=att, cell="pooled", defence="TG")["reading"]]
            body.append(row)
    d.table("r3", "Adaptive Attacks Against the Frozen Defences (Test, Pooled)",
            ["Target", "Attack", "n", "Undefended DSS [95% CI]", "TextGuard v1", "Clip",
             "Sceptical", "TextGuard verdict"], body,
            widths=[0.9, 0.5, 0.45, 1.85, 0.8, 0.7, 0.75, 1.6], wide=True,
            note="DSS in σ. Verdicts follow the pre-registered rules with Holm correction inside each "
                 "target’s family (105, 105, 75, 75 tests). Windows after the Amendment 1 scan.")
    d.p("One appended sentence about next period evades TextGuard v1 on both LLMs ({T:r3}, "
        "{F:r3}). It moves Model 5 by +0.456σ, as far as rewriting the whole report "
        "(+0.457σ), and Qwen2.5-7B by +0.403σ; TextGuard leaves +0.400σ and "
        "+0.352σ. Asked what happened, the judge answers from the true past-tense claim in "
        "91.3% of these reports, so the gate discounts 17.0% of the fakes, no more than the 15.4% "
        "of genuine reports for those windows, while the forecaster acts on the forward claim. Clip "
        "is the only defence that reduces A2, to +0.115σ and +0.120σ, still above the "
        "SESOI. The sceptical prompt’s “pushed away” label on Model 5 rests on a "
        "single window in which the model copied a cumulative “2,730.6 billion” from the "
        "report; without it the prompt is as fooled as the plain model (+0.483σ, descriptive).",
        runin="A2")
    d.figure("r3", FIG / "r3_dss.png",
             "Pooled DSS for A1–A3 with no defence and with each frozen defence, with 95% "
             "bootstrap CIs. Dashed: the 0.05σ SESOI. A single window drags the sceptical "
             "prompt’s A2 estimate for Model 5 below zero.", wide=True)
    d.p("A1 passes the gate as designed (7.1% of fakes gated), while TextGuard discounts 40.5% of "
        "the genuine reports for those windows; yet it does not make TextGuard worse. Our pre-stated "
        "expectation had the sign wrong: discounting a genuine report that pushes in its own "
        "direction pulls the protected forecast for the genuine report toward the forecast without "
        "text and narrows the "
        "fake–real gap. On Model 5, A1 is inconclusive on the pre-registered windows "
        "(+0.007σ, n = 42) and fooled under both alternative window sets (+0.087σ with all "
        "texts; +0.082σ with a stricter check, C8, found only after the metrics existed and "
        "therefore reported as sensitivity, not adopted); fewer than 20 texts carry the effect. "
        "A1 does evade TextGuard on the DLinear+FinBERT head (+0.093σ to +0.072σ). A3, "
        "the tenfold magnitude lie, pushes no target.", runin="A1 and A3")


def _r4(d):
    O, Fb = "summary_r4_operating.csv", "summary_r4_fbr.csv"
    d.h2("F. TextGuard v2 on Held-Out Domains (R4)")
    d.p("v2 and v1, both frozen, were evaluated once on Energy and FNSPID against the flip and the "
        "three adaptive attacks, with clip, never_trust and anchor15, our version of Market Signal "
        "Injection’s output projection [@msi], which projects the forecast onto ±15% of the "
        "last value without tuning. Canonicalization corresponds to never_trust here, since the "
        "text to delete is the genuine report. Targets are Model 5 (primary) and Qwen2.5-7B, each "
        "with a 79-test Holm family; all periods with text serve as test data.")
    body = []
    defs = ("none", "TG1", "TG2", "CL", "anchor15")
    for cell in ("Energy", "FNSPID", "pooled"):
        for att, lab in (("flipped", "flip"), ("a1_selective", "A1"), ("a2_forward", "A2"),
                         ("a3_magnitude", "A3")):
            if att == "a3_magnitude" and cell != "Energy":
                continue
            if att == "a1_selective" and cell != "pooled":
                continue
            body.append([cell, lab] + [V(O, "dss", target="M5", attack=att, defence=x, cell=cell)
                                       for x in defs])
    body.append(["Pooled", "A2 (Qwen2.5-7B)"] + [V(O, "dss", target="Q7", attack="a2_forward",
                                                    defence=x, cell="pooled") for x in defs])
    for cell in ("Energy", "FNSPID", "pooled"):
        body.append([f"FBR {cell}", "genuine"] + [V(Fb, "fbr", pct, target="M5", defence=x, cell=cell)
                                                  for x in defs])
    d.table("r4", "Held-Out Test: DSS (σ) and False Block Rate, Model 5",
            ["Cell", "Attack", "Undefended", "TextGuard v1", "TextGuard v2", "Clip", "anchor15"],
            body, widths=[1.0, 1.15, 0.9, 0.95, 0.95, 0.75, 0.8], wide=True,
            note="Pooled = Energy and FNSPID windows with equal weight. A1 per domain and the full "
                 "Qwen2.5-7B grid are in the released tables. FBR = genuine reports with applied "
                 "weight below 0.5.")
    d.p("By the pre-registered rule the pooled reading for Model 5 is “v2 buys protection at "
        "a cost (FBR)” ({T:r4}, {F:r4dss}). On A2, the strongest attack, v2 lowers DSS from "
        "+0.180σ to +0.021σ, below v1 (+0.096σ) by 0.075σ (Holm p < 0.001), "
        "and below clip. Its residual interval [+0.004, +0.037] lies inside the SESOI but excludes "
        "zero, and the plan reserves “bounded” for non-significant estimates, so the "
        "verdict is “reduces; residual not shown to be below the SESOI”. The work is done "
        "by the per-sentence split: v2’s judge gave the appended sentence the direction the "
        "attacker intended in all 345 Energy texts and all 538 FNSPID texts, whereas the cap alone "
        "leaves A2 at +0.133σ. “v2 closes R3’s A2 gap” is not met, because "
        "pooled v1 is no longer evaded: on FNSPID’s short headline bundles v1’s judge "
        "picked up the appended sentence in 54% of the bundles. The flip and A1 are small on held-out data "
        "(pooled +0.054σ and +0.043σ), and no flip or A1 cell separates v2 from v1 by the "
        "SESOI. anchor15 barely touches A2 (+0.150σ).", runin="Protection")
    d.figure("r4dss", FIG / "r4_dss.png",
             "Held-out DSS by attack and defence. Top row: Model 5; bottom row: Qwen2.5-7B; "
             "columns: Energy, FNSPID, pooled; 95% CIs. Dashed: the 0.05σ SESOI. A3 runs on "
             "Energy only.", wide=True)
    d.p("On genuine reports v2’s block rate is 18.4% and v1’s 7.0% (FNSPID 22.4% and 9.0%), a "
        "difference of +11.4 percentage points (exact McNemar, Holm p < 0.001) that reverses the "
        "validation ordering (6.3% against 11.3%). The gate rather than the cap does the blocking: "
        "without the cap v2 still blocks 16.3%, the cap alone 3.3%. A check against 50 FNSPID "
        "headlines labelled by hand (added after the runs, not pre-registered) locates the cause. "
        "v2’s judge does not once invert a direction and detects 16 of the 17 directional "
        "headlines, yet labels 11 of the 33 neutral-marked headlines as directional (agreement 76%, κ = 0.60). A bundle "
        "averages about ten judged units and the gate takes their minimum, so a single misread claim "
        "that disagrees with a weak weekly trend is enough to block the bundle ({F:r4fbr}).", runin="Cost")
    d.figure("r4fbr", FIG / "r4_fbr.png",
             "Share of genuine held-out reports blocked by each defence, with 95% CIs. Dashed: the "
             "pre-registered 10% budget.", wide=True)
    d.p("Genuine FNSPID headlines make Model 5 less accurate (MSE 1.06 with them, 0.48 without, "
        "persistence 0.32), so any damping helps; v2’s real-text MSE (0.60) beats v1’s "
        "(0.93), partly by damping harmful text rather than by checking claims. For Qwen2.5-7B the "
        "reading is “no improvement over v1 shown”: A2 fools it pooled (+0.074σ) and "
        "both versions are evaded (+0.064σ and +0.054σ). One wording error was disclosed "
        "before any output was seen: the Energy prompt calls the all-grades series regular gasoline "
        "(about $0.09 apart); every test compares answers to the same prompt, so the label cannot "
        "bias a pre-registered comparison. Validation overstated v2, as the plan predicted, and the held-out "
        "test showed by how much.", runin="Accuracy and the larger model")
