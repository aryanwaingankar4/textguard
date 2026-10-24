"""Paper content: Sections VII-IX and Appendices A-F.

Appendix A builds its prompt strings by calling the project's own prompt modules, so the paper
shows exactly what the models saw (placeholders replace the data values). Fake texts quoted here
are synthetic and labelled as such.
"""

import sys

from build_paper import ROOT, V, J, pct, pick, plain, signed

sys.path.insert(0, str(ROOT))
from src.models.qwen_prompt import SYSTEM, build_prompt  # noqa: E402
from src.models.r2_prompts import CAUTION  # noqa: E402
from src.textguard import judge_prompt as J1  # noqa: E402
from src.textguard import judge_prompt_v2 as J2  # noqa: E402


def write(d):
    _errors(d)
    _limitations(d)
    _conclusion(d)


def write_appendices(d):
    """After the References, as in the IEEE template."""
    _appendices(d)


# ===========================================================================================
def _errors(d):
    d.h1("VII. Error Analysis")
    d.p("Aggregates hide how TextGuard fails. {T:errors} collects the failure types we observed, "
        "each with a concrete case; quoted fakes are synthetic research text. They fall into four "
        "groups: errors of the judge, errors of the evidence, limits of blocking itself, and "
        "properties of the attack texts.")
    rows = [
        ["1", "Judge reads a direction into neutral text (v2)",
         "FNSPID headlines labelled neutral by hand, e.g. routine analyst notes",
         "A direction for 11 of 33 neutral headlines; with the minimum over about ten units per "
         "bundle, a single misread claim can block all of it", "FNSPID FBR 22.4%"],
        ["2", "Level statement read as a change (v1)",
         "Health_US: “ILI was 0.9%, below the national baseline” answered "
         "“down” at c = 0.98",
         "The change question checks a claim the report does not make; every Health_US false block "
         "comes from it", "17% agreement on Health_US; TextGuard-L FBR 9.4% pooled"],
        ["3", "Confidence floor drops a clear claim (v2)",
         "NFLX 2018-11-23: “Netflix shares trading down 4.3% after competitor Disney provided "
         "additional details …”",
         "Read correctly as down, but at c = 0.77 < c₀ = 0.8, so never checked", "Single window"],
        ["4", "Genuine news against the trend is blocked",
         "GOOGL 2019-06-21: a genuine headline that technology shares were trading higher, in a "
         "falling week; Traffic year-over-year claims",
         "The evidence, not the judge, disagrees with a true report; the oracle judge blocks as "
         "much", "Traffic FBR 32.1% (oracle 30.2%); GOOGL blocked by v1 and v2"],
        ["5", "Blocking discards genuine information",
         "NFLX 2018-11-23, A2 (synthetic): “Netflix shares are expected to rise next "
         "week.”",
         "v2 flags the fake (weight 0.06), but its fallback, the no-text forecast (343.09), "
         "coincides with the fake forecast (343.09) and differs from the genuine one (316.78) by "
         "the same amount",
         "DSS +0.380σ undefended, under v1 and under v2"],
        ["6", "Forward claim never asked (v1)",
         "A2 on Model 5: genuine report plus one sentence about next period",
         "The judge answers from the true past-tense claim, so the gate lets the report through, "
         "and the model acts on the added sentence", "Past claim read in 91.3%; DSS +0.400σ under v1"],
        ["7", "Gate passes the lie and gates the truth (v1)",
         "A1: reversals chosen to agree with the trend",
         "Fakes pass, while the genuine reports for those windows disagree with the trend and get "
         "discounted",
         "7.1% of fakes vs 40.5% of genuine reports gated"],
        ["8", "Fake retains an explanation that no longer matches (attack side)",
         "FNSPID flip (synthetic): “… trading higher after the Phase 3 study did not meet its "
         "primary endpoint”",
         "Lexical edits cannot make the stated cause agree; a reader or model may discount the fake",
         "52 of 108 flip bundles flagged; A1 on FNSPID +0.048σ to +0.006σ without them"],
        ["9", "Forecaster copies a report figure",
         "Traffic: answer “259.7” (billions) for a series in millions; sceptical prompt "
         "copying “2,730.6 billion”",
         "A unit slip, counted as a failure by the off-scale rule, never as a forecast",
         "16 Model 5 failures in P5, all Traffic"],
    ]
    d.table("errors", "Error Taxonomy With Concrete Cases",
            ["#", "Failure", "Case", "What happens", "Size / evidence"], rows,
            widths=[0.25, 1.35, 2.05, 2.15, 1.45], wide=True, size=7.5,
            align=["c", "l", "l", "l", "l"],
            note="Cases 3 and 5 are single windows, observed while building the demo; they illustrate "
                 "mechanisms and enter no aggregate. FNSPID headlines © their publishers, via "
                 "FNSPID (CC BY-NC 4.0).")
    d.p("Rows 1–3 are judge errors. A 1.5B judge reads directions into neutral sentences, "
        "confuses a level with a change, and is sometimes unsure of a sentence it reads correctly; "
        "a fixed confidence floor removes doubtful and clear claims alike. Because the gate takes "
        "the minimum over claims, more judged units means more chances for one error to block a "
        "report, which is why splitting reports helped against A2 but hurt the FBR on headline "
        "bundles.", runin="Judge")
    d.p("Row 4 is the evidence failing a true report. A genuine report can legitimately disagree "
        "with a short trend: a reversal of fortune, a year-over-year comparison that a three-month "
        "window cannot see, or a statement about a related but different series. No judge "
        "improvement removes these blocks, which is why the oracle judge blocks as much on Traffic "
        "and Economy.", runin="Evidence")
    d.p("Row 5 shows that detection is not protection. When a report is blocked, the output "
        "reverts to the forecast made without text, which also drops whatever the genuine report "
        "contributed; if that fallback is no closer to the genuine forecast than the fake forecast "
        "is, catching the lie leaves DSS where it was. Rows 6 "
        "and 7 show the two ways an adaptive attacker uses the gate’s blind spots. Rows 8 and 9 "
        "belong to the attack and the forecaster rather than to TextGuard: fakes whose stated cause "
        "no longer fits make our attack a conservative lower bound for a fluent attacker, and unit "
        "slips are text-induced failures of their own.", runin="Blocking and the attacker")


# ===========================================================================================
def _limitations(d):
    d.h1("VIII. Limitations")
    items = [
        ("Fusion and model type are confounded",
         "the only early-fusion model in P5 is also the only LLM. R1 and R2 narrow the question "
         "but do not test an early-fusion non-LLM."),
        ("The Health_US reversal edits a number",
         "mirroring the stated ILI level is partly a numeric injection; the change-reversal domains "
         "are reported separately and still show the effect."),
        ("Exaggeration is out of TextGuard’s reach",
         "it keeps the true direction, so the gate passes it; it raised Model 5’s error by "
         "54%, and clipping, not TextGuard, limited that damage."),
        ("False blocks exceed the budget",
         "v1 blocked 14.4% of genuine test reports, v2 18.4% on held-out data, against a 10% "
         "budget; validation understated both."),
        ("The judge is small",
         "it confuses levels with changes, reads directions into neutral headlines and does not "
         "separate outlook from change claims; a larger judge was not tested."),
        ("Evidence frames",
         "a short trend cannot check year-over-year claims (v1); v2 needs the series’ history "
         "beyond the lookback, an input requirement, and cannot verify claims about a different "
         "series, which leaves Economy’s residual false blocks."),
        ("Small samples",
         "each original domain has 46 to 130 test windows with a report and as few as 26 attacked "
         "ones, A1 has 42 valid windows, and the FNSPID judge check rests on 50 labels."),
        ("No forecaster that gains from text",
         "for Model 5 the text was never shown to improve accuracy (its usage gap interval spans zero, "
         "and FNSPID headlines worsen it), so “TextGuard keeps the benefit of text” is "
         "not shown."),
        ("Memorisation",
         "Qwen may have seen these public statistics, stocks and reports; our exact-match checks "
         "flagged nothing, but they cannot exclude recall."),
        ("Reproduction of published accuracy",
         "our chronological splits differ from the reference library’s, so its published MSE "
         "is not reproduced; the networks and recipe are the same."),
        ("Side effects of gating",
         "on Model 3, which ignores direction, TextGuard itself introduces a slight directional "
         "shift, and the Flip Rate inflates the count of reversals for a forecaster that copies the "
         "last value (Phi-3.5)."),
        ("Attack realism",
         "fakes are rule-based and tuned to the report styles; the FNSPID reversals may retain an "
         "explanation that no longer matches the reversed move, 19% of the original attacked test texts held a numeric "
         "inconsistency (conclusions unchanged without them), and no LLM-written or human-rated "
         "attack was run."),
    ]
    for head, text in items:
        d.p(text[0].upper() + text[1:], runin=head)


def _conclusion(d):
    d.h1("IX. Conclusion and Future Work")
    d.p("A single word changed in a genuine report moves an LLM forecaster that reads the report, "
        "and the forecasters that resisted did so because they ignored what the report said. The "
        "Directional Shift Score makes this visible where error metrics cannot. TextGuard, a "
        "training-free, inference-time, black-box claim-vs-evidence consistency gate, removes much "
        "of the manipulation on the plain reversal and, in its revised form, on an appended "
        "forward-looking lie in held-out domains; in every version it blocks more genuine text "
        "than our budget allowed, and validation understated that cost. We read the results as a "
        "map of the trade-off rather than a solved problem.")
    d.p("Three directions follow from the failures. The judge needs to distinguish levels, changes "
        "and outlooks and to abstain on neutral sentences, and evidence should match each "
        "claim’s reference frame and series. A gate should be calibrated on the deployment "
        "domain, since validation on other domains understated v2’s false blocks. Two planned "
        "extensions were not run and remain future work: realistic attacks written by LLMs of "
        "another family, with a human plausibility study, and a drift monitor that flags a "
        "sustained campaign of fake reports from shifts in their embeddings. Finally, the target "
        "scenario, a forecaster that profits from the text yet can be manipulated through it, still needs "
        "a stronger model or domain than ours.")
    d.p("The fake reports were generated for defence research only and are released with a "
        "synthetic label on every row; FNSPID text is used and redistributed under CC BY-NC 4.0 "
        "with attribution, and Time-MMD report text is not redistributed by the demo.",
        runin="Ethics and data use")


# ===========================================================================================
def _appendices(d):
    d.h1("Appendix A: Prompts")
    d.p("The strings below are generated by the code that produced every answer; angle brackets "
        "mark values filled in per window. Weekly series say “week” where monthly ones "
        "say “month”.", indent=False)
    d.p("Forecaster (Models 5, Qwen2.5-7B and Phi-3.5; system message, then user message):",
        indent=False)
    user = build_prompt("<series description>", "<unit>", "monthly", [0.0, 0.0, 0.0], "<report>")
    user = (user.replace("Most recent 3 values", "Most recent <L> values")
            .replace("t-3: 0", "t-<L>: <oldest value>").replace("t-2: 0", "…")
            .replace("t-1: 0", "t-1: <latest value>"))
    d.code([SYSTEM, ""] + user.split("\n"))
    d.p("Sceptical variant (R2): the same prompt with this line after the closing quotes:",
        indent=False)
    d.code([CAUTION])
    d.p("TextGuard v1 judge (system message, then one question per report; Health_US also gets the "
        "level question):", indent=False)
    d.code([J1.JUDGE_SYSTEM, ""] + J1.build_judge_prompt("SocialGood", "change", "<report>").split("\n"))
    d.code(J1.build_judge_prompt("Health_US", "level", "<report>").split("\n")[4:])
    d.p("TextGuard v2 judge (one unit, one of three questions):", indent=False)
    d.code([J2.JUDGE_SYSTEM, ""] + J2.build_judge_prompt("SocialGood", "change", "<sentence>").split("\n"))
    for kind in ("outlook", "level"):
        d.code(J2.build_judge_prompt("Health_US" if kind == "level" else "SocialGood", kind,
                                     "<sentence>").split("\n")[4:])
    d.p("A2 appended sentence (original domains; the held-out templates follow the same form): "
        "“<Subject> is expected to <rise | fall> next <month | week>.”", indent=False)

    d.h1("Appendix B: Reproduction")
    for i, t in enumerate([
        "Clone the repository and create a Python 3.14.3 environment from requirements-lock.txt "
        "(116 pinned packages).",
        "Run python -m src.data.download, which fetches Time-MMD and the FNSPID files and checks "
        "the SHA-256 of their raw bytes.",
        "Run python reproduce.py. It executes every step from the data download to the R4 analysis, and redraws Fig. 1, on a CPU "
        "(about an hour on a laptop) and compares every regenerated file with the committed one.",
        "LLM answers are not regenerated by default: they were produced on T4 GPUs from committed "
        "input files and are loaded with a fingerprint check of every prompt; the GPU notebooks "
        "regenerate them.",
        "In the recorded clean-room run all 47 steps passed, 147 committed files were byte-identical "
        "and all 9 GPU input files were regenerated byte for byte.",
        "python paper/build_paper.py rebuilds this document, paper/check_numbers.py matches its "
        "numbers to their sources and paper/check_overlap.py measures its overlap with the "
        "repository text."], 1):
        d.item(f"{i})", t, size=9)

    d.h1("Appendix C: Sensitivity Tables")
    _sens_p5(d)
    _sens_r3(d)
    _sens_r4(d)

    d.h1("Appendix D: Configurations")
    d.table("config", "Configurations and Hyperparameters",
            ["Component", "Setting"],
            [["Lookback / horizon", "8 months or 24 weeks (24 on held-out data) / 1 step"],
             ["Splits", "Chronological 70/10/20 within the text window; held-out: all test"],
             ["Models 1–4 training", "Adam, ≤ 200 epochs, patience 20, learning rate from "
              "{10^{−4}, 10^{−3}, 10^{−2}} on validation, seeds 2021–2023"],
             ["PatchTST patching", "Patch L/2, stride L/4"],
             ["Text head (M3, M4, R1)", "Frozen encoder, mean-pooled, per-dimension standardised; "
              "MLP 768→96→1; fusion weight w = 0.1"],
             ["Encoders", "bert-base-uncased; FinBERT; Qwen2.5-1.5B last layer (R1)"],
             ["LLM forecasters", "fp16 (7B: 4-bit NF4), greedy, batch 1, ≤ 24 new tokens, "
              "off-scale rule 10× the window range"],
             ["Judge", "Qwen2.5-1.5B-Instruct, pinned revision; scored three-way answers"],
             ["TextGuard v1", "Soft gate τ = 0; w = 3 (monthly), 6 (weekly)"],
             ["TextGuard v2", "w = 4 (monthly), 12 (weekly); c₀ = 0.8; τ = 0; cap 1σ; "
              "period 12 / 52"],
             ["Clip / anchor15", "k = 0.5; projection onto y_last ± 15%"],
             ["Statistics", "Bootstrap 10,000 resamples, seed 0; α = 0.05; Holm per family"],
             ["Holm family sizes", "P5 76; P6 20; R1 108; R2 44, 44, 58, 54; R3 105, 105, 75, 75; "
              "R4 79 per target"],
             ["SESOI and budget", "0.05σ DSS; 5 pp Flip Rate and FBR; 10% MSE; FBR budget 10%"]],
            widths=[1.2, 2.4], size=7.5)

    d.h1("Appendix E: Verification Checks")
    d.table("checks", "Verification Checks Run Before or With the Analyses",
            ["Check", "Result"],
            [["Models 1–2 text-blind", "Real, none and fake forecasts bit-identical (asserted)"],
             ["text = None reproduces the backbone", "DLinear heads bit-identical; PatchTST heads "
              "within float32 precision"],
             ["Timing and leakage", "0 violations after alignment; known leakers dropped; year "
              "typos kept and listed"],
             ["Prompt provenance", "Every LLM answer carries its prompt fingerprint; loaders refuse "
              "a mismatch"],
             ["GPU stack reproducibility", "A 40-prompt probe reproduces earlier answers before "
              "answers from a new run are combined"],
             ["Re-runs reproduce frozen results", "R2 reproduces P5’s Model 5 rows exactly; R3 "
              "reproduces P5, P6, R1 and R2 reference values"],
             ["Memorisation", "No model × domain flagged against a placebo rate"],
             ["Attack texts", "Rule checks plus a consistency scan; reviewed by eye in rounds"],
             ["Clean-room reproduction", "47 / 47 steps, 147 files identical, 9 / 9 GPU inputs"],
             ["Test suite", "484 tests passed"],
             ["This document", "Every number matched to a summary table or the results log "
              "(check_numbers.py)"]],
            widths=[1.3, 2.3], size=7.5)

    d.h1("Appendix F: Methodological Lessons")
    for i, t in enumerate([
        "State a smallest effect of interest with every decision rule. R1 lacked one, and a "
        "+0.0075σ shift counted as “fooled”.",
        "Write expectations with their sign and check the algebra. R3 expected the selective flip "
        "to make TextGuard worse; discounting a genuine report narrows the fake–real gap "
        "instead.",
        "Keep fixes found after results as sensitivity analyses. The C8 consistency check would "
        "have changed one R3 reading; it was reported, not adopted.",
        "Validation overstates a design iterated on it. v2 was chosen among five variants on 159 "
        "genuine texts; its held-out FBR was nearly three times its validation FBR.",
        "Scan generated attacks for internal contradictions before computing metrics (R3 "
        "Amendment 1), and report results with and without the flagged texts.",
        "Choose comparison yardsticks before looking where possible, and disclose the ones chosen "
        "later (the off-scale rule and the failure-concentration yardstick).",
        "Reproduce from a clean clone. Four bugs (line endings, a truncated download, a missing "
        "cached prompt, a hash computed on checked-out bytes) appeared only there; none changed a "
        "number."], 1):
        d.item(f"{i})", t, size=9)


# ---- appendix tables ----------------------------------------------------------------------
def _sens_p5(d):
    P5, N5 = "summary_p5_attack.csv", "summary_p5_attack_no2020.csv"
    P6, N6, X = "summary_p6_operating.csv", "summary_p6_operating_no2020.csv", "summary_p3_exclusion_check.csv"
    q5 = dict(short="M5", condition="flipped", domain="pooled")
    q6 = dict(short="M5", attack="flipped", defence="TextGuard τ=0", domain="pooled")
    body = [
        ["Model 5 flip DSS", V(P5, "dss", **q5), V(N5, "dss", **q5),
         V(X, "dss", analysis="flagged texts excluded", table="p5_attack", **q5)],
        ["Model 5 Flip Rate", V(P5, "flip_rate", pct, **q5), V(N5, "flip_rate", pct, **q5),
         V(X, "flip_rate", pct, analysis="flagged texts excluded", table="p5_attack", **q5)],
        ["TextGuard v1 DSS", V(P6, "dss", **q6), V(N6, "dss", **q6),
         V(X, "dss", analysis="flagged texts excluded", table="p6_operating", attack="flipped",
           defence="TextGuard τ=0", domain="pooled")],
        ["TextGuard v1 FBR", V(P6, "fbr", pct, **q6), V(N6, "fbr", pct, **q6), "unchanged"],
    ]
    d.table("sens_p5", "P5/P6 Sensitivity (Model 5, Flip, Test, Pooled)",
            ["Quantity", "As reported", "Without 2020", "Flagged texts out"], body,
            widths=[1.4, 0.9, 0.9, 1.0], size=7.5,
            note="Flagged texts: attacked P3 texts that the later R3 consistency scan flags "
                 "(exploratory). FBR concerns genuine texts, which no exclusion changes.")


def _sens_r3(d):
    sets = (("Amendment 1 (pre-registered)", "summary_r3_readings.csv", {}),
            ("All attacked texts", "summary_r3_robustness_readings.csv", {}),
            ("With C8 (not adopted)", "summary_r3_sensitivity_c8_readings.csv", {}))
    body = []
    for t, lab in (("M5", "Model 5"), ("DLinear+FinBERT", "M3-F")):
        for name, f, extra in sets:
            r = pick(f, target=t, attack="a1_selective", cell="pooled", defence="TG", **extra)
            body.append([lab, name, r["U"],
                         V(f, "dss_none", target=t, attack="a1_selective", cell="pooled", defence="TG"),
                         V(f, "dss_defended", target=t, attack="a1_selective", cell="pooled", defence="TG"),
                         r["reading"]])
    d.table("sens_r3", "R3 Selective Flip (A1) Under Three Window Sets (Pooled)",
            ["Target", "Window set", "Undefended", "DSS", "DSS TextGuard", "TextGuard verdict"],
            body, widths=[0.55, 1.25, 0.75, 0.55, 0.65, 1.05], size=7)


def _sens_r4(d):
    C, S, A = "summary_r4_covid.csv", "summary_r4_sigma_alt.csv", "summary_r4_v2_ablations.csv"
    O, Fb, M = "summary_r4_operating.csv", "summary_r4_fbr.csv", "summary_r4_semantic.csv"
    W = "summary_r4_single_window.csv"

    def a2(f):
        return " / ".join(V(f, "estimate", target="M5", block=b, attack="a2_forward", cell="pooled")
                          for b in ("U", "T10", "T20"))

    frozen_a2 = " / ".join(V(O, "dss", target="M5", attack="a2_forward", defence=x, cell="pooled")
                           for x in ("none", "TG1", "TG2"))
    frozen_fbr = " / ".join(V(Fb, "fbr", pct, target="M5", defence=x, cell="pooled") for x in ("TG1", "TG2"))
    sig_fbr = " / ".join(V("summary_r4_sigma_alt_fbr.csv", "fbr", pct, target="M5", defence=x, cell="pooled")
                         for x in ("TG1", "TG2"))
    abl = []
    for var in ("V0 raw claims (as built)", "no cap (k ∞)"):
        abl.append([f"v2 ablation: {var.split(' (')[0] if 'V0' in var else 'no cap'}",
                    "FBR / A2 (v2)", f"{V(Fb, 'fbr', pct, target='M5', defence='TG2', cell='pooled')} / "
                                     f"{V(O, 'dss', target='M5', attack='a2_forward', defence='TG2', cell='pooled')}",
                    f"{V(A, 'fbr', pct, target='M5', variant=var, cell='pooled')} / "
                    f"{V(A, 'dss_a2_forward', target='M5', variant=var, cell='pooled')}"])
    cap = [r["variant"] for r in __import__("build_paper").rows(A) if r["variant"].startswith("cap only")][0]
    abl.append(["v2 ablation: cap only", "FBR / A2 (v2)",
                f"{V(Fb, 'fbr', pct, target='M5', defence='TG2', cell='pooled')} / "
                f"{V(O, 'dss', target='M5', attack='a2_forward', defence='TG2', cell='pooled')}",
                f"{V(A, 'fbr', pct, target='M5', variant=cap, cell='pooled')} / "
                f"{V(A, 'dss_a2_forward', target='M5', variant=cap, cell='pooled')}"])
    body = [
        ["Feb–May 2020 out", "A2: undefended / v1 / v2", frozen_a2, a2(C)],
        ["FNSPID σ = 520 weeks", "A2: undefended / v1 / v2", frozen_a2, a2(S)],
        ["FNSPID σ = 520 weeks", "FBR v1 / v2", frozen_fbr, sig_fbr],
        ["Connector-flagged bundles out", "Flip, FNSPID, undefended",
         V(M, "dss", target="M5", attack="flipped", windows="with flagged (all)", cell="FNSPID", defence="none"),
         V(M, "dss", target="M5", attack="flipped", windows="without flagged", cell="FNSPID", defence="none")],
        ["Connector-flagged bundles out", "A1, FNSPID, undefended",
         V(M, "dss", target="M5", attack="a1_selective", windows="with flagged (all)", cell="FNSPID", defence="none"),
         V(M, "dss", target="M5", attack="a1_selective", windows="without flagged", cell="FNSPID", defence="none")],
        ["Most extreme window out", "A2, undefended",
         V(W, "mean", target="M5", attack="a2_forward", defence="none", cell="pooled"),
         V(W, "mean_without", target="M5", attack="a2_forward", defence="none", cell="pooled")],
    ] + abl
    d.table("sens_r4", "R4 Robustness (Model 5, Pooled Unless Stated)",
            ["Check", "Quantity", "As reported", "Sensitivity"], body,
            widths=[1.2, 1.1, 1.15, 1.15], size=7,
            note="All rows descriptive and pre-stated in the R4 plan, except the most-extreme-window "
                 "row, which the plan also lists. DSS in σ.")
