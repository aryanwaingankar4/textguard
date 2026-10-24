"""Paper content: title block, abstract, Sections I-IV (Introduction to Experimental Setup).

Prose numbers come from PROJECT_STATUS §9 or the summary tables (checked by check_numbers.py);
table cells are read through V() in build_paper.py.
"""

from build_paper import FIG, V

AUTHORS = [
    ("Aryan Manoj Waingankar", "aryan.waingankar.btech2023@sitpune.edu.in"),
    ("Devansh Kumar", "devansh.kumar.btech2023@sitpune.edu.in"),
    ("Nivedita Mishra", "nivedita.mishra@sitpune.edu.in"),
]
AFFIL = ["Department of Artificial Intelligence and Machine Learning",
         "Symbiosis Institute of Technology",
         "Symbiosis International (Deemed University)", "Pune, India"]


def write(d):
    d.title("TextGuard: Detecting and Defending Against Misleading Text Attacks on Multimodal "
            "Time-Series Forecasting Models")
    d.authors([(n, AFFIL, e) for n, e in AUTHORS])
    d.columns(2)

    d.abstract("Abstract", (
        "Forecasters that read news or official reports next to the numbers acquire an input "
        "channel that nobody verifies. We ask what happens when an adversary rewrites such a report "
        "so that it states the opposite of what happened while every other word stays in place. On "
        "four text-rich domains of Time-MMD we attack five forecasters with deterministic, minimal "
        "edits and score the outcome with the Directional Shift Score (DSS): how far the forecast "
        "moves toward the planted claim, in units of the series’ standard deviation σ. A "
        "Qwen2.5-1.5B forecaster that reads the report inside its prompt moves +0.457σ toward "
        "the lie and turns its predicted direction in 23.6% of attacked windows. Late-fusion heads on "
        "a frozen BERT encoder move by at most 0.005σ, because that encoder barely separates "
        "“rose” from “fell”; with FinBERT in its place the same heads become "
        "steerable. A 7B model is steered as well, and telling the model to be sceptical does not "
        "help. We then propose TextGuard, a training-free, inference-time, black-box claim-vs-evidence "
        "consistency gate that discounts a report whose claims contradict the numbers. Its first "
        "version removes the pooled manipulation (+0.451σ to −0.042σ) but blocks "
        "14.4% of genuine reports, above our 10% budget, and a single appended forward-looking "
        "sentence evades it. A revised version, developed on validation data and tested once on two "
        "held-out domains (U.S. gasoline prices and FNSPID stock headlines), lowers that attack from "
        "+0.180σ to +0.021σ but blocks 18.4% of genuine reports. All analyses were "
        "pre-registered, and every reported number regenerates from the released code."))
    d.abstract("Index Terms", (
        "Adversarial machine learning, large language models, misinformation, multimodal "
        "time-series forecasting, pre-registration, robustness."))

    # ------------------------------------------------------------------------------------
    d.h1("I. Introduction")
    d.p("Forecasting systems increasingly receive text together with numbers. A demand model may "
        "be handed a weekly market note, an epidemic forecaster a surveillance bulletin, a price "
        "model the morning’s headlines. Benchmarks such as Time-MMD [@timemmd] pair numerical "
        "series with aligned reports so that a model can draw on both, and large language models "
        "(LLMs) can combine them with no task-specific training [@gruver; @cik]. Once a model reacts "
        "to the text, that text becomes an input an outsider can influence. Reports are written by "
        "people, copied between outlets and seldom verified before a model reads them.")
    d.p("Robustness studies of this channel have so far perturbed text without intent. PA-RNet "
        "[@parnet] bounds the error caused by zero-mean noise on the text, and Sridhar et al. "
        "[@sridhar] empty, shuffle or swap texts and find that three Time-MMD architectures barely "
        "notice. Attacks with a goal have been studied on other inputs and systems: the numerical "
        "history of LLM forecasters [@liu_adv], tweets fed to stock-movement classifiers [@xie], "
        "headlines read by an LLM-sentiment trading pipeline [@rizvani], news retrieved by event "
        "forecasters [@corroboration] and market commentary given to LLM pricing agents [@msi]. "
        "Untested is the plainest adversary a report-reading forecaster faces: someone who takes a "
        "genuine report and reverses what it says (“the deficit increased” becomes "
        "“the deficit decreased”) and changes nothing else. The edit is visible, yet it is "
        "small and plausible, and an error metric cannot tell whether the forecast went wrong in the "
        "direction the attacker wanted.")
    d.p("We study that adversary and a defence against it. Manipulation is measured by the "
        "Directional Shift Score (DSS), the signed movement of the forecast toward the direction "
        "the fake claims, divided by the series’ standard deviation. We first screen five "
        "forecasters for whether text moves them at all (a step that replicates [@sridhar]), and "
        "then attack them with rule-written fakes whose edit is exact and auditable. The defence, "
        "TextGuard, asks a small LLM what each claim in a report asserts and compares the answer with "
        "what the numbers show; a claim the evidence contradicts shrinks the weight given to the "
        "text-informed forecast. TextGuard calls only the forecaster’s predict() method, so it "
        "applies to any model without training, at the cost of one extra forecast and a few judge "
        "queries per window.")
    d.p("Four findings stand out. First, the attack works on the forecaster that reads the report "
        "in its prompt and hardly moves late-fusion models built on frozen BERT; those are not "
        "robust but deaf, since their encoder scarcely distinguishes a report from its reversal, and "
        "swapping in an encoder that registers direction makes those heads steerable. Second, none "
        "of the forecasters we tested both registered a report’s direction and withstood the "
        "attack. Third, "
        "TextGuard removes most of the manipulation caused by the plain reversal, but blocks more "
        "genuine text than its pre-registered budget allows, and an attacker who appends a single "
        "sentence about the future gets through. Fourth, a revised gate that judges each sentence "
        "on its own closes much of that gap on held-out data, while blocking more genuine reports "
        "than its validation results suggested. Each analysis followed a plan committed before its "
        "results existed, and we report outcomes under the stated rules, unfavourable ones included.")
    d.p("Our contributions are:", indent=False)
    for i, t in enumerate([
        "A threat model and benchmark for visible, minimal, meaning-inverting edits of genuine "
        "reports that feed multimodal time-series forecasters, with deterministic rule-based "
        "attackers (direction reversal, exaggeration and three adaptive variants) whose fakes are "
        "released as labelled synthetic research text.",
        "DSS, a metric for directional manipulation that separates a forecast that is merely wrong "
        "from one that is wrong in the attacker’s chosen direction, reported together with a "
        "Flip Rate and a False Block Rate.",
        "A comparison of early- and late-fusion forecasters under this attack, with encoder and "
        "model swaps that locate where the vulnerability comes from.",
        "TextGuard, a training-free, inference-time, black-box claim-vs-evidence consistency gate, "
        "in a first version and a revised version evaluated once on held-out domains, against "
        "clipping, never or always trusting the text, a cautionary prompt and an output projection.",
        "A pre-registered and byte-reproducible evaluation: plans, amendments, code and summary "
        "tables accompany the paper, and a fresh clone regenerates every reported number."], 1):
        d.item(f"{i})", t)

    # ------------------------------------------------------------------------------------
    d.h1("II. Related Work")
    d.h2("A. Text in Multimodal Time-Series Forecasting")
    d.p("Time-MMD [@timemmd] aligns numerical series from nine domains with filtered reports and "
        "search snippets, and its companion library attaches a text branch to standard forecasters "
        "such as DLinear [@dlinear] and PatchTST [@patchtst]. Context is Key [@cik] constructs tasks "
        "whose solution requires the accompanying text, and LLMs given the numbers as a string can "
        "forecast zero-shot [@gruver]. Whether text helps is disputed. Nguyen et al. [@collapse] "
        "describe text collapse, in which fused models learn to ignore the text; Sridhar et al. "
        "[@sridhar] report that replacing Time-MMD text with empty, shuffled or cross-domain text "
        "hardly changes the error of three architectures, and trace part of the reported gain to a "
        "numeric column shipped alongside the text. Lee et al. [@cfa] show that naive fusion lets "
        "irrelevant text degrade accuracy and propose a Controlled Fusion Adapter (CFA) that filters "
        "it with trained low-rank adapters. Training-free text conditioning of foundation models "
        "[@lafp] and information-theoretic audits that rank correct, incorrect and irrelevant "
        "annotations [@andrews] extend this line. None of these works models an adversary.")
    d.h2("B. Attacks and Defences for Forecasters")
    d.p("Adversarial examples for forecasting usually perturb the numbers. Liu et al. [@liu_adv] "
        "derail LLM-based forecasters with small perturbations of the input series, and Ma et al. "
        "[@ma_detect] detect such perturbations in energy forecasting by checking whether forecasts "
        "from shortened inputs agree. On the text side, PA-RNet [@parnet] adds a perturbation-aware "
        "residual branch with an error bound for zero-mean textual noise; a reversal of meaning is "
        "not zero-mean, so the bound does not cover our setting. CFA [@cfa] is the closest plug-in "
        "control of fusion, but it is trained and addresses irrelevant rather than adversarial text.")
    d.h2("C. Manipulating LLM Decision Systems Through Text")
    d.p("Indirect prompt injection [@greshake] showed that text an application retrieves can steer "
        "the LLM behind it. In finance, Xie et al. [@xie] fooled tweet-based stock-movement "
        "classifiers with a few crafted tweets, and Rizvani et al. [@rizvani] manipulated an "
        "LLM-sentiment-to-LSTM trading system with imperceptible headline injections (homoglyphs and "
        "hidden clauses), measuring lost returns; AutoRedTrader [@autoredtrader] red-teams trading "
        "agents with generated misinformation. Lu and Zhang [@corroboration] move retrieval-augmented "
        "event forecasts in a chosen direction by publishing corroborating articles. Closest to our "
        "defence, Market Signal Injection [@msi] edits the commentary shown to LLM pricing agents and "
        "proposes two training-free, inference-time defences: input canonicalization, which deletes "
        "the injected commentary, and decision boundary anchoring, which constrains the prompt and "
        "projects the price into a fixed band.")
    d.h2("D. Positioning")
    d.p("{T:related} lists the closest works. Ours differs from each in at least one of three ways. "
        "The edit is a visible, minimal reversal of a genuine report, not noise, an imperceptible "
        "injection or added commentary. The target is a numerical forecaster that reads the report, "
        "scored by directional displacement rather than by error or returns. The defence scores a "
        "report by checking its claims against the numerical evidence, whereas canonicalization "
        "discards the text, anchoring and clipping bound the output whatever the text says, and CFA "
        "trains a filter. We claim neither the first text attack on a price forecaster "
        "[@xie; @rizvani] nor the first training-free defence [@msi].")
    d.table("related", "Closest Work Compared With Ours",
            ["Work", "Task", "Mechanism", "Evaluated on"],
            [["Sridhar et al. [@sridhar]", "Audit of text use", "Empty, shuffled and cross-domain text",
              "Time-MMD, three architectures"],
             ["PA-RNet [@parnet]", "Robust multimodal forecasting",
              "Residual branch; zero-mean text noise", "Multimodal forecasting benchmarks"],
             ["CFA [@cfa]", "Fusion control", "Trained low-rank adapters filter irrelevant text",
              "Multimodal forecasting benchmarks"],
             ["Liu et al. [@liu_adv]", "Attack", "Perturbs the numerical input",
              "LLM-based forecasters"],
             ["Ma et al. [@ma_detect]", "Detection", "Forecast divergence on shortened inputs",
              "Time-series LLMs, energy data"],
             ["Xie et al. [@xie]", "Attack", "Crafted tweets", "Tweet-based stock-movement models"],
             ["Rizvani et al. [@rizvani]", "Attack", "Imperceptible headline injections",
              "LLM-sentiment trading, stock data"],
             ["Lu and Zhang [@corroboration]", "Attack", "Corroborating news articles",
              "Retrieval-augmented event forecasters"],
             ["Lee and Park [@msi]", "Attack and defence",
              "Commentary injection; canonicalization, anchoring", "Simulated pricing markets"],
             ["**Ours**", "**Attack and defence**",
              "**Rule-based reversal of genuine reports; DSS; claim-vs-evidence gate**",
              "**Time-MMD (4 domains); held-out Energy and FNSPID**"]],
            widths=[1.25, 1.1, 2.0, 1.65], wide=True)
    d.figure("pipeline", FIG / "fig1_pipeline.png",
             "The TextGuard pipeline. The forecaster is queried without (A) and with (B) the report. "
             "A judge reads the report’s claims, each claim is compared with evidence computed "
             "from the numbers, and the gate g sets how much of B − A reaches the output. The "
             "adversary’s edit (red) enters B directly; TextGuard sees it only through the judge.",
             wide=True, width_frac=0.92)

    # ------------------------------------------------------------------------------------
    d.h1("III. Methodology")
    d.h2("A. Threat Model")
    d.p("The adversary controls the text of a report that a forecaster will read for one forecast "
        "window. It may edit a genuine report or append to it, but it cannot alter the numerical "
        "history, the model or the prompt template. Its goal is to move the forecast in a chosen "
        "direction, for example toward a lower price, and it wants the edit to survive a casual "
        "reading, so it changes as little as possible and keeps the report’s own numbers "
        "consistent with the new claim. In the main attack the adversary does not know the defence. "
        "In the adaptive attacks of Section VI-E it knows TextGuard’s code and frozen settings "
        "and sees the same past values, but it cannot query the forecaster or the judge. The defender "
        "operates a black-box forecaster exposing predict(numbers, text), may call it with and "
        "without the report and may read the series’ history; it cannot retrain the forecaster "
        "and does not know whether a given report is genuine.")
    d.h2("B. Forecasters")
    d.p("All forecasters implement one interface, predict(numbers, text), on a univariate window in "
        "the series’ own units. Models 1 and 2 are DLinear [@dlinear] and PatchTST [@patchtst] "
        "without text; they never see a report and act as the control. Models 3 and 4 add a "
        "late-fusion text head to the same trained networks. A frozen bert-base-uncased encoder "
        "[@bert] embeds the report (all layers, mean-pooled over tokens), each dimension is "
        "standardised with statistics of the training texts, and a two-layer MLP adds w·MLP(z) "
        "to the frozen backbone’s forecast, with w = 0.1. Because the backbone stays frozen, "
        "Models 3 and 4 with text = None reproduce Models 1 and 2 exactly, so B − A isolates "
        "the effect of the text. Model 5 is Qwen2.5-1.5B-Instruct [@qwen25] used as an early-fusion "
        "forecaster: a single chat prompt holds the lookback values, labelled by relative step "
        "without dates, followed by the report when there is one, and greedy decoding returns one "
        "number. Its text = None prompt is the same prompt with the report block removed. The "
        "extensions add late-fusion heads on FinBERT [@finbert] and on mean-pooled Qwen embeddings "
        "(R1), and Qwen2.5-7B-Instruct in 4-bit NF4 [@qlora], Phi-3.5-mini-instruct [@phi3] and "
        "three prompt variants of Model 5 (R2).")
    d.h2("C. Attacks")
    d.p("Every fake is written by deterministic rules rather than by an LLM. Rules make the edit "
        "exact and auditable, and they avoid a circularity: Model 5 and the TextGuard judge are both "
        "Qwen models, so a Qwen-written fake would test Qwen against itself. The rules locate the "
        "report’s headline claim about the forecast target, edit it, and record the direction "
        "the fake now claims, d ∈ {+1, −1}, expressed for the target series. A report that "
        "cannot be edited cleanly is left unattacked; no fake that contradicts itself is emitted.")
    d.p("Direction words are swapped (increased/decreased, above/below) and dependent numbers are "
        "recomputed so that the current level stays fixed: “X, up Y from Z” becomes "
        "“X, down Y from X + Y”. Economy, Traffic and SocialGood reports state changes, "
        "so the flip reverses the change. Health_US reports state a level against a national "
        "baseline B; the flip mirrors the stated value V to 2B − V (or to B²/V when "
        "2B − V ≤ 0), which keeps the fake plausible but also edits a number.",
        runin="Flip")
    d.p("The direction is kept and each stated change is tripled, with an intensifier such as "
        "“sharply”.", runin="Exaggeration")
    d.p("The flip, restricted to windows in which the reversed claim matches the trend that "
        "TextGuard v1 derives from the numbers, which gives the gate no grounds to object.",
        runin="A1, selective flip")
    d.p("The genuine report stays word for word and gains one extra sentence asserting the "
        "opposite direction for the coming period, for example “The unemployment rate is expected to rise "
        "next month.”", runin="A2, forward-looking lie")
    d.p("The direction stays correct but each stated change is scaled up tenfold; values that "
        "depend on it are recomputed and implausible outcomes are dropped.",
        runin="A3, magnitude lie")
    d.p("For the held-out domains the same rules were reimplemented for weekly gasoline reports and "
        "for weekly bundles of stock headlines (Section IV-A); A3 is run on Energy only, since few "
        "headlines state a size of move.")
    d.h2("D. Metrics, Statistics and Pre-Registration")
    d.p("Let ŷ_real and ŷ_fake be the forecasts given the genuine and the fake report "
        "for the same window, d the direction the fake claims and σ the standard deviation of "
        "the series over its training span. The Directional Shift Score is", indent=True)
    d.eq("DSS = mean over attacked windows of d·(ŷ_fake − ŷ_real) / σ.")
    d.p("A positive DSS means the forecast moved toward the lie; it is zero for a model that "
        "ignores text and negative when the fake pushed the forecast the other way. For exaggeration "
        "d is the direction of the genuine claim, so a positive value means the forecast was pushed "
        "further. The Flip Rate is the share of attacked windows in which the real-text forecast "
        "does not point in direction d relative to the last observed value y_last, while the "
        "fake-text forecast does. When a defence is active, the genuine-report and fake-report "
        "forecasts are each filtered by it, so DSS then captures the displacement of the protected "
        "system’s output caused by swapping the true report for the false one. The False Block Rate (FBR) is the share of genuine reports whose applied text weight "
        "falls below 0.5. It is the honesty check, because a defence that discards all text scores "
        "perfectly on DSS. Accuracy is the mean squared error of (y − ŷ)/σ.")
    d.p("All quantities are means of per-window terms. Confidence intervals are paired percentile "
        "bootstraps [@efron] with 10,000 resamples, stratified by domain for pooled values. DSS "
        "and paired DSS differences are tested with the Wilcoxon signed-rank test [@wilcoxon], flip "
        "and block counts with an exact McNemar test [@mcnemar] on discordant windows, and agreement "
        "with Cohen’s κ [@cohen]. p-values are Holm-adjusted [@holm] within each "
        "pre-registered family; a directional conclusion requires both a rejection and an estimate "
        "of the stated sign.",
        runin="Statistics")
    d.p("From R2 onward every decision also requires a smallest effect size of interest (SESOI) "
        "[@lakens]: 0.05σ for DSS, 5 percentage points for Flip Rate and FBR differences and "
        "10% of the comparator’s MSE for accuracy. The verdict words are fixed in advance. An "
        "attack *fools* a model when DSS is significantly positive and at least 0.05σ; a "
        "non-significant DSS is *bounded* only when its whole interval lies within ±0.05σ "
        "and *inconclusive* otherwise; a significant estimate below the SESOI is *significant but "
        "negligible*. A defence is *evaded* when the defended model is still fooled and DSS falls by "
        "less than the SESOI; it *reduces but does not stop* an attack when DSS falls and the model "
        "is still fooled; it *stops* the attack when what remains is bounded; and a reduction that "
        "leaves a residual neither fooled nor bounded is reported as “reduces; residual not shown to "
        "be below the SESOI”.", runin="Effect sizes")
    d.p("Each stage (P5 attack, P6 defence) and each extension (R1 to R4) had its hypotheses, Holm "
        "family, decision rules and, from R2, its SESOI written down and committed on their own "
        "before the first output they govern existed [@nosek]. Any later change was added as a "
        "dated amendment and is reported here as a deviation (Section V, Appendix F). Models and defences already reported are "
        "never re-tuned; a revised defence is evaluated on new data only.",
        runin="Pre-registration")
    d.h2("E. TextGuard v1")
    d.p("With A = predict(numbers, None) and B = predict(numbers, text), TextGuard returns")
    d.eq("ŷ = A + g·(B − A),   g ∈ [0, 1].")
    d.p("A judge, Qwen2.5-1.5B-Instruct at a pinned revision, answers a fixed question about the "
        "domain’s subject (e.g. whether the U.S. unemployment rate went up or down) with one of "
        "three words. We score the three answers by their sequence probability, take the most likely "
        "and use its probability renormalised over the three as the confidence c, so "
        "c ∈ [1/3, 1]. Health_US reports also receive a level question (above or below the "
        "national baseline). The judge is deliberately not the rule engine that wrote the fakes, "
        "which would recognise its own edits.", runin="Claim")
    d.p("For a change claim, the least-squares slope β over the last w lookback values gives "
        "the direction e = sign(β) and the strength s = min(|β|(w − 1)/σ, 1), "
        "that is, the size of the fitted rise or fall over the window, measured in σ. For a level claim against a "
        "stated baseline B the evidence is e = sign(y_last − B) with strength "
        "min(|y_last − B|/σ, 1).", runin="Evidence")
    d.p("Claim k with direction d_{k}, confidence c_{k}, evidence e_{k} and strength s_{k} "
        "contributes", runin="Gate")
    d.eq("g_{k} = 1 if d_{k}·e_{k} ≥ 0, else 1 − c_{k}·s_{k};   "
         "g = min_{k} g_{k}.")
    d.p("A warning is raised when g < 0.5. In block mode with threshold τ a gate below τ "
        "is set to zero. The window w and τ were chosen on validation (Section V). Baselines: "
        "always_trust (g = 1, the undefended system), never_trust (g = 0) and clip, which caps "
        "|B − A| at kσ and so applies the weight min(1, kσ/|B − A|).")
    d.h2("F. TextGuard v2")
    d.p("The P6 and R3 results (Sections VI-B and VI-E) exposed four weaknesses of v1: the judge "
        "reads only the report’s main past claim, so a forward claim is never examined; its "
        "change question treats Health_US level statements as changes; Traffic’s "
        "year-over-year claims are compared with a short trend that cannot confirm them; and "
        "exaggerations pass because their direction is right. v2 keeps the interface and changes "
        "four things. (i) *Claim units*: the report is split into sentences, bullet points or "
        "headlines, each judged in isolation, with an additional outlook question asking whether the "
        "unit predicts a rise or a fall. (ii) *Frame-matched evidence*: a unit that names a "
        "year-over-year comparison is compared with the change y_last − y_last−P over one "
        "seasonal period (P = 12 months or 52 weeks), computed from older values that the "
        "defender supplies explicitly; short-term and outlook claims are compared with the trend; cumulative and multi-month "
        "averages are not gated. (iii) A *confidence floor* c₀ below which claims are ignored. "
        "(iv) A *cap* on the applied change:")
    d.eq("ŷ = A + clamp(g·(B − A), −kσ, +kσ).")
    d.p("v2 still calls only predict(), but it requires more of the series’ past than the "
        "model’s lookback window holds, an extra input requirement for whoever deploys it.")

    # ------------------------------------------------------------------------------------
    d.h1("IV. Experimental Setup")
    d.h2("A. Datasets")
    d.p("From Time-MMD [@timemmd] we use the report text (the fact field) of the four domains with "
        "the densest aligned reports: Economy (U.S. goods trade balance, monthly), SocialGood "
        "(U.S. unemployment rate, monthly), Traffic (U.S. vehicle miles travelled, monthly) and "
        "Health_US (proportion of outpatient visits due to influenza-like illness, weekly). The dataset’s "
        "LLM-written forecast field is excluded, because attacking machine-generated predictions is "
        "a different threat.", runin="Original domains")
    d.p("Two further domains were selected and fixed before any model had produced an output on "
        "them. "
        "*Energy* is Time-MMD’s weekly U.S. retail gasoline price with its weekly price "
        "reports; it is an easy case, because the reports restate the target’s latest movement. "
        "*FNSPID* [@fnspid] supplies Benzinga headlines for ten S&P 500 stocks, selected by a fixed "
        "rule written before any price was examined (most distinct news days between June 2018 and "
        "May 2020). Headlines are bundled weekly, Friday to Thursday before the forecast origin, and "
        "the target is the next Friday close. FNSPID is the difficult case, since a week of stock "
        "prices rarely shows a clear trend and many headlines say nothing about the stock’s "
        "direction. FNSPID is used under its CC BY-NC 4.0 licence "
        "for non-commercial research.", runin="Held-out domains")
    d.h2("B. Cleaning and Leakage Checks")
    d.p("Each report is assigned to the period in which it ends and is given to the forecast one "
        "period later; anchoring on the start date had let annual reports reach forecasts within the "
        "same year (37 metadata violations). A content check then flagged five reports that describe "
        "their own future, which were dropped, and seven whose written year is mistyped although the "
        "figures match the period they are filed under, which were kept. The final timing check "
        "reports no violation. Part of Time-MMD’s SocialGood text was produced with an "
        "influenza prompt; reports that state no unemployment figure were removed (347 to 216 "
        "reports). Energy reports quote the price of the Monday after their stated week, so Energy "
        "uses a two-period lag, and one report quoting a later week was dropped. One Energy report "
        "filed for the last week of 2023 names “January 1, 2023”; its price and the "
        "following report show that it describes January 1, 2024, so it is another year typo and "
        "was kept. In FNSPID, price-move headlines agree with the sign of the same day’s return "
        "in 91% of cases and only at chance level on adjacent days, so their dates are reliable.")
    d.p("Splits are chronological (70/10/20) within each domain’s text window, and scalers and "
        "σ use the training span only. On the held-out domains no model is trained, so all "
        "periods with text serve as test data; σ is the standard deviation of the 520 weeks before "
        "Energy’s text window and of the 104 weekly closes before the FNSPID period, per stock "
        "(closes converted back to as-traded prices).")
    d.h2("C. Evaluation Sets")
    _sets_table(d)
    d.h2("D. Model Configurations")
    d.p("The lookback is 8 months or 24 weeks (24 weeks on both held-out domains) and the horizon "
        "is one step. Models 1 to 4 are trained with Adam for up to 200 epochs with patience 20, "
        "the learning rate is chosen per domain on validation from {10^{−4}, 10^{−3}, "
        "10^{−2}}, and three seeds are averaged per window before any metric. PatchTST uses "
        "patches of half the lookback with a quarter-lookback stride, because the reference "
        "library’s patch-length clamp leaves most inputs unread at horizon one; we also drop "
        "that library’s horizon-normalised text term, which is identically zero at horizon one, "
        "and its seasonal prior column, which carries numbers rather than text. The Qwen models run "
        "in fp16 (the 7B model in 4-bit NF4) with greedy decoding, batch size one and at most 24 new "
        "tokens; the first number in the answer is the forecast, and an answer more than ten times "
        "outside its window’s range counts as a failure (a unit slip), never as a forecast. "
        "Greedy decoding makes Model 5 a single deterministic run. Appendix D lists every setting.")
    d.h2("E. Prompts")
    d.p("The forecaster prompt names the series and its unit, lists the lookback values from oldest "
        "to newest with relative labels, places the report between triple quotes under the line "
        "“Report published before period t”, and asks for one number. The v1 judge "
        "receives the whole report and one question; the v2 judge receives a single unit and one "
        "question. The exact strings are in Appendix A.")
    d.h2("F. Result Generation")
    d.p("Every number in this paper is read from a summary table written by the analysis scripts "
        "(results/summary_*.csv) or from the project’s results log; none was copied from a "
        "notebook. LLM answers were produced on Colab or Kaggle T4 GPUs and committed with the "
        "fingerprint of every prompt; all later analysis runs on a CPU. From a fresh clone and a "
        "pinned environment, one script reruns the 47 pipeline steps and reproduced all 147 "
        "committed result and figure files byte for byte, together with the 9 GPU input files "
        "(Appendix B). A separate script extracts the text of this document and matches every "
        "number to its source. Code, plans and the labelled fake-text benchmark: "
        "https://github.com/aryanwaingankar4/textguard.")


def _sets_table(d):
    P5 = "summary_p5_attack.csv"
    split = {"Economy": "244 / 36 / 70", "SocialGood": "243 / 36 / 69",
             "Traffic": "186 / 27 / 53", "Health_US": "470 / 68 / 134"}
    a2 = {"Economy": "70", "SocialGood": "27", "Traffic": "52", "Health_US": "104"}
    body = []
    for dom, freq in (("Economy", "monthly"), ("SocialGood", "monthly"), ("Traffic", "monthly"),
                      ("Health_US", "weekly")):
        body.append([dom, freq, split[dom],
                     V(P5, "n_text", lambda x: f"{x:.0f}", model="Qwen2.5-1.5B", condition="flipped",
                       domain=dom),
                     V(P5, "n_attacked", lambda x: f"{x:.0f}", model="Qwen2.5-1.5B",
                       condition="flipped", domain=dom) + " / " + a2[dom],
                     "training span"])
    C = "summary_r4_attack_coverage.csv"
    comma = lambda x: f"{x:,.0f}"
    for dom, series, sig in (("Energy", "Energy", "520 weeks before text"),
                             ("FNSPID", "(10 stocks)", "104 weeks, per stock")):
        q = dict(domain=dom, series=series)
        body.append([f"{dom} (held-out)", "weekly",
                     "test only: " + V(C, "windows", comma, attack="flipped", **q),
                     V(C, "with_text", comma, attack="flipped", **q),
                     V(C, "clean", comma, attack="flipped", **q) + " / "
                     + V(C, "clean", comma, attack="a2_forward", **q), sig])
    d.table("sets", "Splits and Evaluation Sets",
            ["Domain", "Freq.", "Train / val / test", "Test windows with report",
             "Attacked: flip / A2", "σ from"], body,
            widths=[1.3, 0.85, 1.3, 1.0, 1.05, 1.25], wide=True,
            note="Original domains: periods per split; attack counts are test windows whose report "
                 "the rules could edit cleanly (A2 after the consistency scan of Section VI-E). "
                 "Held-out domains: every window is a test window.")
