"""TextGuard — Streamlit demo (P7.2). Run: ``streamlit run app/app.py``.

Shows saved results only (``app/data/demo.json``, built by ``python -m app.build_data``). No model,
no judge and no custom text: Model 5 and the judge need a GPU, and every number shown here already
exists in ``results/``. Four screens: the problem, the attack, the defence, the results.

Every fake text is SYNTHETIC — written by the project's deterministic attack rules for defence
research — and is labelled as such wherever it appears (ETHICS.md).
"""

from __future__ import annotations

import difflib
import html
import json
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "demo.json"
TEXTS = HERE / "data" / "demo_timemmd_texts.json"
FIGS = HERE / "figures"
WARN = 0.5
DECIMALS = {"Energy": 3, "FNSPID": 2}
DOMAIN_LABEL = {"Energy": "Energy (Time-MMD, held-out)", "FNSPID": "FNSPID stock (held-out)",
                "SocialGood": "SocialGood (original Time-MMD)"}
ATTACK_LABEL = {"flipped": "flip — the report's direction reversed",
                "a2_forward": "A2 — one forward-looking sentence appended"}
DEFENCES = {"off": "TextGuard off", "v1": "TextGuard v1", "v2": "TextGuard v2"}

st.set_page_config(page_title="TextGuard demo", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
.tg-text {border:1px solid rgba(128,128,128,.35); border-radius:6px; padding:.6rem .8rem;
          max-height:340px; overflow-y:auto; font-size:.9rem; line-height:1.5;}
mark.add {background:rgba(230,80,60,.30); color:inherit; padding:0 2px; border-radius:3px;}
mark.del {background:rgba(80,140,230,.28); color:inherit; padding:0 2px; border-radius:3px;
          text-decoration:line-through;}
.tg-synth {display:inline-block; background:rgba(230,80,60,.18); border:1px solid rgba(230,80,60,.6);
           border-radius:4px; padding:1px 8px; font-size:.78rem; font-weight:600; letter-spacing:.03em;}
.tg-unit {border-left:5px solid; border-radius:4px; padding:.35rem .6rem; margin:.3rem 0;
          font-size:.88rem;}
.tg-unit small {opacity:.75;}
.tg-green {border-color:#2e9e57; background:rgba(46,158,87,.10);}
.tg-amber {border-color:#d99a1e; background:rgba(217,154,30,.12);}
.tg-red   {border-color:#d6453d; background:rgba(214,69,61,.12);}
.tg-grey  {border-color:rgba(128,128,128,.5); background:rgba(128,128,128,.06);}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load() -> tuple[dict, dict]:
    d = json.loads(DATA.read_text(encoding="utf-8"))
    t = json.loads(TEXTS.read_text(encoding="utf-8")) if TEXTS.exists() else {}
    return d, t


D, PRIVATE = load()
EX = {e["id"]: e for e in D["examples"]}
H = D["headline"]


# =============================================================================================
# helpers
# =============================================================================================

def sig(v: float) -> str:
    return f"{v:+.3f}σ"


def pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def fmt(e: dict, v: float | None) -> str:
    return "—" if v is None else f"{v:,.{DECIMALS.get(e['domain'], 2)}f}"


def word(d: int) -> str:
    return {1: "up", -1: "down", 0: "no direction"}[d]


def texts_of(e: dict) -> dict | None:
    return e["text"] if e["text"] is not None else PRIVATE.get(e["id"])


def synth_badge() -> str:
    return '<span class="tg-synth">SYNTHETIC — written by our attack rules, not real news</span>'


def _tokens(s: str) -> list[str]:
    return re.findall(r"\n|[^\S\n]+|\S+", s)


def diff_html(real: str, fake: str) -> tuple[str, str]:
    """Word-level diff: words removed from the genuine report (blue, struck) and words the fake
    added or changed (red)."""
    a, b = _tokens(real), _tokens(fake)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    ra, fb = [], []

    def esc(toks):
        return "".join("<br>" if t == "\n" else html.escape(t) for t in toks)

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            ra.append(esc(a[i1:i2]))
            fb.append(esc(b[j1:j2]))
            continue
        if i2 > i1:
            ra.append(f'<mark class="del">{esc(a[i1:i2])}</mark>')
        if j2 > j1:
            fb.append(f'<mark class="add">{esc(b[j1:j2])}</mark>')
    return "".join(ra), "".join(fb)


def text_box(inner: str) -> None:
    st.markdown(f'<div class="tg-text">{inner}</div>', unsafe_allow_html=True)


def attribution(e: dict) -> None:
    if e["domain"] == "FNSPID":
        st.caption("Headlines: " + D["fnspid_attribution"])
    else:
        st.caption("Report: Time-MMD (Liu et al., NeurIPS 2024 Datasets & Benchmarks).")


def missing_texts_notice() -> None:
    st.info("Time-MMD report texts are not stored in the repository (Time-MMD states no licence). "
            "Download the data and run `python -m app.build_data` to see them here. "
            "Every number on this screen is still shown.")


def chart(e: dict, marks: list[tuple[str, float | None, str]]) -> alt.Chart:
    """The lookback series plus one point per forecast at the target date."""
    hist = pd.DataFrame({"date": pd.to_datetime(e["dates"]), "value": e["lookback"]})
    pts = pd.DataFrame([dict(date=pd.to_datetime(e["target_date"]), value=v, what=lab, color=c)
                        for lab, v, c in marks if v is not None])
    lo = min(hist.value.min(), pts.value.min())
    hi = max(hist.value.max(), pts.value.max())
    pad = (hi - lo) * 0.08 or 1
    y = alt.Y("value:Q", title=e["unit"], scale=alt.Scale(domain=[lo - pad, hi + pad], zero=False))
    line = alt.Chart(hist).mark_line(point=alt.OverlayMarkDef(size=18), color="#7f8a99").encode(
        x=alt.X("date:T", title=None), y=y, tooltip=[alt.Tooltip("date:T"), alt.Tooltip("value:Q", format=",.3f")])
    dots = alt.Chart(pts).mark_point(size=170, filled=True, opacity=.9).encode(
        x="date:T", y=y,
        color=alt.Color("what:N", scale=alt.Scale(domain=list(pts.what), range=list(pts.color)),
                        legend=alt.Legend(title=None, orient="bottom", columns=2)),
        shape=alt.Shape("what:N", legend=None),
        tooltip=["what:N", alt.Tooltip("value:Q", format=",.3f")])
    return (line + dots).properties(height=320)


def header(e: dict) -> None:
    st.subheader(e["title"])
    st.caption(f"{e['description']} · {e['series'] if e['domain'] == 'FNSPID' else e['domain']} · "
               f"forecast for **{e['target_date']}** · source: {e['source']} · σ (train std) = "
               f"{fmt(e, e['sigma'])}")
    with st.expander("How this example was chosen (a fixed rule, not hand-picked)"):
        st.markdown(e["rule"] + f"\n\n{e['n_candidates']} windows met the criterion.")


# =============================================================================================
# sidebar
# =============================================================================================

st.sidebar.title("🛡️ TextGuard")
page = st.sidebar.radio("Screen", ["1 · The problem", "2 · The attack", "3 · The defence", "4 · Results"])
st.sidebar.divider()
domains = list(dict.fromkeys(e["domain"] for e in D["examples"]))
dom = st.sidebar.selectbox("Domain", domains, format_func=DOMAIN_LABEL.get,
                           disabled=page.startswith(("1", "4")))
choices = [e["id"] for e in D["examples"] if e["domain"] == dom]
eid = st.sidebar.radio("Example", choices, format_func=lambda i: EX[i]["title"].split("— ", 1)[-1],
                       disabled=page.startswith(("1", "4")))
E = EX[eid]
st.sidebar.divider()
st.sidebar.caption("Precomputed results only — no model runs in this app. Every fake text is "
                   "**synthetic**, made for defence research.")


# =============================================================================================
# 1 · the problem
# =============================================================================================

def page_problem() -> None:
    st.title("Can a forecaster be steered by editing a news report?")
    st.markdown(
        "Some forecasting models read a **text report** alongside the numbers. Where a model really "
        "reads it, the report becomes an unverified input: **change a few words of a genuine report "
        "and the forecast moves where the attacker wants.**")
    c1, c2, c3 = st.columns(3)
    c1.metric("LLM forecaster pushed toward the lie", sig(H["m5_dss"]),
              help="Qwen2.5-1.5B (Model 5), original Time-MMD domains, pooled DSS under the "
                   "rule-based direction flip (P5).")
    c1.caption(f"Qwen2.5-1.5B reads the report in its prompt. A one-direction flip moves its "
               f"forecast toward the lie and **reverses its direction in {pct(H['m5_flip'])}** of "
               f"attacked windows.")
    c2.metric("Late-fusion BERT models", f"|DSS| ≤ {H['late_max_abs_dss']:.3f}σ",
              help="Models 3–4 (DLinear/PatchTST + frozen BERT), pooled DSS under the flip (P5).")
    c2.caption("Not robust, just **deaf**: the pooled BERT embedding doesn't carry the report's "
               "direction, so the lie has nothing to push on.")
    c3.metric("TextGuard v2 vs a stealthier attack (A2)",
              f"{H['r4_a2_none']:+.3f}σ → {H['r4_a2_tg2']:+.3f}σ",
              help="Held-out Energy + FNSPID, Model 5, pooled (R4).")
    c3.caption(f"…but it discounts **{pct(H['r4_fbr_tg2'])} of genuine reports** "
               f"(v1: {pct(H['r4_fbr_tg1'])}). Protection has a price.")
    st.divider()
    st.markdown("#### How TextGuard works: a claim-vs-evidence gate around any `predict(numbers, text)`")
    st.code("A = model.predict(numbers, text=None)      # forecast without the report\n"
            "B = model.predict(numbers, text=report)    # forecast with the report\n"
            "for each claim in the report:              # judge: does it say up or down? how sure?\n"
            "    compare with the trend in the numbers  # disagree strongly -> low trust g\n"
            "final = A + g * (B - A)                    # g = 1 trust fully, g = 0 ignore the report\n"
            "warn  = applied weight < 0.5",
            language="python")
    st.markdown(
        "Training-free, inference-time and black-box: no retraining, only the model's outputs. "
        "**v1** asks one question about the whole report; **v2** checks every sentence or headline "
        "against evidence in its own frame and caps how far text may move the forecast.\n\n"
        "**DSS** (directional shift score) = d · (forecast with fake − forecast with genuine) / σ, "
        "where d = the direction the fake claims. Positive = moved toward the lie.")
    st.caption("Numbers from PROJECT_STATUS.md §9 / `results/summary_*.csv`. Use the sidebar to walk "
               "through the attack and the defence on real examples.")


# =============================================================================================
# 2 · the attack
# =============================================================================================

def page_attack(e: dict) -> None:
    header(e)
    f, d = e["forecast"], e["claimed"]
    dss = d * (f["fake"] - f["real"]) / e["sigma"]
    st.markdown(f"**Attack:** {ATTACK_LABEL[e['fake_cond']]}. The fake claims the series will go "
                f"**{word(d)}**.")
    left, right = st.columns([3, 2])
    with left:
        st.altair_chart(chart(e, [("true value", e["y_true"], "#222222"),
                                  ("Model 5, no text", f["none"], "#9aa5b1"),
                                  ("Model 5 + genuine report", f["real"], "#2f6fd6"),
                                  ("Model 5 + synthetic fake", f["fake"], "#d6453d")]),
                        width="stretch")
    with right:
        st.metric("Forecast with the genuine report", fmt(e, f["real"]))
        st.metric("Forecast with the synthetic fake", fmt(e, f["fake"]),
                  delta=f"{f['fake'] - f['real']:+,.{DECIMALS.get(e['domain'], 2)}f}", delta_color="off")
        st.metric("True value", fmt(e, e["y_true"]))
        st.metric("DSS (this window)", sig(dss),
                  help="d · (ŷ_fake − ŷ_genuine) / σ. Positive = moved toward the lie.")
        st.caption("One window. Aggregates over all windows are on screen 4.")
    tx = texts_of(e)
    if tx is None:
        missing_texts_notice()
        return
    real_h, fake_h = diff_html(tx["real"], tx["fake"])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Genuine report**")
        text_box(real_h)
        st.caption("Blue, struck through: words the attack removed or changed.")
    with c2:
        st.markdown(f"**Fake report** &nbsp; {synth_badge()}", unsafe_allow_html=True)
        text_box(fake_h)
        st.caption("Red: words the attack added or changed. Everything else is the genuine text.")
    attribution(e)


# =============================================================================================
# 3 · the defence
# =============================================================================================

def _unit_class(g: float | None, checked: bool) -> str:
    if not checked:
        return "tg-grey"
    return "tg-green" if g >= 0.999 else ("tg-amber" if g >= WARN else "tg-red")


def _check_line(k: dict) -> str:
    what = {"change": "says", "outlook": "predicts", "level": "level"}[k["kind"]]
    ev = "no evidence" if k["evidence"] == 0 else f"numbers: {word(k['evidence'])} (strength {k['strength']:.2f})"
    return (f"{what} <b>{word(k['direction'])}</b> (judge conf. {k['confidence']:.2f}) · {ev} "
            f"→ trust {k['g']:.2f}")


def trust_v2(e: dict, which: str, units_text: list[str] | None) -> None:
    rows = e["v2"][which]["units"]
    for i, r in enumerate(rows):
        body = html.escape(units_text[i]) if units_text else f"<i>unit {i + 1} (text not stored)</i>"
        notes = [_check_line(k) for k in r["checks"]]
        if not r["checks"]:
            notes.append("no claim the judge was sure of (confidence below 0.8): not checked"
                         if r["ignored"] else "no directional claim: not checked")
        cls = _unit_class(r["g"], bool(r["checks"]))
        st.markdown(f'<div class="tg-unit {cls}">{body}<br><small>{"<br>".join(notes)}</small></div>',
                    unsafe_allow_html=True)


def trust_v1(e: dict, which: str, text: str | None) -> None:
    v = e["v1"][which]
    body = html.escape(text).replace("\n", "<br>") if text else "<i>report text not stored</i>"
    notes = [_check_line(k) for k in v["checks"]] or ["no directional claim: not checked"]
    cls = _unit_class(v["g"], bool(v["checks"]))
    st.markdown(f'<div class="tg-unit {cls}" style="max-height:340px;overflow-y:auto">{body}<br>'
                f'<small>v1 reads the whole report as one claim: {"; ".join(notes)}</small></div>',
                unsafe_allow_html=True)


def page_defence(e: dict) -> None:
    header(e)
    default = 1 if e["id"] == "fnspid_fbr" else 0
    which_lab = st.radio("Text fed to the model", ["Synthetic fake", "Genuine report"], index=default,
                         horizontal=True)
    which = "fake" if which_lab.startswith("Synthetic") else "real"
    opts = list(DEFENCES) if e["v2_available"] else ["off", "v1"]
    dfn = st.radio("Defence", opts, index=len(opts) - 1, format_func=DEFENCES.get, horizontal=True)
    if not e["v2_available"]:
        st.caption("TextGuard v2 is not shown for this domain: v2 was developed on the original domains' "
                   "validation split and, by the project's rules, is never evaluated on their test split.")

    f = e["forecast"]
    res = e["defended"][which][dfn]
    B = f["fake"] if which == "fake" else f["real"]
    warn = res["weight"] < WARN

    left, right = st.columns([3, 2])
    with left:
        st.altair_chart(chart(e, [("true value", e["y_true"], "#222222"),
                                  ("A: no text", f["none"], "#9aa5b1"),
                                  (f"B: with {'fake' if which == 'fake' else 'genuine report'}", B,
                                   "#d6453d" if which == "fake" else "#2f6fd6"),
                                  (f"final ({DEFENCES[dfn]})", res["final"], "#2e9e57")]),
                        width="stretch")
    with right:
        st.metric(f"Final forecast — {DEFENCES[dfn]}", fmt(e, res["final"]))
        st.metric("Weight given to the text", f"{res['weight']:.2f}",
                  help="(final − A) / (B − A): 1 = text fully used, 0 = ignored. v2's 1σ cap counts too.")
        if dfn == "off":
            st.info("No defence: the model's forecast with the text is used as is.")
        elif warn:
            st.error("⚠ Warning: the text was mostly discounted (weight < 0.5)."
                     + (" This is a **genuine** report: a false block." if which == "real" else ""))
        else:
            st.success("No warning: the text was mostly kept (weight ≥ 0.5)."
                       + (" The fake got through." if which == "fake" else ""))
        st.metric("True value", fmt(e, e["y_true"]))

    tx = texts_of(e)
    st.markdown(f"#### Per-sentence trust — {DEFENCES[dfn]}"
                + (f" &nbsp; {synth_badge()}" if which == "fake" else ""), unsafe_allow_html=True)
    if tx is None:
        missing_texts_notice()
    text = tx[which] if tx else None
    if dfn == "off":
        if text:
            text_box(html.escape(text).replace("\n", "<br>"))
        st.caption("Off: nothing is checked.")
    elif dfn == "v1":
        trust_v1(e, which, text)
    else:
        trust_v2(e, which, tx["units"][which] if tx else None)
        st.caption("v2 checks every sentence / headline; the least trusted one sets the gate "
                   f"(g = {e['v2'][which]['g']:.2f}), then the text's effect is capped at 1σ.")
    st.caption("Green = agrees with the numbers · amber = partly discounted · red = trust < 0.5 · "
               "grey = not checked.")
    attribution(e)

    st.markdown("#### This window under every defence")
    rows = []
    for k in opts:
        rf, ff = e["defended"]["real"][k], e["defended"]["fake"][k]
        rows.append({"defence": DEFENCES[k],
                     "final (genuine)": fmt(e, rf["final"]), "weight (genuine)": f"{rf['weight']:.2f}",
                     "final (fake)": fmt(e, ff["final"]), "weight (fake)": f"{ff['weight']:.2f}",
                     "DSS": sig(e["claimed"] * (ff["final"] - rf["final"]) / e["sigma"]),
                     "error with genuine (σ)": f"{abs(rf['final'] - e['y_true']) / e['sigma']:.3f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Settings, frozen before testing — v1: {e['settings']['v1']}"
               + (f"; v2: {e['settings']['v2']}" if e["settings"]["v2"] else "")
               + ". One window; aggregates are on screen 4.")


# =============================================================================================
# 4 · results
# =============================================================================================

def page_results() -> None:
    st.title("Results")
    st.markdown("#### Who can be fooled? DSS under the flip attack, per model")
    m = pd.DataFrame(D["models"])
    bars = alt.Chart(m).mark_bar().encode(
        y=alt.Y("model:N", sort=list(m.model), title=None),
        x=alt.X("dss:Q", title="DSS (σ toward the lie), pooled, original domains"),
        color=alt.Color("reads:N", title="How the model reads the report",
                        legend=alt.Legend(orient="bottom", columns=2)),
        tooltip=["model", "reads", "source", alt.Tooltip("dss:Q", format="+.3f"),
                 alt.Tooltip("flip:Q", format=".1%", title="flip rate"), "n"])
    err = alt.Chart(m).mark_rule().encode(y=alt.Y("model:N", sort=list(m.model)), x="dss_lo:Q", x2="dss_hi:Q")
    st.altair_chart((bars + err).properties(height=360), width="stretch")
    st.markdown(
        "**The \"immune\" models ignore the text.** Models that pool the report into a BERT or Qwen "
        "embedding barely move (DSS ≈ 0). They aren't resisting the lie: the embedding doesn't carry "
        "the report's direction. A direction-aware encoder (FinBERT) makes the same late-fusion heads "
        "attackable, and the LLMs that read the report in their prompt are moved most. Phi-3.5 is "
        "the exception: it ignores what the report says. **Being vulnerable to text attacks is the "
        "price of actually using the text.**")
    st.caption("Bars: pooled DSS; lines: 95% bootstrap CI. Sources: `summary_p5_attack.csv`, "
               "`summary_r1_attack.csv`, `summary_r2_attack.csv`.")

    st.divider()
    st.markdown("#### The defence trade-off — held-out Energy + FNSPID, Model 5 (R4)")
    t = pd.DataFrame(D["tradeoff"])
    cl = t[t.defence.str.startswith("clip")].iloc[0]
    show = pd.DataFrame({"defence": t.defence,
                         "DSS · flip": t.dss_flipped.map(sig), "DSS · A2": t.dss_a2.map(sig),
                         "False Block Rate": t.fbr.map(pct)})
    st.dataframe(show, hide_index=True, width="stretch")
    st.markdown(
        f"v2 nearly closes the A2 gap that v1 left open ({H['r4_a2_tg1']:+.3f}σ → {H['r4_a2_tg2']:+.3f}σ), "
        f"but it blocks **{pct(H['r4_fbr_tg2'])}** of genuine reports ({pct(H['r4_fbr_tg2_fnspid'])} on "
        f"FNSPID) against v1's {pct(H['r4_fbr_tg1'])}: over the 10% budget. clip (k 0.5) sits in "
        f"between ({sig(cl.dss_a2)} on A2, {pct(cl.fbr)} FBR). No setting is free: the FNSPID example on "
        "screens 2–3 shows what a false block looks like.")
    st.caption("Source: `summary_r4_operating.csv`, `summary_r4_fbr.csv`. FBR = share of genuine "
               "reports given weight < 0.5.")

    st.divider()
    st.markdown("#### Paper figures")
    tabs = st.tabs(["P5 · DSS", "P5 · Flip Rate", "P6 · trade-off", "R3 · adaptive attacks",
                    "R4 · DSS", "R4 · FBR"])
    for tab, name in zip(tabs, ["p5_dss", "p5_flip_rate", "p6_tradeoff", "r3_dss", "r4_dss", "r4_fbr"]):
        with tab:
            st.image(str(FIGS / f"{name}.png"), width="stretch")


if page.startswith("1"):
    page_problem()
elif page.startswith("2"):
    page_attack(E)
elif page.startswith("3"):
    page_defence(E)
else:
    page_results()

st.divider()
st.caption("TextGuard — final-year MMAI project, Symbiosis Institute of Technology. Research demo: "
           "fake texts are synthetic and for defence research only. FNSPID material: CC BY-NC 4.0, "
           "non-commercial.")
