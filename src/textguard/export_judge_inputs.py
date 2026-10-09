"""Export every distinct text TextGuard will judge into ONE CSV for the Colab judge run (P6a).

Why export: the judge is Qwen2.5-1.5B (GPU), and Colab cannot see this laptop (issue #5) — the
same reason Model 5 is exported. The notebook needs nothing from our pipeline.

What is exported (``row_type = "judge"``), one row per distinct (domain, question, text):

* every non-empty text of every val + test window under **every** condition in ``texts/``
  (real, shuffled, cross_domain, time_shifted, flipped, exaggerated). A text is judged with the
  questions of the domain it is *fed to* — a cross-domain text sent to Economy is asked about
  the trade deficit (it should answer "none"). Identical texts are judged once;
* the **human-labelled** P1 reports (``results/summary_directional_sample.csv``, 50; plus the
  25 filtered-SocialGood re-check reports — Energy is not one of our domains), so the judge can
  be validated against human labels in P6b (issue #25). Some predate val: that is fine, the
  judge is never tuned.

Health_US texts get two questions (level and change), the others one (``judge_prompt``).
``sources`` lists where a text occurs (e.g. ``real;time_shifted``) — informational only.

Plus the **reproducibility probe** (``row_type = "probe"``): the 40 Model 5 forecasting prompts
P5 used as its probe, with P2.4's raw answers. Re-running them on Colab certifies the judge ran
on the same software setup class as P5 (and pins the same revision).

Output: ``results/qwen/judge_inputs.csv`` (gitignored: raw text).
Run: ``python -m src.textguard.export_judge_inputs``
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.attacks.common import CONDITIONS, load
from src.models.export_qwen_inputs import INPUTS_P5
from src.textguard.judge_prompt import QUESTIONS, build_judge_prompt, judge_sha
from src.textguard.text_direction import JUDGE_INPUTS, ROOT

HUMAN_LABELS = [ROOT / "results" / "summary_directional_sample.csv",
                ROOT / "results" / "summary_directional_socialgood_energy.csv"]
PROBE_COLS = ["split", "sample_id", "condition", "freq", "description", "unit", "lookback",
              "expected_raw"]


def condition_texts(conditions=CONDITIONS) -> pd.DataFrame:
    """(domain, text, source) for every non-empty val/test text in ``texts/``."""
    parts = []
    for cond in conditions:
        t = load(cond)
        t = t[t.text.str.strip().str.len() > 0]
        parts.append(pd.DataFrame({"domain": t.domain, "text": t.text.str.strip(),
                                   "source": cond}))
    return pd.concat(parts, ignore_index=True)


def human_texts(paths=HUMAN_LABELS) -> pd.DataFrame:
    parts = []
    for p in paths:
        h = pd.read_csv(p)
        h = h[h.domain.isin(QUESTIONS) & h.fact.notna()]
        parts.append(pd.DataFrame({"domain": h.domain, "text": h.fact.str.strip(),
                                   "source": "human:" + Path(p).stem}))
    return pd.concat(parts, ignore_index=True)


def build(texts: pd.DataFrame, probes: pd.DataFrame) -> pd.DataFrame:
    texts = texts[texts.text.str.len() > 0]
    src = (texts.groupby(["domain", "text"], sort=False)["source"]
           .agg(lambda s: ";".join(dict.fromkeys(s))).reset_index())
    rows = [dict(row_type="judge", domain=d, question=kind, text=t,
                 prompt_sha=judge_sha(build_judge_prompt(d, kind, t)), sources=s)
            for d, t, s in zip(src.domain, src.text, src.source)
            for kind, _ in QUESTIONS[d]]
    judge = pd.DataFrame(rows)
    if judge.prompt_sha.duplicated().any():
        raise ValueError("two judge questions share a fingerprint")
    probe = probes[["domain", "text"] + PROBE_COLS].assign(row_type="probe")
    return pd.concat([judge, probe], ignore_index=True)[
        ["row_type", "domain", "question", "text", "prompt_sha", "sources"] + PROBE_COLS
    ].fillna("")


def export(path: Path = JUDGE_INPUTS, p5_inputs: Path = INPUTS_P5) -> pd.DataFrame:
    p5 = pd.read_csv(p5_inputs, keep_default_na=False, dtype={"sample_id": str})
    probes = p5[p5.probe.astype(str) == "True"]
    if len(probes) != 40:
        raise ValueError(f"expected P5's 40 probe rows, found {len(probes)}")
    df = build(pd.concat([condition_texts(), human_texts()], ignore_index=True), probes)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def main() -> None:
    df = export()
    j = df[df.row_type == "judge"]
    print(j.groupby(["domain", "question"]).size().rename("questions").to_string())
    by_src = j.assign(source=j.sources.str.split(";")).explode("source")
    print(by_src.groupby("source").size().rename("questions containing").to_string())
    print(f"{len(j)} judge questions ({j[['domain', 'text']].drop_duplicates().shape[0]} distinct "
          f"texts) + {int((df.row_type == 'probe').sum())} probe prompts -> {JUDGE_INPUTS} "
          f"({JUDGE_INPUTS.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
