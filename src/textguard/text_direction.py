"""What does a report claim about the forecast target? — TextGuard's direction judge (P6).

TextGuard needs, for each text, the direction it claims and how sure we are of that reading.
Any object with ``__call__(text) -> list[Claim]`` is a judge (``DirectionJudge``): TextGuard
never looks inside it, so a different reader (a bigger LLM, a human-in-the-loop) can be
plugged in without touching the gate. A judge instance belongs to one domain, because the
question it asks names that domain's series.

Our judge, ``QwenJudge``, is Qwen2.5-1.5B-Instruct, **not** the P3 rules: the rules wrote the
fakes, so rules judging them would be circular (issue #25). Qwen needs a GPU, so — exactly
like Model 5 — it runs once on Colab over every distinct text
(``notebooks/p6_direction_judge.ipynb``, inputs from ``export_judge_inputs.py``) and this
module serves the answers from the CSV. A text that was not exported raises ``KeyError``: that
is a pipeline bug, never a silent "no claim".

From the judge's answer to a ``Claim``:

* **direction** is mapped from the literal subject to the forecast series: Economy's question
  is about the trade *deficit*, its target is the trade *balance* — deficit up = series −1
  (``POLARITY``). Other domains ask about the series itself.
* **confidence** = the chosen answer's probability renormalised over the three allowed
  answers (∈ [⅓, 1]), computed on Colab from the model's own token probabilities.
* **reference** (level claims only) = the baseline number the text states, read by a regex
  (``stated_baseline``). That is reading a *number*, not judging a direction, and the attack
  never edits it (the level-flip mirrors the value around it), so it is not circular.

Integrity at load (same pattern as ``qwen_forecaster``): every answer's prompt fingerprint is
recomputed from its own text with the *current* ``judge_prompt``; one software setup per file;
the reproducibility probe's outcome is carried on every row.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from src.textguard.judge_prompt import ANSWERS, QUESTIONS, build_judge_prompt, judge_sha

ROOT = Path(__file__).resolve().parents[2]
JUDGE_DIR = ROOT / "results" / "qwen"
JUDGE_INPUTS = JUDGE_DIR / "judge_inputs.csv"
JUDGE_PREDICTIONS = JUDGE_DIR / "p6_judge_predictions.csv"
JUDGE_META = JUDGE_DIR / "p6_judge_run_meta.json"

# literal subject -> forecast series. Economy: the reports talk about the deficit, the target
# is the balance (exports − imports), so "deficit increased" means the series went down.
POLARITY = {"Economy": -1, "Traffic": 1, "SocialGood": 1, "Health_US": 1}


@dataclass(frozen=True)
class Claim:
    """One claim a text makes about the forecast series."""
    kind: str                       # "change" | "level"
    direction: int                  # series direction: +1 up/above, −1 down/below, 0 none
    confidence: float               # in [0, 1]
    reference: float | None = None  # level claims: the reference level the text states


@runtime_checkable
class DirectionJudge(Protocol):
    """Anything that turns a text into claims about one domain's series."""

    def __call__(self, text: str | None) -> list[Claim]:
        ...


_BASELINE = re.compile(r"national\s+baseline(?:\s+level)?\s*(?:of|\(|was|is|=)?\s*"
                       r"(\d+(?:\.\d+)?)\s*(?:%|percent)", re.I)


def stated_baseline(text: str | None) -> float | None:
    """The national baseline the text states ("… the national baseline of 2.1%"), or None.

    Only the *national* baseline: Health_US reports also mention "region-specific baseline
    levels", which are not the series' reference (the series is national ILI).
    """
    if not isinstance(text, str):
        return None
    m = _BASELINE.search(text)
    return float(m.group(1)) if m else None


def answer_direction(kind: str, answer: str) -> int:
    """'up'/'above' → +1, 'down'/'below' → −1, 'none' → 0 (literal, before POLARITY)."""
    pos, neg, none = ANSWERS[kind]
    return {pos: 1, neg: -1, none: 0}[answer]


def text_key(domain: str, kind: str, text: str) -> str:
    """Join key of one judge question: the prompt fingerprint (domain, kind and text in it)."""
    return judge_sha(build_judge_prompt(domain, kind, text))


# --- loading the Colab output ---------------------------------------------------------------

def load_judgements(predictions: Path | str = JUDGE_PREDICTIONS, inputs: Path | str = JUDGE_INPUTS,
                    meta: Path | str = JUDGE_META) -> pd.DataFrame:
    """Every judge answer joined with its text, fingerprints verified, one setup, complete.

    Columns: domain, question, text, answer, confidence, p_pos, p_neg, p_none, literal
    (±1/0 about the subject), direction (±1/0 about the series), raw_output, prompt_sha,
    setup, probe_ok.
    """
    inp = pd.read_csv(inputs, keep_default_na=False, dtype=str)
    inp = inp[inp.row_type == "judge"]
    out = pd.read_csv(predictions, keep_default_na=False, dtype={"prompt_sha": str})
    run = json.loads(Path(meta).read_text(encoding="utf-8"))
    if out.duplicated("prompt_sha").any():
        raise ValueError("duplicate judge answers for one prompt")
    df = inp.merge(out.drop(columns=["domain", "question"], errors="ignore"),
                   on="prompt_sha", how="left", validate="one_to_one", indicator=True)
    if (df["_merge"] != "both").any():
        raise ValueError(f"{int((df['_merge'] != 'both').sum())} exported questions have no "
                         "judge answer — finish the Colab run")
    if len(out) != len(inp):
        raise ValueError("judge CSV has answers for prompts that were not exported")
    recomputed = [text_key(d, q, t) for d, q, t in zip(df.domain, df.question, df.text)]
    if list(recomputed) != list(df.prompt_sha):
        raise ValueError("judge prompt changed since the Colab run (fingerprint mismatch)")
    if df.setup.nunique() != 1 or df.setup.iloc[0] != run.get("setup"):
        raise ValueError("judge answers come from more than one software setup")
    df["confidence"] = df["confidence"].astype(float)
    for c in ("p_pos", "p_neg", "p_none"):
        df[c] = df[c].astype(float)
    df["literal"] = [answer_direction(q, a) for q, a in zip(df.question, df.answer)]
    df["direction"] = df["literal"] * df["domain"].map(POLARITY)
    df["probe_ok"] = bool(run.get("probe_ok"))
    return df.drop(columns=["_merge", "row_type"])


class QwenJudge:
    """The Qwen direction judge for one domain, served from the Colab CSV."""

    def __init__(self, domain: str, data: pd.DataFrame | None = None):
        if domain not in QUESTIONS:
            raise ValueError(f"no judge questions for domain {domain!r}")
        self.domain = domain
        df = load_judgements() if data is None else data
        df = df[df.domain == domain]
        self._answers = {(q, t.strip()): (int(d), float(c))
                         for q, t, d, c in zip(df.question, df.text, df.direction, df.confidence)}

    def __call__(self, text: str | None) -> list[Claim]:
        if not isinstance(text, str) or not text.strip():
            return []
        claims = []
        for kind, _ in QUESTIONS[self.domain]:
            key = (kind, text.strip())
            if key not in self._answers:
                raise KeyError(f"{self.domain}/{kind}: text was not judged on Colab "
                               "(re-export judge inputs and re-run the notebook)")
            d, c = self._answers[key]
            ref = stated_baseline(text) if kind == "level" else None
            claims.append(Claim(kind, d, c, ref))
        return claims


def fixed_judge(*claims: Claim) -> DirectionJudge:
    """A judge that returns the same claims for every non-empty text (tests, worked example)."""
    def judge(text: str | None) -> list[Claim]:
        return list(claims) if isinstance(text, str) and text.strip() else []
    return judge


def judge_from(fn, kind: str = "change") -> DirectionJudge:
    """Adapt any ``fn(text) -> (direction, confidence)`` into a judge (one change claim)."""
    def judge(text: str | None) -> list[Claim]:
        if not isinstance(text, str) or not text.strip():
            return []
        d, c = fn(text)
        return [Claim(kind, int(np.sign(d)), float(c))]
    return judge
