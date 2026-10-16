"""Export TextGuard v2's VAL judge job — original domains, val split only (R4 development).

v2 is developed on the original domains' **val** split only (PROJECT_STATUS §1a, §10g): never on
their test split (its P5/P6/R1–R3 results are known and taught v2's design) and never on the
held-out domains (Energy, FNSPID — R4's only test). This export is the guard: it reads only
``split == "val"`` rows of the four original domains and refuses anything else.

Rows (``row_type``), one CSV so Kaggle checks one sha256:

* ``judge`` — one per distinct (domain, question, unit) over every val text of ``real``,
  ``flipped``, ``exaggerated`` and R3's ``a1_selective``, ``a2_forward``, ``a3_magnitude``
  (``units.units``; ``v2.questions_for``: change + outlook, + level where a baseline is stated);
  plus the units of the P1 **human-labelled** reports of the four domains (all dated before any
  test window — asserted), for judge validation. Energy's human labels are skipped (held out).
* ``probe`` — P5's 40 reproducibility prompts (setup class, recorded only), as in P6 / R3.

Output: ``results/qwen/r4v2_val_inputs.csv`` (gitignored: raw text and synthetic fakes).
Run: ``python -m src.textguard.export_v2_inputs``
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.attacks import common
from src.models.export_qwen_inputs import INPUTS_P5
from src.textguard.export_judge_inputs import HUMAN_LABELS, PROBE_COLS
from src.textguard.judge_prompt_v2 import SUBJECTS, build_judge_prompt, judge_sha
from src.textguard.units import units
from src.textguard.v2 import QWEN_DIR, questions_for

V2_VAL_INPUTS = QWEN_DIR / "r4v2_val_inputs.csv"
DOMAINS = tuple(SUBJECTS)                         # the four original domains, nothing else
CONDITIONS = ("real", "flipped", "exaggerated", "a1_selective", "a2_forward", "a3_magnitude")
COLUMNS = ["row_type", "domain", "question", "text", "prompt_sha", "sources"] + PROBE_COLS


def val_texts(conditions=CONDITIONS) -> pd.DataFrame:
    """(domain, text, source) for every non-empty VAL text of the original domains."""
    parts = []
    for cond in conditions:
        t = common.load(cond)
        t = t[(t.split == "val") & t.domain.isin(DOMAINS) & (t.text.str.strip().str.len() > 0)]
        parts.append(pd.DataFrame({"domain": t.domain, "text": t.text, "source": cond}))
    return pd.concat(parts, ignore_index=True)


def human_texts() -> pd.DataFrame:
    """P1's human-labelled reports of the original domains, each dated before that domain's test."""
    real = common.load("real")
    first_test = real[real.split == "test"].groupby("domain").sample_id.min()
    parts = []
    for p in HUMAN_LABELS:
        h = pd.read_csv(p, dtype=str)
        h = h[h.domain.isin(DOMAINS) & h.fact.notna()]
        late = h[h.start_date >= h.domain.map(first_test)]
        if len(late):
            raise ValueError(f"{len(late)} human-labelled reports fall in a test period")
        parts.append(pd.DataFrame({"domain": h.domain, "text": h.fact,
                                   "source": "human:" + Path(p).stem}))
    return pd.concat(parts, ignore_index=True)


def unit_rows(texts: pd.DataFrame) -> pd.DataFrame:
    rows = [(d, u, s) for d, t, s in zip(texts.domain, texts.text, texts.source) for u in units(t)]
    u = pd.DataFrame(rows, columns=["domain", "text", "source"])
    return (u.groupby(["domain", "text"], sort=False)["source"]
            .agg(lambda s: ";".join(dict.fromkeys(s))).rename("sources").reset_index())


def build(texts: pd.DataFrame, probes: pd.DataFrame) -> pd.DataFrame:
    if not set(texts.domain) <= set(DOMAINS):
        raise ValueError("only the original domains may be exported for v2 development")
    u = unit_rows(texts)
    judge = pd.DataFrame([dict(row_type="judge", domain=d, question=k, text=t,
                               prompt_sha=judge_sha(build_judge_prompt(d, k, t)), sources=s)
                          for d, t, s in zip(u.domain, u.text, u.sources)
                          for k in questions_for(t)])
    if judge.prompt_sha.duplicated().any():
        raise ValueError("two judge questions share a fingerprint")
    probe = probes[["domain", "text"] + PROBE_COLS].assign(row_type="probe")
    df = pd.concat([judge, probe], ignore_index=True)
    for c in COLUMNS:
        if c not in df:
            df[c] = ""
    return df[COLUMNS].fillna("")


def sha256(path: Path = V2_VAL_INPUTS) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(path: Path = V2_VAL_INPUTS) -> pd.DataFrame:
    p5 = pd.read_csv(INPUTS_P5, keep_default_na=False, dtype={"sample_id": str})
    probes = p5[p5.probe.astype(str) == "True"]
    if len(probes) != 40:
        raise ValueError(f"expected P5's 40 probe rows, found {len(probes)}")
    df = build(pd.concat([val_texts(), human_texts()], ignore_index=True), probes)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def main() -> None:
    df = export()
    j = df[df.row_type == "judge"]
    print(j.groupby(["domain", "question"]).size().rename("questions").to_string())
    print(f"{len(j)} judge questions over {j[['domain', 'text']].drop_duplicates().shape[0]} units + "
          f"{int((df.row_type == 'probe').sum())} probe prompts -> {V2_VAL_INPUTS} "
          f"({V2_VAL_INPUTS.stat().st_size / 1024:.0f} KB), sha256 {sha256()[:16]}")


if __name__ == "__main__":
    main()
