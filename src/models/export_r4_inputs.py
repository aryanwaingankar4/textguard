"""Export everything R4's four held-out Kaggle jobs need into ONE CSV (pre-registration §10g, "Runs").

One file, so Aryan uploads one Kaggle dataset and every notebook checks the same sha256
(``notebooks/build_r4_heldout_notebooks.py`` embeds it). Row types:

* ``forecast`` — one row per held-out window × condition, for M5 and Qwen2.5-7B: ``real``, ``none``,
  ``flipped``, ``a1_selective``, ``a2_forward`` (both domains) and ``a3_magnitude`` (Energy only).
  Everything is regenerated (real and none too) so no comparison mixes software setups. Domain key
  ``Energy`` / ``FNSPID:<TICKER>`` (stocks share dates); prompt fields from ``SERIES`` (§10g).
* ``judge`` — TextGuard **v1**'s questions: one row per distinct (domain key, text) over real and
  every attack (v1's frozen prompt + ``heldout_questions.HELDOUT_QUESTIONS``).
* ``judge_v2`` — TextGuard **v2**'s questions: one row per distinct (domain key, question, unit)
  over the same texts (``units.units``, ``v2.questions_for``; ``HELDOUT_SUBJECTS``).
* ``probe`` — P5's 40 reproducibility prompts (recorded only).

Guards: the texts must have the sha256 prefixes pre-registered in §10g (else refuse); only the
held-out domains are exported. Output: ``results/qwen/r4_heldout_inputs.csv`` (gitignored: report
text and synthetic fakes). Run after ``python -m src.attacks.heldout``:
``python -m src.models.export_r4_inputs``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.attacks.common import TEXTS_DIR
from src.data import heldout
from src.models.export_qwen_inputs import INPUTS_P5, OUT_DIR
from src.textguard import heldout_questions as hq
from src.textguard import judge_prompt as jp1
from src.textguard import judge_prompt_v2 as jp2
from src.textguard.units import units
from src.textguard.v2 import questions_for

R4_INPUTS = OUT_DIR / "r4_heldout_inputs.csv"
CONDITIONS = ("real", "flipped", "a1_selective", "a2_forward", "a3_magnitude")
# sha256 prefixes of texts/r4/*.csv as pre-registered (§10g, commit 1e13815)
TEXT_SHA = {"real": "514b5388e451cace", "flipped": "5509881b5beb3759", "a1_selective": "50f14b3680d2cac0",
            "a2_forward": "07119b20ca8a4b3f", "a3_magnitude": "652bb51659a7a56c"}
SERIES = {"Energy": ("United States weekly average retail price of regular gasoline", "US dollars per gallon"),
          **{f"FNSPID:{t}": (f"Weekly closing share price of {c} (ticker {t}), as traded", "US dollars")
             for t, c in hq.COMPANY_NAMES.items()}}
COLUMNS = ["row_type", "domain", "split", "sample_id", "condition", "freq", "description", "unit",
           "lookback", "text", "y_true", "train_mean", "train_std", "attacked", "question", "prompt_sha",
           "sources", "expected_raw"]


def sha256(path: Path = R4_INPUTS) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_texts() -> dict[str, pd.DataFrame]:
    out = {}
    for c in CONDITIONS:
        p = TEXTS_DIR / "r4" / f"{c}.csv"
        if not sha256(p).startswith(TEXT_SHA[c]):
            raise ValueError(f"{p.name} differs from the pre-registered texts (§10g) — do not export")
        t = pd.read_csv(p, keep_default_na=False, dtype={"sample_id": str})
        t["key"] = [hq.domain_key(d, s) for d, s in zip(t.domain, t.series)]
        out[c] = t
    return out


def forecast_rows(texts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    w = heldout.windows()
    w["key"] = [hq.domain_key(d, s) for d, s in zip(w.domain, w.series)]
    rows = []
    for cond in ("none",) + CONDITIONS:
        if cond == "none":
            tx, att = [""] * len(w), [False] * len(w)
        else:
            t = texts[cond]
            if list(zip(t.key, t.sample_id)) != list(zip(w.key, w.sample_id)):
                raise ValueError(f"{cond}: windows differ from heldout.windows()")
            if cond == "a3_magnitude":
                keep = (t.domain == "Energy").to_numpy()
            tx, att = list(t.text), list(t.attacked.astype(str) == "True")
        for i, r in enumerate(w.itertuples()):
            if cond == "a3_magnitude" and not keep[i]:
                continue                                   # A3 dropped on FNSPID (§10g)
            desc, unit = SERIES[r.key]
            rows.append(dict(row_type="forecast", domain=r.key, split="test", sample_id=r.sample_id,
                             condition=cond, freq="weekly", description=desc, unit=unit,
                             lookback=json.dumps([float(v) for v in r.lookback]), text=tx[i],
                             y_true=r.y_true, train_mean="", train_std=r.sigma, attacked=att[i]))
    return pd.DataFrame(rows)


def judge_rows(texts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    hq.register()
    seen: dict[tuple, list] = {}
    for cond, t in texts.items():
        for k, x in zip(t.key, t.text):
            if isinstance(x, str) and x.strip():
                seen.setdefault((k, x.strip()), []).append(cond)
    v1 = [dict(row_type="judge", domain=k, question=q, text=x,
               prompt_sha=jp1.judge_sha(jp1.build_judge_prompt(k, q, x)), sources=";".join(dict.fromkeys(c)))
          for (k, x), c in seen.items() for q, _ in jp1.QUESTIONS[k]]
    units_seen: dict[tuple, list] = {}
    for (k, x), c in seen.items():
        for u in units(x):
            units_seen.setdefault((k, u), []).extend(c)
    v2 = [dict(row_type="judge_v2", domain=k, question=q, text=u,
               prompt_sha=jp2.judge_sha(jp2.build_judge_prompt(k, q, u)), sources=";".join(dict.fromkeys(c)))
          for (k, u), c in units_seen.items() for q in questions_for(u)]
    df = pd.DataFrame(v1 + v2)
    if df.prompt_sha.duplicated().any():
        raise ValueError("two judge questions share a fingerprint")
    return df


def probe_rows() -> pd.DataFrame:
    p5 = pd.read_csv(INPUTS_P5, keep_default_na=False, dtype={"sample_id": str})
    p = p5[p5.probe.astype(str) == "True"]
    if len(p) != 40:
        raise ValueError(f"expected P5's 40 probe rows, found {len(p)}")
    return p.assign(row_type="probe")


def build() -> pd.DataFrame:
    texts = load_texts()
    df = pd.concat([forecast_rows(texts), judge_rows(texts), probe_rows()], ignore_index=True)
    if not df[df.row_type != "probe"].domain.str.match(r"^(Energy|FNSPID:[A-Z]+)$").all():
        raise ValueError("only held-out domain keys may be exported")
    for c in COLUMNS:
        if c not in df:
            df[c] = ""
    return df[COLUMNS].fillna("")


def export(path: Path = R4_INPUTS) -> pd.DataFrame:
    df = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    # CRLF on every OS: the Kaggle runs recorded this file's sha256 as written on Windows (R7)
    df.to_csv(path, index=False, lineterminator="\r\n")
    return df


def main() -> None:
    df = export()
    f = df[df.row_type == "forecast"]
    print(f.groupby(["condition", f.domain.str.split(":").str[0]]).size().unstack().to_string())
    for rt in ("judge", "judge_v2"):
        j = df[df.row_type == rt]
        print(rt, j.groupby([j.domain.str.split(":").str[0], "question"]).size().to_dict())
    print(f"{len(f)} forecast rows ({f[['description', 'lookback', 'text']].drop_duplicates().shape[0]} distinct "
          f"prompts), {int((df.row_type == 'judge').sum())} v1 + {int((df.row_type == 'judge_v2').sum())} v2 judge "
          f"questions, {int((df.row_type == 'probe').sum())} probe -> {R4_INPUTS} "
          f"({R4_INPUTS.stat().st_size / 1024:.0f} KB), sha256 {sha256()[:16]}")


if __name__ == "__main__":
    main()
