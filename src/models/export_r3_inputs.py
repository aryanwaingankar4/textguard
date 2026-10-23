"""Export everything R3's three Kaggle jobs need into ONE CSV (PROJECT_STATUS §10f, "Runs").

One file, so Aryan uploads a single Kaggle dataset and every notebook checks the same sha256
(``notebooks/build_r3_notebooks.py`` embeds it). Three kinds of row (``row_type``):

* ``forecast`` — one row per val + test window × condition, for Model 5 and Qwen2.5-7B:
  ``real``, ``none`` (no report), P3's ``flipped`` and ``exaggerated`` (within-run reference
  rows) and R3's ``a1_selective``, ``a2_forward``, ``a3_magnitude``. **Everything is
  regenerated in the R3 run** — real and none included — because Kaggle's software stack is not
  P5's Colab stack, and a stack change must never be measured as attack effect (§2, §10f).
  Same columns as P5's ``qwen_inputs_p5.csv`` (lookback, unit, …) plus ``attacked``.
* ``judge`` — one row per distinct (domain, question, text) over real, flipped, exaggerated and
  the three R3 attacks (val + test), built by P6's own ``export_judge_inputs.build`` — so the
  question wording and fingerprints are P6's. Real texts are re-judged too: TextGuard's gates in
  R3 come from one judge run for genuine and fake texts alike.
* ``probe`` — P5's 40 reproducibility prompts with P2.4's raw answers (run by the M5 and the
  judge notebooks; recorded, gates nothing).

Output: ``results/qwen/r3_inputs.csv`` (gitignored: raw report text and synthetic fakes).
Run after ``python -m src.attacks.adaptive`` and ``validate_r3``:
``python -m src.models.export_r3_inputs``
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.attacks import common
from src.attacks.adaptive import R3_CONDITIONS
from src.models.export_qwen_inputs import INPUTS_P5, OUT_DIR, build_inputs
from src.textguard import export_judge_inputs as ej

R3_INPUTS = OUT_DIR / "r3_inputs.csv"
FORECAST_CONDITIONS = ("real", "none", "flipped", "exaggerated") + R3_CONDITIONS
JUDGE_CONDITIONS = ("real", "flipped", "exaggerated") + R3_CONDITIONS
COLUMNS = ["row_type", "domain", "split", "sample_id", "condition", "freq", "description", "unit",
           "lookback", "text", "y_true", "train_mean", "train_std", "attacked", "question",
           "prompt_sha", "sources", "expected_raw"]
KEYS = ["domain", "split", "sample_id"]


def forecast_rows() -> pd.DataFrame:
    base = build_inputs()
    parts = []
    for cond in FORECAST_CONDITIONS:
        if cond == "none":
            parts.append(base.assign(condition=cond, text="", attacked=False))
            continue
        t = common.load(cond)[KEYS + ["text", "attacked"]].rename(columns={"text": "cond_text"})
        m = base.merge(t, on=KEYS, how="left", validate="one_to_one", indicator=True)
        if (m["_merge"] != "both").any():
            raise ValueError(f"texts/{cond}.csv lacks some windows — re-run the generators")
        if cond == "real" and not (m.cond_text.str.strip() == m.text.str.strip()).all():
            raise ValueError("texts/real.csv differs from the windows' aligned reports")
        parts.append(m.drop(columns=["text", "_merge"]).rename(columns={"cond_text": "text"})
                     .assign(condition=cond))
    df = pd.concat(parts, ignore_index=True).assign(row_type="forecast")
    df["attacked"] = df["attacked"].astype(bool)
    if df.duplicated(KEYS + ["condition"]).any():
        raise ValueError("duplicate (window, condition) forecast rows")
    return df


def judge_and_probe_rows() -> pd.DataFrame:
    p5 = pd.read_csv(INPUTS_P5, keep_default_na=False, dtype={"sample_id": str})
    probes = p5[p5.probe.astype(str) == "True"]
    if len(probes) != 40:
        raise ValueError(f"expected P5's 40 probe rows, found {len(probes)}")
    return ej.build(ej.condition_texts(JUDGE_CONDITIONS), probes)


def build() -> pd.DataFrame:
    df = pd.concat([forecast_rows(), judge_and_probe_rows()], ignore_index=True)
    for c in COLUMNS:
        if c not in df:
            df[c] = ""
    return df[COLUMNS].fillna("")


def sha256(path: Path = R3_INPUTS) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(path: Path = R3_INPUTS) -> pd.DataFrame:
    df = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    # CRLF on every OS: the Kaggle runs recorded this file's sha256 as written on Windows (R7)
    df.to_csv(path, index=False, lineterminator="\r\n")
    return df


def main() -> None:
    df = export()
    f = df[df.row_type == "forecast"]
    print(f.groupby(["condition", "split"]).size().unstack().to_string())
    j = df[df.row_type == "judge"]
    print(j.groupby(["domain", "question"]).size().rename("judge questions").to_string())
    print(f"{len(f)} forecast rows, {len(j)} judge questions, "
          f"{int((df.row_type == 'probe').sum())} probe rows -> {R3_INPUTS} "
          f"({R3_INPUTS.stat().st_size / 1024:.0f} KB), sha256 {sha256()[:16]}")


if __name__ == "__main__":
    main()
