"""Export every val + test window for Model 5 (Qwen) into ONE self-contained CSV for Colab.

Why export instead of running our pipeline on Colab: Qwen needs a GPU (free Colab T4), and
the Colab runtime cannot see this laptop (issue #5). Exporting the exact windows means Colab
needs nothing from our code — no loader, no splits, no Time-MMD clone — and Model 5 is
guaranteed to forecast the *same* samples as Models 1–4 (same chronological splits, same
lookback, same lag-1 / end-anchored text alignment).

Why only val + test: Qwen is used zero-shot (no training), so train windows are not needed.
Val is kept so any Qwen-side choice (prompt wording, parser) can be made on val, never test.

One row per window. Columns:

* ``domain, split, sample_id`` — sample_id is the target period's start date, the same id
  used in every other predictions CSV (``windows.Windows.target_start``)
* ``freq, description, unit`` — what the series is, in words, for the prompt. **No dates**:
  the prompt labels values t-L … t-1 to reduce recall of memorised historical values
* ``lookback`` — JSON list of the L most recent values, raw units, oldest first
* ``text`` — the aligned report text (empty string when the period has no report)
* ``y_true`` — the target, raw units
* ``train_mean, train_std`` — the train split's scaler, so normalised MSE can be computed
  exactly as for Models 1–4 without our pipeline

Output: ``results/qwen/qwen_inputs.csv`` (gitignored: raw data).
Run: ``python -m src.models.export_qwen_inputs``

**P5a — the altered conditions (issue #29):** ``python -m src.models.export_qwen_inputs --p5``
writes ``results/qwen/qwen_inputs_p5.csv``: the same columns plus ``condition``, one row per
(window, condition) for shuffled, cross_domain, time_shifted, flipped and exaggerated, with
``text`` = that condition's text from ``texts/<condition>.csv``. Extra columns:

* ``reuse`` — ``"real"`` when the condition's text is identical to the real report (e.g. a
  report the flip rules could not attack), ``"none"`` when it is empty. Those prompts are
  *byte-identical* to a P2.4 prompt, so the notebook can skip them and P2.4's prediction is
  reused (``qwen_forecaster.load_all_predictions``) — a shorter Colab run, same answers.
* **baseline rows** — every window's ``real`` and ``none`` prompt (986 rows, ``reuse`` =
  ``"p2"``): P2.4 already answered them, so they are skipped too — *unless the probe fails*.
* ``probe`` / ``expected_raw`` — 40 of the baseline rows (val only) re-run first as a
  **reproducibility probe**, compared with P2.4's raw output. Reuse is only valid if today's
  Colab stack reproduces P2.4's greedy answers exactly. **If any probe differs, the notebook
  regenerates every row — the 986 baseline rows included** — so that DSS never compares a
  fake-text answer from today's stack with a real-text answer from P2.4's: a software change
  would otherwise be counted as attack effect.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.splits import load_splits
from src.data.windows import DOMAIN_FREQ, lookback_for, split_windows
from src.models.train_baselines import DOMAINS

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "qwen"
INPUTS = OUT_DIR / "qwen_inputs.csv"
INPUTS_P5 = OUT_DIR / "qwen_inputs_p5.csv"
P2_PREDICTIONS = OUT_DIR / "p2_qwen_predictions.csv"
P5_CONDITIONS = ("shuffled", "cross_domain", "time_shifted", "flipped", "exaggerated")
PROBES_PER_DOMAIN = 5

# What each target column is (verified against the Time-MMD / MM-TSFlib source files and the
# report texts). Deliberately undated and unbranded beyond what the reports themselves say.
# Economy's target is the GOODS balance (May 2015: −58,365), while its reports quote the goods
# AND services deficit ("$41.9 billion in May"); the ~$19B gap is the services surplus. So a
# report's headline number is not the series value — Qwen must not copy it (PROJECT_STATUS §7b).
SERIES = {
    "Economy": ("United States monthly trade balance in goods (exports minus imports; "
                "negative = deficit)",
                "million US dollars"),
    "SocialGood": ("United States monthly unemployment rate, not seasonally adjusted",
                   "percent"),
    "Traffic": ("United States monthly vehicle miles traveled on all roads",
                "million vehicle miles"),
    "Health_US": ("United States weekly share of outpatient visits for influenza-like illness "
                  "(unweighted)", "percent"),
}


def export(domains=DOMAINS, path: Path = INPUTS) -> pd.DataFrame:
    df = build_inputs(domains)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def build_inputs(domains=DOMAINS) -> pd.DataFrame:
    """One row per val + test window with its real report — the P2.4 table."""
    rows = []
    for domain in domains:
        splits = load_splits(domain)
        train_values = splits.train["value"].to_numpy(float)
        mean, std = float(train_values.mean()), float(train_values.std())
        w = split_windows(splits, lookback_for(domain), horizon=1)
        desc, unit = SERIES[domain]
        for split in ("val", "test"):
            win = w[split]
            for i in range(len(win)):
                rows.append({
                    "domain": domain, "split": split,
                    "sample_id": win.target_start[i].strftime("%Y-%m-%d"),
                    "freq": DOMAIN_FREQ[domain], "description": desc, "unit": unit,
                    "lookback": json.dumps([float(v) for v in win.X[i]]),
                    "text": win.text[i] or "",
                    "y_true": float(win.Y[i, 0]),
                    "train_mean": mean, "train_std": std,
                })
    df = pd.DataFrame(rows)
    if df.duplicated(["domain", "split", "sample_id"]).any():
        raise ValueError("duplicate (domain, split, sample_id) — sample ids must be unique")
    return df


def _same_text(a: str, b: str) -> bool:
    """Same prompt? ``build_prompt`` strips the report and drops a blank one."""
    return str(a).strip() == str(b).strip()


def export_conditions(conditions=P5_CONDITIONS, domains=DOMAINS, path: Path = INPUTS_P5,
                      p2_predictions: Path = P2_PREDICTIONS, n_probe: int = PROBES_PER_DOMAIN
                      ) -> pd.DataFrame:
    """The P5a Qwen input table: baseline (real / none) rows + every (window, altered
    condition) row. Probes are a flagged subset of the baseline rows."""
    from src.attacks.common import load
    base = build_inputs(domains)
    keys = ["domain", "split", "sample_id"]
    parts = [baseline(base, p2_predictions, n_probe)]
    for cond in conditions:
        t = load(cond)[keys + ["text"]].rename(columns={"text": "cond_text"})
        m = base.merge(t, on=keys, how="left", validate="one_to_one", indicator=True)
        if (m["_merge"] != "both").any():
            raise ValueError(f"texts/{cond}.csv lacks some exported windows — re-run "
                             "python -m src.attacks.generate")
        m["reuse"] = ["none" if not str(c).strip() else "real" if _same_text(c, r) else ""
                      for c, r in zip(m.cond_text, m.text)]
        m["text"] = m["cond_text"].fillna("")
        parts.append(m.drop(columns=["cond_text", "_merge"])
                     .assign(condition=cond, probe=False, expected_raw=""))
    df = pd.concat(parts, ignore_index=True)
    df = df[["domain", "split", "sample_id", "condition", "freq", "description", "unit",
             "lookback", "text", "y_true", "train_mean", "train_std", "reuse", "probe",
             "expected_raw"]]
    if df.duplicated(keys + ["condition"]).any():
        raise ValueError("duplicate (domain, split, sample_id, condition) rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def baseline(base: pd.DataFrame, p2_predictions: Path, n_probe: int) -> pd.DataFrame:
    """Every window's real and none rows (``reuse="p2"``), with P2.4's raw output.

    The first ``n_probe`` val windows with a report per domain are flagged ``probe``. Val
    only: the probe tests the software stack, not a model choice, and never reads test.
    """
    p2 = pd.read_csv(p2_predictions, keep_default_na=False, dtype={"sample_id": str})
    keys = ["domain", "split", "sample_id"]
    probe_win = (base[(base.split == "val") & (base.text.str.strip().str.len() > 0)]
                 .groupby("domain", sort=False).head(n_probe)[keys])
    probe_set = set(map(tuple, probe_win.to_numpy()))
    out = []
    for cond in ("real", "none"):
        r = base.merge(p2[p2.condition == cond][keys + ["raw_output"]], on=keys,
                       how="left", validate="one_to_one")
        if r.raw_output.isna().any():
            raise ValueError("P2.4 predictions lack some windows")
        r["text"] = r["text"] if cond == "real" else ""
        r["probe"] = [k in probe_set for k in zip(r.domain, r.split, r.sample_id)]
        r["expected_raw"] = r["raw_output"].where(r["probe"], "")
        out.append(r.drop(columns="raw_output").assign(condition=cond, reuse="p2"))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    import sys
    if "--p5" in sys.argv[1:]:
        df = export_conditions()
        print(df.assign(generate=df.reuse == "").groupby(["condition", "split"]).agg(
            rows=("sample_id", "size"), to_generate=("generate", "sum"),
            reuse_p2=("reuse", lambda r: (r == "p2").sum()),
            reuse_real=("reuse", lambda r: (r == "real").sum()),
            reuse_none=("reuse", lambda r: (r == "none").sum())).to_string())
        n_gen = int((df.reuse == "").sum())
        print(f"{len(df)} rows ({n_gen} to generate if the probe passes, all {len(df)} if it "
              f"fails; {int(df.probe.sum())} probe rows) -> {INPUTS_P5} "
              f"({INPUTS_P5.stat().st_size / 1024:.0f} KB)")
        return
    df = export()
    summary = df.assign(has_text=df.text.str.len() > 0).groupby(["domain", "split"]).agg(
        windows=("sample_id", "size"), with_text=("has_text", "sum"))
    print(summary.to_string())
    print(f"{len(df)} windows -> {INPUTS} ({INPUTS.stat().st_size / 1024:.0f} KB); "
          f"Qwen prompts: {2 * len(df)} (real + none)")


if __name__ == "__main__":
    main()
