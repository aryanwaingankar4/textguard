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
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def main() -> None:
    df = export()
    summary = df.assign(has_text=df.text.str.len() > 0).groupby(["domain", "split"]).agg(
        windows=("sample_id", "size"), with_text=("has_text", "sum"))
    print(summary.to_string())
    print(f"{len(df)} windows -> {INPUTS} ({INPUTS.stat().st_size / 1024:.0f} KB); "
          f"Qwen prompts: {2 * len(df)} (real + none)")


if __name__ == "__main__":
    main()
