"""P1.5 directional check: do the reports make up/down claims the flip attack can invert?

The flipped-text attack (P3) rewrites "rose" -> "fell". If reports mostly state levels
("the rate was 5.3 percent") rather than changes, there is nothing to flip and the core
attack is not viable. PHASES.md sets the bar at >= 30% of reports making a clear claim.

Sampling choices:
* **Individual reports**, not per-period concatenations — a report is the unit an attacker
  would rewrite.
* **Stratified across the four selected domains** (13/13/12/12) so the weekly Health_US,
  which has the most reports, does not dominate the estimate.
* **Only reports dated before each domain's test split.** This check decides whether the
  attack is viable, which is a design decision, so the test period is not looked at.
* Fixed seed, so the same 50 reports are drawn every time.

Labels:
* ``heuristic`` — keyword lexicon: ``up``, ``down``, ``mixed`` (both), ``none``. Fast,
  reproducible, but blind to meaning (e.g. "deficit increased" means the balance fell).
* ``manual`` — a human judgement, filled into the saved CSV. The recorded viability % uses
  the manual labels; the heuristic is reported alongside so its agreement is visible,
  since P3's ``validate_attacks.py`` will need an automatic direction check anyway.

Output: prints every sampled report, and writes
``results/summary_directional_sample.csv`` (tracked in git as the audit trail).

Note: ``summary_directional_sample.csv`` was drawn *before* SocialGood's influenza texts were
filtered (5 of its 13 SocialGood rows are that junk). It is kept as the record of that run;
the filtered-SocialGood estimate lives in ``summary_directional_socialgood_energy.csv``.

Run:  python -m src.data.check_directional
      python -m src.data.check_directional --domains SocialGood Energy --per-domain 25           --out results/summary_directional_socialgood_energy.csv
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

from src.data.loader import load_text
from src.data.splits import load_splits

SELECTED_DOMAINS = ("Economy", "SocialGood", "Traffic", "Health_US")
N_SAMPLES = 50
SEED = 0
OUT = Path(__file__).resolve().parents[2] / "results" / "summary_directional_sample.csv"

UP = re.compile(r"\b(increas\w*|rose|rise[sn]?|rising|higher|grew|grow\w*|gain\w*|climb\w*|"
                r"jump\w*|surg\w*|widen\w*|rebound\w*|up)\b", re.I)
DOWN = re.compile(r"\b(decreas\w*|fell|fall\w*|declin\w*|lower|drop\w*|dipped|down|"
                  r"shr[ai]nk\w*|narrow\w*|plung\w*|reduc\w*|contract\w*)\b", re.I)


def heuristic_direction(text: str) -> str:
    up, down = bool(UP.search(text)), bool(DOWN.search(text))
    return "mixed" if up and down else "up" if up else "down" if down else "none"


def sample_reports(domains=SELECTED_DOMAINS, n: int = N_SAMPLES, seed: int = SEED,
                   per_domain: int | None = None) -> pd.DataFrame:
    """``n`` reports split as evenly as possible over ``domains``, or ``per_domain`` each."""
    counts = ([per_domain] * len(domains) if per_domain else
              [n // len(domains) + (i < n % len(domains)) for i in range(len(domains))])
    parts = []
    for domain, k in zip(domains, counts):
        test_start = load_splits(domain).test["start_date"].min()
        reports = load_text(domain)
        pool = reports[reports["end_date"] < test_start]
        parts.append(pool.sample(n=k, random_state=seed).assign(domain=domain))
    out = pd.concat(parts, ignore_index=True)[["domain", "start_date", "end_date", "fact"]]
    out["heuristic"] = out["fact"].map(heuristic_direction)
    out["manual"] = ""
    return out


def summarise(df: pd.DataFrame, column: str) -> str:
    counts = df[column].value_counts()
    claim = df[column].isin(["up", "down", "mixed"]).mean()
    clear = df[column].isin(["up", "down"]).mean()
    return (f"{column}: {counts.to_dict()} -> any directional claim {claim:.0%}, "
            f"single clear direction {clear:.0%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--domains", nargs="+", default=list(SELECTED_DOMAINS))
    ap.add_argument("--per-domain", type=int, default=None,
                    help="reports per domain (default: 50 split evenly)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out
    logging.disable(logging.WARNING)
    sys.stdout.reconfigure(encoding="utf-8")

    if out.exists() and (existing := pd.read_csv(out))["manual"].notna().any():
        labelled = existing[existing["manual"].notna()]
        print(f"{out.name} already has manual labels — not overwriting. Summary:")
        print(" ", summarise(existing, "heuristic"))
        print(" ", summarise(labelled, "manual"))
        for domain, part in labelled.groupby("domain", sort=False):
            print(f"    {domain}: {summarise(part, 'manual')}")
        return

    df = sample_reports(args.domains, per_domain=args.per_domain)
    for i, r in df.iterrows():
        text = " ".join(r["fact"].split())  # collapse newlines for display
        print(f"[{i:2d}] {r['domain']:10s} {r['start_date'].date()}→{r['end_date'].date()}  "
              f"heuristic={r['heuristic']}\n     {text}\n")
    print(summarise(df, "heuristic"))
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved to {out} — fill the 'manual' column (up/down/mixed/none), then re-run.")


if __name__ == "__main__":
    main()
