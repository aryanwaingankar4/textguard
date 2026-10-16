"""R4 — semantic-mismatch robustness flags for the FNSPID flips (text only, no model output).

The FNSPID engine (``heldout_rules``) reverses the *move* a headline states but never its *cause*:
"Netflix Shares Trading Lower After Phase 3 Study …" becomes "… Trading Higher After …". Where the
cause only fits the original direction ("trading higher after the Phase 3 study did not meet its
primary endpoint"), the fake is lexically consistent but semantically odd — a reader (or an LLM)
may discount it. The scan (``consistency_r4``) cannot see this: it checks words and numbers, not
whether a cause fits a move (PROJECT_STATUS §8 "R4 held-out attacks", limitation).

We cannot judge "fits" automatically without a reader, so we flag the *possibility* with a fixed,
broad, text-only rule (Aryan, 2026-10-16, written before any model output on FNSPID): an
**edited** headline (a line of the fake that differs from the same line of the real bundle)
containing a causal connector — ``after``, ``on``, ``following``, ``amid``, ``as``, ``due to``
(whole words, any case). It over-flags on purpose ("on CNBC", "as CEO" are not causes): the
robustness table drops every possibly-mismatched fake, so the "without" result is conservative.

Pre-stated use (§10g): every FNSPID ``flipped`` and ``a1_selective`` result is reported **with and
without** the flagged bundles. A1 ⊆ flip with identical texts, so its flags are the flip's.

Output: ``results/r4_connector_flags.csv`` (one row per clean attacked bundle) and
``results/summary_r4_connector_flags.csv`` (per stock × attack). Run after
``python -m src.attacks.heldout``: ``python -m src.attacks.connectors_r4``.
"""

from __future__ import annotations

import re

import pandas as pd

from src.attacks import common
from src.attacks.heldout import EXCLUSIONS, OUT_DIR

CONNECTORS = ["after", "on", "following", "amid", "as", "due to"]
CONNECTOR_RE = {c: re.compile(r"\b" + c.replace(" ", r"\s+") + r"\b", re.I) for c in CONNECTORS}
ATTACKS = ["flipped", "a1_selective"]
FLAGS = common.ROOT / "results" / "r4_connector_flags.csv"
SUMMARY = common.ROOT / "results" / "summary_r4_connector_flags.csv"


def edited_lines(real: str, fake: str) -> list[str]:
    """Lines of the fake that differ from the real bundle's line at the same position.

    The FNSPID flip edits headlines in place and never adds or drops a line, so a positional
    comparison is exact; a length mismatch is a pipeline bug, not a case to handle.
    """
    r, f = real.split("\n"), fake.split("\n")
    if len(r) != len(f):
        raise ValueError("flip changed the number of headlines")
    return [b for a, b in zip(r, f) if a != b]


def connectors_in(line: str) -> list[str]:
    return [c for c in CONNECTORS if CONNECTOR_RE[c].search(line)]


def flags() -> pd.DataFrame:
    real = pd.read_csv(OUT_DIR / "real.csv", keep_default_na=False)
    real = real[real.domain == "FNSPID"].set_index(["series", "sample_id"]).text
    excl = pd.read_csv(EXCLUSIONS, keep_default_na=False)
    rows = []
    for attack in ATTACKS:
        df = pd.read_csv(OUT_DIR / f"{attack}.csv", keep_default_na=False)
        df = df[(df.domain == "FNSPID") & (df.attacked.astype(str) == "True")]
        bad = set(zip(*[excl.loc[excl.attack == attack, c] for c in ("series", "sample_id")]))
        for s, sid, text in zip(df.series, df.sample_id, df.text):
            if (s, sid) in bad:
                continue                                  # scan-flagged → not attacked (§10f A1)
            lines = edited_lines(real[(s, sid)], text)
            hits = [connectors_in(x) for x in lines]
            found = sorted({c for h in hits for c in h}, key=CONNECTORS.index)
            rows.append({"attack": attack, "series": s, "sample_id": sid,
                         "edited_lines": len(lines), "lines_with_connector": sum(bool(h) for h in hits),
                         "connectors": "|".join(found), "flagged": bool(found)})
    return pd.DataFrame(rows)


def summary(f: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (attack, s), g in f.groupby(["attack", "series"], sort=False):
        row = {"attack": attack, "series": s, "clean_attacked": len(g), "flagged": int(g.flagged.sum()),
               "kept_without_flagged": int((~g.flagged).sum())}
        for c in CONNECTORS:
            row[f"has_{c.replace(' ', '_')}"] = int(g.connectors.str.split("|").map(lambda x: c in x).sum())
        out.append(row)
    s = pd.DataFrame(out)
    tot = s.groupby("attack", sort=False).sum(numeric_only=True).reset_index()
    tot.insert(1, "series", "ALL")
    return pd.concat([s, tot], ignore_index=True)


def main() -> None:
    f = flags()
    f.to_csv(FLAGS, index=False)
    s = summary(f)
    s.to_csv(SUMMARY, index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
