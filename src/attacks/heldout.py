"""R4 — attack texts for the held-out domains (Energy, FNSPID), scanned, with coverage and a
review file. No model or judge is run here.

Conditions (PROJECT_STATUS §10g "Held-out freeze"; engines in ``heldout_rules``):

* ``real`` — the window's genuine text (Energy report at lag 2 / FNSPID weekly headline bundle).
* ``flipped`` — the P3-style minimal flip (Energy: verb + week-over-week comparator; FNSPID: every
  headline claim about the stock, with coupled numbers and ratings).
* ``a1_selective`` — R3's A1: the flip, **only** where TextGuard v1's change evidence agrees with
  the claimed direction (or is 0) at the frozen weekly window w = 6 (§10g) — else the real text.
  Neither held-out domain states a baseline, so v1's level branch never fires.
* ``a2_forward`` — R3's A2: the real text kept as an exact prefix + one sentence claiming the
  reverse direction for next week (Energy: appended as R3 does; FNSPID: as a new headline line).
* ``a3_magnitude`` — R3's A3 at ×10, **Energy only** (dropped on FNSPID, Aryan 2026-10-12), with
  the Energy plausibility bound (the largest pre-text weekly move).

Then the R4 consistency scan (``consistency_r4``; R3 Amendment 1's rule) — a flagged text is
treated as **not attacked** in every R4 metric.

Outputs: ``texts/r4/<condition>.csv`` (gitignored; P3 columns + ``series``; fake rows
``synthetic=True``), ``results/r4_text_exclusions.csv``, ``results/summary_r4_attack_coverage.csv``,
``texts/r4_review_pairs.md`` (15 before/after pairs for Aryan, seed 0).

Run: ``python -m src.attacks.heldout``.
"""

from __future__ import annotations

import re
import sys

import pandas as pd

from src.attacks import common
from src.attacks import consistency_r4 as c4
from src.attacks import heldout_rules as hr
from src.attacks.adaptive import append_claim
from src.attacks.rules import EditResult
from src.attacks.validate_attacks import clean, diff
from src.data import heldout
from src.textguard.trend_detector import change_evidence

OUT_DIR = common.TEXTS_DIR / "r4"
EXCLUSIONS = common.ROOT / "results" / "r4_text_exclusions.csv"
COVERAGE = common.ROOT / "results" / "summary_r4_attack_coverage.csv"
REVIEW = common.TEXTS_DIR / "r4_review_pairs.md"
REVIEW_2 = common.TEXTS_DIR / "r4_review_pairs_2.md"
# (series, sample_id) of the FNSPID windows Aryan reviewed in round 1 (2026-10-16), pairs 9-15
SHOWN_1 = frozenset({("MU", "2019-02-15"), ("PFE", "2018-12-28"), ("PCG", "2020-04-24"), ("FDX", "2019-05-24"),
                     ("PFE", "2019-10-25"), ("NVDA", "2019-02-01"), ("PCG", "2019-10-18")})
PAIRS_2 = [("FNSPID", "flipped", 3), ("FNSPID", "a1_selective", 1), ("FNSPID", "a2_forward", 1)]
V1_WEEKLY_WINDOW = 6                  # TextGuard v1's frozen weekly window (Health_US), §10g
A3_FACTOR = 10.0
ATTACKS = ["flipped", "a1_selective", "a2_forward", "a3_magnitude"]
COLUMNS = ["domain", "series"] + common.COLUMNS[1:]


def _frame(w: pd.DataFrame, condition: str, res: list[EditResult]) -> pd.DataFrame:
    out = pd.DataFrame({
        "domain": w["domain"].values, "series": w["series"].values, "split": "test",
        "sample_id": w["sample_id"].values, "condition": condition,
        "text": [r.text if isinstance(r.text, str) else "" for r in res],
        "attacked": [r.attacked for r in res], "reason": [r.reason for r in res],
        "real_direction": [r.real_direction for r in res], "claimed_direction": [r.claimed_direction for r in res],
        "variant": [r.variant for r in res], "source_domain": w["domain"].values,
        "source_sample_id": w["sample_id"].values, "synthetic": [r.attacked for r in res]})
    out["has_text"] = out["text"].str.len() > 0
    return out[COLUMNS]


def _flip(text, domain, series) -> EditResult:
    return hr.energy_edit(text, "flip") if domain == "Energy" else hr.fnspid_edit(text, series)


def build_real(w):
    return _frame(w, "real", [EditResult(t, False, "", 0, 0, "") for t in w["text"]])


def build_flipped(w):
    return _frame(w, "flipped", [_flip(t, d, s) for t, d, s in zip(w["text"], w["domain"], w["series"])])


def build_a1(w):
    res = []
    for r in w.itertuples():
        f = _flip(r.text, r.domain, r.series)
        if f.attacked:
            ev = change_evidence(r.lookback, r.sigma, V1_WEEKLY_WINDOW).direction
            if ev not in (0, f.claimed_direction):
                f = EditResult(r.text, False, "flip would disagree with TextGuard's trend evidence",
                               f.real_direction, 0, f.variant)
        res.append(f)
    return _frame(w, "a1_selective", res)


def build_a2(w):
    res = []
    for r in w.itertuples():
        if r.domain == "Energy":
            d, why = hr.energy_direction(r.text)
        else:
            d = hr.bundle_direction(r.text, r.series)
            why = "no text" if not isinstance(r.text, str) else "no net direction about the stock"
        if d == 0:
            res.append(EditResult(r.text, False, why, 0, 0, ""))
            continue
        fake = (append_claim(r.text, hr.energy_forward(-d)) if r.domain == "Energy"
                else r.text.rstrip() + "\n" + hr.fnspid_forward(r.series, -d))
        res.append(EditResult(fake, True, "", d, -d, "forward"))
    return _frame(w, "a2_forward", res)


def build_a3(w):
    max_step = heldout.energy_max_step()
    res = [hr.energy_edit(t, "exaggerate", A3_FACTOR, max_step) if d == "Energy"
           else EditResult(t, False, "A3 not run on FNSPID (§10g)", 0, 0, "")
           for t, d in zip(w["text"], w["domain"])]
    return _frame(w, "a3_magnitude", res)


def coverage(tables: dict[str, pd.DataFrame], excl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for attack, df in tables.items():
        for (dom, ser), g in df.groupby(["domain", "series"], sort=False):
            if attack == "a3_magnitude" and dom == "FNSPID":
                continue
            flagged = int(((excl.attack == attack) & (excl.domain == dom) & (excl.series == ser)).sum())
            rows.append(dict(attack=attack, domain=dom, series=ser, windows=len(g), with_text=int(g.has_text.sum()),
                             attacked=int(g.attacked.sum()), flagged=flagged, clean=int(g.attacked.sum()) - flagged))
    c = pd.DataFrame(rows)
    pooled = (c[c.domain == "FNSPID"].groupby("attack", sort=False)[["windows", "with_text", "attacked", "flagged", "clean"]]
              .sum().reset_index().assign(domain="FNSPID", series="(10 stocks)"))
    return pd.concat([c, pooled], ignore_index=True)[c.columns]


PAIRS = [("Energy", "flipped", 2), ("Energy", "a1_selective", 2), ("Energy", "a2_forward", 2),
         ("Energy", "a3_magnitude", 2), ("FNSPID", "flipped", 3), ("FNSPID", "a1_selective", 2),
         ("FNSPID", "a2_forward", 2)]


def review_pairs(tables, real, excl, w, pairs=None, seed: int = 0, skip=frozenset(),
                 title: str = "15 before/after pairs for review by eye (seed 0)") -> str:
    """Attacked, scan-clean pairs. Round 1 (seed 0, ``PAIRS``): Energy 2 per attack, FNSPID flip 3,
    A1 2, A2 2. Round 2 (``PAIRS_2``, seed 1): 5 FNSPID pairs not shown in round 1 (``SHOWN_1``)."""
    pairs = PAIRS if pairs is None else pairs
    r = real.set_index(c4.KEYS)["text"]
    lb = w.set_index(c4.KEYS)["lookback"]
    bad = set(map(tuple, excl[["attack"] + c4.KEYS].values))
    out = [f"# R4 — {title}", "",
           f"> Fake texts are {common.SYNTHETIC_NOTE}", "",
           "Edits are shown inline as `[old -> new]`; headlines of a bundle are separated by ` / `.",
           "Check: is the lie fluent and coherent, is the claimed direction right, are the numbers",
           "plausible, does any headline left unedited contradict an edited one? Problems found here",
           "become a dated amendment (R3 rule).", ""]
    i = 0
    for dom, attack, n in pairs:
        df = tables[attack]
        a = df[df.attacked & (df.domain == dom)]
        a = a[[(attack, *k) not in bad and k[1:] not in skip
               for k in a[c4.KEYS].itertuples(index=False, name=None)]]
        pick = a.sample(min(n, len(a)), random_state=seed)
        out += [f"## {dom} — {attack}", ""]
        for x in pick.itertuples():
            i += 1
            k = (x.domain, x.series, x.sample_id)
            out.append(f"### {i}. {attack} — {x.series} {x.sample_id} | real direction {x.real_direction:+d} "
                       f"→ claimed {x.claimed_direction:+d}")
            if attack == "a1_selective":
                vals = ", ".join(f"{v:,.4g}" for v in lb.loc[k][-V1_WEEKLY_WINDOW:])
                out.append(f"*TextGuard v1's evidence (last {V1_WEEKLY_WINDOW} values, oldest first): {vals}*")
            out += ["", _shown(r.loc[k], x.text, attack, dom), ""]
    return "\n".join(out) + "\n"


def _shown(real: str, fake: str, attack: str, domain: str) -> str:
    """The review rendering. A2 is real + one appended claim, shown as an addition: on FNSPID as its
    own headline (`` / [+ …]``) — a word diff would drop the line break and glue the new headline to
    the last real one (the text itself has it on its own line; review pair 15, 2026-10-16)."""
    if attack == "a2_forward" and fake.startswith(real.rstrip()):
        added = fake[len(real.rstrip()):].strip()
        return clean(real) + (" / " if domain == "FNSPID" else " ") + f"[+ {added}]"
    return clean(diff(real, fake))


FORWARD = re.compile(r"(?P<subject>.+?) (?:is|are) expected to (?P<verb>rise|fall) next week\.$")


def validate(tables: dict[str, pd.DataFrame], real: pd.DataFrame) -> None:
    """R3-style bookkeeping, all automatic (raises on any failure)."""
    for name, df in tables.items():
        assert (df[c4.KEYS].values == real[c4.KEYS].values).all(), f"{name}: windows differ from real"
        assert (df.synthetic == df.attacked).all() and ((df.claimed_direction != 0) == df.attacked).all(), name
        assert (df.loc[~df.attacked, "text"] == real.loc[~df.attacked, "text"]).all(), f"{name}: unattacked != real"
    fl, a1, a2 = tables["flipped"], tables["a1_selective"], tables["a2_forward"]
    assert (fl.attacked | ~a1.attacked).all() and (a1.loc[a1.attacked, "text"] == fl.loc[a1.attacked, "text"]).all()
    for i in a2.index[a2.attacked]:
        t, r = a2.at[i, "text"], real.at[i, "text"]
        assert t.startswith(r.rstrip()), "A2: real text is not an exact prefix"
        last = re.split(r"\n|(?<=[.!?]) ", t[len(r.rstrip()):].strip())[-1]
        m = FORWARD.match(last)
        assert m and (1 if m.group("verb") == "rise" else -1) == a2.at[i, "claimed_direction"] == -a2.at[i, "real_direction"]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    w = heldout.windows()
    real = build_real(w)
    tables = {"flipped": build_flipped(w), "a1_selective": build_a1(w), "a2_forward": build_a2(w),
              "a3_magnitude": build_a3(w)}
    validate(tables, real)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in {"real": real, **tables}.items():
        # CRLF on every OS: export_r4_inputs checks these bytes against pre-registered sha256s
        # computed on Windows, where pandas writes CRLF (R7; Linux / macOS would write LF).
        df.to_csv(OUT_DIR / f"{name}.csv", index=False, lineterminator="\r\n")
    (OUT_DIR / "README.txt").write_text(f"R4 held-out text conditions.\n\nRows with synthetic=True are "
                                        f"{common.SYNTHETIC_NOTE}\nGenerated by: python -m src.attacks.heldout\n",
                                        encoding="utf-8")
    excl = c4.exclusions(tables, real)
    excl.to_csv(EXCLUSIONS, index=False)
    cov = coverage(tables, excl)
    cov.to_csv(COVERAGE, index=False)
    REVIEW.write_text(review_pairs(tables, real, excl, w), encoding="utf-8")
    REVIEW_2.write_text(review_pairs(tables, real, excl, w, PAIRS_2, seed=1, skip=SHOWN_1,
                                     title="round 2: 5 new FNSPID pairs after the 2026-10-16 engine fixes (seed 1)"),
                        encoding="utf-8")
    print(cov.to_string(index=False))
    print("\nflags by check:\n" + (excl.assign(c=excl.checks.str.split(",")).explode("c").groupby(["attack", "c"]).size()
                                   .to_string() if len(excl) else "  none"))
    print("\nnot attacked — reasons (texts with text only):")
    for name, df in tables.items():
        t = df[df.has_text & ~df.attacked]
        print(f"  {name}: " + str(t.groupby("domain").reason.apply(lambda s: s.str[:48].value_counts().head(5).to_dict()).to_dict()))
    print(f"\n-> {OUT_DIR}, {EXCLUSIONS.name}, {COVERAGE.name}, {REVIEW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
