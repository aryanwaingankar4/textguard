"""Check every number in paper/TextGuard_paper.docx against its source.

Run with the paper venv after build_paper.py:
    .venv-paper\\Scripts\\python paper\\check_numbers.py

Three checks, strongest first:
1. **Headline claims** (KEY below): the sentence that states a key result must contain the
   number, and the number must equal the named CSV cell at the printed precision.
2. **Table cells**: every cell build_paper.py logged in paper/build/provenance.json is re-read from
   its CSV / JSON here, independently of the build, and must reproduce the printed text, which
   must appear in the document.
3. **Every other number** in the text (the reference list, verbatim prompt blocks, dates, citation and
   equation labels excluded) must be found in a source: a value in results/summary_*.csv or
   *.json at the printed precision, a number written in PROJECT_STATUS.md §9 (Results), or a
   number in the design / pre-registration sections (§1a, §6, §7, §8, §10), which are reported
   as a separate category. Anything else is UNMATCHED and fails the check.

Why this split: the sweep (3) proves that a number exists in a source at that precision, not that
it is the right one for the sentence. Tables are pinned cell by cell (2), and the claims a reader
will quote are pinned sentence by sentence (1).
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
DOCX = HERE / "TextGuard_paper.docx"
OUT = HERE / "build" / "number_check.csv"

NUM = re.compile(r"(?<![\w.\-+/^])([+\-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[+\-]?\d+(?:\.\d+)?)"
                 r"(?![\w/]|\.\d)")
SCI = re.compile(r"(?<![\w.])(?:(\d+(?:\.\d+)?)×)?10([\-+]\d{1,2})(?![\d.])")
STRIP = [re.compile(p) for p in (
    r"\b\d{4}-\d{2}-\d{2}\b", r"\b\d{4}-\d{2}\b", r"https?://\S+", r"arXiv:\d{4}\.\d{4,5}",
    r"\[\d+\](?:, \[\d+\])*", r"Fig\. \d+", r"\t\(\d+\)$", r"^\d+\)\t", r"S&P 500",
    r"results/summary_\*\.csv")]
YEAR = re.compile(r"^(19|20)\d\d$")


# ---- the document ----------------------------------------------------------------------------
def blocks(doc):
    """Yield (kind, text, is_code) in reading order, skipping the reference list (from the
    References heading to the next top-level heading, i.e. the appendices)."""
    body = doc.element.body
    in_refs = False
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, doc)
            if p.style.name == "Heading 1":
                in_refs = p.text.strip().lower() == "references"
            if in_refs:
                continue
            code = any(r.font.name == "Courier New" for r in p.runs)
            yield "p", p.text, code
        elif tag == "tbl":
            t = Table(child, doc)
            for row in t.rows:
                seen = set()
                for cell in row.cells:
                    if id(cell._tc) in seen:
                        continue
                    seen.add(id(cell._tc))
                    yield "cell", cell.text, False


def norm(s: str) -> str:
    return (s.replace("−", "-").replace("–", "–").replace(" ", " ")
            .replace(" ", " "))


def tokens(text: str):
    """(token, value, decimals, context) for every number in ``text``."""
    t = norm(text)
    for rx in STRIP:
        t = rx.sub(" ", t)
    out = []
    for m in SCI.finditer(t):
        mant = float(m.group(1)) if m.group(1) else 1.0
        out.append((m.group(0), mant * 10 ** int(m.group(2)), None, t[max(0, m.start() - 50):m.end() + 30]))
    t2 = SCI.sub(" ", t)
    for m in NUM.finditer(t2):
        raw = m.group(1)
        clean = raw.replace(",", "")
        dec = len(clean.split(".")[1]) if "." in clean else 0
        out.append((raw, float(clean), dec, t2[max(0, m.start() - 50):m.end() + 30]))
    return out


# ---- the sources ---------------------------------------------------------------------------
def _walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield path, float(obj)


def result_pool():
    """printed-form -> first source, for every numeric value in the summaries."""
    pool = {}

    def add(v, src):
        for x, tag in ((v, ""), (100 * v, " (×100)")):
            for d in range(0, 5):
                key = f"{abs(round(x, d)):.{d}f}"
                pool.setdefault(key, src + tag)
    for f in sorted(RES.glob("summary_*.csv")):
        with open(f, newline="", encoding="utf-8") as fh:
            for i, row in enumerate(csv.DictReader(fh), 2):
                for col, val in row.items():
                    try:
                        v = float(val)
                    except (TypeError, ValueError):
                        continue
                    if v == v and abs(v) != float("inf"):
                        add(v, f"{f.name}:{i}:{col}")
    for f in sorted(RES.glob("summary_*.json")):
        for path, v in _walk(json.loads(f.read_text(encoding="utf-8"))):
            add(v, f"{f.name}{path}")
    return pool


def status_pools():
    """Numbers written in PROJECT_STATUS.md: §9 (results) and the design sections."""
    lines = (ROOT / "PROJECT_STATUS.md").read_text(encoding="utf-8").splitlines()
    sec, results, design = None, {}, {}
    for n, line in enumerate(lines, 1):
        m = re.match(r"^## (\d+[a-z]?)\.", line)
        if m:
            sec = m.group(1)
        if line.startswith("## 1a."):
            sec = "1a"
        target = results if sec == "9" else design if sec in {"1a", "6", "6a", "7", "7a", "7b", "7c", "7d", "7e", "7f", "8", "10"} else None
        if target is None:
            continue
        for tok, v, dec, _ in tokens(line):
            for key in _keys(tok, v, dec):
                target.setdefault(key, f"PROJECT_STATUS.md:{n} (§{sec})")
    return results, design


def _keys(tok, v, dec):
    if dec is None:
        return [f"sci:{v:.3g}"]
    return [f"{abs(v):.{dec}f}"]


# ---- headline claims pinned to a cell --------------------------------------------------------
KEY = [  # (regex that must match the document text, file, filter, column, scale, decimals)
    (r"moves \+0\.457σ toward the lie", "summary_p5_attack.csv",
     dict(short="M5", condition="flipped", domain="pooled"), "dss", 1, 3),
    (r"23\.6% of attacked windows", "summary_p5_attack.csv",
     dict(short="M5", condition="flipped", domain="pooled"), "flip_rate", 100, 1),
    (r"\+0\.451σ to −0\.042σ", "summary_p6_operating.csv",
     dict(short="M5", attack="flipped", defence="always_trust", domain="pooled"), "dss", 1, 3),
    (r"\+0\.451σ to −0\.042σ", "summary_p6_operating.csv",
     dict(short="M5", attack="flipped", defence="TextGuard τ=0", domain="pooled"), "dss", 1, 3),
    (r"blocks 14\.4% of genuine reports", "summary_p6_operating.csv",
     dict(short="M5", attack="flipped", defence="TextGuard τ=0", domain="pooled"), "fbr", 100, 1),
    (r"\+0\.180σ to \+0\.021σ", "summary_r4_operating.csv",
     dict(target="M5", attack="a2_forward", defence="none", cell="pooled"), "dss", 1, 3),
    (r"\+0\.180σ to \+0\.021σ", "summary_r4_operating.csv",
     dict(target="M5", attack="a2_forward", defence="TG2", cell="pooled"), "dss", 1, 3),
    (r"blocks 18\.4% of genuine reports", "summary_r4_fbr.csv",
     dict(target="M5", defence="TG2", cell="pooled"), "fbr", 100, 1),
    (r"v1’s 7\.0%", "summary_r4_fbr.csv", dict(target="M5", defence="TG1", cell="pooled"),
     "fbr", 100, 1),
    (r"FNSPID 22\.4%", "summary_r4_fbr.csv", dict(target="M5", defence="TG2", cell="FNSPID"),
     "fbr", 100, 1),
    (r"below v1 \(\+0\.096σ\)", "summary_r4_operating.csv",
     dict(target="M5", attack="a2_forward", defence="TG1", cell="pooled"), "dss", 1, 3),
    (r"\+0\.206σ toward the lie", "summary_r2_attack.csv",
     dict(short="Q7", condition="flipped", domain="pooled"), "dss", 1, 3),
    (r"fooled \(\+0\.362σ\)", "summary_r2_attack.csv",
     dict(short="M5-sk", condition="flipped", domain="pooled"), "dss", 1, 3),
    (r"moves Model 5 by \+0\.456σ", "summary_r3_operating.csv",
     dict(target="M5", attack="a2_forward", defence="none", cell="pooled"), "dss", 1, 3),
    (r"Qwen2\.5-7B by \+0\.403σ", "summary_r3_operating.csv",
     dict(target="Q7", attack="a2_forward", defence="none", cell="pooled"), "dss", 1, 3),
    (r"TextGuard leaves \+0\.400σ", "summary_r3_operating.csv",
     dict(target="M5", attack="a2_forward", defence="TG", cell="pooled"), "dss", 1, 3),
    (r"M3-F \+0\.049σ", "summary_r1_attack.csv",
     dict(short="M3-F", condition="flipped", domain="pooled"), "dss", 1, 3),
    (r"FBR of 14\.4%", "summary_p6_operating.csv",
     dict(short="M5", attack="flipped", defence="TextGuard τ=0", domain="pooled"), "fbr", 100, 1),
    (r"agreement 76%, κ = 0\.60", "summary_r4_judge_labels.csv", None, None, None, None),
]


def _cell(file, flt, col):
    with open(RES / file, newline="", encoding="utf-8") as fh:
        hits = [r for r in csv.DictReader(fh) if all(r.get(k) == v for k, v in flt.items())]
    if len(hits) != 1:
        raise KeyError(f"{file} {flt}: {len(hits)} rows")
    return float(hits[0][col])


def check_keys(fulltext):
    fails, n = [], 0
    for rx, file, flt, col, scale, dec in KEY:
        n += 1
        m = re.search(rx, fulltext)
        if not m:
            fails.append(f"claim not found: /{rx}/")
            continue
        if flt is None:                      # sentence-only check (source checked by the sweep)
            continue
        want = f"{abs(_cell(file, flt, col) * scale):.{dec}f}"
        nums = [x.replace("−", "") for x in re.findall(r"−?\d+\.\d+", m.group(0))]
        if want not in nums:
            fails.append(f"/{rx}/: source {file} {flt} {col} = {want}, text has {nums}")
    return n, fails


def check_provenance(doctext):
    prov = json.loads((HERE / "build" / "provenance.json").read_text(encoding="utf-8"))
    fails = []
    for e in prov:
        if "json_path" in e:
            v = 1.0
            obj = json.loads((RES / e["file"]).read_text(encoding="utf-8"))
            for k in e["json_path"].split("/"):
                obj = obj[k]
            v = float(obj)
        else:
            v = _cell(e["file"], e["filter"], e["column"])
        if v != e["value"] and not (v != v and e["value"] != e["value"]):
            fails.append(f"value changed: {e}")
        printed = re.sub(r"\^\{(.+?)\}|_\{(.+?)\}", lambda m: m.group(1) or m.group(2), e["text"])
        if printed not in doctext:
            fails.append(f"printed text not in document: {e['text']} ({e['file']} {e.get('column')})")
    return len(prov), fails


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    doc = Document(str(DOCX))
    items = list(blocks(doc))
    fulltext = "\n".join(t for _, t, _ in items)
    pool = result_pool()
    s9, design = status_pools()
    prov = json.loads((HERE / "build" / "provenance.json").read_text(encoding="utf-8"))
    pinned = {norm(e["text"]).lstrip("+-") for e in prov}

    rows, cats = [], Counter()
    for kind, text, code in items:
        if code:
            continue
        for tok, v, dec, ctx in tokens(text):
            keys = _keys(tok, v, dec)
            key = keys[0]
            plainkey = norm(tok).replace(",", "").lstrip("+-")
            if kind == "cell" and plainkey in {p.replace(",", "") for p in pinned}:
                cat, src = "table cell (pinned, re-verified)", "provenance.json"
            elif dec is not None and key in pool:
                cat, src = "summary table", pool[key]
            elif key in s9:
                cat, src = "PROJECT_STATUS §9 (results)", s9[key]
            elif dec is None:
                cat, src = _sci_lookup(v)
            elif key in design:
                cat, src = "PROJECT_STATUS design / pre-registration", design[key]
            elif YEAR.match(plainkey):
                cat, src = "year", ""
            else:
                cat, src = "UNMATCHED", ""
            cats[cat] += 1
            rows.append({"kind": kind, "token": tok, "category": cat, "source": src,
                         "context": ctx.replace("\n", " ")})

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    nk, kfail = check_keys(fulltext)
    np_, pfail = check_provenance(fulltext)
    unmatched = [r for r in rows if r["category"] == "UNMATCHED"]
    print(f"numbers found in the text: {len(rows)}")
    for c, n in cats.most_common():
        print(f"  {c:45s} {n}")
    print(f"headline claims pinned to a cell: {nk - len(kfail)} / {nk} pass")
    print(f"table cells re-verified from source: {np_ - len(pfail)} / {np_} pass")
    for f in kfail + pfail:
        print("  FAIL", f)
    for r in unmatched:
        print(f"  UNMATCHED {r['token']!r}: …{r['context']}…")
    print(f"details: {OUT.relative_to(ROOT)}")
    ok = not (kfail or pfail or unmatched)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


_SCI_CACHE = None


def _sci_lookup(v):
    """A number printed as m×10^e (p-values): find a summary value with the same 2 digits."""
    global _SCI_CACHE
    if _SCI_CACHE is None:
        _SCI_CACHE = {}
        for f in sorted(RES.glob("summary_*.csv")):
            with open(f, newline="", encoding="utf-8") as fh:
                for i, row in enumerate(csv.DictReader(fh), 2):
                    for col, val in row.items():
                        try:
                            x = float(val)
                        except (TypeError, ValueError):
                            continue
                        if 0 < x < 1e-2:
                            _SCI_CACHE.setdefault(f"{x:.1e}", f"{f.name}:{i}:{col}")
    key = f"{v:.1e}"
    if key in _SCI_CACHE:
        return "summary table", _SCI_CACHE[key]
    return "UNMATCHED", ""


if __name__ == "__main__":
    sys.exit(main())
