"""Measure 8-word-sequence overlap between the paper and every text file in the repository.

Run with the paper venv after build_paper.py:
    .venv-paper\\Scripts\\python paper\\check_overlap.py

Why: the paper goes through Turnitin, and the repository (README, cards, docs, the results log,
code comments, the section drafts) will be public later. Text shared with those files would show
up as matched text. We count 8-word shingles (lower-cased words, punctuation removed) of the
paper that also occur in a repository file.

Two numbers are reported for every file:
* **all text** - every paragraph, table cell and caption except the reference list;
* **body** - the same without the verbatim prompt blocks of Appendix A (Courier) and without text
  inside quotation marks (quoted prompts, report snippets, pre-registered rule wording), which is
  meant to match its source.
The References are excluded from both. The paper's own build sources (paper/build_paper.py,
content_*.py, this script) hold the paper text by construction and are skipped; everything else
tracked by git (and the untracked README / docs, if any) is scanned.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DOCX = HERE / "TextGuard_paper.docx"
OUT = HERE / "build" / "overlap_report.txt"
N = 8
SELF = {"paper/build_paper.py", "paper/content_front.py", "paper/content_results.py",
        "paper/content_back.py", "paper/check_overlap.py", "paper/check_numbers.py"}
TEXT_EXT = {".md", ".py", ".tex", ".txt", ".bib", ".json", ".csv", ".ipynb", ".sh", ".ps1",
            ".toml", ".cfg", ".yml", ".yaml", ".ini", ""}
MAX_BYTES = 30_000_000
QUOTED = re.compile(r"“[^”]*”|\"[^\"]*\"")
WORD = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")


def words(text: str) -> list[str]:
    return WORD.findall(text.lower().replace("’", "'"))


def shingles(ws: list[str]) -> set[tuple]:
    return {tuple(ws[i:i + N]) for i in range(len(ws) - N + 1)}


def paper_text():
    doc = Document(str(DOCX))
    allp, body = [], []
    in_refs = False
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, doc)
            if p.style.name == "Heading 1":       # the reference list runs to the next heading
                in_refs = p.text.strip().lower() == "references"
            if in_refs:
                continue
            allp.append(p.text)
            if not any(r.font.name == "Courier New" for r in p.runs):
                body.append(QUOTED.sub(" | ", p.text))
        elif tag == "tbl":
            for row in Table(child, doc).rows:
                for cell in row.cells:
                    allp.append(cell.text)
                    body.append(QUOTED.sub(" | ", cell.text))
    return allp, body


def para_shingles(paras):
    out = set()
    for t in paras:
        for chunk in t.split(" | "):          # a removed quotation breaks the sequence
            out |= shingles(words(chunk))
    return out


def repo_files():
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.splitlines()
    for rel in tracked:
        p = ROOT / rel
        if rel in SELF or p.suffix.lower() not in TEXT_EXT or not p.is_file():
            continue
        if p.stat().st_size > MAX_BYTES:
            continue
        yield rel, p


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    allp, body = paper_text()
    s_all, s_body = para_shingles(allp), para_shingles(body)
    results, examples = [], {}
    for rel, p in repo_files():
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        fs = shingles(words(text))
        a, b = s_all & fs, s_body & fs
        if a:
            results.append((rel, len(a), len(b)))
            examples[rel] = sorted(b or a)[:3]
    results.sort(key=lambda r: (-r[2], -r[1]))
    lines = [f"Paper: {len(s_all)} distinct {N}-word sequences (all text), {len(s_body)} (body, "
             f"without prompt blocks and quotations); References excluded.",
             f"Files scanned: {sum(1 for _ in repo_files())} (git-tracked text files; paper build "
             f"sources skipped).", ""]
    union_all = set().union(*(s_all & shingles(words((ROOT / r).read_text(encoding='utf-8', errors='ignore'))) for r, _, _ in results)) if results else set()
    union_body = set().union(*(s_body & shingles(words((ROOT / r).read_text(encoding='utf-8', errors='ignore'))) for r, _, _ in results)) if results else set()
    lines.append(f"Overlap with the whole repository: all text {len(union_all)} / {len(s_all)} = "
                 f"{100 * len(union_all) / max(1, len(s_all)):.2f}%; body {len(union_body)} / "
                 f"{len(s_body)} = {100 * len(union_body) / max(1, len(s_body)):.2f}%")
    lines.append("")
    lines.append(f"{'file':60s} {'all':>6s} {'body':>6s}  body %")
    for rel, a, b in results:
        lines.append(f"{rel:60s} {a:6d} {b:6d}  {100 * b / max(1, len(s_body)):.2f}%")
    lines.append("")
    lines.append("Examples of shared body sequences (up to 3 per file):")
    for rel, _, b in results:
        if b:
            for ex in examples[rel]:
                lines.append(f"  {rel}: {' '.join(ex)}")
    report = "\n".join(lines)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
