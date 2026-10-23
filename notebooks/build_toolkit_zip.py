"""Build ``dist/textguard_toolkit.zip`` — the toolkit code for a Kaggle / Colab run without GitHub (R7).

Why a zip: the repository is private until P8, so ``textguard_your_model.ipynb`` cannot
``git clone`` it. The zip is uploaded once as a private Kaggle dataset; cell 1 of the notebook
finds it under ``/kaggle/input`` and copies it to the working directory.

What goes in — exactly what the toolkit needs, nothing else:

* every repository module that ``import textguard`` loads (found by importing it in a fresh
  interpreter, not by a hand-kept list, so a new import cannot be forgotten);
* the worked example (``examples/sample_series.csv``, a synthetic series; ``audit_dummy.py``);
* the notebook itself, ``LICENSE`` and ``README.md``;
* ``TOOLKIT_MANIFEST.json``: the commit it was built from (``-dirty`` if the working tree had
  changes) and the sha256 of every file. Cell 1 checks the hashes, so a stale or edited upload is
  refused rather than silently audited.

No data, results, texts or model weights are included (no FNSPID or Time-MMD content). The build
then unpacks the zip into a temporary folder and runs the worked example there with an isolated
interpreter (``-I``), which proves that the file list is complete.

Run: ``python notebooks/build_toolkit_zip.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "textguard_toolkit.zip"
EXTRA = ["examples/sample_series.csv", "examples/audit_dummy.py",
         "notebooks/textguard_your_model.ipynb", "LICENSE", "README.md"]

# Imports textguard in a clean interpreter and prints the repository files it loaded.
_LIST_MODULES = """
import sys
from pathlib import Path
root = Path.cwd().resolve()
sys.path.insert(0, str(root))
import textguard
for m in list(sys.modules.values()):
    f = getattr(m, "__file__", None)
    if f and root in Path(f).resolve().parents:
        rel = Path(f).resolve().relative_to(root)
        if rel.parts[0] in ("src", "textguard"):  # not a venv that happens to live in the repo
            print(rel.as_posix())
"""

# Runs the worked example (no judge) inside the unpacked zip.
_SELF_TEST = """
import sys
sys.path.insert(0, ".")
from textguard import DummyForecaster, audit, load_series
from src.textguard import live_judge
data = load_series("examples/sample_series.csv", lookback=12, freq="weekly")
rep = audit(DummyForecaster(sensitivity=1.0, step=data.sigma / 4, lookback=12), data,
            subject="the average widget price")
assert len(rep.summary) > 0
print("self-test ok:", len(data.windows), "windows")
"""


def modules() -> list[str]:
    out = subprocess.run([sys.executable, "-I", "-c", _LIST_MODULES], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout.split()
    files = set(out)
    for f in list(files):  # package __init__ files, so every import path resolves
        p = Path(f).parent
        while p != Path("."):
            if (ROOT / p / "__init__.py").exists():
                files.add((p / "__init__.py").as_posix())
            p = p.parent
    return sorted(files)


def commit() -> str:
    h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True,
                       text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True,
                           capture_output=True, text=True).stdout.strip()
    return h + ("-dirty" if dirty else "")


def build() -> Path:
    files = modules() + EXTRA
    manifest = {"commit": commit(),
                "files": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}}
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(ROOT / f, f)
        z.writestr("TOOLKIT_MANIFEST.json", json.dumps(manifest, indent=1) + "\n")
    with tempfile.TemporaryDirectory() as d:
        zipfile.ZipFile(OUT).extractall(d)
        r = subprocess.run([sys.executable, "-I", "-c", _SELF_TEST], cwd=d, capture_output=True,
                           text=True, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        if r.returncode != 0:
            raise SystemExit(f"self-test failed inside the unpacked zip:\n{r.stdout}\n{r.stderr}")
        print(r.stdout.strip())
    print(f"{len(files)} files + manifest, commit {manifest['commit']}")
    for f in files:
        print("  ", f)
    print(f"-> {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    return OUT


if __name__ == "__main__":
    build()
