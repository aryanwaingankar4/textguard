"""Download every raw dataset (and MM-TSFlib) at a pinned revision, then check it by sha256 (R7).

Why this exists: ``data/`` and ``external/`` are gitignored. The data are not ours to
redistribute (FNSPID is CC BY-NC 4.0; Time-MMD's repository states no licence), so a clone
of this repo must fetch them from their owners, and must get **exactly** the bytes our
results came from. Every source is pinned to a commit / revision, and every file a result
depends on is checked against ``src/data/manifest.json``. A changed upstream file fails
loudly instead of silently changing a table.

Sources (pinned):

* **Time-MMD** — ``github.com/AdityaLab/Time-MMD`` @ ``00281e2`` (2025-01-14, the commit
  cloned on 2026-09-28): ``numerical/`` and ``textual/`` (~35 MB), from GitHub's zip archive
  (no git needed).
* **FNSPID** — ``huggingface.co/datasets/Zihan1004/FNSPID`` @ ``bf9189c``: only the Benzinga
  headline block and 15 price files (``src/data/fnspid.py`` streams them, ~340 MB read,
  ~10 MB kept, and re-derives the frozen stock selection).
* **MM-TSFlib** (MIT) — ``github.com/AdityaLab/MM-TSFlib`` @ ``e789ce7``: the DLinear /
  PatchTST network code Models 1–4 use. Only needed to **retrain** Models 1–4 / the R1 heads;
  ``reproduce.py`` does not retrain, so it is optional (``--with-mmtsflib``).

Hashes: sha256 of each file's **raw bytes**. Line endings matter: Time-MMD is written in the
form our results were computed from (a Windows ``git clone``, CRLF inside the report texts too —
:func:`to_windows_checkout`); a plain LF copy fails the check, because it changes 166 prompts.
Only ``selection_ranking.csv`` (written by pandas, platform newline) is hashed with CRLF -> LF.
**Use this script, not ``git clone``, to get Time-MMD.** The FNSPID headlines are kept
as parquet, whose bytes depend on the pyarrow version, so their digest is taken over the
table's content (a canonical CSV dump), not the file.

Run: ``python -m src.data.download`` (``--skip-fnspid``, ``--with-mmtsflib``, ``--verify-only``).
Maintainer only: ``--write-manifest`` records the digests of the files currently on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MANIFEST = Path(__file__).with_name("manifest.json")

TIMEMMD_COMMIT = "00281e2d86058286d5548b15a7670e8eda57ef62"
MMTSFLIB_COMMIT = "e789ce78c9bafd8e3ba0d8850f9ad2becbe83548"
TIMEMMD_DIR = DATA / "Time-MMD"
FNSPID_DIR = DATA / "FNSPID"
MMTSFLIB_DIR = ROOT / "external" / "MM-TSFlib"


def _zip_url(repo: str, commit: str) -> str:
    return f"https://codeload.github.com/{repo}/zip/{commit}"


def _fetch_github(repo: str, commit: str, dest: Path, keep=None) -> None:
    """Extract ``repo@commit`` into ``dest`` (only top-level entries in ``keep``, if given)."""
    print(f"downloading {repo}@{commit[:7]} ...", flush=True)
    with urllib.request.urlopen(_zip_url(repo, commit), timeout=600) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    prefix = z.namelist()[0].split("/")[0] + "/"
    dest.mkdir(parents=True, exist_ok=True)
    for name in z.namelist():
        rel = name[len(prefix):]
        if not rel or name.endswith("/"):
            continue
        if keep is not None and rel.split("/")[0] not in keep:
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        with z.open(name) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)


def to_windows_checkout(b: bytes) -> bytes:
    """Lone LF -> CRLF: what ``git clone`` with ``core.autocrlf=true`` (Windows) writes.

    Why this exists (found by R7's clean-room test): Time-MMD's blobs use LF, but every committed
    result was computed from a Windows clone of it, whose checkout turned LF into CRLF — **including
    the line breaks inside the report texts**. Those ``\\r\\n`` reached the prompts, so the committed
    GPU answers' fingerprints encode them (166 Model 5 prompts differ otherwise) and the loaders
    rightly refuse LF data. Rewriting the prompts would invalidate every frozen result, so the
    download reproduces the bytes the results were computed from — on every OS — and the manifest
    checks them raw. (Verified: the conversion reproduces all 30 files of the original clone.)"""
    return re.sub(rb"(?<!\r)\n", b"\r\n", b)


def fetch_timemmd() -> None:
    _fetch_github("AdityaLab/Time-MMD", TIMEMMD_COMMIT, TIMEMMD_DIR,
                  keep={"numerical", "textual", "readme.MD"})
    for p in list((TIMEMMD_DIR / "numerical").rglob("*.csv")) + list((TIMEMMD_DIR / "textual").rglob("*.csv")):
        p.write_bytes(to_windows_checkout(p.read_bytes()))


def fetch_mmtsflib() -> None:
    _fetch_github("AdityaLab/MM-TSFlib", MMTSFLIB_COMMIT, MMTSFLIB_DIR)


def fetch_fnspid() -> None:
    from src.data import fnspid
    if fnspid.main() != 0:
        raise SystemExit("FNSPID download failed")


# --- digests ------------------------------------------------------------------------------------

def file_digest(path: Path, normalise: bool = False) -> str:
    """sha256 of the file's bytes. Raw for downloaded data (line endings are content here: see
    :func:`to_windows_checkout`); ``normalise`` (CRLF -> LF) only for files *we* write with pandas,
    whose newline is the platform's."""
    b = path.read_bytes()
    if normalise:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def table_digest(path: Path) -> str:
    """sha256 of a parquet table's content (canonical CSV), independent of the pyarrow version."""
    df = pd.read_parquet(path)
    return hashlib.sha256(df.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def tracked_files() -> dict[str, Path]:
    """Every raw file a reported result is derived from (relative path -> absolute path)."""
    files = {}
    for sub in ("numerical", "textual"):
        for p in sorted((TIMEMMD_DIR / sub).rglob("*.csv")):
            files[p.relative_to(ROOT).as_posix()] = p
    for p in sorted((FNSPID_DIR / "prices").glob("*.csv")):
        files[p.relative_to(ROOT).as_posix()] = p
    for p in (FNSPID_DIR / "selection_ranking.csv", FNSPID_DIR / "benzinga_frozen_tickers.parquet"):
        files[p.relative_to(ROOT).as_posix()] = p
    return files


def digest(path: Path) -> str:
    if path.suffix == ".parquet":
        return table_digest(path)
    return file_digest(path, normalise=path.name == "selection_ranking.csv")


def write_manifest() -> dict:
    m = {"Time-MMD": TIMEMMD_COMMIT, "MM-TSFlib": MMTSFLIB_COMMIT,
         "files": {rel: digest(p) for rel, p in tracked_files().items()}}
    MANIFEST.write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8")
    return m


def verify(skip_fnspid: bool = False) -> int:
    """Compare every manifest entry with the file on disk; return the number of problems."""
    want = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    bad = 0
    for rel, sha in want.items():
        if skip_fnspid and rel.startswith("data/FNSPID/"):
            continue
        p = ROOT / rel
        if not p.exists():
            print(f"  MISSING  {rel}")
            bad += 1
        elif digest(p) != sha:
            print(f"  CHANGED  {rel}")
            bad += 1
    n = sum(1 for r in want if not (skip_fnspid and r.startswith("data/FNSPID/")))
    print(f"data check: {n - bad} / {n} files match src/data/manifest.json")
    return bad


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-fnspid", action="store_true", help="Time-MMD only (P1–R3)")
    ap.add_argument("--with-mmtsflib", action="store_true", help="also fetch MM-TSFlib (retraining)")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--write-manifest", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.write_manifest:
        print(f"{len(write_manifest()['files'])} digests -> {MANIFEST}")
        return 0
    if not a.verify_only:
        if not (TIMEMMD_DIR / "numerical").exists():
            fetch_timemmd()
        if not a.skip_fnspid and not (FNSPID_DIR / "benzinga_frozen_tickers.parquet").exists():
            fetch_fnspid()
        if a.with_mmtsflib and not (MMTSFLIB_DIR / "models").exists():
            fetch_mmtsflib()
    return 1 if verify(a.skip_fnspid) else 0


if __name__ == "__main__":
    raise SystemExit(main())
