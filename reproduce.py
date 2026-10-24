"""Regenerate every reported table and paper figure from the committed outputs (R7).

    python reproduce.py                 # everything: data check -> P1 ... R4 -> byte comparison
    python reproduce.py --phases P5 P6  # only some phases (their inputs must already exist)
    python reproduce.py --list          # print the steps, run nothing

What "from the committed outputs" means — nothing is retrained and no GPU is used:

* **GPU answers** (Model 5, Qwen2.5-7B, Phi-3.5, the direction judge, Qwen embeddings) are
  committed in ``results/qwen/``. Their loaders re-derive every prompt from the data, check
  each prompt's fingerprint and refuse a mismatch, so a table can only be rebuilt from answers
  to exactly the prompts that were asked. How to rerun the notebooks: ``docs/GPU_RUNS.md``.
* **CPU-trained models** (Models 1–4, the R1 heads, the R3 FinBERT runs) are committed as their
  per-unit raw predictions (``results/p2_baselines/``, ``p5_units/``, ``r1_train_units/``,
  ``r1_units/``, ``r3_finbert_units/``); the runners' ``--collect-only`` mode rebuilds the
  combined tables from them. Retraining needs MM-TSFlib, the BERT cache and ~2 h of CPU
  (``docs/REPRODUCE.md`` "Retraining").
* **Everything else** — data checks, attack texts, exports, embeddings for the similarity
  tables (BERT / FinBERT, CPU, pinned revisions), every analysis and figure — is recomputed.

At the end every committed file under ``results/`` and ``paper/figures/`` is compared byte for
byte with the version that was checked out before the run (the snapshot is taken first, so it
works in a fresh clone without touching git). Files no step rewrites are inputs by design
(``INPUTS``); a committed file that is neither rewritten nor an input is reported as an error.
Line endings: pandas writes the platform's newline and git checks text files out the same way
(``core.autocrlf``), so a difference *only* in CRLF vs LF is reported separately, never hidden.

PDF figures: ``SOURCE_DATE_EPOCH`` is fixed so matplotlib writes the same creation date every
time (the committed PDFs were re-rendered with it in R7).

Logs: ``logs/reproduce/<nn>_<step>.log``. Exit code 0 only if every step ran and every
regenerated file is identical.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs" / "reproduce"
COMPARED = ("results", "paper/figures")
SOURCE_DATE_EPOCH = "1792368000"  # 2026-10-19 00:00 UTC, the day R7 started

# Committed files that are inputs to the pipeline, never rewritten by it (each with its reason).
INPUTS = {
    "results/qwen/*_predictions.csv": "GPU answers (Colab / Kaggle) — docs/GPU_RUNS.md",
    "results/qwen/*_run_meta.json": "GPU run metadata (setup string, pins, probe results)",
    "results/qwen/r1_qwen_embeddings.npz": "Qwen embeddings (Colab) — docs/GPU_RUNS.md",
    "results/qwen/r1_qwen_embed_meta.json": "Qwen embedding run metadata",
    "results/p2_baselines/*": "Models 1–2 + persistence + Models 3–4 per-unit raw predictions (CPU)",
    "results/p5_units/*": "Models 3–4 per-unit raw predictions on the P3 conditions (CPU)",
    "results/r1_train_units/*": "R1 heads, per-unit raw predictions at training time (CPU)",
    "results/r1_units/*": "R1 heads, per-unit raw predictions on all conditions (CPU)",
    "results/r3_finbert_units/*": "R3 FinBERT heads, per-unit raw predictions (CPU)",
    "results/summary_directional_sample.csv": "P1 human labels (Aryan) — hand-made",
    "results/summary_directional_socialgood_energy.csv": "P1 human labels (Aryan) — hand-made",
    "results/summary_directional_fnspid.csv": "R4 human labels (Aryan) + Claude labels — hand-made",
    "results/summary_r4_v2_frozen.json": "TextGuard v2 frozen settings (checked, not rewritten, by `analyze_r4_val freeze`)",
    "results/r7_toolkit_kaggle/*": "R7 toolkit test on Kaggle (live Qwen judge, synthetic example) — notebook output",
    "results/.gitkeep": "placeholder",
}


@dataclass
class Step:
    phase: str
    name: str
    argv: list[str]
    note: str = ""


def _m(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


STEPS = [
    Step("data", "download+verify", _m("src.data.download"),
         "Time-MMD @00281e2 + FNSPID @bf9189c, sha256 against src/data/manifest.json"),
    Step("P1", "check_timing", _m("src.data.check_timing"), "0 timing violations"),
    Step("P2", "baselines_collect", _m("src.models.train_baselines", "--collect-only")),
    Step("P2", "export_qwen_inputs", _m("src.models.export_qwen_inputs")),
    Step("P2", "qwen_summary", _m("src.models.qwen_forecaster")),
    Step("P3", "generate_texts", _m("src.attacks.generate")),
    Step("P3", "validate_attacks", _m("src.attacks.validate_attacks")),
    Step("P5", "export_qwen_inputs_p5", _m("src.models.export_qwen_inputs", "--p5"),
         "the Model 5 prompts of the P3 conditions (also R2's Kaggle input)"),
    Step("P5", "bert_embed_real", _m("src.models.embed_texts"), "CPU, BERT @86b5e09"),
    Step("P5", "bert_embed_conditions", _m("src.models.embed_texts", "--conditions")),
    Step("P5", "models34_collect", _m("src.run_experiments", "--collect-only")),
    Step("P5", "analyze_p5", _m("src.analyze_p5")),
    Step("P5", "figures_p5", _m("src.figures_p5")),
    Step("P6", "export_judge_inputs", _m("src.textguard.export_judge_inputs")),
    Step("P6", "analyze_p6_judge", _m("src.analyze_p6", "judge")),
    Step("P6", "analyze_p6_tune", _m("src.analyze_p6", "tune")),
    Step("P6", "analyze_p6_test", _m("src.analyze_p6", "test")),
    Step("P6", "figures_p6", _m("src.figures_p6")),
    Step("R1", "qwen_embed_export", _m("src.models.qwen_embeddings", "export")),
    Step("R1", "qwen_embed_import", _m("src.models.qwen_embeddings", "import")),
    Step("R1", "finbert_embed", _m("src.analyze_r1", "embed-finbert"), "CPU, FinBERT @4556d13"),
    Step("R1", "analyze_r1_mechanism", _m("src.analyze_r1", "mechanism")),
    Step("R1", "train_r1_collect", _m("src.models.train_r1", "--collect-only")),
    Step("R1", "run_r1_collect", _m("src.run_r1", "--collect-only")),
    Step("R1", "analyze_r1_test", _m("src.analyze_r1", "test")),
    Step("R2", "analyze_r2", _m("src.analyze_r2")),
    Step("R2", "figures_r2", _m("src.figures_r2")),
    Step("R3", "adaptive_texts", _m("src.attacks.adaptive")),
    Step("R3", "validate_r3", _m("src.attacks.validate_r3")),
    Step("R3", "consistency_r3", _m("src.attacks.consistency_r3")),
    Step("R3", "export_r3_inputs", _m("src.models.export_r3_inputs")),
    Step("R3", "finbert_r3_collect", _m("src.run_r3_finbert", "--collect-only")),
    Step("R3", "analyze_r3", _m("src.analyze_r3")),
    Step("R3", "figures_r3", _m("src.figures_r3")),
    Step("R3", "explore_p3_exclusions", _m("src.explore_p3_exclusions")),
    Step("R4", "heldout_texts", _m("src.attacks.heldout")),
    Step("R4", "connectors_r4", _m("src.attacks.connectors_r4")),
    Step("R4", "export_v2_val_inputs", _m("src.textguard.export_v2_inputs")),
    Step("R4", "analyze_r4_val_v1j", _m("src.analyze_r4_val", "v1j")),
    Step("R4", "analyze_r4_val_v2", _m("src.analyze_r4_val", "v2")),
    Step("R4", "analyze_r4_val_freeze", _m("src.analyze_r4_val", "freeze"),
         "refuses if the val grid no longer gives the frozen v2 settings"),
    Step("R4", "explore_r4_v2_variants", _m("src.explore_r4_v2_variants")),
    Step("R4", "export_r4_inputs", _m("src.models.export_r4_inputs"), "asserts sha256 cecbdc9b…"),
    Step("R4", "verify_r4_runs", _m("src.models.r4_answers")),
    Step("R4", "analyze_r4", _m("src.analyze_r4")),
    Step("R4", "energy_label_check", _m("src.analyze_r4", "--energy-label-check"),
         "exploratory, outside §10g (Note 1)"),
    Step("R4", "figures_r4", _m("src.figures_r4")),
    Step("P8", "figure_pipeline", [sys.executable, str(ROOT / "paper" / "make_fig1.py")],
         "Fig. 1 of the paper (no data)"),
]
PHASES = list(dict.fromkeys(s.phase for s in STEPS))


# --- snapshot and comparison --------------------------------------------------------------------

def committed_files() -> list[Path]:
    """Files under COMPARED as checked out — from git when available, else everything on disk."""
    try:
        out = subprocess.run(["git", "ls-files", "-z", "--", *COMPARED], cwd=ROOT, check=True,
                             capture_output=True).stdout
        return [ROOT / p for p in out.decode().split("\0") if p]
    except (OSError, subprocess.CalledProcessError):
        return [p for c in COMPARED for p in (ROOT / c).rglob("*") if p.is_file()]


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def snapshot(files: list[Path]) -> dict[str, dict]:
    snap = {}
    for p in files:
        if p.exists():
            b = p.read_bytes()
            snap[p.relative_to(ROOT).as_posix()] = dict(sha=_sha(b), lf=_sha(b.replace(b"\r\n", b"\n")),
                                                        mtime=p.stat().st_mtime_ns)
    return snap


def is_input(rel: str) -> str | None:
    return next((why for pat, why in INPUTS.items() if fnmatch.fnmatch(rel, pat)), None)


def compare(before: dict[str, dict]) -> dict[str, list[str]]:
    res = {"identical": [], "identical_up_to_line_endings": [], "differs": [], "inputs": [],
           "not_regenerated": [], "deleted": []}
    for rel, s in sorted(before.items()):
        p = ROOT / rel
        if not p.exists():
            res["deleted"].append(rel)
            continue
        b = p.read_bytes()
        rewritten = p.stat().st_mtime_ns != s["mtime"]
        if not rewritten:
            res["inputs" if is_input(rel) else "not_regenerated"].append(rel)
        elif _sha(b) == s["sha"]:
            res["identical"].append(rel)
        elif _sha(b.replace(b"\r\n", b"\n")) == s["lf"]:
            res["identical_up_to_line_endings"].append(rel)
        else:
            res["differs"].append(rel)
    return res


def gpu_input_checks() -> list[tuple[str, str, bool]]:
    """Each Kaggle run recorded its input file's sha256: the regenerated export must match it."""
    rows = []
    for meta in sorted((ROOT / "results" / "qwen").glob("*_run_meta.json")):
        m = json.loads(meta.read_text(encoding="utf-8"))
        if "input_sha256" not in m:
            continue
        p = ROOT / "results" / "qwen" / m["inputs"]
        ok = p.exists() and _sha(p.read_bytes()) == m["input_sha256"]
        rows.append((meta.name.removesuffix("_run_meta.json"), m["inputs"], ok))
    return rows


# --- running ------------------------------------------------------------------------------------

def run_step(i: int, s: Step, env: dict) -> tuple[bool, float]:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{i:02d}_{s.phase}_{s.name}.log"
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(s.argv, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    return r.returncode == 0, dt


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phases", nargs="+", choices=PHASES, help="run only these phases")
    ap.add_argument("--list", action="store_true", help="print the steps and exit")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed step")
    a = ap.parse_args()
    steps = [s for s in STEPS if not a.phases or s.phase in a.phases]
    if a.list:
        for i, s in enumerate(steps, 1):
            print(f"{i:2d}. [{s.phase:>4}] {s.name:<24} {' '.join(s.argv[1:])}"
                  + (f"   # {s.note}" if s.note else ""))
        return 0

    before = snapshot(committed_files())
    env = dict(os.environ, SOURCE_DATE_EPOCH=SOURCE_DATE_EPOCH, PYTHONIOENCODING="utf-8",
               PYTHONHASHSEED="0", MPLBACKEND="Agg")
    failed, ran, t_all = [], 0, time.time()
    for i, s in enumerate(steps, 1):
        print(f"[{i:2d}/{len(steps)}] {s.phase:>4} {s.name:<24}", end=" ", flush=True)
        ok, dt = run_step(i, s, env)
        ran += 1
        print(f"{'ok  ' if ok else 'FAIL'} {dt:7.1f}s", flush=True)
        if not ok:
            failed.append(s.name)
            if not a.keep_going:
                print(f"stopped — see {LOGS}/{i:02d}_{s.phase}_{s.name}.log")
                break
    print(f"\nsteps: {ran - len(failed)} ok, {len(failed)} failed, {len(steps) - ran} not run "
          f"({(time.time() - t_all) / 60:.1f} min)")

    res = compare(before)
    if a.phases:  # a partial run cannot regenerate the other phases' files
        res["not_regenerated"] = []
    print("\ncomparison with the committed files:")
    for k in ("identical", "identical_up_to_line_endings", "differs", "deleted", "inputs",
              "not_regenerated"):
        print(f"  {k:<30} {len(res[k]):4d}")
        if k in ("identical_up_to_line_endings", "differs", "deleted", "not_regenerated"):
            for rel in res[k]:
                print(f"      {rel}")
    checks = gpu_input_checks()
    if checks and not a.phases:
        print("\nGPU-run inputs regenerated byte for byte (sha256 recorded by each Kaggle run):")
        for run, inp, ok in checks:
            print(f"  {'ok  ' if ok else 'FAIL'} {run:<28} {inp}")
    report = dict(steps_failed=failed, **res, gpu_inputs=[dict(run=r, inputs=i, ok=o)
                                                          for r, i, o in checks])
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    bad = failed or res["differs"] or res["deleted"] or res["not_regenerated"] or \
        (not a.phases and not all(o for _, _, o in checks))
    print(f"\n{'REPRODUCED' if not bad else 'NOT REPRODUCED'} — report: {LOGS / 'report.json'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
