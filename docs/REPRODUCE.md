# Reproducing the results

Aim: from a fresh clone, rebuild every table and figure without help.

## 1. Environment

* Python 3.14.3 (`.python-version`). The results were made on Windows 11 x64 with a CPU only.
* `python -m pip install -r requirements-lock.txt` gives the exact environment (116 packages;
  `pywin32` on Windows only). `requirements.txt` holds the direct, pinned dependencies.
* Models fetched from the Hugging Face Hub on first use: BERT `google-bert/bert-base-uncased` @
  `86b5e09` and FinBERT `ProsusAI/finbert` @ `4556d13` (CPU embeddings for the similarity tables).
  Qwen is not downloaded; its answers are in the repository.
* The GPU runs (Colab / Kaggle) had their own setups, listed in `docs/GPU_RUNS.md`.

## 2. Data

```bash
python -m src.data.download
```

Gets Time-MMD @ `00281e2` (~35 MB zip) and the FNSPID subset @ `bf9189c` (reads the 332 MB Benzinga
block, keeps ~10 MB, and re-checks the stock selection), then checks all 47 files against
`src/data/manifest.json`. Time-MMD is written with Windows line endings, also inside the reports,
because every stored result came from a Windows clone (`DATA_CARD.md`); an LF copy changes 166
prompts and fails, so don't use `git clone` for it. `--skip-fnspid` gets Time-MMD only (enough for
P1–R3); `--with-mmtsflib` adds the MM-TSFlib network code (only for retraining).

## 3. Run

```bash
python reproduce.py              # or ./reproduce.sh: builds .venv-repro, installs, downloads, runs
python reproduce.py --list       # show the steps (47 in the R7 clean room; P8 added Fig. 1)
python reproduce.py --phases R4  # one phase (its inputs must exist)
```

| Phase | Rebuilt |
|---|---|
| data | download and hash check |
| P1 | timing check (must report 0 problems) |
| P2 | Models 1–4 predictions from the stored units, `summary_p2_baselines`, `summary_p2_lr_tuning`, Model 5's prompts, `summary_p2_qwen` |
| P3 | the 6 text conditions (`texts/`, marked synthetic), `summary_p3_attack_validation` |
| P5 | Model 5 prompts, BERT embeddings, Models 3–4 predictions, all `summary_p5_*`, P5 figures |
| P6 | judge inputs and checks, validation tuning (`summary_p6_frozen.json` must come out the same), test analysis, P6 figures |
| R1 | Qwen embeddings (from the stored `.npz`), FinBERT embeddings, mechanism table, R1 predictions, all `summary_r1_*` |
| R2 | all `summary_r2_*` and the figure (the GPU answers are stored) |
| R3 | A1–A3 texts and checks, the Amendment 1 scan, the Kaggle input (hash checked), FinBERT predictions, all `summary_r3_*`, figure, the P3 exclusion recheck |
| R4 | held-out texts and scan, connector flags, v2's validation export and grid, `freeze` (fails if v2's settings would change), dev variants, the held-out Kaggle input (hash checked), run checks, all `summary_r4_*`, figures |

Not rerun:
* GPU jobs (Qwen2.5-1.5B forecaster and judge, Qwen2.5-7B, Phi-3.5, Qwen embeddings). Their
  outputs are in `results/qwen/`; each loader rebuilds the prompts and checks every answer's
  fingerprint. To redo them see `docs/GPU_RUNS.md`.
* CPU training (Models 1–4, R1 heads, R3 FinBERT). Their per-unit predictions are stored
  (`results/p2_baselines/`, `p5_units/`, `r1_train_units/`, `r1_units/`, `r3_finbert_units/`), and
  `--collect-only` rebuilds the combined tables.

Never rewritten, on purpose: GPU answers and run metadata, per-unit predictions, the hand labels
(`summary_directional_*.csv`) and v2's frozen settings (`summary_r4_v2_frozen.json`, checked by
`analyze_r4_val freeze`). `reproduce.py` lists them as `inputs` and flags any other stored file that
no step rewrote.

**How the check works.** Before step 1, `reproduce.py` hashes every stored file in `results/` and
`paper/figures/`. At the end each one is `identical`, `identical_up_to_line_endings` (shown, never
hidden), `differs`, `deleted`, `inputs` or `not_regenerated`, and each regenerated GPU input is
compared with the sha256 its Kaggle run recorded. Report: `logs/reproduce/report.json`, step logs
beside it. PDFs use a fixed `SOURCE_DATE_EPOCH` (2026-10-19) so they rebuild byte for byte.

The paper document is rebuilt separately (`paper/build_paper.py`, in its own venv with
python-docx); `paper/check_numbers.py` and `paper/check_overlap.py` check it.

## 4. Retraining (optional; no table needs it)

```bash
python -m src.data.download --with-mmtsflib
python -m src.models.embed_texts && python -m src.models.embed_texts --conditions
python -m src.models.train_baselines --fresh --no-mlflow     # Models 1–4, ~1 h CPU
python -m src.run_experiments --fresh --no-mlflow            # Models 3–4 on the conditions, ~15 min
python -m src.analyze_r1 embed-finbert
python -m src.models.train_r1 --fresh --no-mlflow             # R1 heads, ~25 min
python -m src.run_r1 --fresh --no-mlflow                      # ~15 min
python -m src.run_r3_finbert --no-mlflow                      # ~10 min
```

On one machine training is deterministic (a second P2 run matched to ≤ 1.3e-7, CSV rounding). Other
hardware or library builds can shift the last digit of a float, which is why the stored units, not a
retrain, are the reference.

## 5. Clean-room test (R7): `REPRODUCED`, 2026-10-24

| | |
|---|---|
| Code | fresh `git clone` of `b275568` from GitHub |
| Environment | new venv from `requirements-lock.txt` (Python 3.14.3; 116 / 116 packages match); empty `HF_HOME` |
| Machine | Windows 11 x64, 16 GB RAM, CPU, plain PowerShell window (Claude Code's memory guard had stopped two earlier tries) |
| Command | `python reproduce.py` |
| Steps | **47 / 47 ok** in 61.6 min (download 147 s, BERT 533 + 275 s, FinBERT 1,176 s) |
| Stored files | **147 identical**, 0 line-ending-only, **0 differ**, 0 deleted, 658 inputs, 0 not regenerated |
| GPU inputs | **9 / 9 identical** to the recorded sha256 |

Report: `logs/reproduce/report.json` (not stored; rerun to get it).

Four bugs showed up only in the clean room (the working copy already had what a fresh clone lacks).
All are fixed and **no reported number changed**.

| # | Where | Problem | Fix |
|---|---|---|---|
| 1 | step 1 → P2 | Time-MMD line endings: results came from a CRLF clone; an LF copy changes 166 prompts and the stored answers are rejected. The first manifest normalised endings and hid this | `download.py` writes CRLF exactly like the original clone (all 30 files); the manifest hashes raw bytes |
| 2 | step 1 | FNSPID download silently cut short by a dropped connection | resumable range download with a size check and retries (`fnspid.py`) |
| 3 | step 22 | FinBERT's empty-report prompt was never embedded in a fresh cache, but `analyze_r1 mechanism` needs it | `embed_finbert` embeds it in a separate call; other embeddings unchanged; that vector enters no number |
| 4 | R4 `freeze` | the judge-answer hash was taken on checked-out bytes (CRLF on Windows) instead of the stored LF form | `analyze_r4_val.git_content_sha` hashes the git-stored form; same value as before, frozen file unchanged |

Also fixed earlier: the 8 PDF figures held their creation date; they now use the fixed
`SOURCE_DATE_EPOCH`.

**Other systems.** Checked on Windows. For Linux / macOS a fix is in place but not yet run: pandas
writes the system newline, so every file with a recorded sha256 (the four GPU inputs
`qwen_inputs_p5.csv`, `r3_inputs.csv`, `r4v2_val_inputs.csv`, `r4_heldout_inputs.csv`, and R4's
attack texts `texts/r4/*.csv`) is written with `lineterminator="\r\n"`. On Windows the bytes didn't
change and all 9 hashes still match; `tests/test_crlf_exports.py` fakes a Linux newline. Time-MMD
download and `freeze` are already system-independent. On Linux, summary tables should compare as
`identical`; anything differing only in line endings is listed separately. A GitHub Actions run on
Linux is planned (P8).

**Toolkit.** `notebooks/textguard_your_model.ipynb` ran on Kaggle with the live Qwen judge on the
synthetic example: CPU, about 47 min, no errors (`results/r7_toolkit_kaggle/`,
`PROJECT_STATUS.md` §9 "R7").
