# Reproducing the results

Goal: clone the repository and regenerate every reported table and figure without asking anyone.

## 1. Environment

* **Python 3.14.3** (`.python-version`). The results were produced on Windows 11 x64, CPU only.
* `python -m pip install -r requirements-lock.txt` installs the exact environment (`pip freeze`
  of the development venv, 120 packages; `pywin32` only on Windows). `requirements.txt` lists the
  direct dependencies, pinned.
* Pinned model weights, downloaded from the Hugging Face Hub on first use:
  BERT `google-bert/bert-base-uncased` @ `86b5e09` and FinBERT `ProsusAI/finbert` @ `4556d13` (CPU
  embeddings for the similarity tables). Qwen is **not** downloaded by `reproduce.py`; its answers
  are committed.
* GPU runs (Colab / Kaggle) used their own images, recorded per run in `docs/GPU_RUNS.md`.

## 2. Data

```bash
python -m src.data.download
```

This fetches Time-MMD @ `00281e2` (GitHub zip, ~35 MB) and the FNSPID subset @ `bf9189c` (streams
the 332 MB Benzinga block, keeps ~10 MB, re-derives and asserts the frozen stock selection). It
then checks all 47 files against `src/data/manifest.json` (sha256 of the raw bytes). Time-MMD is
written with CRLF line endings, including inside report texts, because that is the form every
committed result was computed from (a Windows `git clone`; see the top of `DATA_CARD.md`). An LF copy
changes 166 prompts and fails. Use the script, not `git clone`.
`--skip-fnspid` fetches Time-MMD only (enough for P1–R3); `--with-mmtsflib` also fetches the
MM-TSFlib network code (retraining only). Licences: `DATA_CARD.md`.

## 3. Run

```bash
python reproduce.py              # or ./reproduce.sh (makes .venv-repro, installs the lock, downloads, runs)
python reproduce.py --list       # the 47 steps
python reproduce.py --phases R4  # one phase (its inputs must exist)
```

| Phase | What is regenerated |
|---|---|
| data | download + sha256 check |
| P1 | timing check (0 violations, exit code) |
| P2 | Models 1–4 combined predictions + `summary_p2_baselines`, `summary_p2_lr_tuning` (from the committed units), Model 5's prompts and `summary_p2_qwen` |
| P3 | the 6 text conditions (`texts/`, labelled synthetic) + `summary_p3_attack_validation` |
| P5 | Model 5 prompts of the P3 conditions, BERT embeddings, Models 3–4 combined predictions, every `summary_p5_*` table, P5 figures |
| P6 | judge inputs, judge validation, val tuning (rewrites `summary_p6_frozen.json`, which must come out identical), the test analysis, P6 figures |
| R1 | Qwen embeddings (imported from the committed `.npz`), FinBERT embeddings, the mechanism table, R1 heads' combined predictions, every `summary_r1_*` table |
| R2 | every `summary_r2_*` table + figure (Model 5 prompt variants, Phi-3.5, Qwen2.5-7B answers committed) |
| R3 | A1–A3 texts, validation, the Amendment 1 scan, the Kaggle input (sha256 checked), FinBERT predictions, every `summary_r3_*` table, figure, the exploratory P3-exclusion recheck |
| R4 | held-out texts + scan, connector flags, v2's val export and val grid, `freeze` (refuses if v2's frozen settings would change), dev variants, the held-out Kaggle input (sha256 asserted), run verification, every `summary_r4_*` table, figures |

**Not re-executed:**
* GPU jobs (Qwen2.5-1.5B forecaster and judge, Qwen2.5-7B, Phi-3.5, Qwen embeddings): their
  outputs are in `results/qwen/` and every loader rebuilds the prompts from the data and checks each
  answer's fingerprint. To rerun them: `docs/GPU_RUNS.md`.
* CPU training (Models 1–4, R1 heads, R3 FinBERT inference): their per-unit raw predictions are
  committed (`results/p2_baselines/`, `p5_units/`, `r1_train_units/`, `r1_units/`,
  `r3_finbert_units/`) and the runners' `--collect-only` mode rebuilds the combined tables.

**Inputs that no step rewrites, by design:** GPU answers and run metadata, the per-unit
predictions, the hand-made labels (`summary_directional_*.csv`) and v2's frozen settings
(`summary_r4_v2_frozen.json`, checked by `analyze_r4_val freeze`). `reproduce.py` lists them under
`inputs` and flags any other committed file that no step rewrote.

**Comparison.** Before the first step, `reproduce.py` hashes every committed file under `results/`
and `paper/figures/`. Afterwards it reports each as `identical`, `identical_up_to_line_endings`
(pandas writes the platform newline, git checks text out per `core.autocrlf`; reported, never
hidden), `differs`, `deleted`, `inputs` or `not_regenerated`. It also checks the sha256 of every
regenerated GPU input against the value its Kaggle run recorded. Report:
`logs/reproduce/report.json`; per-step logs beside it. PDFs are written with
`SOURCE_DATE_EPOCH` fixed (2026-10-19), so they are byte-reproducible.

## 4. Retraining (optional, not needed for any table)

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

Training is deterministic on one machine (P2: an independent re-run reproduced predictions to
≤ 1.3e-7, CSV round-off). On other hardware or library builds, expect tiny float differences in
the unit files and therefore possibly in the last printed digit of a table. That is why the
committed units, not a retrain, are the reference.

## 5. Clean-room test (R7)

_Not finished yet (R7 paused 2026-10-21): a fresh clone + fresh venv passed steps 1–9 of 47, then the run was stopped for low system memory. Result to be recorded here; see PROJECT_STATUS.md §1a._
