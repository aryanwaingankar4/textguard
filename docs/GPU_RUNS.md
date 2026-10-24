# GPU runs: what ran on Colab / Kaggle and how to run it again

`reproduce.py` never needs a GPU. All GPU outputs are stored in `results/qwen/`, and each loader
rebuilds every prompt from the data and checks its fingerprint. This page is only for redoing a
GPU job, for example to see whether a new run gives our answers.

The table comes from the stored `results/qwen/*_run_meta.json` (R7, 2026-10-19). The **setup
string** identifies the hardware and software of a run. One file holds one setup, and the loaders
never mix answers from two setups in one comparison (`qwen_forecaster.qwen_rows`, `r3_answers`,
`r4_answers`).

| Output (in `results/qwen/`) | Notebook | Platform | Model @ revision | Setup string | Input ← export | Input sha256 (recorded by the run) | Rows | Reproducibility probe |
|---|---|---|---|---|---|---|---|---|
| `p2_qwen_predictions.csv` | `notebooks/p2_qwen_forecaster.ipynb` | Colab | `Qwen/Qwen2.5-1.5B-Instruct` @ `main (not recorded)` | `Tesla T4 \| torch 2.11.0+cu130 \| transformers 5.18.0 \| Qwen/Qwen2.5-1.5B-Instruct` | `qwen_inputs.csv` ← `python -m src.models.export_qwen_inputs` | — | 986 | — |
| `p5_qwen_predictions.csv` | `notebooks/p5_qwen_conditions.ipynb` | Colab | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa79` | `Tesla T4 \| torch 2.11.0+cu130 \| transformers 5.18.0 \| Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | `qwen_inputs_p5.csv` ← `python -m src.models.export_qwen_inputs --p5` | — | 2215 | 40 prompts, 0 mismatches |
| `p6_judge_predictions.csv` | `notebooks/p6_direction_judge.ipynb` | Colab | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa79` | `Tesla T4 \| torch 2.11.0+cu130 \| transformers 5.18.0 \| Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | `judge_inputs.csv` ← `python -m src.textguard.export_judge_inputs` | — | 2490 | 40 prompts, 0 mismatches |
| `r1_qwen_embeddings.npz` | `notebooks/r1_qwen_embeddings.ipynb` | Colab | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa79` | `Tesla T4 \| torch 2.11.0+cu130 \| transformers 5.18.0 \| Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | `r1_embed_inputs.csv` ← `python -m src.models.qwen_embeddings export` | — | 1994 | 40 prompts, 0 mismatches |
| `r2_m5prompts_fp16_predictions.csv` | `notebooks/r2_m5_prompts.ipynb` | Colab | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa79` | `Tesla T4 \| torch 2.11.0+cu130 \| transformers 5.18.0 \| fp16 \| Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | `qwen_inputs_p5.csv` ← `python -m src.models.export_qwen_inputs --p5` | — | 9099 | 40 prompts, 0 mismatches |
| `r2_phi35_fp16_predictions.csv` | `notebooks/r2_phi35.ipynb` | Colab | `microsoft/Phi-3.5-mini-instruct` @ `2fe1924` | `Tesla T4 \| torch 2.11.0+cu130 \| transformers 5.18.0 \| fp16 \| microsoft/Phi-3.5-mini-instruct@2fe192450127e6a83f7441aef6e3ca586c338b77` | `qwen_inputs_p5.csv` ← `python -m src.models.export_qwen_inputs --p5` | — | 3451 | — |
| `r2_qwen7b_nf4_kaggle_predictions.csv` | `notebooks/r2_qwen7b_kaggle.ipynb` | Kaggle | `Qwen/Qwen2.5-7B-Instruct` @ `a09a354` | `Tesla T4 \| torch 2.11.0+cu128 \| transformers 5.16.1 \| nf4 bitsandbytes 0.50.2 \| Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28 \| kaggle` | `qwen_inputs_p5.csv` ← `python -m src.models.export_qwen_inputs --p5` | `89260e2a627a…` | 3451 | — |
| `r3_judge_kaggle_predictions.csv` | `notebooks/r3_judge_kaggle.ipynb` | Kaggle | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa79` | `Tesla T4 \| torch 2.11.0+cu128 \| transformers 5.16.1 \| Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \| kaggle` | `r3_inputs.csv` ← `python -m src.models.export_r3_inputs` | `136a14160935…` | 2411 | 40 prompts, 0 mismatches |
| `r3_m5_fp16_kaggle_predictions.csv` | `notebooks/r3_m5_kaggle.ipynb` | Kaggle | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa79` | `Tesla T4 \| torch 2.11.0+cu128 \| transformers 5.16.1 \| fp16 \| Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \| kaggle` | `r3_inputs.csv` ← `python -m src.models.export_r3_inputs` | `136a14160935…` | 6409 | 40 prompts, 0 mismatches |
| `r3_qwen7b_nf4_kaggle_predictions.csv` | `notebooks/r3_qwen7b_kaggle.ipynb` | Kaggle | `Qwen/Qwen2.5-7B-Instruct` @ `a09a354` | `Tesla T4 \| torch 2.11.0+cu128 \| transformers 5.16.1 \| nf4 bitsandbytes 0.50.2 \| Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28 \| kaggle` | `r3_inputs.csv` ← `python -m src.models.export_r3_inputs` | `136a14160935…` | 6409 | — |
| `r4_judge_v1_kaggle_predictions.csv` | `notebooks/r4_judge_v1_kaggle.ipynb` | Kaggle | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa79` | `Tesla T4 \| torch 2.11.0+cu128 \| transformers 5.16.1 \| Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \| kaggle` | `r4_heldout_inputs.csv` ← `python -m src.models.export_r4_inputs` | `cecbdc9b1408…` | 2913 | 40 prompts, 0 mismatches |
| `r4_judge_v2_kaggle_predictions.csv` | `notebooks/r4_judge_v2_kaggle.ipynb` | Kaggle | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa79` | `Tesla T4 \| torch 2.11.0+cu128 \| transformers 5.16.1 \| Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \| kaggle` | `r4_heldout_inputs.csv` ← `python -m src.models.export_r4_inputs` | `cecbdc9b1408…` | 21312 | 40 prompts, 0 mismatches |
| `r4_m5_fp16_kaggle_predictions.csv` | `notebooks/r4_m5_kaggle.ipynb` | Kaggle | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa79` | `Tesla T4 \| torch 2.11.0+cu128 \| transformers 5.16.1 \| fp16 \| Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \| kaggle` | `r4_heldout_inputs.csv` ← `python -m src.models.export_r4_inputs` | `cecbdc9b1408…` | 9145 | 40 prompts, 0 mismatches |
| `r4_qwen7b_nf4_kaggle_predictions.csv` | `notebooks/r4_qwen7b_kaggle.ipynb` | Kaggle | `Qwen/Qwen2.5-7B-Instruct` @ `a09a354` | `Tesla T4 \| torch 2.11.0+cu128 \| transformers 5.16.1 \| nf4 bitsandbytes 0.50.2 \| Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28 \| kaggle` | `r4_heldout_inputs.csv` ← `python -m src.models.export_r4_inputs` | `cecbdc9b1408…` | 9145 | — |
| `r4v2_val_judge_kaggle_predictions.csv` | `notebooks/r4_v2_val_judge_kaggle.ipynb` | Kaggle | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa79` | `Tesla T4 \| torch 2.11.0+cu128 \| transformers 5.16.1 \| Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \| kaggle` | `r4v2_val_inputs.csv` ← `python -m src.textguard.export_v2_inputs` | `a155c85c0766…` | 2898 | 40 prompts, 0 mismatches |

Notes:
* **P2.4** (`p2_qwen`) used Qwen2.5-1.5B without a fixed revision (Hub `main`, 2026-10-08). P5 fixed
  `989aa79`, and its 40-prompt probe got every P2.4 answer back exactly, so those answers count as
  that revision and P5 reuses them for identical prompts.
* **Colab** (P2–R2): torch 2.11.0+cu130, transformers 5.18.0, Tesla T4, fp16, greedy, batch 1.
  **Kaggle** (R2 Qwen-7B, R3, R4): torch 2.11.0+cu128, transformers 5.16.1, bitsandbytes 0.50.2
  (NF4 for Qwen2.5-7B), Tesla T4. Qwen-7B moved to Kaggle in R2 by amendment (Colab ran out of RAM).
* Each Kaggle run saved the sha256 of its input file; `reproduce.py` checks the rebuilt input
  against it (the "GPU-run inputs" part of its report).
* Forecasters: greedy, batch 1 (no padding), 24 new tokens. Judges: the three allowed answers are
  scored, nothing is sampled. Every run that tested it regenerated 20 / 20 answers identically.

## Running one job again

1. **Data and texts:** `python -m src.data.download`, then `python reproduce.py --phases P2 P3`
   (plus R3 / R4 for those runs) so the text conditions exist.
2. **Input file:** run the command in the "Input ← export" column (e.g.
   `python -m src.models.export_qwen_inputs --p5`). For Kaggle runs compare
   `sha256sum results/qwen/<input>` with the table; if equal, you are asking our exact prompts.
3. **Notebook:** `python notebooks/build_<...>.py` writes the `.ipynb` from the prompt modules (the
   prompt code is copied in as is; a test fails if it drifts).
4. **Run:**
   * *Colab:* Runtime → T4 GPU; upload the input CSV to `/content` (or `MyDrive/textguard/` with
     Drive mounted); Run all. Results are saved as it goes and it resumes after a disconnect.
   * *Kaggle:* make a Dataset with the input CSV, attach it, Accelerator → GPU T4, Internet on;
     Run all. Results go to `/kaggle/working/`.
   The first cells re-ask 40 known prompts and stop with a message if the answers differ. **A
   different GPU, torch or transformers gives a different setup string**, and our loaders will not
   mix those answers with ours.
5. **Bring results back:** copy `<run>_predictions.csv` and `<run>_run_meta.json` into
   `results/qwen/` (renamed if you want to compare with ours), then run the matching phase with
   `python reproduce.py --phases <phase>`.

Free Colab / Kaggle images change. Our setup strings record what we had; a newer image usually gives
the same greedy answers for most prompts but not always, hence the probe.
