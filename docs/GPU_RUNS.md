# GPU runs — what ran on Colab / Kaggle, with which pins, and how to rerun it

`reproduce.py` never uses a GPU: every GPU output is committed in `results/qwen/` and re-verified
by its loader (each answer's prompt is rebuilt from the data and its fingerprint must match). This
page is for re-running those GPU jobs, e.g. to check that a fresh run reproduces our answers.

Generated from the committed `results/qwen/*_run_meta.json` (R7, 2026-10-19). The **setup string**
is the provenance key every loader checks: one file = one setup; answers from two setups are never
mixed in one comparison (`qwen_forecaster.qwen_rows`, `r3_answers`, `r4_answers`).

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
* **P2.4** (`p2_qwen`) loaded Qwen2.5-1.5B without a pinned revision (Hub `main`, 2026-10-08). P5 pinned
  `989aa79` and its 40-prompt probe reproduced every P2.4 answer exactly (0 mismatches), so P2.4's
  answers are certified for that revision and P5 reuses them for byte-identical prompts.
* **Colab** runs (P2–R2): torch 2.11.0+cu130, transformers 5.18.0, Tesla T4, fp16, greedy, batch 1.
  **Kaggle** runs (R2 Qwen-7B, R3, R4): torch 2.11.0+cu128, transformers 5.16.1, bitsandbytes 0.50.2
  (NF4 for Qwen2.5-7B), Tesla T4. R2 moved Qwen-7B to Kaggle by amendment (Colab RAM).
* Every Kaggle run recorded the sha256 of its input file; `reproduce.py` checks that the regenerated
  export matches it byte for byte (the "GPU-run inputs" block of its report).
* Forecasters: greedy decoding, batch size 1 (no padding), 24 new tokens. Judges: scored answers
  (teacher forcing over the 3 allowed answers), no sampling. Determinism checks: 20 / 20 identical
  re-generations in every run that recorded one.

## How to rerun one job

1. **Data and texts** — `python -m src.data.download`, then `python reproduce.py --phases P2 P3`
   (and R3 / R4 for those runs) so the text conditions exist.
2. **Export the input** — the command in the "Input ← export" column (for `qwen_inputs_p5.csv`:
   `python -m src.models.export_qwen_inputs --p5`). For Kaggle runs, check
   `sha256sum results/qwen/<input>` against the table: equal means your run asks exactly our
   prompts.
3. **Build the notebook** — `python notebooks/build_<...>.py` regenerates the `.ipynb` from the
   prompt modules (the prompt code is pasted in verbatim; tests fail if it drifts).
4. **Run it**
   * *Colab:* Runtime → T4 GPU; upload the input CSV to `/content` (or `MyDrive/textguard/` with Drive
     mounted); Run all. Outputs are written incrementally and the loop resumes after a disconnect.
   * *Kaggle:* create a Dataset holding the input CSV, attach it, Accelerator → GPU T4, Internet on;
     Run all. Outputs go to `/kaggle/working/`.
   The first cells run the reproducibility probe (40 prompts answered before) and stop with a
   message if the setup does not reproduce them. **A different GPU / torch / transformers gives a
   different setup string**; our loaders then refuse to mix its answers with ours, by design.
5. **Bring the outputs back** — copy `<run>_predictions.csv` and `<run>_run_meta.json` into
   `results/qwen/` (under a new name if you want to compare with ours), then run the matching
   analysis (`python reproduce.py --phases <phase>`).

Free-tier note: Colab / Kaggle images change over time. Our setup strings record the versions we
had. A newer image will usually produce identical greedy answers for most prompts, but not
guaranteed for all, which is why the probe exists.
