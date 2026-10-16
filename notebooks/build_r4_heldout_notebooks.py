"""Build R4's four held-out Kaggle notebooks (pre-registration PROJECT_STATUS §10g, "Runs").

* ``r4_m5_kaggle.ipynb``       — Model 5 (Qwen2.5-1.5B @ M5's revision, fp16), M5's prompt only,
  P5's 40-prompt probe first (recorded only)
* ``r4_qwen7b_kaggle.ipynb``   — Qwen2.5-7B (R2's revision, 4-bit NF4), M5's prompt, R2's smoke test
* ``r4_judge_v1_kaggle.ipynb`` — TextGuard v1's judge (P6's prompt + the held-out questions)
* ``r4_judge_v2_kaggle.ipynb`` — TextGuard v2's judge (``judge_prompt_v2.py`` + held-out subjects)

All are R3's notebooks (``build_r3_notebooks``) with exact, asserted edits only: the input file
(``r4_heldout_inputs.csv``, sha256 embedded), output names, test split only, M5's prompt only, the
held-out question module pasted verbatim (``src/textguard/heldout_questions.py``) and registered in
the notebook — v1's ``judge_prompt.py`` and v2's ``judge_prompt_v2.py`` are pasted unchanged. The
model / scoring / generation code is R3's byte for byte.

After changing the texts, a prompt module or the export:
``python -m src.models.export_r4_inputs``, then ``python notebooks/build_r4_heldout_notebooks.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT))
import build_r3_notebooks as r3  # noqa: E402
from build_p2_qwen_notebook import code, md  # noqa: E402
from build_r2_notebooks import _edit, _src  # noqa: E402
from src.models.export_r4_inputs import R4_INPUTS, sha256  # noqa: E402

INPUT_NAME = R4_INPUTS.name
HQ_SRC = ROOT / "src" / "textguard" / "heldout_questions.py"
V2_SRC = ROOT / "src" / "textguard" / "judge_prompt_v2.py"
FILES = {"m5": "r4_m5_kaggle.ipynb", "qwen7b": "r4_qwen7b_kaggle.ipynb",
         "judge_v1": "r4_judge_v1_kaggle.ipynb", "judge_v2": "r4_judge_v2_kaggle.ipynb"}
SETTINGS = r3.KAGGLE_SETTINGS.replace("r3_inputs.csv", INPUT_NAME).replace(
    "`flipped`, `exaggerated`, `a1_selective`, `a2_forward`, `a3_magnitude` rows",
    "`flipped`, `a1_selective`, `a2_forward`, `a3_magnitude` rows")


def _sub(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"{old!r} occurs {text.count(old)}x"
    return text.replace(old, new)


def _edit_all(cells: list, key: str, old: str, new: str, n: int) -> list:
    """Replace every occurrence of ``old`` in the code cell starting with ``key`` — exactly ``n`` of them."""
    hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and _src(c).startswith(key)]
    assert len(hits) == 1, f"{len(hits)} cells start with {key!r}"
    src = _src(cells[hits[0]])
    assert src.count(old) == n, f"{old!r} occurs {src.count(old)}x in {key!r}, expected {n}"
    out = list(cells)
    out[hits[0]] = code(src.replace(old, new))
    return out


# ---- forecasters ----------------------------------------------------------------------------------

def forecaster(run: str, sha: str, n_forecast: int) -> dict:
    nb = r3.build_forecaster(run, sha)
    cells = nb["cells"]
    cfg = r3.RUNS[run]
    check = ("**Probe (cell 6):** P5's 40 prompts are re-run and compared with P2.4's answers — recorded only."
             if cfg["probe"] else
             "**Smoke test (cell 6):** 20 answers; ≥ 10 unparseable → the run stops (tell Claude, change nothing).")
    cells[0] = md(f"""
# R4 — {cfg['title'].split(' — ', 1)[1]} on the held-out domains — Kaggle

TextGuard R4 (pre-registration PROJECT_STATUS §10g): forecasts every held-out window (Energy weekly
gasoline price; FNSPID weekly closes of 10 stocks) under real, none, flipped, a1_selective, a2_forward
and (Energy) a3_magnitude texts, with **M5's prompt only**. Horizon 1. Test only (every held-out text
period is a test period).

* Input: `{INPUT_NAME}` in any attached dataset; its sha256 must match the export.
* Output: `r4_{run}_{cfg['quant']}_kaggle_predictions.csv` + `…_run_meta.json`, appended every 10 rows.
* {check}
{SETTINGS}""")
    cells = _edit(cells, "# 0.", "(pre-registered, §10f)", "(pre-registered, §10g)")
    cells = _edit(cells, "# 0.", 'RUN_VARIANTS = ["p5", "sk"]', 'RUN_VARIANTS = ["p5"]')
    cells = _edit(cells, "# 0.", 'SPLITS = ["test", "val"]          # test first; val is descriptive in R3 (and R4\'s dev data)',
                  'SPLITS = ["test"]                  # held-out: every window is a test window')
    cells = _edit_all(cells, "# 3.", "r3_inputs.csv", INPUT_NAME, 3)
    cells = _edit(cells, "# 3.", 'TAG = f"r3_{RUN}_{QUANT}_kaggle"', 'TAG = f"r4_{RUN}_{QUANT}_kaggle"')
    cells = _edit(cells, "# 3.", 'assert len(inp) == 3451 or "LOCAL_TEST" in globals(), f"expected 3451 forecast rows, got {len(inp)}"',
                  f'assert len(inp) == {n_forecast} or "LOCAL_TEST" in globals(), f"expected {n_forecast} forecast rows, got {{len(inp)}}"')
    cells = _edit(cells, "# 3.", "# 3. Paths and inputs (one CSV for all three R3 notebooks; forecast + probe rows used here)",
                  "# 3. Paths and inputs (one CSV for all four R4 notebooks; forecast + probe rows used here)")
    i = next(k for k, c in enumerate(cells) if _src(c).startswith("# 4."))
    cells[i] = code("""
# 4. Look at the R4 prompts before spending GPU time (one per attack, M5's prompt)
for cond in ("flipped", "a1_selective", "a2_forward", "a3_magnitude"):
    r = inp[(inp.condition == cond) & (inp.attacked == "True")].iloc[0]
    print(f"===== {cond} | {r.domain} | system: {system_for('p5')}")
    print(build_variant_prompt("p5", r.description, r.unit, r.freq, json.loads(r.lookback), r.text or None), "\\n")
""")
    return r3._notebook(cells)


# ---- judges ---------------------------------------------------------------------------------------

def _hq_cell(register: str) -> dict:
    return code("# Generated verbatim from src/textguard/heldout_questions.py\n"
                + HQ_SRC.read_text(encoding="utf-8").rstrip("\n")
                + f"\n\n# held-out questions registered in place (v1's / v2's prompt files are unchanged)\n{register}\n")


def judge(which: str, sha: str) -> dict:
    cells = r3.build_judge(sha)["cells"]
    tag = f"r4_judge_{which}_kaggle"
    if which == "v2":
        i_mod = next(k for k, c in enumerate(cells)
                     if _src(c).startswith("# Generated verbatim from src/textguard/judge_prompt.py"))
        cells[i_mod] = code("# Generated verbatim from src/textguard/judge_prompt_v2.py\n" + V2_SRC.read_text(encoding="utf-8"))
        hq = _hq_cell("SUBJECTS.update(HELDOUT_SUBJECTS)")
        module = "`judge_prompt_v2.py` (v2: every unit × change / outlook)"
    else:
        hq = _hq_cell("QUESTIONS.update(HELDOUT_QUESTIONS)")
        module = "`judge_prompt.py` (v1, P6's prompt unchanged: whole text × change)"
    i_mod = next(k for k, c in enumerate(cells) if _src(c).startswith("# Generated verbatim from src/textguard/judge_prompt"))
    cells.insert(i_mod + 1, hq)
    cells[0] = md(f"""
# R4 — TextGuard {which}'s judge on the held-out domains (Qwen2.5-1.5B-Instruct) — Kaggle

TextGuard R4 (pre-registration PROJECT_STATUS §10g). P6's judge model and scoring code; prompts from
{module}, plus the held-out questions (`heldout_questions.py`, approved 2026-10-16). Every distinct
held-out text (or unit) of real and every attack. P5's 40-prompt probe runs first and is recorded.

* Input: `{INPUT_NAME}` in any attached dataset; sha256 checked.
* Output: `{tag}_predictions.csv` + `{tag}_run_meta.json`, appended every 20.
{SETTINGS}""")
    cells = _edit(cells, "# 0.", "(pre-registered, §10f)", "(pre-registered, §10g)")
    cells = _edit_all(cells, "# 3.", "r3_inputs.csv", INPUT_NAME, 3)
    cells = _edit(cells, "# 3.", 'OUT = f"{OUT_DIR}/r3_judge_kaggle_predictions.csv"', f'OUT = f"{{OUT_DIR}}/{tag}_predictions.csv"')
    cells = _edit(cells, "# 3.", 'META = f"{OUT_DIR}/r3_judge_kaggle_run_meta.json"', f'META = f"{{OUT_DIR}}/{tag}_run_meta.json"')
    cells = _edit(cells, "# 3.", "# 3. Paths and inputs (one CSV for all three R3 notebooks; judge + probe rows used here)",
                  f"# 3. Paths and inputs (one CSV for all four R4 notebooks; {'judge_v2' if which == 'v2' else 'judge'} + probe rows used here)")
    if which == "v2":
        cells = _edit(cells, "# 3.", 'df[df.row_type == "judge"]', 'df[df.row_type == "judge_v2"]')
    cells = _edit(cells, "# 4.", 'rows[rows.domain == "Health_US"].iloc[0]', 'rows[rows.domain == "Energy"].iloc[0]')
    return r3._notebook(cells)


def build_all() -> dict[str, dict]:
    if not R4_INPUTS.exists():
        raise FileNotFoundError("run python -m src.models.export_r4_inputs first")
    import pandas as pd
    sha = sha256()
    n = int((pd.read_csv(R4_INPUTS, usecols=["row_type"]).row_type == "forecast").sum())
    return {FILES["m5"]: forecaster("m5", sha, n), FILES["qwen7b"]: forecaster("qwen7b", sha, n),
            FILES["judge_v1"]: judge("v1", sha), FILES["judge_v2"]: judge("v2", sha)}


if __name__ == "__main__":
    for name, nb in build_all().items():
        path = ROOT / "notebooks" / name
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {path}")
