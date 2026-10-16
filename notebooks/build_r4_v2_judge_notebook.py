"""Build R4's Kaggle notebook for TextGuard v2's judge on the ORIGINAL domains' VAL split.

``r4_v2_val_judge_kaggle.ipynb`` = R3's judge notebook (itself P6's notebook with the platform
plumbing replaced) with two substitutions, both asserted exact:

* the prompt module is ``src/textguard/judge_prompt_v2.py`` (pasted verbatim) instead of v1's —
  same names (``JUDGE_SYSTEM``, ``ANSWERS``, ``build_judge_prompt``, ``judge_sha``), so P6's
  scoring loop runs unchanged: Qwen2.5-1.5B-Instruct at P5's revision, fp16, scored answers,
  confidence renormalised over the three allowed answers;
* the input is ``r4v2_val_inputs.csv`` (``python -m src.textguard.export_v2_inputs``): val units
  of Economy, Traffic, SocialGood, Health_US only — **no test-split text, no held-out text** —
  plus P5's 40-prompt probe (recorded only). Its sha256 is embedded and checked.

Development data, not a test: these answers tune v2 (R4, §10g). The held-out judge run comes
after the pre-registration, from a separate export.

After changing ``judge_prompt_v2.py``, ``units.py`` or the texts:
``python -m src.textguard.export_v2_inputs``, then ``python notebooks/build_r4_v2_judge_notebook.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT))
import build_p6_judge_notebook as p6  # noqa: E402
import build_r3_notebooks as r3  # noqa: E402
from build_p2_qwen_notebook import PROMPT_SRC, code, md  # noqa: E402
from build_r2_notebooks import _edit, _src  # noqa: E402
from src.textguard.export_v2_inputs import V2_VAL_INPUTS, sha256  # noqa: E402

INPUT_NAME = V2_VAL_INPUTS.name
V2_SRC = ROOT / "src" / "textguard" / "judge_prompt_v2.py"
FILE = "r4_v2_val_judge_kaggle.ipynb"
TAG = "r4v2_val_judge_kaggle"

INTRO = f"""
# R4 — TextGuard v2's claim judge on VAL (Qwen2.5-1.5B-Instruct) — Kaggle

TextGuard v2 development (PROJECT_STATUS §1a, §10g draft). P6's judge model and scoring code,
**v2's questions** (`judge_prompt_v2.py`): every claim unit (sentence / bullet) of every **val**
text of the four original domains — real, flipped, exaggerated, a1_selective, a2_forward,
a3_magnitude — is asked `change` (did it go up / down), `outlook` (will it go up / down), and
`level` where the unit states a national baseline. Plus the units of P1's human-labelled reports
(judge validation). **No test-split and no held-out text is in the input.** P5's 40-prompt probe
runs first and is recorded.

* Input: `{INPUT_NAME}` in any attached dataset; sha256 checked.
* Output: `{TAG}_predictions.csv` + `{TAG}_run_meta.json`, appended every 20.
{r3.KAGGLE_SETTINGS.replace("r3_inputs.csv", INPUT_NAME)}"""

CONFIG = """
# 0. Run configuration (R4 v2 development, val only) — do not change
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # one GPU (cuda:0) even on T4 x2; before torch is imported
PLATFORM = "kaggle"
INPUT_SHA256 = "{sha}"
"""

PATHS = r3.FIND_INPUT.replace("r3_inputs.csv", INPUT_NAME) + f"""
OUT = f"{{OUT_DIR}}/{TAG}_predictions.csv"
META = f"{{OUT_DIR}}/{TAG}_run_meta.json\""""


def build(sha: str) -> dict:
    nb_cells = r3.build_judge(sha)["cells"]          # R3's judge notebook, then v2's swaps
    i_mod = next(i for i, c in enumerate(nb_cells)
                 if _src(c).startswith("# Generated verbatim from src/textguard/judge_prompt.py"))
    nb_cells[i_mod] = code("# Generated verbatim from src/textguard/judge_prompt_v2.py\n"
                           + V2_SRC.read_text(encoding="utf-8"))
    nb_cells[0] = md(INTRO)
    i_md = next(i for i, c in enumerate(nb_cells)
                if c["cell_type"] == "markdown" and "## Prompts" in _src(c))
    nb_cells[i_md] = md("\n## Prompts — generated from `src/models/qwen_prompt.py` (probe) and "
                        "`src/textguard/judge_prompt_v2.py` (v2 judge). Do not edit here.\n")
    i_cfg = next(i for i, c in enumerate(nb_cells) if _src(c).startswith("\n# 0.")
                 or _src(c).startswith("# 0."))
    nb_cells[i_cfg] = code(CONFIG.format(sha=sha))
    nb_cells = _edit(nb_cells, "# 3.", r3.JUDGE_PATHS, PATHS)
    nb_cells = _edit(nb_cells, "# 3.",
                     "# 3. Paths and inputs (one CSV for all three R3 notebooks; judge + probe rows used here)",
                     "# 3. Paths and inputs (R4 v2 val judge: judge + probe rows)")
    return r3._notebook(nb_cells)


def main() -> None:
    if not V2_VAL_INPUTS.exists():
        raise FileNotFoundError("run python -m src.textguard.export_v2_inputs first")
    nb = build(sha256())
    path = ROOT / "notebooks" / FILE
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
