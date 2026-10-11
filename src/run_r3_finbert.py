"""R3 — R1's FinBERT heads (M3-F, M4-F) on the adaptive attacks A1–A3 (PROJECT_STATUS §10f).

§10f: "FinBERT heads: R1's checkpoints, R1's runner (CPU, deterministic) on the new texts;
real / none / flipped / exaggerated from R1's existing predictions (identical code path)."
So this is the P5a runner (``run_experiments.run``) — the same one ``run_r1`` calls — pointed
at the three R3 conditions and at **its own** unit folder and predictions table. R1's files
(``results/r1_units/``, ``results/r1_predictions.csv``) are only read, never written; ``main``
fingerprints them before and after and refuses to finish if a byte changed.

Steps:

1. ``embed`` — FinBERT (R1's pinned revision) embeds every distinct A1/A2/A3 text (val + test)
   into the same content-addressed cache R1 used. A1's texts are P3 ``flipped`` or ``real``
   texts, so they are cache hits; A2 / A3 are new. Nothing already cached is recomputed.
2. ``run`` — inference only, 3 seeds × val + test, each unit fingerprinted (checkpoints + texts).
3. Integrity checks (not results): every head's ``EmbeddingScaler`` still equals its
   train-text statistics; **A1 = a selection, not new text**, so its predictions must equal
   R1's ``flipped`` predictions on attacked windows and R1's ``real`` ones elsewhere, exactly.

Outputs: ``results/r3_finbert_units/`` and ``results/r3_finbert_predictions.csv`` (gitignored,
regenerable in ~minutes), MLflow experiment ``r3_finbert``.

Run: ``python -m src.run_r3_finbert`` (``--no-mlflow``).
"""

from __future__ import annotations

import argparse
import hashlib
import sys

import numpy as np
import pandas as pd

from src.attacks import common
from src.attacks.adaptive import R3_CONDITIONS
from src.run_experiments import RESULTS, check_embedding_scalers, run

FINBERT_HEADS = ("DLinear+FinBERT", "PatchTST+FinBERT")
UNIT_DIR = RESULTS / "r3_finbert_units"
PREDICTIONS = RESULTS / "r3_finbert_predictions.csv"
R1_UNITS = RESULTS / "r1_units"
R1_PREDICTIONS = RESULTS / "r1_predictions.csv"
KEYS = ["model", "domain", "split", "seed", "sample_id"]


def r1_fingerprint() -> str:
    """sha256 over R1's predictions table and every R1 unit file (names + bytes)."""
    h = hashlib.sha256(R1_PREDICTIONS.read_bytes())
    for p in sorted(R1_UNITS.iterdir()):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def r3_texts() -> list[str]:
    """Every distinct non-blank A1/A2/A3 text (val + test), cleaned as the heads read it."""
    from src.models.late_fusion import _clean
    t = pd.concat([common.load(c).text for c in R3_CONDITIONS], ignore_index=True)
    return list(dict.fromkeys(x for x in (_clean(v) for v in t) if x is not None))


def embed() -> dict:
    from src.models.text_encoder import FINBERT_NAME, MAX_TOKENS, BertTextEncoder, make_prompt
    enc = BertTextEncoder(FINBERT_NAME)
    texts = r3_texts()
    cached = sum(enc._path(make_prompt(t)).exists() for t in texts)
    try:
        n_tok = enc.token_counts(texts)
        enc.encode(texts)
    finally:
        enc.unload()
    return dict(encoder=FINBERT_NAME, revision=enc.revision, texts=len(texts),
                already_cached=cached, newly_embedded=len(texts) - cached,
                truncated=int((n_tok > MAX_TOKENS).sum()))


def check_a1_is_a_selection(preds: pd.DataFrame) -> pd.DataFrame:
    """A1 rows must equal R1's flipped (attacked windows) / real (the rest) predictions exactly."""
    r1 = pd.read_csv(R1_PREDICTIONS, dtype={"sample_id": str})
    r1 = r1[r1.model.isin(FINBERT_HEADS)]
    a1 = preds[preds.condition == R3_CONDITIONS[0]]
    ref = {c: r1[r1.condition == c].set_index(KEYS).y_pred for c in ("flipped", "real")}
    rows = []
    for attacked, src in ((True, "flipped"), (False, "real")):
        part = a1[a1.attacked == attacked].set_index(KEYS).y_pred
        other = ref[src].reindex(part.index)
        same = np.isclose(part.to_numpy(), other.to_numpy(), rtol=0, atol=0, equal_nan=True)
        rows.append(dict(a1_rows="attacked" if attacked else "not attacked", compared_to=f"R1 {src}",
                         n=len(part), identical=int(same.sum())))
    out = pd.DataFrame(rows)
    if (out.n != out.identical).any():
        raise AssertionError(f"A1 predictions differ from R1's:\n{out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    fp0 = r1_fingerprint()
    print("R1 fingerprint before:", fp0)
    print("embedding:", embed())
    preds = run(models=FINBERT_HEADS, conditions=R3_CONDITIONS, use_mlflow=not a.no_mlflow,
                unit_dir=UNIT_DIR, predictions=PREDICTIONS, experiment="r3_finbert")
    scal = check_embedding_scalers(models=FINBERT_HEADS)
    print(f"embedding scalers: {len(scal)} checkpoints match their train-text statistics "
          f"(max |diff| {scal.max_abs_diff.max():.1e})")
    print("A1 = selection of R1's own predictions:\n" + check_a1_is_a_selection(preds).to_string(index=False))
    print(preds.groupby(["model", "condition", "split"]).size().unstack().to_string())
    print(f"NaN predictions: {int(preds.y_pred.isna().sum())}")
    fp1 = r1_fingerprint()
    print("R1 fingerprint after: ", fp1)
    if fp0 != fp1:
        raise AssertionError("R1's outputs changed")
    print("R1's outputs are byte-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
