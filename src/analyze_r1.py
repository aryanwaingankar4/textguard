"""R1 — LLM vs fusion: pre-training diagnostics (and, later, the analysis against §10d).

Stage implemented so far — **before any R1 head is trained**:

* ``embed-finbert``: FinBERT (``ProsusAI/finbert``, pinned revision) embeds every prompt the R1
  heads need — the same set the Qwen export holds (``qwen_embeddings.window_texts`` +
  ``condition_texts``) — into the shared cache, on the CPU. Truncation (512 tokens) counted.
* ``mechanism``: the pre-registered **val** mechanism check (§10d). For each encoder, P5's
  ``embedding_similarity`` on the val split: how far the flip moves the head's input, relative
  to swapping in a different report (``rel_distance``), in the standardised space that
  encoder's head would see. **Rule fixed in §10d:** an encoder is *direction-aware* if its
  pooled val mean ``rel_distance`` (flipped) is ≥ 2× BERT's on the same windows. Only selects
  between readings R1-b and R1-c; it never touches a head or a forecast.
  Qwen is included once its Colab embeddings are imported (``qwen_embeddings import``);
  until then it is reported as pending, never computed from a partial cache.

Why val: the rule was fixed before any number existed, and val is where a pre-training
diagnostic belongs; the test-split table is reported later, beside P5's BERT table.

Run: ``python -m src.analyze_r1 embed-finbert``, then ``python -m src.analyze_r1 mechanism``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.text_encoder import (FINBERT_NAME, MAX_TOKENS, MODEL_NAME, QWEN_NAME,
                                     BertTextEncoder, cache_dir_for)

RESULTS = __import__("pathlib").Path(__file__).resolve().parents[1] / "results"
ENCODERS = {"BERT": MODEL_NAME, "FinBERT": FINBERT_NAME, "Qwen": QWEN_NAME}
AWARE_FACTOR = 2.0  # §10d: direction-aware ⇔ pooled val mean rel_distance ≥ 2× BERT's


def r1_texts() -> list[str]:
    """Every distinct non-blank text the R1 heads read (train/val/test windows + conditions)."""
    from src.models.late_fusion import _clean
    from src.models.qwen_embeddings import condition_texts, window_texts
    t = pd.concat([window_texts(), condition_texts()], ignore_index=True)
    return list(dict.fromkeys(x for x in (_clean(v) for v in t.text) if x is not None))


def embed_finbert() -> dict:
    enc = BertTextEncoder(FINBERT_NAME)
    texts = r1_texts()
    try:
        n_tok = enc.token_counts(texts)
        enc.encode(texts)
    finally:
        enc.unload()
    return dict(encoder=FINBERT_NAME, revision=enc.revision, texts=len(texts),
                median_tokens=int(np.median(n_tok)), max_tokens=int(n_tok.max()),
                truncated=int((n_tok > MAX_TOKENS).sum()))


def qwen_cached() -> bool:
    """All R1 prompts present in the Qwen cache (i.e. the Colab import has run)."""
    from src.models.qwen_embeddings import INPUTS, load_inputs
    if not INPUTS.exists():
        return False
    e = load_inputs()
    e = e[e.row_type == "embed"]
    d = cache_dir_for(QWEN_NAME)
    return all((d / f"{s}.npy").exists() for s in e.prompt_sha)


def mechanism(split: str = "val") -> pd.DataFrame:
    """The §10d mechanism table: one block per available encoder + the direction-aware rule."""
    from src.analyze_p5 import embedding_similarity
    parts = []
    for label, name in ENCODERS.items():
        if name == QWEN_NAME and not qwen_cached():
            print("Qwen: embeddings not imported yet -> pending (run the Colab notebook, §7c)")
            continue
        parts.append(embedding_similarity(split, name).assign(encoder=label, split=split))
    df = pd.concat(parts, ignore_index=True)
    key = (df.attack == "flipped") & (df.pair == "real vs flipped") & (df.domain == "pooled")
    bert = float(df[key & (df.encoder == "BERT")].rel_distance_mean.iloc[0])
    ratio = df[key].set_index("encoder").rel_distance_mean / bert
    df["ratio_to_bert"] = np.where(key, df.encoder.map(ratio), np.nan)
    df["direction_aware"] = np.where(key, df.ratio_to_bert >= AWARE_FACTOR, None)
    cols = ["encoder", "split", "attack", "pair", "domain", "n"]
    return df[cols + [c for c in df.columns if c not in cols]]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["embed-finbert", "mechanism"])
    a = ap.parse_args()
    if a.stage == "embed-finbert":
        print(embed_finbert())
        return
    df = mechanism("val")
    out = RESULTS / "summary_r1_mechanism_val.csv"
    df.to_csv(out, index=False)
    show = df[(df.domain == "pooled")][["encoder", "attack", "pair", "n", "cos_raw_mean",
                                        "cos_std_mean", "rel_distance_mean",
                                        "rel_distance_median", "ratio_to_bert",
                                        "direction_aware"]]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(show.to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
