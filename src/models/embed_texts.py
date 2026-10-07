"""Pre-compute BERT embeddings for every text in the selected domains, and report truncation.

Why a separate step: embedding is the only expensive part of Models 3/4 on a CPU. Doing it
once up front fills the cache (``text_encoder.CACHE_DIR``) so training never runs BERT, and
gives a single place to report how many texts exceed BERT's 512-token limit.

Counts are per *period* text (the reports of one period joined by the loader), wrapped in
MM-TSFlib's prompt — exactly what the models see. Only periods inside the split range (train,
val, test) are counted.

Run: ``python -m src.models.embed_texts``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.splits import load_splits
from src.models.text_encoder import MAX_TOKENS, BertTextEncoder
from src.models.train_baselines import DOMAINS


def domain_texts(domain: str) -> pd.DataFrame:
    s = load_splits(domain)
    parts = [df.assign(split=name) for name, df in
             (("train", s.train), ("val", s.val), ("test", s.test))]
    full = pd.concat(parts, ignore_index=True)
    return full[full["has_text"]][["start_date", "split", "text"]]


def main() -> None:
    enc = BertTextEncoder()
    rows = []
    for domain in DOMAINS:
        df = domain_texts(domain)
        texts = df["text"].tolist()
        n_tok = enc.token_counts(texts)
        enc.encode(texts + [None])  # fills the cache; None = the null prompt
        rows.append({"domain": domain, "texts": len(texts),
                     "median_tokens": int(np.median(n_tok)), "max_tokens": int(n_tok.max()),
                     "truncated": int((n_tok > MAX_TOKENS).sum()),
                     "truncated_pct": round(100 * float((n_tok > MAX_TOKENS).mean()), 1),
                     "truncated_test": int((n_tok[df["split"].to_numpy() == "test"]
                                            > MAX_TOKENS).sum())})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print(f"total: {out.truncated.sum()} / {out.texts.sum()} texts exceed {MAX_TOKENS} tokens")


if __name__ == "__main__":
    main()
