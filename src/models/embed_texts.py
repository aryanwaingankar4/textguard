"""Pre-compute BERT embeddings for every text in the selected domains, and report truncation.

Why a separate step: embedding is the only expensive part of Models 3/4 on a CPU. Doing it
once up front fills the cache (``text_encoder.CACHE_DIR``) so training never runs BERT, and
gives a single place to report how many texts exceed BERT's 512-token limit.

Counts are per *period* text (the reports of one period joined by the loader), wrapped in
MM-TSFlib's prompt — exactly what the models see. Only periods inside the split range (train,
val, test) are counted.

Run: ``python -m src.models.embed_texts`` (real texts, all splits) or
``python -m src.models.embed_texts --conditions`` (P3's altered conditions, val + test — P5a).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.splits import load_splits
from src.models.text_encoder import MAX_TOKENS, BertTextEncoder, make_prompt
from src.models.train_baselines import DOMAINS

# P3 conditions whose text differs from the real report (P5a embeds them; ``empty`` has none)
ALTERED_CONDITIONS = ("shuffled", "cross_domain", "time_shifted", "flipped", "exaggerated")


def domain_texts(domain: str) -> pd.DataFrame:
    s = load_splits(domain)
    parts = [df.assign(split=name) for name, df in
             (("train", s.train), ("val", s.val), ("test", s.test))]
    full = pd.concat(parts, ignore_index=True)
    return full[full["has_text"]][["start_date", "split", "text"]]


def embed_conditions(conditions=ALTERED_CONDITIONS, enc: BertTextEncoder | None = None
                     ) -> pd.DataFrame:
    """Embed every val + test text of the altered conditions (P5a) into the same cache.

    Why this is all that is needed to standardise them correctly: the cache holds **raw**
    BERT vectors. The per-dimension z-score is the ``EmbeddingScaler`` stored inside each
    Model 3/4 checkpoint, fitted in P2.3 on *train* texts only. Loading a checkpoint applies
    those frozen statistics to any text, so attacked text is standardised exactly like real
    text and can never refit the scaler (which would leak the attack into the model).

    Also reports truncation per condition — exaggerated rewrites are slightly longer.
    BERT is unloaded before returning (RAM is tight on this machine).
    """
    from src.attacks.common import load
    enc = enc or BertTextEncoder()
    rows = []
    try:
        for cond in conditions:
            df = load(cond)
            df = df[df["text"].str.strip().str.len() > 0]
            texts = df["text"].tolist()
            before = sum(enc._path(make_prompt(t)).exists() for t in set(texts))
            n_tok = enc.token_counts(texts)
            enc.encode(texts)
            rows.append({"condition": cond, "texts": len(texts), "unique": len(set(texts)),
                         "already_cached": before, "median_tokens": int(np.median(n_tok)),
                         "max_tokens": int(n_tok.max()),
                         "truncated": int((n_tok > MAX_TOKENS).sum())})
            print(rows[-1], flush=True)
    finally:
        enc.unload()
    return pd.DataFrame(rows)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--conditions", action="store_true",
                    help="embed the P3 altered-condition texts (val + test) instead")
    if ap.parse_args().conditions:
        out = embed_conditions()
        print(out.to_string(index=False))
        return
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
