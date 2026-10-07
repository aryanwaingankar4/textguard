"""Frozen BERT text encoder with an on-disk embedding cache (Models 3 and 4).

Why frozen + cached: BERT is never trained here (MM-TSFlib freezes it too), so a text's
embedding is a pure function of the text. Computing it once and caching it means

* training only touches the small fusion head — fast on a laptop CPU;
* checkpoints hold the head and the time-series network, **never BERT's 440 MB of weights**;
* P3's attacked texts (flipped, exaggerated, …) are embedded once and reused by every model,
  seed and defence run.

Representation (decided 2026-10-07, PROJECT_STATUS §7a): **all 12 layers** of
``bert-base-uncased``, last hidden state **mean-pooled over real tokens** (padding masked).
MM-TSFlib's shipped script instead uses only BERT's word-embedding table
(``use_fullmodel=0``) — a bag of words that cannot tell "rose" from "did not rise". For a
study of *meaning*-inverting attacks the encoder must read context.

Each text is wrapped in MM-TSFlib's prompt template before encoding, as MM-TSFlib does.

Truncation: BERT reads at most 512 tokens. Longer prompts are cut (``truncation=True``) and
counted; ``token_stats`` reports how many texts that affects.

Cache layout: ``cache/text_emb/<model>/<sha256 of the prompt>.npy`` (float32, shape (768,)),
gitignored. Keyed by the exact prompt string, so a changed template or text is a cache miss,
never a stale hit. BERT is loaded lazily, only on a miss.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

MODEL_NAME = "google-bert/bert-base-uncased"  # the id MM-TSFlib loads
MAX_TOKENS = 512
EMB_DIM = 768
ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "cache" / "text_emb"

# MM-TSFlib's prompt (exp_long_term_forecasting.py), verbatim — including its odd tags
PROMPT = ("<|start_prompt|Make predictions about the future based on the following "
          "information: {text}<|<end_prompt>|>")


def make_prompt(text: str | None) -> str:
    """The string BERT sees. ``None`` → the template with nothing inside: the *null prompt*.

    Why the null prompt rather than a zero vector: it is a real point in BERT's embedding
    space ("make predictions… : <nothing>"), so the fusion head receives an ordinary input
    for missing text, and it is the same input for "no report existed" (sparse domains) and
    "text removed" (the ``empty`` condition / TextGuard's A).
    """
    return PROMPT.format(text="" if text is None else text)


class BertTextEncoder:
    """``encode(texts) -> (N, 768)`` float32, cached on disk. ``None`` encodes the null prompt."""

    def __init__(self, model_name: str = MODEL_NAME, cache_dir: Path | str = CACHE_DIR,
                 batch_size: int = 16):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) / model_name.replace("/", "__")
        self.batch_size = batch_size
        self._tok = None
        self._bert = None

    # --- lazy BERT -------------------------------------------------------------------

    def _load(self):
        if self._bert is None:
            from transformers import AutoModel, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._bert = AutoModel.from_pretrained(self.model_name).eval()
            for p in self._bert.parameters():
                p.requires_grad = False
        return self._tok, self._bert

    def unload(self) -> None:
        """Release BERT (~440 MB). Training reads embeddings from the cache only, so BERT is
        loaded only on a cache miss and should be dropped before the next training unit."""
        if self._bert is not None:
            self._bert = self._tok = None
            import gc
            gc.collect()

    @property
    def is_loaded(self) -> bool:
        return self._bert is not None

    @property
    def tokenizer(self):
        return self._load()[0]

    # --- cache -----------------------------------------------------------------------

    def _path(self, prompt: str) -> Path:
        return self.cache_dir / f"{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}.npy"

    def encode(self, texts) -> np.ndarray:
        prompts = [make_prompt(t) for t in texts]
        out = np.empty((len(prompts), EMB_DIM), dtype=np.float32)
        missing: dict[str, list[int]] = {}
        for i, p in enumerate(prompts):
            path = self._path(p)
            if path.exists():
                out[i] = np.load(path)
            else:
                missing.setdefault(p, []).append(i)
        if missing:
            todo = list(missing)
            embs = self._embed(todo)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for p, e in zip(todo, embs):
                np.save(self._path(p), e)
                out[missing[p]] = e
        return out

    def _embed(self, prompts: list[str]) -> np.ndarray:
        import torch
        tok, bert = self._load()
        chunks = []
        with torch.no_grad():
            for i in range(0, len(prompts), self.batch_size):
                batch = tok(prompts[i:i + self.batch_size], padding=True, truncation=True,
                            max_length=MAX_TOKENS, return_tensors="pt")
                h = bert(**batch).last_hidden_state                   # (B, T, 768)
                mask = batch["attention_mask"].unsqueeze(-1).float()  # padding excluded
                chunks.append(((h * mask).sum(1) / mask.sum(1)).numpy().astype(np.float32))
        return np.concatenate(chunks)

    # --- diagnostics -----------------------------------------------------------------

    def token_counts(self, texts) -> np.ndarray:
        """Untruncated token count of each prompt (incl. [CLS]/[SEP])."""
        tok = self.tokenizer
        return np.array([len(tok(make_prompt(t), truncation=False)["input_ids"]) for t in texts])
