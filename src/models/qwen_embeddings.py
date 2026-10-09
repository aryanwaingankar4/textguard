"""Qwen2.5-1.5B text embeddings for R1's late-fusion heads: export → Colab → import (R1).

Why: R1 (PROJECT_STATUS §10d) asks whether Model 5's vulnerability comes from the LLM's
*understanding* of the text or from *early fusion*. It keeps Models 3/4 exactly and swaps
frozen BERT for Model 5's own LLM as a frozen text encoder. Qwen needs a GPU and Colab cannot
see this laptop (issue #5), so — as for Model 5 and the P6 judge — the texts are exported to one
CSV, embedded on Colab (``notebooks/r1_qwen_embeddings.ipynb``), and imported back into the
same on-disk cache BERT uses (``text_encoder.CACHE_DIR``), where ``CachedEncoder`` serves them.

**What is exported** (``row_type = "embed"``): one row per distinct *prompt* — the text wrapped
in MM-TSFlib's template exactly as the heads encode it (``make_prompt(_clean(text))``) — for

* every report of every **train / val / test window** of the four domains (the heads train on
  train, early-stop on val, and the embedding scaler is fit on train texts), and
* every non-empty val + test text of **every condition** in ``texts/`` (real, shuffled,
  cross_domain, time_shifted, flipped, exaggerated) — what P5/R1 evaluate.

plus the **null prompt** (the template with no text, ``make_prompt(None)``): the heads never
use it (their text branch is absent without a report), but P5's ``embedding_similarity`` — the
R1 mechanism check — encodes it for windows whose comparison text is missing, as it did for BERT.

``prompt_sha`` = SHA-256 of the prompt = the cache file name. Because Models 3/4 already
embedded exactly these prompts with BERT, ``export`` checks that every one is a BERT-cache
hit: a miss would mean the exported set differs from what the heads read.

Plus the **reproducibility probe** (``row_type = "probe"``): the 40 Model 5 forecasting prompts
P5 probed with, with P2.4's raw answers, so the run is certified to be P5's setup class.

**Import** checks, before writing a single cache file: the Colab file holds exactly the
exported fingerprints (none missing, none extra, no duplicates), 1536-d, all finite; the meta
file names Model 5's revision. An existing cache file that differs raises — two stacks must
never be mixed silently.

Run: ``python -m src.models.qwen_embeddings export`` → ``results/qwen/r1_embed_inputs.csv``
(gitignored: raw text); after Colab: ``python -m src.models.qwen_embeddings import``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.text_encoder import (EMB_DIMS, MODEL_NAME, QWEN_NAME, QWEN_REVISION,
                                     cache_dir_for, make_prompt, prompt_sha)

ROOT = Path(__file__).resolve().parents[2]
QWEN_DIR = ROOT / "results" / "qwen"
INPUTS = QWEN_DIR / "r1_embed_inputs.csv"
EMBEDDINGS = QWEN_DIR / "r1_qwen_embeddings.npz"
META = QWEN_DIR / "r1_qwen_embed_meta.json"
PROBE_COLS = ["split", "sample_id", "condition", "freq", "description", "unit", "lookback",
              "expected_raw"]


def window_texts(domains=None) -> pd.DataFrame:
    """(domain, text, source) for the report of every train / val / test window."""
    from src.data.splits import load_splits
    from src.data.windows import lookback_for, split_windows
    from src.models.train_baselines import DOMAINS
    rows = []
    for dom in domains or DOMAINS:
        for split, w in split_windows(load_splits(dom), lookback_for(dom), horizon=1).items():
            rows += [(dom, t, f"window:{split}") for t in (w.text if w.text is not None else [])]
    return pd.DataFrame(rows, columns=["domain", "text", "source"])


def condition_texts() -> pd.DataFrame:
    """(domain, text, source) for every val + test text of every condition in ``texts/``."""
    from src.attacks.common import CONDITIONS, load
    parts = [pd.DataFrame({"domain": t.domain, "text": t.text, "source": cond})
             for cond in CONDITIONS for t in [load(cond)]]
    return pd.concat(parts, ignore_index=True)


def build(texts: pd.DataFrame, probes: pd.DataFrame) -> pd.DataFrame:
    """One ``embed`` row per distinct prompt + the probe rows. Blank / missing text is dropped
    (the heads' text branch is absent then — it is never embedded)."""
    from src.models.late_fusion import _clean
    t = texts.assign(text=[_clean(x) for x in texts.text]).dropna(subset=["text"])
    t = t.assign(prompt=[make_prompt(x) for x in t.text])
    t = pd.concat([t, pd.DataFrame([dict(domain="", text="", source="null",
                                         prompt=make_prompt(None))])], ignore_index=True)
    g = t.groupby("prompt", sort=False).agg(
        domains=("domain", lambda s: ";".join(dict.fromkeys(s))),
        sources=("source", lambda s: ";".join(dict.fromkeys(s)))).reset_index()
    embed = g.assign(row_type="embed", prompt_sha=[prompt_sha(p) for p in g.prompt],
                     n_chars=g.prompt.str.len())
    if embed.prompt_sha.duplicated().any():
        raise ValueError("two prompts share a fingerprint")
    probe = probes[["domain", "text"] + PROBE_COLS].assign(row_type="probe",
                                                            domains=probes.domain)
    cols = ["row_type", "domains", "prompt_sha", "prompt", "sources", "n_chars", "text"] + PROBE_COLS
    return pd.concat([embed, probe], ignore_index=True).reindex(columns=cols).fillna("")


def bert_cache_misses(df: pd.DataFrame) -> int:
    """How many exported prompts Models 3/4 never embedded with BERT (must be 0)."""
    d = cache_dir_for(MODEL_NAME)
    e = df[df.row_type == "embed"]
    return int(sum(not (d / f"{s}.npy").exists() for s in e.prompt_sha))


def export(path: Path = INPUTS) -> pd.DataFrame:
    from src.models.export_qwen_inputs import INPUTS_P5
    p5 = pd.read_csv(INPUTS_P5, keep_default_na=False, dtype={"sample_id": str})
    probes = p5[p5.probe.astype(str) == "True"]
    if len(probes) != 40:
        raise ValueError(f"expected P5's 40 probe rows, found {len(probes)}")
    df = build(pd.concat([window_texts(), condition_texts()], ignore_index=True), probes)
    miss = bert_cache_misses(df)
    if miss:
        raise ValueError(f"{miss} exported prompts are not in the BERT cache: the exported set "
                         "differs from what Models 3/4 read")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_inputs(path: Path = INPUTS) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, dtype=str)


def import_embeddings(npz: Path = EMBEDDINGS, meta: Path = META, inputs: Path = INPUTS,
                      cache_dir: Path | None = None, dim: int = EMB_DIMS[QWEN_NAME]) -> dict:
    """Verify the Colab file against the export, then write one cache file per prompt."""
    df = load_inputs(inputs)
    e = df[df.row_type == "embed"]
    bad = [s for s, p in zip(e.prompt_sha, e.prompt) if prompt_sha(p) != s]
    if bad:
        raise ValueError(f"{len(bad)} exported fingerprints do not match their prompt")
    m = json.loads(Path(meta).read_text(encoding="utf-8"))
    if m.get("model_id") != QWEN_NAME or m.get("revision") != QWEN_REVISION:
        raise ValueError(f"embeddings from {m.get('model_id')}@{m.get('revision')}, "
                         f"expected {QWEN_NAME}@{QWEN_REVISION} (Model 5's revision)")
    with np.load(npz, allow_pickle=False) as f:
        z = {k: f[k] for k in f.files}
    shas, emb = [str(s) for s in z["sha"]], z["emb"]
    if len(set(shas)) != len(shas):
        raise ValueError("duplicate fingerprints in the Colab file")
    want, got = set(e.prompt_sha), set(shas)
    if want != got:
        raise ValueError(f"Colab file does not match the export: {len(want - got)} missing, "
                         f"{len(got - want)} extra")
    if emb.shape != (len(shas), dim) or emb.dtype != np.float32:
        raise ValueError(f"expected float32 ({len(shas)}, {dim}), got {emb.dtype} {emb.shape}")
    if not np.isfinite(emb).all():
        raise ValueError("non-finite embedding values")
    out = Path(cache_dir) if cache_dir is not None else cache_dir_for(QWEN_NAME)
    out.mkdir(parents=True, exist_ok=True)
    written = same = 0
    for s, v in zip(shas, emb):
        f = out / f"{s}.npy"
        if f.exists():
            if not np.array_equal(np.load(f), v):
                raise ValueError(f"{f.name} already cached with different values — another "
                                 "run/setup; delete the Qwen cache folder deliberately first")
            same += 1
            continue
        np.save(f, v)
        written += 1
    tok = z.get("n_tokens")
    return dict(prompts=len(shas), written=written, already_identical=same, dim=dim,
                probe_ok=m.get("probe_ok"), setup=m.get("setup"),
                tokens_median=None if tok is None else int(np.median(tok)),
                tokens_max=None if tok is None else int(tok.max()), cache=str(out))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["export", "import"])
    a = ap.parse_args()
    if a.action == "export":
        df = export()
        e = df[df.row_type == "embed"]
        src = e.assign(source=e.sources.str.split(";")).explode("source")
        print(src.groupby("source").size().rename("prompts containing").to_string())
        print(f"{len(e)} distinct prompts (median {int(e.n_chars.median())} chars, max "
              f"{int(e.n_chars.max())}) + {int((df.row_type == 'probe').sum())} probe prompts; "
              f"BERT-cache misses: 0 -> {INPUTS} ({INPUTS.stat().st_size / 1024:.0f} KB)")
    else:
        print(json.dumps(import_embeddings(), indent=1))


if __name__ == "__main__":
    main()
