"""R1 (LLM vs fusion) tooling: encoders, Qwen embedding export / Colab notebook / import.

No GPU: the notebook's own cells are executed with a fake ``embed`` / ``generate``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.qwen_embeddings import build, import_embeddings
from src.models.text_encoder import (FINBERT_NAME, FINBERT_REVISION, MODEL_NAME, QWEN_NAME,
                                     QWEN_REVISION, BertTextEncoder, CachedEncoder, get_encoder,
                                     make_prompt, prompt_sha)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "r1_qwen_embeddings.ipynb"
DIM = 1536


# --- encoders ----------------------------------------------------------------------------------

def test_encoder_factory_pins_revisions_and_dims(tmp_path):
    bert = get_encoder(MODEL_NAME, cache_dir=tmp_path)
    fin = get_encoder(FINBERT_NAME, cache_dir=tmp_path)
    qwen = get_encoder(QWEN_NAME, cache_dir=tmp_path)
    assert type(bert) is BertTextEncoder and bert.dim == 768 and bert.revision is None
    assert type(fin) is BertTextEncoder and fin.dim == 768 and fin.revision == FINBERT_REVISION
    assert type(qwen) is CachedEncoder and qwen.dim == DIM and qwen.revision == QWEN_REVISION
    assert bert.cache_dir != fin.cache_dir != qwen.cache_dir          # never share a cache


def test_cached_encoder_serves_hits_and_raises_on_a_miss(tmp_path):
    enc = CachedEncoder(QWEN_NAME, cache_dir=tmp_path)
    v = np.arange(DIM, dtype=np.float32)
    enc.cache_dir.mkdir(parents=True)
    np.save(enc.cache_dir / f"{prompt_sha(make_prompt('rose'))}.npy", v)
    assert np.array_equal(enc.encode(["rose", "rose"]), np.stack([v, v]))
    with pytest.raises(KeyError, match="missing from the"):
        enc.encode(["fell"])
    assert not enc.is_loaded                                          # never loads a model


# --- export -------------------------------------------------------------------------------------

def _probes(n=4):
    return pd.DataFrame([dict(domain="Economy", text="", split="val", sample_id=f"2015-0{i}-01",
                              condition="none", freq="monthly", description="series",
                              unit="million US dollars", lookback="[1.0, 2.0]", expected_raw="3")
                         for i in range(1, n + 1)])


TEXTS = pd.DataFrame([("Economy", "The deficit increased $2.0 billion.", "window:train"),
                      ("Economy", "The deficit increased $2.0 billion.", "real"),
                      ("Traffic", "The deficit increased $2.0 billion.", "cross_domain"),
                      ("Economy", "The deficit decreased $2.0 billion.", "flipped"),
                      ("Economy", "   ", "empty"), ("Economy", None, "real")],
                     columns=["domain", "text", "source"])


def test_export_one_row_per_distinct_prompt_exactly_as_the_heads_encode_it():
    df = build(TEXTS, _probes())
    e = df[df.row_type == "embed"]
    assert len(e) == 3                          # 2 distinct texts (blank / None dropped) + null prompt
    assert e.iloc[-1].prompt == make_prompt(None) and e.iloc[-1].sources == "null"
    first = e.iloc[0]
    assert first.prompt == make_prompt("The deficit increased $2.0 billion.")
    assert first.prompt_sha == prompt_sha(first.prompt)
    assert first.sources == "window:train;real;cross_domain" and first.domains == "Economy;Traffic"
    assert (df.row_type == "probe").sum() == 4


# --- notebook: in sync, pinned, and its logic with a fake model --------------------------------

def _cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]


def test_notebook_embeds_prompt_module_and_pins_p5_revision():
    src = (ROOT / "src/models/qwen_prompt.py").read_text(encoding="utf-8").strip("\n")
    assert any(s.endswith(src) and t == "code" for s, t in _cells()), \
        "notebook is stale: run python notebooks/build_r1_qwen_embeddings_notebook.py"
    p5 = json.loads((ROOT / "results" / "qwen" / "p5_qwen_run_meta.json").read_text())
    assert p5["revision"] == QWEN_REVISION
    load = next(s for s, _ in _cells() if "def embed" in s)
    assert f'REVISION = "{p5["revision"]}"' in load and "revision=REVISION" in load
    assert "last_hidden_state" in load and ".mean(0)" in load and "h.float()" in load
    assert "apply_chat_template" not in load.split("def embed")[1]    # no chat template
    assert "truncation=" not in load.split("def embed")[1].split("meta = dict")[0]


def test_notebook_setup_string_matches_setup_id():
    from src.models.qwen_forecaster import setup_id
    load = next(s for s, _ in _cells() if "def embed" in s)
    expr = load[load.index("SETUP = ("):load.index('meta["setup"]')]
    meta = dict(model_id=QWEN_NAME, revision="abc", gpu="Tesla T4", torch="2.11.0+cu130",
                transformers="5.18.0")
    ns = {"meta": meta, "MODEL_ID": meta["model_id"], "REVISION": "abc"}
    exec(expr, ns)
    assert ns["SETUP"] == setup_id(meta)


def _fake_embed(prompt):
    rng = np.random.default_rng(int(prompt_sha(prompt)[:8], 16))
    return rng.standard_normal(DIM).astype(np.float32), len(prompt.split())


def _run(tmp_path, generate=lambda p, **kw: "3", setup="SETUP-A"):
    cells = [s for s, t in _cells() if t == "code"]
    pick = lambda key: next(c for c in cells if key in c)  # noqa: E731
    meta = dict(model_id=QWEN_NAME, revision=QWEN_REVISION, setup=setup)
    ns = {"generate": generate, "embed": _fake_embed, "SETUP": setup, "P5_SETUP": "SETUP-A",
          "meta": meta}
    exec(pick("# 3. Paths and inputs").replace("/content", tmp_path.as_posix()), ns)
    exec(pick("# Generated verbatim from src/models/qwen_prompt.py"), ns)
    exec(pick("# 5. Reproducibility probe").replace("len(probes) == 40", "len(probes) == 4"), ns)
    for key in ("# 6. Main loop", "# 7. Checks"):
        exec(pick(key), ns)
    return ns


def _inputs(tmp_path):
    df = build(TEXTS, _probes())
    df.to_csv(tmp_path / "r1_embed_inputs.csv", index=False)
    return df


def test_notebook_embeds_every_prompt_and_import_fills_the_cache(tmp_path):
    inp = _inputs(tmp_path)
    ns = _run(tmp_path)
    assert ns["PROBE_OK"] and ns["DETERMINISTIC"]
    cache = tmp_path / "cache"
    out = import_embeddings(tmp_path / "r1_qwen_embeddings.npz",
                            tmp_path / "r1_qwen_embed_meta.json",
                            tmp_path / "r1_embed_inputs.csv", cache_dir=cache)
    assert out["written"] == len(inp[inp.row_type == "embed"]) == 3 and out["probe_ok"]
    enc = CachedEncoder(QWEN_NAME, cache_dir=tmp_path)
    enc.cache_dir = cache
    got = enc.encode(["The deficit decreased $2.0 billion."])[0]
    assert np.array_equal(got, _fake_embed(make_prompt("The deficit decreased $2.0 billion."))[0])
    again = import_embeddings(tmp_path / "r1_qwen_embeddings.npz",
                              tmp_path / "r1_qwen_embed_meta.json",
                              tmp_path / "r1_embed_inputs.csv", cache_dir=cache)
    assert again["written"] == 0 and again["already_identical"] == 3  # idempotent


def test_probe_mismatch_is_recorded_not_hidden(tmp_path):
    _inputs(tmp_path)
    ns = _run(tmp_path, generate=lambda p, **kw: "999")
    assert not ns["PROBE_OK"]
    meta = json.loads((tmp_path / "r1_qwen_embed_meta.json").read_text())
    assert meta["probe_ok"] is False and meta["probe_mismatches"] == 4


def test_notebook_resumes_after_a_disconnect(tmp_path):
    _inputs(tmp_path)
    _run(tmp_path)
    npz = tmp_path / "r1_qwen_embeddings.npz"
    with np.load(npz) as z:
        z = dict(z)
    np.savez(npz, sha=z["sha"][:1], emb=z["emb"][:1], n_tokens=z["n_tokens"][:1])
    ns = _run(tmp_path)
    with np.load(npz) as z:
        assert len(z["sha"]) == 3 and ns["DETERMINISTIC"]


def test_import_refuses_mismatched_files(tmp_path):
    _inputs(tmp_path)
    _run(tmp_path)
    npz, meta, inp = (tmp_path / "r1_qwen_embeddings.npz", tmp_path / "r1_qwen_embed_meta.json",
                      tmp_path / "r1_embed_inputs.csv")
    with np.load(npz) as z:
        z = dict(z)
    np.savez(tmp_path / "short.npz", sha=z["sha"][:1], emb=z["emb"][:1], n_tokens=z["n_tokens"][:1])
    with pytest.raises(ValueError, match="2 missing"):
        import_embeddings(tmp_path / "short.npz", meta, inp, cache_dir=tmp_path / "c1")
    m = json.loads(meta.read_text())
    (tmp_path / "other.json").write_text(json.dumps({**m, "revision": "latest"}))
    with pytest.raises(ValueError, match="Model 5's revision"):
        import_embeddings(npz, tmp_path / "other.json", inp, cache_dir=tmp_path / "c2")
    np.savez(tmp_path / "nan.npz", sha=z["sha"], emb=np.full_like(z["emb"], np.nan),
             n_tokens=z["n_tokens"])
    with pytest.raises(ValueError, match="non-finite"):
        import_embeddings(tmp_path / "nan.npz", meta, inp, cache_dir=tmp_path / "c3")
    import_embeddings(npz, meta, inp, cache_dir=tmp_path / "c4")
    np.savez(tmp_path / "diff.npz", sha=z["sha"], emb=z["emb"] + 1, n_tokens=z["n_tokens"])
    with pytest.raises(ValueError, match="different values"):
        import_embeddings(tmp_path / "diff.npz", meta, inp, cache_dir=tmp_path / "c4")


# --- R1 heads: only the encoder differs; text=None is the backbone exactly ---------------------

@pytest.mark.parametrize("encoder_dim", [1536, 768])
def test_r1_heads_load_and_reproduce_the_backbone_without_text(tmp_path, monkeypatch, encoder_dim):
    from tests.test_late_fusion import _data, needs_mmtsflib  # noqa: F401
    from src.models import _mmtsflib
    if not _mmtsflib.available():
        pytest.skip("MM-TSFlib not cloned")
    from src.models import late_fusion as lf
    from src.models.dlinear import DLinearForecaster
    from src.models.patchtst import PatchTSTForecaster

    class Enc:
        def encode(self, texts):
            return np.stack([np.random.default_rng(int(prompt_sha(make_prompt(t))[:8], 16))
                             .normal(size=encoder_dim) for t in texts]).astype(np.float32)
    monkeypatch.setattr(lf, "_encoder", lambda name: Enc())
    heads = ({1536: (lf.DLinearQwenForecaster, lf.PatchTSTQwenForecaster),
              768: (lf.DLinearFinBertForecaster, lf.PatchTSTFinBertForecaster)}[encoder_dim])
    s, w = _data()
    for base_cls, cls in zip((DLinearForecaster, PatchTSTForecaster), heads):
        ckpt = tmp_path / f"{base_cls.name}.pt"
        base = base_cls(8, 1, seed=2021, epochs=2).fit(w["train"], w["val"], s.train["value"],
                                                        checkpoint=ckpt)
        m = cls(8, 1, seed=2021, epochs=2, backbone_ckpt=ckpt)
        assert m.encoder_name == (QWEN_NAME if encoder_dim == 1536 else FINBERT_NAME)
        assert m.emb_dim == encoder_dim
        m.fit(w["train"], w["val"], s.train["value"], checkpoint=tmp_path / "head.pt")
        assert m.mlp[0].in_features == encoder_dim and m.mlp[0].out_features == 96
        X = w["test"].X
        assert np.array_equal(m.predict_windows(X, None), base.predict_windows(X, None))
        again = cls.load(tmp_path / "head.pt")
        assert again.encoder_name == m.encoder_name
        assert np.allclose(again.predict_windows(X, w["test"].text),
                           m.predict_windows(X, w["test"].text))


def test_r1_family_has_exactly_108_tests():
    from src.analyze_r1 import FAMILY_SIZE, R1_HEADS
    assert sum(FAMILY_SIZE.values()) == 108 and len(R1_HEADS) == 4
    assert FAMILY_SIZE == {"S": 4 * 4, "A1": 4 * 2 * 5, "A2": 4 * 2 * 5, "E": 4 * 2, "F'": 2 * 2}
