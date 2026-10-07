"""Tests for Models 3/4 (late fusion) and the text pipeline around them.

A fake encoder (deterministic hash → vector) replaces BERT, so these tests never download
or run BERT; ``test_text_encoder_*`` cover the real encoder's cache logic with a stub model.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.splits import chronological_split
from src.data.windows import split_windows
from src.models import _mmtsflib
from src.models import late_fusion as lf
from src.models import text_encoder as te
from src.models.base import Forecaster

needs_mmtsflib = pytest.mark.skipif(not _mmtsflib.available(), reason="MM-TSFlib not cloned")


class FakeEncoder:
    """Deterministic 768-d vector per prompt; records what it was asked to encode."""

    def __init__(self):
        self.seen = []

    def encode(self, texts):
        out = []
        for t in texts:
            p = te.make_prompt(t)
            self.seen.append(t)
            seed = int(hashlib.sha256(p.encode()).hexdigest()[:8], 16)
            out.append(np.random.default_rng(seed).normal(size=te.EMB_DIM))
        return np.asarray(out, dtype=np.float32)


@pytest.fixture
def fake_encoder(monkeypatch):
    enc = FakeEncoder()
    monkeypatch.setattr(lf, "_encoder", lambda name: enc)
    return enc


def _data(n=120, seq_len=8, text_every=1):
    t = np.arange(n)
    start = pd.date_range("2000-01-01", periods=n, freq="MS")
    texts = [f"report {i}: value {'rose' if i % 2 else 'fell'}" if i % text_every == 0 else None
             for i in range(n)]
    df = pd.DataFrame({"start_date": start, "end_date": start + pd.offsets.MonthEnd(0),
                       "value": 1000 + 10 * np.sin(t / 3) + 0.2 * t, "text": texts,
                       "has_text": [x is not None for x in texts]})
    s = chronological_split(df)
    return s, split_windows(s, seq_len, 1)


def _models(tmp_path, s, w, seed=2021):
    """Train Model 1 and 2 (2 epochs) to checkpoints; return Models 3 and 4 built on them."""
    from src.models.dlinear import DLinearForecaster
    from src.models.patchtst import PatchTSTForecaster
    out = []
    for base_cls, cls in ((DLinearForecaster, lf.DLinearBertForecaster),
                          (PatchTSTForecaster, lf.PatchTSTBertForecaster)):
        ckpt = tmp_path / f"{base_cls.name}_seed{seed}.pt"
        if not ckpt.exists():
            base_cls(8, 1, seed=seed, epochs=2).fit(w["train"], w["val"], s.train["value"],
                                                     checkpoint=ckpt)
        out.append(cls(8, 1, seed=seed, epochs=2, backbone_ckpt=ckpt))
    return out


# --- windows carry the right text --------------------------------------------------------

def test_window_text_is_the_first_target_rows_text():
    s, w = _data()
    full = pd.concat([s.train, s.val, s.test], ignore_index=True)
    for name in ("train", "val", "test"):
        dates = w[name].target_start
        expected = full.set_index("start_date").loc[dates, "text"].tolist()
        assert list(w[name].text) == expected


def test_windows_without_text_column_have_none():
    s, _ = _data()
    s2 = type(s)(*(df.drop(columns="text") for df in (s.train, s.val, s.test)))
    assert split_windows(s2, 8, 1)["test"].text is None


# --- MM-TSFlib finding, locked in --------------------------------------------------------

def test_mmtsflib_norm_is_identically_zero_at_horizon_1():
    """MM-TSFlib's text term norm(t) z-scores across the horizon: at h=1 it is always 0."""
    def norm(e):  # verbatim from MM-TSFlib exp_long_term_forecasting.py
        e = e - e.mean(1, keepdim=True).detach()
        return e / torch.sqrt(torch.var(e, dim=1, keepdim=True, unbiased=False) + 1e-5)
    t = torch.randn(16, 1, 1) * 100
    assert torch.count_nonzero(norm(t)) == 0
    assert torch.allclose(norm(torch.randn(4, 6, 1)).mean(1), torch.zeros(4, 1), atol=1e-6)


# --- the fusion models -------------------------------------------------------------------

@needs_mmtsflib
@pytest.mark.parametrize("idx", [0, 1])
def test_text_none_is_exactly_the_frozen_numbers_only_model(idx, fake_encoder, tmp_path):
    """The P2.3 critical check, by construction: A = predict(x, None) = Model 1/2."""
    s, w = _data()
    m = _models(tmp_path, s, w)[idx]
    m.fit(w["train"], w["val"], s.train["value"])
    base = type(m).__mro__[2].load(m.backbone_ckpt)
    X = w["test"].X
    np.testing.assert_array_equal(m.predict_windows(X, None), base.predict_windows(X))
    np.testing.assert_array_equal(m.predict(X[0], "   "), base.predict(X[0]))   # blank ≡ None
    for (k, v), (k0, v0) in zip(m.net.state_dict().items(), base.net.state_dict().items()):
        assert k == k0 and torch.equal(v, v0)                                   # backbone frozen


@needs_mmtsflib
@pytest.mark.parametrize("idx", [0, 1])
def test_predict_contract_and_text_moves_the_forecast(idx, fake_encoder, tmp_path):
    s, w = _data()
    m = _models(tmp_path, s, w)[idx].fit(w["train"], w["val"], s.train["value"])
    assert isinstance(m, Forecaster)
    x = w["test"].X[0]
    b = m.predict(x, "prices collapsed")
    assert b.shape == (1,) and 900 < b[0] < 1100                              # raw units
    assert not np.array_equal(b, m.predict(x, None))                          # text matters
    np.testing.assert_array_equal(b, m.predict(x, "prices collapsed"))        # eval deterministic
    eff = m.text_effect(w["test"].X[:3], ["a", None, "b"])
    assert eff[1, 0] == 0.0 and eff[0, 0] != 0.0


@needs_mmtsflib
def test_backbone_stays_in_eval_mode_while_head_trains(fake_encoder, tmp_path):
    s, w = _data()
    m = _models(tmp_path, s, w)[1]  # PatchTST has dropout: train mode would perturb TS(x)
    m.fit(w["train"], w["val"], s.train["value"])
    m._set_train(True)
    assert not m.net.training and m.mlp.training


@needs_mmtsflib
def test_embedding_scaler_is_fit_on_train_texts_only(fake_encoder, tmp_path):
    s, w = _data()
    m = _models(tmp_path, s, w)[0].fit(w["train"], w["val"], s.train["value"])
    train_real = [t for t in w["train"].text if t is not None]
    e = FakeEncoder().encode(train_real)
    np.testing.assert_allclose(m.emb_scaler.mean.numpy(), e.mean(0), rtol=1e-5, atol=1e-6)
    val_texts = {t for t in w["val"].text if t is not None}
    assert val_texts and not val_texts & set(train_real)


@needs_mmtsflib
@pytest.mark.parametrize("idx", [0, 1])
def test_checkpoint_round_trip_without_bert_weights(idx, fake_encoder, tmp_path):
    s, w = _data()
    m = _models(tmp_path, s, w)[idx]
    m.fit(w["train"], w["val"], s.train["value"], checkpoint=tmp_path / "m.pt")
    ckpt = torch.load(tmp_path / "m.pt", weights_only=False)
    keys = list(ckpt["state_dict"]) + [f"{k}.{p}" for k, st in ckpt["extra_state"].items()
                                       for p in st]
    assert not any("bert" in k.lower() or "encoder.layer" in k for k in keys)
    assert set(ckpt["extra_state"]) == {"mlp", "emb_scaler"}
    n_params = sum(v.numel() for st in [ckpt["state_dict"], *ckpt["extra_state"].values()]
                   for v in st.values())
    assert n_params < 10_000_000                                 # BERT alone is 110M
    Path(m.backbone_ckpt).unlink()           # a saved Model 3/4 must not need Model 1/2's file
    loaded = type(m).load(tmp_path / "m.pt")
    X, T = w["test"].X, w["test"].text
    np.testing.assert_allclose(loaded.predict_windows(X, T), m.predict_windows(X, T))
    np.testing.assert_allclose(loaded.predict_windows(X, None), m.predict_windows(X, None))


@needs_mmtsflib
def test_same_seed_is_reproducible(fake_encoder, tmp_path):
    s, w = _data()
    a = _models(tmp_path, s, w)[0].fit(w["train"], w["val"], s.train["value"])
    b = _models(tmp_path, s, w)[0].fit(w["train"], w["val"], s.train["value"])
    np.testing.assert_array_equal(a.predict_windows(w["test"].X, w["test"].text),
                                  b.predict_windows(w["test"].X, w["test"].text))


@needs_mmtsflib
def test_backbone_must_exist_and_match_seed(fake_encoder, tmp_path):
    s, w = _data()
    with pytest.raises(ValueError, match="backbone_ckpt is required"):
        lf.DLinearBertForecaster(8, 1, epochs=2).fit(w["train"], w["val"], s.train["value"])
    ckpt = _models(tmp_path, s, w, seed=2021)[0].backbone_ckpt
    wrong_seed = lf.DLinearBertForecaster(8, 1, seed=2022, epochs=2, backbone_ckpt=ckpt)
    with pytest.raises(ValueError, match="does not match"):
        wrong_seed.fit(w["train"], w["val"], s.train["value"])


@needs_mmtsflib
def test_fit_without_any_text_raises(fake_encoder, tmp_path):
    s, w = _data(text_every=10_000)  # only row 0 has text, and it is context-only
    with pytest.raises(ValueError, match="no text"):
        _models(tmp_path, s, w)[0].fit(w["train"], w["val"], s.train["value"])


def test_patchtst_bert_uses_window_patching():
    assert lf.PatchTSTBertForecaster(8, 1).patching == "window"


# --- text encoder cache --------------------------------------------------------------------

def test_text_encoder_caches_and_only_embeds_misses(tmp_path, monkeypatch):
    enc = te.BertTextEncoder(cache_dir=tmp_path)
    calls = []

    def fake_embed(prompts):
        calls.append(list(prompts))
        return np.stack([np.full(te.EMB_DIM, len(p), dtype=np.float32) for p in prompts])

    monkeypatch.setattr(enc, "_embed", fake_embed)
    first = enc.encode(["a", "bb", None, "a"])
    assert calls == [[te.make_prompt("a"), te.make_prompt("bb"), te.make_prompt(None)]]
    np.testing.assert_array_equal(first[0], first[3])
    second = enc.encode(["bb", "a", None])
    assert len(calls) == 1                                   # all cache hits
    np.testing.assert_array_equal(second, first[[1, 0, 2]])


def test_null_prompt_is_the_empty_template():
    assert te.make_prompt(None) == te.PROMPT.format(text="")
    assert "nan" not in te.make_prompt(None)


def test_text_encoder_unload_releases_bert(tmp_path):
    enc = te.BertTextEncoder(cache_dir=tmp_path)
    enc._bert, enc._tok = object(), object()  # pretend a cache miss loaded BERT
    assert enc.is_loaded
    enc.unload()
    assert not enc.is_loaded and enc._tok is None
