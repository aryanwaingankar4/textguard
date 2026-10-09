"""MM-TSFlib-style late fusion of a numbers-only backbone and frozen-BERT text (Models 3, 4).

What MM-TSFlib does (``exp/exp_long_term_forecasting.py``, read 2026-10-07 — issue #3):

    ŷ = (1 − w)·TS(x) + w·( norm(pool(MLP(E_tokens))) + prior_y )      (TS and MLP trained jointly)

with ``E_tokens`` BERT's *word-embedding table* lookups (``use_fullmodel=0``), an MLP
768 → 96 → pred_len applied per token, average pooling over tokens, ``norm`` a z-score
**across the forecast horizon**, ``prior_y`` a numeric seasonal prior column
(``prior_history_avg``: same month 1–2 years earlier) and ``w = prompt_weight``.

What we implement (decided 2026-10-07, PROJECT_STATUS §7a):

    ŷ = TS(x) + w·MLP(z(e_text))        TS = the trained Model 1/2 of the same seed, FROZEN

* **Kept from MM-TSFlib:** late additive fusion with weight ``w`` (0.1, as in its only shipped
  script), the MLP shape 768 → 96 → horizon with ReLU and dropout 0.3, the prompt template,
  frozen BERT, and the backbone networks (DLinear / PatchTST). ``(1 − w)·TS`` vs ``TS`` is
  the same function class — the backbone absorbs the constant.
* **Dropped ``norm()``** — at horizon 1 it is identically 0 (a value minus its own mean),
  so MM-TSFlib's text branch contributes *nothing* at h=1; at h>1 it can only reshape the
  forecast, never shift its level. A directional attack could not move it by construction.
* **Dropped ``prior_y``** — a numeric lag-12 feature riding in the "text" branch. Models 1/2
  do not get it, so it would make the Text Usage Gap measure a seasonal prior rather than
  text (the co-shipped-column confound of Sridhar et al. 2026).
* **Full 12-layer BERT**, mean-pooled over real tokens (MM-TSFlib averages padding too, so a
  text's vector depended on its batch-mates). Missing text is ``None``, not the string "nan".
* **Embeddings standardised per dimension (``z``), mean/std fit on the train split's texts
  only** (like the numeric scaler). Mean-pooled BERT vectors are nearly identical across
  texts (a large shared component; per-dimension spread across texts 0.03–0.25). Fed raw at
  lr 1e-2, every ReLU unit died within 1–3 epochs on Economy (0/96 alive) and the model gave
  every text the same output — text-deaf by optimisation, not by choice.
* **Two-stage training instead of joint:** the backbone is Model 1/2's checkpoint for the
  same seed, frozen (and kept in eval mode, so its dropout never perturbs TS(x)); only the
  text head (scaler + MLP) is trained, under the P2.2 protocol — lr from ``LR_GRID`` chosen
  on val, up to 200 epochs, patience 20, 3 seeds. MM-TSFlib's fixed head lr 1e-2 is one
  point of that grid.

**What ``text=None`` means** (also blank text, and periods with no report): **the text
branch is absent**, so

    predict(x, None) = TS(x) = Model 1/2's forecast, exactly.

Why two-stage, and why this definition — P2.3's critical check. With joint training (first
implementation, text dropout 0.5) the backbone co-adapted with the text and was a weaker
numbers-only model on its own: Economy DLinear+BERT with ``text=None`` scored val
0.109–0.128 over 3 seeds vs Model 1's 0.101, while real text scored 0.093–0.095. Most of the
apparent "text gain" was the no-text path being out of distribution, which inflates the Text
Usage Gap. With a frozen backbone, ``A = predict(x, None)`` *is* the numbers-only model, so

    Text Usage Gap = MSE(Model 1/2) − MSE(Model 3/4, real)     B − A = w·MLP(z(e_text))

measures the text and nothing else, and TextGuard's ``A + g·(B − A)`` interpolates between
the numbers-only forecast and the text-informed one. On val (seed 2021) the frozen variant
also beat the joint one with real text in all 8 domain × model cells (PROJECT_STATUS §7a).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data.windows import Windows
from src.models.dlinear import DLinearForecaster
from src.models.patchtst import PatchTSTForecaster
from src.models.text_encoder import EMB_DIM, MODEL_NAME, BertTextEncoder, get_encoder

PROMPT_WEIGHT = 0.1  # MM-TSFlib scripts/week_health.sh (its run.py default is 0.01)
MLP_DROPOUT = 0.3    # MM-TSFlib MLP(..., dropout_rate=0.3)

_ENCODERS: dict[str, BertTextEncoder] = {}


def _encoder(name: str) -> BertTextEncoder:
    """One encoder per process: the cache is on disk, BERT is loaded at most once."""
    if name not in _ENCODERS:
        _ENCODERS[name] = get_encoder(name)
    return _ENCODERS[name]


def unload_encoders() -> None:
    """Drop any BERT loaded by a cache miss (embeddings stay on disk)."""
    for enc in _ENCODERS.values():
        enc.unload()


def _clean(t) -> str | None:
    """Blank / missing text → None, matching ``BaseForecaster``'s rule."""
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return None
    t = str(t)
    return t if t.strip() else None


class EmbeddingScaler(nn.Module):
    """Per-dimension z-score of text embeddings. Buffers, so it is saved and restored with
    the checkpoint; fit on the train split only, never on val/test texts."""

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        super().__init__()
        self.register_buffer("mean", torch.zeros(EMB_DIM) if mean is None
                             else torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.ones(EMB_DIM) if std is None
                             else torch.as_tensor(std, dtype=torch.float32))

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return (e - self.mean) / self.std


class LateFusionMixin:
    """Adds a text head to a frozen, trained ``NeuralForecaster`` backbone via its hooks.

    ``fit`` needs ``backbone_ckpt`` — the Model 1/2 checkpoint of the same domain and seed —
    and ``lr`` is the *head's* learning rate (tuned on val like any other lr).
    """

    uses_text = True

    def __init__(self, seq_len: int, horizon: int = 1, prompt_weight: float = PROMPT_WEIGHT,
                 encoder_name: str = MODEL_NAME, backbone_ckpt: str | Path | None = None,
                 **kw):
        super().__init__(seq_len=seq_len, horizon=horizon, **kw)
        if not 0.0 < prompt_weight <= 1.0:
            raise ValueError("prompt_weight must be in (0, 1]")
        self.prompt_weight = prompt_weight
        self.encoder_name = encoder_name
        self.backbone_ckpt = None if backbone_ckpt is None else str(backbone_ckpt)
        self.mlp: nn.Module | None = None
        self.emb_scaler: EmbeddingScaler | None = None
        self._emb_stats: tuple[np.ndarray, np.ndarray] | None = None
        self._backbone_state: dict | None = None

    def hyperparams(self) -> dict:
        # backbone_ckpt is provenance only: a saved Model 3/4 checkpoint contains the frozen
        # backbone's weights itself and never re-reads the Model 1/2 file
        return {**super().hyperparams(), "prompt_weight": self.prompt_weight,
                "encoder_name": self.encoder_name, "backbone_ckpt": self.backbone_ckpt}

    def fit(self, train: Windows, val: Windows, train_values, checkpoint=None):
        """Load the frozen backbone, fit the embedding scaler on *train* texts, train the head."""
        if self.backbone_ckpt is None:
            raise ValueError("backbone_ckpt is required: Models 3/4 build on a trained Model 1/2")
        base_cls = next(c for c in type(self).__mro__[1:]
                        if c is not LateFusionMixin and "build_network" in c.__dict__)
        base = base_cls.load(self.backbone_ckpt)
        mine = {k: v for k, v in self.hyperparams().items() if k in base.hyperparams()}
        if base.hyperparams() != mine:  # same seq_len, horizon, patching — and same seed
            raise ValueError(f"backbone {base.hyperparams()} does not match {mine}")
        self._backbone_state = base.net.state_dict()

        real = [t for t in (_clean(t) for t in (train.text if train.text is not None else []))
                if t is not None]
        if not real:
            raise ValueError("no text in the training windows; a text model cannot be fit")
        e = _encoder(self.encoder_name).encode(real)
        self._emb_stats = (e.mean(0), e.std(0) + 1e-6)

        super().fit(train, val, train_values, checkpoint)
        if not (np.isclose(self.mean, base.mean) and np.isclose(self.std, base.std)):
            raise ValueError("train split differs from the backbone's (scaler mismatch)")
        return self

    # --- hooks -----------------------------------------------------------------------

    def _build(self):
        self.net = self.build_network()
        if self._backbone_state is not None:
            self.net.load_state_dict(self._backbone_state)
        for p in self.net.parameters():
            p.requires_grad = False
        self.emb_scaler = EmbeddingScaler(*(self._emb_stats or (None, None)))
        self.mlp = nn.Sequential(nn.Linear(EMB_DIM, EMB_DIM // 8), nn.ReLU(),
                                 nn.Dropout(MLP_DROPOUT), nn.Linear(EMB_DIM // 8, self.horizon))
        return [torch.optim.Adam(self.mlp.parameters(), lr=self.lr)]

    def _modules(self):
        return {**super()._modules(), "mlp": self.mlp, "emb_scaler": self.emb_scaler}

    def _set_train(self, on: bool) -> None:
        super()._set_train(on)
        self.net.eval()  # frozen backbone: never in train mode, so TS(x) is Model 1/2 exactly

    def _inputs(self, w: Windows, texts=None):
        (x,) = super()._inputs(w)
        texts = [None] * len(x) if w.text is None else [_clean(t) for t in w.text]
        has = torch.tensor([t is not None for t in texts])
        e = torch.zeros(len(x), EMB_DIM)
        if has.any():
            e[has] = torch.as_tensor(_encoder(self.encoder_name).encode(
                [t for t in texts if t is not None]))
        return x, e, has

    def _forward(self, inputs, train: bool = False):
        x, e, has = inputs
        ts = super()._forward((x,), train)
        text = self.prompt_weight * self.mlp(self.emb_scaler(e))
        return ts + torch.where(has.unsqueeze(-1), text, torch.zeros_like(text))

    # --- diagnostics -----------------------------------------------------------------

    def text_effect(self, X: np.ndarray, texts) -> np.ndarray:
        """``predict(x, text) − predict(x, None)`` in raw units, (N, horizon) — B − A."""
        return self.predict_windows(X, texts) - self.predict_windows(X, [None] * len(X))


class DLinearBertForecaster(LateFusionMixin, DLinearForecaster):
    """Model 3 — frozen Model 1 (DLinear) + frozen-BERT text head, late fusion."""
    name = "DLinear+BERT"


class PatchTSTBertForecaster(LateFusionMixin, PatchTSTForecaster):
    """Model 4 — frozen Model 2 (PatchTST, ``patching="window"``) + frozen-BERT text head."""
    name = "PatchTST+BERT"
