"""Model 2 — PatchTST, numbers only (Nie et al., ICLR 2023).

PatchTST cuts the window into (overlapping) patches, embeds each patch as a token, and runs
a Transformer encoder over the tokens. It also normalises each window by its own mean and
std (instance normalisation) before the encoder.

The network is MM-TSFlib's ``models/PatchTST.py`` with MM-TSFlib's defaults (d_model 512,
8 heads, 2 layers, d_ff 2048, dropout 0.1).

⚠ **One deliberate deviation — patching at horizon 1.** MM-TSFlib's constructor contains
``if patch_len > pred_len: patch_len = pred_len`` (not in the original PatchTST). With
``pred_len=1`` that gives patch_len 1 with stride 8: on a lookback of 8 the encoder sees
two "patches", ``x[0]`` and the padded copy of ``x[7]``. Six of the eight inputs are never
read. That is a degenerate model, not PatchTST, so for our main experiments we bypass the
clamp:

* ``patching="window"`` (default): patch_len = seq_len // 2, stride = seq_len // 4 →
  four half-window patches covering every input (lookback 8 → patch 4 / stride 2;
  lookback 24 → patch 12 / stride 6). Implemented by building the network with
  ``pred_len = patch_len`` (so the clamp does not fire) and then swapping in a prediction
  head of the real horizon.
* ``patching="mmtsflib"``: the network exactly as shipped (patch 16, stride 8, clamped).
  Used only for the reproduction check at the published horizon (6 / 12), where it matches
  what produced the Time-MMD numbers.
"""

from __future__ import annotations

from src.models import _mmtsflib
from src.models.neural import NeuralForecaster


class PatchTSTForecaster(NeuralForecaster):
    name = "PatchTST"

    def __init__(self, seq_len: int, horizon: int = 1, patching: str = "window", **kw):
        if patching not in ("window", "mmtsflib"):
            raise ValueError("patching must be 'window' or 'mmtsflib'")
        super().__init__(seq_len=seq_len, horizon=horizon, **kw)
        self.patching = patching

    def hyperparams(self) -> dict:
        return {**super().hyperparams(), "patching": self.patching}

    def patch_geometry(self) -> tuple[int, int]:
        """(patch_len, stride) for ``patching="window"``."""
        return max(self.seq_len // 2, 1), max(self.seq_len // 4, 1)

    def build_network(self):
        mod = _mmtsflib.model_module("PatchTST")
        if self.patching == "mmtsflib":
            return mod.Model(_mmtsflib.default_configs(self.seq_len, self.horizon))
        patch_len, stride = self.patch_geometry()
        cfg = _mmtsflib.default_configs(self.seq_len, max(self.horizon, patch_len))
        net = mod.Model(cfg, patch_len=patch_len, stride=stride)
        net.pred_len = self.horizon  # used by its de-normalisation step
        net.head = mod.FlattenHead(cfg.enc_in, net.head_nf, self.horizon,
                                   head_dropout=cfg.dropout)
        return net
