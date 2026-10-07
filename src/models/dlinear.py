"""Model 1 — DLinear, numbers only (Zeng et al., AAAI 2023).

DLinear splits the window into a trend (moving average) and a remainder, and maps each to
the forecast with one linear layer. It is the simplest strong baseline in the field: if a
multimodal model cannot beat DLinear, the text is not helping.

The network is MM-TSFlib's ``models/DLinear.py``, unchanged, with MM-TSFlib's defaults
(``moving_avg=25``, shared weights across channels). With a lookback of 8, the 25-wide
moving average is wider than the window; MM-TSFlib pads by edge replication, so the trend
is close to a flat line. That is the published setting, so we keep it.

Training, scaling and the ``predict`` contract are in ``NeuralForecaster``.
"""

from __future__ import annotations

from src.models import _mmtsflib
from src.models.neural import NeuralForecaster


class DLinearForecaster(NeuralForecaster):
    name = "DLinear"

    def build_network(self):
        cfg = _mmtsflib.default_configs(self.seq_len, self.horizon)
        return _mmtsflib.model_class("DLinear")(cfg)
