"""Persistence baseline: the forecast is the last observed value.

Why it exists: it is the sanity floor. It needs no training and no parameters, so it
cannot be wrong because of a pipeline bug. If a trained model (PatchTST, DLinear) does not
beat persistence on a domain, suspect the pipeline — scaling, windowing, training — before
concluding anything about the model.

For horizon H it repeats the last value H times. ``text`` is ignored.
"""

from __future__ import annotations

import numpy as np

from src.models.base import BaseForecaster


class PersistenceForecaster(BaseForecaster):
    name = "Persistence"

    def _predict(self, numbers: np.ndarray, text: str | None) -> np.ndarray:
        return np.full(self.horizon, numbers[-1])

    def predict_windows(self, X: np.ndarray) -> np.ndarray:
        """Batched forecasts, shape (N, horizon) — same interface as ``NeuralForecaster``."""
        X = np.asarray(X, dtype=float)
        return np.repeat(X[:, -1:], self.horizon, axis=1)
