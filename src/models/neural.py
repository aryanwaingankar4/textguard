"""Shared training / inference for the neural numbers-only forecasters (DLinear, PatchTST).

Why one shared class: DLinear and PatchTST differ only in their network. Scaling,
training, early stopping, checkpointing and the ``predict`` contract must be *identical*,
or a difference between the two models could come from the pipeline instead of the
architecture.

The training loop mirrors MM-TSFlib's ``exp_long_term_forecasting.train``:

* Adam, MSE loss on the **normalised** scale, constant lr (MM-TSFlib has its
  ``adjust_learning_rate`` call commented out), batch 32, shuffled, ``drop_last=True``
* early stopping on validation loss, best epoch restored

The constructor defaults (10 epochs, patience 5, lr 1e-4) are MM-TSFlib's recipe, used for
the reproduction check. Main experiments override them with the val-tuned protocol in
``train_baselines.py`` (up to 200 epochs, patience 20, lr chosen on val).

What deliberately differs: the data. Splits are ours (chronological over the text window,
issue #7) and the scaler is fit on **our train split only**.

Scaling: a z-score with the train split's mean and std (population std, as sklearn's
``StandardScaler``). ``predict`` scales the input, runs the network and inverts the scaling,
so callers — TextGuard, metrics — only ever see raw data units.

These models are numbers-only: ``text`` is accepted (the interface requires it) and
ignored, so ``predict(x, text) == predict(x, None)`` by construction. That is what makes
them the control group: their Text Usage Gap is exactly zero.
"""

from __future__ import annotations

import copy
import random
from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data.windows import Windows
from src.models.base import BaseForecaster


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class NeuralForecaster(BaseForecaster):
    """A torch network behind the ``predict(numbers, text)`` contract. Subclasses build the net."""

    name = "neural"

    def __init__(self, seq_len: int, horizon: int = 1, seed: int = 2021, epochs: int = 10,
                 batch_size: int = 32, lr: float = 1e-4, patience: int = 5):
        super().__init__(horizon=horizon, lookback=seq_len)
        self.seq_len = seq_len
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.mean: float | None = None
        self.std: float | None = None
        self.net: nn.Module | None = None
        self.history: list[dict] = []

    # --- subclass hook ---------------------------------------------------------------

    @abstractmethod
    def build_network(self) -> nn.Module:
        """Return an MM-TSFlib-style module: ``net(x, None, None, None) -> (B, horizon, 1)``."""

    def hyperparams(self) -> dict:
        """Everything needed to rebuild the network from a checkpoint."""
        return dict(seq_len=self.seq_len, horizon=self.horizon, seed=self.seed)

    # --- scaling ---------------------------------------------------------------------

    def _scale(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def _unscale(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self.mean

    # --- training --------------------------------------------------------------------

    def _forward(self, z: np.ndarray) -> np.ndarray:
        """Normalised windows (N, L) -> normalised forecasts (N, H)."""
        x = torch.as_tensor(z, dtype=torch.float32).unsqueeze(-1)
        return self.net(x, None, None, None)[:, -self.horizon:, 0]

    def _loss(self, w: Windows) -> float:
        self.net.eval()
        with torch.no_grad():
            pred = self._forward(self._scale(w.X))
            return float(nn.functional.mse_loss(pred, torch.as_tensor(self._scale(w.Y),
                                                                      dtype=torch.float32)))

    def fit(self, train: Windows, val: Windows, train_values: np.ndarray,
            checkpoint: Path | str | None = None) -> "NeuralForecaster":
        """Train on ``train`` windows, early-stop on ``val``.

        ``train_values`` is the *whole* train split (not just window targets): the scaler is
        fit on it, as MM-TSFlib fits its ``StandardScaler`` on all train rows.
        """
        train_values = np.asarray(train_values, dtype=float)
        self.mean = float(train_values.mean())
        self.std = float(train_values.std()) or 1.0  # constant series: avoid division by 0

        set_seed(self.seed)
        self.net = self.build_network()
        optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        X = torch.as_tensor(self._scale(train.X), dtype=torch.float32)
        Y = torch.as_tensor(self._scale(train.Y), dtype=torch.float32)
        gen = torch.Generator().manual_seed(self.seed)
        # drop_last as in MM-TSFlib, unless that would leave no batch at all (tiny tests)
        drop_last = len(X) >= self.batch_size

        best, best_state, bad_epochs = np.inf, None, 0
        self.history = []
        for epoch in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(X), generator=gen)
            n_batches = len(X) // self.batch_size if drop_last else -(-len(X) // self.batch_size)
            losses = []
            for b in range(n_batches):
                idx = perm[b * self.batch_size:(b + 1) * self.batch_size]
                optim.zero_grad()
                loss = nn.functional.mse_loss(self._forward(X[idx].numpy()), Y[idx])
                loss.backward()
                optim.step()
                losses.append(loss.item())
            val_loss = self._loss(val)
            self.history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)),
                                     val_loss=val_loss))
            if val_loss < best:
                best, best_state, bad_epochs = val_loss, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        if checkpoint is not None:
            self.save(checkpoint)
        return self

    # --- inference -------------------------------------------------------------------

    def predict_windows(self, X: np.ndarray) -> np.ndarray:
        """Batched forecasts for many windows, raw units, shape (N, horizon).

        Same computation as ``predict``, just vectorised for evaluation. Text is not an
        argument because these models ignore it.
        """
        if self.net is None:
            raise RuntimeError("model is not trained; call fit() or load() first")
        self.net.eval()
        with torch.no_grad():
            return self._unscale(self._forward(self._scale(np.asarray(X, float))).numpy())

    def _predict(self, numbers: np.ndarray, text: str | None) -> np.ndarray:
        return self.predict_windows(numbers[None, :])[0]

    # --- checkpoints -----------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"hyperparams": self.hyperparams(), "mean": self.mean, "std": self.std,
                    "state_dict": self.net.state_dict(), "history": self.history}, path)

    @classmethod
    def load(cls, path: Path | str) -> "NeuralForecaster":
        ckpt = torch.load(path, weights_only=False)
        model = cls(**ckpt["hyperparams"])
        model.mean, model.std, model.history = ckpt["mean"], ckpt["std"], ckpt["history"]
        model.net = model.build_network()
        model.net.load_state_dict(ckpt["state_dict"])
        model.net.eval()
        return model
