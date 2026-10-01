"""The uniform forecaster interface: ``predict(numbers, text) -> prediction``.

Why this module matters more than any model wrapper: TextGuard (P6) and the Auditor (P7)
are meant to work on *any* forecaster, including ones we never trained. They can only do
that if every model is reachable through one black-box call. TextGuard needs exactly two
calls per sample::

    A = model.predict(numbers, None)   # numbers only
    B = model.predict(numbers, text)   # numbers + text
    final = A + g * (B - A)

Nothing else — no gradients, no embeddings, no training. If a model can answer those two
calls, it can be audited and defended.

Two layers:

* ``Forecaster`` — a ``Protocol``: the contract. Any object with a ``horizon`` attribute
  and a ``predict(numbers, text)`` method satisfies it, **without inheriting from us**.
  A third-party model can be plugged in as-is (duck typing).
* ``BaseForecaster`` — an optional helper for *our* wrappers. It validates inputs and
  outputs once, here, so every wrapper (DLinear, PatchTST, BERT variants, Qwen, dummy)
  obeys exactly the same rules, then calls the wrapper's ``_predict``.

The contract, and why each rule exists:

1. ``numbers`` is a **1-D univariate lookback window in raw data units.**
   *Univariate is deliberate.* Sridhar et al. (2026) found that apparent "text gains" in
   multimodal TSF can come from a co-shipped numeric column rather than from the text. By
   feeding only the target series, any difference between ``A`` and ``B`` can come from
   the text alone.
   *Raw units are deliberate.* DSS divides by σ in data units, and TextGuard mixes
   ``A`` and ``B`` linearly; both are only meaningful if predictions come back in the
   units of the data. Any scaling is the wrapper's private business — and **a wrapper
   must fit its scaler on the train split only**, never on val/test (that would leak
   future statistics into the model).
2. ``text`` is ``str | None``. ``None`` means "no text channel": the numbers-only
   prediction. Each wrapper documents what that means internally (e.g. a zeroed text
   embedding). **Blank text ("" or whitespace) is normalised to ``None``**, so the
   ``empty`` attack condition, rows with ``has_text=False`` and TextGuard's ``A`` are all
   the *same* prediction. If they could differ, "Text Usage Gap" and TextGuard's
   baseline would silently measure different things.
3. The prediction is always an ``ndarray`` of shape ``(horizon,)`` — even for horizon 1 —
   so metrics and the gate are plain array arithmetic. Main experiments use horizon 1.
4. **Failures return NaN; they never raise and never drop the sample.** An LLM forecaster
   (Qwen, P2.4) sometimes returns unparseable text. Raising would crash a long run;
   dropping would hide the failure. A NaN row lands in the raw-predictions CSV and the
   metrics count it. Infinite values are *not* failures — they are bugs, and raise.
5. Training, dates and domain are **not** part of the interface. TextGuard is inference
   only; how a model is trained is wrapper-specific, and the domain is fixed when the
   model is constructed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Forecaster(Protocol):
    """Anything with this shape can be audited and defended. No inheritance needed."""

    horizon: int

    def predict(self, numbers: np.ndarray, text: str | None) -> np.ndarray:
        """Return a float array of shape ``(horizon,)``; NaN marks a failed prediction."""
        ...


def check_numbers(numbers, lookback: int | None = None) -> np.ndarray:
    """Coerce ``numbers`` to a 1-D finite float array, or raise.

    A 2-D input is rejected rather than flattened: it almost always means someone passed a
    multivariate frame, which would bring back the co-shipped-column confound (see module
    docstring). NaN inputs are rejected because a gap in the history is a data bug that
    must be fixed upstream, not a model failure.
    """
    arr = np.asarray(numbers, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"numbers must be a 1-D univariate window, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("numbers must contain at least one value")
    if not np.all(np.isfinite(arr)):
        raise ValueError("numbers contains NaN or inf; fix the data, not the model")
    if lookback is not None and arr.size != lookback:
        # Strict equality, not "take the last L": silently truncating would hide a
        # windowing bug in the experiment runner.
        raise ValueError(f"expected a lookback window of {lookback} values, got {arr.size}")
    return arr


def normalise_text(text) -> str | None:
    """``None`` / blank string -> ``None``; any other string is returned unchanged."""
    if text is None:
        return None
    if not isinstance(text, str):
        raise TypeError(f"text must be str or None, got {type(text).__name__}")
    return text if text.strip() else None


def check_output(out, horizon: int) -> np.ndarray:
    """Coerce a wrapper's output to shape ``(horizon,)``. NaN allowed, inf is a bug."""
    arr = np.asarray(out, dtype=float)
    if arr.ndim == 0 and horizon == 1:
        arr = arr.reshape(1)
    if arr.shape != (horizon,):
        raise ValueError(f"prediction must have shape ({horizon},), got {arr.shape}")
    if np.any(np.isinf(arr)):
        raise ValueError("prediction contains inf; return NaN for failures instead")
    return arr


class BaseForecaster(ABC):
    """Shared validation for our wrappers. Subclasses implement ``_predict`` only.

    ``predict`` is a template method: validate -> ``_predict`` -> validate. Keeping the
    checks here means a wrapper cannot accidentally break the contract that TextGuard
    relies on.
    """

    def __init__(self, horizon: int = 1, lookback: int | None = None):
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if lookback is not None and lookback < 1:
            raise ValueError("lookback must be >= 1 or None")
        self.horizon = horizon
        self.lookback = lookback

    def predict(self, numbers, text: str | None = None) -> np.ndarray:
        numbers = check_numbers(numbers, self.lookback)
        text = normalise_text(text)
        return check_output(self._predict(numbers, text), self.horizon)

    @abstractmethod
    def _predict(self, numbers: np.ndarray, text: str | None) -> np.ndarray:
        """Receives validated inputs (``text`` already ``None`` if blank)."""

    def predict_batch(self, numbers_list: Sequence, texts: Sequence[str | None]) -> list[np.ndarray]:
        """Loop over ``predict``. BERT / Qwen wrappers override this for speed.

        Batching is an optimisation, not part of the contract: the single-sample
        ``predict`` stays the definition of what the model does.
        """
        if len(numbers_list) != len(texts):
            raise ValueError("numbers_list and texts must have the same length")
        return [self.predict(n, t) for n, t in zip(numbers_list, texts)]
