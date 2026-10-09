"""Mixing A (numbers only) and B (with text) — TextGuard's rule and the baseline defences (P6).

Every defence here is a rule for the **applied weight** g′ on the text's effect:

    final = A + g′ · (B − A)

so all of them are compared on the same axis (PROJECT_STATUS §10b): the False Block Rate is
the share of genuine texts with g′ < 0.5, whatever the defence.

* ``always_trust`` — g′ = 1: the undefended model (final = B). FBR 0%.
* ``never_trust``  — g′ = 0: ignore all text (final = A). FBR 100%; DSS exactly 0.
* ``clip``         — cap the text's effect at k·σ: final = A + clamp(B − A, ±kσ), i.e.
  g′ = min(1, kσ / |B − A|) (1 when B = A). Model-dependent; needs no judge.
* ``textguard``    — g′ from the gate. ``mode="soft"``: g′ = g (CLAUDE.md §8). ``mode="block"``
  at threshold τ: g′ = 0 where g < τ (a warning falls back to A), else g. τ = 0 is the soft
  gate and τ > 1 is never_trust, so sweeping τ traces the trade-off between the two. In soft
  mode τ only sets the warning flag — it never changes a forecast — which is why the
  pre-registered curve uses block mode.

NaN rules, binding (PROJECT_STATUS §6a): B NaN → final = A (fall back to numbers only);
A NaN → final NaN — there is nothing to fall back to, and it is counted as a failure.

All functions are vectorised: A, B of shape (n,) or (n, horizon); g, σ of shape (n,) or
scalar. P6b evaluates from saved predictions, never by re-running a model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Defended:
    final: np.ndarray    # the defended forecast, same shape as A
    weight: np.ndarray   # applied weight g′ per sample, shape (n,)
    warn: np.ndarray     # per sample: did the defence raise a warning (TextGuard only)


def _ab(A, B):
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    if A.shape != B.shape:
        raise ValueError(f"A and B must have the same shape, got {A.shape} and {B.shape}")
    return A, B


def _per_sample(x, A: np.ndarray) -> np.ndarray:
    """Broadcast a per-sample value against A (n,) or (n, horizon)."""
    x = np.asarray(x, dtype=float)
    return x.reshape(x.shape + (1,) * (A.ndim - x.ndim)) if x.ndim else x


def mix(A, B, weight, warn=None) -> Defended:
    """final = A + g′(B − A), with the NaN rules: B NaN → A, A NaN → NaN."""
    A, B = _ab(A, B)
    n = A.shape[0] if A.ndim else 1
    w = np.broadcast_to(np.asarray(weight, dtype=float), (n,)).copy()
    if np.any((w < 0) | (w > 1)):
        raise ValueError("weights must lie in [0, 1]")
    final = A + _per_sample(w, A) * (np.where(np.isnan(B), A, B) - A)
    warn = np.zeros(n, bool) if warn is None else np.broadcast_to(np.asarray(warn, bool), (n,))
    return Defended(final, w, warn.copy())


def always_trust(A, B) -> Defended:
    return mix(A, B, 1.0)


def never_trust(A, B) -> Defended:
    return mix(A, B, 0.0)


def clip(A, B, sigma, k: float) -> Defended:
    """Cap |B − A| at k·σ per sample (per horizon step; weight from the largest step)."""
    if k < 0:
        raise ValueError("k must be >= 0")
    A, B = _ab(A, B)
    diff = np.abs(np.where(np.isnan(B), A, B) - A)
    if diff.ndim > 1:
        diff = np.nanmax(diff, axis=tuple(range(1, diff.ndim)))
    cap = k * np.broadcast_to(np.asarray(sigma, dtype=float), diff.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(diff > cap, cap / diff, 1.0)
    return mix(A, B, np.nan_to_num(w, nan=1.0))


def textguard(A, B, g, threshold: float = 0.5, mode: str = "soft") -> Defended:
    """Apply TextGuard's gate values ``g``: soft (g′ = g) or block (g′ = 0 where g < τ)."""
    g = np.asarray(g, dtype=float)
    if np.any((g < 0) | (g > 1)):
        raise ValueError("gate values must lie in [0, 1]")
    warn = g < threshold
    if mode == "soft":
        w = g
    elif mode == "block":
        w = np.where(warn, 0.0, g)
    else:
        raise ValueError("mode must be 'soft' or 'block'")
    return mix(A, B, w, warn)
