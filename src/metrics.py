"""Metrics (P4): every number the paper reports. Pure functions — P5 runs them.

Design: each metric is the **mean of a per-window term** over the windows that are *valid*:

* inside the optional ``mask`` (True = include; e.g. P5 excludes 2020 with it, issue #23),
* where every input the term needs is finite — the **NaN-pairing rule** (PROJECT_STATUS §6a):
  if either side of a paired comparison is NaN (a parse or format failure), the window is
  dropped from *both* sides, so a model that fails more often under fake text cannot look
  robust because its worst cases vanished,
* and, for the attack metrics, where the window was actually attacked (``d != 0``, issue #30).

Every metric returns a ``Metric``: the value, how many windows it used, how many were dropped
for NaN / mask / not-attacked, and the per-window ``terms`` (NaN where excluded). The terms are
what the statistics use: ``bootstrap_ci`` (paired, stratified by domain, fixed seed),
``wilcoxon`` (paired signed-rank), ``holm`` (family-wise correction), ``grouped`` (pooled
headline + per-domain breakdown, issue #8).

Scale: errors and shifts are divided by ``sigma`` = the **train-split std of the domain's
series** (population std, the models' own scaler) — fixed before any test data is seen, so
MSE here equals the "normalised MSE" of PROJECT_STATUS §7. ``sigma`` may be a scalar or one
value per window (pooled tables mix domains).

Metric definitions (PROJECT_STATUS §9):

* ``mse`` / ``mae`` — mean ((y − ŷ)/σ)² / mean |y − ŷ|/σ
* ``text_usage_gap`` — MSE(text=None) − MSE(real), paired. ``empty`` ≡ ``None``. > 0: text helps
* ``text_sensitivity`` — mean |ŷ_real − ŷ_none|/σ: does text *move* the forecast at all,
  regardless of accuracy. **P5 selects attack targets by sensitivity, not usage gap.**
* ``dss`` — mean d·(ŷ_fake − ŷ_real)/σ over attacked windows, d = the attack's recorded
  ``claimed_direction`` (P3 metadata, never a classifier). > 0: pulled toward the lie.
  For ``exaggerated``, d is the real claim's direction: > 0 = pushed further that way
* ``flip_rate`` — share of attacked windows whose forecast direction (vs the last observed
  value) turns *toward* the lie: sign(ŷ_real − y_last) ≠ d and sign(ŷ_fake − y_last) = d
* ``false_block_rate`` — share of genuine texts with gate g < 0.5 (P6)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BLOCK_THRESHOLD = 0.5  # gate below this = text blocked / warning (CLAUDE.md §8)


# --- result type -----------------------------------------------------------------------------

@dataclass
class Metric:
    """A metric value plus exactly which windows produced it."""
    value: float                 # NaN if no window is valid
    n: int                       # windows used
    n_nan: int = 0               # dropped by the NaN-pairing rule (inside mask, attacked)
    n_masked: int = 0            # excluded by the caller's mask
    n_not_attacked: int = 0      # attack metrics only: windows with d == 0
    terms: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)

    @property
    def coverage(self) -> float:
        """Share of in-mask windows that were used (attack metrics: attacked and valid)."""
        total = self.n + self.n_nan + self.n_not_attacked
        return self.n / total if total else float("nan")

    def __float__(self) -> float:
        return float(self.value)


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _mask(mask, n: int) -> np.ndarray:
    if mask is None:
        return np.ones(n, dtype=bool)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if m.shape != (n,):
        raise ValueError(f"mask has {m.size} entries for {n} windows")
    return m


def _sigma(sigma, n: int) -> np.ndarray:
    s = np.broadcast_to(_arr(sigma), (n,)) if np.ndim(sigma) else np.full(n, float(sigma))
    if not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("sigma must be finite and > 0 (the train-split std of each domain)")
    return s


def _finish(raw: np.ndarray, inputs: list[np.ndarray], mask, attacked=None) -> Metric:
    """Apply mask, attack filter and NaN pairing to per-window ``raw`` terms."""
    n = raw.size
    for a in inputs:
        if a.size != n:
            raise ValueError("all inputs must have one value per window")
    m = _mask(mask, n)
    att = np.ones(n, dtype=bool) if attacked is None else attacked
    finite = np.all([np.isfinite(a) for a in inputs], axis=0) if inputs else np.ones(n, bool)
    use = m & att & finite
    terms = np.where(use, raw, np.nan)
    return Metric(value=float(np.mean(raw[use])) if use.any() else float("nan"),
                  n=int(use.sum()), n_nan=int((m & att & ~finite).sum()),
                  n_masked=int((~m).sum()), n_not_attacked=int((m & ~att).sum()),
                  terms=terms)


# --- accuracy --------------------------------------------------------------------------------

def mse(y_true, y_pred, sigma, mask=None) -> Metric:
    """Mean squared error on the normalised scale: mean(((y − ŷ)/σ)²)."""
    y, p = _arr(y_true), _arr(y_pred)
    s = _sigma(sigma, y.size)
    with np.errstate(invalid="ignore"):
        return _finish(((y - p) / s) ** 2, [y, p], mask)


def mae(y_true, y_pred, sigma, mask=None) -> Metric:
    """Mean absolute error on the normalised scale: mean(|y − ŷ|/σ)."""
    y, p = _arr(y_true), _arr(y_pred)
    s = _sigma(sigma, y.size)
    with np.errstate(invalid="ignore"):
        return _finish(np.abs(y - p) / s, [y, p], mask)


def paired_mse(y_true, preds: dict, sigma, mask=None) -> dict[str, Metric]:
    """MSE of several conditions on the **same** windows (NaN in any condition drops the
    window from all), so e.g. MSE(real) and MSE(none) in one table are comparable."""
    y = _arr(y_true)
    arrays = {k: _arr(v) for k, v in preds.items()}
    common = _mask(mask, y.size) & np.all([np.isfinite(a) for a in arrays.values()], axis=0)
    out = {k: mse(y, a, sigma, common) for k, a in arrays.items()}
    m = _mask(mask, y.size)
    for r in out.values():  # report NaN drops relative to the caller's mask
        r.n_masked, r.n_nan = int((~m).sum()), int((m & ~common).sum())
    return out


# --- does the model use / move with text -----------------------------------------------------

def text_usage_gap(y_true, y_pred_none, y_pred_real, sigma, mask=None) -> Metric:
    """MSE(text=None) − MSE(real), paired per window. > 0: the text improves accuracy."""
    y, a, b = _arr(y_true), _arr(y_pred_none), _arr(y_pred_real)
    s = _sigma(sigma, y.size)
    with np.errstate(invalid="ignore"):
        return _finish(((y - a) / s) ** 2 - ((y - b) / s) ** 2, [y, a, b], mask)


def text_sensitivity(y_pred_real, y_pred_none, sigma, mask=None) -> Metric:
    """mean |ŷ_real − ŷ_none|/σ — how far the text moves the forecast, right or wrong.

    The attack-target criterion (P5): a model whose forecast moves with text can be pushed
    by fake text even if the text does not make it more accurate (Model 5 on Health_US).
    """
    b, a = _arr(y_pred_real), _arr(y_pred_none)
    s = _sigma(sigma, b.size)
    with np.errstate(invalid="ignore"):
        return _finish(np.abs(b - a) / s, [a, b], mask)


def text_moves_share(y_pred_real, y_pred_none, mask=None, atol: float = 0.0) -> Metric:
    """Share of windows where the text changes the forecast at all (|ŷ_real − ŷ_none| > atol)."""
    b, a = _arr(y_pred_real), _arr(y_pred_none)
    with np.errstate(invalid="ignore"):
        return _finish((np.abs(b - a) > atol).astype(float), [a, b], mask)


# --- attack ----------------------------------------------------------------------------------

def _direction(d) -> np.ndarray:
    d = _arr(d)
    d = np.where(np.isfinite(d), d, 0.0)
    if not np.all(np.isin(d, (-1.0, 0.0, 1.0))):
        raise ValueError("d must be the recorded claimed_direction: +1, -1 or 0 (not attacked)")
    return d


def dss(y_pred_fake, y_pred_real, d, sigma, mask=None) -> Metric:
    """Directional Shift Score: mean d·(ŷ_fake − ŷ_real)/σ over attacked windows (d ≠ 0).

    ``d`` is P3's recorded ``claimed_direction`` of the fake text (+1/−1; 0 = not attacked,
    excluded and counted in ``n_not_attacked``). > 0: the forecast moved toward the lie,
    in units of the series' train std. ``coverage`` = attacked-and-valid / in-mask windows.
    """
    f, r, dd = _arr(y_pred_fake), _arr(y_pred_real), _direction(d)
    s = _sigma(sigma, f.size)
    with np.errstate(invalid="ignore"):
        return _finish(dd * (f - r) / s, [f, r], mask, attacked=dd != 0)


def flip_rate(y_pred_fake, y_pred_real, y_last, d, mask=None) -> Metric:
    """Share of attacked windows whose forecast direction turns toward the lie.

    Direction = sign(ŷ − y_last), the forecast's move relative to the last observed value.
    A window counts as flipped iff  sign(ŷ_real − y_last) ≠ d  and  sign(ŷ_fake − y_last) = d.

    Zero-change cases (exact equality, no tolerance — models that copy the last value, like
    Model 5, produce many):
    * real forecast exactly at y_last (sign 0): it was not in the lie's direction, so a fake
      forecast that moves in direction d **counts as a flip** (flat → toward the lie);
    * fake forecast exactly at y_last: it is not in direction d, so it **never counts**;
    * real already in direction d: cannot flip (it is in the denominator, not the numerator).
    Denominator: every attacked, valid window.
    """
    f, r, yl, dd = _arr(y_pred_fake), _arr(y_pred_real), _arr(y_last), _direction(d)
    with np.errstate(invalid="ignore"):
        raw = ((np.sign(r - yl) != dd) & (np.sign(f - yl) == dd)).astype(float)
    return _finish(raw, [f, r, yl], mask, attacked=dd != 0)


# --- defence ---------------------------------------------------------------------------------

def false_block_rate(g, mask=None, threshold: float = BLOCK_THRESHOLD) -> Metric:
    """Share of **genuine** texts that the gate blocks: g < 0.5.

    Pass the gate values TextGuard produced on *real* text (``mask`` selects windows that
    have a report). A defence that blocks every text scores perfectly on DSS and Flip Rate —
    this is the honesty check that catches it.
    """
    gg = _arr(g)
    if np.any((gg < 0) | (gg > 1)):
        raise ValueError("gate values must lie in [0, 1]")
    with np.errstate(invalid="ignore"):
        return _finish((gg < threshold).astype(float), [gg], mask)


# --- sanity ----------------------------------------------------------------------------------

def is_text_blind(y_pred_real, y_pred_none, *others) -> bool:
    """True iff every condition gives bit-identical forecasts — required of Models 1–2,
    whose DSS and sensitivity must then be exactly 0 (P5 asserts this)."""
    base = _arr(y_pred_real)
    return all(np.array_equal(base, _arr(o), equal_nan=True) for o in (y_pred_none, *others))


# --- seeds and failures ----------------------------------------------------------------------

def seed_mean(preds: pd.DataFrame, value: str = "y_pred", seed: str = "seed") -> pd.DataFrame:
    """Average per-window predictions over seeds (Models 1–4; Model 5 is one greedy run).

    A window that failed (NaN) in any seed stays NaN — a failure is never averaged away.
    ``n_seeds`` records how many seeds each window had.
    """
    keys = [c for c in preds.columns if c not in (value, seed) and c not in ("lr",)]
    g = preds.groupby(keys, dropna=False, sort=False)[value]
    out = g.agg(lambda s: s.mean() if s.notna().all() else np.nan).rename(value).reset_index()
    out["n_seeds"] = g.size().values
    return out


def failure_rates(preds: pd.DataFrame, by=("model", "condition"), value: str = "y_pred") -> pd.DataFrame:
    """Failure (NaN prediction) rate per group — reported next to every metric.

    For Model 5 the NaNs are parse failures and format (unit-confusion) failures, issue #27;
    when ``parse_ok`` / ``scale_ok`` columns are present they are broken out separately.
    """
    df = preds.assign(_fail=preds[value].isna())
    g = df.groupby(list(by), dropna=False)
    out = pd.DataFrame({"n": g.size(), "failures": g._fail.sum()})
    if "parse_ok" in df:
        out["parse_failures"] = g.parse_ok.apply(lambda s: int((~s.astype(bool)).sum()))
    if "scale_ok" in df:
        out["format_failures"] = g.scale_ok.apply(lambda s: int((~s.astype(bool)).sum()))
    out["rate"] = out.failures / out.n
    return out.reset_index()


# --- statistics ------------------------------------------------------------------------------

def bootstrap_ci(terms, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0,
                 strata=None) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean of per-window ``terms`` (NaN = excluded window).

    Resamples windows with replacement, **within each stratum** if ``strata`` is given (the
    domain, for pooled numbers — keeps each domain's share fixed). For a paired comparison
    pass the per-window difference (e.g. ``dss_before.terms − dss_after.terms``): resampling
    the pairs together is what makes the CI paired. Fixed ``seed`` → reproducible.
    """
    t = _arr(terms)
    keep = np.isfinite(t)
    t = t[keep]
    if t.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    if strata is None:
        idx = rng.integers(0, t.size, size=(n_boot, t.size))
        means = t[idx].mean(axis=1)
    else:
        st = np.asarray(strata)[keep]
        total = np.zeros(n_boot)
        for s in pd.unique(st):
            ts = t[st == s]
            total += ts[rng.integers(0, ts.size, size=(n_boot, ts.size))].sum(axis=1)
        means = total / t.size
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_difference(a: Metric, b: Metric) -> np.ndarray:
    """Per-window a − b, NaN unless both are valid (NaN pairing for comparisons)."""
    if a.terms.size != b.terms.size:
        raise ValueError("metrics must be computed on the same windows")
    return a.terms - b.terms


def wilcoxon(a, b=None) -> dict:
    """Paired Wilcoxon signed-rank test on per-window values (or differences if ``b`` is None).

    Pairs with a NaN on either side are dropped. Zero differences are dropped (scipy's
    default ``zero_method='wilcox'``); if every difference is zero there is no evidence of a
    difference and p = 1. Two-sided.
    """
    from scipy.stats import wilcoxon as _w
    diff = _arr(a) if b is None else _arr(a) - _arr(b)
    diff = diff[np.isfinite(diff)]
    nz = diff[diff != 0]
    if nz.size == 0:
        return dict(statistic=0.0, p=1.0, n=int(diff.size), n_nonzero=0)
    r = _w(nz)
    return dict(statistic=float(r.statistic), p=float(r.pvalue), n=int(diff.size),
                n_nonzero=int(nz.size))


def holm(pvalues) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values for one family of comparisons (same order as input).

    Sorted ascending, p_(i) × (m − i), made monotone, capped at 1. Reject at level α where
    adjusted p < α. Controls the family-wise error rate without independence assumptions.
    """
    p = _arr(pvalues)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def grouped(metric: Metric, groups, n_boot: int = 10_000, seed: int = 0,
            alpha: float = 0.05) -> pd.DataFrame:
    """Pooled headline + per-group (domain) breakdown of one metric, each with a bootstrap CI.

    Pooled = mean over every valid window (each window weighs the same), CI stratified by
    group. Per-group values are the breakdown, not the headline (issue #8).
    """
    t = metric.terms
    g = np.asarray(groups)
    if g.size != t.size:
        raise ValueError("one group label per window")
    rows = []
    for name in pd.unique(g):
        sel = g == name
        ts = t[sel]
        ok = np.isfinite(ts)
        lo, hi = bootstrap_ci(ts, n_boot, alpha, seed)
        rows.append(dict(group=name, value=float(ts[ok].mean()) if ok.any() else np.nan,
                         n=int(ok.sum()), ci_low=lo, ci_high=hi))
    ok = np.isfinite(t)
    lo, hi = bootstrap_ci(t, n_boot, alpha, seed, strata=g)
    rows.append(dict(group="pooled", value=float(t[ok].mean()) if ok.any() else np.nan,
                     n=int(ok.sum()), ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)
