"""P6 paper figures: the manipulation-vs-False-Block trade-off, and the gate distribution.

Both figures are drawn from the saved tables (``python -m src.analyze_p6 test`` writes them),
never recomputed, so a figure always shows the committed numbers:

* ``p6_tradeoff`` — Model 5, ``flipped``, test. Top row: DSS vs FBR; bottom row: Flip Rate vs
  FBR; columns: pooled + the four domains (Amendment 1 asks for per-domain FBR). Series:
  TextGuard over τ (blue, circles), clip over k (orange, squares), TextGuard-L over τ
  (exploratory, aqua, triangles, dashed). The two endpoints are marked and labelled:
  always_trust (FBR 0) and never_trust (FBR 100%, DSS 0). The val-chosen operating points
  carry their 95% bootstrap CI. Identity never rests on colour alone (marker + line style +
  legend + direct labels on the endpoints).
* ``p6_gate_distribution`` — empirical CDF of the gate g on genuine (real) texts vs attacked
  ``flipped`` texts, pooled + per domain, with the 0.5 line: where the genuine curve crosses
  it is the False Block Rate, where the flipped curve crosses it is the share of lies
  blocked. A good gate separates the two curves; overlapping curves mean the gate cannot
  tell a genuine report from its flipped twin.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analyze_p5 import FIGURES  # noqa: E402
from src.run_experiments import RESULTS  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {  # family -> (colour, marker, line style, legend label)
    "TextGuard": ("#2a78d6", "o", "-", "TextGuard (τ sweep)"),
    "clip": ("#eb6834", "s", "-", "clip (k sweep)"),
    "TextGuard-L (exploratory)": ("#1baf7a", "^", "--", "TextGuard-L, exploratory (τ sweep)"),
}
PANELS = ["pooled", "Economy", "SocialGood", "Traffic", "Health_US"]


def _style(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)


def tradeoff(curve: pd.DataFrame, stem: str = "p6_tradeoff"):
    c = curve[(curve.short == "M5") & (curve.attack == "flipped")]
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.4), dpi=200, sharex=True)
    for row, (q, ylab) in enumerate((("dss", "DSS (σ, toward the lie)"),
                                     ("flip", "Flip Rate"))):
        for col, dom in enumerate(PANELS):
            ax = axes[row, col]
            d = c[c.domain == dom]
            ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
            for fam, (color, marker, ls, label) in SERIES.items():
                s = d[d.family == fam].sort_values("fbr")
                ax.plot(100 * s.fbr, s[q], ls=ls, color=color, lw=1.6, marker=marker, ms=4.5,
                        mec="white", mew=0.8, label=label, zorder=3)
                op = s[s.val_chosen]
                if len(op):
                    r = op.iloc[0]
                    ax.errorbar(100 * r.fbr, r[q],
                                xerr=[[100 * (r.fbr - r.fbr_lo)], [100 * (r.fbr_hi - r.fbr)]],
                                yerr=[[r[q] - r[f"{q}_lo"]], [r[f"{q}_hi"] - r[q]]],
                                fmt=marker, color=color, ms=8, mec=INK, mew=1.2,
                                elinewidth=1.2, capsize=0, zorder=4)
            for name, x, ha in (("always_trust", 0, "left"), ("never_trust", 100, "right")):
                e = d[d.family == name].iloc[0]
                ax.plot(100 * e.fbr, e[q], marker="D", color=INK, ms=5.5, zorder=5,
                        ls="none")
                if col == 0:   # labelled once; the legend names them in every panel
                    ax.annotate(name.replace("_", " "), (100 * e.fbr, e[q]), fontsize=6.5,
                                color=INK, xytext=(6 if ha == "left" else -4, 6),
                                textcoords="offset points", ha=ha)
            _style(ax)
            if row == 0:
                ax.set_title(dom, fontsize=9, color=INK)
            if row == 1:
                ax.set_xlabel("False Block Rate on genuine text (%)", fontsize=7.5, color=INK)
                ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            if col == 0:
                ax.set_ylabel(ylab, fontsize=8, color=INK)
            ax.set_xlim(-4, 104)
    h, lab = axes[0, 0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker="D", color=INK, ls="none", ms=5.5))
    lab.append("endpoints (always / never trust)")
    h.append(plt.Line2D([], [], marker="o", color="white", mec=INK, mew=1.2, ms=8, ls="none"))
    lab.append("val-chosen operating point (95% CI)")
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Model 5 (Qwen2.5-1.5B), flipped attack, test: manipulation vs genuine "
                 "text blocked", fontsize=10, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    _save(fig, stem)


def gate_distribution(hist: pd.DataFrame, stem: str = "p6_gate_distribution"):
    h = hist[(hist.variant == "TextGuard")]
    sets = {"genuine (real)": ("#2a78d6", "-", "genuine report"),
            "flipped (attacked)": ("#eb6834", "--", "flipped report (attacked)")}
    fig, axes = plt.subplots(1, 5, figsize=(13, 2.9), dpi=200, sharey=True)
    for ax, dom in zip(axes, PANELS):
        for s, (color, ls, label) in sets.items():
            d = h[(h.set == s) & (h.domain == dom)].sort_values(["bin_lo", "bin_hi"])
            n = d["count"].sum()
            below = d[d.bin_hi > d.bin_lo]
            x = np.r_[0.0, below.bin_hi.to_numpy(), 1.0]
            y = np.r_[0.0, np.cumsum(below["count"].to_numpy()) / n, 1.0]
            ax.step(x, y, where="post", color=color, ls=ls, lw=1.8, label=f"{label}")
            at05 = below[below.bin_hi <= 0.5]["count"].sum() / n
            fake = s.startswith("f")   # fakes labelled left of the line, genuine right
            ax.annotate(f"{100 * at05:.0f}%", (0.5, at05),
                        xytext=(-4, 4) if fake else (4, -10 if at05 >= 0.1 else 3),
                        textcoords="offset points", ha="right" if fake else "left",
                        fontsize=7, color=INK)
        ax.axvline(0.5, color=MUTED, lw=0.8, ls=":")
        _style(ax)
        ns = {s: int(h[(h.set == s) & (h.domain == dom)]["count"].sum()) for s in sets}
        ax.set_title(f"{dom}  (n = {ns['genuine (real)']} / {ns['flipped (attacked)']})",
                     fontsize=8.5, color=INK)
        ax.set_xlabel("gate g", fontsize=7.5, color=INK)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("share of texts with gate ≤ g", fontsize=8, color=INK)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    hd, lb = axes[0].get_legend_handles_labels()
    hd.append(plt.Line2D([], [], color=MUTED, ls=":", lw=0.8))
    lb.append("g = 0.5: genuine curve height = False Block Rate")
    fig.legend(hd, lb, loc="lower center", ncol=3, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("TextGuard gate on test texts (soft gate, w = 3 monthly / 6 weekly): "
                 "genuine vs flipped reports (the jump at g = 1 = texts passed untouched)",
                 fontsize=9.5, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    _save(fig, stem)


def _save(fig, stem: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_figures():
    tradeoff(pd.read_csv(RESULTS / "summary_p6_curve.csv"))
    gate_distribution(pd.read_csv(RESULTS / "summary_p6_gate_hist.csv"))


if __name__ == "__main__":
    make_figures()
