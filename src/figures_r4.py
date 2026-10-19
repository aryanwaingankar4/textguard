"""R4 paper figures: DSS under each attack and the False Block Rate, per defence, held-out domains.

Read from ``results/summary_r4_operating.csv`` and ``summary_r4_fbr.csv`` (``python -m
src.analyze_r4`` writes them), so the figures are always the saved numbers.

* ``r4_dss`` — small multiples: rows = target (M5, Qwen2.5-7B), columns = cell (Energy, FNSPID,
  pooled), one shared y-axis (DSS in σ). In each panel the attacks on x (A3 on Energy only) and one
  dot per defence with its 95% bootstrap CI. Dashed line = the SESOI (0.05σ).
* ``r4_fbr`` — False Block Rate on genuine reports per defence, Energy / FNSPID / pooled, one panel
  per target, with the 10% budget as a dashed line.

Defences keep R3's hue and marker (undefended blue circle, TextGuard v1 orange square, clip aqua
diamond) plus TextGuard v2 = violet down-triangle and anchor15 = yellow triangle; the 5 hues pass
the dataviz validator (CVD ΔE ≥ 9.1; aqua / yellow are below 3:1 contrast, so every series also has
its own marker and a legend). Verdicts are not drawn — they come from the Holm family table.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analyze_p5 import FIGURES  # noqa: E402
from src.analyze_r2 import SESOI_DSS  # noqa: E402
from src.analyze_r4 import ATTACKS, FBR_BUDGET, TARGETS  # noqa: E402
from src.run_experiments import RESULTS  # noqa: E402

SERIES = {"none": ("#2a78d6", "o", "Undefended"),
          "TG1": ("#eb6834", "s", "TextGuard v1 (frozen)"),
          "TG2": ("#4a3aa7", "v", "TextGuard v2 (frozen)"),
          "CL": ("#1baf7a", "D", "Clip k = 0.5"),
          "anchor15": ("#eda100", "^", "anchor15 (±15% of last value)")}
TICK = {"flipped": "flip", "a1_selective": "A1\nselective", "a2_forward": "A2\nforward",
        "a3_magnitude": "A3\n×10"}
TITLE = {"M5": "M5 (Qwen2.5-1.5B)", "Q7": "Qwen2.5-7B (4-bit)"}
CELLS = ("Energy", "FNSPID", "pooled")
CELL_TITLE = {"Energy": "Energy (easy case)", "FNSPID": "FNSPID (hard case)", "pooled": "pooled"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax) -> None:
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)


def dss_figure(op: pd.DataFrame | None = None, stem: str = "r4_dss") -> None:
    op = pd.read_csv(RESULTS / "summary_r4_operating.csv") if op is None else op
    fig, axes = plt.subplots(len(TARGETS), len(CELLS), figsize=(11.5, 5.6), dpi=200, sharey=True)
    for r, tgt in enumerate(TARGETS):
        for c, cell in enumerate(CELLS):
            ax = axes[r, c]
            a = op[(op.target == tgt) & (op.cell == cell)].set_index(["defence", "attack"])
            atk = [x for x in ATTACKS if (("none", x) in a.index)]
            for j, dfn in enumerate(SERIES):
                color, marker, label = SERIES[dfn]
                rr = a.xs(dfn, level="defence").loc[atk]
                off = (j - (len(SERIES) - 1) / 2) * 0.14
                xs = np.arange(len(atk)) + off
                ax.errorbar(xs, rr.dss, yerr=[rr.dss - rr.dss_lo, rr.dss_hi - rr.dss], fmt=marker, color=color,
                            ms=4.5, mec="white", mew=0.7, elinewidth=1.3, capsize=0, label=label, zorder=3)
            ax.axhline(SESOI_DSS, color=MUTED, lw=0.8, ls="--", zorder=1)
            _style(ax)
            ax.set_xticks(range(len(atk)), [TICK[x] for x in atk])
            if r == 0:
                ax.set_title(CELL_TITLE[cell], fontsize=8.5, color=INK)
            if c == 0:
                ax.set_ylabel(f"{TITLE[tgt]}\nDSS (σ), toward the lie", fontsize=7.5, color=INK)
    axes[0, -1].text(1.0, SESOI_DSS, " SESOI", transform=axes[0, -1].get_yaxis_transform(), fontsize=6.5,
                     color=MUTED, va="center", ha="left")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def fbr_figure(fb: pd.DataFrame | None = None, stem: str = "r4_fbr") -> None:
    fb = pd.read_csv(RESULTS / "summary_r4_fbr.csv") if fb is None else fb
    defs = ["TG1", "TG2", "CL", "anchor15"]
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(9.0, 2.9), dpi=200, sharey=True)
    for ax, tgt in zip(axes, TARGETS):
        a = fb[(fb.target == tgt)].set_index(["defence", "cell"])
        for j, dfn in enumerate(defs):
            color, marker, label = SERIES[dfn]
            rr = a.xs(dfn, level="defence").loc[list(CELLS)]
            xs = np.arange(len(CELLS)) + (j - (len(defs) - 1) / 2) * 0.17
            ax.errorbar(xs, 100 * rr.fbr, yerr=[100 * (rr.fbr - rr.ci_low), 100 * (rr.ci_high - rr.fbr)], fmt=marker,
                        color=color, ms=5, mec="white", mew=0.7, elinewidth=1.3, capsize=0, label=label, zorder=3)
        ax.axhline(100 * FBR_BUDGET, color=MUTED, lw=0.8, ls="--", zorder=1)
        ax.text(1.0, 100 * FBR_BUDGET, " 10% budget", transform=ax.get_yaxis_transform(), fontsize=6.5,
                color=MUTED, va="center", ha="left")
        _style(ax)
        ax.set_xticks(range(len(CELLS)), list(CELLS))
        ax.set_title(TITLE[tgt], fontsize=8.5, color=INK)
    axes[0].set_ylabel("genuine reports blocked (%)", fontsize=7.5, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    dss_figure()
    fbr_figure()
    print(f"wrote {FIGURES}/r4_dss.* and r4_fbr.*")
