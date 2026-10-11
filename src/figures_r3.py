"""R3 paper figure: pooled DSS under each adaptive attack, undefended vs each frozen defence.

Read from ``results/summary_r3_operating.csv`` (``python -m src.analyze_r3`` writes it), so the
figure is always the saved numbers.

Form: small multiples, one panel per target (M5, Qwen2.5-7B, M3-F, M4-F) sharing one y-axis
(DSS in σ, pooled over the 4 domains, Amendment 1's attacked windows). In each panel, the three
attacks on x and one dot per defence with its 95% bootstrap CI. Defences keep the same hue and
marker in every panel (none = blue circle, TextGuard v1 = orange square, clip = aqua diamond,
skeptical prompt = yellow triangle; validated palette, CVD ΔE ≥ 9.1), so identity never depends
on colour alone; the skeptical prompt exists only for the LLM targets. A dashed line marks the
SESOI (0.05σ). Verdicts are not drawn — they come from the Holm family table, not from a CI.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.analyze_p5 import FIGURES  # noqa: E402
from src.analyze_r2 import SESOI_DSS  # noqa: E402
from src.analyze_r3 import ATTACKS, SHORT, TARGETS  # noqa: E402

SERIES = {"none": ("#2a78d6", "o", "Undefended"),
          "TG": ("#eb6834", "s", "TextGuard v1 (frozen)"),
          "CL": ("#1baf7a", "D", "Clip k = 0.5 (frozen)"),
          "SK": ("#eda100", "^", "Skeptical prompt")}
TICK = {"a1_selective": "A1\nselective flip", "a2_forward": "A2\nforward lie",
        "a3_magnitude": "A3\n×10 magnitude"}
TITLE = {"M5": "M5 (Qwen2.5-1.5B)", "Q7": "Qwen2.5-7B (4-bit)", "DLinear+FinBERT": "M3-F (DLinear+FinBERT)",
         "PatchTST+FinBERT": "M4-F (PatchTST+FinBERT)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
YLIM = (-0.3, 0.8)   # one copied-number window stretches an SK whisker to -1.37 (marked, see caption)


def make_figure(op: pd.DataFrame | None = None, stem: str = "r3_dss") -> None:
    if op is None:
        from src.run_experiments import RESULTS
        op = pd.read_csv(RESULTS / "summary_r3_operating.csv")
    op = op[(op.cell == "pooled") & op.attack.isin(ATTACKS)]
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(11.5, 3.4), dpi=200, sharey=True)
    for ax, tgt in zip(axes, TARGETS):
        a = op[op.target == tgt].set_index(["defence", "attack"])
        defs = [d for d in SERIES if d in a.index.get_level_values(0)]
        for j, dfn in enumerate(defs):
            color, marker, label = SERIES[dfn]
            r = a.xs(dfn, level="defence").loc[list(ATTACKS)]
            off = (j - (len(defs) - 1) / 2) * 0.17
            xs = [i + off for i in range(len(ATTACKS))]
            lo = r.dss_lo.clip(lower=YLIM[0])          # a whisker past the axis is cut and marked
            ax.errorbar(xs, r.dss, yerr=[r.dss - lo, r.dss_hi - r.dss], fmt=marker, color=color,
                        ms=5, mec="white", mew=0.8, elinewidth=1.4, capsize=0, label=label, zorder=3)
            for xx, l0 in zip(xs, r.dss_lo):
                if l0 < YLIM[0]:
                    ax.annotate(f"CI to {l0:.2f}", (xx, YLIM[0]), xytext=(4, 6), textcoords="offset points",
                                fontsize=6, color=MUTED, ha="left", va="bottom")
                    ax.plot([xx], [YLIM[0]], marker="v", color=color, ms=4, zorder=4)
        ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
        ax.axhline(SESOI_DSS, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.set_xticks(range(len(ATTACKS)))
        ax.set_xticklabels([TICK[x] for x in ATTACKS], fontsize=7, color=INK)
        ax.set_title(TITLE[tgt], fontsize=8.5, color=INK, loc="left")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=7)
    axes[0].set_ylim(*YLIM)
    axes[0].set_ylabel("DSS, pooled (σ units)\n> 0 = pulled toward the claim", fontsize=8, color=INK)
    axes[0].text(0.01, SESOI_DSS + 0.02, "SESOI 0.05σ", transform=axes[0].get_yaxis_transform(),
                 fontsize=6.5, color=MUTED, ha="left", va="bottom")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(1.0, 1.04))
    fig.text(0.01, -0.03, "Test split, attacked windows after Amendment 1 (A1 n = 42, A2 n = 253, A3 n = 98 "
             "before NaN pairing). Whiskers: 95% bootstrap CI (stratified by domain). For A3 the claim "
             "is the real direction, so DSS > 0 = pushed further in it.\n"
             "M5 skeptical prompt, A2: the mean (−0.11) and the long whisker come from one window where "
             "the model copied the report's 2,730.6 billion; without it +0.48.", fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
