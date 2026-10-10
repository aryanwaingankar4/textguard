"""R2 paper figure: DSS of Model 5, its prompt variants, Phi-3.5 and Qwen2.5-7B.

Read from ``results/summary_r2_attack.csv`` (``python -m src.analyze_r2`` writes it), so the
figure is always the saved numbers.

Form: small multiples — pooled first, then the four domains — sharing one y-axis (DSS in σ).
In each panel one dot per forecaster and attack with its 95% bootstrap CI as a whisker.
Same two series as the P5 figure (flipped = blue circle, exaggerated = orange square; the pair
passes the palette validator, CVD ΔE 24.7), so identity never depends on colour alone. A
dashed line marks the pre-registered SESOI (0.05σ): "fooled" needs a significant DSS at or
above it (§10e). Verdicts are not drawn — they come from the Holm family table, not from a CI.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.analyze_p5 import FIGURES  # noqa: E402
from src.analyze_r2 import FAKES, M5, SESOI_DSS  # noqa: E402
from src.models.train_baselines import DOMAINS  # noqa: E402

ORDER = (M5, "M5-c1", "M5-c2", "M5-sk", "P35", "Q7")
# short codes; the paper caption spells them out (full names collide at this width)
TICK = {M5: "M5", "M5-c1": "c1", "M5-c2": "c2", "M5-sk": "sk", "P35": "Phi", "Q7": "Q7B"}
SERIES = {"flipped": ("#2a78d6", "o", "Flipped (direction reversed)"),
          "exaggerated": ("#eb6834", "s", "Exaggerated (change ×3)")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def make_figure(att: pd.DataFrame | None = None, stem: str = "r2_dss") -> None:
    if att is None:
        from src.run_experiments import RESULTS
        att = pd.read_csv(RESULTS / "summary_r2_attack.csv")
    panels = ["pooled", *DOMAINS]
    fig, axes = plt.subplots(1, len(panels), figsize=(11.5, 3.4), dpi=200, sharey=True,
                             gridspec_kw=dict(width_ratios=[1.25] + [1] * len(DOMAINS)))
    for ax, dom in zip(axes, panels):
        a = att[att.domain == dom].set_index(["model", "condition"])
        for j, cond in enumerate(FAKES):
            color, marker, label = SERIES[cond]
            r = a.xs(cond, level="condition").loc[list(ORDER)]
            xs = [i + (j - 0.5) * 0.28 for i in range(len(ORDER))]
            ax.errorbar(xs, r.dss, yerr=[r.dss - r.dss_ci_low, r.dss_ci_high - r.dss],
                        fmt=marker, color=color, ms=4.5 if dom != "pooled" else 5.5,
                        mec="white", mew=0.8, elinewidth=1.4, capsize=0, label=label, zorder=3)
        ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
        ax.axhline(SESOI_DSS, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.axvline(3.5, color=GRID, lw=0.8, zorder=0)      # M5 prompts | other models
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([TICK[m] for m in ORDER], fontsize=7, color=INK)
        ax.set_title("Pooled (4 domains)" if dom == "pooled" else dom, fontsize=8.5, color=INK,
                     loc="left", fontweight="bold" if dom == "pooled" else "normal")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=7)
    axes[0].set_ylabel("DSS (σ units)\n> 0 = pulled toward the claim", fontsize=8, color=INK)
    axes[0].text(0.99, SESOI_DSS + 0.02, "SESOI 0.05σ", transform=axes[0].get_yaxis_transform(),
                 fontsize=6.5, color=MUTED, ha="right", va="bottom")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=2, frameon=False, fontsize=7.5,
               bbox_to_anchor=(1.0, 1.03))
    fig.text(0.01, -0.02, "M5 = Model 5 (P5 prompt); c1 = report first; c2 = reworded; "
             "sk = skeptical (M5, same weights) | Phi = Phi-3.5-mini; Q7B = Qwen2.5-7B (4-bit). "
             "Whiskers: 95% bootstrap CI.", fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
