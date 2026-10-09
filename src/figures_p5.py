"""P5 paper figures: pooled DSS and Flip Rate per model, with 95% bootstrap CIs.

Read from ``results/summary_p5_attack.csv`` (``python -m src.analyze_p5`` writes it), so
the figures are always the saved numbers — never recomputed separately.

Form: one dot per model × attack with its CI as a whisker, models on the x-axis in
fusion order (no text → late → early). Two series (flipped, exaggerated) get the first two
slots of the validated categorical palette (blue, orange) **and** different marker shapes,
so identity never depends on colour alone (print, colour-blind readers). Models 1–2 are
plotted at exactly 0 — they are asserted text-blind, which is part of the result.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.analyze_p5 import ATTACKS, FIGURES, MODELS, SHORT  # noqa: E402

SERIES = {"flipped": ("#2a78d6", "o", "Flipped (direction reversed)"),
          "exaggerated": ("#eb6834", "s", "Exaggerated (change ×3)")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FUSION_LABEL = {"DLinear": "no text", "PatchTST": "no text", "DLinear+BERT": "late",
                "PatchTST+BERT": "late", "Qwen2.5-1.5B": "early"}


def _axes(ax, ylabel: str):
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([f"{SHORT[m]}\n{m}\n({FUSION_LABEL[m]})" for m in MODELS],
                       fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)


def _plot(att: pd.DataFrame, value: str, lo: str, hi: str, ylabel: str, title: str,
          path_stem: str, fmt: str):
    pooled = att[att.domain == "pooled"].set_index(["model", "condition"])
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    for j, a in enumerate(ATTACKS):
        color, marker, label = SERIES[a]
        xs = [i + (j - 0.5) * 0.22 for i in range(len(MODELS))]
        r = pooled.xs(a, level="condition").loc[list(MODELS)]
        ax.errorbar(xs, r[value], yerr=[r[value] - r[lo], r[hi] - r[value]], fmt=marker,
                    color=color, ms=6, mec="white", mew=1.0, elinewidth=1.6, capsize=0,
                    label=label, zorder=3)
        # direct label on the largest effect only (selective labelling)
        k = r[value].abs().values.argmax()
        v = r[value].iloc[k]
        if abs(v) > 0.02:   # beside the point, clear of the whisker and the axes
            ax.annotate(fmt.format(v), (xs[k], v), xytext=(-7, 0),
                        textcoords="offset points", ha="right", va="center", fontsize=8,
                        color=INK)
    _axes(ax, ylabel)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    fig.text(0.01, 0.005, "Test split, pooled over 4 domains, attacked windows only. "
             "Whiskers: 95% domain-stratified bootstrap CI.", fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{path_stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def make_figures(att: pd.DataFrame | None = None) -> None:
    if att is None:
        from src.analyze_p5 import RESULTS
        att = pd.read_csv(RESULTS / "summary_p5_attack.csv")
    _plot(att, "dss", "dss_ci_low", "dss_ci_high", "DSS (σ units; > 0 = toward the claim)",
          "Directional Shift Score under fake text", "p5_dss", "{:+.2f}")
    att = att.assign(flip_pct=100 * att.flip_rate, flo=100 * att.flip_ci_low,
                     fhi=100 * att.flip_ci_high)
    _plot(att, "flip_pct", "flo", "fhi", "Flip Rate (% of attacked windows)",
          "Forecast direction turned toward the fake claim", "p5_flip_rate", "{:.0f}%")


if __name__ == "__main__":
    make_figures()
