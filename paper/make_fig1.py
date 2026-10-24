"""Fig. 1 of the paper: the TextGuard pipeline, with the attack path marked.

Drawn with matplotlib boxes and arrows (no data). Run with the project venv:
    .venv\\Scripts\\python paper\\make_fig1.py
Writes paper/figures/fig1_pipeline.png (300 dpi). Why a script and not a drawing tool: the figure
regenerates from the repo like every other figure, and the labels can be checked against the gate
definition (CLAUDE.md §8) by reading this file.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures" / "fig1_pipeline.png"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 7})

INK, GREY, BLUE, RED, GREEN = "#1f1f1f", "#f2f2f2", "#dce8f7", "#c0392b", "#e3f2e6"


def box(ax, x, y, w, h, text, fc=GREY, ec=INK, ls="-", weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.03",
                                fc=fc, ec=ec, lw=0.8, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=INK, weight=weight,
            linespacing=1.3)


def arrow(ax, p, q, color=INK, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.8, color=color,
                                 ls=ls, shrinkA=0, shrinkB=0))


def line(ax, xs, ys, color=INK, ls="-"):
    ax.plot(xs, ys, color=color, lw=0.8, ls=ls, solid_capstyle="butt")


fig, ax = plt.subplots(figsize=(7.2, 2.9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.0)
ax.axis("off")

H = 0.8                       # box height, top row
Y1 = 2.55                     # top row (text path)
# --- attack -------------------------------------------------------------------------------
box(ax, 0.05, 3.62, 1.6, 0.34, "Adversary", fc="#fbe3e0", ec=RED, ls="--", weight="bold")
arrow(ax, (0.85, 3.62), (0.85, Y1 + H), color=RED)
ax.text(1.8, 3.79, "Attack path: the edited report (flip, A1 selective, A2 forward sentence, "
        "A3 ×10) goes straight into B;\nTextGuard can only see it through the judge.",
        ha="left", va="center", color=RED, style="italic", linespacing=1.3)

# --- text path (top row) --------------------------------------------------------------------
cols = [(0.05, 1.6, "Report text\n(genuine, or edited\nby the adversary)", BLUE),
        (2.0, 1.6, "Split into claims\nv1: whole report\nv2: each sentence", GREY),
        (3.95, 1.6, "LLM judge\n(Qwen2.5-1.5B)\ndirection d, conf. c", GREY),
        (5.9, 1.75, "Evidence check:\nclaim vs. numbers\n(trend / level / YoY) → s", GREY),
        (8.0, 1.95, "Gate per claim:\n1 if it agrees, else 1 − c·s;\ng = min over claims",
         GREEN)]
for i, (x, w, t, fc) in enumerate(cols):
    box(ax, x, Y1, w, H, t, fc=fc)
    if i:
        px, pw = cols[i - 1][0], cols[i - 1][1]
        arrow(ax, (px + pw, Y1 + H / 2), (x, Y1 + H / 2))

# --- forecaster path ------------------------------------------------------------------------
box(ax, 0.05, 0.1, 1.6, 0.7, "Numerical history\n(lookback window)", fc=BLUE)
box(ax, 2.0, 1.25, 2.6, 0.55, "B = predict(numbers, text)")
box(ax, 2.0, 0.17, 2.6, 0.55, "A = predict(numbers, None)")
box(ax, 6.9, 0.45, 3.05, 1.05, "Output = A + g·(B − A)\nv2 also caps the change at ±kσ;"
    "\nwarning when g < 0.5", fc=GREEN, weight="bold")

# report -> B (down, then right)
line(ax, [0.85, 0.85], [Y1, 1.525])
arrow(ax, (0.85, 1.525), (2.0, 1.525))
# numbers -> A and -> B
arrow(ax, (1.65, 0.445), (2.0, 0.445))
line(ax, [1.8, 1.8], [0.445, 1.4])
arrow(ax, (1.8, 1.4), (2.0, 1.4))
# A, B -> output
arrow(ax, (4.6, 1.525), (6.9, 1.2))
arrow(ax, (4.6, 0.445), (6.9, 0.75))
# gate -> output
arrow(ax, (8.975, Y1), (8.975, 1.5))
# numbers -> evidence (dashed, along the bottom then up)
line(ax, [1.0, 1.0], [0.1, 0.03], ls=(0, (3, 2)))
line(ax, [1.0, 6.3], [0.03, 0.03], ls=(0, (3, 2)))
line(ax, [6.3, 6.3], [0.03, 0.3], ls=(0, (3, 2)))
arrow(ax, (6.3, 1.6), (6.3, Y1), ls=(0, (3, 2)))
line(ax, [6.3, 6.3], [0.3, 1.6], ls=(0, (3, 2)))
ax.text(6.38, 2.05, "numbers (v2: plus\nlonger history)", ha="left", va="center", color="#444",
        fontsize=6.5)
ax.text(5.0, 1.03, "black-box: only predict() is called", ha="center", va="center",
        color="#444", fontsize=6.5, style="italic")

fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.03, metadata={"Software": None})
print(OUT)
