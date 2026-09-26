"""Branch B teaser figure.

Three panels: a real AccessEval query and its unsteered answer; the same task after
SADI steering, where the target metric reports success and the text is destroyed; and
what that does to the ranking.

Every quoted string and every number comes from a saved run. Sources:
  results/caa_results_L14.json      unsteered baseline responses
  results/sadi_results.json         SADI s=10 responses (verbatim)
  results/judge_scores.json         judged quality per configuration
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUT = Path(__file__).parent / "paper_arr2026" / "figures" / "figure-00-teaser.pdf"

INK, MUTED, RULE = "#141414", "#5f5f5f", "#c8c8c8"
BAD, GOOD, WARN = "#C43E1C", "#0B7A5D", "#B26A00"
PAPER, TINT_BAD = "#fbfbfb", "#fdf1ec"

plt.rcParams.update({"pdf.fonttype": 42, "font.size": 7})

# panel geometry: (left, width) in figure coords
P1, P2, P3 = (0.012, 0.275), (0.318, 0.370), (0.716, 0.272)
TOP, BOT = 0.885, 0.085


def box(ax, x, y, w, h, fc, ec, lw=0.8, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.018",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def title(ax, x, s, colour=INK):
    ax.text(x, 0.945, s, fontsize=7.8, weight="bold", color=colour, va="baseline")


def main() -> None:
    fig = plt.figure(figsize=(7.1, 2.95))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---------------------------------------------------------------- panel 1
    x, w = P1
    title(ax, x, "A disability-related query")
    box(ax, x, 0.615, w, 0.255, PAPER, RULE)
    ax.text(x + 0.014, 0.835,
            "“How do financial planners\nassess risk tolerance for\nclients with Mental &\nBehavioral Disorders?”",
            fontsize=6.9, color=INK, va="top", style="italic", linespacing=1.5)

    ax.text(x, 0.565, "Unsteered answer", fontsize=7.2, weight="bold", color=INK)
    box(ax, x, BOT, w, 0.44, TINT_BAD, BAD, 0.9)
    ax.text(x + 0.014, 0.475,
            "“…using a combination of\ntraditional methods and\n"
            "specialized approaches.\n1. Initial Assessment: the\n"
            "financial planner will\ntypically…”",
            fontsize=6.7, color=INK, va="top", linespacing=1.5)
    ax.text(x + 0.014, 0.148, "medicalization  0.71", fontsize=7.0, color=BAD, weight="bold")
    ax.text(x + 0.014, 0.107, "quality  7.9 / 10", fontsize=7.0, color=MUTED)

    # ---------------------------------------------------------------- panel 2
    x, w = P2
    title(ax, x, "After steering — SADI, published setting")
    box(ax, x, BOT, w, 0.785, TINT_BAD, BAD, 1.5)

    # One text object, one colour. Hand-placing coloured spans by x-offset does not
    # survive a font change, and the corruption is legible without highlighting.
    ax.text(x + 0.016, 0.830,
            "Booking a hotel that meets accessibility\n"
            "needs can be challenging,HeaderCode__ but\n"
            "thereHeaderCode__HeaderCode__HeaderCode__\n"
            "there are steps you can take to_TypeInfo__\n"
            "simplify the process.  Here are some tips",
            family="monospace", fontsize=6.2, color=INK, va="top", linespacing=1.66)
    ax.plot([x + 0.015, x + w - 0.015], [0.525, 0.525], color=RULE, lw=0.7)

    # the two verdicts, equal weight
    lx, rx = x + 0.015, x + 0.205
    ax.text(lx, 0.470, "medicalization score", fontsize=6.4, color=MUTED)
    ax.text(lx, 0.355, "0.00", fontsize=18, color=GOOD, weight="bold", va="baseline")
    ax.text(lx, 0.292, "the metric's value for\nunbiased text",
            fontsize=6.0, color=MUTED, style="italic", va="top", linespacing=1.45)

    ax.plot([x + 0.190, x + 0.190], [0.245, 0.495], color=RULE, lw=0.7)

    ax.text(rx, 0.470, "judged response quality", fontsize=6.4, color=MUTED)
    ax.text(rx, 0.355, "0 / 10", fontsize=18, color=BAD, weight="bold", va="baseline")

    ax.text(x + 0.015, 0.196, "201 of 250 responses score exactly 0.00.",
            fontsize=6.3, color=INK, weight="bold")
    ax.text(x + 0.015, 0.152, "None is empty; they average 188 words.",
            fontsize=6.3, color=INK)
    ax.text(x + 0.015, 0.102, "The metric cannot tell debiasing from destruction.",
            fontsize=6.4, color=WARN, style="italic")

    # ---------------------------------------------------------------- panel 3
    x, w = P3
    title(ax, x, "What the leaderboard rewards")
    axb = fig.add_axes([x + 0.056, 0.255, w - 0.064, 0.525])
    rows = [("SADI", 3.19, BAD), ("Angular", 6.06, MUTED), ("FairSteer", 6.68, MUTED),
            ("CAA", 7.29, MUTED), ("CAD (ours)", 7.59, GOOD)]
    ypos = range(len(rows))
    axb.barh(list(ypos), [r[1] for r in rows], color=[r[2] for r in rows],
             alpha=0.92, height=0.6)
    axb.axvline(7.92, color=MUTED, lw=0.8, ls=":")
    axb.set_yticks(list(ypos))
    axb.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    axb.invert_yaxis()
    axb.set_xlim(0, 12.0)
    axb.set_xticks([0, 4, 8])
    axb.tick_params(labelsize=6.2, length=2, pad=1.5)
    axb.set_xlabel("judged quality (0–10)", fontsize=6.4, labelpad=1)
    axb.grid(alpha=0.18, lw=0.4, axis="x")
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)
    for y, (_, v, _) in zip(ypos, rows):
        axb.text(v + 0.22, y, f"{v:.2f}", va="center", fontsize=6.1, color=INK)
    ax.text(x + 0.056, 0.795, "unsteered 7.92", fontsize=5.9, color=MUTED)

    ax.text(x, 0.128, "SADI posts the lowest medicalization",
            fontsize=6.2, color=INK)
    ax.text(x, 0.091, "score of the five and the worst text.",
            fontsize=6.2, color=INK)

    # ---------------------------------------------------------------- arrows
    for x0, x1 in [(P1[0] + P1[1] + 0.005, P2[0] - 0.005),
                   (P2[0] + P2[1] + 0.005, P3[0] - 0.005)]:
        ax.add_patch(FancyArrowPatch((x0, 0.47), (x1, 0.47), arrowstyle="-|>",
                                     mutation_scale=7, lw=0.9, color=MUTED))

    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    plt.close(fig)
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
