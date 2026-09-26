"""Standalone middle panel for the Canva teaser.

Portrait, transparent background, 400 dpi, sized to drop into the existing three-panel
layout in place of the old "CAD Bias Geometry Diagnostic" panel.

Content is verbatim from results/sadi_results.json (SADI s=10) and
results/judge_scores.json. Nothing here is illustrative.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUT = Path(__file__).parent / "paper_arr2026" / "figures" / "teaser-panel2.png"

INK, MUTED, RULE = "#141414", "#5f5f5f", "#cfcfcf"
BAD, GOOD, WARN = "#C43E1C", "#0B7A5D", "#B26A00"
TINT = "#fdf1ec"

plt.rcParams.update({"pdf.fonttype": 42})


def main() -> None:
    fig = plt.figure(figsize=(2.75, 4.35))
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.035, 0.012), 0.93, 0.872,
                                boxstyle="round,pad=0.012,rounding_size=0.035",
                                fc=TINT, ec=BAD, lw=2.0))

    ax.text(0.5, 0.955, "After steering", fontsize=11.5, weight="bold",
            color=INK, ha="center")
    ax.text(0.5, 0.910, "SADI, published setting", fontsize=8.6,
            color=MUTED, ha="center")

    ax.text(0.075, 0.845,
            "Booking a hotel that meets\n"
            "accessibility needs can be\n"
            "challenging,HeaderCode__ but\n"
            "thereHeaderCode__HeaderCode__\n"
            "HeaderCode__there are steps\n"
            "you can take to_TypeInfo__\n"
            "simplify the process.",
            family="monospace", fontsize=7.4, color=INK, va="top", linespacing=1.72)

    ax.plot([0.075, 0.925], [0.560, 0.560], color=RULE, lw=1.0)

    ax.text(0.5, 0.505, "medicalization score", fontsize=8.4, color=MUTED, ha="center")
    ax.text(0.5, 0.405, "0.00", fontsize=34, color=GOOD, weight="bold",
            ha="center", va="baseline")
    ax.text(0.5, 0.358, "the metric's value for unbiased text",
            fontsize=7.2, color=MUTED, ha="center", style="italic")

    ax.plot([0.235, 0.765], [0.322, 0.322], color=RULE, lw=1.0)

    ax.text(0.5, 0.268, "judged response quality", fontsize=8.4, color=MUTED, ha="center")
    ax.text(0.5, 0.168, "0 / 10", fontsize=34, color=BAD, weight="bold",
            ha="center", va="baseline")

    ax.text(0.5, 0.128, "201 of 250 responses score exactly 0.00.",
            fontsize=7.4, color=INK, ha="center", weight="bold")
    ax.text(0.5, 0.094, "None is empty; they average 188 words.",
            fontsize=7.4, color=INK, ha="center")

    ax.text(0.5, 0.044, "The metric cannot tell",
            fontsize=7.9, color=WARN, ha="center", style="italic")
    ax.text(0.5, 0.020, "debiasing from destruction.",
            fontsize=7.9, color=WARN, ha="center", style="italic")

    fig.savefig(OUT, dpi=400, transparent=True)
    fig.savefig(OUT.with_suffix(".pdf"), transparent=True)
    plt.close(fig)
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
