"""Contrast-set controls figure for the ARR draft.

Replots controls_results/day3_subsample_results.json at ACL two-column width with a
single shared legend below both panels, so neither plot area is obscured.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTS = Path(__file__).parent / "controls_results" / "day3_subsample_results.json"
OUT = Path(__file__).parent / "paper_arr2026" / "figures" / "figure-03-contrast-set-controls.pdf"

# EVR_1 and k@80% reported for DiscrimEval Target A in the earlier version of this work
TARGET_A = {"evr1": 67.9, "k": 2}

SERIES = [
    ("L21_pair",    "L21, pair-level",    "#0072B2", "-",  "o"),
    ("L21_cluster", "L21, cluster-level", "#0072B2", "--", "s"),
    ("L0_pair",     "L0, pair-level",     "#D55E00", "-",  "o"),
    ("L0_cluster",  "L0, cluster-level",  "#D55E00", "--", "s"),
    ("L21_null",    "L21, permuted null", "#7f7f7f", ":",  "^"),
]

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def main() -> None:
    res = json.loads(RESULTS.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85))

    for key, label, colour, ls, marker in SERIES:
        rows = res["curves"].get(key, [])
        if not rows:
            continue
        x = [r["n_actual_mean"] for r in rows]
        for ax, mean_k, lo_k, hi_k, scale in (
            (axes[0], "evr1_mean", "evr1_lo", "evr1_hi", 100.0),
            (axes[1], "k_mean", "k_lo", "k_hi", 1.0),
        ):
            ax.plot(x, [r[mean_k] * scale for r in rows], marker=marker, color=colour,
                    ls=ls, lw=1.5, ms=3.4, label=label if ax is axes[0] else "_nolegend_")
            if lo_k in rows[0]:
                ax.fill_between(x, [r[lo_k] * scale for r in rows],
                                [r[hi_k] * scale for r in rows],
                                color=colour, alpha=0.11, lw=0)

    for ax, ref, note in (
        (axes[0], TARGET_A["evr1"], "Target A, previously reported: 67.9%"),
        (axes[1], TARGET_A["k"], "Target A, previously reported: $k{=}2$"),
    ):
        ax.axhline(ref, color="#1a1a1a", ls="-.", lw=1.0, zorder=1)

    # place the reference annotations in empty regions of each panel
    axes[0].annotate("Target A, previously\nreported: $\\mathrm{EVR}_1{=}67.9\\%$",
                     xy=(430, TARGET_A["evr1"]), xytext=(150, 47), fontsize=6.4,
                     color="#1a1a1a", linespacing=1.4,
                     arrowprops=dict(arrowstyle="-", lw=0.6, color="#666666", shrinkB=1))
    # one line, sitting just above the reference line in the empty band at the right,
    # so it crosses no data line (an arrowed two-line label overlapped the cluster curve)
    axes[1].text(1000, TARGET_A["k"] + 0.8, "Target A, previously reported: $k{=}2$",
                 fontsize=6.4, color="#1a1a1a", ha="right", va="bottom")

    axes[0].set_ylabel("$\\mathrm{EVR}_1$ (\\%)" if False else "EVR$_1$ (%)")
    axes[1].set_ylabel("$k$@80%")
    axes[0].set_title("Spectral concentration", loc="left", fontsize=9)
    axes[1].set_title("Prescribed intervention rank", loc="left", fontsize=9)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("number of contrastive pairs ($n$)")
        ax.grid(alpha=0.22, lw=0.45, ls=":")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               fontsize=6.9, bbox_to_anchor=(0.5, -0.045), handlelength=2.2,
               columnspacing=1.4, handletextpad=0.5)

    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
