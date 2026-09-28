"""Figure for the pre-registered frozen-set test (E1 + A4).

Reads analysis-output/e1/e1_scores.json (written by e1_analysis.py) and writes
paper_arr2026/figures/figure-02-e1-frozen-set.pdf. CPU only.

    python e1_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCORES = Path("analysis-output/e1/e1_scores.json")
OUT = Path("paper_arr2026/figures/figure-02-e1-frozen-set.pdf")

# Okabe-Ito
BLUE, SKY, ORANGE, GREEN, VERM, PURPLE, GREY, BLACK = (
    "#0072B2", "#56B4E9", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#999999", "#000000")

LINES = [  # (label, arms in order, colour, marker). LEACE and additive CAD are in the appendix table.
    ("CAD projection, rank 1 to 80", [f"cad_proj_k{k}" for k in (1, 5, 10, 20, 40, 80)], BLUE, "o"),
]
RANK_LABELS = {"cad_proj_k1": ("1", (0, -9)), "cad_proj_k40": ("40", (-13, -9)),
               "cad_proj_k80": ("80", (5, -9))}
POINTS = [  # (label, arm, colour, marker, size)
    ("CAA", "caa", ORANGE, "o", 38),
    ("FairSteer", "fairsteer", GREEN, "o", 38),
    ("SADI", "sadi", VERM, "o", 38),
    ("Angular", "angular", PURPLE, "o", 38),
    ("Prompt (minimal)", "prompt_min", BLACK, "x", 34),
    ("Prompt (explicit)", "prompt_explicit", BLACK, "P", 38),
    ("Prompt + CAD projection", "prompt_explicit_cad_k40", BLUE, "*", 120),
]


def main() -> int:
    """Two panels, one per judge. Vertical bars: 95% intervals over the 50 questions. Intervals on
    d are in the paper's table, not drawn, to keep the dense region readable."""
    S = json.loads(SCORES.read_text())["summary"]
    plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
                         "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.25), sharex=True)
    for ax, judge, name in ((axes[0], "llama", "Llama-3.1-8B judge"),
                            (axes[1], "qwen", "Qwen2.5-7B judge")):
        ax.axhline(0, color=GREY, lw=0.6, ls=":")
        for label, arms, col, mk in LINES:
            ax.plot([S[a]["d"] for a in arms], [S[a][f"dq_{judge}"] for a in arms], color=col,
                    lw=1.0, marker=mk, ms=3.0, label=label, zorder=2)
            for a in arms:
                if a in RANK_LABELS:
                    txt, off = RANK_LABELS[a]
                    ax.annotate(f"$k{{=}}{txt}$", (S[a]["d"], S[a][f"dq_{judge}"]),
                                xytext=off, textcoords="offset points", ha="center",
                                fontsize=5.8, color=BLUE)
        for label, arm, col, mk, sz in POINTS:
            s = S[arm]
            broken = bool(s["degenerate"])
            y, lo, hi = s[f"dq_{judge}"], *s[f"dq_{judge}_ci"]
            ax.errorbar(s["d"], y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=col,
                        elinewidth=0.7, capsize=1.5, zorder=3)
            style = ({"c": col} if mk == "x" else
                     {"facecolors": "white" if broken and mk not in ("P", "*") else col,
                      "edgecolors": col})
            ax.scatter(s["d"], y, s=sz * 0.8, marker=mk, zorder=5, linewidths=1.0,
                       label=label + (" (flagged)" if broken else ""), **style)
        ax.set_title(name, fontsize=7.5)
        ax.set_xlabel("Debiasing, Cohen's $d$")
        ax.grid(alpha=0.2, lw=0.4)
    axes[0].set_ylabel("Quality change vs. unsteered")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.02),
               handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
