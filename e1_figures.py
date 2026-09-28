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

LINES = [  # (label, arms in order, colour, marker)
    ("CAD projection, rank 1 to 80", [f"cad_proj_k{k}" for k in (1, 5, 10, 20, 40, 80)], BLUE, "o"),
    ("CAD additive, probe", [f"add_probe_a{a}" for a in (3, 5, 7, 9)], SKY, "^"),
    ("CAD additive, mean-diff.", [f"add_meandiff_a{a}" for a in (3, 5, 7, 8)], SKY, "s"),
    ("LEACE, rank 1 to 80", [f"leace_k{k}" for k in (1, 5, 10, 20, 40, 80)], GREY, "D"),
]
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
    S = json.loads(SCORES.read_text())["summary"]
    plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 8, "legend.fontsize": 6.5,
                         "font.family": "serif", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.45), sharex=True)
    for ax, judge, name in ((axes[0], "llama", "Llama-3.1-8B judge"),
                            (axes[1], "qwen", "Qwen2.5-7B judge")):
        ax.axhline(0, color=GREY, lw=0.6, ls=":")
        for label, arms, col, mk in LINES:
            xs = [S[a]["d"] for a in arms]
            ys = [S[a][f"dq_{judge}"] for a in arms]
            ax.plot(xs, ys, color=col, lw=0.9, marker=mk, ms=3.2, label=label, zorder=2)
        k40 = S["cad_proj_k40"]
        ax.errorbar(k40["d"], k40[f"dq_{judge}"],
                    xerr=[[k40["d"] - k40["d_ci"][0]], [k40["d_ci"][1] - k40["d"]]],
                    yerr=[[k40[f"dq_{judge}"] - k40[f"dq_{judge}_ci"][0]],
                          [k40[f"dq_{judge}_ci"][1] - k40[f"dq_{judge}"]]],
                    fmt="o", color=BLUE, ms=5, lw=0.7, capsize=1.5, zorder=4)
        for label, arm, col, mk, sz in POINTS:
            s = S[arm]
            broken = bool(s["degenerate"])
            ax.errorbar(s["d"], s[f"dq_{judge}"],
                        xerr=[[s["d"] - s["d_ci"][0]], [s["d_ci"][1] - s["d"]]],
                        yerr=[[s[f"dq_{judge}"] - s[f"dq_{judge}_ci"][0]],
                              [s[f"dq_{judge}_ci"][1] - s[f"dq_{judge}"]]],
                        fmt="none", ecolor=col, elinewidth=0.6, capsize=1.2, zorder=3, alpha=0.8)
            ax.scatter(s["d"], s[f"dq_{judge}"], s=sz, marker=mk, zorder=5,
                       **({"c": col} if mk == "x" else
                          {"facecolors": "white" if broken and mk not in ("P", "*") else col,
                           "edgecolors": col}), linewidths=1.0,
                       label=label + (" (flagged)" if broken else ""))
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("Debiasing, Cohen's $d$ (higher = less medicalized)")
        ax.grid(alpha=0.2, lw=0.4)
    axes[0].set_ylabel("Quality change vs. unsteered")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02),
               handletextpad=0.3, columnspacing=0.9)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
