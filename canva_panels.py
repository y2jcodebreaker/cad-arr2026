"""Exact chart panels for the Canva teaser (styled like the GPT draft), read from committed outputs.

    python canva_panels.py   # writes paper_arr2026/figures/canva/{scatter_b,bars_c}.{svg,pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

import teaser_figure as tf        # noqa: E402  (same data loading as the earlier teaser)

OUT = Path("paper_arr2026/figures/canva")
INK, MUTED, TEAL, TEAL_T, RED = "#1C2420", "#5A675F", "#2E6B5B", "#E2EEE9", "#C0462A"
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 6.2, "svg.fonttype": "path", "pdf.fonttype": 42,
                     "axes.linewidth": 0.7, "axes.edgecolor": INK})


def scatter(pts, salad_x):
    fig, ax = plt.subplots(figsize=(560 / 300, 545 / 300))   # Canva slot: 560 x 545 px
    ax.axhspan(0, 0.36, color=TEAL_T, lw=0, zorder=0)
    ax.axhline(-0.46, color="#C9D1CB", lw=0.6, ls=(0, (2, 2)))
    # (dx, dy, ha, va, label) per method: chosen so no label touches another label or point at this size
    lab = {"fc_remove_a1": (0.03, 0.045, "left", "center", "FCS (ours)"),
           "prompt_explicit": (-0.015, -0.04, "left", "top", "Explicit\nprompt"),
           "caa": (0.045, 0.0, "left", "center", "CAA"),
           "cad_proj_k40": (0.04, -0.04, "left", "center", "CAD proj."),
           "fairsteer": (-0.02, -0.04, "center", "top", "FairSteer"),
           "angular": (0.0, -0.045, "center", "top", "Angular\n(\u22122.1 quality)"),
           "sadi": (0.035, 0.0, "left", "center", "SADI (\u22123.9 quality)")}
    for p in pts:
        fcs = p["arm"] == "fc_remove_a1"
        c = TEAL if fcs else (INK if not p["erases"] else RED)
        ax.scatter(p["x"], p["y"], s=34 if fcs else 18, facecolor="white" if p["erases"] else c, edgecolor=c, lw=1.0, zorder=3)
        dx, dy, ha, va, text = lab[p["arm"]]
        ax.text(p["x"] + dx, p["y"] + dy, text, fontsize=5.8, color=TEAL if fcs else INK,
                fontweight="bold" if fcs else "normal", ha=ha, va=va, linespacing=1.05, zorder=4)
    ax.scatter(salad_x, -0.53, marker="x", s=22, color=RED, lw=1.1, zorder=3)
    ax.text(salad_x + 0.035, -0.53, "Word salad\n(unratable,\n\u22126.5 quality)", fontsize=5.6, color=INK, ha="left",
            va="center", linespacing=1.05)
    ax.set_xlim(-0.06, 1.0)
    ax.set_ylim(-0.6, 0.36)
    ax.set_xticks([0, 0.4, 0.8])
    ax.set_yticks([-0.6, -0.3, 0.0, 0.2])
    ax.set_xlabel("lexical score: less medical \u2192", fontsize=6.0, labelpad=1.5)
    ax.set_ylabel("judges: less medicalization \u2192", fontsize=6.0, labelpad=1.5)
    ax.tick_params(labelsize=5.8, length=2, pad=1.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.scatter([], [], s=18, facecolor=INK, edgecolor=INK, label="keeps the disability")
    ax.scatter([], [], s=18, facecolor="white", edgecolor=RED, lw=1.0, label="erases it")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.19), ncol=2, frameon=False, fontsize=5.8, handletextpad=0.2,
              columnspacing=1.2)
    fig.tight_layout(pad=0.15)
    return fig


def bars(b):
    fig, ax = plt.subplots(figsize=(600 / 300, 300 / 300))   # Canva slot: 600 x 300 px
    cols = [TEAL, "#7FB3A3", "#BFD9CF"]
    for gi, (model, vals) in enumerate(b.items()):
        for k, (est, ci) in enumerate(vals):
            x = gi + (k - 1) * 0.27
            ax.bar(x, est, width=0.24, color=cols[k], zorder=2)
            ax.errorbar(x, est, yerr=[[est - ci[0]], [ci[1] - est]], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.3, zorder=3)
            ax.text(x, ci[0] - 0.015, f"{est:.2f}".replace("-", "\u2212"), fontsize=5.0, ha="center", va="top", color=INK)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(list(b), fontsize=5.9)
    ax.tick_params(axis="x", length=0, pad=1)
    ax.tick_params(axis="y", labelsize=5.6, length=2, pad=1)
    ax.set_ylim(-0.47, 0.17)
    ax.set_yticks([-0.4, -0.2, 0.0])
    ax.set_ylabel("\u0394 medicalization", fontsize=5.9, labelpad=1)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(cols, ("Llama judge", "Qwen judge", "Mistral-24B judge"))],
              fontsize=5.2, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.04), handlelength=0.8,
              handletextpad=0.3, columnspacing=0.7, borderaxespad=0)
    fig.tight_layout(pad=0.15)
    return fig


def main() -> int:
    pts, salad_x, b = tf.load()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fig in (("scatter_b", scatter(pts, salad_x)), ("bars_c", bars(b))):
        for ext in ("svg", "pdf"):
            fig.savefig(OUT / f"{name}.{ext}", transparent=True)
        fig.savefig(OUT / f"{name}.png", dpi=400, transparent=True)
        print("wrote", OUT / f"{name}.svg")
    for model, vals in b.items():
        print(model, [(round(e, 3), [round(c, 3) for c in ci]) for e, ci in vals])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
