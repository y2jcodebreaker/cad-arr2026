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
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 7, "svg.fonttype": "path", "pdf.fonttype": 42,
                     "axes.linewidth": 0.7, "axes.edgecolor": INK})


def scatter(pts, salad_x):
    fig, ax = plt.subplots(figsize=(2.4, 2.1))
    ax.axhspan(0, 0.33, color=TEAL_T, lw=0, zorder=0)
    ax.axhline(-0.46, color="#C9D1CB", lw=0.6, ls=(0, (2, 2)))
    off = {"fc_remove_a1": (0.025, 0.04), "prompt_explicit": (-0.02, -0.035), "cad_proj_k40": (0.03, 0.0), "caa": (0.03, 0.02),
           "fairsteer": (0.03, 0.0), "angular": (0.0, 0.03), "sadi": (0.03, 0.03)}
    for p in pts:
        fcs = p["arm"] == "fc_remove_a1"
        c = TEAL if fcs else (INK if not p["erases"] else RED)
        ax.scatter(p["x"], p["y"], s=40 if fcs else 22, facecolor="white" if p["erases"] else c, edgecolor=c, lw=1.1, zorder=3)
        sep = "\n" if p["arm"] == "angular" else " "
        lab = p["name"] + (f"{sep}(−{abs(p['dq']):.1f} quality)" if p["dq"] < -1.5 else "")
        if p["arm"] == "prompt_explicit":
            lab = "Explicit\nprompt"
        dx, dy = off[p["arm"]]
        va = {"angular": "bottom", "prompt_explicit": "top"}.get(p["arm"], "center")
        ax.text(p["x"] + dx, p["y"] + dy, lab, fontsize=6.3, color=TEAL if fcs else INK, fontweight="bold" if fcs else "normal",
                ha="center" if p["arm"] == "angular" else "left", va=va, linespacing=1.1)
    ax.scatter(salad_x, -0.53, marker="x", s=26, color=RED, lw=1.2, zorder=3)
    ax.text(salad_x + 0.03, -0.53, "Word salad\n(unratable, −6.5 quality)", fontsize=6.0, color=INK, va="center")
    ax.set_xlim(-0.06, 1.0)
    ax.set_ylim(-0.6, 0.33)
    ax.set_xticks([0, 0.4, 0.8])
    ax.set_yticks([-0.6, -0.3, 0.0, 0.2])
    ax.set_xlabel("lexical score says: less medical →")
    ax.set_ylabel("judges say: less medicalization →")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.2)
    return fig


def bars(b):
    fig, ax = plt.subplots(figsize=(2.4, 1.5))
    cols = [TEAL, "#7FB3A3", "#BFD9CF"]
    for gi, (model, vals) in enumerate(b.items()):
        for k, (est, ci) in enumerate(vals):
            x = gi + (k - 1) * 0.27
            ax.bar(x, est, width=0.24, color=cols[k], zorder=2)
            ax.errorbar(x, est, yerr=[[est - ci[0]], [ci[1] - est]], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.5, zorder=3)
            ax.text(x, ci[0] - 0.02, f"{est:.2f}".replace("-", "−"), fontsize=5.6, ha="center", va="top", color=INK)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(list(b))
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(-0.5, 0.02)
    ax.set_yticks([-0.4, -0.2, 0.0])
    ax.set_ylabel("Δ medicalization")
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(cols, ("Llama", "Qwen", "Mistral-24B"))], title="judge",
              title_fontsize=5.8, fontsize=5.8, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.42),
              handlelength=0.9, columnspacing=0.8)
    fig.tight_layout(pad=0.2)
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
