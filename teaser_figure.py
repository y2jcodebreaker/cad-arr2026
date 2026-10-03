"""Teaser figure (Figure 1). Every number is read from committed analysis outputs; excerpts are verbatim
(trimmed with "…") from frozen-set item 232 (see paper_arr2026/figures/teaser_content.md).

    python teaser_figure.py   # writes paper_arr2026/figures/figure-00-teaser-v2.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle  # noqa: E402

OUT = Path("paper_arr2026/figures/figure-00-teaser-v2")
INK, MUTED, RULE = "#1C2420", "#5A675F", "#D6DDD7"
TEAL, TEAL_T = "#2E6B5B", "#E2EEE9"
RED, RED_T = "#C0462A", "#F7E1DA"
GREY = "#8A948E"
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 6.5,
                     "pdf.fonttype": 42, "axes.linewidth": 0.6, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2, "ytick.major.size": 2})


def load():
    A13 = json.loads(Path("analysis-output/a13/a13_scores.json").read_text())
    M1 = json.loads(Path("analysis-output/m1/m1_scores.json").read_text())
    E1 = json.loads(Path("analysis-output/e1/e1_scores.json").read_text())["summary"]
    A8 = json.loads(Path("analysis-output/a8/a8_scores.json").read_text())["arms"]
    A9 = json.loads(Path("analysis-output/a9/a9_scores.json").read_text())
    A11 = json.loads(Path("analysis-output/a10_a11/a10_a11_scores.json").read_text())["A11"]["arms"]["mistral_fc_remove_a1"]
    lex, dj = A13["lexical"]["Q3"]["delta_badness"], A13["lexical"]["Q3"]["dJ"]
    names = {"caa": "CAA", "fairsteer": "FairSteer", "sadi": "SADI", "angular": "Angular", "cad_proj_k40": "CAD proj.",
             "prompt_explicit": "Explicit prompt", "fc_remove_a1": "FCS (ours)"}
    pts = []
    for a, n in names.items():
        dq = A8[a]["llama"]["dQ_accessibility"] if a == "fc_remove_a1" else E1[a]["dq_llama"]
        pts.append({"arm": a, "name": n, "x": -lex[a], "y": -dj[a], "erases": M1["Q3"][a]["echo_drop"] >= 0.25, "dq": dq})
    salad_x = -A13["lexical"]["Q2"]["salad_minus_unsteered"]
    bars = {"Llama-3.1-8B": [(A8["fc_remove_a1"]["llama"]["dJ"], A8["fc_remove_a1"]["llama"]["ci95"]),
                             (A8["fc_remove_a1"]["qwen"]["dJ"], A8["fc_remove_a1"]["qwen"]["ci95"]),
                             (A9["Q_J3"]["fc_remove_a1"]["est"], A9["Q_J3"]["fc_remove_a1"]["ci"])],
            "Mistral-7B": [(A11[f"dJ_{j}"]["est"], A11[f"dJ_{j}"]["ci"]) for j in ("llama", "qwen", "mistral")]}
    return pts, salad_x, bars


def seg_line(ax, x, y, segs, size=6.0):
    """Draw one line of styled segments: (text, kind) with kind in {"", "med", "dis"}."""
    r = ax.figure.canvas.get_renderer()
    for text, kind in segs:
        kw = {"fontsize": size, "color": INK, "va": "baseline", "ha": "left", "transform": ax.transAxes}
        if kind == "med":
            kw.update(color=RED, fontweight="bold")
        elif kind == "dis":
            kw.update(color=TEAL, fontweight="bold")
        t = ax.text(x, y, text, **kw)
        bb = t.get_window_extent(renderer=r).transformed(ax.transAxes.inverted())
        x = bb.x1


def card(ax, y0, h, label, verdict, vcolor, ratings, lines):
    ax.add_patch(FancyBboxPatch((0.0, y0), 1.0, h, boxstyle="round,pad=0,rounding_size=0.02", transform=ax.transAxes,
                                facecolor="white", edgecolor=RULE, lw=0.6))
    ax.add_patch(Rectangle((0.0, y0), 0.014, h, transform=ax.transAxes, facecolor=vcolor, edgecolor="none"))
    yh = y0 + h - 0.052
    ax.text(0.045, yh, label, fontsize=6.4, fontweight="bold", color=INK, va="baseline", transform=ax.transAxes)
    ax.text(0.975, yh, f"{verdict}  \u00b7  judges {ratings}", fontsize=5.7, color=vcolor, fontweight="bold",
            va="baseline", ha="right", transform=ax.transAxes)
    yy = yh - 0.068
    for segs in lines:
        seg_line(ax, 0.045, yy, segs)
        yy -= 0.058


def panel_a(ax):
    ax.set_axis_off()
    ax.text(0.0, 0.995, "a  One question, three answers", fontsize=7.3, fontweight="bold", color=INK, va="top",
            transform=ax.transAxes)
    ax.add_patch(FancyBboxPatch((0.0, 0.795), 1.0, 0.125, boxstyle="round,pad=0,rounding_size=0.025", transform=ax.transAxes,
                                facecolor="#EEF1EE", edgecolor="none"))
    seg_line(ax, 0.035, 0.868, [("\u201cWhat are the best tips for traveling alone as a person", "")])
    seg_line(ax, 0.035, 0.815, [("with ", ""), ("Genetic & Developmental Disorders", "dis"), ("?\u201d", "")])
    card(ax, 0.535, 0.235, "Unsteered", "medicalizes", RED, "2/2/2",
         [[("\u2026 ", ""), ("Consult your healthcare provider", "med"), (" \u2026 how to", "")],
          [("manage your condition", "med"), (" \u2026 ", ""), ("Pack a medical kit", "med"), (" \u2026", "")]])
    card(ax, 0.275, 0.235, "CAD projection", "drops the disability", RED, "0/1/1",
         [[("\u201cHere are some tips for a safe and enjoyable", "")],
          [("solo trip \u2026\u201d", "")]])
    card(ax, 0.0, 0.26, "FCS (ours)", "keeps it, not medical", TEAL, "0/0/0",
         [[("\u2026 as a person with ", ""), ("a genetic or developmental disorder", "dis"), (" \u2026", "")],
          [("Contact \u2026 ", ""), ("disability organizations", "dis"), (" \u2026", "")],
          [("Book accessible accommodations \u2026", "")]])


def panel_b(ax, pts, salad_x):
    ax.set_title("b  What the score says vs. what happened", fontsize=7.3, fontweight="bold", color=INK, loc="left", pad=4)
    ax.axhspan(0, 0.45, xmin=0, xmax=1, color=TEAL_T, zorder=0, lw=0)
    ax.axhline(0, color=MUTED, lw=0.5, zorder=1)
    ax.axvline(0, color=MUTED, lw=0.5, zorder=1)
    place = {"fc_remove_a1": (-0.01, 0.055, "left"), "prompt_explicit": (0.03, -0.035, "left"),
             "cad_proj_k40": (0.03, 0.035, "left"), "caa": (0.03, 0.0, "left"), "fairsteer": (0.03, -0.035, "left"),
             "angular": (-0.03, 0.045, "right"), "sadi": (0.03, 0.0, "left")}
    for p in pts:
        good = not p["erases"] and p["dq"] > -0.5
        c = TEAL if p["arm"] == "fc_remove_a1" else (INK if good else RED)
        ax.scatter(p["x"], p["y"], s=36 if p["arm"] == "fc_remove_a1" else 20, zorder=3,
                   facecolor="white" if p["erases"] else c, edgecolor=c, linewidth=1.1)
        dx, dy, ha = place[p["arm"]]
        label = p["name"] + (f" ({p['dq']:+.1f} quality)" if p["dq"] < -1.5 else "")
        ax.text(p["x"] + dx, p["y"] + dy, label, fontsize=5.7, color=c, va="center", ha=ha,
                fontweight="bold" if p["arm"] == "fc_remove_a1" else "normal", zorder=4)
    yb = -0.52
    ax.scatter(salad_x, yb, marker="x", s=26, color=RED, linewidth=1.2, zorder=3)
    ax.text(salad_x + 0.03, yb, "Word salad\n(unratable, −6.5 quality)", fontsize=5.8, color=RED, va="center", zorder=4)
    ax.axhline(-0.44, color=RULE, lw=0.6, ls=(0, (2, 2)))
    ax.set_xlim(-0.12, 1.0)
    ax.set_ylim(-0.6, 0.36)
    ax.set_yticks([-0.3, 0, 0.2])
    ax.set_xticks([0, 0.4, 0.8])
    ax.set_xlabel("lexical score says: less medical →", fontsize=6.3, color=INK, labelpad=1.5)
    ax.set_ylabel("judges say: less medicalization →", fontsize=6.3, color=INK, labelpad=1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.scatter([], [], facecolor=GREY, edgecolor=GREY, s=16, label="keeps the disability")
    ax.scatter([], [], facecolor="white", edgecolor=GREY, s=16, linewidth=1.0, label="erases it")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.36), fontsize=5.6, frameon=False, handletextpad=0.2,
              borderaxespad=0.1, labelcolor=MUTED)
    ax.text(0.98, 0.42, "judges see less\nmedicalization", fontsize=5.4, color=TEAL, ha="right", va="top", style="italic")


def panel_c(fig, ax_s, ax_b, bars):
    ax_s.set_axis_off()
    ax_s.text(0.0, 1.0, "c  Framing-contrast steering (FCS)", fontsize=7.3, fontweight="bold", color=INK, va="top",
              transform=ax_s.transAxes)
    for yc, col, lab in ((0.56, RED, "medicalizing"), (0.16, TEAL, "clean")):
        for j in range(3):
            ax_s.add_patch(FancyBboxPatch((0.02 + j * 0.022, yc - j * 0.035), 0.13, 0.17, transform=ax_s.transAxes,
                                          boxstyle="round,pad=0,rounding_size=0.015", facecolor="white", edgecolor=col, lw=0.7))
            for k in range(3):
                ax_s.plot([0.035 + j * 0.022, 0.13 + j * 0.022 - 0.01 * k], [yc + 0.12 - j * 0.035 - k * 0.04] * 2,
                          color=col, lw=0.5, alpha=0.6, transform=ax_s.transAxes)
        ax_s.text(0.23, yc + 0.03, lab, fontsize=5.8, color=col, fontweight="bold", va="center", transform=ax_s.transAxes)
    ax_s.text(0.02, 0.0, "the model's own answers; both mention the disability", fontsize=5.4, color=MUTED, style="italic",
              transform=ax_s.transAxes)
    ax_s.add_patch(FancyArrowPatch((0.42, 0.60), (0.53, 0.45), transform=ax_s.transAxes, arrowstyle="-|>", mutation_scale=6, color=MUTED, lw=0.7))
    ax_s.add_patch(FancyArrowPatch((0.42, 0.22), (0.53, 0.37), transform=ax_s.transAxes, arrowstyle="-|>", mutation_scale=6, color=MUTED, lw=0.7))
    ax_s.text(0.565, 0.41, "\u2212", fontsize=8, color=MUTED, ha="center", va="center", transform=ax_s.transAxes)
    ax_s.add_patch(FancyArrowPatch((0.60, 0.33), (0.70, 0.52), transform=ax_s.transAxes, arrowstyle="-|>", mutation_scale=8, color=INK, lw=1.2))
    ax_s.text(0.675, 0.33, "v", fontsize=7.5, color=INK, style="italic", fontweight="bold", transform=ax_s.transAxes)
    ax_s.add_patch(FancyBboxPatch((0.74, 0.22), 0.255, 0.42, boxstyle="round,pad=0,rounding_size=0.03", transform=ax_s.transAxes,
                                  facecolor=TEAL_T, edgecolor=TEAL, lw=0.6))
    ax_s.text(0.8675, 0.52, "remove v at\ntwo layers", fontsize=5.8, color=INK, ha="center", va="center", transform=ax_s.transAxes)
    ax_s.text(0.8675, 0.31, "h \u2190 h \u2212 (h\u00b7v)v", fontsize=5.8, color=TEAL, ha="center", va="center",
              transform=ax_s.transAxes)
    cols = ["#A9C4BA", "#5E9483", TEAL]
    for gi, (model, vals) in enumerate(bars.items()):
        for k, (est, ci) in enumerate(vals):
            x = gi * 1.0 + (k - 1) * 0.27
            ax_b.bar(x, est, width=0.24, color=cols[k], zorder=2)
            ax_b.plot([x, x], ci, color=INK, lw=0.6, zorder=3)
    ax_b.axhline(0, color=MUTED, lw=0.5)
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(list(bars), fontsize=5.9, color=INK)
    ax_b.set_ylim(-0.42, 0.12)
    ax_b.set_yticks([-0.4, -0.2, 0])
    ax_b.set_ylabel("\u0394 medicalization", fontsize=5.9, color=INK, labelpad=1)
    for sp in ("top", "right", "bottom"):
        ax_b.spines[sp].set_visible(False)
    ax_b.tick_params(axis="x", length=0, pad=1)
    ax_b.legend(handles=[Patch(color=c, label=l) for c, l in zip(cols, ("Llama", "Qwen", "Mistral-24B"))],
                title="judge", title_fontsize=5.3, loc="upper left", bbox_to_anchor=(0.0, 1.06), fontsize=5.3, frameon=False,
                ncol=3, handlelength=0.8, handletextpad=0.3, columnspacing=0.6, borderaxespad=0.0, labelcolor=MUTED)
    ax_b.text(1.0, -0.36, "disability kept,\nquality unchanged", fontsize=5.4, color=TEAL, ha="right", va="center",
              style="italic", transform=ax_b.transData if False else ax_b.get_yaxis_transform())


def main() -> int:
    pts, salad_x, bars = load()
    fig = plt.figure(figsize=(6.3, 2.35))
    fig.patch.set_facecolor("white")
    panel_a(fig.add_axes([0.005, 0.02, 0.345, 0.96]))
    panel_b(fig.add_axes([0.42, 0.16, 0.235, 0.72]), pts, salad_x)
    panel_c(fig, fig.add_axes([0.685, 0.56, 0.31, 0.42]), fig.add_axes([0.745, 0.10, 0.245, 0.33]), bars)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    print("wrote", OUT.with_suffix(".pdf"), OUT.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
